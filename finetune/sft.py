import time
import os
import tiktoken

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader

from gpt2.train import load_checkpoint, setup_device, get_lr
from gpt2.utils import unwrap_model
from finetune.data import AlpacaDataset, DollyDataset, collate_fn


def extend_encoder(enc):
    enc_extended = tiktoken.Encoding(
        name="gpt2_chat",
        pat_str=enc._pat_str,  # pylint: disable=protected-access
        mergeable_ranks=enc._mergeable_ranks,  # pylint: disable=protected-access
        special_tokens={
            **enc._special_tokens,  # pylint: disable=protected-access
            "<|im_start|>": 50257,
            "<|im_end|>": 50258,
        },
    )
    return enc_extended


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    enc_extended = extend_encoder(enc)

    ctx = setup_device()

    ddp = ctx.ddp
    ddp_rank = ctx.ddp_rank
    ddp_local_rank = ctx.ddp_local_rank
    ddp_world_size = ctx.ddp_world_size
    device = ctx.device
    device_type = ctx.device_type
    master_process = ctx.master_process

    grad_accum_steps = 1
    max_lr = 1e-5
    min_lr = 1e-6
    warmup_steps = 700
    weight_decay = 0.1

    epochs = 5

    checkpoint_file = "/workspace/gpt-2/gpt2/log/model_19072.pt"

    model, checkpoint = load_checkpoint(checkpoint_file, device, weights_only=True)
    optimizer = model.configure_optimizer(
        weight_decay=weight_decay, lr=max_lr, device=device_type
    )

    model = torch.compile(model)
    raw_model = unwrap_model(model)

    train_dataset = ConcatDataset(
        [
            AlpacaDataset(enc_extended, split="train"),
            DollyDataset(enc_extended, split="train"),
        ]
    )
    val_dataset = ConcatDataset(
        [
            AlpacaDataset(enc_extended, split="val"),
            DollyDataset(enc_extended, split="val"),
        ]
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=8, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=8, collate_fn=collate_fn
    )

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(BASE_DIR, ".", "log")
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "sft_finetune.log")

    # reuse train logic from train.py
    max_steps = len(train_dataset) // 8 * epochs  # dataloader bs
    eval_every = max_steps // 3
    for step in range(max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        if step > 0 and (step % eval_every == 0 or last_step):
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    try:
                        x, y = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        x, y = next(val_iter)
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step % eval_every == 0 or last_step:
                    checkpoint_path = os.path.join(log_dir, f"ft_model_{step:05d}.pt")
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "config": raw_model.config,
                    }
                    torch.save(checkpoint, checkpoint_path)

        model.train()
        loss_accum = 0.0
        optimizer.zero_grad()
        for micro_step in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)

            if torch.isnan(loss):
                print(f"NaN loss detected at step {step}, skipping batch")
                print(x, y)
                optimizer.zero_grad()
                loss_accum = torch.tensor(float("nan"))
                continue
            loss /= grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        t1 = time.time()
        dt = t1 - t0
        if master_process:
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} |"
                f", lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms",
            )
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
