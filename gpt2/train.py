import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from types import SimpleNamespace

from .data import DataLoader
from .model import GPT, GPTConfig
from .utils import unwrap_model
from .hellaswag import iterate_examples, render_example, get_most_likely_row

import sys
sys.path.insert(0, os.path.dirname(__file__))

def setup_device():
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        assert torch.cuda.is_available(), "No cuda available for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    if master_process:
        print(f"using device: {device}")

    return SimpleNamespace(
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_local_rank=ddp_local_rank,
        ddp_world_size=ddp_world_size,
        device=device,
        device_type=device_type,
        master_process=master_process,
    )


def load_checkpoint(path, device, device_type):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = GPT(checkpoint["config"])
    model.to(device)
    model.load_state_dict(checkpoint["model"])

    opt_state = checkpoint["optimizer"]
    lr = opt_state["param_groups"][0]["lr"]
    wd = opt_state["param_groups"][0]["weight_decay"]
    optimizer = model.configure_optimizer(weight_decay=wd, lr=lr, device=device_type)
    optimizer.load_state_dict(opt_state)
    return model, optimizer, checkpoint


def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay <= 1
    cos_coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return min_lr + cos_coeff * (max_lr - min_lr)

def main():
    # hardcoded from gpt-3 paper
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    weight_decay = 0.1
    warmup_steps = 715
    max_steps = 19073

    ctx = setup_device()

    ddp = ctx.ddp
    ddp_rank = ctx.ddp_rank
    ddp_local_rank = ctx.ddp_local_rank
    ddp_world_size = ctx.ddp_world_size
    device = ctx.device
    device_type = ctx.device_type
    master_process = ctx.master_process

    torch.manual_seed(333)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(333)
    torch.set_float32_matmul_precision("high")

    sim_batch_size = 524288
    B, T = 16, 1024
    assert (
        sim_batch_size % (B * T * ddp_world_size) == 0
    ), "sim_batch_size must be divisible by B * T * ddp_world_size"
    grad_accum_steps = sim_batch_size // (B * T * ddp_world_size)

    train_loader = DataLoader(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
    )
    val_loader = DataLoader(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val"
    )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(BASE_DIR, "..", "log")
    log_dir = os.path.abspath(log_dir)  # normaliza el path
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    resume_training = True
    if resume_training:
        checkpoint_files = sorted(
            f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")
        )
        assert checkpoint_files, "no checkpoints found"

        model, optimizer, checkpoint = load_checkpoint(
            os.path.join(log_dir, checkpoint_files[-1]), device, device_type
        )
        model.load_state_dict(checkpoint["model"])
        train_loader.set(checkpoint["train_loader"])
        current_step = checkpoint["step"] + 1

        if master_process:
            print(
                f"resuming training from step {current_step}",
                f"with val_loss={checkpoint['val_loss']:.4f}",
            )
    else:
        model = GPT(GPTConfig(vocab_size=50304))
        model.to(device)
        optimizer = model.configure_optimizer(
            weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type
        )
        current_step = 0

        if master_process:
            open(log_file, "w").close()

    model = torch.compile(model)
    raw_model = unwrap_model(model)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[ddp_local_rank]
        )

    for step in range(current_step, max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        if step > 0 and (step % 2500 == 0 or last_step):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.get_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": raw_model.config,
                        "step": step,
                        "val_loss": val_loss_accum.item(),
                        "train_loader": {
                            "current_shard": train_loader.current_shard,
                            "current_position": train_loader.current_pos,
                            "rng_state": train_loader.rng.bit_generator.state,
                        },
                        "torch_rng_state": torch.get_rng_state(),
                    }
                    torch.save(checkpoint, checkpoint_path)

        if step > 0 and (step % 2500 == 0 or last_step):
            num_correct_norm = 0
            num_total = 0
            raw_model.eval()
            for i, example in enumerate(iterate_examples("val")):
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = raw_model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(
                    num_correct_norm, dtype=torch.long, device=device
                )
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(
                    f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}"
                )
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # forward pass, backward pass
        model.train()
        loss_accum = 0.0
        optimizer.zero_grad()
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.get_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss /= grad_accum_steps
            loss_accum += loss.detach()
            if ddp:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        tokens_processed = (
            train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        )
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(
                f"step {step:5d} | loss: {loss_accum.item():.6f} |"
                f", lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms",
                f"| tok/sec: {tokens_per_sec:.2f}",
            )
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()