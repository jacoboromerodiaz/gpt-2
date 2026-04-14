from datasets import load_dataset
import tiktoken
import time
import torch
from torch.utils.data import Dataset, DataLoader

from gpt2.model import GPT, GPTConfig
from gpt2.train import load_checkpoint, setup_device

from dataclasses import fields


class AlpacaDataset(Dataset):
    def __init__(self, enc, max_length=1024):
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        self.examples = []
        for row in ds:
            text = (
                f"<|im_start|>user\n{row['instruction']}"
                + (f"\n{row['input']}" if row["input"] else "")
                + f"<|im_end|>\n<|im_start|>assistant\n{row['output']}<|im_end|>"
            )
            tokens = enc.encode(text, allowed_special={"<|im_start|>", "<|im_end|>"})
            if len(tokens) <= max_length:
                self.examples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        x = tokens[:-1]
        y = tokens[1:]
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)

    max_len = max(x.size(0) for x in xs)

    xs_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    ys_pad = torch.full((len(ys), max_len), fill_value=-100, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        xs_pad[i, : x.size(0)] = x
        ys_pad[i, : y.size(0)] = y

    return xs_pad, ys_pad


def extend_encoder(enc):
    enc_extended = tiktoken.Encoding(
        name="gpt2_chat",
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens={
            **enc._special_tokens,
            "<|im_start|>": 50257,
            "<|im_end|>": 50258,
        },
    )
    return enc_extended


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    device = "mps"
    device_type = "cpu"
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

    checkpoint_file = "/Users/jacoboromerodiaz/Projects/gpt-2/gpt2/model_10000.pt"

    model, optimizer, checkpoint = load_checkpoint(checkpoint_file, device, device_type)

    dataset = AlpacaDataset(enc_extended)
    train_loader = DataLoader(
        dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
    )

    # reuse train logic from train.py
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
