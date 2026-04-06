import math
import os
import time

import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from data import DataLoader
from model import GPT, GPTConfig
from utils import unwrap_model

# hardcoded from gpt-3 paper
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073


def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay <= 1
    cos_coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return max_lr + cos_coeff * (max_lr - min_lr)


ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "No cuda available for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device_type = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device=device_type)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

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
    B=4, T=32, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
)
val_loader = DataLoader(
    B=4, T=32, process_rank=ddp_rank, num_processes=ddp_world_size, split="val"
)

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

if ddp:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[ddp_local_rank]
    )

optimizer = model.configure_optimizer(weight_decay=0.1, lr=6e-4, device=device)
raw_model = unwrap_model(model)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train.log")


def main():
    for step in range(max_steps):
        t0 = time.time()
        last_step = step == max_steps - 1

        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
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
                if step > 0 and (step % 2500 == 0 or last_step):
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": raw_model.config,
                        "step": step,
                        "val_loss": val_loss_accum.item(),
                        "train_loader": {
                            "current_shard": train_loader.current_shard,
                            "current_position": train_loader.current_position,
                            "rng_state": train_loader.rng.bit_generator.state,
                        },
                        "torch_rng_state": torch.get_rng_state(),
                    }
                    torch.save(checkpoint, checkpoint_path)

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
