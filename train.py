import math
import os

import torch
import torch.distributed as dist
from torch.distributed import init_process_group

from data import DataLoader
from model import GPT, GPTConfig

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
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device=device)
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
for step in range(max_steps):
    loss_accum = 0.0
    optimizer.zero_grad()
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
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
    if master_process:
        print(
            f"STEP {step} -> loss: {loss_accum.item():.4f},\t",
            f"lr: {lr:.2e},\t",
            f"grad norm: {norm:.2e}",
        )
