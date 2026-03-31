from dataclasses import dataclass

import math
import inspect
import os

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed import init_process_group


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.head_size, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.head_size
        self.register_buffer(
            "bias", torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)  # [B, T, 192] (3 * 64)
        q, k, v = qkv.split(self.head_size, dim=2)  # Each: [B, T, 64]
        weights = q @ k.transpose(-2, -1) * self.head_size**-0.5  # Scale by head_size
        weights = weights.masked_fill(self.bias[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        out = weights @ v  # [B, T, 64]
        return out


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1  # flag
        # self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        y = torch.cat([h(x) for h in self.heads], dim=-1)
        y = self.c_proj(y)
        # y = self.dropout(self.proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1  # flag
        # self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:  # 124M
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.2

    @property
    def head_size(self):
        return self.n_embd // self.n_head


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # follow source code
        std = 0.02
        if hasattr(module, "GPT_SCALE_INIT"):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def configure_optimizer(self, weight_decay, lr, device):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.ndim < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)
        print(
            f"num decay params: {num_decay_params}, "
            f"num no decay params: {num_no_decay_params}"
        )

        fused_avail = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avail and device == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of len {T} as it exceeds "
            f"block_size = {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb  # broadcasting
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def load_tokens(file):
    npt = np.load(file)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt


class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}, "Invalid split"

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = self.B * self.T * self.process_rank

    def get_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos : self.current_pos + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_pos += B * T * self.num_processes
        if self.current_pos + B * T * self.num_processes + 1 >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = self.B * self.T * self.process_rank
        return x, y


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
