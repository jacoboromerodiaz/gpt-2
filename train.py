from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias", torch.tril(torch.ones(config.block_size, config.block_size))
        )

        # self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        weights = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T)
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        # weights = self.dropout(weights)
        out = weights @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(config.head_size) for _ in range(config.num_heads)]
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1  # flag
        # self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        y = torch.cat([h(x) for h in self.heads], dim=-1)
        y = self.c_proj(x)
        # y = self.dropout(self.proj(y))
        return y


class FeedFoward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = (nn.Linear(config.n_embd, 4 * config.n_embd),)
        self.gelu = (nn.GELU(approximate="tanh"),)
        self.c_proj = (nn.Linear(4 * config.n_embd, config.n_embd),)
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
        self.ln_1 = nn.LayerNorm(config)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config)
        self.mlp = FeedFoward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
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
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
            return logits, loss


class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.current_pos = 0

    def get_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_pos:self.current_pos+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_pos += B * T
        if self.current_pos + B * T + 1 >= len(self.tokens):
            self.current_pos = 0
        return x, y


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print("using device: {device}")

train_loader = DataLoader(B=4, T=32)

enc = tiktoken.get_encoding("gpt2")
with open("input.txt", "r") as f:
    text = f.read()
tokens = enc.encode(text[:1000])
B, T = 4, 32
buf = torch.tensor(tokens[: B * T + 1]).to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for i in range(50):
    x, y = train_loader.get_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print(f"step {i}: loss: {loss.item()}")
