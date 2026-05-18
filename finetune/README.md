# Finetuning Pipeline

Post-training stages to transform the pretrained base model into an instruction-following assistant.

Sources:

- *[Reinforcement Learning from Human Feedback](https://rlhfbook.com)*. Nathan Lambert. Online, 2026.
- [Video series on RLHF](https://www.youtube.com/watch?v=jQPiH-KB4B0&list=PLL1tdVxB1CpVpEtMHxwuR4uI4Lxjw00_y). Nathan Lambert. Online, 2026

---

## Stage 1: Supervised Fine-Tuning (`sft.py`)

During pretraining, the model learns to predict the next token on raw internet text with no structure. SFT keeps the same objective (next-token prediction) but restricts it to assistant response tokens on formatted conversations.

Therefore, we will train the model on instruction-response pairs from Alpaca and Databricks Dolly datasets to shift to following assistant formatting:

```
<|im_start|>user
What is machine learning?
<|im_end|>
<|im_start|>assistant
Machine learning is...
<|im_end|>
```

This allows role-based separation and structured generation.

#### Datasets

1. [`AlpacaDataset`](https://huggingface.co/datasets/yahma/alpaca-cleaned): 52K instruction-response pairs from Stanford's self-instruct on GPT-3
2. [`Databricks Dolly 15K`](https://huggingface.co/datasets/databricks/databricks-dolly-15k): 15K enterprise question-answering pairs

---

### Implementations

#### 1. Extended Tokenizer Vocabulary

Base GPT-2 tokenizer (50,257 tokens) extended with 2 special tokens:
- `<|im_start|>`: Message boundary (user/assistant)
- `<|im_end|>`: Message terminator

```python
# sft.py
enc_extended = tiktoken.Encoding(
    name="gpt2_chat",
    ...
    special_tokens={
        "<|im_start|>": 50257,
        "<|im_end|>": 50258,
    },
)
# vocab_size: 50257 -> 50259
```

#### 2. Prompt Masking

Only train the model to predict assistant responses, not user prompts.

```python
# data.py
def __getitem__(self, idx):
    tokens = self.examples[idx]
    x = tokens[:-1]
    y = tokens[1:].clone()

    # find where assistant response begins
    assistant_start_seq = torch.tensor(
        [IM_START] + self.enc_extended.encode("assistant\n"),
        dtype=torch.long
    )
    mask_until = find_subsequence(y, assistant_start_seq)

    if mask_until is None:
        y[:] = -100  # no assistant response, mask everything
    else:
        y[:mask_until] = -100  # mask until assistant starts

    return x, y
```

The loss function ignores tokens marked with `-100`, the model only learns from assistant tokens.

#### 3. Hyperparameter changes from pre-training

- **Learning rate**: pretraining used `6e-4`, supervised fine-tuning uses `1e-5`. The reasons are:
    - Catastrophic forgetting: Large updates destroy pretraining knowledge.
    - Smaller dataset, less data diversity, overfitting risk (67K examples vs 10B tokens pretraining).
- **Effective batch size**: pretraining used a large token budget per step (~500K tokens). SFT uses 8 * 4 = 32 sequences per update, which is a much smaller absolute volume of data per step, so learning rate stays conservative.

#### 4. Training Loop logic

Recycled from `gpt2/train.py`, the changes are:

- **Goal**: starts from an existing checkpoint instead of from scratch.
- **Data**: shard-based token `DataLoader` (raw FineWeb-style streams) → HuggingFace-style `ConcatDataset(AlpacaDataset, DollyDataset)` wrapped in a standard `torch.utils.data.DataLoader` with `shuffle` and `collate_fn`. Batches are pulled via `next(iter)` with `StopIteration` → re-iter for epoch wraparound.
- **Tokenizer**: `standard GPT-2 tokenizer → extended via `extend_encoder(enc)` to add the instruction special tokens.
- **Loop length**: `max_steps = len(train_dataset) // (B * grad_accum) * epochs`, epoch-driven.
- **Hyperparameters**: changes mentioned before (`max_lr=1e-5`, `min_lr=1e-6`, `B=16`, `grad_accum=4`), plus 3 epochs and no `sim_batch_size` math.
- **Evaluation**: eval only 3 times total. HellaSwag block removed.
- **Checkpoints**: save just `{model, config}`.
- **Robustness**: added a `torch.isnan(loss)` guard that zeroes grads.
- **Logging**: dropped `tokens/sec` (not useful when batches are variable-length padded sequences)

**DDP support** is unchanged from pretraining.

---

## Stage 2: Reinforcement Learning from Human Feedback (`rlhf.py`)

After SFT, the model follows the assistant format but isn't optimized for response quality. RLHF keeps the same model architecture but replaces the supervised objective with a policy gradient signal: the reward model scores the full completion, and that reward is distributed back across the completion tokens via the policy gradient loss.

Therefore, we sample `G=16` completions per prompt, score them with OpenAssistant's DeBERTa-v3 reward model, and update the policy via Group Relative Policy Optimization (GRPO) with a KL penalty to stay close to the SFT reference.

#### Dataset

[`AlpacaPromptDataset`](data.py): same Alpaca source as SFT, but only prompts,completions are generated at rollout time, not loaded from disk.

---

### Implementations

#### 1. Reward Model

Uses OpenAssistant's pretrained (DeBERTa-v3)[https://huggingface.co/OpenAssistant/reward-model-deberta-v3-base] to score prompt-completion pairs:

```python
# rlhf.py
RM_NAME = "OpenAssistant/reward-model-deberta-v3-base"

rm_tokenizer = AutoTokenizer.from_pretrained(RM_NAME)
rm_model = AutoModelForSequenceClassification.from_pretrained(
    RM_NAME, torch_dtype=torch.bfloat16
).to(device).eval()
```

Scoring strips everything after the stop token before passing to DeBERTa:

#### 2. Left-Padded Prompt Dataset

Only prompts are loaded! Completions are generated at rollout time:

```python
# data.py
class AlpacaPromptDataset(Dataset):
    def __init__(self, enc_extended, max_prompt_length=256, split="train", ...):
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        self.prompts = []
        for row in ds:
            text = (
                f"<|im_start|>user\n{row['instruction']}"
                + (f"\n{row['input']}" if row["input"] else "")
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            tokens = enc_extended.encode(text, allowed_special={...})
            if len(tokens) <= max_prompt_length:
                self.prompts.append(torch.tensor(tokens, dtype=torch.long))

    def __getitem__(self, idx):
        return self.prompts[idx]


def collate_prompts(batch):
    max_len = max(p.size(0) for p in batch)
    padded = torch.full((len(batch), max_len), fill_value=EOS, dtype=torch.long)
    for i, p in enumerate(batch):
        padded[i, max_len - p.size(0):] = p  # Left-pad
    return padded
```

**Why left-pad:** Generation always starts from the same position relative to the sequence end, making `model.generate()` deterministic across variable-length prompts.

#### 3. Group Relative Policy Optimization (GRPO)

Each prompt generates `G=16` completions at rollout time, advantages are computed relative to the group:

**Rollout:**
```python
# rlhf.py
expanded = prompt_batch.repeat_interleave(group_size, dim=0)  # (B*G, T)

with torch.no_grad():
    sequence_ids = raw_model.generate(
        expanded, max_new_tokens=128, temperature=0.8, top_k=50,
        stop_tokens=(IM_END, EOS),
    )
```

**Advantages:**
```python
# rlhf.py
def grpo_advantages(rewards, group_size, eps=1e-4):
    r = rewards.view(-1, group_size)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + eps)
    return adv.view(-1)
```

Group normalization prevents reward hacking, since rewards are always relative to the other completions in the group, the model can't game the RM by outputting high-scoring arbitrary text.

**Fixed-length loss normalization:** Dividing by actual completion length upweights short completions. A 20-token response gets ~13× more gradient per token than a 256-token one, inadvertently rewarding terseness. Instead, `policy_gradient_loss` and `kl_penalty_k3` always divide by `L_max = max_new_tokens`, so short completions are naturally downweighted (smaller masked sum, same denominator):

```python
# rlhf.py
def policy_gradient_loss(..., fixed_len_norm=False, L_max=128):
    ...
    denom = L_max if fixed_len_norm else mask.sum(dim=1).clamp(min=1.0)
    loss = -((surrogate * mask).sum(dim=1) / denom).mean()

def kl_penalty_k3(..., fixed_len_norm=False, L_max=128):
    ...
    denom = L_max if fixed_len_norm else mask.sum(dim=1).clamp(min=1.0)
    return ((per_token_kl * mask).sum(dim=1) / denom).mean()
```

**Action mask:** Only completion tokens contribute to the loss:
```python
# rlhf.py
def build_action_mask(sequence_ids, prompt_len, stop_tokens=(IM_END, EOS)):
    B, T = sequence_ids.shape
    mask = torch.zeros(B, T - 1, dtype=torch.bool, device=sequence_ids.device)
    for i, comp in enumerate(sequence_ids[:, prompt_len:].tolist()):
        stop_pos = next((j for j, t in enumerate(comp) if t in stop_tokens), len(comp))
        n_keep = min(stop_pos + 1, len(comp))
        mask[i, prompt_len - 1: prompt_len - 1 + n_keep] = True
    return mask
```

#### 4. Policy Gradient Loss with KL Penalty

PPO-style clipped surrogate loss with a KL penalty to prevent the policy from drifting too far from the SFT reference:

```python
# rlhf.py
def policy_gradient_loss(log_probs_new, log_probs_old, advantages, action_mask, clip_eps=0.2):
    ratio = torch.exp(log_probs_new - log_probs_old.detach())
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surrogate = torch.min(ratio * adv, clipped * adv)
    mask = action_mask.float()
    seq_lengths = mask.sum(dim=1).clamp(min=1.0)
    return -((surrogate * mask).sum(dim=1) / seq_lengths).mean()

def kl_penalty_k3(log_probs_new, log_probs_ref, action_mask):
    log_ratio = log_probs_ref - log_probs_new
    per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
    mask = action_mask.float()
    seq_lengths = mask.sum(dim=1).clamp(min=1.0)
    return ((per_token_kl * mask).sum(dim=1) / seq_lengths).mean()

loss = pg_loss + beta_kl * kl_loss  # beta_kl = 0.2
```

The reference model keeps SFT weights frozen so the KL term has a fixed target:

```python
# rlhf.py
ref_model, _ = load_checkpoint(checkpoint_file, device, weights_only=True)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)
```

#### 5. Training Loop Logic

Recycled from `sft.py`, the changes are:

- **Goal**: starts from the SFT checkpoint instead of the pretrained base.
- **Data**: `AlpacaPromptDataset` with `collate_prompts`, left-padded prompts only.
- **Rollout**: each step generates `G=16` completions per prompt before any gradient update.
- **Inner loop**: 2 PPO-style gradient steps per rollout — reuses the same samples to squeeze more signal before discarding them.
- **Gradient accumulation**: `batch_size=1` prompt × `group_size=16` completions = 16 completions per micro-step. A `sim_batch_size=64` sets `grad_accum_steps=4`, so 4 independent micro-batches are collected during rollout before any gradient update. GRPO advantages are still computed per-group (within each prompt's 16 completions), preserving the group-relative property, but the gradient itself is smoother because it spans more prompts per step.
- **Hyperparameters**: `max_lr=2e-6`, `min_lr=2e-7`, `warmup_steps=300`, `B=1`, `group_size=16`, `sim_batch_size=64`, `grad_accum_steps=4`, `beta_kl=0.4`, 1 epoch.

**DDP support** is unchanged from pretraining, with **Slow Merged Generation**.

---

