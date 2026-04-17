<div align="center">
  <img src="assets/repo_logo.png" width="320px"/>

# ChatGPT-2 (124M): from Pretraining to Instruct

A full LLM pipeline built from scratch: raw text → pretrained base → SFT → instruct model.

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-in%20progress-orange)

</div>

This follows [Karpathy's](https://www.youtube.com/@AndrejKarpathy) nanoGPT/GPT-tokenizer series as a foundation, then extends it into a complete post-training pipeline: tokenizer, supervised fine-tuning on instruction data, and an in-progress RLHF stage with a learned reward model.

The goal was to understand every layer of the stack by building it, not importing it.

---

## Pipeline

```
 BPE Tokenizer → FineWeb-Edu 10B → Pretrained Base → SFT (Alpaca) → [RLHF in progress] → Instruct Model
```

---

## Pretraining

Architecture and data pipeline follow Karpathy's nanoGPT, trained on FineWeb-Edu 10B with GPT-3 paper hyperparameters. Supports single-GPU, multi-GPU (DDP), and Apple Silicon (MPS).

**Results after 1 epoch:**

| Metric | Value |
|---|---|
| HellaSwag | 0.3066 |
| Val loss | 3.0155 |
| Min train loss | 2.7657 |

**Extensions over the baseline:**

- Seasonality-aware data sampling to reduce distribution skew [`374e541`](https://github.com/jacoboromerodiaz/gpt-2/commit/374e541)
- Unwrapping the model before evaluation to allow `torch.compile` to eval HellaSwag cutting eval time [`0d4bfa7`](https://github.com/jacoboromerodiaz/gpt-2/commit/0d4bfa7)
- Checkpoint resume [`c555f96`](https://github.com/jacoboromerodiaz/gpt-2/commit/c555f96)

---

## Tokenizer

Byte Pair Encoding implemented from scratch on the [`tokenizer`](https://github.com/jacoboromerodiaz/gpt-2/tree/tokenizer) branch, no libraries. Replicates the exact algorithm used in GPT-2: iterative pair merging over a UTF-8 byte vocabulary. Built to understand the internals, not to replace tiktoken.

---

## Fine-Tuning

Inspired by Karpathy's [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI&t=10811s). The pretrained base is extended into an instruct model through two post-training stages.

### Stage 1 — Supervised Fine-Tuning

Fine-tuned on the Alpaca instruction dataset to shift the model from a next-token predictor to one that follows the assistant format. This is the part of the pipeline I implemented independently of the tutorial — dataset formatting, training loop modifications, and prompt templating.

### Stage 2 — RLHF (in progress)

Training a reward model on human preference data, then using PPO to optimize the SFT model against that signal. The goal is to close the gap between "responds in assistant format" and "responds helpfully."

---

## Infrastructure

Full pretraining on [Vast.ai](https://vast.ai) for ~$7.50 (~37 h of training).

| Resource | Config |
|---|---|
| GPU | 1× 24 GB VRAM |
| Storage | 64 GB (FineWeb-Edu dataset + checkpoints) |
| Pricing | ~$0.201/h on-demand |
| Total cost | ~$7.50 |

Use **on-demand** (not interruptible) to avoid losing a mid-epoch checkpoint.

---

### Option A — Docker template (recommended)

Select the **Docker** template on Vast.ai and point it to this repo's image. The container clones the repo, installs dependencies via `uv`, and downloads the pretrained checkpoint from Google Drive automatically.

```dockerfile
FROM python:3.11

RUN apt-get update && apt-get install -y curl git

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

RUN git clone https://github.com/jacoboromerodiaz/gpt-2.git /workspace/gpt-2

WORKDIR /workspace/gpt-2

ENV PATH="/workspace/gpt-2/.venv/bin:$PATH"

RUN uv sync

RUN uv pip install gdown

RUN gdown --id [YOUR_DRIVE_FILE_ID] -O /workspace/gpt-2/finetune/log/model.pt
```

---

### Option B — PyTorch template

Select the **PyTorch** template, then run:

```bash
git clone https://github.com/jacoboromerodiaz/gpt-2.git
cd gpt-2

curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# download the pretrained checkpoint from drive
uv pip install gdown
gdown [YOUR_DRIVE_FILE_ID] -O finetune/log/model.pt
```

---

### Running pretraining from scratch

The setup above pulls the pretrained checkpoint and is ready for fine-tuning. To reproduce pretraining from scratch, upload the FineWeb-Edu dataset shards to Google Drive and download them into the container the same way:

```bash
gdown [YOUR_DATASET_DRIVE_ID] -O data/
```

Then update the data path in `data.py` to point to your local shard directory and run:

```bash
python train.py
```

> **Note:** path configuration will be moved to a `config.yaml` file in a future update.