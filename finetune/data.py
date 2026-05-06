import random

import torch
from torch.utils.data import Dataset

from datasets import load_dataset

IM_START = 50257
IM_END = 50258
EOS = 50256


def find_subsequence(tensor, subseq):
    n, m = tensor.size(0), subseq.size(0)
    for i in range(n - m + 1):
        if torch.all(tensor[i : i + m] == subseq):
            return i + m
    return None


class _SFTDataset(Dataset):
    """Base class"""

    def __init__(
        self, enc_extended, max_length=1024, split="train", val_ratio=0.1, seed=42
    ):
        self.enc_extended = enc_extended

        all_examples = self.load(enc_extended, max_length)
        rng = random.Random(seed)
        rng.shuffle(all_examples)

        n_val = int(len(all_examples) * val_ratio)
        if split == "val":
            self.examples = all_examples[:n_val]
        else:
            self.examples = all_examples[n_val:]

    def load(self, enc_extended, max_length):
        raise NotImplementedError

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        x = tokens[:-1]
        y = tokens[1:].clone()

        assistant_start_seq = torch.tensor(
            [IM_START] + self.enc_extended.encode("assistant\n"), dtype=torch.long
        )
        mask_until = find_subsequence(y, assistant_start_seq)
        if mask_until is None:
            y[:] = -100
        else:
            y[:mask_until] = -100
        return x, y


class AlpacaDataset(_SFTDataset):
    def load(self, enc_extended, max_length):
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        examples = []
        for row in ds:
            text = (
                f"<|im_start|>user\n{row['instruction']}"
                + (f"\n{row['input']}" if row["input"] else "")
                + f"<|im_end|>\n<|im_start|>assistant\n{row['output']}<|im_end|>"
            )
            tokens = enc_extended.encode(
                text, allowed_special={"<|im_start|>", "<|im_end|>"}
            )
            if len(tokens) <= max_length:
                examples.append(torch.tensor(tokens, dtype=torch.long))
        return examples


class DollyDataset(_SFTDataset):
    def load(self, enc_extended, max_length):
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        examples = []
        for row in ds:
            text = (
                f"<|im_start|>user\n{row['instruction']}"
                + (f"\n{row['context']}" if row["context"] else "")
                + f"<|im_end|>\n<|im_start|>assistant\n{row['response']}<|im_end|>"
            )
            tokens = enc_extended.encode(
                text, allowed_special={"<|im_start|>", "<|im_end|>"}
            )
            if len(tokens) <= max_length:
                examples.append(torch.tensor(tokens, dtype=torch.long))
        return examples


def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    xs_pad = torch.full((len(xs), max_len), fill_value=EOS, dtype=torch.long)
    ys_pad = torch.full((len(ys), max_len), fill_value=-100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xs_pad[i, : x.size(0)] = x
        ys_pad[i, : y.size(0)] = y
    return xs_pad, ys_pad


class AlpacaPromptDataset(Dataset):
    """Only the user prompt tokens, used by the RLHF rollout."""

    def __init__(
        self, enc_extended, max_prompt_length=256, split="train", val_ratio=0.1, seed=42
    ):
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        all_prompts = []
        for row in ds:
            text = (
                f"<|im_start|>user\n{row['instruction']}"
                + (f"\n{row['input']}" if row["input"] else "")
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            tokens = enc_extended.encode(
                text, allowed_special={"<|im_start|>", "<|im_end|>"}
            )
            if len(tokens) <= max_prompt_length:
                all_prompts.append(torch.tensor(tokens, dtype=torch.long))

        rng = random.Random(seed)
        rng.shuffle(all_prompts)
        n_val = int(len(all_prompts) * val_ratio)
        self.prompts = all_prompts[:n_val] if split == "val" else all_prompts[n_val:]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def collate_prompts(batch):
    """Left-pad prompts."""
    max_len = max(p.size(0) for p in batch)
    padded = torch.full((len(batch), max_len), fill_value=EOS, dtype=torch.long)
    for i, p in enumerate(batch):
        padded[i, max_len - p.size(0) :] = p
    return padded
