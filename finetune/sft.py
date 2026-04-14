from datasets import load_dataset
import tiktoken
import torch
from torch.nn import nn
from torch.utils.data import Dataset

from gpt2.model import GPT, GPTConfig
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


def load_checkpoint(path, device, device_type):
    checkpoint = torch.load(path, map_location=device)
    gpt_fields = {f.name for f in fields(GPTConfig)}
    config_kwargs = {k: v for k, v in checkpoint["config"].items() if k in gpt_fields}
    config = GPTConfig(**config_kwargs)

    model = GPT(config)
    model.to(device)

    # artifact of torch.compile
    def _remove_unwanted_prefix(state_dict):
        unwanted_prefix = "_orig_mod."
        clean_state_dict = {
            (k[len(unwanted_prefix) :] if k.startswith(unwanted_prefix) else k): v
            for k, v in state_dict.items()
        }
        return clean_state_dict

    model_state_dict = _remove_unwanted_prefix(checkpoint["model"])
    model.load_state_dict(model_state_dict, strict=False)

    opt_state = checkpoint["optimizer"]
    lr = opt_state["param_groups"][0]["lr"]
    wd = opt_state["param_groups"][0]["weight_decay"]
    optimizer = model.configure_optimizer(weight_decay=wd, lr=lr, device=device_type)
    opt_state_dict = _remove_unwanted_prefix(checkpoint["optimizer"])
    optimizer.load_state_dict(opt_state_dict)

    return model, optimizer, checkpoint


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


def extend_embd(model):
    wte = model.transformer.wte
    n_embd = model.transformer.wte.weight.shape[1]

    extended_wte = nn.Embedding(50259, n_embd)
    extended_wte.weight.data[:50257] = wte.weight.data
    model.transformer.wte = nn.Embedding(50259, n_embd)

    with torch.no_grad():
        mean_emb = model.transformer.wte.weight[:50257].mean(0)
        model.transformer.wte.weight[50257] = mean_emb  # <|im_start|>
        model.transformer.wte.weight[50258] = mean_emb  # <|im_end|>
    print("Model loaded and embeddings extended")


def format_example(enc, row):
    text = (
        f"<|im_start|>user\n{row['instruction']}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{row['output']}<|im_end|>"
    )
    return enc.encode(text, allowed_special={"<|im_start|>", "<|im_end|>"})


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    device = "mps"
    device_type = "cpu"
    enc_extended = extend_encoder(enc)

    checkpoint_file = "/Users/jacoboromerodiaz/Projects/gpt-2/gpt2/best_model.pt"

    model, optimizer, checkpoint = load_checkpoint(checkpoint_file, device, device_type)
    extended_model = extend_embd(model)
