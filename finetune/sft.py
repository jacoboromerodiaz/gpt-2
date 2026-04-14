from datasets import load_dataset
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

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


def collate_fn(batch):
    xs, ys = zip(*batch)

    max_len = max(x.size(0) for x in xs)

    xs_pad = torch.zeros(len(xs), max_len, dtype=torch.long)
    ys_pad = torch.full((len(ys), max_len), fill_value=-100, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        xs_pad[i, : x.size(0)] = x
        ys_pad[i, : y.size(0)] = y

    return xs_pad, ys_pad


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


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    device = "mps"
    device_type = "cpu"
    enc_extended = extend_encoder(enc)

    checkpoint_file = "/Users/jacoboromerodiaz/Projects/gpt-2/gpt2/best_model.pt"

    model, optimizer, checkpoint = load_checkpoint(checkpoint_file, device, device_type)

    dataset = AlpacaDataset(enc_extended)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    x, y = next(iter(loader))
    print(x.shape, y.shape)
    print(x[0])
