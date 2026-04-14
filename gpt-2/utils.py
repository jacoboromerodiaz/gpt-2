import argparse
import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT with YAML config")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_torch_dtype(dtype_str):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported autocast dtype: {dtype_str}. "
            f"Choose from {list(dtype_map.keys())}."
        )
    return dtype_map[dtype_str]


def unwrap_model(model):
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model
