import os
import numpy as np
import torch
import tiktoken

# TODO implement our own tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]


def load_tokens(file):
    npt = np.load(file)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt


class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, split, master_process=False):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}, "Invalid split"
        self.rng = np.random.default_rng(1337)

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
        self.tokens = None
        self.current_pos = 0
        self.reset()

    def load_shard(
        self, filename
    ):  # added from PR: https://github.com/karpathy/build-nanogpt/pull/52/files
        shard = load_tokens(filename)
        if self.split == "train":
            eot_positions = (torch.where(shard == enc.eot_token)[0] + 1).tolist()
            documents = [
                shard[start:end]
                for start, end in zip([0] + eot_positions[:-1], eot_positions)
            ]
            self.rng.shuffle(documents)
            shard = torch.cat(documents)
        return shard

    def set(self, loader_checkpoint):
        self.current_position = loader_checkpoint["current_position"]
        self.current_shard = loader_checkpoint["current_shard"]
        self.rng.bit_generator.state = loader_checkpoint["rng_state"]
        self.tokens = load_tokens(self.shards[self.current_shard])
        if self.current_position + (self.B * self.T * self.num_processes + 1) > len(
            self.tokens
        ):
            self.current_shard += 1
            if self.current_shard == len(self.shards):
                self.reset()
            else:
                self.tokens = self.load_shard(self.shards[self.current_shard])
                self.current_position = self.B * self.T * self.process_rank

    def reset(self):
        self.current_shard = 0
        self.rng.shuffle(self.shards) if self.split == "train" else None
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
