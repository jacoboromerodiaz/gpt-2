"""
https://github.com/openai/gpt-2/blob/master/src/encoder.py
"""

import unicodedata


def get_counts(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):  # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")  # escape
    return "".join(chars)


def render_token(t: bytes) -> str:
    s = t.decode("utf-8", errors="replace")
    s = replace_control_characters(s)
    return s


class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def encode(self, text):
        if not self.merges:
            print("[WARN] Tokenizer not trained: using byte-level encoding.")

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_counts(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        if not self.merges:
            print("[WARN] Tokenizer not trained: decoding with base byte vocabulary.")

        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def train(self, text, vocab_size=276, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_counts(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {i+1}/{num_merges}: {pair} -> {idx} "
                    f"({vocab[idx]}) had {stats[pair]} occurrences"
                )

        self.merges = merges
        self.vocab = vocab

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix: str) -> None:
        model_path = f"{file_prefix}.model"
        vocab_path = f"{file_prefix}.vocab"

        self._save_model_file(model_path)
        self._save_vocab_file(vocab_path)

    def _save_model_file(self, model_path: str) -> None:
        with open(model_path, "w", encoding="utf-8") as file:
            file.write("minbpe v1\n")
            file.write(f"{self.pattern}\n")
            file.write(f"{len(self.special_tokens)}\n")

            for token_text, token_id in self.special_tokens.items():
                file.write(f"{token_text} {token_id}\n")

            for left_id, right_id in self.merges:
                file.write(f"{left_id} {right_id}\n")

    def _save_vocab_file(self, vocab_path: str) -> None:
        merge_parents_by_token_id = {
            token_id: pair for pair, token_id in self.merges.items()
        }

        with open(vocab_path, "w", encoding="utf-8") as file:
            for token_id, token_bytes in self.vocab.items():
                rendered_token = render_token(token_bytes)

                if token_id in merge_parents_by_token_id:
                    left_id, right_id = merge_parents_by_token_id[token_id]
                    left_token = render_token(self.vocab[left_id])
                    right_token = render_token(self.vocab[right_id])
                    file.write(
                        f"[{left_token}][{right_token}] -> "
                        f"[{rendered_token}] {token_id}\n"
                    )
                else:
                    file.write(f"[{rendered_token}] {token_id}\n")

    def load(self, model_file: str) -> None:
        if not model_file.endswith(".model"):
            raise ValueError("model_file must end with '.model'")

        with open(model_file, "r", encoding="utf-8") as file:
            version = file.readline().strip()
            pattern = file.readline().strip()
            special_tokens_count = int(file.readline().strip())

            self._validate_version(version)

            special_tokens = self._read_special_tokens(
                file, num_special_tokens=special_tokens_count
            )
            merges = self._read_merges(file)

        self.pattern = pattern
        self.special_tokens = special_tokens
        self.merges = merges
        self.vocab = self._build_vocab()

    def _validate_version(self, version: str) -> None:
        if version != "minbpe v1":
            raise ValueError(f"Unsupported model version: {version}")

    def _read_special_tokens(self, file, num_special_tokens: int) -> dict[str, int]:
        special_tokens = {}

        for _ in range(num_special_tokens):
            token_text, token_id = file.readline().strip().split()
            special_tokens[token_text] = int(token_id)

        return special_tokens

    def _read_merges(self, file) -> dict[tuple[int, int], int]:
        merges = {}
        next_token_id = 256

        for line in file:
            left_id, right_id = map(int, line.split())
            merges[(left_id, right_id)] = next_token_id
            next_token_id += 1

        return merges
