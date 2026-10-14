"""nano_gpt.data -- character tokenizer, corpus loading, and batching.

A character-level language model treats the raw text as a sequence of single
characters, so the vocabulary is just the set of unique characters in the
corpus. ``Dataset`` slices random context windows out of the text, which is
how every mini-batch is produced during training.
"""

from __future__ import annotations

import numpy as np


class CharTokenizer:
    """Map between raw text and integer ids at the character level."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}   # char -> id
        self.itos = {i: c for i, c in enumerate(self.chars)}   # id -> char

    def encode(self, text: str):
        """Text -> list of integer ids (one per character)."""
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        """List of integer ids -> text."""
        return "".join(self.itos[i] for i in ids)


def load_text(path: str) -> str:
    """Read a UTF-8 text file into a single string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class Dataset:
    """Random context windows from a corpus, for training / validation."""

    def __init__(self, tokenizer: CharTokenizer, text: str,
                 block_size: int, batch_size: int, val_frac: float = 0.1,
                 seed: int = 0):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        rng = np.random.default_rng(seed)

        ids = np.array(tokenizer.encode(text), dtype=np.int64)
        n_val = max(1, int(len(ids) * val_frac))
        self.train_ids = ids[:-n_val]
        self.val_ids = ids[-n_val:]

    def get_batch(self, split: str):
        """Return (inputs, targets) of shape (batch_size, block_size).

        targets[t] is the character that should follow inputs[t], i.e. the
        labels are the input window shifted by one position.
        """
        ids = self.train_ids if split == "train" else self.val_ids
        rng = np.random.default_rng()  # fresh generator: batches are i.i.d.
        n = len(ids)
        if n < self.block_size + 1:
            raise ValueError("corpus too short for the requested block size")
        starts = rng.integers(0, n - self.block_size, size=self.batch_size)
        x = np.stack([ids[s:s + self.block_size] for s in starts])
        y = np.stack([ids[s + 1:s + self.block_size + 1] for s in starts])
        return x, y
