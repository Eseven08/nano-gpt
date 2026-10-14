"""test_smoke.py -- end-to-end sanity: shapes, learning, and sampling.

Run:  python tests/test_smoke.py

Trains a tiny model on a toy corpus and asserts that
  1. forward produces the right logits shape,
  2. training actually decreases the loss (the gradients are usable),
  3. sampling produces exactly `length` characters, all in the vocabulary,
  4. greedy / temperature / top-k sampling all run.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_gpt import model as M  # noqa: E402
from nano_gpt.data import CharTokenizer, Dataset  # noqa: E402
from nano_gpt.model import GPT, softmax_cross_entropy  # noqa: E402
from nano_gpt.sampler import generate  # noqa: E402
from nano_gpt.trainer import Adam  # noqa: E402

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the fox likes the dog. the dog chases the fox. "
    "quick brown foxes jump. lazy dogs sleep. "
    "foxes and dogs run over the hill. the sun sets, the fox rests. "
) * 40


def main():
    print("=" * 62)
    print("Smoke tests (shapes / learning / sampling)")
    print("=" * 62)

    M.DTYPE = np.float32
    tokenizer = CharTokenizer(CORPUS)
    dataset = Dataset(tokenizer, CORPUS, block_size=32, batch_size=8, seed=0)
    model = GPT(vocab_size=tokenizer.vocab_size, d_model=16, n_head=2,
                n_layer=2, block_size=32, rng=np.random.default_rng(0))

    # 1. forward shape
    x, y = dataset.get_batch("train")
    logits = model.forward(x, {})
    assert logits.shape == (8, 32, tokenizer.vocab_size), logits.shape
    print("[PASS] forward logits shape (B, T, V)")

    # 2. training decreases the loss
    optim = Adam(model.parameters(), lr=1e-3)
    first_loss = None
    for _ in range(200):
        x, y = dataset.get_batch("train")
        cache = {}
        loss, dlogits = softmax_cross_entropy(model.forward(x, cache), y)
        if first_loss is None:
            first_loss = loss
        optim.zero_grad()
        model.backward(dlogits, cache)
        optim.step()
    assert loss < first_loss * 0.5, f"loss did not drop: {first_loss:.4f} -> {loss:.4f}"
    print(f"[PASS] training decreases loss: {first_loss:.4f} -> {loss:.4f}")

    # 3. greedy sampling: exact length, all in-vocabulary
    out = generate(model, tokenizer, prompt="the fox", length=40,
                   temperature=0.0, top_k=None, seed=1)
    assert len(out) == len("the fox") + 40, len(out)
    assert all(c in tokenizer.stoi for c in out), "out-of-vocabulary character"
    print(f"[PASS] greedy sampling: length ok, all in-vocab -> {out[:50]!r}")

    # 4. temperature / top-k sampling
    for kwargs in [{"temperature": 1.0}, {"top_k": 5}, {"temperature": 0.8, "top_k": 10}]:
        out = generate(model, tokenizer, prompt="", length=25, seed=2, **kwargs)
        assert len(out) == 25 and all(c in tokenizer.stoi for c in out)
    print("[PASS] temperature / top-k sampling all run")

    print("\nAll smoke tests PASSED.")


if __name__ == "__main__":
    main()
