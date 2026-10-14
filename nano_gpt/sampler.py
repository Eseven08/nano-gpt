"""nano_gpt.sampler -- autoregressive text generation.

Generation feeds the model's own previous output back in as input, one
character at a time. ``temperature`` reshapes the softmax (lower = greedier,
higher = more random) and ``top_k`` restricts the choice to the k most
probable characters.
"""

from __future__ import annotations

import numpy as np


def generate(model, tokenizer, prompt: str = "", length: int = 300,
             temperature: float = 1.0, top_k: int | None = None, seed: int | None = None) -> str:
    rng = np.random.default_rng(seed)
    vocab = model.vocab_size

    if prompt:
        ids = tokenizer.encode(prompt)
        if len(ids) == 0:
            raise ValueError("prompt must contain at least one in-vocabulary character")
        ids = ids[-(model.block_size):]
    else:
        ids = [tokenizer.stoi.get("\n", 0)]   # internal seed; not part of the output

    for _ in range(length):
        idx = np.array(ids[-model.block_size:], dtype=np.int64)[None, :]   # (1, T)
        logits = model.forward(idx, {})[0, -1].astype(np.float64)          # (V,)

        if temperature <= 0:
            # Greedy decoding: always take the most likely character.
            next_id = int(np.argmax(logits))
            ids.append(next_id)
            continue

        if temperature != 1.0:
            logits = logits / temperature
        if top_k is not None:
            k = min(int(top_k), vocab)
            threshold = np.sort(logits)[-k]
            logits = np.where(logits < threshold, -np.inf, logits)

        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        ids.append(int(rng.choice(vocab, p=probs)))

    text = tokenizer.decode(ids)
    if not prompt and len(ids) > 1:
        text = text[1:]   # drop the artificial seed character
    return text
