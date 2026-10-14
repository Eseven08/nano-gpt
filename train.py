"""train.py -- train a character-level nano GPT on a text corpus.

Example:
    python train.py --data data/tinyshakespeare.txt --steps 3000

The model and tokenizer are saved to --out (an .npz checkpoint) so that
generate.py can load them later.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from nano_gpt import model as M
from nano_gpt.data import CharTokenizer, Dataset, load_text
from nano_gpt.model import GPT, softmax_cross_entropy
from nano_gpt.sampler import generate
from nano_gpt.trainer import Adam


def parse_args():
    ap = argparse.ArgumentParser(description="Train a character-level nano GPT.")
    ap.add_argument("--data", default="data/tinyshakespeare.txt", help="UTF-8 text corpus")
    ap.add_argument("--steps", type=int, default=3000, help="number of training steps")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=96, help="context length")
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-head", type=int, default=3)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log-every", type=int, default=200, help="print train loss every N steps")
    ap.add_argument("--val-every", type=int, default=500, help="print val loss + sample every N steps")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/model.npz", help="checkpoint output path")
    return ap.parse_args()


def save_checkpoint(path: str, model: GPT, tokenizer: CharTokenizer, config: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {name: p.data for name, p in model.parameters()}
    arrays["chars"] = np.array(tokenizer.chars, dtype="<U1")
    arrays["config_json"] = np.array([json.dumps(config)])
    np.savez(path, **arrays)
    print(f"saved checkpoint -> {path}")


def main():
    args = parse_args()
    M.DTYPE = np.float64 if args.dtype == "float64" else np.float32

    print("=" * 62)
    print("nano-gpt: a character-level GPT in pure NumPy")
    print("=" * 62)

    text = load_text(args.data)
    tokenizer = CharTokenizer(text)
    dataset = Dataset(tokenizer, text, block_size=args.block_size,
                      batch_size=args.batch_size, seed=args.seed)
    print(f"corpus     : {args.data} ({len(text):,} chars)")
    print(f"vocab size : {tokenizer.vocab_size} unique characters")

    rng = np.random.default_rng(args.seed)
    model = GPT(vocab_size=tokenizer.vocab_size, d_model=args.d_model,
                n_head=args.n_head, n_layer=args.n_layer,
                block_size=args.block_size, rng=rng)
    print(f"model      : {args.n_layer} layers, {args.n_head} heads, "
          f"d={args.d_model}, block={args.block_size}")
    print(f"parameters : {model.num_params():,}  (dtype {args.dtype})")

    optim = Adam(model.parameters(), lr=args.lr)
    t0 = time.time()
    best_val = float("inf")

    for step in range(1, args.steps + 1):
        x, y = dataset.get_batch("train")
        cache = {}
        logits = model.forward(x, cache)
        loss, dlogits = softmax_cross_entropy(logits, y)
        optim.zero_grad()
        model.backward(dlogits, cache)
        optim.step()

        if step % args.log_every == 0 or step == 1:
            el = time.time() - t0
            print(f"step {step:>6d} | train loss {loss:.4f} | "
                  f"{(step * args.batch_size * args.block_size / el) / 1e3:,.0f}k tok/s")

        if step % args.val_every == 0 or step == args.steps:
            vx, vy = dataset.get_batch("val")
            vloss, _ = softmax_cross_entropy(model.forward(vx, {}), vy)
            best_val = min(best_val, float(vloss))
            sample = generate(model, tokenizer, prompt="To be", length=120,
                              temperature=0.8, top_k=30, seed=0)
            print(f"step {step:>6d} | val loss   {vloss:.4f} (best {best_val:.4f})")
            print("  sample: " + sample.replace("\n", " / ")[:160])

    config = vars(args)
    save_checkpoint(args.out, model, tokenizer, config)
    print(f"done in {time.time() - t0:.1f}s | best val loss {best_val:.4f}")


if __name__ == "__main__":
    main()
