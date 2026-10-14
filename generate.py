"""generate.py -- load a trained checkpoint and generate text.

Example:
    python generate.py --checkpoint checkpoints/model.npz \
        --prompt "To be, or not to be" --length 300

Sampling knobs:
    --temperature  lower = more deterministic, higher = more creative
    --top-k        restrict sampling to the k most likely next characters
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from nano_gpt import model as M
from nano_gpt.data import CharTokenizer
from nano_gpt.model import GPT
from nano_gpt.sampler import generate


def parse_args():
    ap = argparse.ArgumentParser(description="Generate text with a trained nano GPT.")
    ap.add_argument("--checkpoint", default="checkpoints/model.npz")
    ap.add_argument("--prompt", default="To be, or not to be")
    ap.add_argument("--length", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_checkpoint(path: str):
    data = np.load(path, allow_pickle=False)
    config = json.loads(str(data["config_json"][0]))
    tokenizer = CharTokenizer("".join(str(c) for c in data["chars"]))

    M.DTYPE = np.float64 if config.get("dtype", "float32") == "float64" else np.float32
    model = GPT(vocab_size=tokenizer.vocab_size,
                d_model=config["d_model"], n_head=config["n_head"],
                n_layer=config["n_layer"], block_size=config["block_size"])
    for name, p in model.parameters():
        p.data = data[name]
    return model, tokenizer, config


def main():
    args = parse_args()
    model, tokenizer, config = load_checkpoint(args.checkpoint)
    print(f"loaded checkpoint: {args.checkpoint} "
          f"({config['n_layer']}L/{config['n_head']}H, d={config['d_model']})")

    out = generate(model, tokenizer, prompt=args.prompt, length=args.length,
                   temperature=args.temperature, top_k=args.top_k, seed=args.seed)
    print("\n" + "=" * 62)
    print(f"generated {args.length} characters (temperature={args.temperature}, "
          f"top_k={args.top_k}):")
    print("=" * 62)
    print(out)


if __name__ == "__main__":
    main()
