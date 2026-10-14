# nano-gpt

A **minimal GPT — a decoder-only Transformer for character-level text generation — written entirely from scratch in NumPy.**

No PyTorch. No autograd. Every forward pass and every backward pass (through
LayerNorm, multi-head causal attention, GeLU MLPs, embeddings, and the softmax
cross-entropy loss) is hand-derived from the chain rule and **numerically
verified against central-difference gradients** in `tests/test_gradient.py`.

This is a companion project to
[mini-dl-framework](https://github.com/Eseven08/mini-dl-framework) (an
autograd engine). Here we take the opposite route: we derive the gradients by
hand, so nothing is hidden behind an engine.

```
prompt: "To be, or not to be"
  ┌──────────────────────────────────────────────────────────────┐
  │ token + position embeddings                                  │
  │   ┌──────────────────────────── block ×N ─────────────────┐  │
  │   │  LayerNorm → Multi-Head Causal Attention ──┐           │  │
  │   │  residual add ◄────────────────────────────┘           │  │
  │   │  LayerNorm → MLP (GeLU) ──┐                            │  │
  │   │  residual add ◄──────────┘                            │  │
  │   └────────────────────────────────────────────────────────┘  │
  │ LayerNorm → linear head → softmax over vocabulary             │
  └──────────────────────────────────────────────────────────────┘
  sample the next character → feed it back as input → repeat
```

## Features

- **Decoder-only Transformer** (GPT-style): token + positional embeddings,
  N transformer blocks (multi-head *causal* self-attention + position-wise
  MLP with GeLU), final LayerNorm, linear head.
- **Hand-written backward passes** for every layer — no autograd engine.
- **Adam optimizer** with bias correction, implemented from scratch.
- **Character-level tokenizer**: vocabulary = the unique characters of your
  corpus; swap any UTF-8 text file and retrain.
- **Sampling knobs**: temperature and top-k, plus greedy decoding.
- **Checkpointing** to `.npz` so you can train once and generate many times.
- **Verified**: gradient checks (float64, tight tolerance + float32 sanity),
  shape checks, loss-descent check, sampling checks — all in `tests/`.
- Only dependency: **NumPy**. Runs on any CPU.

## Quickstart

```bash
pip install numpy

# train (a few minutes on CPU; flags match the results below)
python train.py --data data/tinyshakespeare.txt --steps 2000 --block-size 64

# generate with a trained checkpoint
python generate.py --checkpoint checkpoints/model.npz \
    --prompt "To be, or not to be" --length 300
```

## Usage

### train.py

| flag | default | meaning |
|---|---|---|
| `--data` | `data/tinyshakespeare.txt` | UTF-8 corpus to learn from |
| `--steps` | 3000 | number of optimizer steps |
| `--batch-size` | 32 | sequences per batch |
| `--block-size` | 96 | context length (tokens the model can "see") |
| `--d-model` | 96 | hidden dimension |
| `--n-head` | 3 | attention heads |
| `--n-layer` | 4 | transformer blocks |
| `--lr` | 3e-4 | Adam learning rate |
| `--dtype` | `float32` | `float32` (fast) or `float64` |
| `--out` | `checkpoints/model.npz` | checkpoint path |

The training loop prints train loss every `--log-every` steps and validation
loss + a live text sample every `--val-every` steps, so you can watch the
model get better in real time.

### generate.py

| flag | default | meaning |
|---|---|---|
| `--checkpoint` | `checkpoints/model.npz` | trained model |
| `--prompt` | `"To be, or not to be"` | seed text |
| `--length` | 300 | characters to generate |
| `--temperature` | 0.8 | lower = more conservative, higher = more creative |
| `--top-k` | 40 | only sample among the k most likely characters |

## Bring your own corpus

Any UTF-8 `.txt` file works — a novel, song lyrics, your notes, code:

```bash
python train.py --data my_book.txt --block-size 128 --steps 4000
python generate.py --prompt "Chapter 1" --length 500
```

The tokenizer is built from the corpus automatically: more unique characters
→ larger vocabulary → slower but more expressive generation.

## How the gradients work (the short version)

Every layer implements:

```python
forward(x, cache)  -> y
backward(dy, cache) -> dx
```

and `backward` also accumulates gradients into the layer's parameters. The
non-obvious ones:

- **LayerNorm** — `y = (x - μ)/σ·g + b`. Because μ and σ depend on the whole
  vector, the backward pass subtracts the mean of the gradient and a
  `x̂ · mean(d̂·x̂)` correction term:
  `dx = (d̂ - mean(d̂) - x̂·mean(d̂·x̂)) / σ`.
- **Causal attention** — `ctx = softmax(QKᵀ/√d)·V`. The softmax backward is
  `dS = P·(dP - Σ dP·P)`, then `dQ = dS·K/√d`, `dK = dSᵀ·Q/√d`,
  `dV = Pᵀ·dctx`. The causal mask simply zeroes the masked entries (their
  softmax probability is 0).
- **GeLU** (sigmoid approximation) — `y = x·σ(1.702x)`, so
  `dy/dx = σ(1.702x) + 1.702x·σ(1.702x)·(1 - σ(1.702x))`.

The tests compare every one of these against numerical central-difference
gradients on a full model — a wrong sign, wrong scale, or wrong broadcast
cannot slip through.

## Project layout

```
nano-gpt/
├── nano_gpt/
│   ├── __init__.py
│   ├── data.py       # char tokenizer, corpus loading, batching
│   ├── model.py      # layers with forward/backward, GPT, cross-entropy
│   ├── trainer.py    # Adam optimizer
│   └── sampler.py    # greedy / temperature / top-k generation
├── train.py          # training CLI
├── generate.py       # generation CLI
├── data/
│   └── tinyshakespeare.txt   # built-in public-domain corpus (~1.1 MB)
├── tests/
│   ├── test_gradient.py      # numerical gradient verification (float64/32)
│   └── test_smoke.py         # shapes, learning, sampling sanity
└── checkpoints/              # created by train.py (gitignored)
```

## Results

Trained on tiny Shakespeare (1,115,394 characters, vocabulary of 65
characters) with `d=96, 3 heads, 4 layers, block=64, batch=32, Adam lr=3e-4`.

| metric | value |
|---|---|
| parameters | 466,241 |
| train loss (start) | ~4.17 (random guessing, ln 65) |
| final train loss | **1.6729** (step 2000) |
| final validation loss | **1.8421** |

Sample generation (temperature 0.6, top-k 30, seed 1):

```
To be, or not to be were to and the hast
True have stired and it thee, and this and many.

Secent:
Then, the see had likes the you dom mear
To make in with will death, as it the throw!

Thy lord, my they hungere the mak
```

Rough around the edges — exactly what a 466k-parameter model trained for
~15 minutes on a CPU looks like, and a great baseline to beat. The loss
curve drops from 4.20 to 1.67, and the text visibly moves from random
gibberish to word-like Shakespearean fragments.

## Why this project

Large language models are usually treated as a black box. `nano-gpt` opens
that box: with a few hundred lines of readable NumPy you can trace exactly
where every gradient comes from, watch attention patterns learn, and generate
your own Shakespeare — all on a laptop CPU. It is an ideal starting point for
reading the real GPT literature (Attention Is All You Need, GPT-2/3 papers,
nanoGPT) afterward.

## Extensions / ideas

- Add `top-p` (nucleus) sampling and a `--seed` for reproducible demos.
- Implement KV-cache sampling for much faster generation.
- Add a small learning-rate schedule (warmup + cosine decay) — typically
  worth a solid loss improvement.
- Train on more text (or a bigger model) and compare loss curves.
- Write an attention-pattern visualizer to "see" what each head learns.

## License

MIT — see [LICENSE](LICENSE). Built for learning; use it however you like.
