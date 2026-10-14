"""nano_gpt -- a minimal GPT (decoder-only Transformer) written from scratch in NumPy.

This package implements a character-level generative language model with NO
autograd engine: every backward pass is hand-derived from the chain rule and
numerically verified (see ``tests/test_gradient.py``). It is meant to be read:
the whole training stack -- data, model, optimizer, sampler -- is a few
hundred lines of plain Python.

Modules
-------
data     : character tokenizer, corpus loading, random-context batching
model    : Transformer layers with forward/backward, GPT, cross-entropy loss
trainer  : Adam optimizer
sampler  : autoregressive text generation (temperature / top-k)
"""

from .model import GPT, LayerNorm, CausalSelfAttention, MLP, Block, softmax_cross_entropy
from .data import CharTokenizer, Dataset
from .trainer import Adam
from .sampler import generate

__version__ = "0.1.0"
