"""nano_gpt.model -- a decoder-only Transformer (GPT-style) in pure NumPy.

Every trainable parameter is a ``Param`` (``data`` + accumulated ``grad``).
Every layer exposes two methods:

    forward(x, cache)  -> y
    backward(dy, cache)-> dx

There is deliberately no autograd engine: all backward passes are derived by
hand from the chain rule, which keeps every formula readable and verifiable
(the tests compare them against numerical central-difference gradients).

Shapes
------
x, h           : (B, T, d)      batch, time, hidden
q, k, v, ctx   : (B, H, T, dh)  per-head views of the same data
logits         : (B, T, V)      V = vocabulary size
"""

from __future__ import annotations

import math
import numpy as np

DTYPE = np.float32  # training default; tests override with float64


class Param:
    """A trainable array: ``.data`` is the value, ``.grad`` accumulates."""

    __slots__ = ("data", "grad")

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)


def _randn(shape, rng, scale: float = 0.02):
    return (rng.standard_normal(shape) * scale).astype(DTYPE)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def gelu_forward(x, cache):
    """GeLU with the sigmoid approximation: y = x * sigmoid(1.702 * x)."""
    s = 1.0 / (1.0 + np.exp(-1.702 * x))
    cache["s"] = s
    cache["x"] = x
    return x * s


def gelu_backward(dy, cache):
    """d/dx [x * sigmoid(1.702x)] = s + 1.702 * x * s * (1 - s)."""
    x = cache["x"]
    s = cache["s"]
    return dy * (s + 1.702 * x * s * (1.0 - s))


def softmax_last_axis(x):
    """Numerically stable softmax over the last axis."""
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class LayerNorm:
    """y = (x - mean) / sqrt(var + eps) * g + b, normalized over the last axis."""

    def __init__(self, d: int, rng=None, eps: float = 1e-5):
        self.eps = eps
        self.g = Param(np.ones(d, dtype=DTYPE))
        self.b = Param(np.zeros(d, dtype=DTYPE))

    def forward(self, x, cache):
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = np.mean(xc * xc, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        xhat = xc / std
        cache["xhat"] = xhat
        cache["std"] = std
        return xhat * self.g.data + self.b.data

    def backward(self, dy, cache):
        xhat = cache["xhat"]
        std = cache["std"]
        dhat = dy * self.g.data                       # dL / d xhat
        # Standard LayerNorm backward (see README for the derivation):
        #   dx = (dhat - mean(dhat) - xhat * mean(dhat * xhat)) / std
        dx = (dhat - dhat.mean(axis=-1, keepdims=True)
              - xhat * (dhat * xhat).mean(axis=-1, keepdims=True)) / std
        self.g.grad += np.sum(dy * xhat, axis=(0, 1))
        self.b.grad += np.sum(dy, axis=(0, 1))
        return dx


class CausalSelfAttention:
    """Multi-head scaled dot-product attention with a causal (triangular) mask.

    A token may only attend to itself and earlier tokens, which is what makes
    the model autoregressive: predictions never peek into the future.
    """

    def __init__(self, d: int, n_head: int, block_size: int, rng, scale: float = 0.02):
        assert d % n_head == 0, "d_model must be divisible by n_head"
        self.d = d
        self.n_head = n_head
        self.dh = d // n_head
        self.W_qkv = Param(_randn((d, 3 * d), rng, scale))     # q, k, v side by side
        self.b_qkv = Param(np.zeros(3 * d, dtype=DTYPE))
        self.W_o = Param(_randn((d, d), rng, scale))
        self.b_o = Param(np.zeros(d, dtype=DTYPE))
        # Causal mask: a token may only attend to itself and earlier tokens.
        self.mask = np.triu(np.ones((block_size, block_size), dtype=bool), k=1)

    def forward(self, x, cache):
        B, T, d = x.shape
        H, dh = self.n_head, self.dh
        cache["x"] = x

        qkv = x @ self.W_qkv.data + self.b_qkv.data            # (B, T, 3d)
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, H, dh).transpose(0, 2, 1, 3)       # (B, H, T, dh)
        k = k.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, dh).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(dh)
        scores = q @ k.transpose(0, 1, 3, 2) * scale           # (B, H, T, T)
        scores = np.where(self.mask[:T, :T], -np.inf, scores)  # hide the future
        probs = softmax_last_axis(scores)

        ctx = probs @ v                                        # (B, H, T, dh)
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, d)
        out = ctx @ self.W_o.data + self.b_o.data

        cache.update(q=q, k=k, v=v, probs=probs, ctx=ctx)
        return out

    def backward(self, dy, cache):
        q = cache["q"]
        k = cache["k"]
        v = cache["v"]
        probs = cache["probs"]
        ctx = cache["ctx"]
        B, T, d = dy.shape
        H, dh = self.n_head, self.dh
        scale = 1.0 / math.sqrt(dh)

        # Output projection. dW_o = sum_{b,t} ctx^T dy  (a single BLAS matmul
        # on the flattened batch*time axis -- much faster than an einsum).
        dctx = dy @ self.W_o.data.T                            # (B, T, d)
        self.W_o.grad += ctx.reshape(-1, d).T @ dy.reshape(-1, d)
        self.b_o.grad += dy.sum(axis=(0, 1))
        dctx = dctx.reshape(B, T, H, dh).transpose(0, 2, 1, 3)  # (B, H, T, dh)

        # Context = probs @ v  ->  v and probs
        dV = probs.transpose(0, 1, 3, 2) @ dctx
        dP = dctx @ v.transpose(0, 1, 3, 2)

        # Softmax backward: dS = P * (dP - sum(dP * P))
        dS = probs * (dP - (dP * probs).sum(axis=-1, keepdims=True))
        dS = np.where(self.mask[:T, :T], 0.0, dS)             # masked: prob 0

        # scores = q @ k^T * scale
        dQ = (dS @ k) * scale
        dK = (dS.transpose(0, 1, 3, 2) @ q) * scale

        d_qkv = np.concatenate(
            [dQ.transpose(0, 2, 1, 3).reshape(B, T, d),
             dK.transpose(0, 2, 1, 3).reshape(B, T, d),
             dV.transpose(0, 2, 1, 3).reshape(B, T, d)],
            axis=-1)

        # qkv = x @ W_qkv + b_qkv
        self.W_qkv.grad += cache["x"].reshape(-1, d).T @ d_qkv.reshape(-1, 3 * d)
        self.b_qkv.grad += d_qkv.sum(axis=(0, 1))
        return d_qkv @ self.W_qkv.data.T


class MLP:
    """Two-layer feed-forward network with GeLU: mlp(x) = gelu(x W1 + b1) W2 + b2."""

    def __init__(self, d: int, rng, scale: float = 0.02):
        self.d = d
        self.W1 = Param(_randn((d, 4 * d), rng, scale))
        self.b1 = Param(np.zeros(4 * d, dtype=DTYPE))
        self.W2 = Param(_randn((4 * d, d), rng, scale))
        self.b2 = Param(np.zeros(d, dtype=DTYPE))

    def forward(self, x, cache):
        h = gelu_forward(x @ self.W1.data + self.b1.data, cache)
        cache["x_mlp"] = x
        cache["h_mlp"] = h
        return h @ self.W2.data + self.b2.data

    def backward(self, dy, cache):
        h = cache["h_mlp"]
        x = cache["x_mlp"]
        self.W2.grad += h.reshape(-1, 4 * self.d).T @ dy.reshape(-1, self.d)
        self.b2.grad += dy.sum(axis=(0, 1))
        dh = dy @ self.W2.data.T
        dg = gelu_backward(dh, cache)
        self.W1.grad += x.reshape(-1, self.d).T @ dg.reshape(-1, 4 * self.d)
        self.b1.grad += dg.sum(axis=(0, 1))
        return dg @ self.W1.data.T


class Block:
    """One Transformer block: ln -> attn -> residual, ln -> mlp -> residual."""

    def __init__(self, d: int, n_head: int, block_size: int, rng, scale: float = 0.02):
        self.ln1 = LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head, block_size, rng, scale)
        self.ln2 = LayerNorm(d)
        self.mlp = MLP(d, rng, scale)

    def forward(self, x, cache):
        cache["x_in"] = x
        cache["x1"] = x + self.attn.forward(self.ln1.forward(x, cache.setdefault("ln1", {})),
                                            cache.setdefault("attn", {}))
        return cache["x1"] + self.mlp.forward(self.ln2.forward(cache["x1"], cache.setdefault("ln2", {})),
                                              cache.setdefault("mlp", {}))

    def backward(self, dy, cache):
        # Residual stream: the gradient splits back into both branches.
        dm = self.mlp.backward(dy, cache["mlp"])
        dx1_ln = self.ln2.backward(dm, cache["ln2"])
        da = self.attn.backward(dy + dx1_ln, cache["attn"])
        dx_ln = self.ln1.backward(da, cache["ln1"])
        return (dy + dx1_ln) + dx_ln


class LinearHead:
    """Final projection from hidden states to vocabulary logits."""

    def __init__(self, d: int, vocab_size: int, rng, scale: float = 0.02):
        self.W = Param(_randn((vocab_size, d), rng, scale))
        self.b = Param(np.zeros(vocab_size, dtype=DTYPE))

    def forward(self, x):
        return x @ self.W.data.T + self.b.data


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def softmax_cross_entropy(logits, targets):
    """Mean cross-entropy over every (batch, time) position.

    Returns (loss, dlogits); dlogits is the gradient of the loss with respect
    to the logits: (softmax(logits) - onehot(targets)) / (B * T).
    """
    B, T, V = logits.shape
    flat = logits.reshape(-1, V)
    probs = softmax_last_axis(flat)
    loss = -np.log(probs[np.arange(B * T), targets.reshape(-1)]).mean()
    d = probs.copy()
    d[np.arange(B * T), targets.reshape(-1)] -= 1.0
    return float(loss), (d / (B * T)).reshape(B, T, V)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GPT:
    """A decoder-only Transformer: token + position embeddings, N blocks,
    final LayerNorm, and a linear head. ``block_size`` is the context length."""

    def __init__(self, vocab_size: int, d_model: int, n_head: int, n_layer: int,
                 block_size: int, rng=None, scale: float = 0.02):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        rng = rng if rng is not None else np.random.default_rng(0)

        self.Wte = Param(_randn((vocab_size, d_model), rng, scale))  # token embedding
        self.Wpe = Param(_randn((block_size, d_model), rng, scale))  # position embedding
        self.blocks = [Block(d_model, n_head, block_size, rng, scale) for _ in range(n_layer)]
        self.ln_f = LayerNorm(d_model)
        self.head = LinearHead(d_model, vocab_size, rng, scale)

    def parameters(self):
        """(name, Param) pairs; names are stable keys for checkpoints."""
        out = [("Wte", self.Wte), ("Wpe", self.Wpe)]
        for i, blk in enumerate(self.blocks):
            out += [
                (f"b{i}.ln1.g", blk.ln1.g), (f"b{i}.ln1.b", blk.ln1.b),
                (f"b{i}.attn.W_qkv", blk.attn.W_qkv), (f"b{i}.attn.b_qkv", blk.attn.b_qkv),
                (f"b{i}.attn.W_o", blk.attn.W_o), (f"b{i}.attn.b_o", blk.attn.b_o),
                (f"b{i}.ln2.g", blk.ln2.g), (f"b{i}.ln2.b", blk.ln2.b),
                (f"b{i}.mlp.W1", blk.mlp.W1), (f"b{i}.mlp.b1", blk.mlp.b1),
                (f"b{i}.mlp.W2", blk.mlp.W2), (f"b{i}.mlp.b2", blk.mlp.b2),
            ]
        out += [("lnf.g", self.ln_f.g), ("lnf.b", self.ln_f.b),
                ("head.W", self.head.W), ("head.b", self.head.b)]
        return out

    def zero_grad(self):
        for _, p in self.parameters():
            p.grad.fill(0.0)

    def num_params(self) -> int:
        return sum(int(p.data.size) for _, p in self.parameters())

    def forward(self, idx, cache):
        """idx: (B, T) integer ids -> logits (B, T, V). Records `cache` for backward."""
        B, T = idx.shape
        assert T <= self.block_size, f"context {T} exceeds block_size {self.block_size}"
        cache["idx"] = idx
        h = self.Wte.data[idx] + self.Wpe.data[:T]               # (B, T, d)
        cache["h0"] = h
        block_caches = {}
        for i, blk in enumerate(self.blocks):
            block_caches[i] = {}
            h = blk.forward(h, block_caches[i])
        cache["blocks"] = block_caches
        h = self.ln_f.forward(h, cache.setdefault("lnf", {}))
        cache["h_pre_head"] = h
        return self.head.forward(h)

    def backward(self, dlogits, cache):
        """Accumulate parameter gradients for a given dL/dlogits."""
        h_pre = cache["h_pre_head"]
        self.head.W.grad += dlogits.reshape(-1, self.vocab_size).T @ h_pre.reshape(-1, self.d_model)
        self.head.b.grad += dlogits.sum(axis=(0, 1))
        dh = dlogits @ self.head.W.data
        dh = self.ln_f.backward(dh, cache["lnf"])
        for i in range(self.n_layer - 1, -1, -1):
            dh = self.blocks[i].backward(dh, cache["blocks"][i])
        # Embeddings: h0 = Wte[idx] + Wpe[:T]
        # np.add.at is pathologically slow for scatter-add; bincount per column
        # is ~100x faster and mathematically identical (both sum over dups).
        idx = cache["idx"]
        idx_flat = idx.reshape(-1)
        dh_flat = dh.reshape(-1, self.d_model)
        for j in range(self.d_model):
            self.Wte.grad[:, j] += np.bincount(
                idx_flat, weights=dh_flat[:, j], minlength=self.vocab_size)
        self.Wpe.grad[:idx.shape[1]] += dh.sum(axis=0)
