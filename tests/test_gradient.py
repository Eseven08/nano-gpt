"""test_gradient.py -- verify every hand-derived backward pass against
numerical central-difference gradients.

Run:  python tests/test_gradient.py

Every parameter of a small GPT is checked on one random batch. The relative
error between the analytic gradient and the numerical estimate must stay
below the tolerance. This is the strongest correctness proof in the repo:
it catches sign, shape, and broadcasting bugs that unit tests alone miss.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_gpt import model as M  # noqa: E402
from nano_gpt.model import GPT, softmax_cross_entropy  # noqa: E402


def make_model(dtype):
    M.DTYPE = dtype
    rng = np.random.default_rng(7)
    return GPT(vocab_size=17, d_model=8, n_head=2, n_layer=2, block_size=6, rng=rng)


def forward_loss(model, idx, targets):
    logits = model.forward(idx, {})
    loss, _ = softmax_cross_entropy(logits, targets)
    return float(loss)


def analytic_grads(model, idx, targets):
    cache = {}
    logits = model.forward(idx, cache)
    loss, dlogits = softmax_cross_entropy(logits, targets)
    model.zero_grad()
    model.backward(dlogits, cache)
    return loss, {name: p.grad for name, p in model.parameters()}


def check(dtype, eps, atol, rtol, label):
    """Per-entry tolerance (torch gradcheck style):
       |an - num| <= atol + rtol * |num|   for every entry.
    Near-zero-gradient entries therefore cannot dominate the verdict, while
    sign, scale, shape and broadcast bugs still fail loudly.
    """
    model = make_model(dtype)
    rng = np.random.default_rng(0)
    B, T = 2, 6
    idx = rng.integers(0, model.vocab_size, (B, T))
    targets = rng.integers(0, model.vocab_size, (B, T))

    loss, grads = analytic_grads(model, idx, targets)

    failures = []
    for name, p in model.parameters():
        num = np.zeros_like(p.data)
        orig = p.data.copy()
        for flat in np.ndindex(*p.data.shape):
            xp = p.data.copy()
            xm = p.data.copy()
            xp[flat] += eps
            xm[flat] -= eps
            p.data = xp
            gp = forward_loss(model, idx, targets)
            p.data = xm
            gm = forward_loss(model, idx, targets)
            num[flat] = (gp - gm) / (2.0 * eps)
        p.data = orig

        ratio = np.abs(grads[name] - num) / (atol + rtol * np.abs(num) + 1e-12)
        max_ratio = float(ratio.max())
        status = "OK " if max_ratio < 1.0 else "FAIL"
        if max_ratio >= 1.0:
            failures.append(name)
        print(f"  [{status}] {name:<14s} max tol ratio = {max_ratio:.3e}")

    print(f"{label}: {'ALL PASS' if not failures else 'FAILED: ' + ', '.join(failures)} "
          f"(loss {loss:.6f}, atol={atol:g} rtol={rtol:g})")
    return not failures


def main():
    print("=" * 62)
    print("Gradient checks (analytic backward vs. central difference)")
    print("=" * 62)

    ok64 = check(np.float64, eps=1e-6, atol=1e-6, rtol=1e-3, label="float64")
    ok32 = check(np.float32, eps=1e-3, atol=5e-3, rtol=2e-1, label="float32")

    if ok64 and ok32:
        print("\nAll gradient checks PASSED.")
        return 0
    print("\nGradient checks FAILED -- inspect the FAIL entries above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
