"""nano_gpt.trainer -- Adam optimizer and the parameter-update bookkeeping."""

from __future__ import annotations

import numpy as np


class Adam:
    """Adam (Kingma & Ba, 2015) with bias correction, implemented by hand.

    Maintains per-parameter first (m) and second (v) moment estimates and
    rescales updates by sqrt(v) + eps, which gives every parameter an
    effectively normalized step size.
    """

    def __init__(self, params, lr: float = 3e-4, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for _, p in self.params]
        self.v = [np.zeros_like(p.data) for _, p in self.params]
        self.t = 0

    def zero_grad(self):
        for _, p in self.params:
            p.grad.fill(0.0)

    def step(self):
        self.t += 1
        b1, b2, eps, lr = self.b1, self.b2, self.eps, self.lr
        for i, (_, p) in enumerate(self.params):
            g = p.grad
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * (g * g)
            m_hat = self.m[i] / (1.0 - b1 ** self.t)
            v_hat = self.v[i] / (1.0 - b2 ** self.t)
            p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)
