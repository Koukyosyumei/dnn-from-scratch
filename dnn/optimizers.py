"""Gradient-based parameter update rules."""
from __future__ import annotations
import numpy as np


class SGD:
    """Stochastic Gradient Descent with optional momentum.

    Parameters
    ----------
    lr:       Learning rate.
    momentum: Momentum coefficient (0 disables momentum).
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")
        self.lr = lr
        self.momentum = momentum
        self._velocity: dict = {}

    def step(self, layers) -> None:
        """Apply one gradient-descent update to all layer parameters."""
        for i, layer in enumerate(layers):
            for j, (param, grad) in enumerate(layer.parameters()):
                if grad is None:
                    continue
                key = (i, j)
                if self.momentum > 0.0:
                    v = self._velocity.get(key, np.zeros_like(param))
                    v = self.momentum * v - self.lr * grad
                    self._velocity[key] = v
                    param += v
                else:
                    param -= self.lr * grad

    def zero_grad(self, layers) -> None:
        """Reset accumulated gradients to None."""
        for layer in layers:
            for param, _ in layer.parameters():
                pass  # gradients are overwritten each backward pass


class Adam:
    """Adam optimiser (Kingma & Ba, 2015).

    Parameters
    ----------
    lr:      Learning rate (alpha).
    beta1:   Exponential decay rate for the first moment estimate.
    beta2:   Exponential decay rate for the second moment estimate.
    eps:     Small constant for numerical stability.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: dict = {}   # first moment
        self._v: dict = {}   # second moment
        self._t: int = 0     # step counter

    def step(self, layers) -> None:
        """Apply one Adam update to all layer parameters."""
        self._t += 1
        for i, layer in enumerate(layers):
            for j, (param, grad) in enumerate(layer.parameters()):
                if grad is None:
                    continue
                key = (i, j)
                m = self._m.get(key, np.zeros_like(param))
                v = self._v.get(key, np.zeros_like(param))

                m = self.beta1 * m + (1.0 - self.beta1) * grad
                v = self.beta2 * v + (1.0 - self.beta2) * grad ** 2

                self._m[key] = m
                self._v[key] = v

                m_hat = m / (1.0 - self.beta1 ** self._t)
                v_hat = v / (1.0 - self.beta2 ** self._t)

                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self, layers) -> None:
        """Reset accumulated gradients to None."""
        for layer in layers:
            for param, _ in layer.parameters():
                pass  # gradients are overwritten each backward pass
