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
