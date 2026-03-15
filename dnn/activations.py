"""Element-wise activation functions."""
import numpy as np


class ReLU:
    """Rectified Linear Unit: f(x) = max(0, x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask

    def parameters(self):
        return []


class Sigmoid:
    """Logistic sigmoid: f(x) = 1 / (1 + exp(-x))."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Clip to avoid overflow in exp
        self._out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self._out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        s = self._out
        return grad_out * s * (1.0 - s)

    def parameters(self):
        return []
