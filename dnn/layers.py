"""Parameterised layers for a fully-connected network."""
import numpy as np


class Linear:
    """Affine transformation: y = x @ W + b.

    Parameters
    ----------
    in_features:  Number of input dimensions.
    out_features: Number of output dimensions.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        # He initialisation — works well with ReLU activations.
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        # Gradients (populated by backward())
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self._x: np.ndarray | None = None  # cached input for backward

    # ── forward / backward ─────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """y = x @ W + b  (shape: [batch, out_features])"""
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Accumulate gradients and return grad w.r.t. input."""
        assert self._x is not None, "Call forward() before backward()"
        self.dW = self._x.T @ grad_out          # [in, out]
        self.db = grad_out.sum(axis=0)          # [out]
        return grad_out @ self.W.T              # [batch, in]

    # ── parameter access ───────────────────────────────────────────────────

    def parameters(self):
        """Yield (param, grad) pairs for the optimiser."""
        return [(self.W, self.dW), (self.b, self.db)]
