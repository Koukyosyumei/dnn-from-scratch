"""Loss functions with forward/backward interface."""
import numpy as np


class MSELoss:
    """Mean Squared Error: L = mean((pred - target)^2)."""

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._diff = pred - target
        return float(np.mean(self._diff ** 2))

    def backward(self) -> np.ndarray:
        n = self._diff.size
        return 2.0 * self._diff / n


class BCELoss:
    """Binary Cross-Entropy: L = -mean(t*log(p) + (1-t)*log(1-p))."""

    _EPS = 1e-12

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        p = np.clip(pred, self._EPS, 1.0 - self._EPS)
        self._p = p
        self._t = target
        return float(-np.mean(target * np.log(p) + (1.0 - target) * np.log(1.0 - p)))

    def backward(self) -> np.ndarray:
        p, t = self._p, self._t
        n = p.size
        return (-(t / p) + (1.0 - t) / (1.0 - p)) / n
