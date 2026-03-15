"""High-level Sequential container."""
from __future__ import annotations
import numpy as np


class Sequential:
    """Chains layers so that output of layer[i] feeds layer[i+1].

    Example
    -------
    >>> import numpy as np
    >>> from dnn import Linear, ReLU, Sequential
    >>> net = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
    >>> x = np.random.randn(8, 2)
    >>> y = net.forward(x)
    >>> y.shape
    (8, 1)
    """

    def __init__(self, *layers) -> None:
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        """Flat list of layers passed to the optimiser."""
        return self.layers
