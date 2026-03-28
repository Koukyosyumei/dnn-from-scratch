"""Parameterised layers for a fully-connected network."""
import numpy as np


def _im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int) -> np.ndarray:
    """Rearrange image patches into columns for efficient convolution.

    Parameters
    ----------
    x:      Input array of shape (N, C, H, W).
    kh, kw: Kernel height and width.
    stride: Convolution stride.
    pad:    Zero-padding added to both spatial sides.

    Returns
    -------
    col: Array of shape (N, C, kh, kw, out_h, out_w) rearranged as
         (N * out_h * out_w, C * kh * kw).
    """
    N, C, H, W = x.shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    col = np.zeros((N, C, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            col[:, :, i, j, :, :] = x_pad[:, :, i:i_max:stride, j:j_max:stride]

    # (N, C, kh, kw, out_h, out_w) -> (N * out_h * out_w, C * kh * kw)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def _col2im(col: np.ndarray, x_shape: tuple, kh: int, kw: int,
            stride: int, pad: int) -> np.ndarray:
    """Inverse of _im2col — scatter column values back into an image tensor.

    Parameters
    ----------
    col:     Array of shape (N * out_h * out_w, C * kh * kw).
    x_shape: Original input shape (N, C, H, W).
    """
    N, C, H, W = x_shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    col = col.reshape(N, out_h, out_w, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)

    x_pad = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1),
                     dtype=col.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            x_pad[:, :, i:i_max:stride, j:j_max:stride] += col[:, :, i, j, :, :]

    if pad == 0:
        return x_pad[:, :, :H, :W]
    return x_pad[:, :, pad:pad + H, pad:pad + W]


class Conv2D:
    """2-D convolution layer.

    Performs cross-correlation (the standard "convolution" in deep learning):
        y[n, f, i, j] = sum_{c,kh,kw} W[f,c,kh,kw] * x[n, c, i*s+kh, j*s+kw] + b[f]

    Parameters
    ----------
    in_channels:  Number of input channels C.
    out_channels: Number of filters (output channels) F.
    kernel_size:  Spatial size of each square kernel (K×K).
    stride:       Step size for the sliding window (default 1).
    padding:      Zero-padding applied to each spatial side (default 0).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialisation (fan_in = C * K * K)
        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)

        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self._x_shape: tuple | None = None
        self._col: np.ndarray | None = None

    # ── forward / backward ─────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """y shape: (N, F, out_h, out_w)"""
        N, C, H, W = x.shape
        K = self.kernel_size
        S, P = self.stride, self.padding
        out_h = (H + 2 * P - K) // S + 1
        out_w = (W + 2 * P - K) // S + 1

        col = _im2col(x, K, K, S, P)           # (N*out_h*out_w, C*K*K)
        W_col = self.W.reshape(self.out_channels, -1)  # (F, C*K*K)

        out = col @ W_col.T + self.b            # (N*out_h*out_w, F)
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

        self._x_shape = x.shape
        self._col = col
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Accumulate dW, db and return gradient w.r.t. input x."""
        assert self._col is not None, "Call forward() before backward()"
        N, F, out_h, out_w = grad_out.shape
        K = self.kernel_size
        S, P = self.stride, self.padding

        # grad_out: (N, F, out_h, out_w) -> (N*out_h*out_w, F)
        d_out = grad_out.transpose(0, 2, 3, 1).reshape(-1, F)

        self.db = d_out.sum(axis=0)                                  # (F,)
        self.dW = (d_out.T @ self._col).reshape(self.W.shape)        # (F, C, K, K)

        W_col = self.W.reshape(F, -1)
        d_col = d_out @ W_col                                        # (N*out_h*out_w, C*K*K)
        dx = _col2im(d_col, self._x_shape, K, K, S, P)
        return dx

    # ── parameter access ───────────────────────────────────────────────────

    def parameters(self):
        """Yield (param, grad) pairs for the optimiser."""
        return [(self.W, self.dW), (self.b, self.db)]


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
