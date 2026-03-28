"""Tests for the Conv2D layer."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dnn.layers import Conv2D


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def conv_1x1():
    """1×1 convolution — equivalent to a channel-wise linear layer."""
    np.random.seed(0)
    return Conv2D(in_channels=3, out_channels=8, kernel_size=1)


@pytest.fixture
def conv_3x3():
    np.random.seed(42)
    return Conv2D(in_channels=2, out_channels=4, kernel_size=3, padding=1)


# ── output shape ─────────────────────────────────────────────────────────────

def test_forward_shape_no_padding():
    layer = Conv2D(1, 4, kernel_size=3)
    x = np.random.randn(2, 1, 8, 8)
    assert layer.forward(x).shape == (2, 4, 6, 6)


def test_forward_shape_with_same_padding(conv_3x3):
    x = np.random.randn(3, 2, 5, 5)
    # padding=1, stride=1 → output has same H, W as input
    assert conv_3x3.forward(x).shape == (3, 4, 5, 5)


def test_forward_shape_stride_2():
    layer = Conv2D(1, 2, kernel_size=3, stride=2, padding=1)
    x = np.random.randn(1, 1, 8, 8)
    assert layer.forward(x).shape == (1, 2, 4, 4)


def test_1x1_conv_shape(conv_1x1):
    x = np.random.randn(4, 3, 6, 6)
    assert conv_1x1.forward(x).shape == (4, 8, 6, 6)


# ── correctness ───────────────────────────────────────────────────────────────

def test_1x1_conv_matches_linear():
    """A 1×1 conv with a single spatial position equals a Linear layer."""
    np.random.seed(7)
    layer = Conv2D(in_channels=3, out_channels=5, kernel_size=1)
    x = np.random.randn(2, 3, 1, 1)
    out = layer.forward(x)                    # (2, 5, 1, 1)
    expected = x[:, :, 0, 0] @ layer.W[:, :, 0, 0].T + layer.b  # (2, 5)
    np.testing.assert_allclose(out[:, :, 0, 0], expected, rtol=1e-6)


def test_zero_kernel_produces_bias_only():
    layer = Conv2D(1, 3, kernel_size=3, padding=1)
    layer.W[:] = 0.0
    layer.b = np.array([1.0, 2.0, 3.0])
    x = np.random.randn(2, 1, 4, 4)
    out = layer.forward(x)
    np.testing.assert_allclose(out[:, 0], 1.0)
    np.testing.assert_allclose(out[:, 1], 2.0)
    np.testing.assert_allclose(out[:, 2], 3.0)


# ── gradient shapes ───────────────────────────────────────────────────────────

def test_backward_dx_shape():
    layer = Conv2D(2, 4, kernel_size=3, padding=1)
    x = np.random.randn(3, 2, 5, 5)
    out = layer.forward(x)
    dx = layer.backward(np.ones_like(out))
    assert dx.shape == x.shape


def test_backward_dW_shape():
    layer = Conv2D(2, 4, kernel_size=3, padding=1)
    x = np.random.randn(3, 2, 5, 5)
    out = layer.forward(x)
    layer.backward(np.ones_like(out))
    assert layer.dW.shape == layer.W.shape


def test_backward_db_shape():
    layer = Conv2D(2, 4, kernel_size=3, padding=1)
    x = np.random.randn(3, 2, 5, 5)
    out = layer.forward(x)
    layer.backward(np.ones_like(out))
    assert layer.db.shape == (4,)


# ── numerical gradient check ──────────────────────────────────────────────────

def _numerical_grad(layer, x, grad_out, param, eps=1e-5):
    """Finite-difference gradient for `param` (W or b)."""
    num_grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = param[idx]
        param[idx] = orig + eps
        f1 = np.sum(layer.forward(x) * grad_out)
        param[idx] = orig - eps
        f2 = np.sum(layer.forward(x) * grad_out)
        param[idx] = orig
        num_grad[idx] = (f1 - f2) / (2 * eps)
        it.iternext()
    return num_grad


def test_gradient_check_W():
    np.random.seed(3)
    layer = Conv2D(2, 3, kernel_size=3, padding=1)
    x = np.random.randn(2, 2, 5, 5)
    grad_out = np.random.randn(2, 3, 5, 5)

    layer.forward(x)
    layer.backward(grad_out)
    analytical = layer.dW.copy()
    numerical = _numerical_grad(layer, x, grad_out, layer.W)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)


def test_gradient_check_b():
    np.random.seed(5)
    layer = Conv2D(1, 4, kernel_size=3, padding=1)
    x = np.random.randn(2, 1, 6, 6)
    grad_out = np.random.randn(2, 4, 6, 6)

    layer.forward(x)
    layer.backward(grad_out)
    analytical = layer.db.copy()
    numerical = _numerical_grad(layer, x, grad_out, layer.b)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)


def test_gradient_check_x():
    np.random.seed(9)
    layer = Conv2D(2, 3, kernel_size=3, stride=1, padding=1)
    x = np.random.randn(2, 2, 5, 5)
    grad_out = np.random.randn(2, 3, 5, 5)

    layer.forward(x)
    layer.backward(grad_out)
    analytical_dx = layer.backward(grad_out)  # second call for clean dx

    eps = 1e-5
    numerical_dx = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x[idx] += eps
        f1 = np.sum(layer.forward(x) * grad_out)
        x[idx] -= 2 * eps
        f2 = np.sum(layer.forward(x) * grad_out)
        x[idx] += eps
        numerical_dx[idx] = (f1 - f2) / (2 * eps)

    np.testing.assert_allclose(analytical_dx, numerical_dx, rtol=1e-4, atol=1e-6)


# ── parameters() ─────────────────────────────────────────────────────────────

def test_parameters_returns_W_and_b(conv_3x3):
    params = conv_3x3.parameters()
    assert len(params) == 2
    assert params[0][0] is conv_3x3.W
    assert params[1][0] is conv_3x3.b
