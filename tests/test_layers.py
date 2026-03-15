import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dnn.layers import Linear


@pytest.fixture
def linear_2_3():
    np.random.seed(0)
    return Linear(2, 3)


def test_forward_shape(linear_2_3):
    x = np.ones((5, 2))
    out = linear_2_3.forward(x)
    assert out.shape == (5, 3)


def test_forward_values():
    layer = Linear(2, 2)
    layer.W = np.eye(2)
    layer.b = np.zeros(2)
    x = np.array([[1.0, 2.0]])
    np.testing.assert_allclose(layer.forward(x), x)


def test_backward_grad_input_shape(linear_2_3):
    x = np.ones((5, 2))
    linear_2_3.forward(x)
    grad = linear_2_3.backward(np.ones((5, 3)))
    assert grad.shape == (5, 2)


def test_backward_grad_W_shape(linear_2_3):
    x = np.ones((5, 2))
    linear_2_3.forward(x)
    linear_2_3.backward(np.ones((5, 3)))
    assert linear_2_3.dW.shape == (2, 3)


def test_backward_grad_b_shape(linear_2_3):
    x = np.ones((5, 2))
    linear_2_3.forward(x)
    linear_2_3.backward(np.ones((5, 3)))
    assert linear_2_3.db.shape == (3,)


def test_numerical_gradient():
    """Finite-difference check for W gradient."""
    np.random.seed(1)
    layer = Linear(3, 4)
    x = np.random.randn(2, 3)
    grad_out = np.random.randn(2, 4)

    layer.forward(x)
    layer.backward(grad_out)
    analytical_dW = layer.dW.copy()

    eps = 1e-5
    numerical_dW = np.zeros_like(layer.W)
    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            layer.W[i, j] += eps
            f1 = np.sum(layer.forward(x) * grad_out)
            layer.W[i, j] -= 2 * eps
            f2 = np.sum(layer.forward(x) * grad_out)
            layer.W[i, j] += eps
            numerical_dW[i, j] = (f1 - f2) / (2 * eps)

    np.testing.assert_allclose(analytical_dW, numerical_dW, rtol=1e-4)


def test_parameters_returns_W_and_b(linear_2_3):
    params = linear_2_3.parameters()
    assert len(params) == 2
    assert params[0][0] is linear_2_3.W
    assert params[1][0] is linear_2_3.b
