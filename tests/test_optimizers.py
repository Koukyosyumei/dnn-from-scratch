import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dnn.layers import Linear
from dnn.optimizers import SGD, Adam


def make_layer_with_grad(W_val, b_val, dW_val, db_val):
    layer = Linear(W_val.shape[0], W_val.shape[1])
    layer.W = W_val.copy()
    layer.b = b_val.copy()
    layer.dW = dW_val.copy()
    layer.db = db_val.copy()
    return layer


def test_sgd_basic_step():
    W = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([0.5, -0.5])
    dW = np.ones_like(W)
    db = np.ones_like(b)
    layer = make_layer_with_grad(W, b, dW, db)

    sgd = SGD(lr=0.1)
    sgd.step([layer])

    np.testing.assert_allclose(layer.W, W - 0.1 * dW)
    np.testing.assert_allclose(layer.b, b - 0.1 * db)


def test_sgd_lr_zero_raises():
    # lr=0.0 is non-positive so the constructor should reject it
    with pytest.raises(ValueError):
        SGD(lr=0.0)


def test_sgd_invalid_lr_raises():
    with pytest.raises(ValueError):
        SGD(lr=-0.1)


def test_sgd_invalid_momentum_raises():
    with pytest.raises(ValueError):
        SGD(lr=0.01, momentum=1.0)


def test_sgd_momentum_accumulates():
    W = np.array([[1.0]])
    b = np.array([0.0])
    dW = np.array([[1.0]])
    db = np.array([0.0])
    layer = make_layer_with_grad(W, b, dW, db)

    sgd = SGD(lr=0.1, momentum=0.9)
    sgd.step([layer])
    W_after_1 = layer.W.copy()

    layer.dW = np.array([[1.0]])
    sgd.step([layer])
    W_after_2 = layer.W.copy()

    # Second step should move further due to accumulated momentum
    delta1 = abs(1.0 - W_after_1[0, 0])
    delta2 = abs(W_after_1[0, 0] - W_after_2[0, 0])
    assert delta2 > delta1, "Momentum should accelerate updates"


def test_sgd_skips_none_grad():
    layer = Linear(2, 2)
    layer.dW = None
    layer.db = None
    sgd = SGD(lr=0.1)
    W_before = layer.W.copy()
    sgd.step([layer])
    np.testing.assert_allclose(layer.W, W_before)  # unchanged


# ---------------------------------------------------------------------------
# Adam tests
# ---------------------------------------------------------------------------

def test_adam_basic_step():
    """Single step matches the closed-form Adam update."""
    W = np.array([[1.0, 2.0]])
    b = np.array([0.5])
    dW = np.array([[0.1, 0.2]])
    db = np.array([0.3])
    layer = make_layer_with_grad(W, b, dW, db)

    lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8
    adam = Adam(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
    adam.step([layer])

    # Expected: first step (t=1), bias-corrected moments
    # m = (1-beta1)*dW  →  m_hat = m / (1 - beta1^1) = dW
    # v = (1-beta2)*dW² →  v_hat = v / (1 - beta2^1) = dW²
    m_hat = ((1.0 - beta1) * dW) / (1.0 - beta1 ** 1)
    v_hat = ((1.0 - beta2) * dW ** 2) / (1.0 - beta2 ** 1)
    W_expected = W - lr * m_hat / (np.sqrt(v_hat) + eps)
    np.testing.assert_allclose(layer.W, W_expected, rtol=1e-6)


def test_adam_invalid_lr_raises():
    with pytest.raises(ValueError):
        Adam(lr=0.0)


def test_adam_invalid_beta1_raises():
    with pytest.raises(ValueError):
        Adam(beta1=1.0)


def test_adam_invalid_beta2_raises():
    with pytest.raises(ValueError):
        Adam(beta2=-0.1)


def test_adam_invalid_eps_raises():
    with pytest.raises(ValueError):
        Adam(eps=0.0)


def test_adam_skips_none_grad():
    layer = Linear(2, 2)
    layer.dW = None
    layer.db = None
    adam = Adam()
    W_before = layer.W.copy()
    adam.step([layer])
    np.testing.assert_allclose(layer.W, W_before)


def test_adam_step_counter_increments():
    layer = make_layer_with_grad(
        np.array([[1.0]]), np.array([0.0]),
        np.array([[0.1]]), np.array([0.0]),
    )
    adam = Adam()
    adam.step([layer])
    adam.step([layer])
    assert adam._t == 2


def test_adam_decreases_loss():
    """Adam should reduce a simple quadratic loss over multiple steps."""
    np.random.seed(0)
    W = np.array([[2.0, -3.0]])
    b = np.array([1.0])
    layer = make_layer_with_grad(W, b, np.zeros_like(W), np.zeros_like(b))
    adam = Adam(lr=0.1)

    for _ in range(200):
        layer.dW = layer.W  # grad of 0.5*||W||^2
        layer.db = layer.b
        adam.step([layer])

    assert np.sum(layer.W ** 2) < 0.01
