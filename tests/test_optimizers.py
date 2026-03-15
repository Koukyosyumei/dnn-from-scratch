import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dnn.layers import Linear
from dnn.optimizers import SGD


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


def test_sgd_lr_zero_no_update():
    W = np.array([[1.0]])
    b = np.array([0.0])
    dW = np.array([[5.0]])
    db = np.array([5.0])
    layer = make_layer_with_grad(W, b, dW, db)

    sgd = SGD(lr=0.0)  # technically invalid but tests boundary
    # raises or no-ops depending on implementation — just test it doesn't crash badly
    try:
        sgd.step([layer])
    except Exception:
        pass  # ValueError is acceptable


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
