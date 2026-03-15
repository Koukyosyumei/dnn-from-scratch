import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dnn import Linear, ReLU, Sigmoid, Sequential, SGD, MSELoss, BCELoss


@pytest.fixture
def tiny_net():
    np.random.seed(42)
    return Sequential(Linear(2, 4), ReLU(), Linear(4, 1), Sigmoid())


def test_forward_output_shape(tiny_net):
    x = np.random.randn(8, 2)
    out = tiny_net.forward(x)
    assert out.shape == (8, 1)


def test_forward_sigmoid_output_in_01(tiny_net):
    x = np.random.randn(16, 2)
    out = tiny_net.forward(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_backward_does_not_crash(tiny_net):
    x = np.random.randn(4, 2)
    out = tiny_net.forward(x)
    loss_fn = BCELoss()
    target = np.random.randint(0, 2, (4, 1)).astype(float)
    loss_fn.forward(out, target)
    grad = loss_fn.backward()
    tiny_net.backward(grad)  # should not raise


def test_parameters_returns_all_layers(tiny_net):
    params = tiny_net.parameters()
    assert len(params) == 4  # Linear, ReLU, Linear, Sigmoid


def test_xor_converges():
    """End-to-end: network should learn XOR within 5000 steps."""
    np.random.seed(0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    net = Sequential(Linear(2, 8), ReLU(), Linear(8, 1), Sigmoid())
    loss_fn = BCELoss()
    opt = SGD(lr=0.1, momentum=0.9)

    for _ in range(5000):
        pred = net.forward(X)
        loss_fn.forward(pred, y)
        grad = loss_fn.backward()
        net.backward(grad)
        opt.step(net.parameters())

    pred = net.forward(X)
    preds_bin = (pred > 0.5).astype(int)
    acc = (preds_bin == y).mean()
    assert acc == 1.0, f"XOR should be learned with acc=1.0, got {acc}"


def test_mse_regression_converges():
    """Network should overfit a tiny linear dataset with MSE loss."""
    np.random.seed(7)
    X = np.random.randn(20, 3)
    y = X @ np.array([1.0, -2.0, 0.5]).reshape(3, 1) + 0.1

    net = Sequential(Linear(3, 8), ReLU(), Linear(8, 1))
    loss_fn = MSELoss()
    opt = SGD(lr=0.01, momentum=0.9)

    initial_loss = None
    for step in range(2000):
        pred = net.forward(X)
        loss = loss_fn.forward(pred, y)
        if initial_loss is None:
            initial_loss = loss
        grad = loss_fn.backward()
        net.backward(grad)
        opt.step(net.parameters())

    final_loss = loss_fn.forward(net.forward(X), y)
    assert final_loss < initial_loss * 0.01, "Loss should decrease by 99%"
