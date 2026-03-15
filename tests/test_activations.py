import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dnn.activations import ReLU, Sigmoid


class TestReLU:
    def test_forward_positive(self):
        relu = ReLU()
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(relu.forward(x), x)

    def test_forward_negative_zeroed(self):
        relu = ReLU()
        x = np.array([-1.0, -2.0])
        np.testing.assert_allclose(relu.forward(x), np.zeros(2))

    def test_forward_mixed(self):
        relu = ReLU()
        x = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(relu.forward(x), [0.0, 0.0, 1.0])

    def test_backward_passes_grad_for_positive(self):
        relu = ReLU()
        relu.forward(np.array([1.0, -1.0, 2.0]))
        grad = relu.backward(np.ones(3))
        np.testing.assert_allclose(grad, [1.0, 0.0, 1.0])

    def test_parameters_empty(self):
        assert ReLU().parameters() == []


class TestSigmoid:
    def test_forward_zero(self):
        sig = Sigmoid()
        np.testing.assert_allclose(sig.forward(np.array([0.0])), [0.5])

    def test_forward_large_positive(self):
        sig = Sigmoid()
        assert sig.forward(np.array([1000.0]))[0] > 0.999

    def test_forward_large_negative(self):
        sig = Sigmoid()
        assert sig.forward(np.array([-1000.0]))[0] < 0.001

    def test_backward_numerical(self):
        sig = Sigmoid()
        x = np.array([0.5])
        sig.forward(x)
        grad = sig.backward(np.ones(1))
        s = 1 / (1 + np.exp(-0.5))
        np.testing.assert_allclose(grad, [s * (1 - s)], rtol=1e-6)

    def test_parameters_empty(self):
        assert Sigmoid().parameters() == []
