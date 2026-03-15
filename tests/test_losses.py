import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dnn.losses import MSELoss, BCELoss


class TestMSELoss:
    def test_zero_loss_when_equal(self):
        loss = MSELoss()
        p = np.array([1.0, 2.0, 3.0])
        assert loss.forward(p, p) == pytest.approx(0.0)

    def test_known_value(self):
        loss = MSELoss()
        pred = np.array([2.0, 4.0])
        target = np.array([0.0, 0.0])
        assert loss.forward(pred, target) == pytest.approx(10.0)

    def test_backward_shape(self):
        loss = MSELoss()
        pred = np.random.randn(3, 2)
        target = np.zeros((3, 2))
        loss.forward(pred, target)
        grad = loss.backward()
        assert grad.shape == pred.shape

    def test_backward_numerical(self):
        loss = MSELoss()
        pred = np.array([1.0, 2.0, 3.0])
        target = np.zeros(3)
        loss.forward(pred, target)
        analytical = loss.backward()

        eps = 1e-5
        numerical = np.zeros_like(pred)
        for i in range(len(pred)):
            pred[i] += eps
            f1 = loss.forward(pred, target)
            pred[i] -= 2 * eps
            f2 = loss.forward(pred, target)
            pred[i] += eps
            numerical[i] = (f1 - f2) / (2 * eps)

        np.testing.assert_allclose(analytical, numerical, rtol=1e-4)


class TestBCELoss:
    def test_forward_perfect_prediction(self):
        loss = BCELoss()
        pred = np.array([0.9999, 0.9999])
        target = np.array([1.0, 1.0])
        assert loss.forward(pred, target) < 0.01

    def test_forward_bad_prediction(self):
        loss = BCELoss()
        pred = np.array([0.0001, 0.0001])
        target = np.array([1.0, 1.0])
        assert loss.forward(pred, target) > 5.0

    def test_backward_numerical(self):
        loss = BCELoss()
        pred = np.array([0.3, 0.7, 0.5])
        target = np.array([0.0, 1.0, 1.0])
        loss.forward(pred, target)
        analytical = loss.backward()

        eps = 1e-5
        numerical = np.zeros_like(pred)
        for i in range(len(pred)):
            pred[i] += eps
            f1 = loss.forward(pred, target)
            pred[i] -= 2 * eps
            f2 = loss.forward(pred, target)
            pred[i] += eps
            numerical[i] = (f1 - f2) / (2 * eps)

        np.testing.assert_allclose(analytical, numerical, rtol=1e-3)
