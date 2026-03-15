#!/usr/bin/env python3
"""Train a tiny MLP to learn the XOR function.

XOR truth table:
  (0,0) -> 0   (0,1) -> 1   (1,0) -> 1   (1,1) -> 0

We solve it with:  Linear(2→4) → ReLU → Linear(4→1) → Sigmoid
Optimiser: SGD with momentum 0.9, lr=0.1
Loss: Binary Cross-Entropy

Run:
    python examples/train_xor.py
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dnn import Linear, ReLU, Sigmoid, BCELoss, SGD, Sequential

np.random.seed(42)

# Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# Model
net = Sequential(
    Linear(2, 8),
    ReLU(),
    Linear(8, 1),
    Sigmoid(),
)
loss_fn = BCELoss()
opt = SGD(lr=0.1, momentum=0.9)

# Training loop
for epoch in range(1, 5001):
    pred = net.forward(X)
    loss = loss_fn.forward(pred, y)
    grad = loss_fn.backward()
    net.backward(grad)
    opt.step(net.parameters())

    if epoch % 1000 == 0:
        preds_bin = (pred > 0.5).astype(int)
        acc = (preds_bin == y).mean() * 100
        print(f"Epoch {epoch:5d} | loss={loss:.4f} | acc={acc:.1f}%")

# Final evaluation
pred = net.forward(X)
preds_bin = (pred > 0.5).astype(int)
print("\nFinal predictions:")
for xi, yi, pi in zip(X, y, preds_bin):
    print(f"  XOR{tuple(int(v) for v in xi)} = {int(yi[0])}  →  {int(pi[0])}")

acc = (preds_bin == y).mean()
if acc == 1.0:
    print("\n✓ XOR learned perfectly!")
    sys.exit(0)
else:
    print(f"\n✗ Accuracy {acc*100:.1f}% — did not converge.")
    sys.exit(1)
