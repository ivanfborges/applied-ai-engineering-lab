"""Small, independently testable calculations for the Day 54 visual lab."""

from __future__ import annotations

import importlib.util

import numpy as np


def synthetic_data(n: int = 41):
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer >= 2")
    x = np.linspace(-1, 1, n, dtype=np.float64)
    return x, np.sin(3 * x) + 0.2 * x


def _data(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or y.shape != x.shape:
        raise ValueError("x and y must be nonempty one-dimensional arrays of equal shape")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must be finite")
    return x, y


def _theta(theta):
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (2,) or not np.isfinite(theta).all():
        raise ValueError("theta must be finite with shape (2,), ordered [w1,w2]")
    return theta


def loss_surface(w1, w2, x, y):
    """Actual half-MSE, broadcasting parameter grids with observations last."""
    x, y = _data(x, y)
    w1, w2 = np.broadcast_arrays(np.asarray(w1, float), np.asarray(w2, float))
    if not np.isfinite(w1).all() or not np.isfinite(w2).all():
        raise ValueError("surface parameters must be finite")
    with np.errstate(over="raise", invalid="raise"):
        residual = w2[..., None] * np.tanh(w1[..., None] * x) - y
        return 0.5 * np.mean(residual**2, axis=-1)


def loss_gradient(theta, x, y):
    """Gradients of mean(0.5*(w2*tanh(w1*x)-y)**2), in float64."""
    w1, w2 = _theta(theta)
    x, y = _data(x, y)
    with np.errstate(over="raise", invalid="raise"):
        h = np.tanh(w1 * x)
        residual = w2 * h - y
        loss = float(0.5 * np.mean(residual**2))
        gradient = np.array([np.mean(residual * w2 * (1 - h**2) * x),
                             np.mean(residual * h)])
    return loss, gradient


def central_difference(theta, x, y, epsilon=1e-5):
    theta = _theta(theta)
    if not np.isscalar(epsilon) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    gradient = np.empty(2)
    for j in range(2):
        offset = np.zeros(2)
        offset[j] = epsilon
        plus, minus = theta + offset, theta - offset
        gradient[j] = (loss_surface(*plus, x, y) - loss_surface(*minus, x, y)) / (2 * epsilon)
    return gradient


def descent(x, y, initial=(0.5, 0.8), learning_rate=0.1, steps=180):
    """Store each evaluated position and its gradient; stop on nonfinite arithmetic."""
    theta = _theta(initial).copy()
    _data(x, y)
    if not np.isscalar(learning_rate) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    positions, losses, gradients = [], [], []
    status = "completed"
    for step in range(steps + 1):
        try:
            loss, gradient = loss_gradient(theta, x, y)
            if not np.isfinite(loss) or not np.isfinite(gradient).all():
                raise FloatingPointError("nonfinite loss or gradient")
        except FloatingPointError:
            status = f"stopped before nonfinite evaluation at step {step}"
            break
        positions.append(theta.copy())
        losses.append(loss)
        gradients.append(gradient)
        if step < steps:
            with np.errstate(over="ignore", invalid="ignore"):
                theta = theta - learning_rate * gradient
            if not np.isfinite(theta).all():
                status = f"stopped before nonfinite update {step + 1}"
                break
    return {"theta": np.asarray(positions), "loss": np.asarray(losses),
            "gradient": np.asarray(gradients), "learning_rate": float(learning_rate),
            "requested_steps": steps, "status": status}


def scalar_graph(x=0.8, y=0.6, w1=0.7, b=-0.1, w2=1.2):
    values = np.asarray([x, y, w1, b, w2], dtype=np.float64)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("graph inputs must be finite scalars")
    x, y, w1, b, w2 = values
    with np.errstate(over="raise", invalid="raise"):
        z = w1 * x + b
        h = np.tanh(z)
        pred = w2 * h
        residual = pred - y
        local = np.array([residual, w2, 1 - h*h, x])
        dw1 = np.prod(local)
        db = residual * w2 * (1 - h*h)
        dw2 = residual * h
        return {"x": float(x), "y": float(y), "w1": float(w1), "b": float(b), "w2": float(w2),
                "z": float(z), "h": float(h), "prediction": float(pred),
                "loss": float(0.5 * residual**2), "local": local,
                "gradient": np.array([dw1, db, dw2]), "dL_dh": float(residual * w2),
                "dL_dz": float(db)}


def activations(z):
    z = np.asarray(z, dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError("activation inputs must be finite")
    sigmoid = np.exp(-np.logaddexp(0, -z))
    tanh = np.tanh(z)
    return {"sigmoid": (sigmoid, sigmoid * (1 - sigmoid)),
            "tanh": (tanh, 1 - tanh**2),
            "ReLU": (np.maximum(0, z), (z > 0).astype(float))}


def gradient_products(factor, depth=16, upstream=1.0):
    if not np.isscalar(factor) or not np.isfinite(factor) or factor <= 0:
        raise ValueError("factor must be positive and finite")
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 100:
        raise ValueError("depth must be an integer in [1,100]")
    if not np.isscalar(upstream) or not np.isfinite(upstream) or upstream == 0:
        raise ValueError("upstream must be finite and nonzero for logarithmic display")
    return abs(upstream) * factor ** np.arange(depth + 1, dtype=float)


def clip_global_norm(gradient, threshold=1.0):
    gradient = np.asarray(gradient, dtype=float)
    if gradient.ndim != 1 or gradient.size == 0 or not np.isfinite(gradient).all():
        raise ValueError("gradient must be a finite, nonempty vector")
    if not np.isscalar(threshold) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    norm = float(np.linalg.norm(gradient))
    if not np.isfinite(norm):
        raise ValueError("gradient norm overflowed")
    return gradient.copy() if norm <= threshold else gradient * (threshold / norm)


def accumulation(x, y, theta=(0.5, 0.8), sizes=(13, 28)):
    x, y = _data(x, y)
    if not sizes or any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in sizes):
        raise ValueError("sizes must contain positive integers")
    if sum(sizes) != len(x):
        raise ValueError("microbatch sizes must partition the full batch")
    pieces, start = [], 0
    for size in sizes:
        pieces.append(loss_gradient(theta, x[start:start+size], y[start:start+size])[1])
        start += size
    pieces = np.asarray(pieces)
    weighted = np.average(pieces, axis=0, weights=sizes)
    full = loss_gradient(theta, x, y)[1]
    return {"pieces": pieces, "sum": pieces.sum(axis=0), "unweighted": pieces.mean(axis=0),
            "weighted": weighted, "full": full, "sizes": list(sizes)}


def optional_autograd_check(x, y, theta=(0.5, 0.8)):
    """Reading reference remains optional; absence never counts as a passing comparison."""
    if importlib.util.find_spec("torch") is None:
        return {"status": "not performed: PyTorch is not installed"}
    import torch
    x, y = _data(x, y)
    weights = torch.tensor(_theta(theta), dtype=torch.float64, requires_grad=True)
    tx, ty = torch.tensor(x), torch.tensor(y)
    loss = 0.5 * ((weights[1] * torch.tanh(weights[0] * tx) - ty)**2).mean()
    loss.backward()
    manual = loss_gradient(theta, x, y)[1]
    error = float(np.max(np.abs(manual - weights.grad.detach().numpy())))
    if not np.allclose(manual, weights.grad.detach().numpy(), rtol=1e-9, atol=1e-12):
        raise ValueError("manual gradients disagree with available PyTorch autograd")
    return {"status": "compared", "torch": torch.__version__, "max_absolute_error": error}
