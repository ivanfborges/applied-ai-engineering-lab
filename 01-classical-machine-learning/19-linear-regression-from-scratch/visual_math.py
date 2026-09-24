"""Small numerical helpers for the Day 19 visual experiments (MSE convention)."""

from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class Trace:
    parameters: np.ndarray
    losses: np.ndarray
    status: str
    stop_iteration: int


def linear_data(seed: int = 19):
    """Return 45 synthetic observations; shared across the first six views."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-2, 4, 45)
    return x[:, None], 1.5 + 1.8 * x + rng.normal(0, 0.9, len(x))


def design(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite matrix.")
    return np.column_stack((np.ones(len(X)), X))


def validate_system(A, y):
    A, y = np.asarray(A, dtype=float), np.asarray(y, dtype=float)
    if A.ndim != 2 or min(A.shape) == 0 or y.shape != (len(A),):
        raise ValueError("Expected nonempty A (n, p) and y (n,).")
    if not np.isfinite(A).all() or not np.isfinite(y).all():
        raise ValueError("A and y must be finite.")
    return A, y


def mse_gradient(A, y, beta):
    """A already includes an intercept if one is to be optimized."""
    residual = A @ beta - y
    return float(np.mean(residual ** 2)), 2 * A.T @ residual / len(y)


def gradient_trace(A, y, start, rate, steps=200, tol=1e-9, loss_limit=1e10):
    """Record initial and accepted iterates; stop before unsafe values enter plots."""
    A, y = validate_system(A, y)
    beta = np.asarray(start, dtype=float).copy()
    if beta.shape != (A.shape[1],) or not np.isfinite(beta).all():
        raise ValueError("start must contain one finite value per design column.")
    for value in (rate, tol, loss_limit):
        if not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or value <= 0:
            raise ValueError("rate, tol and loss_limit must be positive finite scalars.")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    positions, losses = [], []
    status = "budget exhausted"
    for iteration in range(steps + 1):
        try:
            with np.errstate(over="raise", invalid="raise"):
                loss, gradient = mse_gradient(A, y, beta)
            safe = np.isfinite(loss) and np.isfinite(gradient).all() and loss <= loss_limit
        except FloatingPointError:
            safe = False
        if not safe:
            status = "diverged (safety cutoff)"
            break
        positions.append(beta.copy())
        losses.append(loss)
        if np.max(np.abs(gradient)) <= tol:
            status = "converged"
            break
        if iteration < steps:
            try:
                with np.errstate(over="raise", invalid="raise"):
                    beta = beta - rate * gradient
            except FloatingPointError:
                status = "diverged (safety cutoff)"
                iteration += 1
                break
    if not positions:
        raise ValueError("Initial loss is nonfinite or exceeds loss_limit.")
    return Trace(np.array(positions), np.array(losses), status, iteration)


def loss_surface(A, y, first, second):
    """MSE grid for an exactly two-parameter design, not a hidden projection."""
    A, y = validate_system(A, y)
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if A.shape[1] != 2 or any(v.ndim != 1 or v.size < 2 or not np.isfinite(v).all() for v in (first, second)):
        raise ValueError("Use a two-column design and finite 1-D axes of length >= 2.")
    B0, B1 = np.meshgrid(first, second)
    # Row-wise evaluation avoids an n x grid x grid allocation.
    losses = np.empty_like(B0)
    for row in range(len(second)):
        predictions = A[:, :1] * first + A[:, 1:2] * second[row]
        losses[row] = np.mean((predictions - y[:, None]) ** 2, axis=0)
    return B0, B1, losses


def scaling_data():
    """Centering profiles out the intercept, leaving a genuine 2-D slope loss."""
    rng = np.random.default_rng(1907)
    X = rng.uniform(size=(160, 2)) * [1, 10000]
    y = 2 + X @ np.array([8., 0.0008]) + rng.normal(0, 0.2, len(X))
    scaler = StandardScaler().fit(X)
    centered, target = X - scaler.mean_, y - y.mean()
    return centered, scaler.transform(X), target, scaler


def comparison_data():
    rng = np.random.default_rng(1911)
    X = rng.normal(size=(320, 3)) * [1, 10, 0.2]
    y = 2.5 + X @ np.array([3., -0.3, 4.]) + rng.normal(0, 0.5, len(X))
    return X, y

