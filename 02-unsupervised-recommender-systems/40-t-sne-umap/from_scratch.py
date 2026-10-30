"""Dense fixed-bandwidth t-SNE objective: an educational simplification."""

import json
import platform

import numpy as np


def finite_matrix(X, name):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1 or not np.isfinite(X).all():
        raise ValueError(f"{name} must be a finite matrix with >=2 rows and >=1 feature")
    return X


def gaussian_affinities(X, sigma=2.0):
    """Symmetric P with one fixed sigma, NOT perplexity-matched t-SNE."""
    X = finite_matrix(X, "X")
    if not np.isscalar(sigma) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = (X[:, None, :] - X[None, :, :]) / sigma
        logits = -0.5 * np.sum(delta * delta, axis=2)
    np.fill_diagonal(logits, -np.inf)
    # Subtracting each row maximum avoids an all-zero row at small bandwidths.
    row_max = logits.max(axis=1, keepdims=True)
    if not np.isfinite(row_max).all():
        raise ValueError("distances overflowed; rescale X or increase sigma")
    weights = np.exp(logits - row_max)
    conditional = weights / weights.sum(axis=1, keepdims=True)
    return (conditional + conditional.T) / (2 * len(X))


def loss_and_gradient(Y, P):
    """Return KL(P||Q) and its exact gradient for symmetric normalized P."""
    Y = finite_matrix(Y, "Y")
    P = np.asarray(P, dtype=float)
    if P.shape != (len(Y), len(Y)) or not np.isfinite(P).all() or np.any(P < 0):
        raise ValueError("P must be a finite nonnegative n-by-n matrix")
    if (not np.allclose(P, P.T, atol=1e-12, rtol=0)
            or np.any(np.diag(P) != 0) or not np.isclose(P.sum(), 1, atol=1e-12, rtol=0)):
        raise ValueError("P must be symmetric, zero-diagonal and sum to one")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = Y[:, None, :] - Y[None, :, :]
        distance2 = np.sum(delta * delta, axis=2)
    if not np.isfinite(distance2).all():
        raise ValueError("embedding distances overflowed; rescale Y")
    numerator = 1.0 / (1.0 + distance2)
    np.fill_diagonal(numerator, 0)
    normalizer = numerator.sum()
    Q = numerator / normalizer
    positive = P > 0
    # Use log Q without clipping; clipping would change the differentiated loss.
    log_Q = -np.log1p(distance2) - np.log(normalizer)
    loss = np.sum(P[positive] * (np.log(P[positive]) - log_Q[positive]))
    gradient = 4 * np.sum(((P - Q) * numerator)[:, :, None] * delta, axis=1)
    return float(loss), gradient


def optimize(X, sigma=2.0, steps=300, learning_rate=10.0, seed=42):
    """Plain fixed-step descent; intentionally omits production t-SNE machinery."""
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not np.isscalar(learning_rate) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    P = gaussian_affinities(X, sigma)
    Y = np.random.default_rng(seed).normal(0, 0.01, size=(len(P), 2))
    Y -= Y.mean(axis=0)
    history = []
    for _ in range(steps):
        loss, gradient = loss_and_gradient(Y, P)
        history.append(loss)
        Y -= learning_rate * gradient
        Y -= Y.mean(axis=0)
    history.append(loss_and_gradient(Y, P)[0])
    return Y, np.asarray(history)


def run_demo():
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(-1.5, 0.5, (10, 4)), rng.normal(1.5, 0.5, (10, 4))])
    Y, history = optimize(X)
    return {
        "data": "synthetic: two Gaussian groups, 10 rows each, 4 features, std=0.5",
        "configuration": {"data_seed": 42, "initialization_seed": 42, "sigma": 2.0,
                          "steps": 300, "learning_rate": 10.0, "initialization_std": 0.01},
        "python": platform.python_version(), "numpy": np.__version__,
        "initial_kl": float(history[0]), "final_kl": float(history[-1]),
        "finite_coordinates": bool(np.isfinite(Y).all()),
        "interpretation_status": "author review required; simplified objective only",
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, allow_nan=False))
