"""Educational untied linear autoencoder with explicit MSE gradients."""

from __future__ import annotations

import json
import numpy as np


def matrix(values, name="X"):
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or min(result.shape) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return result


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def linear_loss_gradients(X, W, V):
    X, W, V = matrix(X), matrix(W, "W"), matrix(V, "V")
    if W.shape[0] != X.shape[1] or V.shape != (W.shape[1], X.shape[1]):
        raise ValueError("Expected X[n,d], W[d,k], V[k,d]")
    hidden = X @ W
    error = hidden @ V - X
    scale = 2 / X.size
    return float(np.mean(error**2)), scale * X.T @ (error @ V.T), scale * hidden.T @ error


def fit_linear_autoencoder(X, latent_dim=2, *, steps=2000, learning_rate=0.4, seed=42):
    """Fit centered inputs; return weights and loss before/after each full update."""
    X = matrix(X)
    positive_integer(latent_dim, "latent_dim")
    positive_integer(steps, "steps")
    if latent_dim > X.shape[1]:
        raise ValueError("latent_dim must not exceed the input dimension")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, (X.shape[1], latent_dim))
    V = rng.normal(0, 0.1, (latent_dim, X.shape[1]))
    losses = []
    for _ in range(steps):
        loss, grad_W, grad_V = linear_loss_gradients(X, W, V)
        if not np.isfinite(loss) or not np.isfinite(grad_W).all() or not np.isfinite(grad_V).all():
            raise ValueError("Optimization diverged; reduce learning_rate")
        losses.append(loss)
        # Both gradients use the same pre-update weights.
        W = W - learning_rate * grad_W
        V = V - learning_rate * grad_V
    loss, _, _ = linear_loss_gradients(X, W, V)
    if not np.isfinite(loss):
        raise ValueError("Optimization diverged; reduce learning_rate")
    return W, V, np.asarray([*losses, loss])


def run_demo():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 2)) @ rng.normal(size=(2, 8))
    X += 0.1 * rng.normal(size=X.shape)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    W, V, losses = fit_linear_autoencoder(X)
    # SVD gives the optimal training rank-2 reconstruction for this objective.
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    pca_reconstruction = X @ vt[:2].T @ vt[:2]
    return {
        "synthetic": True, "seed": 42, "shape": list(X.shape), "latent_dim": 2,
        "steps": 2000, "learning_rate": 0.4, "numpy": np.__version__,
        "initial_mse": float(losses[0]), "linear_ae_mse": float(losses[-1]),
        "pca_mse": float(np.mean((X - pca_reconstruction)**2)),
        "reconstruction_rank": int(np.linalg.matrix_rank(X @ W @ V)),
        "interpretation_status": "author review required",
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, allow_nan=False))
