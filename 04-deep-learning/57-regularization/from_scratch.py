"""Educational forward mechanisms; no autodiff or production layer replacement."""

from __future__ import annotations

import json

import numpy as np


def finite_array(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty and finite.")
    return result


def inverted_dropout(values, p: float, rng: np.random.Generator, *, training=True):
    """Keep the conditional activation mean, not the whole nonlinear prediction."""
    x = finite_array(values, "values")
    if not np.isfinite(p) or not 0 <= p < 1:
        raise ValueError("p must be finite and in [0, 1).")
    if not isinstance(training, bool):
        raise ValueError("training must be a bool.")
    if not training or p == 0:
        return x.copy()
    mask = rng.binomial(1, 1 - p, size=x.shape)
    return x * mask / (1 - p)


def batch_norm_training(values, gamma, beta, *, eps=1e-5):
    """2D training forward only: population variance, no running state/backward."""
    x = finite_array(values, "values")
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("values must have shape (batch >= 2, features).")
    scale = finite_array(gamma, "gamma")
    shift = finite_array(beta, "beta")
    if scale.shape != (x.shape[1],) or shift.shape != scale.shape:
        raise ValueError("gamma and beta must have one value per feature.")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive.")
    return scale * (x - x.mean(axis=0)) / np.sqrt(x.var(axis=0) + eps) + shift


def sgd_l2_pair(weights, gradient, *, lr: float, decay: float):
    """L + decay/2 * ||w||^2 versus decoupled vanilla SGD; no momentum."""
    w = finite_array(weights, "weights")
    g = finite_array(gradient, "gradient")
    if w.shape != g.shape:
        raise ValueError("weights and gradient must have identical shapes.")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive.")
    if not np.isfinite(decay) or decay < 0:
        raise ValueError("decay must be finite and nonnegative.")
    return w - lr * (g + decay * w), (1 - lr * decay) * w - lr * g


def run_demo():
    rng = np.random.default_rng(57)
    dropped = inverted_dropout(np.ones(100_000), 0.3, rng)
    x = rng.normal(loc=5, scale=3, size=(32, 4))
    normalized = batch_norm_training(x, np.ones(4), np.zeros(4))
    coupled, decoupled = sgd_l2_pair([2, -1], [0.5, -0.2], lr=0.1, decay=0.2)
    return {
        "hypothesis": "Dropout moments approach their conditional expectations; "
                      "BatchNorm standardizes a batch; vanilla SGD/L2 updates agree.",
        "configuration": {"seed": 57, "dropout_n": 100_000, "p": 0.3,
                          "batch_shape": [32, 4], "eps": 1e-5,
                          "weights": [2, -1], "gradient": [0.5, -0.2],
                          "lr": 0.1, "decay": 0.2},
        "numpy_version": np.__version__,
        "dropout_mean": float(dropped.mean()), "dropout_variance": float(dropped.var()),
        "expected_mean": 1.0, "expected_variance": 0.3 / 0.7,
        "batch_norm_means": normalized.mean(axis=0).tolist(),
        "batch_norm_variances": normalized.var(axis=0).tolist(),
        "sgd_coupled": coupled.tolist(), "sgd_decoupled": decoupled.tolist(),
        "sgd_max_absolute_difference": float(np.max(np.abs(coupled - decoupled))),
        "interpretation_candidate": "These controls illustrate mechanism identities only.",
        "interpretation_status": "author review required",
        "limitation": "Synthetic activations; finite-sample moments; no gradients or inference "
                      "state in NumPy BatchNorm; SGD equivalence does not extend to Adam/L2.",
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, allow_nan=False))
