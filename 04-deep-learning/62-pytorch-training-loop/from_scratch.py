"""Educational logistic BCE gradients and mini-batch SGD using NumPy only."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import numpy as np


def loss_and_gradients(features: np.ndarray, targets: np.ndarray,
                       weights: np.ndarray, bias: float) -> tuple:
    features, targets, weights = (np.asarray(a, dtype=float) for a in (features, targets, weights))
    if features.ndim != 2 or min(features.shape) < 1:
        raise ValueError("features must have nonempty shape (N, D)")
    if targets.shape != (len(features),) or weights.shape != (features.shape[1],):
        raise ValueError("targets must have shape (N,) and weights shape (D,)")
    if np.ndim(bias) != 0 or not np.isfinite(bias):
        raise ValueError("bias must be a finite scalar")
    if not all(np.isfinite(a).all() for a in (features, targets, weights)):
        raise ValueError("all inputs must be finite")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("targets must be binary, 0 or 1")
    logits = features @ weights + bias
    if not np.isfinite(logits).all():
        raise ValueError("computed logits must be finite")
    # exp(-logaddexp(0, -z)) is a stable sigmoid without clipping the objective.
    probabilities = np.exp(-np.logaddexp(0, -logits))
    signed_logits = np.where(targets == 1, -logits, logits)
    loss = np.logaddexp(0, signed_logits).mean()
    errors = probabilities - targets
    return float(loss), features.T @ errors / len(features), float(errors.mean())


def sgd_epoch(features: np.ndarray, targets: np.ndarray, weights: np.ndarray,
              bias: float, *, learning_rate: float, batch_size: int,
              rng: np.random.Generator) -> tuple:
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    loss_and_gradients(features, targets, weights, bias)
    weights = np.asarray(weights, dtype=float).copy()
    total_loss = 0.0
    order = rng.permutation(len(features))
    for start in range(0, len(features), batch_size):
        idx = order[start:start + batch_size]
        loss, grad_w, grad_b = loss_and_gradients(features[idx], targets[idx], weights, bias)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        total_loss += loss * len(idx)
    return weights, float(bias), total_loss / len(features)


def run_demo() -> dict:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(400, 2))
    targets = (features[:, 0] - 0.7 * features[:, 1] + rng.normal(scale=0.3, size=400) > 0).astype(float)
    weights, bias = np.zeros(2), 0.0
    initial_loss = loss_and_gradients(features, targets, weights, bias)[0]
    for _ in range(30):
        weights, bias, online_loss = sgd_epoch(
            features, targets, weights, bias, learning_rate=0.1, batch_size=50, rng=rng,
        )
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "configuration": {"seed": 42, "samples": 400, "epochs": 30,
                          "batch_size": 50, "learning_rate": 0.1, "steps": 240},
        "hypothesis": "Manual mini-batch SGD can reduce training BCE on a noisy linear synthetic signal.",
        "initial_training_loss": initial_loss,
        "final_online_loss": online_loss,
        "final_training_loss": loss_and_gradients(features, targets, weights, bias)[0],
        "training_accuracy": float(((features @ weights + bias >= 0) == targets).mean()),
        "interpretation_candidate": "A training-only optimization demonstration; author review required.",
        "limitation": "Different synthetic data and optimizer from example.py; no holdout or generalization comparison.",
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, allow_nan=False))
