"""Deterministic synthetic training and unequal-microbatch gradient check."""

import json
import platform

import numpy as np

from from_scratch import initialize_parameters, loss_and_gradients, sgd_step


def run_study() -> dict:
    X = np.linspace(-1, 1, 128).reshape(-1, 1)
    y = np.sin(3 * X) + 0.2 * X
    p = initialize_parameters(seed=7)
    initial = {name: value.copy() for name, value in p.items()}
    _, full_grad = loss_and_gradients(X, y, p)
    accumulated = {name: np.zeros_like(value) for name, value in p.items()}
    for indices in (slice(0, 37), slice(37, 128)):
        _, gradient = loss_and_gradients(X[indices], y[indices], p)
        weight = len(X[indices]) / len(X)
        for name in p:
            accumulated[name] += weight * gradient[name]

    checkpoints = []
    for step in range(2001):
        loss, gradients = loss_and_gradients(X, y, p)
        if step in (0, 1, 100, 1000, 2000):
            checkpoints.append({
                "completed_updates": step, "loss": loss,
                "gradient_norms": {name: float(np.linalg.norm(g)) for name, g in gradients.items()},
            })
        if step < 2000:
            p = sgd_step(p, gradients, learning_rate=0.1)

    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "data": "synthetic y=sin(3x)+0.2x, 128 evenly spaced x in [-1,1]; no holdout",
        "configuration": {"seed": 7, "dtype": "float64", "hidden": 8,
                          "initial_weight_std": 0.3, "learning_rate": 0.1,
                          "updates": 2000, "loss": "half mean squared error"},
        "microbatch_sizes": [37, 91],
        "weighted_microbatch_max_absolute_error": max(
            float(np.max(np.abs(full_grad[name] - accumulated[name]))) for name in p
        ),
        "parameter_displacement_norm": float(np.sqrt(sum(
            np.sum((p[name] - initial[name])**2) for name in p
        ))),
        "checkpoints": checkpoints,
        "interpretation_status": "author review required",
    }


if __name__ == "__main__":
    print(json.dumps(run_study(), indent=2, allow_nan=False))
