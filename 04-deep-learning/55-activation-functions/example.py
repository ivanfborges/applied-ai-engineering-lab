"""Deterministic activation diagnostics on synthetic inputs; prints JSON only."""

import json
import platform

import numpy as np
import scipy
from scipy.special import expit, logsumexp, softmax as scipy_softmax

from from_scratch import (
    ELEMENTWISE, activation_and_derivative, cross_entropy_with_logits,
    finite_difference_jacobian, softmax,
)


def run_study():
    x = np.array([-8.0, -2.0, -0.5, 0.5, 2.0, 8.0])
    elementwise = {}
    for name in ELEMENTWISE:
        values, derivative = activation_and_derivative(x, name)
        numerical = finite_difference_jacobian(
            lambda z: activation_and_derivative(z, name)[0], x
        )
        error = float(np.max(np.abs(np.diag(derivative) - numerical)))
        np.testing.assert_allclose(np.diag(derivative), numerical, rtol=1e-6, atol=1e-9)
        elementwise[name] = {"values": values.tolist(), "derivatives": derivative.tolist(),
                             "jacobian_max_absolute_error": error}
    np.testing.assert_allclose(activation_and_derivative(x, "sigmoid")[0], expit(x))

    z = np.array([[2.0, 1.0, -1.0], [-1.0, 0.0, 3.0]])
    y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    temperatures = {}
    for t in (0.5, 1.0, 2.0):
        p = softmax(z, temperature=t)
        np.testing.assert_allclose(p, scipy_softmax(z / t, axis=-1), atol=1e-15)
        shift_error = float(np.max(np.abs(p - softmax(z + 10000, temperature=t))))
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-15)
        loss, gradient = cross_entropy_with_logits(z, y, temperature=t)
        reference = float(np.mean(logsumexp(z / t, axis=1) - np.sum(y * z / t, axis=1)))
        np.testing.assert_allclose(loss, reference, atol=1e-15)
        epsilon = 1e-5
        numerical = np.empty_like(z)
        for index in np.ndindex(z.shape):
            plus, minus = z.copy(), z.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numerical[index] = (
                cross_entropy_with_logits(plus, y, t)[0]
                - cross_entropy_with_logits(minus, y, t)[0]
            ) / (2 * epsilon)
        error = float(np.max(np.abs(gradient - numerical)))
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-9)
        temperatures[str(t)] = {
            "probabilities": p.tolist(), "mean_cross_entropy": loss,
            "entropy_per_row_nats": (-np.sum(p * np.log(p), axis=1)).tolist(),
            "shift_max_absolute_error": shift_error,
            "loss_gradient_max_absolute_error": error,
        }
    extreme_loss, _ = cross_entropy_with_logits([[1000.0, -1000.0]], [[0.0, 1.0]])
    np.testing.assert_allclose(extreme_loss, 2000.0)
    return {
        "data": "fixed synthetic preactivations and two three-class logit rows; no training",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "preactivations": x.tolist(), "negative_slope": 0.01,
        "finite_difference_epsilon": 1e-5, "elementwise": elementwise,
        "logits": z.tolist(), "targets": y.tolist(), "temperatures": temperatures,
        "extreme_incorrect_prediction_loss": extreme_loss,
        "interpretation_status": "author review required",
    }


if __name__ == "__main__":
    print(json.dumps(run_study(), indent=2, allow_nan=False))
