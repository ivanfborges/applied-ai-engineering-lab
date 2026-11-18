"""Educational reverse pass for a tanh regressor; no autodiff framework."""

from __future__ import annotations

import json

import numpy as np


def initialize_parameters(n_features: int = 1, hidden: int = 8, seed: int = 7) -> dict:
    for name, value in (("n_features", n_features), ("hidden", hidden)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, 0.3, (n_features, hidden)),
        "b1": np.zeros((1, hidden)),
        "W2": rng.normal(0, 0.3, (hidden, 1)),
        "b2": np.zeros((1, 1)),
    }


def _parameters(parameters: dict) -> dict:
    if set(parameters) != {"W1", "b1", "W2", "b2"}:
        raise ValueError("parameters must contain exactly W1, b1, W2 and b2")
    p = {name: np.asarray(value, dtype=np.float64) for name, value in parameters.items()}
    if any(value.ndim != 2 or not np.isfinite(value).all() for value in p.values()):
        raise ValueError("parameters must be finite two-dimensional arrays")
    d, h = p["W1"].shape
    if min(d, h) < 1 or any(
        p[name].shape != shape
        for name, shape in (("b1", (1, h)), ("W2", (h, 1)), ("b2", (1, 1)))
    ):
        raise ValueError("expected shapes W1=(d,h), b1=(1,h), W2=(h,1), b2=(1,1)")
    return p


def _forward(X, y, parameters):
    p = _parameters(parameters)
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != p["W1"].shape[0]:
        raise ValueError("X must be a nonempty (n,d) matrix matching W1")
    if y.shape != (X.shape[0], 1):
        raise ValueError("y must have shape (n,1); flat targets would broadcast incorrectly")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must be finite")
    with np.errstate(over="raise", invalid="raise"):
        H = np.tanh(X @ p["W1"] + p["b1"])
        residual = H @ p["W2"] + p["b2"] - y
        loss = float(0.5 * np.mean(residual**2))
    return loss, X, H, residual, p


def loss_value(X, y, parameters) -> float:
    """Half mean squared error for exactly one output per observation."""
    return _forward(X, y, parameters)[0]


def loss_and_gradients(X, y, parameters) -> tuple[float, dict]:
    """Compute all gradients at the same parameters without updating them."""
    loss, X, H, residual, p = _forward(X, y, parameters)
    with np.errstate(over="raise", invalid="raise"):
        delta2 = residual / X.shape[0]
        dW2 = H.T @ delta2
        db2 = delta2.sum(axis=0, keepdims=True)
        delta1 = (delta2 @ p["W2"].T) * (1 - H**2)
        dW1 = X.T @ delta1
        db1 = delta1.sum(axis=0, keepdims=True)
    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def sgd_step(parameters: dict, gradients: dict, learning_rate: float) -> dict:
    """Return a simultaneous full-batch gradient descent update in new arrays."""
    p = _parameters(parameters)
    if not np.isscalar(learning_rate) or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if set(gradients) != set(p):
        raise ValueError("gradients must have the same keys as parameters")
    result = {}
    for name, value in p.items():
        gradient = np.asarray(gradients[name], dtype=np.float64)
        if gradient.shape != value.shape or not np.isfinite(gradient).all():
            raise ValueError(f"gradient {name} must be finite and match its parameter shape")
        with np.errstate(over="raise", invalid="raise"):
            result[name] = value - learning_rate * gradient
    return result


def finite_difference_gradients(X, y, parameters, epsilon: float = 1e-5) -> dict:
    """Central differences for every coordinate; intentionally expensive."""
    if not np.isscalar(epsilon) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    p = _parameters(parameters)
    result = {}
    for name, value in p.items():
        numerical = np.empty_like(value)
        for index in np.ndindex(value.shape):
            plus = {key: array.copy() for key, array in p.items()}
            minus = {key: array.copy() for key, array in p.items()}
            plus[name][index] += epsilon
            minus[name][index] -= epsilon
            numerical[index] = (loss_value(X, y, plus) - loss_value(X, y, minus)) / (2 * epsilon)
        result[name] = numerical
    return result


def shared_path_trace(x=3.0, w=0.2, b=-0.1, y=0.4) -> dict:
    """Hand-written backward pass for z=wx+b, a=tanh(z), prediction=a+z."""
    if not all(np.isscalar(v) and np.isfinite(v) for v in (x, w, b, y)):
        raise ValueError("x, w, b and y must be finite scalars")
    z = w * x + b
    a = float(np.tanh(z))
    prediction = a + z
    residual = prediction - y
    through_activation = residual * (1 - a**2)
    through_skip = residual
    dz = through_activation + through_skip  # Both downstream paths contribute.
    return {
        "z": z, "a": a, "prediction": prediction, "loss": 0.5 * residual**2,
        "dL_dz_via_tanh": through_activation, "dL_dz_via_skip": through_skip,
        "dL_dz": dz, "dL_dw": dz * x, "dL_db": dz, "dL_dx": dz * w,
    }


def main() -> None:
    X = np.linspace(-1, 1, 9).reshape(-1, 1)
    y = np.sin(3 * X) + 0.2 * X
    p = initialize_parameters()
    loss, analytical = loss_and_gradients(X, y, p)
    numerical = finite_difference_gradients(X, y, p)
    errors = {key: float(np.max(np.abs(analytical[key] - numerical[key]))) for key in p}
    print(json.dumps({
        "data": "synthetic, nine evenly spaced inputs in [-1,1]",
        "numpy": np.__version__, "seed": 7, "dtype": "float64",
        "epsilon": 1e-5, "parameters_checked": sum(v.size for v in p.values()),
        "loss": loss, "max_absolute_gradient_errors": errors,
        "shared_path_trace": shared_path_trace(),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
