"""Testable numerical models for the Day 53 visuals; no rendering or I/O."""

import math

import numpy as np

from from_scratch import initialize_parameters, loss_and_gradients, predict_proba

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([0, 1, 1, 0])
ACTIVATIONS = ("identity", "step", "sigmoid", "tanh", "relu", "gelu")


def activation(name, z, *, derivative=False):
    """Exact erf-based GELU; NaN denotes an undefined derivative at zero."""
    z = np.asarray(z, dtype=float)
    if not np.isfinite(z).all():
        raise ValueError("activation inputs must be finite")
    if name == "identity":
        return np.ones_like(z) if derivative else z.copy()
    if name == "step":
        return np.where(z == 0, np.nan, 0.0) if derivative else (z > 0).astype(float)
    if name == "sigmoid":
        p = np.exp(-np.logaddexp(0.0, -z))
        return p * (1 - p) if derivative else p
    if name == "tanh":
        return 1 - np.tanh(z)**2 if derivative else np.tanh(z)
    if name == "relu":
        return np.where(z == 0, np.nan, (z > 0).astype(float)) if derivative else np.maximum(0, z)
    if name == "gelu":
        erf = np.vectorize(math.erf, otypes=[float])(z / np.sqrt(2))
        cdf = 0.5 * (1 + erf)
        return cdf + z * np.exp(-z**2 / 2) / np.sqrt(2 * np.pi) if derivative else z * cdf
    raise ValueError(f"unknown activation: {name}")


def relu_features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or not np.isfinite(X).all():
        raise ValueError("X must be a finite (n,2) array")
    summed = X.sum(axis=1)
    return np.column_stack((np.maximum(0, summed), np.maximum(0, summed - 1)))


def grid_points(low=-0.5, high=1.5, size=65):
    if not isinstance(size, int) or size < 3 or not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("grid needs size >= 3 and finite low < high")
    axis = np.linspace(low, high, size)
    xx, yy = np.meshgrid(axis, axis)
    return xx, yy, np.column_stack((xx.ravel(), yy.ravel()))


def boundary_segment(weights, bias, low, high):
    """Clip w1*x+w2*y+b=0 to a square, including vertical/degenerate cases."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (2,) or not np.isfinite([*w, bias, low, high]).all() or low >= high:
        raise ValueError("boundary requires two finite weights and low < high")
    points = []
    if abs(w[1]) > 1e-12:
        points.extend((x, -(w[0] * x + bias) / w[1]) for x in (low, high))
    if abs(w[0]) > 1e-12:
        points.extend((-(w[1] * y + bias) / w[0], y) for y in (low, high))
    valid = []
    for point in points:
        if all(low - 1e-9 <= v <= high + 1e-9 for v in point) and not any(np.allclose(point, p) for p in valid):
            valid.append(point)
    return np.asarray(valid[:2], dtype=float).reshape(-1, 2)


def perceptron_trace(seed=42, max_epochs=100):
    """Record each genuine mistake/zero-margin update, using the core rule."""
    if not isinstance(max_epochs, int) or max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(-1, 1, (400, 2))
    scores = candidates @ np.array([1.0, 0.6]) - 0.1
    X = candidates[np.abs(scores) > 0.3][:48]
    y = np.where(X @ np.array([1.0, 0.6]) - 0.1 > 0, 1, -1)
    weights, bias, updates = np.zeros(2), 0.0, 0
    states = []

    def record(epoch):
        scores = X @ weights + bias
        states.append({"epoch": epoch, "update": updates, "weights": weights.copy(), "bias": bias,
                       "training_errors": int(np.sum(np.where(scores > 0, 1, -1) != y)),
                       "margin_violations": int(np.sum(y * scores <= 0))})

    record(0)
    certified = False
    for epoch in range(1, max_epochs + 1):
        epoch_updates = 0
        for row, target in zip(X, y):
            if target * (row @ weights + bias) <= 0:
                weights += target * row
                bias += float(target)
                updates += 1
                epoch_updates += 1
                record(epoch)
        if epoch_updates == 0:
            certified = True
            record(epoch)
            break
    return {"X": X, "y": y, "states": states, "certified": certified}


def training_trace(seed=42, *, epochs=10000, frames=40, learning_rate=0.5):
    """Checkpoints are exact parameters after the indicated number of updates."""
    if not isinstance(epochs, int) or epochs < 1 or not isinstance(frames, int) or frames < 2:
        raise ValueError("epochs >= 1 and frames >= 2 are required")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    selected = set(np.unique(np.rint(np.geomspace(1, epochs + 1, frames) - 1).astype(int)))
    selected.update((0, epochs))
    parameters = initialize_parameters(2, 8, seed=seed)
    losses, states = [], []
    step_norm = 0.0
    for epoch in range(epochs + 1):
        loss, grads = loss_and_gradients(XOR_X, XOR_Y, parameters)
        losses.append(loss)
        if epoch in selected:
            states.append({"epoch": epoch, "loss": loss,
                           "parameters": {k: v.copy() for k, v in parameters.items()},
                           "gradient_norm": float(np.sqrt(sum(np.sum(g*g) for g in grads.values()))),
                           "previous_step_norm": step_norm,
                           "probabilities": predict_proba(XOR_X, parameters)})
        if epoch < epochs:
            step_norm = learning_rate * np.sqrt(sum(np.sum(g*g) for g in grads.values()))
            parameters = {k: v - learning_rate * grads[k] for k, v in parameters.items()}
    return {"losses": np.array(losses), "states": states, "learning_rate": learning_rate}


def check_gradients(parameters, epsilon=1e-6):
    """Central differences for every coordinate on the actual XOR objective."""
    _, analytical = loss_and_gradients(XOR_X, XOR_Y, parameters)
    errors = []
    for name, array in parameters.items():
        for index in np.ndindex(array.shape):
            plus = {k: v.copy() for k, v in parameters.items()}
            minus = {k: v.copy() for k, v in parameters.items()}
            plus[name][index] += epsilon
            minus[name][index] -= epsilon
            numerical = (loss_and_gradients(XOR_X, XOR_Y, plus)[0] - loss_and_gradients(XOR_X, XOR_Y, minus)[0]) / (2 * epsilon)
            errors.append(abs(numerical - analytical[name][index]))
    return float(max(errors))


def fixed_forward(seed=42, x=(0.0, 1.0)):
    """Fixed, untrained 2-4-4-1 network: tanh hidden layers, sigmoid output."""
    rng = np.random.default_rng(seed)
    widths = (2, 4, 4, 1)
    weights = [rng.normal(0, 0.8, (a, b)) for a, b in zip(widths[:-1], widths[1:])]
    biases = [rng.normal(0, 0.15, b) for b in widths[1:]]
    values, logits = [np.asarray(x, dtype=float)], []
    if values[0].shape != (2,) or not np.isfinite(values[0]).all():
        raise ValueError("forward input needs two finite values")
    for index, (w, b) in enumerate(zip(weights, biases)):
        z = values[-1] @ w + b
        logits.append(z)
        values.append(activation("sigmoid" if index == 2 else "tanh", z))
    return {"weights": weights, "biases": biases, "values": values, "logits": logits,
            "parameters": sum(w.size + b.size for w, b in zip(weights, biases))}


def loss_slice(parameters, *, size=41, extent=4.0):
    """Vary only W2[0,0] and b2[0], keeping all other coordinates fixed."""
    if not isinstance(size, int) or size < 3 or not np.isfinite(extent) or extent <= 0:
        raise ValueError("slice needs size >= 3 and finite positive extent")
    offsets = np.linspace(-extent, extent, size)
    hidden = np.tanh(XOR_X @ parameters["W1"] + parameters["b1"])
    base = (hidden @ parameters["W2"] + parameters["b2"]).ravel()
    da, db = np.meshgrid(offsets, offsets)
    logits = base[None, None, :] + da[..., None] * hidden[:, 0] + db[..., None]
    losses = np.logaddexp(0, (1 - 2 * XOR_Y) * logits).mean(axis=-1)
    return da, db, losses


def target_function(x):
    x = np.asarray(x, dtype=float)
    return np.sin(2 * np.pi * x) + 0.3 * x**2


def relu_approximation(x, width):
    """Linear interpolant on [0,1], represented with exactly width ReLU units.

    One unit implements x on this domain; the remaining units change slope
    at interior knots. Output bias is f(0). No fitting/optimization occurs.
    """
    if not isinstance(width, int) or width < 2:
        raise ValueError("width must be an integer >= 2")
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("constructive approximation is restricted to [0,1]")
    knots = np.linspace(0, 1, width + 1)
    values = target_function(knots)
    slopes = np.diff(values) / np.diff(knots)
    coefficients = np.r_[slopes[0], np.diff(slopes)]
    features = np.maximum(0, x[..., None] - knots[:-1])
    return values[0] + features @ coefficients
