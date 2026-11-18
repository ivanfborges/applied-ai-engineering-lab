"""Educational perceptron and one-hidden-layer binary MLP; NumPy only."""

import json

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_rate(value):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError("learning_rate must be finite and positive")


def _features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite 2D matrix")
    return X


def _targets(y, n, allowed):
    y = np.asarray(y, dtype=float)
    if y.shape != (n,) or not np.isin(y, allowed).all():
        raise ValueError(f"y must have shape ({n},) and values in {allowed}")
    return y


def fit_perceptron(X, y, *, learning_rate=1.0, max_epochs=100):
    """Fixed-order updates for labels -1/+1, including zero-margin cases.

    A zero-mistake epoch certifies separation of these training points.
    Reaching the epoch cap is not a proof of nonseparability.
    """
    X = _features(X)
    y = _targets(y, len(X), (-1, 1))
    _positive_rate(learning_rate)
    _positive_integer(max_epochs, "max_epochs")
    weights, bias = np.zeros(X.shape[1]), 0.0
    mistakes = []
    for _ in range(max_epochs):
        count = 0
        for row, target in zip(X, y):
            if target * (row @ weights + bias) <= 0:
                weights += learning_rate * target * row
                bias += learning_rate * target
                count += 1
        mistakes.append(count)
        if count == 0:
            break
    return weights, float(bias), mistakes


def constructive_xor(X):
    """Two fixed ReLU units; exact XOR values only on binary inputs."""
    X = _features(X)
    if X.shape[1] != 2 or not np.isin(X, (0, 1)).all():
        raise ValueError("constructive_xor requires two binary features")
    summed = X.sum(axis=1)
    return np.maximum(0, summed) - 2 * np.maximum(0, summed - 1)


def initialize_parameters(n_features, n_hidden=8, *, seed=42):
    """Independent, fan-in-scaled weights break hidden-unit symmetry."""
    _positive_integer(n_features, "n_features")
    _positive_integer(n_hidden, "n_hidden")
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(size=(n_features, n_hidden)) / np.sqrt(n_features),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(size=(n_hidden, 1)) / np.sqrt(n_hidden),
        "b2": np.zeros(1),
    }


def _parameters(parameters, n_features):
    if set(parameters) != {"W1", "b1", "W2", "b2"}:
        raise ValueError("parameters must contain W1, b1, W2 and b2")
    arrays = {key: np.asarray(value, dtype=float) for key, value in parameters.items()}
    W1 = arrays["W1"]
    if W1.ndim != 2 or W1.shape[0] != n_features or W1.shape[1] < 1:
        raise ValueError("W1 shape must be (n_features, n_hidden)")
    h = W1.shape[1]
    expected = {"W1": (n_features, h), "b1": (h,), "W2": (h, 1), "b2": (1,)}
    if any(a.shape != expected[key] or not np.isfinite(a).all() for key, a in arrays.items()):
        raise ValueError("parameter shapes must agree and values must be finite")
    return arrays


def _sigmoid(logits):
    # logaddexp avoids overflow without clipping the objective or its gradient.
    return np.exp(-np.logaddexp(0.0, -logits))


def _forward(X, parameters):
    hidden = np.tanh(X @ parameters["W1"] + parameters["b1"])
    logits = (hidden @ parameters["W2"] + parameters["b2"]).ravel()
    if not np.isfinite(logits).all():
        raise ValueError("non-finite logits; reduce input or parameter magnitudes")
    return hidden, logits


def predict_proba(X, parameters):
    """Return an (n_samples,) class-1 score vector; calibration is untested."""
    X = _features(X)
    parameters = _parameters(parameters, X.shape[1])
    _, logits = _forward(X, parameters)
    return _sigmoid(logits)


def loss_and_gradients(X, y, parameters):
    """Mean BCE from logits and analytical gradients, without regularization."""
    X = _features(X)
    y = _targets(y, len(X), (0, 1))
    parameters = _parameters(parameters, X.shape[1])
    hidden, logits = _forward(X, parameters)
    # Label-dependent sign avoids subtracting two large nearly equal values.
    loss = float(np.logaddexp(0.0, (1 - 2 * y) * logits).mean())
    delta2 = ((_sigmoid(logits) - y) / len(X))[:, None]
    delta1 = (delta2 @ parameters["W2"].T) * (1 - hidden**2)
    gradients = {
        "W1": X.T @ delta1,
        "b1": delta1.sum(axis=0),
        "W2": hidden.T @ delta2,
        "b2": delta2.sum(axis=0),
    }
    if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients.values()):
        raise ValueError("non-finite loss or gradients; rescale inputs or parameters")
    return loss, gradients


def train_mlp(X, y, *, n_hidden=8, epochs=10000, learning_rate=0.5, seed=42):
    """Full-batch gradient descent; history includes initial and final loss."""
    X = _features(X)
    y = _targets(y, len(X), (0, 1))
    _positive_integer(epochs, "epochs")
    _positive_rate(learning_rate)
    parameters = initialize_parameters(X.shape[1], n_hidden, seed=seed)
    history = []
    for _ in range(epochs):
        loss, gradients = loss_and_gradients(X, y, parameters)
        history.append(loss)
        parameters = {key: value - learning_rate * gradients[key] for key, value in parameters.items()}
    history.append(loss_and_gradients(X, y, parameters)[0])
    return parameters, np.asarray(history)


def run_demo():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    xor = np.array([0, 1, 1, 0])
    records = {}
    for name, y in {"AND": np.array([-1, -1, -1, 1]), "XOR": 2 * xor - 1}.items():
        weights, bias, mistakes = fit_perceptron(X, y)
        predictions = np.where(X @ weights + bias > 0, 1, -1)
        records[name] = {
            "epochs": len(mistakes), "zero_mistake_epoch": mistakes[-1] == 0,
            "training_accuracy": float(np.mean(predictions == y)),
            "updates": sum(mistakes), "weights": weights.tolist(), "bias": bias,
        }
    parameters, history = train_mlp(X, xor)
    scores = predict_proba(X, parameters)
    return {
        "data": "synthetic four-corner truth tables; all points used for training",
        "config": {"seed": 42, "hidden": 8, "epochs": 10000, "learning_rate": 0.5,
                   "activation": "tanh", "perceptron_epoch_cap": 100},
        "perceptron": records,
        "constructive_xor": constructive_xor(X).tolist(),
        "mlp": {"initial_loss": float(history[0]), "final_loss": float(history[-1]),
                "probabilities": scores.tolist(), "predictions": (scores >= 0.5).astype(int).tolist(),
                "training_accuracy": float(np.mean((scores >= 0.5) == xor)),
                "parameters": sum(value.size for value in parameters.values())},
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
