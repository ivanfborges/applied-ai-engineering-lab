"""Educational classification tree with exhaustive Gini split selection."""

from dataclasses import dataclass

import numpy as np


def _counts(y):
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("y must be a nonempty one-dimensional array")
    if np.issubdtype(labels.dtype, np.number) and not np.all(np.isfinite(labels)):
        raise ValueError("y must contain only finite labels")
    return np.unique(labels, return_counts=True)[1]


def gini(y) -> float:
    counts = _counts(y)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p**2))


def entropy(y) -> float:
    counts = _counts(y)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("X must be a nonempty two-dimensional feature matrix")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must have one label per row")
    _counts(y)
    return X, y


def best_gini_split(X, y, min_samples_leaf=1):
    """Return (feature, threshold, gain), or None when no positive gain exists.

    Ties keep the first feature and lowest threshold. Midpoints between
    distinct observed values keep both children nonempty.
    """
    X, y = _validate_xy(X, y)
    min_samples_leaf = _positive_int(min_samples_leaf, "min_samples_leaf")
    parent = gini(y)
    best = None
    best_gain = 0.0
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        for lower, upper in zip(values[:-1], values[1:]):
            threshold = lower + (upper - lower) / 2.0
            left = X[:, feature] <= threshold
            n_left = int(left.sum())
            n_right = len(y) - n_left
            if min(n_left, n_right) < min_samples_leaf:
                continue
            weighted = (n_left * gini(y[left]) + n_right * gini(y[~left])) / len(y)
            gain = parent - weighted
            if gain > best_gain + 1e-12:
                best_gain = gain
                best = (feature, float(threshold), float(gain))
    return best


@dataclass
class Node:
    prediction: object
    feature: int | None = None
    threshold: float | None = None
    gain: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None


class GiniTreeClassifier:
    """Small greedy tree for finite numeric features and class labels."""

    def __init__(self, max_depth=3, min_samples_leaf=1):
        self.max_depth = _positive_int(max_depth, "max_depth")
        self.min_samples_leaf = _positive_int(min_samples_leaf, "min_samples_leaf")

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.root_ = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        classes, counts = np.unique(y, return_counts=True)
        node = Node(prediction=classes[np.argmax(counts)])
        if depth >= self.max_depth or len(classes) == 1 or len(y) < 2 * self.min_samples_leaf:
            return node
        split = best_gini_split(X, y, self.min_samples_leaf)
        if split is None:
            return node
        node.feature, node.threshold, node.gain = split
        left = X[:, node.feature] <= node.threshold
        node.left = self._build(X[left], y[left], depth + 1)
        node.right = self._build(X[~left], y[~left], depth + 1)
        return node

    def predict(self, X):
        if not hasattr(self, "root_"):
            raise ValueError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X must be two-dimensional with the fitted feature count")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        predictions = []
        for row in X:
            node = self.root_
            while node.feature is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right
            predictions.append(node.prediction)
        return np.asarray(predictions, dtype=self.classes_.dtype)


if __name__ == "__main__":
    X = np.array([[0.0, 0.0], [0.2, 1.0], [0.8, 0.0], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    model = GiniTreeClassifier(max_depth=2).fit(X, y)
    print("Synthetic four-row split:", best_gini_split(X, y))
    print("Predictions:", model.predict([[0.1, 0.4], [0.9, 0.4]]).tolist())
