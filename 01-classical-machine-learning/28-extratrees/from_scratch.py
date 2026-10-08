"""Educational binary ExtraTrees: recursive random splits, no bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _integer(name: str, value: int, minimum: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _features(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty, finite 2D numeric array.")
    return X


def _labels(y, n_samples: int) -> np.ndarray:
    y = np.asarray(y)
    if y.shape != (n_samples,) or not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain one binary label (0 or 1) per row.")
    return y.astype(int)


def gini(y) -> float:
    """Binary Gini impurity; empty nodes are invalid."""
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("y must be a nonempty 1D binary array.")
    y = _labels(y, len(y))
    p = float(y.mean())
    return 2.0 * p * (1.0 - p)


@dataclass
class Node:
    probability: float
    n_samples: int
    feature: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None


class ExtraTreesBinary:
    """Small finite-data learner for labels 0/1, with full-data trees.

    Exactly one random threshold per selected nonconstant feature is tried.
    Invalid proposals are skipped; features are not resampled to force a split.
    This is an educational subset of the algorithm, not sklearn parity.
    """

    def __init__(
        self,
        n_estimators: int = 30,
        max_depth: int = 6,
        min_samples_leaf: int = 1,
        max_features: int | None = None,
        random_state: int = 42,
    ):
        _integer("n_estimators", n_estimators, 1)
        _integer("max_depth", max_depth, 0)
        _integer("min_samples_leaf", min_samples_leaf, 1)
        _integer("random_state", random_state, 0)
        if max_features is not None:
            _integer("max_features", max_features, 1)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y) -> ExtraTreesBinary:
        X = _features(X)
        y = _labels(y, len(X))
        n_features = X.shape[1]
        k = self.max_features if self.max_features is not None else max(1, int(np.sqrt(n_features)))
        if k > n_features:
            raise ValueError("max_features cannot exceed the number of input features.")
        # Reset the RNG on every fit; repeated fits with the same seed agree.
        rng = np.random.default_rng(self.random_state)
        roots = [self._grow(X, y, 0, k, rng) for _ in range(self.n_estimators)]
        self.n_features_in_ = n_features
        self.roots_ = roots
        return self

    def _grow(self, X, y, depth, k, rng) -> Node:
        node = Node(float(y.mean()), len(y))
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or np.all(y == y[0]):
            return node

        best_score = np.inf
        best_split = None
        for feature in rng.choice(X.shape[1], size=k, replace=False):
            values = X[:, feature]
            low, high = float(values.min()), float(values.max())
            if low == high:
                continue
            # Uniform draw via a convex combination avoids overflow in high - low.
            u = float(rng.random())
            threshold = (1.0 - u) * low + u * high
            left = values <= threshold
            n_left = int(left.sum())
            if min(n_left, len(y) - n_left) < self.min_samples_leaf:
                continue
            score = (n_left * gini(y[left]) + (len(y) - n_left) * gini(y[~left])) / len(y)
            if score < best_score:
                best_score = score
                best_split = (int(feature), threshold, left)

        if best_split is not None:
            # Zero immediate gain is allowed: a later split can uncover an interaction.
            node.feature, node.threshold, left = best_split
            node.left = self._grow(X[left], y[left], depth + 1, k, rng)
            node.right = self._grow(X[~left], y[~left], depth + 1, k, rng)
        return node

    def tree_probabilities(self, X) -> np.ndarray:
        """Return P(y=1) with shape (n_trees, n_rows)."""
        if not hasattr(self, "roots_"):
            raise RuntimeError("Fit the model before prediction.")
        X = _features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number of features as training data.")
        probabilities = np.empty((len(self.roots_), len(X)))
        for i, root in enumerate(self.roots_):
            for j, row in enumerate(X):
                node = root
                while node.feature is not None:
                    node = node.left if row[node.feature] <= node.threshold else node.right
                probabilities[i, j] = node.probability
        return probabilities

    def predict_proba(self, X) -> np.ndarray:
        """Columns correspond to classes [0, 1]."""
        positive = self.tree_probabilities(X).mean(axis=0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X) -> np.ndarray:
        # argmax resolves an exact probability tie in favor of class 0.
        return self.predict_proba(X).argmax(axis=1)


def main() -> None:
    # Code-defined synthetic truth table: XOR needs two recursive splits.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    model = ExtraTreesBinary(n_estimators=5, max_depth=2, max_features=2).fit(X, y)
    print("Synthetic XOR truth table; training-only mechanics demonstration.")
    print("Root thresholds:", [round(root.threshold, 4) for root in model.roots_])
    print("P(y=1):", model.predict_proba(X)[:, 1].tolist())
    print("Predictions:", model.predict(X).tolist())


if __name__ == "__main__":
    main()
