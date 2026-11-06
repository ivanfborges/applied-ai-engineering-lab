"""Educational MAD baseline and isolation forest; not library replacements."""

from dataclasses import dataclass

import numpy as np


def feature_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite 2D feature matrix")
    return X


def score_vector(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be a nonempty finite vector")
    return scores


def calibrate_threshold(normal_scores, alpha=0.02):
    """Empirical normal-reference quantile; alerts use strict score > threshold."""
    scores = score_vector(normal_scores)
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    return float(np.quantile(scores, 1 - alpha, method="higher"))


class RobustMAD:
    """Maximum absolute featurewise deviation, scaled by Gaussian-consistent MAD."""

    def fit(self, X):
        X = feature_matrix(X)
        center = np.median(X, axis=0)
        scale = 1.482602218505602 * np.median(np.abs(X - center), axis=0)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("each feature needs positive finite MAD; review constant/discrete features")
        self.center_, self.scale_ = center, scale
        return self

    def anomaly_score(self, X):
        if not hasattr(self, "center_"):
            raise ValueError("fit the MAD baseline before scoring")
        X = feature_matrix(X)
        if X.shape[1] != self.center_.size:
            raise ValueError("feature count differs from fitted data")
        return np.max(np.abs((X - self.center_) / self.scale_), axis=1)


def average_path_length(n):
    """Exact harmonic-number c(n), including c(0)=c(1)=0 and c(2)=1."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if n <= 1:
        return 0.0
    return float(2 * np.sum(1 / np.arange(1, n)) - 2 * (n - 1) / n)


@dataclass(frozen=True)
class IsolationNode:
    size: int
    feature: int | None = None
    threshold: float | None = None
    left: "IsolationNode | None" = None
    right: "IsolationNode | None" = None


def _build_tree(X, depth, max_depth, rng):
    if len(X) <= 1 or depth >= max_depth:
        return IsolationNode(len(X))
    low, high = X.min(axis=0), X.max(axis=0)
    candidates = np.flatnonzero(high > low)
    if not candidates.size:
        return IsolationNode(len(X))
    feature = int(rng.choice(candidates))
    threshold = float(rng.uniform(low[feature], high[feature]))
    # The <= split and clipping keep both children nonempty even at float endpoints.
    threshold = min(threshold, float(np.nextafter(high[feature], low[feature])))
    left = X[:, feature] <= threshold
    return IsolationNode(
        len(X), feature, threshold,
        _build_tree(X[left], depth + 1, max_depth, rng),
        _build_tree(X[~left], depth + 1, max_depth, rng),
    )


def path_length(x, node, depth=0):
    """Add c(leaf size) when duplicates or the depth limit stop isolation."""
    if node.feature is None:
        return depth + average_path_length(node.size)
    child = node.left if x[node.feature] <= node.threshold else node.right
    return path_length(x, child, depth + 1)


class EducationalIsolationForest:
    """Small dense NumPy forest with exact leaf correction and local RNG state."""

    def __init__(self, n_estimators=100, max_samples=128, random_state=41):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X):
        X = feature_matrix(X)
        for name, value, minimum in (
            ("n_estimators", self.n_estimators, 1), ("max_samples", self.max_samples, 2),
            ("random_state", self.random_state, 0),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if len(X) < 2:
            raise ValueError("at least two training rows are required")
        with np.errstate(over="ignore"):
            if not np.isfinite(np.ptp(X, axis=0)).all():
                raise ValueError("feature ranges must be finite; rescale extreme inputs")
        sample_size = min(self.max_samples, len(X))
        max_depth = int(np.ceil(np.log2(sample_size)))
        rng = np.random.default_rng(self.random_state)
        trees = [
            _build_tree(X[rng.choice(len(X), sample_size, replace=False)], 0, max_depth, rng)
            for _ in range(self.n_estimators)
        ]
        self.trees_, self.sample_size_, self.n_features_in_ = trees, sample_size, X.shape[1]
        return self

    def anomaly_score(self, X):
        if not hasattr(self, "trees_"):
            raise ValueError("fit the forest before scoring")
        X = feature_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("feature count differs from fitted data")
        paths = np.array([[path_length(x, tree) for tree in self.trees_] for x in X])
        return 2.0 ** (-paths.mean(axis=1) / average_path_length(self.sample_size_))


if __name__ == "__main__":
    forest = EducationalIsolationForest(n_estimators=3).fit(np.ones((8, 2)))
    print("Duplicate-only reference: no split is possible; corrected scores =",
          forest.anomaly_score([[1, 1], [100, 100]]))
    print("This invariant illustrates why an unsplittable baseline cannot identify novelty.")
