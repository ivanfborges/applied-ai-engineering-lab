"""Educational binary resampling and empirical cost threshold selection."""
from __future__ import annotations

import numpy as np


def _binary_labels(y):
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.size == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("y must be a nonempty one-dimensional array of 0/1 labels")
    return labels.astype(int)


def _features(X):
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all():
        raise ValueError("X must be a nonempty finite numerical matrix")
    return values


def _positive_costs(false_positive_cost, false_negative_cost):
    costs = np.asarray([false_positive_cost, false_negative_cost], dtype=float)
    if costs.shape != (2,) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("both error costs must be finite positive scalars")
    return costs


def simple_smote(X_minority, n_samples, k_neighbors=5, seed=42):
    """Interpolate numerical minority rows; return only new observations.

    Uses dense pairwise distances, with stable index ordering for ties.
    This small-data implementation is not an imbalanced-learn replacement.
    """
    X = _features(X_minority)
    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, (int, np.integer))
        or n_samples < 0
    ):
        raise ValueError("n_samples must be a nonnegative integer")
    if (
        isinstance(k_neighbors, (bool, np.bool_))
        or not isinstance(k_neighbors, (int, np.integer))
        or not 1 <= k_neighbors < len(X)
    ):
        raise ValueError("k_neighbors must be an integer in [1, minority_count - 1]")
    if n_samples == 0:
        return np.empty((0, X.shape[1]), dtype=float)
    distances = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    # Exclude rows by identity: duplicates can tie at zero distance.
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :k_neighbors]
    rng = np.random.default_rng(seed)
    anchors = rng.integers(len(X), size=n_samples)
    partners = neighbors[anchors, rng.integers(k_neighbors, size=n_samples)]
    fraction = rng.random((n_samples, 1))
    return X[anchors] + fraction * (X[partners] - X[anchors])


def resample_training(X, y, method, seed=42, k_neighbors=5):
    """Balance two classes on supplied training rows; never mutate inputs."""
    values, labels = _features(X), _binary_labels(y)
    if len(values) != len(labels):
        raise ValueError("X and y must have matching row counts")
    if method not in {"random_over", "random_under", "smote"}:
        raise ValueError("method must be random_over, random_under, or smote")
    counts = np.bincount(labels, minlength=2)
    if (counts == 0).any():
        raise ValueError("resampling requires both classes")
    if counts[0] == counts[1]:
        return values.copy(), labels.copy()
    minority = int(np.argmin(counts))
    minor_indices = np.flatnonzero(labels == minority)
    major_indices = np.flatnonzero(labels != minority)
    rng = np.random.default_rng(seed)
    if method == "random_under":
        chosen = rng.choice(major_indices, size=len(minor_indices), replace=False)
        indices = np.concatenate([minor_indices, chosen])
        return values[indices].copy(), labels[indices].copy()
    extra_count = len(major_indices) - len(minor_indices)
    if method == "random_over":
        extra = values[rng.choice(minor_indices, size=extra_count, replace=True)]
    else:
        extra = simple_smote(values[minor_indices], extra_count, k_neighbors, seed)
    return (
        np.vstack([values, extra]),
        np.concatenate([labels, np.full(extra_count, minority, dtype=int)]),
    )


def select_cost_threshold(
    y_validation, probabilities, false_positive_cost=1.0, false_negative_cost=10.0
):
    """Minimize validation FP*C_FP + FN*C_FN under score >= threshold.

    Evaluate every distinct decision set, including no alerts. Ties choose
    the highest threshold (fewest alerts). This is empirical selection,
    not a calibrated Bayes threshold or unbiased validation estimate.
    """
    y = _binary_labels(y_validation)
    p = np.asarray(probabilities, dtype=float)
    if (
        p.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("probabilities must match y and contain finite values in [0, 1]")
    c_fp, c_fn = _positive_costs(false_positive_cost, false_negative_cost)
    candidates = np.append(np.unique(p), np.nextafter(p.max(), np.inf))
    best = None
    for threshold in candidates:
        predicted = p >= threshold
        fp = int(np.count_nonzero(predicted & (y == 0)))
        fn = int(np.count_nonzero(~predicted & (y == 1)))
        cost = float(c_fp * fp + c_fn * fn)
        # Ascending candidates and <= implement the stated tie policy.
        if best is None or cost <= best["validation_cost"]:
            best = {
                "threshold": float(threshold), "validation_cost": cost,
                "fp": fp, "fn": fn,
            }
    return best


def main():
    minority = np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
    print("Synthetic numerical interpolation demo:")
    print(simple_smote(minority, n_samples=4, k_neighbors=1))
    print("Toy validation cost policy:")
    print(select_cost_threshold([0, 1, 0, 1], [0.1, 0.3, 0.4, 0.9]))


if __name__ == "__main__":
    main()
