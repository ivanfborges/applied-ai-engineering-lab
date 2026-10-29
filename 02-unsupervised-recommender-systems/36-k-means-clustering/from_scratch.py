"""Educational, dense NumPy implementation of Lloyd's assignment/update loop."""

from __future__ import annotations

from numbers import Integral, Real
from typing import NamedTuple

import numpy as np


class KMeansResult(NamedTuple):
    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    inertia_history: np.ndarray
    n_iter: int
    converged: bool


def _matrix(values, name):
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric matrix") from error
    if matrix.ndim != 2 or 0 in matrix.shape or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a nonempty, finite 2D matrix")
    return matrix


def squared_distances(X, centers):
    """Return an (n_samples, n_centers) matrix of squared Euclidean distances."""
    X = _matrix(X, "X")
    centers = _matrix(centers, "centers")
    if X.shape[1] != centers.shape[1]:
        raise ValueError("X and centers must have the same number of features")
    try:
        with np.errstate(over="raise", invalid="raise"):
            return np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    except FloatingPointError as error:
        raise ValueError("Distances overflowed; rescale the input features") from error


def predict_labels(X, centers):
    """Assign to nearest centers; exact ties choose the lowest center index."""
    return squared_distances(X, centers).argmin(axis=1)


def kmeans_numpy(X, k, *, init=None, max_iter=100, tol=1e-8, seed=42):
    """Fit one Lloyd run; retain previous centers for empty clusters.

    History includes nearest-center inertia before updates and after each update.
    On every exit, labels are nearest-center assignments to the returned centers.
    A tolerance or iteration stop can leave centers unequal to final-label means.
    This dense teaching implementation allocates an n-by-k-by-d temporary.
    """
    X = _matrix(X, "X")
    for name, value in (("k", k), ("max_iter", max_iter)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if k > len(X):
        raise ValueError("k must not exceed n_samples")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, Real) or not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")

    if init is None:
        rng = np.random.default_rng(seed)
        centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    else:
        centers = _matrix(init, "init").copy()
        if centers.shape != (k, X.shape[1]):
            raise ValueError("init must have shape (k, n_features)")

    distances = squared_distances(X, centers)
    labels = distances.argmin(axis=1)
    history = [float(distances[np.arange(len(X)), labels].sum())]
    converged = False
    for iteration in range(1, max_iter + 1):
        updated = centers.copy()
        for cluster in range(k):
            members = X[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        movement = np.linalg.norm(updated - centers)
        distances = squared_distances(X, updated)
        new_labels = distances.argmin(axis=1)
        history.append(float(distances[np.arange(len(X)), new_labels].sum()))
        converged = np.array_equal(new_labels, labels) or movement <= tol
        centers, labels = updated, new_labels
        if converged:
            break

    return KMeansResult(centers, labels, history[-1], np.asarray(history), iteration, converged)


def main():
    X = np.array([[0, 0], [0, 2], [2, 0], [10, 10], [10, 12], [12, 10]])
    result = kmeans_numpy(X, 2, init=[[0, 0], [10, 10]], tol=0)
    print("Synthetic six-point centroid check; fixed initial centers")
    print("Centers:\n", result.centers)
    print("Labels:", result.labels.tolist())
    print("Inertia history:", result.inertia_history.tolist())
    print(f"Iterations: {result.n_iter}; converged: {result.converged}")


if __name__ == "__main__":
    main()
