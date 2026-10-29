"""Small synthetic controls and exact Lloyd states for the Day 36 visual lab."""

from __future__ import annotations

from numbers import Integral
from typing import NamedTuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, kmeans_plusplus
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from from_scratch import _matrix, squared_distances


class LloydState(NamedTuple):
    iteration: int
    phase: str
    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    converged: bool


def lloyd_trace(X, initial, *, max_iter=80):
    """Store actual assignment/update states; no interpolated optimization steps.

    Update states retain the preceding assignments and measure their objective
    at the new means. Assignment states use the current nearest centers.
    Ties choose the lowest ID; empty centers retain their previous coordinates.
    """
    X = _matrix(X, "X")
    centers = _matrix(initial, "initial").copy()
    if len(centers) > len(X):
        raise ValueError("The number of centers cannot exceed n_samples")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    distances = squared_distances(X, centers)
    labels = distances.argmin(axis=1)

    def state(iteration, phase, stable=False):
        inertia = float(np.sum((X - centers[labels]) ** 2))
        if not np.isfinite(inertia):
            raise ValueError("Inertia overflowed; rescale input features")
        return LloydState(iteration, phase, centers.copy(), labels.copy(), inertia, stable)

    states = [state(0, "initial"), state(0, "assignment")]
    for iteration in range(1, max_iter + 1):
        updated = centers.copy()
        for cluster in range(len(centers)):
            members = X[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        centers = updated
        states.append(state(iteration, "update"))
        new_labels = squared_distances(X, centers).argmin(axis=1)
        stable = np.array_equal(new_labels, labels)
        labels = new_labels
        states.append(state(iteration, "assignment", stable))
        if stable:
            break
    return states


def fit_reference(X, k, *, seed=42, init="k-means++", n_init=10):
    model = KMeans(n_clusters=k, init=init, n_init=n_init, random_state=seed,
                   max_iter=300, tol=0, algorithm="lloyd").fit(X)
    np.testing.assert_allclose(model.inertia_, np.sum((X - model.cluster_centers_[model.labels_]) ** 2),
                               rtol=1e-10, atol=1e-8)
    return model


def valid_silhouette(X, labels):
    occupied = len(np.unique(labels))
    return float(silhouette_score(X, labels)) if 2 <= occupied < len(X) else None


def convergence_data(seed):
    X, _ = make_blobs(n_samples=240, centers=[(-4, -1), (0, 4), (4, -1)],
                      cluster_std=0.7, random_state=seed)
    initial = np.array([[-4., -3.], [-2., -3.], [1., -3.]])
    return X, initial


def scaling_experiment(seed):
    latent, truth = make_blobs(n_samples=360, centers=[(-4, -1), (0, 4), (4, -1)],
                               cluster_std=[0.65, 0.85, 0.65], random_state=seed)
    X = np.column_stack(((latent[:, 1] + 3) * 60, latent[:, 0] + 6))
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    raw_model = fit_reference(X, 3, seed=seed)
    scaled_model = fit_reference(scaled, 3, seed=seed)
    delta = np.array([60., 1.])
    contributions = np.vstack((delta ** 2, (delta / scaler.scale_) ** 2))
    shares = contributions / contributions.sum(axis=1, keepdims=True)
    return X, truth, scaler, raw_model, scaled_model, shares


def selection_experiment(seed, k_min=2, k_max=10):
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) for k in (k_min, k_max)):
        raise ValueError("K range must contain integers")
    if not 2 <= k_min <= k_max < 360:
        raise ValueError("Require 2 <= k_min <= k_max < 360")
    X, _ = make_blobs(n_samples=360, centers=[(-4, -1), (0, 4), (4, -1)],
                      cluster_std=0.85, random_state=seed)
    rows = []
    for k in range(k_min, k_max + 1):
        model = fit_reference(X, k, seed=seed)
        rows.append({"k": k, "inertia": float(model.inertia_),
                     "silhouette": valid_silhouette(X, model.labels_),
                     "sizes": np.bincount(model.labels_, minlength=k).tolist()})
    return X, rows


def limitations_experiment(seed):
    spherical, _ = make_blobs(n_samples=280, centers=[(-3, -2), (0, 3), (3, -2)],
                              cluster_std=0.5, random_state=seed)
    elongated = spherical @ np.array([[1., 0.7], [-0.4, 0.25]])
    moons, _ = make_moons(n_samples=280, noise=0.06, random_state=seed)
    varied, _ = make_blobs(n_samples=[100, 100, 80], centers=[(-3, -2), (0, 3), (3, -2)],
                          cluster_std=[0.16, 0.65, 1.3], random_state=seed)
    outliers = np.random.default_rng(seed).normal(loc=[14, 13], scale=1.3, size=(12, 2))
    cases = [
        ("Compact groups", spherical, 3, 0.55, "Favorable Euclidean geometry"),
        ("Elongated groups", elongated, 3, 0.38, "Means do not model covariance"),
        ("Two moons", moons, 2, 0.22, "Nonconvex groups cross Voronoi cells"),
        ("Unequal densities", varied, 3, 0.55, "One density radius can also fail"),
        ("Distant observations", np.vstack((spherical, outliers)), 3, 0.65, "Squared loss can spend a center on extremes"),
    ]
    return [{"name": name, "X": X, "k": k, "eps": eps, "min_samples": 6,
             "note": note, "kmeans": fit_reference(X, k, seed=seed),
             "dbscan": DBSCAN(eps=eps, min_samples=6).fit(X)}
            for name, X, k, eps, note in cases]


def three_dimensional_data(seed):
    X, _ = make_blobs(n_samples=240, centers=[(2, 3, 1), (7, 10, 4), (11, 4, 8)],
                      cluster_std=[0.6, 0.9, 0.7], random_state=seed)
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    initial = scaled[np.random.default_rng(seed).choice(len(X), 3, replace=False)].copy()
    return X, scaler, scaled, initial


def initialization_experiment(seed):
    X, _ = make_blobs(n_samples=400, centers=[(-4, -2), (-4, 2), (4, -2), (4, 2)],
                      cluster_std=0.55, random_state=seed)
    runs = []
    for strategy in ("random", "k-means++"):
        for run_seed in (seed, seed + 1, seed + 2):
            if strategy == "random":
                initial = X[np.random.default_rng(run_seed).choice(len(X), 3, replace=False)].copy()
            else:
                initial, _ = kmeans_plusplus(X, n_clusters=3, random_state=run_seed)
            final = fit_reference(X, 3, seed=run_seed, init=initial.copy(), n_init=1)
            runs.append({"strategy": strategy, "seed": run_seed, "initial": initial, "model": final})
    budgets = {strategy: float(fit_reference(X, 3, seed=seed, init=strategy, n_init=10).inertia_)
               for strategy in ("random", "k-means++")}
    return X, runs, budgets
