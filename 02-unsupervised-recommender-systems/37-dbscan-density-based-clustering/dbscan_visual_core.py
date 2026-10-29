"""Small deterministic datasets and numerical results, independent of rendering."""

from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array

from from_scratch import dbscan_trace
from example import make_data, kth_neighbor_distances


EPS_VALUES = (0.08, 0.12, 0.18, 0.25, 0.35, 0.50)
MIN_SAMPLES_VALUES = (3, 5, 7, 10, 15)


def neighborhood_data(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cloud = rng.uniform([-1.4, -1.15], [1.4, 1.15], size=(30, 2))
    return np.vstack([[0., 0.], cloud])


def expansion_data() -> np.ndarray:
    """Two short curves; first point is initially noise and later becomes border."""
    curve = np.array([[-2, 0], [-1.75, 0.1], [-1.5, 0.3], [-1.3, 0.55],
                      [-1.15, 0.8], [-1, 1.05], [-1, 1.3], [-1.1, 1.55],
                      [-1.3, 1.75], [-1.55, 1.9], [-1.8, 2]])
    return np.vstack([[[-2.5, -0.1]], curve, curve + [4.4, 0],
                      [[0, -1.2], [0, 2.7], [-3.3, 2.7], [3.5, -1.0]]])


def fit_density(X: np.ndarray, eps: float, min_samples: int) -> dict:
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(X)
    core = np.zeros(len(X), dtype=bool)
    core[model.core_sample_indices_] = True
    return {"labels": model.labels_, "core": core, **summarize(model.labels_, core),
            "eps": eps, "min_samples": min_samples}


def summarize(labels: np.ndarray, core: np.ndarray) -> dict:
    return {"clusters": int(len(np.unique(labels[labels >= 0]))),
            "core_count": int(core.sum()),
            "border_count": int(((labels >= 0) & ~core).sum()),
            "noise_count": int((labels == -1).sum()),
            "noise_fraction": float(np.mean(labels == -1))}


def expansion_experiment() -> dict:
    X = expansion_data()
    labels, core, steps = dbscan_trace(X, eps=0.64, min_samples=4)
    return {"X": X, "labels": labels, "core": core, "steps": steps,
            "eps": 0.64, "min_samples": 4, **summarize(labels, core)}


def classification_names(labels: np.ndarray, core: np.ndarray) -> np.ndarray:
    return np.where(core, "core", np.where(labels >= 0, "border", "noise"))


def parameter_grid() -> dict:
    X, _ = make_data()
    results = [[fit_density(X, eps, m) for eps in EPS_VALUES] for m in MIN_SAMPLES_VALUES]
    return {"X": X, "results": results,
            "cluster_counts": np.array([[r["clusters"] for r in row] for row in results]),
            "noise_fractions": np.array([[r["noise_fraction"] for r in row] for row in results])}


def scaling_experiment() -> dict:
    X, _ = make_blobs(n_samples=300, centers=[[-2, -1], [2, 1]], cluster_std=0.27, random_state=42)
    distorted = X * [30, 1]
    scaler = StandardScaler().fit(distorted)
    standardized = scaler.transform(distorted)
    spaces = [X, distorted, standardized]
    return {"spaces": spaces, "results": [fit_density(points, 0.3, 7) for points in spaces],
            "scale_factor": 30, "fitted_scales": scaler.scale_.tolist(),
            "fitted_means": scaler.mean_.tolist()}


def three_dimensional_data() -> np.ndarray:
    """Two separated curved tubes with independent 3D jitter and box background."""
    rng = np.random.default_rng(42)
    t = np.linspace(-1.6, 1.6, 140)
    first = np.c_[t, -1.8 + 0.45*np.sin(2*t), -0.9 + 0.4*np.cos(1.5*t)]
    second = np.c_[t, 1.8 + 0.45*np.sin(2*t), 0.9 + 0.4*np.cos(1.5*t)]
    tubes = np.vstack([first, second]) + rng.normal(0, 0.045, size=(280, 3))
    background = rng.uniform([-2.0, -2.7, -1.6], [2.0, 2.7, 1.8], size=(24, 3))
    return np.vstack([tubes, background])


def fixed_radius_density(X: np.ndarray, queries: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """A 2D count/area illustration, not an additional DBSCAN estimator."""
    X, queries = check_array(X), check_array(queries)
    if X.shape[1] != 2 or queries.shape[1] != 2:
        raise ValueError("Density illustration requires two-dimensional coordinates")
    if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, Real) or not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    counts = np.empty(len(queries), dtype=int)
    for start in range(0, len(queries), 128):
        distance = np.linalg.norm(queries[start:start+128, None] - X[None, :], axis=2)
        counts[start:start+128] = (distance <= eps).sum(axis=1)
    return counts, counts / (len(X) * np.pi * eps**2)


def variable_density_experiment() -> dict:
    rng = np.random.default_rng(42)
    dense = rng.normal([-0.65, 0], 0.055, size=(120, 2))
    sparse = rng.normal([0.65, 0], 0.32, size=(120, 2))
    background = rng.uniform([-1.3, -1.2], [1.7, 1.2], size=(15, 2))
    X = np.vstack([dense, sparse, background])
    return {"X": X, "provenance": np.repeat([0, 1, 2], [120, 120, 15]),
            "results": [fit_density(X, eps, 7) for eps in (0.08, 0.18, 0.36)]}


def distance_concentration() -> dict:
    rng = np.random.default_rng(42)
    dimensions = [2, 10, 50, 100]
    relative_distances, cvs, ratios = [], [], []
    for dimension in dimensions:
        X = rng.normal(size=(180, dimension))
        distances = pdist(X, metric="euclidean")
        relative_distances.append(distances / np.sqrt(2*dimension))
        cvs.append(float(distances.std()/distances.mean()))
        matrix = squareform(distances)
        np.fill_diagonal(matrix, np.nan)
        ratios.append(float(np.median(np.nanmin(matrix, axis=1) / np.nanmax(matrix, axis=1))))
    return {"dimensions": dimensions, "n_each": 180, "relative_distances": relative_distances,
            "coefficient_of_variation": cvs, "median_nearest_farthest_ratio": ratios}
