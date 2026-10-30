"""Numerical neighborhood audit of PCA, t-SNE and UMAP; no visual exports."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import platform

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits


def neighbor_indices(X, k):
    """Euclidean kNN excluding self; equal distances use input-row order."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a finite, nonempty-feature matrix")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    if not 1 <= k < len(X):
        raise ValueError("require 1 <= k < number of rows")
    distances = pairwise_distances(X, metric="euclidean", n_jobs=1)
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1, kind="stable")[:, :k]


def neighborhood_overlap(A, B):
    """Mean fraction of shared neighbors for two equal-k index matrices."""
    A, B = np.asarray(A), np.asarray(B)
    if A.ndim != 2 or A.shape != B.shape or min(A.shape) == 0:
        raise ValueError("neighbor matrices must have the same nonempty shape")
    n, k = A.shape
    for indices in (A, B):
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("neighbor indices must be integers")
        if np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("neighbor index outside the row range")
        if np.any(indices == np.arange(n)[:, None]):
            raise ValueError("self must not appear in neighbor lists")
        if any(len(set(row)) != k for row in indices):
            raise ValueError("each neighbor list must contain unique indices")
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(A, B)]))


def make_methods():
    try:
        from umap import UMAP
    except ModuleNotFoundError as exc:
        if exc.name != "umap":
            raise
        raise ImportError('Install the topic extra: python -m pip install -e ".[dev,manifold]"') from exc

    # scikit-learn 1.4 is still supported by the repository dependency contract.
    iteration_key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    methods = {"pca": PCA(n_components=2, svd_solver="full")}
    for perplexity, seed in ((30, 42), (50, 42), (30, 21)):
        methods[f"tsne_p{perplexity}_s{seed}"] = TSNE(
            n_components=2, perplexity=perplexity, init="pca",
            learning_rate="auto", early_exaggeration=12.0, metric="euclidean",
            method="barnes_hut", angle=0.5, random_state=seed, n_jobs=1,
            **{iteration_key: 1000},
        )
    for min_dist, seed in ((0.1, 42), (0.8, 42), (0.1, 21)):
        methods[f"umap_md{min_dist}_s{seed}"] = UMAP(
            n_components=2, n_neighbors=20, min_dist=min_dist, spread=1.0,
            metric="euclidean", init="spectral", n_epochs=500,
            random_state=seed, transform_seed=seed, n_jobs=1,
        )
    return methods


def run_comparison(n_samples=800, sample_seed=42):
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError("n_samples must be an integer")
    # k=30 trustworthiness requires strictly more than 60 rows.
    if not 61 <= n_samples <= 1797:
        raise ValueError("n_samples must be between 61 and 1797")
    digits = load_digits()
    indices = np.random.default_rng(sample_seed).choice(len(digits.data), n_samples, replace=False)
    X = digits.data[indices].astype(float) / 16.0
    methods = make_methods()
    original_neighbors = {k: neighbor_indices(X, k) for k in (10, 30)}
    rows, embedded_neighbors = {}, {}
    with threadpool_limits(limits=1):
        for name, model in methods.items():
            # Do not pass digit labels: UMAP also supports supervised fitting.
            Y = model.fit_transform(X)
            if Y.shape != (n_samples, 2) or not np.isfinite(Y).all():
                raise RuntimeError(f"{name} returned an invalid embedding")
            embedded_neighbors[name] = {}
            scores = {}
            for k in (10, 30):
                neighbors = neighbor_indices(Y, k)
                embedded_neighbors[name][k] = neighbors
                scores[f"trustworthiness@{k}"] = float(trustworthiness(X, Y, n_neighbors=k))
                scores[f"neighbor_recall@{k}"] = neighborhood_overlap(original_neighbors[k], neighbors)
            rows[name] = {"parameters": model.get_params(deep=False), "scores": scores}
            if name == "pca":
                rows[name]["explained_variance_ratio_sum"] = float(model.explained_variance_ratio_.sum())
    stability = {}
    for baseline, variant in (
        ("tsne_p30_s42", "tsne_p30_s21"),
        ("umap_md0.1_s42", "umap_md0.1_s21"),
    ):
        stability[f"{baseline}_vs_{variant}"] = {
            f"neighbor_overlap@{k}": neighborhood_overlap(
                embedded_neighbors[baseline][k], embedded_neighbors[variant][k]
            ) for k in (10, 30)
        }
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "scikit-learn", "umap-learn", "numba", "pynndescent", "threadpoolctl"
        )},
        "data": {"source": "sklearn.datasets.load_digits", "n_samples": n_samples,
                 "n_features": 64, "sample_seed": sample_seed,
                 "sample_index_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                 "preprocessing": "pixel counts divided by 16; no fitted scaling or pre-PCA",
                 "metric": "euclidean", "labels_used": False},
        "methods": rows, "seed_stability": stability,
        "interpretation_status": "author review required; exploratory in-sample diagnostics",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_comparison(args.samples, args.sample_seed), indent=2, allow_nan=False))
