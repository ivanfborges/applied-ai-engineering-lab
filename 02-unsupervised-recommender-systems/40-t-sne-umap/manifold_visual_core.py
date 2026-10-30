"""Small, inspectable numerical mechanisms for the Day 40 visual laboratory."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import numpy as np
from sklearn.datasets import load_digits, make_blobs, make_moons, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_limits

from example import neighbor_indices, neighborhood_overlap
from from_scratch import finite_matrix, loss_and_gradient


SEED = 42


def versions():
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "scikit-learn", "umap-learn", "numba", "pynndescent",
            "matplotlib", "plotly", "pillow", "threadpoolctl"
        )}}


def array_hash(X):
    X = np.ascontiguousarray(X, dtype="<f8")
    return hashlib.sha256(str(X.shape).encode() + X.tobytes()).hexdigest()


def digits_sample(n=400):
    if not isinstance(n, int) or not 101 <= n <= 1797:
        raise ValueError("digits sample must contain 101..1797 observations")
    data = load_digits()
    rows = np.random.default_rng(SEED).choice(len(data.data), n, replace=False)
    return data.data[rows] / 16.0, data.target[rows], rows


def swiss_sample(n=500):
    if not isinstance(n, int) or not 61 <= n <= 2000:
        raise ValueError("Swiss Roll sample must contain 61..2000 observations")
    return make_swiss_roll(n_samples=n, noise=0.05, random_state=SEED)


def probability_sample():
    return make_blobs(n_samples=72, centers=[[-2, 0], [1, 1], [2, -2]],
                      cluster_std=[0.35, 0.7, 0.45], random_state=SEED)


def entropy_probabilities(distance2, beta):
    """A conditional distribution over non-self squared distances, in nats."""
    distance2 = np.asarray(distance2, dtype=float)
    if distance2.ndim != 1 or len(distance2) < 2 or not np.isfinite(distance2).all() or np.any(distance2 < 0):
        raise ValueError("distance2 must be a finite nonnegative vector with >=2 entries")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    logits = -(distance2 - distance2.min()) * beta
    weights = np.exp(logits)
    p = weights / weights.sum()
    positive = p > 0
    entropy = -np.sum(p[positive] * np.log(p[positive]))
    return p, float(entropy)


def match_perplexity(distance2, perplexity, tolerance=1e-7, max_iter=80):
    """Binary search Gaussian precision beta=1/(2 sigma^2); retain real search states."""
    distance2 = np.asarray(distance2, dtype=float)
    if not np.isfinite(perplexity) or not 1 < perplexity < len(distance2):
        raise ValueError("perplexity must lie strictly between 1 and the non-self count")
    if not isinstance(max_iter, int) or max_iter < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid search controls")
    beta, low, high = 1.0, 0.0, np.inf
    states = []
    for _ in range(max_iter):
        p, entropy = entropy_probabilities(distance2, beta)
        states.append({"beta": beta, "sigma": float(np.sqrt(0.5 / beta)),
                       "probabilities": p.copy(), "entropy_bits": entropy / np.log(2),
                       "perplexity": float(np.exp(entropy))})
        error = entropy - np.log(perplexity)
        if abs(error) <= tolerance:
            return states[-1], states
        if error > 0:
            low = beta
            beta = 2 * beta if np.isinf(high) else (beta + high) / 2
        else:
            high = beta
            beta = (beta + low) / 2
    raise ValueError("perplexity search failed; tied distances may make the target unattainable")


def perplexity_affinities(X, perplexity=10):
    X = finite_matrix(X, "X")
    if len(X) > 100:
        raise ValueError("educational affinity computation is capped at 100 observations")
    distances = pairwise_distances(X, squared=True)
    conditional, sigmas, achieved = np.zeros((len(X), len(X))), [], []
    for i in range(len(X)):
        mask = np.arange(len(X)) != i
        solution, _ = match_perplexity(distances[i, mask], perplexity)
        conditional[i, mask] = solution["probabilities"]
        sigmas.append(solution["sigma"])
        achieved.append(solution["perplexity"])
    P = (conditional + conditional.T) / (2 * len(X))
    return P, conditional, np.asarray(sigmas), np.asarray(achieved)


def descent_forces(Y, P):
    """Exact negative-gradient contributions, before any display rescaling."""
    Y = finite_matrix(Y, "Y")
    delta = Y[:, None] - Y[None, :]
    weights = 1 / (1 + np.sum(delta ** 2, axis=2))
    np.fill_diagonal(weights, 0)
    Q = weights / weights.sum()
    attraction = -4 * np.sum((P * weights)[:, :, None] * delta, axis=1)
    repulsion = 4 * np.sum((Q * weights)[:, :, None] * delta, axis=1)
    return attraction, repulsion


def optimization_trace(X, perplexity=10, steps=300, learning_rate=100, stride=10, seed=SEED):
    """Genuine fixed-P descent snapshots, with backtracking and centering."""
    if not isinstance(steps, int) or steps < 1 or not isinstance(stride, int) or stride < 1:
        raise ValueError("steps and stride must be positive integers")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be positive and finite")
    P, conditional, sigmas, achieved = perplexity_affinities(X, perplexity)
    Y = np.random.default_rng(seed).normal(0, 0.01, (len(P), 2))
    Y -= Y.mean(axis=0)
    positions, losses, iterations, accepted_rates = [], [], [], []
    full_losses = []
    for step in range(steps + 1):
        loss, gradient = loss_and_gradient(Y, P)
        full_losses.append(loss)
        if step % stride == 0 or step == steps:
            positions.append(Y.copy())
            losses.append(loss)
            iterations.append(step)
        if step == steps:
            break
        rate = float(learning_rate)
        for _ in range(40):
            candidate = Y - rate * gradient
            candidate -= candidate.mean(axis=0)
            next_loss = loss_and_gradient(candidate, P)[0]
            if next_loss <= loss + 1e-14:
                break
            rate /= 2
        else:
            raise RuntimeError("line search could not find a finite descent step")
        accepted_rates.append(rate)
        Y = candidate
    return {"positions": np.asarray(positions), "losses": np.asarray(losses),
            "iterations": np.asarray(iterations), "full_losses": np.asarray(full_losses),
            "accepted_rates": np.asarray(accepted_rates), "P": P,
            "conditional": conditional, "sigmas": sigmas, "achieved_perplexities": achieved}


def make_estimator(method, n_components=2, seed=SEED, iterations=1000, epochs=400, **parameters):
    if method == "PCA":
        return PCA(n_components=n_components, svd_solver="full", **parameters)
    if method == "t-SNE":
        iteration_key = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
        settings = dict(n_components=n_components, random_state=seed, n_jobs=1,
                        init="pca", perplexity=30, metric="euclidean", learning_rate="auto",
                        **{iteration_key: iterations})
        settings.update(parameters)
        return TSNE(**settings)
    if method == "UMAP":
        try:
            from umap import UMAP
        except ModuleNotFoundError as exc:
            raise ImportError('Install: python -m pip install -e ".[dev,manifold]"') from exc
        settings = dict(n_components=n_components, random_state=seed, transform_seed=seed,
                        n_jobs=1, n_neighbors=15, min_dist=0.1, init="spectral",
                        metric="euclidean", n_epochs=epochs)
        settings.update(parameters)
        return UMAP(**settings)
    raise ValueError("method must be PCA, t-SNE or UMAP")


def fit_embedding(X, method, cache_dir, n_components=2, seed=SEED, iterations=1000, epochs=400, **parameters):
    """Cache real fits by input bytes, full parameters, versions and numerical source."""
    X = finite_matrix(X, "X")
    if len(X) > 2000 or X.shape[1] > 256 or n_components not in (2, 3):
        raise ValueError("lab limits: <=2000 rows, <=256 features, and 2 or 3 components")
    estimator = make_estimator(method, n_components, seed, iterations, epochs, **parameters)
    config = {"method": method, "shape": list(X.shape), "input_sha256": array_hash(X),
              "parameters": estimator.get_params(deep=False), "versions": versions(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            Y, record = saved["Y"], json.loads(str(saved["record"]))
        record["cache_hit"] = True
    else:
        with threadpool_limits(limits=1):
            start = perf_counter()
            Y = estimator.fit_transform(X)  # Labels deliberately excluded, including UMAP.
            elapsed = perf_counter() - start
        record = {**config, "fit_seconds": elapsed, "cache_hit": False,
                  "computed_at_utc": datetime.now(timezone.utc).isoformat()}
        if method == "PCA":
            record["explained_variance_ratio_sum"] = float(estimator.explained_variance_ratio_.sum())
        if Y.shape != (len(X), n_components) or not np.isfinite(Y).all():
            raise RuntimeError("estimator produced invalid coordinates")
        np.savez_compressed(path, Y=Y, record=json.dumps(record))
    if Y.shape != (len(X), n_components) or not np.isfinite(Y).all():
        raise RuntimeError("cache contained invalid coordinates; remove that cache file")
    return Y, record


def evaluate(X, Y, ks=(5, 10, 30)):
    with threadpool_limits(limits=1):
        result = {}
        for k in ks:
            if not isinstance(k, int) or not 1 <= k < len(X) / 2:
                raise ValueError("metric k must be an integer with 1 <= k < n/2")
            result[f"T@{k}"] = float(trustworthiness(X, Y, n_neighbors=k, metric="euclidean"))
            result[f"R@{k}"] = neighborhood_overlap(neighbor_indices(X, k), neighbor_indices(Y, k))
    return result


def umap_graph_stages(n_neighbors=15, epochs=400):
    """Exact supplied kNN -> library local scales -> union -> genuine UMAP fit."""
    X, labels = make_moons(n_samples=75, noise=0.06, random_state=SEED)
    if not isinstance(n_neighbors, int) or not 2 <= n_neighbors < len(X):
        raise ValueError("invalid graph neighborhood count")
    nonself = neighbor_indices(X, n_neighbors - 1)
    # UMAP's implementation expects self at column zero of the kNN table.
    indices = np.column_stack([np.arange(len(X)), nonself]).astype(np.int32)
    distances = np.take_along_axis(pairwise_distances(X), indices, axis=1).astype(np.float32)
    estimator = make_estimator("UMAP", n_neighbors=n_neighbors, epochs=epochs,
                               force_approximation_algorithm=True,
                               precomputed_knn=(indices, distances, None))
    graph_snapshot = {}
    fit_embed = estimator._fit_embed_data
    def capture_graph(*args, **kwargs):
        # UMAP can prune weak edges in-place when scheduling finite epochs.
        # Preserve the graph actually passed to its optimizer, before pruning.
        graph_snapshot["before_schedule"] = estimator.graph_.copy()
        return fit_embed(*args, **kwargs)
    estimator._fit_embed_data = capture_graph
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        start = perf_counter()
        Y = estimator.fit_transform(X)
        elapsed = perf_counter() - start
    sigma, rho = estimator._sigmas, estimator._rhos
    directed = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j, distance in zip(indices[i, 1:], distances[i, 1:]):
            directed[i, j] = np.exp(-max(0, float(distance - rho[i])) / float(sigma[i]))
    fuzzy_union = directed + directed.T - directed * directed.T
    graph = graph_snapshot["before_schedule"].toarray()
    error = float(np.max(np.abs(fuzzy_union - graph)))
    if error > 2e-6 or not np.isfinite(Y).all():
        raise RuntimeError("display graph differs from the fitted UMAP graph")
    return {"X": X, "labels": labels, "Y": Y, "indices": indices, "distances": distances,
            "sigma": sigma, "rho": rho, "directed": directed, "graph": graph,
            "max_graph_error": error, "fit_seconds": elapsed,
            "scheduling_prune_max_change": float(np.max(np.abs(graph - estimator.graph_.toarray()))),
            "warnings": [str(w.message) for w in caught]}


def geometry_sample():
    rng = np.random.default_rng(SEED)
    X = np.vstack([rng.normal([center, 0], spread, (60, 2))
                   for center, spread in zip((0, 4, 12), (0.15, 0.5, 1.1))])
    return X, np.repeat(np.arange(3), 60)


def group_geometry(X, labels):
    centers = np.asarray([X[labels == c].mean(axis=0) for c in range(3)])
    distances = pairwise_distances(centers)
    pairs = np.array([distances[0, 1], distances[0, 2], distances[1, 2]])
    radii = []
    for c in range(3):
        group = X[labels == c]
        neighbors = neighbor_indices(group, 5)
        radii.append(float(np.median(np.linalg.norm(group - group[neighbors[:, -1]], axis=1))))
    return {"centroid_pair_distances": pairs.tolist(),
            "relative_centroid_distances": (pairs / pairs.max()).tolist(),
            "within_group_median_5nn_radius": radii,
            "relative_5nn_radius": (np.asarray(radii) / radii[0]).tolist()}
