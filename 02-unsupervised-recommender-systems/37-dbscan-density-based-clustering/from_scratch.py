"""Educational, unweighted Euclidean DBSCAN with quadratic distance storage."""

from collections import deque
from numbers import Integral, Real
from typing import NamedTuple

import numpy as np


class DBSCANStep(NamedTuple):
    """An exact snapshot; -2 is pending, -1 is provisional/final noise."""

    operation: str
    caption: str
    current: int | None
    neighborhood: np.ndarray
    labels: np.ndarray
    visited: np.ndarray
    expanded: np.ndarray


def _prepare_neighbors(X, eps, min_samples):
    raw = np.asarray(X)
    if np.iscomplexobj(raw):
        raise ValueError("X must contain real coordinates")
    try:
        X = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain numeric coordinates") from exc
    if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite 2D array with features")
    if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, Real) or not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a finite positive real number")
    if isinstance(min_samples, (bool, np.bool_)) or not isinstance(min_samples, Integral) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")

    n = len(X)
    distances = np.empty((n, n), dtype=float)
    # Row-wise differences avoid an additional n x n x features tensor.
    try:
        with np.errstate(over="raise", invalid="raise"):
            for i in range(n):
                distances[i] = np.linalg.norm(X - X[i], axis=1)
    except FloatingPointError as exc:
        raise ValueError("Coordinate magnitudes overflow Euclidean distance; rescale X") from exc
    neighbors = [np.flatnonzero(row <= eps) for row in distances]
    core = np.array([len(indices) >= min_samples for indices in neighbors])
    return neighbors, core


def _run_dbscan(X, eps, min_samples, *, record):
    neighbors, core = _prepare_neighbors(X, eps, min_samples)
    n = len(neighbors)
    if record and n > 200:
        raise ValueError("The educational trace is limited to 200 points")
    labels = np.full(n, -2, dtype=int)
    visited = np.zeros(n, dtype=bool)
    expanded = np.zeros(n, dtype=bool)
    steps = []

    def emit(operation, caption, current=None):
        if record:
            neighborhood = np.array([], dtype=int) if current is None else neighbors[current].copy()
            steps.append(DBSCANStep(operation, caption, current, neighborhood,
                                    labels.copy(), visited.copy(), expanded.copy()))

    emit("initial", "All observations are pending; no cluster exists yet.")
    cluster_id = 0
    for seed in range(n):
        if visited[seed]:
            continue
        emit("select", f"Select unvisited point {seed}.", seed)
        visited[seed] = True
        emit("inspect", f"Point {seed}: {len(neighbors[seed])} neighbors including self; need {min_samples}.", seed)
        if not core[seed]:
            labels[seed] = -1
            emit("noise", f"Point {seed} is not core: mark noise for now; a later core may claim it.", seed)
            continue
        labels[seed] = cluster_id
        emit("start", f"Point {seed} is core: start cluster {cluster_id}.", seed)
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            expanded[current] = True
            emit("expand", f"Expand core {current}: inspect its epsilon neighborhood.", current)
            for neighbor in neighbors[current]:
                if labels[neighbor] >= 0:
                    continue
                was_noise = labels[neighbor] == -1
                visited[neighbor] = True
                labels[neighbor] = cluster_id
                if core[neighbor]:
                    queue.append(neighbor)
                    emit("add_core", f"Point {neighbor}: {len(neighbors[neighbor])} neighbors; core, add to expansion queue.", neighbor)
                else:
                    operation = "noise_to_border" if was_noise else "border"
                    prefix = "Reclassify earlier noise" if was_noise else "Attach border"
                    emit(operation, f"{prefix} {neighbor}: only {len(neighbors[neighbor])} neighbors; never expand from it.", neighbor)
        emit("complete", f"Cluster {cluster_id} complete; continue searching for unvisited observations.")
        cluster_id += 1
    emit("finished", "Finished: core components plus attached borders; remaining -1 points are noise.")
    return labels, core, tuple(steps)


def dbscan_numpy(X, eps=0.18, min_samples=7):
    """Return labels and core mask for dense unweighted Euclidean observations.

    Counts include self and the epsilon boundary. IDs follow input order;
    a shared border attaches to the first discovered component. No predict API.
    """
    labels, core, _ = _run_dbscan(X, eps, min_samples, record=False)
    return labels, core


def dbscan_trace(X, eps=0.18, min_samples=7):
    """Return labels, core mask and exact teaching snapshots (at most 200 points).

    Uses the same engine as dbscan_numpy. Intermediate noise is provisional;
    only core points enter the queue. Trace storage adds O(number_of_steps * n).
    """
    return _run_dbscan(X, eps, min_samples, record=True)


if __name__ == "__main__":
    points = np.array([[0.], [0.1], [0.2], [0.3], [0.55], [2.]])
    labels, core = dbscan_numpy(points, eps=0.3, min_samples=4)
    print("Code-defined 1D points:", points.ravel().tolist())
    print("eps=0.3; min_samples=4 (includes self)")
    print("Labels:", labels.tolist())
    print("Core mask:", core.tolist())
