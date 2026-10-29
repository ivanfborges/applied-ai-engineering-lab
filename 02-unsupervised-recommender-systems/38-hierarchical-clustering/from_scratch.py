"""Transparent Euclidean average linkage for small educational inputs."""

from itertools import combinations

import numpy as np


def agglomerative_average(X):
    """Return SciPy-format [left, right, height, count] rows.

    Recompute mean cross-cluster distances from a fixed point-distance matrix.
    Equal distances are resolved by the smallest pair of active cluster IDs.
    Input rows are observations, even for a square array; distances are not input.
    """
    try:
        raw = np.asarray(X)
        if np.iscomplexobj(raw):
            raise ValueError("complex coordinates are unsupported")
        X = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("X must contain real numeric coordinates") from error
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must be a 2D array with at least two rows and one feature")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite coordinates")

    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    if not np.isfinite(distances).all():
        raise ValueError("distance overflow: rescale the coordinates")

    n = len(X)
    clusters = {i: [i] for i in range(n)}
    Z = np.empty((n - 1, 4), dtype=float)
    for step in range(n - 1):
        candidates = []
        for left, right in combinations(sorted(clusters), 2):
            cross = distances[np.ix_(clusters[left], clusters[right])]
            # Scale the positive distances before averaging to avoid sum overflow.
            scale = float(cross.max())
            height = scale * float((cross / scale).mean()) if scale else 0.0
            candidates.append((height, left, right))
        height, left, right = min(candidates)
        members = clusters.pop(left) + clusters.pop(right)
        Z[step] = [left, right, height, len(members)]
        clusters[n + step] = members
    return Z


def main():
    X = np.array([[1.], [2.], [8.], [9.]])
    print("Synthetic 1D observations:", X.ravel().tolist())
    print("Average linkage: left ID, right ID, height, observation count")
    print(agglomerative_average(X))
    print("Leaves use row indices 0..3; new nodes use IDs 4..6.")


if __name__ == "__main__":
    main()
