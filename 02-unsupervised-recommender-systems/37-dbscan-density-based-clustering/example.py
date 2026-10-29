"""Synthetic density sensitivity and K-Means comparison; writes no files."""

from numbers import Integral
import platform

import numpy as np
import scipy
import sklearn
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array
from threadpoolctl import threadpool_limits

from from_scratch import dbscan_numpy


SEED = 42
MOON_COUNT = 450


def make_data():
    """Synthetic moons followed by background; provenance is not anomaly truth."""
    moons, generating_labels = make_moons(
        n_samples=MOON_COUNT, noise=0.06, random_state=SEED,
    )
    background = np.random.default_rng(SEED).uniform(
        low=[-1.4, -0.9], high=[2.4, 1.4], size=(35, 2),
    )
    return np.vstack([moons, background]), generating_labels


def kth_neighbor_distances(X, min_samples):
    """Sorted distances to neighbor m INCLUDING self, matching DBSCAN counts."""
    X = check_array(X, ensure_min_samples=1, ensure_min_features=1)
    if (isinstance(min_samples, (bool, np.bool_))
            or not isinstance(min_samples, Integral)
            or not 1 <= min_samples <= len(X)):
        raise ValueError("min_samples must be an integer between 1 and len(X)")
    neighbors = NearestNeighbors(n_neighbors=min_samples, metric="euclidean").fit(X)
    # Passing X explicitly includes self; kneighbors() without X excludes it.
    distances, _ = neighbors.kneighbors(X)
    return np.sort(distances[:, -1])


def run_experiment():
    X, generating_labels = make_data()
    radii = kth_neighbor_distances(X, min_samples=7)
    print("7-distance quantiles (includes self); q=[0.5, 0.9, 0.95, 0.99, 1]")
    print(np.round(np.quantile(radii, [0.5, 0.9, 0.95, 0.99, 1]), 6).tolist())
    print("\nAll fits use the same 485 raw 2D points and Euclidean distance.")
    print("ARI compares ONLY the 450 moon points with generator labels; -1 stays a label.")
    print(" eps    m  clusters core border noise noise_frac moon_noise bg_noise moon_ARI")
    settings = [(eps, 7) for eps in (0.08, 0.12, 0.18, 0.30, 0.45)]
    settings += [(0.18, m) for m in (3, 5, 12, 20)]
    reference = None
    for eps, m in settings:
        model = DBSCAN(eps=eps, min_samples=m, metric="euclidean").fit(X)
        labels = model.labels_
        core = np.zeros(len(X), dtype=bool)
        core[model.core_sample_indices_] = True
        noise = labels == -1
        clusters = len(np.unique(labels[~noise]))
        ari = adjusted_rand_score(generating_labels, labels[:MOON_COUNT])
        print(f"{eps:5.2f} {m:4d} {clusters:9d} {core.sum():4d} "
              f"{((~core) & (~noise)).sum():6d} {noise.sum():5d} "
              f"{noise.mean():10.6f} {noise[:MOON_COUNT].sum():10d} "
              f"{noise[MOON_COUNT:].sum():8d} {ari:8.6f}")
        if eps == 0.18 and m == 7:
            reference = model, core

    kmeans = KMeans(n_clusters=2, n_init=10, random_state=SEED, algorithm="lloyd").fit(X)
    print("\nK-Means K=2, n_init=10, seed=42; all points assigned")
    print(f"Moon-only ARI={adjusted_rand_score(generating_labels, kmeans.labels_[:MOON_COUNT]):.6f}")
    print("K=2 uses generator knowledge; neither method was tuned against these labels.")

    reference_model, reference_core = reference
    labels, core = dbscan_numpy(X, eps=0.18, min_samples=7)
    np.testing.assert_array_equal(core, reference_core)
    np.testing.assert_array_equal(labels == -1, reference_model.labels_ == -1)
    core_ari = adjusted_rand_score(reference_model.labels_[core], labels[core])
    assert np.isclose(core_ari, 1.0)
    print("\nNumPy/sklearn default-fit check: core mask and noise mask agree")
    print(f"Core partition ARI={core_ari:.6f}; full partition ARI="
          f"{adjusted_rand_score(reference_model.labels_, labels):.6f}")


def main():
    print(f"Python {platform.python_version()}; NumPy {np.__version__}; "
          f"SciPy {scipy.__version__}; scikit-learn {sklearn.__version__}")
    print("Synthetic corpus; seed=42; 450 moons (noise=0.06) + 35 uniform background")
    with threadpool_limits(limits=1):
        run_experiment()


if __name__ == "__main__":
    main()
