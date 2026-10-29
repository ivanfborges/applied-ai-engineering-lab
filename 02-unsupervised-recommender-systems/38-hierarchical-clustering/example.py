"""Compare linkage criteria and inspect dendrogram data without plotting."""

from numbers import Integral
import platform

import numpy as np
import scipy
from scipy.cluster.hierarchy import cut_tree, is_monotonic, is_valid_linkage, linkage
import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from from_scratch import agglomerative_average


def model_to_linkage(model):
    """Convert a complete fitted sklearn tree with distances to SciPy format."""
    required = ("children_", "distances_", "n_leaves_")
    if any(not hasattr(model, attr) for attr in required):
        raise ValueError("fit a full tree with compute_distances=True first")
    n = model.n_leaves_
    if model.children_.shape != (n - 1, 2) or len(model.distances_) != n - 1:
        raise ValueError("a complete n-1 merge tree is required")
    counts = np.zeros(n - 1)
    for step, children in enumerate(model.children_):
        counts[step] = sum(1 if child < n else counts[child - n] for child in children)
    Z = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    is_valid_linkage(Z, throw=True)
    return Z


def cut_height_for_k(Z, k):
    """Return an interior horizontal cut for 2 <= k < n, or None at a tie.

    This requires monotone heights. An exact-k merge-order cut can still exist
    when no strictly interior floating-point height separates its boundaries.
    """
    Z = np.asarray(Z, dtype=float)
    is_valid_linkage(Z, throw=True)
    if not np.isfinite(Z).all() or not is_monotonic(Z):
        raise ValueError("Z must have finite, monotone heights")
    n = len(Z) + 1
    if (isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral)
            or not 2 <= k < n):
        raise ValueError("k must be an integer with 2 <= k < n")
    lower, upper = Z[n - k - 1, 2], Z[n - k, 2]
    height = lower + (upper - lower) / 2
    return float(height) if lower < height < upper else None


def make_data():
    """Return 90 synthetic observations; generator labels are not business truth."""
    X, generating_labels = make_blobs(
        n_samples=90, centers=[[-3, -2], [0, 3], [3, -1]],
        cluster_std=[0.65, 0.9, 0.75], random_state=42,
    )
    return StandardScaler().fit_transform(X), generating_labels


def run_experiment():
    X, generating_labels = make_data()
    print("Synthetic blobs: n=90; seed=42; centers=[[-3,-2],[0,3],[3,-1]]")
    print("std=[0.65,0.9,0.75]; StandardScaler on all 90 exploratory rows")
    print("Euclidean; full unstructured trees; K=3 is fixed using generator knowledge")
    print("method     sorted_sizes  silhouette  generator_ARI")
    models = {}
    for method in ("single", "complete", "average", "ward"):
        model = AgglomerativeClustering(
            n_clusters=3, metric="euclidean", linkage=method,
            compute_full_tree=True, compute_distances=True,
        ).fit(X)
        models[method] = model
        sizes = sorted(np.bincount(model.labels_).tolist())
        print(f"{method:9s} {str(sizes):13s} "
              f"{silhouette_score(X, model.labels_):10.6f} "
              f"{adjusted_rand_score(generating_labels, model.labels_):14.6f}")

    Z = model_to_linkage(models["ward"])
    print("\nSame Ward tree, exact-K cuts (no refitting or label-based selection)")
    print("K  sorted_sizes          silhouette  interior_cut_height")
    for k in (2, 3, 4, 5):
        labels = cut_tree(Z, n_clusters=[k]).ravel()
        sizes = sorted(np.bincount(labels).tolist())
        height = cut_height_for_k(Z, k)
        print(f"{k}  {str(sizes):21s} {silhouette_score(X, labels):10.6f}  {height}")
        if k == 3:
            assert np.isclose(adjusted_rand_score(labels, models["ward"].labels_), 1)
    print("Last four Ward merges: left ID, right ID, height, observation count")
    print(np.round(Z[-4:], 6))

    tiny = np.array([[1.], [2.], [8.], [9.]])
    educational = agglomerative_average(tiny)
    np.testing.assert_allclose(educational, linkage(tiny, method="average"))
    print("\nFour-point average linkage: NumPy/SciPy matrix agreement passed")
    print("Heights:", educational[:, 2].tolist())
    print("K=3 interior height:", cut_height_for_k(educational, 3),
          "(equal heights; use merge order for an exact-K partition)")
    print("Interpretation candidates require author review; no files written.")


def main():
    print(f"Python {platform.python_version()}; NumPy {np.__version__}; "
          f"SciPy {scipy.__version__}; scikit-learn {sklearn.__version__}")
    with threadpool_limits(limits=1):
        run_experiment()


if __name__ == "__main__":
    main()
