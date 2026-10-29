"""Merge semantics, tree conversion, Ward height units and cut boundaries."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.cluster.hierarchy import cut_tree, fcluster, is_valid_linkage, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits


TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load_module("day38_scratch", "from_scratch.py")
previous = sys.modules.get("from_scratch")
sys.modules["from_scratch"] = scratch
try:
    example = load_module("day38_example", "example.py")
finally:
    if previous is None:
        del sys.modules["from_scratch"]
    else:
        sys.modules["from_scratch"] = previous


def test_hand_computed_merges_and_deterministic_tie_order():
    Z = scratch.agglomerative_average([[1.], [2.], [8.], [9.]])
    np.testing.assert_allclose(Z, [[0, 1, 1, 2], [2, 3, 1, 2], [4, 5, 7, 4]])
    assert is_valid_linkage(Z)


def test_average_uses_all_cross_pairs_with_unequal_cluster_sizes():
    # Final merge: distances from {0,1,3} to {10} are 10,9,7; mean = 26/3.
    Z = scratch.agglomerative_average([[0.], [1.], [3.], [10.]])
    np.testing.assert_allclose(Z[:, 2], [1, 2.5, 26 / 3])
    np.testing.assert_allclose(Z[:, 3], [2, 3, 4])


@pytest.mark.parametrize("seed", [0, 17, 42])
def test_random_small_hierarchy_agrees_with_scipy_without_distance_ties(seed):
    X = np.random.default_rng(seed).normal(size=(12, 3))
    Z = scratch.agglomerative_average(X)
    reference = linkage(X, method="average", metric="euclidean")
    np.testing.assert_allclose(Z, reference, rtol=1e-12, atol=1e-12)
    assert np.all(np.diff(Z[:, 2]) >= -1e-12)


def test_duplicates_zero_heights_and_two_observations():
    Z = scratch.agglomerative_average(np.zeros((4, 2)))
    assert is_valid_linkage(Z)
    assert np.all(Z[:, 2] == 0)
    assert Z[-1, 3] == 4
    np.testing.assert_allclose(scratch.agglomerative_average([[0.], [3.]]),
                               [[0, 1, 3, 2]])


@pytest.mark.parametrize("X", [[], [1, 2], [[]], [[1]], [[np.nan], [0]],
                                   [[np.inf], [0]], [[1j], [0]], [["bad"], [0]]])
def test_invalid_coordinates_raise_explicit_errors(X):
    with pytest.raises(ValueError):
        scratch.agglomerative_average(X)


def test_distance_overflow_requests_rescaling():
    with pytest.raises(ValueError, match="rescale"):
        scratch.agglomerative_average([[1e308], [-1e308]])


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_conversion_preserves_complete_sklearn_tree_and_partitions(method):
    X, _ = example.make_data()
    with threadpool_limits(limits=1):
        model = AgglomerativeClustering(
            n_clusters=3, linkage=method, metric="euclidean",
            compute_full_tree=True, compute_distances=True,
        ).fit(X)
    Z = example.model_to_linkage(model)
    np.testing.assert_array_equal(Z[:, :2], model.children_)
    np.testing.assert_allclose(Z[:, 2], model.distances_)
    assert Z.shape == (89, 4) and Z[-1, 3] == 90
    assert adjusted_rand_score(cut_tree(Z, n_clusters=[3]).ravel(), model.labels_) == 1
    # A coarser cut of this same tree must never split a finer cluster.
    coarse, fine = cut_tree(Z, n_clusters=[2, 5]).T
    for label in np.unique(fine):
        assert len(np.unique(coarse[fine == label])) == 1


def test_ward_height_squared_is_twice_the_actual_sse_increase():
    X = np.array([[0., 0.], [1., 0.], [3., 2.], [8., -1.], [9., 0.]])
    Z = linkage(X, method="ward")
    members = {i: [i] for i in range(len(X))}

    def sse(indices):
        points = X[indices]
        return np.square(points - points.mean(axis=0)).sum()

    for step, (left, right, height, count) in enumerate(Z):
        a, b = members[int(left)], members[int(right)]
        delta = sse(a + b) - sse(a) - sse(b)
        np.testing.assert_allclose(height ** 2, 2 * delta, rtol=1e-12)
        assert len(a + b) == count
        members[len(X) + step] = a + b


def test_cut_height_reproduces_exact_partition_when_a_gap_exists():
    Z = scratch.agglomerative_average([[1.], [2.], [8.], [9.]])
    height = example.cut_height_for_k(Z, 2)
    assert height == 4
    assert adjusted_rand_score(fcluster(Z, height, criterion="distance"),
                               cut_tree(Z, n_clusters=[2]).ravel()) == 1


def test_tied_heights_do_not_claim_a_three_cluster_horizontal_cut():
    Z = scratch.agglomerative_average([[1.], [2.], [8.], [9.]])
    assert example.cut_height_for_k(Z, 3) is None
    assert len(np.unique(cut_tree(Z, n_clusters=[3]))) == 3
    assert len(np.unique(fcluster(Z, 0.99, criterion="distance"))) == 4
    assert len(np.unique(fcluster(Z, 1, criterion="distance"))) == 2


@pytest.mark.parametrize("k", [0, 1, 4, 5, -1, 2.5, True, "2"])
def test_invalid_internal_cut_counts_raise(k):
    Z = scratch.agglomerative_average([[1.], [2.], [8.], [9.]])
    with pytest.raises(ValueError, match="k"):
        example.cut_height_for_k(Z, k)


def test_nonmonotone_or_nonfinite_heights_raise():
    Z = scratch.agglomerative_average([[1.], [2.], [8.], [9.]])
    for heights in ([2, 1, 7], [1, 1, np.inf], [1, 1, np.nan]):
        Z[:, 2] = heights
        with pytest.raises(ValueError):
            example.cut_height_for_k(Z, 2)


def test_unfitted_missing_distance_and_partial_trees_raise():
    for model in (AgglomerativeClustering(), SimpleNamespace(
        n_leaves_=4, children_=np.array([[0, 1]]), distances_=np.array([1.]),
    )):
        with pytest.raises(ValueError):
            example.model_to_linkage(model)


def test_ward_rejects_cosine_in_the_library_interface():
    with pytest.raises(ValueError, match="[Ee]uclidean"):
        AgglomerativeClustering(n_clusters=2, linkage="ward", metric="cosine").fit(
            [[1., 0.], [1., 1.], [0., 1.]],
        )
