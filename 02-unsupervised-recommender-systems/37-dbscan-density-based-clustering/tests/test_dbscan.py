"""DBSCAN semantics, ambiguous borders, and a self-inclusive k-distance check."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits


TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load_module("day37_scratch", "from_scratch.py")
previous = sys.modules.get("from_scratch")
sys.modules["from_scratch"] = scratch
try:
    example = load_module("day37_example", "example.py")
finally:
    if previous is None:
        del sys.modules["from_scratch"]
    else:
        sys.modules["from_scratch"] = previous


def test_self_and_epsilon_boundary_are_included():
    labels, core = scratch.dbscan_numpy([[0.], [1.], [2.]], eps=1, min_samples=3)
    np.testing.assert_array_equal(core, [False, True, False])
    np.testing.assert_array_equal(labels, [0, 0, 0])


def test_earlier_noncore_point_can_later_be_assigned_as_border():
    X = np.array([[0.], [0.75], [1.], [1.25], [4.]])
    labels, core = scratch.dbscan_numpy(X, eps=0.8, min_samples=4)
    np.testing.assert_array_equal(core, [False, True, False, False, False])
    np.testing.assert_array_equal(labels, [0, 0, 0, 0, -1])


def test_border_chain_cannot_propagate_a_cluster():
    X = np.array([[-0.3], [-0.2], [-0.1], [0.], [0.95], [1.9]])
    labels, core = scratch.dbscan_numpy(X, eps=1, min_samples=4)
    assert not core[-2:].any()
    assert labels[-2] == labels[0] >= 0
    assert labels[-1] == -1


def test_shared_border_does_not_merge_components_and_can_switch_on_reorder():
    X = np.array([[-1.6], [-1.5], [-1.4], [-0.9], [0.], [0.9], [1.4], [1.5], [1.6]])
    labels, core = scratch.dbscan_numpy(X, eps=1, min_samples=4)
    reverse_labels, reverse_core = scratch.dbscan_numpy(X[::-1], eps=1, min_samples=4)
    reverse_labels, reverse_core = reverse_labels[::-1], reverse_core[::-1]
    np.testing.assert_array_equal(core, [True, True, True, True, False, True, True, True, True])
    np.testing.assert_array_equal(core, reverse_core)
    assert len(np.unique(labels[core])) == 2
    assert adjusted_rand_score(labels[core], reverse_labels[core]) == 1
    assert labels[4] == labels[0]
    assert reverse_labels[4] == reverse_labels[-1]
    assert reverse_labels[4] != reverse_labels[0]
    with threadpool_limits(limits=1):
        reference = DBSCAN(eps=1, min_samples=4).fit(X)
    assert adjusted_rand_score(labels, reference.labels_) == 1


@pytest.mark.parametrize("eps,m", [(0.08, 7), (0.18, 7), (0.45, 7), (0.18, 20)])
def test_synthetic_core_connectivity_and_noise_agree_with_sklearn(eps, m):
    X, _ = example.make_data()
    labels, core = scratch.dbscan_numpy(X, eps=eps, min_samples=m)
    with threadpool_limits(limits=1):
        model = DBSCAN(eps=eps, min_samples=m).fit(X)
    expected_core = np.zeros(len(X), dtype=bool)
    expected_core[model.core_sample_indices_] = True
    np.testing.assert_array_equal(core, expected_core)
    np.testing.assert_array_equal(labels == -1, model.labels_ == -1)
    assert adjusted_rand_score(labels[core], model.labels_[core]) == 1


def test_duplicates_singleton_all_noise_and_oversized_density_threshold():
    labels, core = scratch.dbscan_numpy([[0.], [0.], [3.]], eps=0.1, min_samples=2)
    np.testing.assert_array_equal(labels, [0, 0, -1])
    np.testing.assert_array_equal(core, [True, True, False])
    labels, core = scratch.dbscan_numpy([[0.]], eps=1, min_samples=1)
    assert labels[0] == 0 and core[0]
    labels, core = scratch.dbscan_numpy([[0.], [2.]], eps=0.1, min_samples=3)
    assert np.all(labels == -1) and not core.any()


@pytest.mark.parametrize("X", [[], [1, 2], [[]], [[np.nan]], [[np.inf]], [[1j]], [["bad"]]])
def test_invalid_points_raise(X):
    with pytest.raises(ValueError):
        scratch.dbscan_numpy(X)


@pytest.mark.parametrize("eps", [0, -1, np.inf, np.nan, True, "0.1"])
def test_invalid_radius_raises(eps):
    with pytest.raises(ValueError, match="eps"):
        scratch.dbscan_numpy([[0.]], eps=eps)


@pytest.mark.parametrize("m", [0, -1, 1.5, True, "2"])
def test_invalid_min_samples_raises(m):
    with pytest.raises(ValueError, match="min_samples"):
        scratch.dbscan_numpy([[0.]], min_samples=m)


def test_extreme_coordinates_raise_instead_of_silent_overflow():
    with pytest.raises(ValueError, match="rescale"):
        scratch.dbscan_numpy([[1e308], [-1e308]])


def test_k_distance_includes_self_and_reaches_last_observation():
    X = [[0.], [1.], [3.]]
    with threadpool_limits(limits=1):
        np.testing.assert_allclose(example.kth_neighbor_distances(X, 1), [0, 0, 0])
        np.testing.assert_allclose(example.kth_neighbor_distances(X, 2), [1, 1, 2])
        np.testing.assert_allclose(example.kth_neighbor_distances(X, 3), [2, 3, 3])


@pytest.mark.parametrize("m", [0, 4, True, 1.5])
def test_k_distance_rejects_invalid_neighbor_count(m):
    with pytest.raises(ValueError, match="min_samples"):
        example.kth_neighbor_distances([[0.], [1.], [3.]], m)
