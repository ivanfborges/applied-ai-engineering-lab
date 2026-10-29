"""Numerical invariants, explicit stopping behavior, and preprocessing boundaries."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits


TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load_module("day36_scratch", "from_scratch.py")
previous = sys.modules.get("from_scratch")
sys.modules["from_scratch"] = scratch
try:
    example = load_module("day36_example", "example.py")
finally:
    if previous is None:
        del sys.modules["from_scratch"]
    else:
        sys.modules["from_scratch"] = previous


def test_distances_known_answer_ties_and_feature_validation():
    np.testing.assert_array_equal(
        scratch.squared_distances([[0, 0], [3, 4]], [[0, 0], [0, 4]]),
        [[0, 16], [25, 9]],
    )
    np.testing.assert_array_equal(scratch.predict_labels([[1]], [[0], [2]]), [0])
    with pytest.raises(ValueError, match="features"):
        scratch.squared_distances([[1, 2]], [[1]])
    with pytest.raises(ValueError, match="rescale"):
        scratch.squared_distances([[1e308]], [[-1e308]])


def test_centroid_means_inertia_history_and_no_mutation():
    X = np.array([[0., 0], [0, 2], [2, 0], [10, 10], [10, 12], [12, 10]])
    initial = X[[0, 3]].copy()
    original_X, original_initial = X.copy(), initial.copy()
    result = scratch.kmeans_numpy(X, 2, init=initial, tol=0)
    np.testing.assert_allclose(result.centers, [[2 / 3, 2 / 3], [32 / 3, 32 / 3]])
    np.testing.assert_array_equal(result.labels, [0, 0, 0, 1, 1, 1])
    assert result.inertia == pytest.approx(32 / 3)
    assert result.inertia_history[0] == pytest.approx(16)
    assert result.inertia_history[-1] == result.inertia
    assert len(result.inertia_history) == result.n_iter + 1
    assert np.all(np.diff(result.inertia_history) <= 1e-10)
    assert result.converged
    for cluster in range(2):
        np.testing.assert_allclose(result.centers[cluster], X[result.labels == cluster].mean(axis=0))
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(initial, original_initial)


def test_iteration_cap_reassigns_to_returned_centers():
    X = np.array([[0.], [1], [4], [8], [10]])
    result = scratch.kmeans_numpy(X, 2, init=[[0], [1]], max_iter=1, tol=0)
    np.testing.assert_allclose(result.centers, [[0], [5.75]])
    np.testing.assert_array_equal(result.labels, [0, 0, 1, 1, 1])
    assert result.inertia == pytest.approx(27.1875)
    assert result.n_iter == 1
    assert not result.converged
    np.testing.assert_array_equal(result.labels, scratch.predict_labels(X, result.centers))


def test_tolerance_stop_can_precede_final_label_mean_consistency():
    result = scratch.kmeans_numpy([[0], [1], [4], [8], [10]], 2, init=[[0], [1]], tol=100)
    assert result.converged
    assert result.n_iter == 1
    assert result.centers[0, 0] == 0
    np.testing.assert_array_equal(result.labels, [0, 0, 1, 1, 1])


def test_k_one_and_k_n():
    X = np.array([[0., 2], [2, 4], [4, 0]])
    one = scratch.kmeans_numpy(X, 1)
    np.testing.assert_allclose(one.centers, X.mean(axis=0, keepdims=True))
    assert one.inertia == pytest.approx(np.sum((X - X.mean(axis=0)) ** 2))
    each = scratch.kmeans_numpy(X, len(X))
    assert each.inertia == 0
    assert len(np.unique(each.labels)) == len(X)


def test_empty_cluster_retains_previous_center_and_duplicates_are_supported():
    result = scratch.kmeans_numpy([[0], [0], [10]], 3, init=[[0], [0], [10]])
    np.testing.assert_array_equal(result.centers, [[0], [0], [10]])
    np.testing.assert_array_equal(result.labels, [0, 0, 2])
    assert result.inertia == 0
    assert result.converged
    same = scratch.kmeans_numpy([[3, 3]] * 4, 3)
    assert len(np.unique(same.labels)) == 1
    assert same.inertia == 0


def test_seed_reproducibility_and_monotonic_objective():
    X = np.random.default_rng(8).normal(size=(70, 3))
    first = scratch.kmeans_numpy(X, 4, seed=9, tol=0)
    second = scratch.kmeans_numpy(X, 4, seed=9, tol=0)
    np.testing.assert_array_equal(first.centers, second.centers)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert np.all(np.diff(first.inertia_history) <= 1e-10)
    assert first.inertia == pytest.approx(np.sum((X - first.centers[first.labels]) ** 2))


@pytest.mark.parametrize("X", [[], [[]], [1, 2], [[[1]]], [[np.nan]], [[np.inf]], [[-np.inf]], [["text"]]])
def test_invalid_matrices(X):
    with pytest.raises(ValueError):
        scratch.kmeans_numpy(X, 1)
    with pytest.raises(ValueError):
        scratch.squared_distances([[0]], X)


@pytest.mark.parametrize("kwargs", [
    {"k": 0}, {"k": 4}, {"k": 1.5}, {"k": True},
    {"max_iter": 0}, {"max_iter": -1}, {"max_iter": 2.5}, {"max_iter": True},
    {"tol": -1}, {"tol": np.nan}, {"tol": np.inf}, {"tol": True}, {"tol": "small"},
    {"seed": -1}, {"seed": True}, {"seed": 1.5},
    {"init": [[0, 0]]}, {"init": [[0], [1]]}, {"init": [[np.nan]]},
])
def test_invalid_configuration(kwargs):
    options = {"k": 1, **kwargs}
    with pytest.raises(ValueError):
        scratch.kmeans_numpy([[0], [1], [2]], **options)


def test_matches_sklearn_with_identical_initialization_up_to_label_permutation():
    X, _ = example.make_data()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    initial = X[[0, 100, 200]].copy()
    actual = scratch.kmeans_numpy(X, 3, init=initial, tol=0, max_iter=300)
    with threadpool_limits(limits=1):
        expected = KMeans(n_clusters=3, init=initial.copy(), n_init=1, tol=0, max_iter=300, algorithm="lloyd").fit(X)
    assert actual.converged
    assert actual.inertia == pytest.approx(expected.inertia_, rel=1e-10)
    assert adjusted_rand_score(actual.labels, expected.labels_) == pytest.approx(1)


def test_pipeline_inference_keeps_training_scaler_and_centers_frozen():
    X = np.array([[-2., 100], [-1, 200], [1, 400], [2, 500]])
    with threadpool_limits(limits=1):
        pipeline = example.build_model(2, standardized=True).fit(X)
        scaler = pipeline.named_steps["standardscaler"]
        centers = pipeline.named_steps["kmeans"].cluster_centers_.copy()
        mean, scale = scaler.mean_.copy(), scaler.scale_.copy()
        future = np.array([[100., 8000], [-200, -4000]])
        assigned = pipeline.predict(future)
    np.testing.assert_allclose(mean, X.mean(axis=0))
    np.testing.assert_allclose(scale, X.std(axis=0))
    np.testing.assert_array_equal(assigned, scratch.predict_labels(scaler.transform(future), centers))
    np.testing.assert_array_equal(scaler.mean_, mean)
    np.testing.assert_array_equal(scaler.scale_, scale)
    np.testing.assert_array_equal(pipeline.named_steps["kmeans"].cluster_centers_, centers)


def test_diagnostics_recomputes_known_inertia_and_handles_one_cluster():
    inertia, silhouette = example.diagnostics(np.array([[0.], [2]]), np.array([0, 0]), np.array([[1.]]))
    assert inertia == 2
    assert silhouette is None
