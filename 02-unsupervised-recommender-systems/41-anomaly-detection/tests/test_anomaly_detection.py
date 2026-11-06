"""Numerical identities, score contracts and separation of fit/calibration/test."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import median_abs_deviation


TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scratch = load("from_scratch", "from_scratch.py")
example = load("day41_example", "example.py")


@pytest.mark.parametrize("n,expected", [(0, 0), (1, 0), (2, 1), (3, 5 / 3), (4, 13 / 6)])
def test_exact_path_correction(n, expected):
    assert scratch.average_path_length(n) == pytest.approx(expected)


@pytest.mark.parametrize("n", [-1, 1.5, True])
def test_invalid_leaf_sizes(n):
    with pytest.raises(ValueError):
        scratch.average_path_length(n)


def test_hand_built_paths_include_terminal_correction():
    tree = scratch.IsolationNode(4, 0, 2.0, scratch.IsolationNode(1), scratch.IsolationNode(3))
    assert scratch.path_length([2], tree) == 1
    assert scratch.path_length([3], tree) == pytest.approx(1 + 5 / 3)


def test_duplicate_data_and_subsample_cap():
    model = scratch.EducationalIsolationForest(n_estimators=4, max_samples=128).fit(np.ones((8, 2)))
    assert model.sample_size_ == 8
    assert all(tree.feature is None for tree in model.trees_)
    np.testing.assert_allclose(model.anomaly_score([[1, 1], [100, 100]]), 0.5)


def test_forest_determinism_nonempty_splits_and_score_orientation():
    X = np.random.default_rng(5).normal(size=(200, 2))
    X[:, 1] = 7  # The constant feature must never be chosen for splitting.
    model = scratch.EducationalIsolationForest(n_estimators=40, max_samples=64).fit(X)
    scores = model.anomaly_score(np.vstack((X, [15, 7])))
    again = scratch.EducationalIsolationForest(n_estimators=40, max_samples=64).fit(X)
    np.testing.assert_array_equal(scores, again.anomaly_score(np.vstack((X, [15, 7]))))
    assert np.all((scores > 0) & (scores <= 1))
    assert scores[-1] > np.median(scores[:-1])

    def inspect(node, depth=0):
        assert node.size > 0
        assert depth <= 6
        if node.feature is not None:
            assert node.feature == 0
            assert node.left.size + node.right.size == node.size
            inspect(node.left, depth + 1)
            inspect(node.right, depth + 1)

    for tree in model.trees_:
        assert tree.size == 64
        inspect(tree)


def test_mad_matches_scipy_and_positive_affine_units():
    X = np.array([[0, 2], [1, 4], [2, 6], [3, 8], [1000, 9000]], dtype=float)
    model = scratch.RobustMAD().fit(X)
    np.testing.assert_allclose(model.center_, [2, 6])
    np.testing.assert_allclose(model.scale_, median_abs_deviation(X, axis=0, scale="normal"))
    scores = model.anomaly_score(X)
    transformed = X * [100, 0.1] + [30, -70]
    np.testing.assert_allclose(scores, scratch.RobustMAD().fit(transformed).anomaly_score(transformed))
    center_before = model.center_.copy()
    model.anomaly_score([[1e6, 1e6]])
    np.testing.assert_array_equal(model.center_, center_before)


@pytest.mark.parametrize("X", [np.ones((5, 2)), [[0], [0], [0], [0], [1]]])
def test_zero_mad_is_explicit_error(X):
    with pytest.raises(ValueError, match="positive finite MAD"):
        scratch.RobustMAD().fit(X)


@pytest.mark.parametrize("X", [[], [1, 2], [[np.nan]], [[np.inf]], np.empty((2, 0))])
@pytest.mark.parametrize("factory", [scratch.RobustMAD, scratch.EducationalIsolationForest])
def test_invalid_feature_matrix(factory, X):
    with pytest.raises(ValueError):
        factory().fit(X)


@pytest.mark.parametrize("factory", [scratch.RobustMAD, scratch.EducationalIsolationForest])
def test_unfitted_and_feature_mismatch(factory):
    model = factory()
    with pytest.raises(ValueError, match="fit"):
        model.anomaly_score([[1]])
    model.fit(np.arange(12).reshape(6, 2))
    with pytest.raises(ValueError, match="feature count"):
        model.anomaly_score([[1]])


@pytest.mark.parametrize("kwargs", [{"n_estimators": 0}, {"max_samples": 1}, {"max_samples": 2.5}, {"random_state": -1}])
def test_invalid_forest_parameters(kwargs):
    with pytest.raises(ValueError):
        scratch.EducationalIsolationForest(**kwargs).fit([[0], [1]])


def test_forest_needs_two_rows():
    with pytest.raises(ValueError, match="two"):
        scratch.EducationalIsolationForest().fit([[1]])


def test_threshold_order_statistic_strict_ties_and_monotonicity():
    scores = np.array([0, 1, 2, 3, 4])
    threshold = scratch.calibrate_threshold(scores, alpha=0.25)
    assert threshold == 3
    assert (scores > threshold).tolist() == [False, False, False, False, True]
    assert scratch.calibrate_threshold([1, 1, 1, 1], 0.25) == 1
    assert not np.any(np.ones(4) > 1)
    assert scratch.calibrate_threshold(scores, 0.1) >= threshold


@pytest.mark.parametrize("scores,alpha", [([], 0.02), ([np.nan], 0.02), ([[1]], 0.02), ([1], 0), ([1], 1), ([1], np.nan)])
def test_invalid_calibration(scores, alpha):
    with pytest.raises(ValueError):
        scratch.calibrate_threshold(scores, alpha)


def test_policy_metrics_from_known_confusion_matrix():
    result = example.evaluate_policy(["normal", "normal", "relationship", "extreme"], [0, 3, 4, 1], 2)
    assert [result[k] for k in ("tn", "fp", "fn", "tp")] == [1, 1, 1, 1]
    assert result["precision"] == result["recall"] == result["fpr"] == 0.5
    assert result["recall_by_injection"] == {"relationship": 1, "extreme": 0}
    assert result["ap"] == pytest.approx((1 + 2 / 3) / 2)
    assert example.evaluate_policy(["normal", "extreme"], [0, 1], 9)["precision"] == 0


def test_scaler_fit_boundary_and_library_score_signs():
    train, cal, test, _ = example.make_telemetry()
    models = example.fit_detectors(train)
    pipeline = models["OneClassSVM"]
    scaler = pipeline.steps[0][1]
    np.testing.assert_allclose(scaler.center_, np.median(train, axis=0))
    before = scaler.center_.copy()
    for name, model in models.items():
        scores = example.anomaly_scores(name, model, test)
        assert scores.shape == (340,)
        example.anomaly_scores(name, model, cal * 100)
    np.testing.assert_array_equal(scaler.center_, before)
    isolation = models["IsolationForest"]
    np.testing.assert_allclose(example.anomaly_scores("IsolationForest", isolation, test), -isolation.score_samples(test))
    np.testing.assert_allclose(isolation.decision_function(test), isolation.score_samples(test) - isolation.offset_)
    np.testing.assert_allclose(example.anomaly_scores("OneClassSVM", pipeline, test), -pipeline.decision_function(test))


def test_full_synthetic_comparison_is_deterministic_and_thresholds_do_not_change_ap():
    first, second = example.run_experiment(), example.run_experiment()
    assert first["results"] == second["results"]
    assert first["interpretation_status"] == "author review required"
    assert len(first["results"]) == 9
    for name in ("MAD", "IsolationForest", "OneClassSVM"):
        rows = [r for r in first["results"] if r["detector"] == name]
        assert len({r["ap"] for r in rows}) == 1
        assert rows[0]["alerts"] >= rows[1]["alerts"] >= rows[2]["alerts"]
        assert rows[0]["threshold"] <= rows[1]["threshold"] <= rows[2]["threshold"]
        assert all(r["tn"] + r["fp"] == 300 and r["tp"] + r["fn"] == 40 for r in rows)
