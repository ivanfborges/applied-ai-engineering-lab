"""Geometry, decision-set optima, invalid inputs and data boundaries."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load_module("day33_scratch", "from_scratch.py")
previous = sys.modules.get("from_scratch")
sys.modules["from_scratch"] = scratch
try:
    example = load_module("day33_example", "example.py")
finally:
    if previous is None:
        del sys.modules["from_scratch"]
    else:
        sys.modules["from_scratch"] = previous


def test_smote_stays_on_segment_and_is_seeded_without_mutation():
    X = np.array([[2.0, 4.0], [6.0, 8.0]])
    original = X.copy()
    generated = scratch.simple_smote(X, 100, k_neighbors=1)
    assert generated.shape == (100, 2)
    assert ((generated[:, 0] > 2) & (generated[:, 0] < 6)).all()
    np.testing.assert_allclose(generated[:, 1], generated[:, 0] + 2)
    np.testing.assert_array_equal(generated, scratch.simple_smote(X, 100, 1))
    assert not np.array_equal(generated, scratch.simple_smote(X, 100, 1, seed=7))
    np.testing.assert_array_equal(X, original)
    assert scratch.simple_smote(X, 0, 1).shape == (0, 2)


def test_smote_uses_local_neighbors_and_handles_duplicate_coordinates():
    X = np.array([[0.0], [1.0], [100.0], [101.0]])
    generated = scratch.simple_smote(X, 200, 1).ravel()
    assert ((generated <= 1) | (generated >= 100)).all()
    duplicate = scratch.simple_smote([[0.0], [0.0], [10.0]], 200, 1)
    assert np.isfinite(duplicate).all()
    assert ((duplicate >= 0) & (duplicate < 10)).all()


@pytest.mark.parametrize("n,k", [(-1, 1), (1.5, 1), (True, 1), (1, 0), (1, 2), (1, 1.5), (1, True)])
def test_smote_rejects_invalid_counts(n, k):
    with pytest.raises(ValueError):
        scratch.simple_smote([[0.0], [1.0]], n, k)


@pytest.mark.parametrize("X", [[], [1, 2], [[np.nan], [1]], [[np.inf], [1]], [[1]]])
def test_smote_rejects_invalid_geometry(X):
    with pytest.raises(ValueError):
        scratch.simple_smote(X, 1, 1)


@pytest.mark.parametrize("method", ["random_over", "random_under", "smote"])
@pytest.mark.parametrize("minority_label", [0, 1])
def test_resampling_balances_preserves_minority_and_does_not_mutate(method, minority_label):
    X = np.arange(20.0).reshape(10, 2)
    y = np.array([minority_label] * 3 + [1 - minority_label] * 7)
    old_X, old_y = X.copy(), y.copy()
    new_X, new_y = scratch.resample_training(X, y, method, k_neighbors=2)
    size = 3 if method == "random_under" else 7
    np.testing.assert_array_equal(np.bincount(new_y), [size, size])
    assert all(any(np.array_equal(row, candidate) for candidate in new_X) for row in X[:3])
    if method in {"random_over", "random_under"}:
        assert all(any(np.array_equal(row, source) for source in X) for row in new_X)
    else:
        np.testing.assert_array_equal(new_X[:len(X)], X)
        assert (new_y[len(y):] == minority_label).all()
    repeat_X, repeat_y = scratch.resample_training(X, y, method, k_neighbors=2)
    np.testing.assert_array_equal(new_X, repeat_X)
    np.testing.assert_array_equal(new_y, repeat_y)
    np.testing.assert_array_equal(X, old_X)
    np.testing.assert_array_equal(y, old_y)


def test_balanced_data_is_copied():
    X, y = np.array([[1.0], [2.0]]), np.array([0, 1])
    new_X, new_y = scratch.resample_training(X, y, "smote")
    np.testing.assert_array_equal(new_X, X)
    assert not np.shares_memory(new_X, X)
    assert not np.shares_memory(new_y, y)


@pytest.mark.parametrize("X,y,method", [
    ([[0], [1]], [0], "random_over"),
    ([[0], [1]], [0, 0], "random_over"),
    ([[0], [1]], [0, 2], "random_under"),
    ([[0], [1]], [[0], [1]], "smote"),
    ([[0], [1]], [0, 1], "unknown"),
])
def test_resampling_rejects_invalid_inputs(X, y, method):
    with pytest.raises(ValueError):
        scratch.resample_training(X, y, method)


def test_cost_optimum_matches_exhaustive_distinct_label_sets():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.1, 0.3, 0.7, 0.7, 1.0])
    result = scratch.select_cost_threshold(y, p, 2, 5)
    # Midpoints enumerate the same feasible policies independently of scores.
    policies = [0, 0.2, 0.5, 0.85, np.nextafter(1.0, np.inf)]
    costs = []
    for threshold in policies:
        predicted = p >= threshold
        costs.append(2 * sum(predicted & (y == 0)) + 5 * sum(~predicted & (y == 1)))
    assert result["validation_cost"] == min(costs)
    predicted = p >= result["threshold"]
    assert result["fp"] == sum(predicted & (y == 0))
    assert result["fn"] == sum(~predicted & (y == 1))


def test_tied_scores_are_not_split_and_ties_choose_fewest_alerts():
    result = scratch.select_cost_threshold([0, 1], [0.5, 0.5], 1, 1)
    assert result["threshold"] > 0.5
    assert result["validation_cost"] == 1
    assert (result["fp"], result["fn"]) == (0, 1)


def test_all_negative_and_all_positive_policies_are_reachable():
    assert scratch.select_cost_threshold([0, 0], [1.0, 1.0])["threshold"] > 1
    result = scratch.select_cost_threshold([1, 1], [0.0, 1.0])
    assert result["threshold"] == 0
    assert result["validation_cost"] == 0


@pytest.mark.parametrize("y,p,c_fp,c_fn", [
    ([], [], 1, 10), ([0, 2], [0.1, 0.2], 1, 10),
    ([0, 1], [0.1], 1, 10), ([0, 1], [[0.1], [0.2]], 1, 10),
    ([0, 1], [np.nan, 0.2], 1, 10), ([0, 1], [-0.1, 0.2], 1, 10),
    ([0, 1], [0.1, 1.1], 1, 10), ([0, 1], [0.1, 0.2], 0, 10),
    ([0, 1], [0.1, 0.2], 1, -1), ([0, 1], [0.1, 0.2], np.inf, 10),
])
def test_threshold_selection_rejects_invalid_inputs(y, p, c_fp, c_fn):
    with pytest.raises(ValueError):
        scratch.select_cost_threshold(y, p, c_fp, c_fn)


@pytest.mark.parametrize("strategy", example.STRATEGIES)
def test_scaling_fits_original_training_rows_before_resampling(strategy):
    X = np.arange(60.0).reshape(30, 2)
    y = np.array([1] * 8 + [0] * 22)
    original = X.copy()
    scaler, model, counts = example.fit_strategy(X, y, strategy)
    np.testing.assert_allclose(scaler.mean_, X.mean(axis=0))
    assert scaler.n_samples_seen_ == len(X)
    scaler.transform([[1e9, -1e9]])
    np.testing.assert_allclose(scaler.mean_, X.mean(axis=0))
    assert len(model.classes_) == 2
    assert counts == ([8, 8] if strategy == "random_under" else [22, 22] if strategy in {"random_over", "smote"} else [22, 8])
    np.testing.assert_array_equal(X, original)


def test_probability_metrics_do_not_change_with_threshold():
    y, p = [0, 0, 1, 1], [0.1, 0.4, 0.3, 0.9]
    low, high = example.evaluate(y, p, 0.2), example.evaluate(y, p, 0.8)
    for metric in ("average_precision", "roc_auc", "brier"):
        assert low[metric] == high[metric]
    assert low["alert_rate"] > high["alert_rate"]
    assert low["recall"] > high["recall"]
    assert low["cost"] == low["fp"] + 10 * low["fn"]


def test_experiment_selects_all_policies_before_test_evaluation(monkeypatch):
    X = np.column_stack([np.arange(100) / 100, np.arange(100)])
    y = np.array([0] * 80 + [1] * 20)
    train, rest = example.train_test_split(np.arange(100), test_size=0.4, stratify=y, random_state=42)
    validation, test = example.train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=42)
    monkeypatch.setattr(example, "make_classification", lambda **kwargs: (X, y))
    selection_calls = []

    class IdentityScaler:
        def transform(self, values):
            return values

    class FixedScores:
        def predict_proba(self, values):
            p = values[:, 0]
            return np.column_stack([1 - p, p])

    def fit(values, labels, strategy):
        np.testing.assert_array_equal(values, X[train])
        np.testing.assert_array_equal(labels, y[train])
        return IdentityScaler(), FixedScores(), example.class_counts(labels)

    def select(labels, scores, *costs):
        np.testing.assert_array_equal(labels, y[validation])
        np.testing.assert_array_equal(scores, X[validation, 0])
        selection_calls.append(1)
        return scratch.select_cost_threshold(labels, scores, *costs)

    original_evaluate = example.evaluate

    def evaluate(labels, scores, threshold):
        assert len(selection_calls) == len(example.STRATEGIES)
        np.testing.assert_array_equal(labels, y[test])
        if np.any(scores):
            np.testing.assert_array_equal(scores, X[test, 0])
        return original_evaluate(labels, scores, threshold)

    monkeypatch.setattr(example, "fit_strategy", fit)
    monkeypatch.setattr(example, "select_cost_threshold", select)
    monkeypatch.setattr(example, "evaluate", evaluate)
    report = example.run_experiment()
    assert len(report["test_results"]) == 11
    np.testing.assert_array_equal(X[:, 1], np.arange(100))
    np.testing.assert_array_equal(y, [0] * 80 + [1] * 20)


@pytest.mark.parametrize("y,p,t", [
    ([0, 2], [0.1, 0.9], 0.5), ([0, 0], [0.1, 0.9], 0.5),
    ([0, 1], [0.1], 0.5), ([0, 1], [0.1, float("nan")], 0.5),
    ([0, 1], [0.1, 1.1], 0.5), ([0, 1], [0.1, 0.9], float("nan")),
    ([0, 1], [0.1, 0.9], -0.1),
])
def test_evaluation_rejects_invalid_inputs(y, p, t):
    with pytest.raises(ValueError):
        example.evaluate(y, p, t)


def test_training_rejects_nonbinary_labels_and_unknown_strategy():
    with pytest.raises(ValueError):
        example.fit_strategy([[0], [1]], [0, 2], "baseline")
    with pytest.raises(ValueError):
        example.fit_strategy([[0], [1]], [0, 1], "unknown")
