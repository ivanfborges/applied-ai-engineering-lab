"""Numerical checks for the stump and sequential squared-error correction."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor


TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load_module("day29_scratch", "from_scratch.py")
example = load_module("day29_example", "example.py")
Stump = scratch.DecisionStumpRegressor
Boosting = scratch.SimpleGradientBoostingRegressor


def test_stump_matches_exhaustive_partition_oracle():
    x = np.array([3, 0, 1, 1, 2, 4, 4, 5], dtype=float)
    y = np.array([7, -1, 2, 4, 3, 10, 9, 11], dtype=float)
    minimum = 2
    candidates = [(float(np.sum((y - y.mean()) ** 2)), None)]
    for threshold in np.unique(x)[:-1]:
        left = x <= threshold
        if min(left.sum(), (~left).sum()) >= minimum:
            score = np.sum((y[left] - y[left].mean()) ** 2)
            score += np.sum((y[~left] - y[~left].mean()) ** 2)
            candidates.append((float(score), threshold))
    expected_sse, expected_boundary = min(candidates, key=lambda item: item[0])
    stump = Stump(min_samples_leaf=minimum).fit(x, y)
    prediction = stump.predict(x)
    assert np.sum((y - prediction) ** 2) == pytest.approx(expected_sse)
    np.testing.assert_array_equal(x <= stump.threshold_, x <= expected_boundary)
    assert stump.left_value_ == pytest.approx(y[x <= expected_boundary].mean())
    assert stump.right_value_ == pytest.approx(y[x > expected_boundary].mean())


@pytest.mark.parametrize(
    "x,y,min_leaf",
    [
        ([1, 1, 1], [0, 3, 6], 1),
        ([0, 1, 2], [4, 4, 4], 1),
        ([0, 1, 2], [0, 3, 6], 2),
        ([2], [7], 1),
    ],
)
def test_unsplittable_or_constant_data(x, y, min_leaf):
    stump = Stump(min_samples_leaf=min_leaf).fit(x, y)
    assert stump.threshold_ is None
    np.testing.assert_allclose(stump.predict([-10, 10]), np.mean(y))
    model = Boosting(n_estimators=3, min_samples_leaf=min_leaf).fit(x, y)
    np.testing.assert_allclose(model.predict(x), np.mean(y))


def test_midpoint_boundary_and_adjacent_floating_point_values():
    stump = Stump().fit([0, 0, 2, 2], [1, 1, 5, 5])
    assert stump.threshold_ == 1
    np.testing.assert_array_equal(stump.predict([1, 1.001]), [1, 5])
    low = 1.0
    high = np.nextafter(low, np.inf)
    adjacent = Stump().fit([low, high], [0, 1])
    np.testing.assert_array_equal(adjacent.predict([low, high]), [0, 1])


def test_closed_form_shrinkage_and_independent_stage_snapshots():
    x, y = [0, 1, 2, 3], np.array([1, 1, 5, 5])
    model = Boosting(n_estimators=3, learning_rate=0.5).fit(x, y)
    stages = list(model.staged_predict(x))
    for stage, prediction in enumerate(stages):
        expected = 3 + (1 - 0.5 ** stage) * (y - 3)
        np.testing.assert_allclose(prediction, expected)
    np.testing.assert_allclose(model.train_mse_, [4, 1, 0.25, 0.0625])
    np.testing.assert_array_equal(model.predict(x), stages[-1])
    stages[0][:] = 100
    assert stages[1][0] == 2
    np.testing.assert_array_equal(model.predict(x), [1.25, 1.25, 4.75, 4.75])


def test_each_learner_uses_current_residual_and_training_mse_does_not_increase():
    x = np.arange(20, dtype=float)
    y = np.sin(x) + 0.2 * x
    model = Boosting(n_estimators=15, learning_rate=0.3, min_samples_leaf=2).fit(x, y)
    current = np.full(len(x), y.mean())
    for stump in model.estimators_:
        residual = y - current
        if stump.threshold_ is not None:
            left = x <= stump.threshold_
            assert stump.left_value_ == pytest.approx(residual[left].mean())
            assert stump.right_value_ == pytest.approx(residual[~left].mean())
        current += 0.3 * stump.predict(x)
        assert (y - current).mean() == pytest.approx(0, abs=1e-14)
    assert np.all(np.diff(model.train_mse_) <= 1e-12)


def test_refit_resets_ensemble_and_repeatability():
    x = np.arange(8, dtype=float)
    y = np.array([1, 1, 2, 3, 5, 7, 8, 10], dtype=float)
    model = Boosting(n_estimators=4).fit(x, y)
    original = model.predict(x)
    model.fit(x, y)
    np.testing.assert_array_equal(model.predict(x), original)
    model.fit(x, np.full(8, 6.0))
    assert len(model.estimators_) == 4
    np.testing.assert_array_equal(model.predict(x), np.full(8, 6.0))
    stump = Stump().fit(x, y).fit(x, np.full(8, 6.0))
    assert stump.threshold_ is None


def test_matched_stump_boosting_agrees_with_sklearn_on_controlled_data():
    # Integer features are exactly representable in sklearn's float32 tree input.
    x = np.arange(30, dtype=float).reshape(-1, 1)
    y = 0.1 * x[:, 0] ** 2 + np.random.default_rng(9).normal(0, 0.1, len(x))
    custom = Boosting(n_estimators=12, learning_rate=0.2, min_samples_leaf=2).fit(x, y)
    library = GradientBoostingRegressor(
        n_estimators=12, learning_rate=0.2, max_depth=1,
        min_samples_leaf=2, loss="squared_error", subsample=1.0, random_state=42,
    ).fit(x, y)
    probe = np.arange(-2, 33, dtype=float).reshape(-1, 1)
    for actual, expected in zip(list(custom.staged_predict(probe))[1:], library.staged_predict(probe)):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_estimators": 0}, {"n_estimators": True}, {"n_estimators": 1.5},
        {"min_samples_leaf": 0}, {"min_samples_leaf": False},
        {"learning_rate": 0}, {"learning_rate": -0.1}, {"learning_rate": 1.1},
        {"learning_rate": np.nan}, {"learning_rate": np.inf},
        {"learning_rate": True}, {"learning_rate": "0.1"},
    ],
)
def test_invalid_boosting_parameters(kwargs):
    with pytest.raises(ValueError):
        Boosting(**kwargs)


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5])
def test_invalid_stump_leaf_size(minimum):
    with pytest.raises(ValueError):
        Stump(min_samples_leaf=minimum)


@pytest.mark.parametrize(
    "x,y",
    [
        ([], []), ([0, np.nan], [1, 2]), ([0, np.inf], [1, 2]),
        ([[0, 1], [2, 3]], [1, 2]), ([[[1]]], [1]), ([0, 1], [1]),
        ([0, 1], [[1], [2]]), ([0, 1], [np.nan, 1]), ([0, 1], [1, np.inf]),
    ],
)
def test_invalid_training_data(x, y):
    for model in (Stump(), Boosting()):
        with pytest.raises(ValueError):
            model.fit(x, y)


def test_prediction_errors_and_single_column_shape():
    for model in (Stump(), Boosting()):
        with pytest.raises(RuntimeError):
            model.predict([0])
        model.fit([[0], [1]], [1, 3])
        np.testing.assert_array_equal(model.predict([0, 1]), model.predict([[0], [1]]))
        for invalid in ([], [np.nan], [[0, 1]]):
            with pytest.raises(ValueError):
                model.predict(invalid)


def test_experiment_is_repeatable_uses_disjoint_splits_and_validation_selection():
    record = example.run_experiment()
    assert record == example.run_experiment()
    split = record["configuration"]["split_indices"]
    train, validation, test = (set(split[name]) for name in ("train", "validation", "test"))
    assert not (train & validation or train & test or validation & test)
    assert train | validation | test == set(range(300))
    comparisons = record["results"]["learning_rate_comparison"]
    expected = min(comparisons, key=lambda item: min(item["validation_rmse_by_stage"]))
    assert record["results"]["selected_learning_rate"] == expected["learning_rate"]
    assert record["results"]["selected_stage"] == int(np.argmin(expected["validation_rmse_by_stage"]))
    assert all(len(item["validation_rmse_by_stage"]) == 301 for item in comparisons)

def test_test_targets_cannot_change_validation_selection(monkeypatch):
    record = example.run_experiment()
    x, y = example.make_data()
    changed_y = y.copy()
    test_indices = record["configuration"]["split_indices"]["test"]
    changed_y[test_indices] += 1000
    monkeypatch.setattr(example, "make_data", lambda: (x.copy(), changed_y.copy()))
    changed = example.run_experiment()
    assert changed["results"]["learning_rate_comparison"] == record["results"]["learning_rate_comparison"]
    assert changed["results"]["selected_stage"] == record["results"]["selected_stage"]
    assert changed["results"]["selected_learning_rate"] == record["results"]["selected_learning_rate"]
    assert changed["results"]["selected_test_rmse"] > 900