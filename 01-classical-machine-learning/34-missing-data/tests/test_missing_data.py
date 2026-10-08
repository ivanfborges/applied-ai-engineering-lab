"""Numerical imputation, missingness provenance, and evaluation boundaries."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load_module("day34_scratch", "from_scratch.py")
previous = sys.modules.get("from_scratch")
sys.modules["from_scratch"] = scratch
try:
    example = load_module("day34_example", "example.py")
finally:
    if previous is None:
        del sys.modules["from_scratch"]
    else:
        sys.modules["from_scratch"] = previous


def test_training_medians_indicators_parity_and_no_mutation():
    train = np.array([[1, 10, 0], [2, np.nan, 1], [3, 30, 2], [np.nan, 40, 3]])
    test = np.array([[np.nan, 15, np.nan], [4000, np.nan, 4]])
    original_train, original_test = train.copy(), test.copy()
    imputer = scratch.MedianImputerWithIndicator().fit(train)
    actual = imputer.transform(test)
    np.testing.assert_array_equal(imputer.medians_, [2, 30, 1.5])
    np.testing.assert_array_equal(imputer.indicator_features_, [0, 1])
    np.testing.assert_array_equal(actual, [[2, 15, 1.5, 1, 0], [4000, 30, 4, 0, 1]])
    np.testing.assert_array_equal(actual, SimpleImputer(strategy="median", add_indicator=True).fit(train).transform(test))
    np.testing.assert_array_equal(train, original_train)
    np.testing.assert_array_equal(test, original_test)
    np.testing.assert_array_equal(imputer.medians_, [2, 30, 1.5])


def test_complete_training_and_new_missingness_keep_schema():
    imputer = scratch.MedianImputerWithIndicator().fit([[1, 10], [3, 30]])
    np.testing.assert_array_equal(imputer.transform([[np.nan, np.nan]]), [[2, 20]])
    assert imputer.indicator_features_.size == 0
    imputer.fit([[1, np.nan], [3, 20]])
    np.testing.assert_array_equal(imputer.indicator_features_, [1])
    assert imputer.transform([[np.nan, np.nan]]).shape == (1, 3)


def test_fit_transform_equals_separate_calls():
    X = [[1, np.nan], [3, 20]]
    first = scratch.MedianImputerWithIndicator().fit_transform(X)
    second = scratch.MedianImputerWithIndicator().fit(X).transform(X)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("X", [[], [1, 2], [[]], [[np.inf]], [[-np.inf]], [["text"]], [[[1]]]])
def test_invalid_numerical_inputs(X):
    with pytest.raises(ValueError):
        scratch.MedianImputerWithIndicator().fit(X)


def test_unfitted_wrong_shape_and_empty_training_feature():
    with pytest.raises(RuntimeError, match="fit"):
        scratch.MedianImputerWithIndicator().transform([[1]])
    with pytest.raises(ValueError, match="columns"):
        scratch.MedianImputerWithIndicator().fit([[np.nan, 1], [np.nan, 2]])
    imputer = scratch.MedianImputerWithIndicator().fit([[1, 2]])
    with pytest.raises(ValueError, match="number"):
        imputer.transform([[3]])


def test_seeded_mask_matches_known_mechanisms_and_preserves_values():
    X = np.array([[i, 20, (-1) ** i, i / 20] for i in range(100)], dtype=float)
    original = X.copy()
    seed = 71
    rng = np.random.default_rng(seed)
    expected = X.copy()
    expected[rng.random(len(X)) < 0.15, 0] = np.nan
    expected[rng.random(len(X)) < np.where(X[:, 2] > 0, 0.35, 0.05), 1] = np.nan
    expected[rng.random(len(X)) < np.where(X[:, 3] > 0.8, 0.50, 0.05), 3] = np.nan
    actual = example.inject_missingness(X, seed)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(actual[:, 2], X[:, 2])
    np.testing.assert_array_equal(actual, example.inject_missingness(X, seed))
    assert not np.array_equal(np.isnan(actual), np.isnan(example.inject_missingness(X, seed + 1)))


@pytest.mark.parametrize("X", [[[1, 2, 3]], [[1, 2, 3, np.nan]]])
def test_injection_rejects_incomplete_or_narrow_data(X):
    with pytest.raises(ValueError, match="complete"):
        example.inject_missingness(X, 1)


def test_stress_preserves_rows_existing_missingness_and_input():
    X = np.arange(40, dtype=float).reshape(10, 4)
    X[0, 0] = np.nan
    original = X.copy()
    scenarios = example.stress_scenarios(X, extra_rate=1)
    np.testing.assert_array_equal(scenarios["matched"], X)
    assert np.isnan(scenarios["extra_dropout"][:, [0, 1, 3]]).all()
    np.testing.assert_array_equal(scenarios["extra_dropout"][:, 2], X[:, 2])
    assert np.isnan(scenarios["x2_outage"][:, 2]).all()
    np.testing.assert_array_equal(scenarios["x2_outage"][:, [0, 1, 3]], X[:, [0, 1, 3]])
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(example.stress_scenarios(X, 0)["extra_dropout"], X)
    np.testing.assert_array_equal(
        example.stress_scenarios(X)["extra_dropout"],
        example.stress_scenarios(X)["extra_dropout"],
    )


@pytest.mark.parametrize("rate", [-0.1, 1.1, np.nan, np.inf, True])
def test_stress_rejects_invalid_probability(rate):
    with pytest.raises(ValueError, match="probability"):
        example.stress_scenarios([[1, 2, 3, 4]], rate)


def test_real_pipelines_fit_statistics_only_on_training():
    rng = np.random.default_rng(9)
    train = rng.normal(size=(80, 4))
    train[::4, 0] = np.nan
    labels = np.tile([0, 1], 40)
    validation = rng.normal(loc=1000, size=(20, 4))
    validation[::3, 0] = np.nan
    models = example.build_models()
    with threadpool_limits(limits=1):
        for name, model in models.items():
            if name == "native_hgb":
                model.set_params(max_iter=5)
            else:
                model.set_params(model__n_estimators=5)
            model.fit(train, labels)
            assert np.isfinite(model.predict_proba(validation)).all()
            if name in ("median", "median_indicator"):
                np.testing.assert_allclose(model.named_steps["imputer"].statistics_, np.nanmedian(train, axis=0))
            if name == "scaled_knn":
                scaler = model.named_steps["scaler"]
                np.testing.assert_allclose(scaler.mean_, np.nanmean(train, axis=0))
                frozen_mean = scaler.mean_.copy()
                model.predict_proba(validation * 20)
                np.testing.assert_array_equal(scaler.mean_, frozen_mean)


def test_selection_uses_validation_predictions_and_training_rows(monkeypatch):
    train = np.array([[1], [2], [3], [4]])
    validation = np.array([[50], [60], [70], [80]])
    y_train = np.array([0, 1, 0, 1])
    y_validation = np.array([0, 0, 1, 1])

    class Spy:
        def __init__(self, scores):
            self.scores = np.asarray(scores)

        def fit(self, X, y):
            np.testing.assert_array_equal(X, train)
            np.testing.assert_array_equal(y, y_train)
            return self

        def predict_proba(self, X):
            np.testing.assert_array_equal(X, validation)
            return np.column_stack((1 - self.scores, self.scores))

    candidates = {
        "bad": Spy([0.9, 0.8, 0.2, 0.1]),
        "good": Spy([0.1, 0.2, 0.8, 0.9]),
        "tied": Spy([0.1, 0.2, 0.8, 0.9]),
    }
    monkeypatch.setattr(example, "build_models", lambda: candidates)
    _, scores, selected = example.fit_and_select(train, y_train, validation, y_validation)
    assert scores == {"bad": 0.0, "good": 1.0, "tied": 1.0}
    assert selected == "good"


def test_metrics_single_class_and_empty_groups():
    assert example.classification_metrics([1, 1], [0.5, 0.5]) == {
        "n": 2, "positives": 2, "roc_auc": None, "brier": 0.25
    }

    class ConstantModel:
        def predict_proba(self, X):
            return np.tile([0.5, 0.5], (len(X), 1))

    result = example.evaluate(ConstantModel(), [[np.nan], [np.nan]], [0, 1])
    assert result["groups"]["complete_rows"] is None
    assert result["groups"]["any_missing"]["n"] == 2
    assert result["overall"]["roc_auc"] == 0.5


@pytest.mark.parametrize("y,p", [
    ([], []), ([0], [0.1, 0.2]), ([2], [0.1]),
    ([0], [np.nan]), ([0], [1.1]), ([[0]], [[0.1]]),
])
def test_metrics_reject_invalid_inputs(y, p):
    with pytest.raises(ValueError):
        example.classification_metrics(y, p)
