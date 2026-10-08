"""Known-function, permutation semantics, library agreement, and SHAP checks."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance as sklearn_permutation
from sklearn.linear_model import LinearRegression

TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load_module("day35_scratch", "from_scratch.py")
previous = sys.modules.get("from_scratch")
sys.modules["from_scratch"] = scratch
try:
    example = load_module("day35_example", "example.py")
finally:
    if previous is None:
        del sys.modules["from_scratch"]
    else:
        sys.modules["from_scratch"] = previous


def test_permutation_known_signal_unused_feature_determinism_and_no_mutation():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(100, 2))
    y = 2 * X[:, 0]
    original_X, original_y = X.copy(), y.copy()
    model = LinearRegression().fit(X, y)
    coef = model.coef_.copy()
    kwargs = dict(scoring="neg_mean_squared_error", repeats=50, seed=9)
    result = scratch.permutation_importance(model, X, y, **kwargs)
    repeated = scratch.permutation_importance(model, X, y, **kwargs)
    np.testing.assert_array_equal(result["importances"], repeated["importances"])
    np.testing.assert_allclose(result["importances_mean"][1], 0, atol=1e-20)
    assert result["importances_mean"][0] > 1
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(model.coef_, coef)
    np.testing.assert_allclose(result["importances_std"], result["importances"].std(axis=1))


def test_permutation_library_monte_carlo_agreement():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 2))
    y = 2 * X[:, 0] + 0.5 * X[:, 1]
    model = LinearRegression().fit(X, y)
    own = scratch.permutation_importance(model, X, y, scoring="neg_mean_squared_error", repeats=400)
    library = sklearn_permutation(
        model, X, y, scoring="neg_mean_squared_error", n_repeats=400, random_state=35
    )
    # Random streams differ; compare the estimated expectation within MC error.
    standard_error = np.sqrt(
        (own["importances_std"] ** 2 + library.importances_std ** 2) / 400
    )
    assert np.all(np.abs(own["importances_mean"] - library.importances_mean) < 4 * standard_error)


def test_group_shuffle_preserves_within_group_relationship():
    X = np.column_stack([np.arange(20), 3 * np.arange(20), np.ones(20)])
    original = X.copy()

    def support_score(model, rows, y):
        return float(np.mean(rows[:, 1] == 3 * rows[:, 0]))

    result = scratch.permutation_importance(
        LinearRegression(), X, np.zeros(20), scoring=support_score,
        groups=[(0,), (0, 1)], repeats=8
    )
    assert result["importances_mean"][0] > 0
    np.testing.assert_array_equal(result["importances"][1], 0)
    np.testing.assert_array_equal(X, original)


def test_negative_importance_is_retained():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    result = scratch.permutation_importance(
        LinearRegression(), X, np.zeros(12),
        scoring=lambda model, rows, y: -float(np.mean(rows[:, 0] == X[:, 0])),
        repeats=5
    )
    assert np.all(result["importances"] < 0)


@pytest.mark.parametrize("kwargs", [
    {"repeats": 1}, {"repeats": True}, {"repeats": 2.5}, {"groups": []},
    {"groups": [()]}, {"groups": [(0, 0)]}, {"groups": [(2,)]},
    {"groups": [(-1,)]}, {"groups": [(0.5,)]}, {"groups": [(True,)]},
])
def test_invalid_permutation_configuration(kwargs):
    with pytest.raises(ValueError):
        scratch.permutation_importance(
            LinearRegression(), np.ones((5, 2)), np.zeros(5),
            scoring=lambda model, X, y: 1.0, **kwargs
        )


@pytest.mark.parametrize("X,y", [
    ([], []), ([[1, np.nan], [2, 3]], [0, 1]),
    ([[1], [2]], [0]), ([[1], [2]], [[0], [1]]),
])
def test_invalid_permutation_data(X, y):
    with pytest.raises(ValueError):
        scratch.permutation_importance(
            LinearRegression(), X, y, scoring=lambda model, X, y: 1.0
        )


def test_nonfinite_score_rejected():
    with pytest.raises(ValueError, match="finite"):
        scratch.permutation_importance(
            LinearRegression(), np.ones((3, 2)), np.zeros(3),
            scoring=lambda model, X, y: np.nan
        )


def test_known_interaction_pdp_is_ice_mean_and_inputs_unchanged():
    X = np.array([[1, 2], [3, -1], [2, 4]], dtype=float)
    original = X.copy()
    grid = np.array([-2, 0, 2], dtype=float)
    curves = scratch.pdp_ice(lambda rows: rows[:, 0] * rows[:, 1], X, 0, grid)
    np.testing.assert_array_equal(curves["ice"], X[:, 1, None] * grid)
    np.testing.assert_allclose(curves["pdp"], grid * X[:, 1].mean())
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(grid, [-2, 0, 2])


def test_pdp_ice_library_agreement():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(40, 2))
    model = RandomForestRegressor(n_estimators=8, random_state=6).fit(
        X, X[:, 0] * X[:, 1]
    )
    library = partial_dependence(model, X, [0], method="brute", kind="both", grid_resolution=7)
    own = scratch.pdp_ice(model.predict, X, 0, library["grid_values"][0])
    np.testing.assert_allclose(own["ice"], library["individual"][0])
    np.testing.assert_allclose(own["pdp"], library["average"][0])


@pytest.mark.parametrize("feature,grid", [
    (-1, [0]), (2, [0]), (0.5, [0]), (True, [0]),
    (0, []), (0, [[1]]), (0, [np.inf]),
])
def test_invalid_response_inputs(feature, grid):
    with pytest.raises(ValueError):
        scratch.pdp_ice(lambda rows: rows[:, 0], np.ones((3, 2)), feature, grid)


@pytest.mark.parametrize("predict", [
    lambda rows: np.zeros((len(rows), 2)), lambda rows: np.nan,
    lambda rows: np.full(len(rows), np.inf),
])
def test_invalid_response_output(predict):
    with pytest.raises(ValueError, match="scalar"):
        scratch.pdp_ice(predict, np.ones((3, 2)), 0, [0])


def test_class_lookup_handles_reordered_labels():
    model = SimpleNamespace(classes_=np.array([1, 0]))
    assert example.class_index(model) == 0
    with pytest.raises(ValueError):
        example.class_index(model, 7)


def test_generator_reproducible_and_proxy_has_intended_correlation():
    X, y = example.make_data()
    repeated_X, repeated_y = example.make_data()
    np.testing.assert_array_equal(X, repeated_X)
    np.testing.assert_array_equal(y, repeated_y)
    assert np.corrcoef(X[:, :2].T)[0, 1] > 0.97
    assert set(y) == {0, 1}


@pytest.mark.parametrize("kwargs", [
    {"n": 10}, {"n": True}, {"n": 20.5},
    {"proxy_noise": -1}, {"proxy_noise": np.nan}, {"interaction": np.inf},
])
def test_invalid_generator_parameters(kwargs):
    with pytest.raises(ValueError):
        example.make_data(**kwargs)


def test_shap_probability_reconstruction_background_and_class_selection():
    pytest.importorskip("shap")
    X, y = example.make_data(160)
    model = RandomForestClassifier(
        n_estimators=8, min_samples_leaf=3, random_state=35, n_jobs=1
    ).fit(X[:100], y[:100])
    for label in (0, 1):
        result = example.explain_probability(model, X[:25], X[100:105], label=label)
        index = example.class_index(model, label)
        first = result["first_row"]
        np.testing.assert_allclose(
            first["baseline"] + sum(first["contributions"]),
            model.predict_proba(X[100:101])[0, index], atol=1e-6
        )
        assert result["max_reconstruction_error"] < 1e-6
        np.testing.assert_allclose(
            result["background_prediction_mean"], model.predict_proba(X[:25])[:, index].mean()
        )
    with pytest.raises(ValueError):
        example.explain_probability(model, X[:101], X[100:105])
    with pytest.raises(ValueError):
        example.explain_probability(model, X[:25], np.ones((2, 3)))