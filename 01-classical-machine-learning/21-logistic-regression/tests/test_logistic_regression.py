"""Independent gradients, analytic cases, solver parity, and validation checks."""

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / filename,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load_module("from_scratch.py", "day21_core")
with patch.dict(sys.modules, {"from_scratch": core}):
    example = load_module("example.py", "day21_example")


def test_sigmoid_extremes_symmetry_and_logit_inverse():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        p = core.sigmoid(np.array([-1000, -2, 0, 2, 1000]))
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    np.testing.assert_allclose(p + p[::-1], 1)
    assert core.sigmoid(0) == 0.5
    interior = np.array([1e-8, 0.2, 0.5, 0.8, 1 - 1e-8])
    np.testing.assert_allclose(core.sigmoid(core.logit(interior)), interior)


def test_cross_entropy_extremes_and_bernoulli_formula():
    assert core.binary_cross_entropy([0, 1], [0, 0]) == pytest.approx(np.log(2))
    assert core.binary_cross_entropy([1, 0], [-1000, 1000]) == pytest.approx(1000)
    assert core.binary_cross_entropy([0, 1], [-1000, 1000]) == 0
    # Correct confident predictions still retain small positive loss.
    assert core.binary_cross_entropy([1], [40]) == pytest.approx(np.exp(-40), abs=1e-30)
    p = np.array([0.1, 0.7, 0.8])
    y = np.array([0, 1, 0])
    expected = -np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))
    assert core.binary_cross_entropy(y, core.logit(p)) == pytest.approx(expected)


@pytest.mark.parametrize("l2", [0, 0.3])
def test_gradient_against_central_finite_differences(l2):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(13, 3))
    y = rng.integers(0, 2, size=13)
    theta = np.array([0.3, -0.8, 1.1, 0.6])
    _, dw, db = core.loss_and_gradient(X, y, theta[:-1], theta[-1], l2)
    numerical = []
    for j in range(len(theta)):
        offset = np.eye(len(theta))[j] * 1e-6
        plus, minus = theta + offset, theta - offset
        a = core.loss_and_gradient(X, y, plus[:-1], plus[-1], l2)[0]
        b = core.loss_and_gradient(X, y, minus[:-1], minus[-1], l2)[0]
        numerical.append((a - b) / 2e-6)
    np.testing.assert_allclose(np.r_[dw, db], numerical, atol=1e-9)


def test_intercept_only_analytic_optimum_is_not_penalized():
    X, y = np.zeros((8, 2)), np.array([1, 1, 0, 0, 0, 0, 0, 0])
    for l2 in (0, 2):
        fitted = core.fit_logistic(X, y, l2=l2, tol=1e-10)
        assert fitted.converged
        np.testing.assert_array_equal(fitted.coef, [0, 0])
        assert fitted.intercept == pytest.approx(np.log(1 / 3), abs=1e-8)
        np.testing.assert_allclose(fitted.predict_proba(X), 0.25, atol=1e-8)


@pytest.mark.parametrize("l2", [0.01, 0.3])
def test_solver_matches_sklearn_with_same_objective(l2):
    X, y = core.make_synthetic_data(n_samples=250, seed=7)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    fitted = core.fit_logistic(X, y, l2=l2, tol=1e-10)
    reference = LogisticRegression(
        C=1 / (len(X) * l2), solver="lbfgs", tol=1e-12, max_iter=2000,
    ).fit(X, y)
    assert fitted.converged and fitted.gradient_norm <= 1e-10
    np.testing.assert_allclose(fitted.coef, reference.coef_[0], atol=2e-6)
    assert fitted.intercept == pytest.approx(reference.intercept_[0], abs=2e-6)
    unseen = np.random.default_rng(8).normal(size=(20, 2))
    np.testing.assert_allclose(
        fitted.predict_proba(unseen), reference.predict_proba(unseen)[:, 1], atol=1e-6,
    )
    assert np.all(np.diff(fitted.objective_history) <= 1e-14)
    assert len(fitted.objective_history) == fitted.n_iter + 1


def test_iteration_budget_exposes_nonconvergence():
    X, y = core.make_synthetic_data(n_samples=50)
    fitted = core.fit_logistic(X, y, max_iter=1, tol=1e-14)
    assert not fitted.converged and fitted.n_iter == 1


def test_thresholds_have_nested_positive_sets_and_explicit_ties():
    p = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    low, middle, high = (core.classify(p, t) for t in [0.2, 0.5, 0.8])
    assert np.all(high <= middle) and np.all(middle <= low)
    np.testing.assert_array_equal(middle, [0, 0, 1, 1, 1])
    for t in [0.2, 0.5, 0.8]:
        np.testing.assert_array_equal(core.classify(p, t), core.logit(p) >= core.logit(t))


def test_train_only_scaler_and_raw_unit_boundary():
    X, y = core.make_synthetic_data()
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=21,
    )
    pipeline = example.make_pipeline(len(X_train)).fit(X_train, y_train)
    scale, model = pipeline.named_steps["scale"], pipeline.named_steps["classifier"]
    np.testing.assert_allclose(scale.mean_, X_train.mean(axis=0))
    assert scale.n_samples_seen_ == len(X_train)
    assert not np.allclose(scale.mean_, X.mean(axis=0))
    coef = model.coef_[0] / scale.scale_
    intercept = model.intercept_[0] - coef @ scale.mean_
    np.testing.assert_allclose(pipeline.decision_function(X_test), X_test @ coef + intercept)
    np.testing.assert_allclose(
        pipeline.predict_proba(X_test)[:, 1], core.sigmoid(X_test @ coef + intercept),
    )
    original_mean = scale.mean_.copy()
    pipeline.predict_proba(X_test + 1000)
    np.testing.assert_array_equal(scale.mean_, original_mean)


@pytest.mark.parametrize("X,y", [
    ([], []), ([1, 2], [0, 1]), (np.empty((2, 0)), [0, 1]),
    ([[np.nan], [1]], [0, 1]), ([[0], [1]], [0, np.inf]),
    ([[0], [1]], [0]), ([[0], [1]], [[0], [1]]),
    ([[0], [1]], [0, 2]), ([[0], [1]], [1, 1]),
])
def test_invalid_training_data(X, y):
    with pytest.raises(ValueError):
        core.fit_logistic(X, y)


@pytest.mark.parametrize("kwargs", [
    {"l2": -1}, {"l2": np.nan}, {"l2": np.inf}, {"l2": [1]},
    {"tol": 0}, {"tol": np.inf}, {"max_iter": 0},
    {"max_iter": True}, {"max_iter": 1.5},
])
def test_invalid_optimizer_parameters(kwargs):
    with pytest.raises(ValueError):
        core.fit_logistic([[-1], [1]], [0, 1], **kwargs)


@pytest.mark.parametrize("threshold", [0, 1, -0.1, 1.1, np.nan, np.inf, [0.5]])
def test_invalid_thresholds(threshold):
    with pytest.raises(ValueError):
        core.classify([0.2, 0.8], threshold)


def test_invalid_helpers_and_prediction_shapes():
    for p in [0, 1, -1, np.nan]:
        with pytest.raises(ValueError):
            core.logit(p)
    for p in [[], [[0.5]], [1.1], [np.nan]]:
        with pytest.raises(ValueError):
            core.classify(p)
    for y, z in [([0], []), ([0], [[1]]), ([2], [0]), ([0], [np.inf])]:
        with pytest.raises(ValueError):
            core.binary_cross_entropy(y, z)
    with pytest.raises(ValueError):
        core.sigmoid([np.inf])
    fitted = core.fit_logistic([[-1], [1]], [0, 1])
    for X in [[], [1], [[1, 2]], [[np.nan]]]:
        with pytest.raises(ValueError):
            fitted.predict_proba(X)
    with pytest.raises(ValueError):
        core.loss_and_gradient([[-1], [1]], [0, 1], [1, 2], 0)
    with pytest.raises(ValueError):
        core.loss_and_gradient([[-1], [1]], [0, 1], [1], [0])


@pytest.mark.parametrize("n,l2", [(0, 0.1), (True, 0.1), (1.2, 0.1), (2, 0), (2, np.nan)])
def test_invalid_pipeline_parameters(n, l2):
    with pytest.raises(ValueError):
        example.make_pipeline(n, l2)