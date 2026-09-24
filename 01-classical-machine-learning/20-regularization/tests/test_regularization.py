"""Analytic optima, independent solver parity, and validation boundaries."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import cross_validate


def load_topic_module(filename, name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load_topic_module("from_scratch.py", "day20_core")
example = load_topic_module("example.py", "day20_example")


def test_soft_threshold_boundary_and_sign():
    np.testing.assert_array_equal(
        core.soft_threshold([-3, -1, 0, 1, 3], 1), [-2, 0, 0, 0, 2]
    )


@pytest.mark.parametrize("ratio", [0, 0.4, 1])
def test_orthogonal_closed_form(ratio):
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    y = 7 + X @ np.array([3.0, 0.1])
    fitted = core.fit_elastic_net(X, y, alpha=0.5, l1_ratio=ratio)
    expected = (
        np.sign([3.0, 0.1])
        * np.maximum(np.abs([3.0, 0.1]) - 0.5 * ratio, 0)
        / (1 + 0.5 * (1 - ratio))
    )
    np.testing.assert_allclose(fitted.coef, expected, atol=1e-8)
    assert fitted.intercept == pytest.approx(7)
    assert fitted.converged


@pytest.mark.parametrize("ratio", [0, 0.4, 1])
def test_matches_sklearn_on_correlated_shifted_data(ratio):
    rng = np.random.default_rng(21)
    X = rng.normal(size=(120, 5)) + 4
    X[:, 4] = X[:, 0] + rng.normal(scale=0.3, size=120)
    y = 9 + X @ np.array([3, 0, -2, 0, 1]) + rng.normal(scale=0.1, size=120)
    alpha = 0.2
    fitted = core.fit_elastic_net(X, y, alpha=alpha, l1_ratio=ratio, tol=1e-9)
    reference = (
        Ridge(alpha=len(X) * alpha, solver="svd") if ratio == 0
        else ElasticNet(alpha=alpha, l1_ratio=ratio, tol=1e-12, max_iter=100_000)
    ).fit(X, y)
    unseen = rng.normal(size=(20, 5)) + 2
    np.testing.assert_allclose(fitted.coef, reference.coef_, atol=1e-7)
    np.testing.assert_allclose(fitted.predict(unseen), reference.predict(unseen), atol=1e-6)
    assert fitted.converged and fitted.kkt_residual <= 1e-9
    assert np.all(np.diff(fitted.objective_history) <= 1e-12)
    assert len(fitted.objective_history) == fitted.n_iter + 1


def test_zero_penalty_matches_ols_predictions_with_duplicate_columns():
    X = np.arange(12.0).reshape(-1, 1)
    X = np.column_stack([X, X, np.ones(12)])
    y = 4 + 3 * X[:, 0]
    fitted = core.fit_elastic_net(X, y, alpha=0)
    reference = LinearRegression().fit(X, y)
    np.testing.assert_allclose(fitted.predict(X), reference.predict(X), atol=1e-7)
    assert fitted.coef[2] == 0


def test_constant_features_and_intercept_are_not_penalized():
    X = np.full((8, 3), 5.0)
    for alpha in [0, 1e6]:
        fitted = core.fit_elastic_net(X, np.arange(8.0), alpha=alpha)
        np.testing.assert_array_equal(fitted.coef, np.zeros(3))
        assert fitted.intercept == 3.5 and fitted.converged


def test_lasso_zero_solution_at_residual_correlation_threshold():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(30, 4))
    y = X[:, 0] * 2 + 10
    alpha_max = np.max(np.abs((X - X.mean(axis=0)).T @ (y - y.mean()))) / len(y)
    fitted = core.fit_elastic_net(X, y, alpha=alpha_max * 1.01, l1_ratio=1)
    np.testing.assert_array_equal(fitted.coef, np.zeros(4))
    assert fitted.n_iter == 0 and fitted.converged


def test_iteration_budget_reports_nonconvergence():
    X = np.array([[1, 0], [0, 1], [2, 3]], dtype=float)
    fitted = core.fit_elastic_net(X, np.array([1, 2, 4]), max_iter=1, tol=1e-14)
    assert not fitted.converged and fitted.n_iter == 1


@pytest.mark.parametrize("kwargs", [
    {"alpha": -1}, {"alpha": np.nan}, {"alpha": np.inf},
    {"l1_ratio": -0.1}, {"l1_ratio": 1.1}, {"l1_ratio": np.nan},
    {"tol": 0}, {"tol": np.inf}, {"max_iter": 0},
    {"max_iter": 1.5}, {"max_iter": True},
])
def test_invalid_optimizer_parameters(kwargs):
    with pytest.raises(ValueError):
        core.fit_elastic_net(np.eye(3), np.ones(3), **kwargs)


@pytest.mark.parametrize("X,y", [
    ([], []), ([1, 2], [1, 2]), (np.empty((3, 0)), np.ones(3)),
    ([[np.nan]], [1]), ([[1]], [np.inf]), ([[1], [2]], [1]),
    ([[1]], [[1]]),
])
def test_invalid_training_data(X, y):
    with pytest.raises(ValueError):
        core.fit_elastic_net(X, y)


def test_invalid_predictions_and_thresholds():
    fitted = core.fit_elastic_net(np.eye(3), np.ones(3))
    for X in [np.ones((2, 2)), [[np.nan, 1, 2]], [1, 2, 3]]:
        with pytest.raises(ValueError):
            fitted.predict(X)
    for threshold in [-1, np.inf, np.nan]:
        with pytest.raises(ValueError):
            core.soft_threshold([1], threshold)
    with pytest.raises(ValueError):
        core.soft_threshold([np.nan], 1)


def test_scaler_is_fitted_separately_inside_each_cv_fold():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(30, 3)) + np.arange(30)[:, None]
    y = X[:, 0] + rng.normal(size=30)
    search = example.make_searches()["Ridge"]
    splits = list(search.cv.split(X, y))
    result = cross_validate(clone(search.estimator), X, y, cv=splits, return_estimator=True)
    for (train, validation), pipeline in zip(splits, result["estimator"]):
        scaler = pipeline.named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, X[train].mean(axis=0))
        assert scaler.n_samples_seen_ == len(train)
        assert not set(train) & set(validation)
        assert not np.allclose(scaler.mean_, X.mean(axis=0))


def test_lasso_endpoint_matches_lasso_estimator():
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    y = X @ [2, 0.1]
    fitted = core.fit_elastic_net(X, y, alpha=0.5, l1_ratio=1)
    reference = Lasso(alpha=0.5).fit(X, y)
    np.testing.assert_allclose(fitted.coef, reference.coef_)
