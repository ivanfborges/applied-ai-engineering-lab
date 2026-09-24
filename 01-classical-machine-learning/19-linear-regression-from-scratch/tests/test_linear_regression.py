"""Numerical contracts checked against analytic fixtures and sklearn."""

import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

spec = importlib.util.spec_from_file_location("day19_regression", Path(__file__).resolve().parents[1] / "from_scratch.py")
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


@pytest.fixture
def data():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(120, 3))
    y = 2 + X @ np.array([3., -1., 0.5]) + rng.normal(scale=0.1, size=120)
    return X, y


def test_ols_and_gd_match_reference_on_new_rows(data):
    X, y = data
    reference = LinearRegression().fit(X, y)
    unseen = np.random.default_rng(20).normal(size=(30, 3))
    for fitted in (core.fit_ols(X, y), core.fit_gradient_descent(X, y)):
        np.testing.assert_allclose(fitted.predict(unseen), reference.predict(unseen), atol=1e-6, rtol=0)
        np.testing.assert_allclose(fitted.beta, np.r_[reference.intercept_, reference.coef_], atol=1e-6, rtol=0)


def test_gradient_matches_central_difference(data):
    X, y = data
    beta = np.array([1., 2., -3., 4.])
    _, gradient = core.loss_gradient(X, y, beta)
    numerical = []
    for direction in np.eye(4) * 1e-5:
        plus = core.loss_gradient(X, y, beta + direction)[0]
        minus = core.loss_gradient(X, y, beta - direction)[0]
        numerical.append((plus - minus) / 2e-5)
    np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=0)


def test_convergence_history_and_budget(data):
    X, y = data
    fit = core.fit_gradient_descent(X, y)
    assert fit.converged
    assert len(fit.loss_history) == fit.n_iter + 1
    assert np.all(np.diff(fit.loss_history) <= 1e-12)
    loss, gradient = core.loss_gradient(X, y, fit.beta)
    assert loss == pytest.approx(fit.loss_history[-1])
    assert np.max(np.abs(gradient)) <= 1e-8
    assert not core.fit_gradient_descent(X, y, max_iter=1).converged
    zero = core.fit_gradient_descent(X, np.zeros(len(X)))
    assert zero.converged and zero.n_iter == 0


def test_exact_line_and_residual_orthogonality(data):
    X = np.arange(10.).reshape(-1, 1)
    np.testing.assert_allclose(core.fit_ols(X, 3 + 2 * X[:, 0]).beta, [3, 2], atol=1e-12)
    X, y = data
    fitted = core.fit_ols(X, y)
    A = np.column_stack((np.ones(len(X)), X))
    np.testing.assert_allclose(A.T @ (y - fitted.predict(X)), 0, atol=1e-11)


@pytest.mark.parametrize("kind", ["duplicate", "constant", "underdetermined"])
def test_rank_deficient_predictions(kind):
    x = np.linspace(-1, 1, 12)
    X = np.column_stack((x, x if kind == "duplicate" else np.ones(12)))
    y = 3 + 2 * x
    if kind == "underdetermined":
        X, y = np.array([[1., 2., 3.], [3., 1., 4.]]), np.array([2., 5.])
    fitted = core.fit_ols(X, y)
    assert fitted.rank < X.shape[1] + 1
    np.testing.assert_allclose(fitted.predict(X), LinearRegression().fit(X, y).predict(X), atol=1e-12)
    gd = core.fit_gradient_descent(X, y, learning_rate=0.01)
    assert gd.converged
    np.testing.assert_allclose(gd.predict(X), fitted.predict(X), atol=1e-6)


@pytest.mark.parametrize("X,y", [([1, 2], [1, 2]), (np.empty((0, 2)), []), (np.empty((2, 0)), [1, 2]), ([[1], [2]], [[1], [2]]), ([[1], [2]], [1]), ([[np.nan]], [1]), ([[1]], [np.inf])])
@pytest.mark.parametrize("solver", [core.fit_ols, core.fit_gradient_descent])
def test_bad_training_data(X, y, solver):
    with pytest.raises(ValueError):
        solver(X, y)


@pytest.mark.parametrize("kwargs", [{"learning_rate": 0}, {"learning_rate": np.inf}, {"learning_rate": "bad"}, {"tol": -1}, {"tol": np.nan}, {"max_iter": 0}, {"max_iter": 1.5}, {"max_iter": True}])
def test_bad_optimizer_settings(data, kwargs):
    with pytest.raises(ValueError):
        core.fit_gradient_descent(*data, **kwargs)


def test_bad_predictions_and_parameters(data):
    fit = core.fit_ols(*data)
    for X in (np.ones((3, 2)), [[np.inf] * 3], [1, 2, 3]):
        with pytest.raises(ValueError):
            fit.predict(X)
    for beta in ([1, 2], [0, 0, 0, np.nan]):
        with pytest.raises(ValueError):
            core.loss_gradient(*data, beta)


def test_divergence_is_explicit():
    with pytest.raises(FloatingPointError):
        core.fit_gradient_descent([[1.], [2.]], [1., 2.], learning_rate=1e200)
