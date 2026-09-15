"""Check residual semantics, OLS geometry, and diagnostic input boundaries."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression


spec = importlib.util.spec_from_file_location(
    "day18_example", Path(__file__).resolve().parents[1] / "example.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_residual_sign_and_no_mutation():
    actual = np.array([2.0, 5.0, -1.0])
    predicted = np.array([1.0, 7.0, -1.0])
    np.testing.assert_array_equal(example.residuals(actual, predicted), [1, -2, 0])
    np.testing.assert_array_equal(actual, [2, 5, -1])
    np.testing.assert_array_equal(predicted, [1, 7, -1])


def test_noiseless_coefficients_and_independent_solver_agree():
    X = np.array([[0, 0], [1, 0], [0, 1], [2, 1], [-1, 2]], dtype=float)
    y = 5 + X @ np.array([3, -2])
    model = LinearRegression().fit(X, y)
    design = np.column_stack((np.ones(len(X)), X))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(beta, [5, 3, -2], atol=1e-12)
    np.testing.assert_allclose(np.r_[model.intercept_, model.coef_], beta, atol=1e-12)


def test_orthogonality_does_not_imply_correct_specification():
    x = np.arange(-3.0, 4.0)
    X = x[:, None]
    y = 3 * x + 2 * x**2
    model = LinearRegression().fit(X, y)
    errors = example.residuals(y, model.predict(X))
    result = example.residual_diagnostics(X, errors)
    assert abs(result["mean_residual"]) < 1e-12
    assert result["max_normalized_inner_product"] < 1e-12
    assert np.linalg.norm(errors) > 1
    assert np.corrcoef(errors, x**2)[0, 1] == pytest.approx(1.0)


def test_orthogonality_is_a_training_constraint():
    X = np.arange(5.0)[:, None]
    model = LinearRegression().fit(X, 2 + 3 * X[:, 0])
    errors = example.residuals(12 + 3 * X[:, 0], model.predict(X))
    assert example.residual_diagnostics(X, errors)["mean_residual"] == pytest.approx(10)


def test_diagnostics_handle_zero_and_duplicate_columns():
    x = np.arange(5.0)
    X = np.column_stack((x, x, np.zeros(5)))
    errors = np.array([1, -2, 2, -2, 1], dtype=float)
    result = example.residual_diagnostics(X, errors)
    assert result["max_normalized_inner_product"] < 1e-12


def test_normalized_diagnostic_is_invariant_to_nonzero_feature_rescaling():
    X = np.array([[0, 2], [2, -1], [4, 3]], dtype=float)
    errors = np.array([1, -2, 3], dtype=float)
    actual = example.residual_diagnostics(X, errors)
    scaled = example.residual_diagnostics(X * [1000, -0.1], errors)
    assert actual == pytest.approx(scaled)


@pytest.mark.parametrize("actual,predicted", [
    ([], []), ([1, 2], [1]), ([[1], [2]], [[1], [2]]),
    ([np.nan], [0]), ([0], [np.inf]), (1, 1),
])
def test_residuals_reject_invalid_inputs(actual, predicted):
    with pytest.raises(ValueError):
        example.residuals(actual, predicted)


@pytest.mark.parametrize("features,errors", [
    ([], []), ([[1], [2]], [1]), ([[1], [2]], [[1], [2]]),
    ([[np.inf]], [1]), ([[1]], [np.nan]), ([1, 2], [1, 2]),
    (np.empty((2, 0)), [1, 2]), (np.empty((0, 2)), []),
])
def test_diagnostics_reject_invalid_inputs(features, errors):
    with pytest.raises(ValueError):
        example.residual_diagnostics(features, errors)
