"""Mathematical contracts for the visual experiments; no screenshot snapshots."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


spec = importlib.util.spec_from_file_location(
    "day20_visual_math", Path(__file__).resolve().parents[1] / "visual_math.py"
)
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)


def test_geometry_is_feasible_and_satisfies_penalized_kkt():
    problem = math.geometry_problem()
    X, y = problem["X"], problem["y"]
    np.testing.assert_allclose(np.linalg.norm(problem["ridge"]), 1, atol=1e-10)
    np.testing.assert_allclose(np.abs(problem["lasso"]).sum(), 1, atol=1e-10)
    assert problem["lasso"][1] == 0
    gradient = X.T @ (X @ problem["ridge"] - y) / len(X)
    np.testing.assert_allclose(gradient + problem["ridge_alpha"] * problem["ridge"], 0, atol=1e-10)
    w, alpha = problem["lasso"], problem["lasso_alpha"]
    gradient = X.T @ (X @ w - y) / len(X)
    np.testing.assert_allclose(gradient[w != 0] + alpha * np.sign(w[w != 0]), 0, atol=1e-10)
    assert np.max(np.abs(gradient[w == 0])) <= alpha


@pytest.mark.parametrize("ratio", [0, 0.5, 1])
def test_surface_matches_explicit_centered_residuals(ratio):
    rng = np.random.default_rng(10)
    X = rng.normal(size=(17, 2)) + 8
    y = rng.normal(size=17) + 11
    b1, b2 = np.meshgrid([-2.0, 0, 3], [-1.0, 2])
    actual = math.objective_grid(X, y, b1, b2, alpha=0.7, l1_ratio=ratio)
    for index in np.ndindex(b1.shape):
        w = np.array([b1[index], b2[index]])
        intercept = y.mean() - X.mean(axis=0) @ w
        expected = np.mean((y - intercept - X @ w)**2) / 2
        expected += 0.7 * ratio * np.abs(w).sum() + 0.7 * (1 - ratio) * (w @ w) / 2
        assert actual[index] == pytest.approx(expected, abs=1e-12)


def test_svd_ridge_path_matches_independent_library_fit():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(35, 4)) * [1, 10, 100, 0.2] + 2
    y = X @ [2, -0.1, 0.02, 3] + rng.normal(size=35)
    evaluation = rng.normal(size=(11, 4))
    alphas = np.array([0.001, 0.3, 20])
    predicted = math.ridge_prediction_path(X, y, evaluation, alphas)
    for i, alpha in enumerate(alphas):
        reference = make_pipeline(StandardScaler(), Ridge(alpha=len(X) * alpha, solver="svd")).fit(X, y)
        np.testing.assert_allclose(predicted[i], reference.predict(evaluation), atol=1e-9)


def test_empirical_bias_variance_identity_and_reproducibility():
    first = math.simulate_bias_variance(seed=9, repeats=4, alphas=[0.001, 0.1, 10])
    second = math.simulate_bias_variance(seed=9, repeats=4, alphas=[0.001, 0.1, 10])
    np.testing.assert_allclose(first["signal_mse"], first["bias2"] + first["variance"], atol=1e-12)
    np.testing.assert_allclose(first["risk"], first["signal_mse"] + first["noise_variance"], atol=1e-12)
    for key in ["bias2", "variance", "risk", "mean_prediction"]:
        np.testing.assert_array_equal(first[key], second[key])


def test_lasso_zero_bound_and_ridge_norm_path():
    data = math.regularization_paths(alphas=[1e-4, 0.01, 0.1, 1, 1000])
    assert data["alpha_max_lasso"] < data["alphas"][-1]
    np.testing.assert_array_equal(data["lasso"][-1], np.zeros(8))
    assert np.all(np.diff(np.linalg.norm(data["ridge"], axis=1)) <= 0)
    assert np.isfinite(data["lasso"]).all()


def test_soft_threshold_zero_interval_and_boundary():
    np.testing.assert_array_equal(math.soft_threshold([-2, -1, 0, 1, 2], 1), [-1, 0, 0, 0, 1])


def test_bootstrap_selection_frequencies_and_reproducibility():
    result = math.analyze_lasso_stability(seed=10, repeats=6)
    again = math.analyze_lasso_stability(seed=10, repeats=6)
    active = result["coefficients"] != 0
    np.testing.assert_allclose(result["frequency"], active.mean(axis=0))
    assert result["categories"].sum() == pytest.approx(1)
    np.testing.assert_array_equal(result["coefficients"], again["coefficients"])


def test_group_frequencies_match_raw_bootstrap_coefficients():
    data = math.elasticnet_grouping(repeats=3)
    coefficients = data["coefficients"]
    np.testing.assert_allclose(data["mean"], coefficients.mean(axis=0))
    np.testing.assert_allclose(data["frequency"], (coefficients != 0).mean(axis=0))
    np.testing.assert_allclose(data["group_all_frequency"], (coefficients[:, :, :3] != 0).all(axis=2).mean(axis=0))


def test_multicollinearity_repeatability_and_shapes():
    first = math.simulate_multicollinearity(seed=6, repeats=3)
    second = math.simulate_multicollinearity(seed=6, repeats=3)
    assert first["coefficients"].shape == (4, 3, 2, 2)
    assert first["rmse"].shape == (4, 3, 2)
    assert np.all(first["rmse"] > 0)
    np.testing.assert_array_equal(first["coefficients"], second["coefficients"])
    np.testing.assert_array_equal(first["rmse"], second["rmse"])


def test_scaling_effects_preserve_unit_conversion():
    result = math.demonstrate_scaling_effect()
    np.testing.assert_allclose(result["unit_effects"], result["raw_coefficients"] * result["scales"])
    np.testing.assert_allclose(result["rmse"], np.sqrt(np.mean((result["predictions"] - result["y_eval"])**2, axis=1)))


@pytest.mark.parametrize("bad", [0, 1, -1, True, 2.5])
def test_invalid_repeat_counts(bad):
    with pytest.raises(ValueError):
        math.simulate_bias_variance(repeats=bad)


@pytest.mark.parametrize("bad", [[], [0], [-1], [np.nan], [[1]]])
def test_invalid_alpha_grids(bad):
    with pytest.raises(ValueError):
        math.regularization_paths(alphas=bad)


def test_invalid_surface_input():
    with pytest.raises(ValueError):
        math.objective_grid(np.ones((3, 3)), np.ones(3), 0, 0)
    with pytest.raises(ValueError):
        math.objective_grid(np.ones((3, 2)), np.ones(3), 0, 0, l1_ratio=2)
    with pytest.raises(ValueError):
        math.soft_threshold([np.inf], 1)
