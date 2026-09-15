"""Numerical invariants and rendering smoke tests for every Day 18 visual view."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))
import regression_math as rm
import regression_charts as rc


def test_fit_matches_library_and_projection_identities():
    X, y, _ = rm.correlated_data(rho=.9)
    result = rm.fit_ols(X, y)
    model = LinearRegression().fit(X, y)
    np.testing.assert_allclose(result["pred"], model.predict(X), atol=1e-11)
    np.testing.assert_allclose(result["design"].T@result["residual"], 0, atol=1e-10)
    assert abs(result["residual"].sum()) < 1e-10


def test_rank_deficient_fit_still_projects():
    x = np.arange(5.)
    result = rm.fit_ols(np.column_stack((x, x)), 3+2*x)
    assert result["rank"] == 2
    np.testing.assert_allclose(result["residual"], 0, atol=1e-12)


def test_loss_grid_agrees_with_direct_rss():
    x, y, _ = rm.line_data()
    a, b = [-2., 0., 1.], [-1., 3.]
    grid = rm.loss_grid(x, y, a, b)
    for i, slope in enumerate(b):
        for j, intercept in enumerate(a):
            assert grid[i, j] == pytest.approx(np.sum((y-intercept-slope*x)**2))


def test_stable_descent_decreases_mse_and_reaches_ols():
    x, y, _ = rm.line_data()
    result = rm.gradient_descent(x, y, steps=150)
    assert np.all(np.diff(result["losses"]) <= 1e-10)
    np.testing.assert_allclose(result["path"][-1], rm.fit_ols(x, y)["beta"], atol=1e-6)


def test_unstable_descent_remains_bounded_and_reports_stop():
    x, y, _ = rm.line_data()
    result = rm.gradient_descent(x, y, rate_ratio=1.3, steps=150)
    assert np.isfinite(result["path"]).all()
    assert result["losses"][-1] > result["losses"][0]
    assert "divergence" in result["status"]


def test_projection_identity():
    r = rm.projection()
    np.testing.assert_allclose(r["y"], r["pred"]+r["residual"])
    np.testing.assert_allclose(r["design"].T@r["residual"], 0, atol=1e-12)
    assert np.dot(r["pred"], r["residual"]) == pytest.approx(0, abs=1e-12)


def test_ridge_original_unit_coefficients_reconstruct_predictions():
    X, y, _ = rm.correlated_data()
    model, beta = rm.fitted_model(X, y, 3.)
    np.testing.assert_allclose(model.predict(X), rm.design_matrix(X)@beta, atol=1e-12)


def test_finite_ensemble_bias_variance_identity_and_alpha_zero():
    r = rm.stability(repeats=8, alpha=0)
    np.testing.assert_allclose(r["mean_error"], r["bias2"]+r["variance"], atol=1e-12)
    np.testing.assert_allclose(r["predictions"][:, 0], r["predictions"][:, 1], atol=1e-10)


def test_local_rng_reproducibility_and_state_isolation():
    np.random.seed(7)
    expected = np.random.random(4)
    np.random.seed(7)
    a = rm.stability(repeats=5)
    b = rm.stability(repeats=5)
    np.testing.assert_array_equal(a["beta"], b["beta"])
    np.testing.assert_array_equal(np.random.random(4), expected)


def test_curvature_training_rss_and_orthogonality():
    x, y, linear, quad = rm.curvature(noise=0)
    assert rm.scores(y, linear["pred"])["RSS"] > 1
    np.testing.assert_allclose(quad["residual"], 0, atol=1e-12)
    np.testing.assert_allclose(linear["design"].T@linear["residual"], 0, atol=1e-10)


def test_zero_heteroskedasticity_growth_is_identical():
    _, a, b, sigma = rm.variance_data(strength=0)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(sigma, .3)


def test_leverage_range_and_zero_offset_noiseless_fit():
    r = rm.outlier_case(noise=0, offset=0, point_x=30)
    assert 0 < r["leverage"] < 1
    np.testing.assert_allclose(r["before"]["beta"], r["after"]["beta"], atol=1e-12)


def test_polynomial_nested_training_error_and_validation_is_disjoint():
    r = rm.polynomial_experiment()
    assert not np.intersect1d(r["train"], r["valid"]).size
    assert np.all(np.diff(r["errors"][:, 0]) <= 1e-7)
    assert np.isfinite(r["curves"]).all()


def test_well_conditioned_normal_equations_agree():
    r = rm.normal_equations()
    np.testing.assert_allclose(r["normal"], r["stable"], atol=1e-12)
    np.testing.assert_allclose(r["library"], r["stable"], atol=1e-12)


def test_constant_target_r2_is_explicitly_undefined():
    assert np.isnan(rm.scores(np.ones(5), np.ones(5))["R2"])


@pytest.mark.parametrize("call", [
    lambda: rm.line_data(n=2), lambda: rm.line_data(n=True),
    lambda: rm.line_data(noise=-1), lambda: rm.line_data(domain=(1, 1)),
    lambda: rm.correlated_data(rho=1), lambda: rm.generator(-1),
    lambda: rm.fit_ols([], []), lambda: rm.fit_ols([1, 2], [1]),
    lambda: rm.fit_ols([1, np.nan], [1, 2]),
    lambda: rm.gradient_descent([1, 2], [1, 2], start=[0]),
    lambda: rm.gradient_descent([1, 2], [1, 2], rate_ratio=0),
    lambda: rm.projection([1, 2]), lambda: rm.stability(repeats=1),
    lambda: rm.normal_equations(0),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_plotly_animation_links_both_panels():
    x, y, _ = rm.line_data()
    fig = rc.descent_animation(x, y, rm.gradient_descent(x, y))
    assert len(fig.frames) > 2
    assert list(fig.frames[0].traces) == [1, 4]
    fig.to_json()


def test_all_twenty_app_views_render():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(TOPIC/"linear_regression_visual_lab.py"), default_timeout=45)
    app.run()
    assert not app.exception
    options = list(app.sidebar.selectbox(key="view").options)
    assert len(options) == 20
    for option in options:
        app.sidebar.selectbox(key="view").select(option).run()
        assert not app.exception, (option, [e.message for e in app.exception])
        assert any("Interview takeaway" in v.value for v in app.info)
        assert app.get("plotly_chart") or option.startswith("20")


def test_interactive_boundaries():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(TOPIC/"linear_regression_visual_lab.py"), default_timeout=45).run()
    app.sidebar.slider(key="noise").set_value(0.).run()
    app.sidebar.slider(key="slope").set_value(0.).run()
    assert not app.exception
    assert any(m.value == "Undefined" for m in app.metric)
    options = list(app.sidebar.selectbox(key="view").options)
    app.sidebar.selectbox(key="view").select(options[4]).run()
    app.sidebar.slider(key="rate").set_value(1.3).run()
    assert not app.exception
    assert app.warning
    app.sidebar.selectbox(key="view").select(options[14]).run()
    app.sidebar.slider(key="alpha").set_value(0.).run()
    assert not app.exception
    app.sidebar.selectbox(key="view").select(options[19]).run()
    app.sidebar.slider(key="separation").set_value(-10.).run()
    assert not app.exception