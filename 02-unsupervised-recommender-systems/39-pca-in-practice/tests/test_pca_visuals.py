"""Projection geometry, preprocessing boundaries and actual offline export smoke."""

import importlib.util
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image
import pytest
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load_module("day39_visual_core", "pca_visual_core.py")
previous = sys.modules.get("pca_visual_core")
sys.modules["pca_visual_core"] = core
try:
    render = load_module("day39_visuals", "pca_visualizations.py")
finally:
    if previous is None:
        del sys.modules["pca_visual_core"]
    else:
        sys.modules["pca_visual_core"] = previous
cli = load_module("day39_visual_cli", "visualize_pca.py")


@pytest.mark.parametrize("standardize", [False, True])
def test_orthogonality_variance_tail_loss_and_full_reconstruction(standardize):
    view = core.fit_view(core.plane_data(), standardize)
    model = view["model"]
    np.testing.assert_allclose(model.components_ @ model.components_.T, np.eye(3), atol=1e-12)
    assert model.explained_variance_ratio_.sum() == pytest.approx(1)
    states = [core.reconstruct(view, k) for k in (1, 2, 3)]
    np.testing.assert_allclose(states[-1]["Xhat"], view["X"], atol=1e-12)
    assert np.all(np.diff([s["mse"] for s in states]) <= 1e-12)
    for state in states:
        loss = (len(view["X"])-1) * model.explained_variance_[state["k"]:].sum() / view["X"].size
        assert state["mse"] == pytest.approx(loss, abs=1e-12)
        residual = view["working"] - state["working_hat"]
        np.testing.assert_allclose(residual @ model.components_[:state["k"]].T, 0, atol=1e-12)
        if standardize:
            np.testing.assert_allclose(state["Xhat"], view["scaler"].inverse_transform(state["working_hat"]))


def test_plane_projection_uses_correct_affine_mean_and_perpendicular_residual():
    view = core.fit_view(core.plane_data())
    model = view["model"]
    state = core.reconstruct(view, 2)
    centered = view["X"] - model.mean_
    projector = model.components_[:2].T @ model.components_[:2]
    np.testing.assert_allclose(state["Xhat"], centered @ projector + model.mean_)
    np.testing.assert_allclose((state["Xhat"]-model.mean_) @ model.components_[2], 0, atol=1e-12)
    np.testing.assert_allclose((view["X"]-state["Xhat"]) @ model.components_[:2].T, 0, atol=1e-12)


def test_rotating_basis_changes_coordinates_but_not_original_points():
    X = core.geometry_data()
    original = X.copy()
    centered = X-X.mean(axis=0)
    for angle in (0, .4, 1.2):
        basis = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        np.testing.assert_allclose((centered @ basis.T) @ basis, centered, atol=1e-12)
    np.testing.assert_array_equal(X, original)
    view = core.fit_view(X)
    maximum = view["model"].components_[0]
    angle = np.arctan2(maximum[1], maximum[0])
    assert core.direction_variance(X, angle) == pytest.approx(view["model"].explained_variance_[0])


def test_covariance_ellipse_has_constant_mahalanobis_radius():
    view = core.fit_view(core.geometry_data())
    ellipse = core.covariance_ellipse(view)
    scores = (ellipse-view["model"].mean_) @ view["model"].components_.T
    np.testing.assert_allclose(np.sum(scores**2/view["model"].explained_variance_, axis=1), 4, atol=1e-12)


def test_threshold_agrees_with_sklearn_float_selection():
    view = core.fit_view(core.spectrum_data())
    selected = core.threshold_count(view)
    reference = PCA(.95, svd_solver="full").fit(view["X"])
    assert selected == reference.n_components_
    cumulative = np.cumsum(view["model"].explained_variance_ratio_)
    assert cumulative[selected-1] > .95
    if selected > 1:
        assert cumulative[selected-2] <= .95


def test_scaling_reuses_exact_rows_but_changes_directions():
    raw, scaled = core.scaling_views()
    np.testing.assert_array_equal(raw["X"], scaled["X"])
    np.testing.assert_allclose(scaled["working"].var(axis=0), [1, 1])
    alignment = abs(raw["model"].components_[0] @ scaled["model"].components_[0])
    assert alignment < .85  # Controlled different-unit mechanism, not a general bound.


def test_predictive_models_fit_only_training_rows_and_scores_are_recomputed():
    with threadpool_limits(limits=1):
        experiment = core.predictive_experiment()
    train, test, X = experiment["train"], experiment["test"], experiment["X"]
    assert not set(train) & set(test)
    from sklearn.metrics import balanced_accuracy_score
    for name, model in experiment["models"].items():
        first = model.steps[0][1]
        np.testing.assert_allclose(first.mean_, X[train].mean(axis=0))
        assert experiment["balanced_accuracy"][name] == pytest.approx(
            balanced_accuracy_score(experiment["y"][test], model.predict(X[test])))
    # Test labels are not consulted by the separate fitted PCA view.
    np.testing.assert_allclose(experiment["view"]["model"].mean_, X[train].mean(axis=0))


def test_outlier_states_are_actual_augmented_fits():
    clean = core.geometry_data()
    states = core.outlier_states()
    assert len(states) == 5
    for view, count in states:
        np.testing.assert_array_equal(view["X"][:len(clean)], clean)
        assert len(view["X"]) == len(clean) + count
        np.testing.assert_allclose(view["model"].mean_, view["X"].mean(axis=0))
    alignment = abs(states[0][0]["model"].components_[0] @ states[-1][0]["model"].components_[0])
    assert alignment < .9


def test_signed_wine_weights_and_sign_flip_reconstruction_invariance():
    view, features = core.wine_weights()
    model = view["model"]
    assert view["X"].shape == (133, 13)
    assert len(features) == 13
    before = core.reconstruct(view, 2)["Xhat"]
    model.components_[0] *= -1
    view["scores"][:, 0] *= -1
    np.testing.assert_allclose(before, core.reconstruct(view, 2)["Xhat"], atol=1e-12)


def test_plotly_slider_masks_metrics_and_reconstruction_coordinates():
    fig, records = render.interactive_3d_figure()
    view = core.fit_view(core.plane_data())
    for i, step in enumerate(fig.layout.sliders[0].steps):
        mask, layout = step.args
        assert len(mask["visible"]) == len(fig.data)
        assert mask["visible"][5] == (i == 1)  # plane shown precisely at k=2
        state = core.reconstruct(view, i+1)
        trace = fig.data[6+2*i]
        np.testing.assert_allclose(np.column_stack([trace.x, trace.y, trace.z]), state["Xhat"])
        assert mask["visible"][6+2*i]
        assert sum(mask["visible"]) == (8 if i == 1 else 7)
        assert records[i]["mse"] == pytest.approx(state["mse"])
        assert f"k={i+1}" in layout["title.text"]


def test_actual_static_and_offline_html_exports_in_temporary_directory(tmp_path):
    render.style()
    render.scaling(tmp_path, fast=True)
    render.interactive_3d(tmp_path)
    with Image.open(tmp_path / "05_scaling_comparison.png") as image:
        image.verify()
    text = (tmp_path / "03_pca_3d_interactive.html").read_text(encoding="utf-8")
    assert "plotly.js" in text
    assert not re.search(r'<script[^>]+src=["\']https?://', text)
    assert 'id="pca-3d"' in text
    assert "Retained components" in text


@pytest.mark.parametrize("k", [0, 4, True, 1.5])
def test_invalid_retained_dimensions(k):
    with pytest.raises(ValueError):
        core.reconstruct(core.fit_view(core.plane_data()), k)


@pytest.mark.parametrize("value", [0, 1, np.nan, -1])
def test_invalid_variance_threshold(value):
    with pytest.raises(ValueError):
        core.threshold_count(core.fit_view(core.spectrum_data()), value)


def test_list_cli_has_every_supported_demo_without_rendering(capsys):
    assert cli.main(["--list"]) == 0
    output = capsys.readouterr().out
    for demo in cli.DEMOS:
        assert demo in output
