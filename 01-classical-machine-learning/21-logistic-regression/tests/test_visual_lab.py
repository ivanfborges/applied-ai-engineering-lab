"""Numerical invariants and rendering smoke tests for the visual lab."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load("from_scratch.py", "day21_visual_core")
# Restore only our aliases: clearing all newly imported modules breaks GUI
# library class identities when their lazy imports run later.
previous = {name: sys.modules.get(name) for name in ("from_scratch", "visual_experiments")}
try:
    sys.modules["from_scratch"] = core
    experiments = load("visual_experiments.py", "day21_visual_experiments")
    sys.modules["visual_experiments"] = experiments
    lab = load("logistic_regression_visual_lab.py", "day21_visual_lab")
finally:
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.mark.parametrize("coef", [[1, 0], [0, 1], [1, -2], [-1, 2]])
@pytest.mark.parametrize("threshold", [0.2, 0.5, 0.8])
def test_boundary_segment_satisfies_threshold_equation(coef, threshold):
    segment = experiments.boundary_segment(coef, 0.3, threshold, (-3, 3, -3, 3))
    assert segment.shape == (2, 2)
    assert np.max(np.abs(segment)) <= 3 + 1e-10
    np.testing.assert_allclose(segment @ coef + 0.3, core.logit(threshold), atol=1e-10)


def test_constant_model_and_outside_boundary_return_empty():
    assert experiments.boundary_segment([0, 0], 0, 0.5, (-1, 1, -1, 1)).shape == (0, 2)
    assert experiments.boundary_segment([1, 1], 100, 0.5, (-1, 1, -1, 1)).shape == (0, 2)


def test_fixed_model_threshold_counts_and_raw_parameter_conversion():
    case = experiments.classification_case()
    classifier = case.model.steps[-1][1]
    before = classifier.coef_.copy()
    p = case.model.predict_proba(case.X_test)[:, 1]
    counts = [experiments.decision_counts(case.y_test, p, t) for t in (0.05, 0.5, 0.95)]
    assert counts[0]["positives"] >= counts[1]["positives"] >= counts[2]["positives"]
    for result in counts:
        assert sum(result[k] for k in ("TP", "FP", "TN", "FN")) == len(case.y_test)
    np.testing.assert_array_equal(before, classifier.coef_)
    w, b = experiments.raw_parameters(case.model)
    np.testing.assert_allclose(case.X_test @ w + b, case.model.decision_function(case.X_test))
    np.testing.assert_allclose(case.model.steps[0][1].mean_, case.X_train.mean(axis=0))


def test_gradient_trace_records_real_updates_and_decreases_loss():
    case = experiments.classification_case()
    X = case.model.steps[0][1].transform(case.X_train)
    initial = np.array([-1.0, 0.8, 0.4])
    trace = experiments.gradient_trace(X, case.y_train, initial)
    assert trace["iterations"][0] == 0 and trace["iterations"][-1] == 220
    assert 20 <= len(trace["parameters"]) <= 50
    assert np.all(np.diff(trace["losses"]) <= 1e-12)
    assert trace["losses"][-1] < trace["losses"][0]
    _, dw, db = core.loss_and_gradient(X, case.y_train, initial[:-1], initial[-1])
    np.testing.assert_allclose(trace["parameters"][1], initial - 0.3 * np.r_[dw, db])
    for step, theta in zip(trace["iterations"], trace["parameters"]):
        assert trace["losses"][step] == pytest.approx(core.binary_cross_entropy(
            case.y_train, X @ theta[:-1] + theta[-1]))


def test_nonlinear_models_share_training_rows_and_scale_inside_pipeline():
    case, models, metrics = experiments.nonlinear_experiment()
    for name, model in models.items():
        scaler = model.steps[-2][1]
        features = case.X_train if name == "Raw features" else model.steps[0][1].transform(case.X_train)
        assert scaler.n_samples_seen_ == len(case.X_train)
        np.testing.assert_allclose(scaler.mean_, features.mean(axis=0))
        assert np.isfinite(metrics[name]["log_loss"])


def test_calibration_transform_preserves_ranking_and_bin_support():
    series = list(experiments.calibration_experiment().values())
    np.testing.assert_array_equal(np.argsort(series[0]["probabilities"]), np.argsort(series[1]["probabilities"]))
    assert series[0]["auc"] == series[1]["auc"]
    for result in series:
        assert result["counts"].sum() == 1200
        assert len(result["mean"]) == np.count_nonzero(result["counts"])
        assert np.all((result["fraction"] >= 0) & (result["fraction"] <= 1))


def test_loss_grid_convex_along_each_axis_and_path_inside_grid():
    w, b, loss, trace = experiments.loss_landscape()
    assert np.isfinite(loss).all()
    assert np.min(np.diff(loss, n=2, axis=0)) >= -1e-12
    assert np.min(np.diff(loss, n=2, axis=1)) >= -1e-12
    theta = trace["parameters"]
    assert w.min() <= theta[:, 0].min() <= theta[:, 0].max() <= w.max()
    assert b.min() <= theta[:, 1].min() <= theta[:, 1].max() <= b.max()
    assert np.all(np.diff(trace["losses"]) <= 1e-12)


def test_regularization_results_are_finite_and_shapes_match():
    C, paths = experiments.regularization_experiment()
    assert C.shape == (5,)
    for path in paths.values():
        assert path.shape == (5, 8) and np.isfinite(path).all()
    assert np.count_nonzero(paths["L1"][0] == 0) > 0


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"frames": 1}, {"learning_rate": np.inf}])
def test_trace_rejects_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        experiments.gradient_trace([[-1], [1]], [0, 1], [0, 0], **kwargs)


def test_metrics_reject_nonfinite_values():
    with pytest.raises(ValueError):
        experiments.finite_json({"loss": np.array([1, np.nan])})


def test_static_and_offline_html_rendering(tmp_path):
    from PIL import Image
    lab.style()
    lab.create_pipeline_visualization(tmp_path)
    assert (tmp_path / "12_full_pipeline.png").stat().st_size > 10000
    lab.create_sigmoid_visualization(tmp_path)
    with Image.open(tmp_path / "01_sigmoid.png") as image:
        assert image.width >= 1500 and image.height >= 900
        image.verify()
    lab.create_3d_probability_surface(tmp_path, experiments.classification_case())
    html = (tmp_path / "06_probability_surface_3d.html").read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html and 'src="plotly.min.js"' in html
    assert (tmp_path / "plotly.min.js").stat().st_size > 1_000_000