"""Numerical and CLI contracts for the visual laboratory."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))
from visual_math import (
    design, gradient_trace, linear_data, loss_surface, mse_gradient, scaling_data,
)


def test_surface_orientation_and_direct_loss():
    X, y = linear_data()
    A = design(X)
    first, second = np.array([-2., 0., 3.]), np.array([-1., 2.])
    B0, B1, losses = loss_surface(A, y, first, second)
    assert losses.shape == (2, 3)
    for row in range(2):
        for col in range(3):
            assert losses[row, col] == pytest.approx(
                np.mean((A @ [B0[row, col], B1[row, col]] - y) ** 2)
            )


def test_mse_gradient_matches_finite_difference():
    X, y = linear_data()
    A = design(X)
    beta = np.array([-1., 0.3])
    loss, gradient = mse_gradient(A, y, beta)
    assert loss == pytest.approx(np.mean((A @ beta - y) ** 2))
    for index, delta in enumerate(np.eye(2) * 1e-5):
        numerical = (mse_gradient(A, y, beta + delta)[0] -
                     mse_gradient(A, y, beta - delta)[0]) / 2e-5
        assert gradient[index] == pytest.approx(numerical, abs=1e-8)


def test_trace_converges_and_records_returned_iterates():
    X, y = linear_data()
    A = design(X)
    trace = gradient_trace(A, y, [-2.5, -0.8], .08, steps=300)
    assert trace.status == "converged"
    assert trace.parameters.shape == (len(trace.losses), 2)
    np.testing.assert_allclose(trace.losses, [np.mean((A @ b - y) ** 2) for b in trace.parameters])
    assert np.max(np.diff(trace.losses)) < 1e-12
    np.testing.assert_allclose(A @ trace.parameters[-1], A @ np.linalg.lstsq(A, y, rcond=None)[0], atol=1e-8)
    assert gradient_trace(A, y, [0, 0], .001, steps=1).status == "budget exhausted"


@pytest.mark.parametrize("rate", [.5, 1e200])
def test_divergence_retains_only_finite_safe_values(rate):
    X, y = linear_data()
    trace = gradient_trace(design(X), y, [-2.5, -.8], rate)
    assert trace.status.startswith("diverged")
    assert np.isfinite(trace.parameters).all()
    assert np.isfinite(trace.losses).all()
    assert (trace.losses <= 1e10).all()


def test_centering_profiles_intercept_and_scaling_preserves_predictions():
    raw, scaled, y, scaler = scaling_data()
    raw_w = np.linalg.lstsq(raw, y, rcond=None)[0]
    scaled_w = np.linalg.lstsq(scaled, y, rcond=None)[0]
    np.testing.assert_allclose(raw_w, scaled_w / scaler.scale_, atol=1e-10)
    np.testing.assert_allclose(raw @ raw_w, scaled @ scaled_w, atol=1e-10)
    np.testing.assert_allclose(raw.mean(axis=0), 0, atol=1e-11)
    assert abs(y.mean()) < 1e-12
    assert np.linalg.cond(raw.T @ raw) > 1e6
    assert np.linalg.cond(scaled.T @ scaled) < 10
    # Contours are exact ellipses: their MSE excess equals the chosen level.
    gram = raw.T @ raw / len(y)
    values, vectors = np.linalg.eigh(gram)
    offset = vectors @ (np.sqrt(4 / values) * np.array([.6, .8]))
    minimum = np.mean((raw @ raw_w - y) ** 2)
    assert np.mean((raw @ (raw_w + offset) - y) ** 2) - minimum == pytest.approx(4)


@pytest.mark.parametrize("kwargs", [{"rate": 0}, {"rate": np.inf}, {"steps": 0}, {"steps": 1.5}, {"tol": -1}, {"loss_limit": np.nan}])
def test_invalid_trace_options(kwargs):
    options = {"rate": .1, **kwargs}
    with pytest.raises(ValueError):
        gradient_trace(np.eye(2), np.ones(2), [0, 0], **options)


def test_invalid_dimensions_and_initial_loss():
    with pytest.raises(ValueError):
        gradient_trace(np.eye(2), [1], [0, 0], .1)
    with pytest.raises(ValueError):
        gradient_trace(np.eye(2), [1, 2], [0], .1)
    with pytest.raises(ValueError):
        gradient_trace(np.eye(2), [1, 2], [0, 0], .1, loss_limit=.1)
    with pytest.raises(ValueError):
        loss_surface(np.eye(3), np.ones(3), [0, 1], [0, 1])
    with pytest.raises(ValueError):
        loss_surface(np.eye(2), np.ones(2), [0, np.nan], [0, 1])


def test_cli_selects_one_experiment_and_preserves_unrelated_file(tmp_path):
    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_text("keep", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(TOPIC / "visual_experiments.py"), "--experiment", "residuals",
         "--output-dir", str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert (tmp_path / "01_residuals.png").stat().st_size > 1000
    assert sentinel.read_text(encoding="utf-8") == "keep"
    report = json.loads((tmp_path / "results_residuals.json").read_text(encoding="utf-8"))
    assert list(report["experiments"]) == ["residuals"]
    assert report["experiments"]["residuals"]["result"]["mse"] > 0
    assert "Recommended public previews" in result.stdout
    assert not (tmp_path / "04_gradient_descent_path.gif").exists()


def test_plotly_missing_uses_matplotlib_fallback(tmp_path, monkeypatch):
    import builtins
    spec = importlib.util.spec_from_file_location("day19_visual_runner", TOPIC / "visual_experiments.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original = builtins.__import__
    def no_plotly(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("simulated missing Plotly")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_plotly)
    files, result = runner.loss_surface_view(tmp_path)
    assert result["backend"] == "matplotlib fallback"
    assert "03_loss_surface_3d.png" in files
    assert all((tmp_path / file).stat().st_size > 1000 for file in files)

