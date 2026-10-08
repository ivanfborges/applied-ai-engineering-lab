"""Numerical, history, rendering, slider, and CLI checks for the visual lab."""

import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))

import visual_core as core
import visual_lab as lab


@pytest.fixture
def data():
    return core.make_regression_data()


def test_history_is_additive_residual_based_and_reproducible(data):
    history = core.fit_history(data, n_estimators=8)
    repeated = core.fit_history(data, n_estimators=8)
    assert history.predictions_train.shape == (9, 180)
    assert history.corrections_train.shape == (8, 180)
    np.testing.assert_array_equal(history.predictions_grid, repeated.predictions_grid)
    np.testing.assert_allclose(
        np.diff(history.predictions_train, axis=0),
        history.learning_rate * history.corrections_train, atol=1e-14,
    )
    np.testing.assert_allclose(
        np.diff(history.predictions_grid, axis=0),
        history.learning_rate * history.corrections_grid, atol=1e-14,
    )
    np.testing.assert_allclose(history.residuals_train, data.y_train - history.predictions_train)
    np.testing.assert_allclose(history.residuals_validation, data.y_validation - history.predictions_validation)
    assert np.all(np.diff(history.train_rmse) <= 1e-12)
    assert history.best_stage == int(np.argmin(history.validation_rmse))
    for index, tree in enumerate(history.trees):
        leaves = tree.apply(data.x_train)
        for leaf in np.unique(leaves):
            mask = leaves == leaf
            np.testing.assert_allclose(
                history.corrections_train[index, mask],
                history.residuals_train[index, mask].mean(),
            )
    assert not np.shares_memory(history.predictions_train, history.corrections_train)


def test_data_has_known_truth_and_disjoint_reserved_test(data):
    np.testing.assert_allclose(data.truth_grid, core.true_function(data.x_grid[:, 0]))
    train, validation, reserved = map(set, (
        data.train_indices, data.validation_indices, data.reserved_indices
    ))
    assert not (train & validation or train & reserved or validation & reserved)
    assert train | validation | reserved == set(range(300))


def test_gradients_match_finite_differences_and_extreme_logits_are_stable():
    y = np.array([1.0, -2.0, 3.0])
    prediction = np.array([-0.5, 1.2, 2.0])
    np.testing.assert_allclose(
        core.numerical_negative_gradient(core.squared_loss, y, prediction),
        y - prediction, atol=1e-9,
    )
    labels, logits = np.array([0, 1, 0, 1]), np.array([-3.0, -1.0, 0.4, 2.0])
    np.testing.assert_allclose(
        core.numerical_negative_gradient(core.logistic_loss, labels, logits),
        core.logistic_negative_gradient(labels, logits), atol=1e-9,
    )
    np.testing.assert_allclose(core.logistic_negative_gradient([0, 1], [-1000, 1000]), [0, 0])
    assert np.isfinite(core.logistic_loss([1, 0], [-1000, 1000])).all()
    with pytest.raises(ValueError):
        core.logistic_loss([2], [0])
    with pytest.raises(ValueError):
        core.numerical_negative_gradient(core.squared_loss, y, prediction, step=0)


@pytest.mark.parametrize("parameters", [
    {"n_estimators": 0}, {"n_estimators": True}, {"learning_rate": 0},
    {"learning_rate": np.nan}, {"max_depth": 0}, {"min_samples_leaf": 0},
])
def test_invalid_history_configuration(data, parameters):
    with pytest.raises(ValueError):
        core.fit_history(data, **parameters)


def test_saved_history_is_numeric_and_inspectable(data, tmp_path):
    history = core.fit_history(data, n_estimators=4)
    destination = tmp_path / "history.npz"
    core.save_history(destination, data, history)
    with np.load(destination, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["predictions_grid"], history.predictions_grid)
        np.testing.assert_array_equal(saved["corrections_grid"], history.corrections_grid)
        np.testing.assert_array_equal(saved["reserved_indices"], data.reserved_indices)
        assert all(saved[key].dtype != object for key in saved.files)


def test_static_render_and_frame_contents(data, tmp_path):
    history = core.fit_history(data, n_estimators=5)
    fig = lab.one_boosting_step(data, history)
    output = tmp_path / "step.png"
    lab.save_figure(fig, output, dpi=70)
    with Image.open(output) as image:
        assert image.size == (980, 329)
        assert np.std(np.asarray(image)) > 5
    assert not plt.fignum_exists(fig.number)
    explorer = lab.stage_explorer(data, history)
    assert len(explorer.frames) == 6
    assert explorer.frames[0].name == "0"
    assert explorer.frames[-1].name == "5"
    for stage, frame in enumerate(explorer.frames):
        np.testing.assert_allclose(frame.data[0].y, history.predictions_grid[stage])
        np.testing.assert_allclose(frame.data[1].y, history.residuals_train[stage])
        expected = np.zeros(240) if stage == 0 else 0.2 * history.corrections_grid[stage - 1]
        np.testing.assert_allclose(frame.data[3].y, expected)
    html = tmp_path / "explorer.html"
    lab.write_offline_html(explorer, html)
    source = html.read_text(encoding="utf-8")
    assert "Plotly.addFrames" in source
    assert '<script src="' not in source


def test_surface_frames_have_real_stage_shapes():
    figure, record = lab.surface_explorer()
    assert [frame.name for frame in figure.frames] == ["0", "1", "5", "10", "25", "50", "100"]
    for frame in figure.frames:
        assert np.asarray(frame.data[0].z).shape == (32, 32)
    assert record["result"]["validation_rmse"] >= 0
    assert not np.allclose(figure.frames[0].data[0].z, figure.frames[-1].data[0].z)


def test_cli_static_run_and_invalid_dpi(tmp_path):
    result = subprocess.run(
        [sys.executable, str(TOPIC / "visual_lab.py"), "--headless",
         "--skip-html", "--skip-gifs", "--dpi", "60", "--output-dir", str(tmp_path)],
        cwd=tmp_path, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert len(list((tmp_path / "figures").glob("*.png"))) == 12
    record = json.loads((tmp_path / "visual_experiments.json").read_text())
    assert len(record["artifacts"]) == 12
    assert "surface" not in record["experiments"]
    assert record["dataset"]["reserved_test_scored"] is False
    assert (tmp_path / "stage_history.npz").is_file()
    invalid = subprocess.run(
        [sys.executable, str(TOPIC / "visual_lab.py"), "--dpi", "0"],
        capture_output=True, text=True, timeout=20,
    )
    assert invalid.returncode != 0
    assert "--dpi must be between 50 and 200" in invalid.stderr


def test_forest_prediction_is_the_mean_of_member_predictions(data):
    history = core.fit_history(data, n_estimators=3)
    fig, record = lab.boosting_vs_bagging(data, history)
    assert record["result"]["max_absolute_mean_identity_error"] < 1e-12
    assert record["configuration"]["bootstrap"] is True
    assert "one input feature" in record["limitation"].lower()
    plt.close(fig)
