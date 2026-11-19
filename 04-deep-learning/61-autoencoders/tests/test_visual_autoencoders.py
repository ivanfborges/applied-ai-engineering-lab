"""Independent oracles for visual experiments, real exports and widget callbacks."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualize_autoencoders import parse_args
from visual_core import (LabExperiments, VisualAE, evaluate_model, geometry_data, infer,
                         metrics_at, save_checkpoints, scale_splits, tabular_data, train_model)
from visual_figures import (DEMOS, export_animation, latent_view, style, threshold_view)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    plt.close("all")
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def lab():
    result = LabExperiments()
    # Short smoke fits exercise exactly the same code; not a numerical reference run.
    result.epochs, result.frames = 2, 6
    return result


@pytest.mark.parametrize("generator", [tabular_data, geometry_data])
def test_independent_deterministic_splits_and_train_only_scaler(generator):
    data, repeat = generator(), generator()
    for role in ("train", "validation", "calibration", "test"):
        np.testing.assert_array_equal(getattr(data, role), getattr(repeat, role))
    stacked = np.vstack([getattr(data, role) for role in ("train", "validation", "calibration", "test")])
    assert len(np.unique(stacked, axis=0)) == len(stacked)
    scaled, scaler = scale_splits(data)
    np.testing.assert_allclose(scaler.mean_, data.train.mean(axis=0))
    old_train = scaled.train.copy()
    data.test[:] *= 1000
    again, _ = scale_splits(data)
    np.testing.assert_array_equal(old_train, again.train)


@pytest.mark.parametrize("latent", [1, 2, 4, 8, 12, 16])
def test_model_dimensions_include_overcomplete_representations(latent):
    model = VisualAE(12, latent)
    z, rec = infer(model, np.ones((7, 12)))
    assert z.shape == (7, latent) and rec.shape == (7, 12)
    assert np.isfinite(rec).all()
    with pytest.raises(ValueError):
        infer(model, np.ones((7, 11)))


def test_checkpoints_are_real_states_and_replay_after_safe_disk_load(tmp_path):
    data, _ = scale_splits(tabular_data())
    result = train_model(data.train[:64], data.validation, epochs=3, probe=data.test[:10], checkpoint_count=4)
    assert [c["epoch"] for c in result["checkpoints"]] == [0, 1, 2, 3]
    path = tmp_path / "checkpoints.pt"
    save_checkpoints(result["checkpoints"], path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    for checkpoint in loaded:
        result["model"].load_state_dict(checkpoint["state"])
        z, rec = infer(result["model"], data.test[:10])
        np.testing.assert_array_equal(z, checkpoint["latent"].numpy())
        np.testing.assert_array_equal(rec, checkpoint["reconstruction"].numpy())
    assert any(not torch.equal(result["checkpoints"][0]["state"][key], result["checkpoints"][-1]["state"][key])
               for key in result["checkpoints"][0]["state"])
    with pytest.raises(FileExistsError):
        save_checkpoints(result["checkpoints"], path)


def test_calibration_ignores_validation_test_and_labels(lab):
    result = lab.base()
    measured = evaluate_model(result["model"], lab.data)
    assert measured["scores"].shape == (len(lab.data.test),)
    assert (measured["scores"] >= 0).all()
    direct = np.mean((lab.data.test - measured["reconstruction"])**2, axis=1)
    np.testing.assert_array_equal(direct, measured["scores"])
    original_test, original_val, original_labels = lab.data.test.copy(), lab.data.validation.copy(), lab.data.labels.copy()
    try:
        lab.data.test *= 100
        lab.data.validation *= 1000
        lab.data.labels = 1 - lab.data.labels
        changed = evaluate_model(result["model"], lab.data)
        assert measured["thresholds"] == changed["thresholds"]
        sorted_scores = np.sort(measured["calibration_scores"])
        for q, threshold in measured["thresholds"].items():
            assert threshold == sorted_scores[int(np.ceil(float(q) * (len(sorted_scores)-1)))]
    finally:
        lab.data.test[:], lab.data.validation[:], lab.data.labels = original_test, original_val, original_labels


def test_metrics_and_f1_match_hand_count_and_positive_score_direction():
    actual = metrics_at([0, 0, 1, 1], [0, 3, 2, 4], 2)
    assert actual["confusion_matrix"] == [[1, 1], [1, 1]]
    assert actual["f1"] == .5 and actual["precision"] == .5 and actual["recall"] == .5
    assert metrics_at([0, 0, 1, 1], [0, 1, 2, 3], 1)["roc_auc"] == 1


def test_pca_baseline_scores_an_orthogonal_shift_above_exact_plane_points():
    from sklearn.decomposition import PCA
    from example import reconstruction_scores

    train = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0], [0, -1., 0]])
    pca = PCA(2, svd_solver="full").fit(train)
    probe = np.array([[.2, .3, 0], [.2, .3, 3]])
    scores = reconstruction_scores(probe, pca.inverse_transform(pca.transform(probe)))
    np.testing.assert_allclose(scores, [0, 3], atol=1e-14)


def test_deterministic_training_and_invalid_inputs():
    data, _ = scale_splits(tabular_data())
    a = train_model(data.train[:32], data.validation[:32], epochs=2)
    b = train_model(data.train[:32], data.validation[:32], epochs=2)
    np.testing.assert_array_equal(a["history"], b["history"])
    with pytest.raises(ValueError):
        train_model(data.train, data.validation, epochs=0)
    with pytest.raises(ValueError):
        train_model(data.train, data.validation, corruption_sd=-1)
    with pytest.raises(ValueError):
        VisualAE(12, latent=0)


def click_button(button):
    canvas = button.ax.figure.canvas
    canvas.draw()
    x, y = button.ax.transAxes.transform((.5, .5))
    for event_name in ("button_press_event", "button_release_event"):
        event = MouseEvent(event_name, canvas, x, y, button=1)
        canvas.callbacks.process(event_name, event)


def test_threshold_slider_and_buttons_recompute_actual_metrics(lab):
    fig, api = threshold_view(lab)
    maximum = api["max_score"]
    api["controls"]["threshold"].set_val(maximum)
    assert api["state"]["metrics"]["alerts"] == 0
    click_button(api["controls"]["0.95"])
    assert api["state"]["threshold"] == lab.base()["thresholds"]["0.95"]
    expected_alerts = lab.base()["scores"] > api["state"]["threshold"]
    assert api["state"]["metrics"]["alerts"] == expected_alerts.sum()
    api["controls"]["threshold"].set_val(0)
    assert api["state"]["metrics"]["recall"] == 1
    fig.canvas.draw()


def test_latent_camera_and_sample_sliders(lab):
    fig, api = latent_view(lab, True)
    api["controls"]["camera"].set_val(25)
    api["controls"]["sample"].set_val(180)
    assert api["state"] == {"azimuth": 25., "sample": 180}
    assert len([ax for ax in fig.axes if ax.name == "3d"]) == 2
    fig.canvas.draw()


@pytest.mark.rendering
def test_all_real_exports_have_expected_filenames_and_multiple_frames(lab, tmp_path):
    style()
    for name, (render, names) in DEMOS.items():
        render(lab, tmp_path)
        for filename in names:
            with Image.open(tmp_path / filename) as asset:
                assert asset.width >= 600 and asset.height >= 400
                assert asset.n_frames >= 2 if filename.endswith(".gif") else asset.format == "PNG"
                if filename.endswith(".gif"):
                    asset.seek(asset.n_frames - 1)
                    asset.convert("RGB").load()
    assert not plt.get_fignums()


def test_export_rejects_overwriting_and_cli_invalid_combinations(tmp_path):
    path = tmp_path / "existing.gif"
    path.write_bytes(b"preserve")
    fig = plt.figure()
    with pytest.raises(ValueError):
        export_animation(fig, lambda frame: None, 2, path)
    assert path.read_bytes() == b"preserve"
    for argv in ([], ["--all", "--interactive"], ["--demo", "network", "--interactive"],
                 ["--quick", "--seed", "-1"]):
        with pytest.raises(SystemExit):
            parse_args(argv)
