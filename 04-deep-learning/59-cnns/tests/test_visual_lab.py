"""Spatial identities, actual checkpoints, temporary exports and dashboard flows."""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import bars, evaluate, load_model, make_model, run_comparison, translation_predictions
from from_scratch import correlate2d, output_size
from visual_core import (FILTERS, architecture_table, convolution_steps, edge_image,
                         equivariance, receptive_mask, rf_table, rgb_demo, shift_image)


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def tiny_report():
    return run_comparison(seeds=(5,), epochs=2, train_size=16, val_size=8, test_size=8)


def test_each_animation_sum_and_output_position_matches_torch():
    image, kernel = edge_image(), FILTERS["Vertical edge"]
    for stride, padding in ((1, 0), (2, 1), (3, 2)):
        padded, output, steps = convolution_steps(image, kernel, stride=stride, padding=padding)
        oracle = F.conv2d(torch.tensor(image)[None, None], torch.tensor(kernel)[None, None],
                          stride=stride, padding=padding)[0, 0].numpy()
        np.testing.assert_allclose(output, oracle)
        assert len(steps) == output.size
        assert padded.shape == (8 + 2 * padding,) * 2
        for index, step in enumerate(steps):
            assert divmod(index, output.shape[1]) == (step["row"], step["col"])
            assert step["sum"] == output[step["row"], step["col"]]
            np.testing.assert_array_equal(step["products"], step["patch"] * kernel)


@pytest.mark.parametrize("size,kernel,stride,padding", [
    (8, 3, 1, 0), (8, 5, 2, 1), (16, 7, 3, 3), (4, 7, 1, 2),
])
def test_interactive_dimensions_match_actual_outputs(size, kernel, stride, padding):
    output = correlate2d(np.ones((size, size)), np.ones((kernel, kernel)), stride=stride, padding=padding)
    assert output.shape == (output_size(size, kernel, stride, padding),) * 2


@pytest.mark.parametrize("out,row,col", [(0, 0, 0), (0, 3, 4), (1, 5, 5)])
def test_rgb_channel_sums_and_both_outputs_match_conv2d(out, row, col):
    data = rgb_demo(row, col, out)
    expected = F.conv2d(torch.tensor(data["input"])[None], torch.tensor(data["weights"]),
                        bias=torch.tensor(data["bias"]))[0].numpy()
    np.testing.assert_allclose(data["output"], expected, atol=1e-14)
    assert data["contributions"].sum() + data["bias"][out] == pytest.approx(expected[out, row, col])


def test_zero_filled_translation_has_no_wrap_and_handles_large_shifts():
    image = np.arange(16).reshape(4, 4)
    shifted = shift_image(image, 1, -1)
    np.testing.assert_array_equal(shifted[:3, 1:], image[1:, :3])
    assert not shifted[:, 0].any() and not shifted[-1].any()
    assert not shift_image(image, 4).any()
    assert not shift_image(image, -7).any()


@pytest.mark.parametrize("dx", [-8, -3, -1, 0, 1, 3, 8])
def test_equivariance_compares_only_valid_interior_overlap(dx):
    image = np.random.default_rng(59).normal(size=(24, 24))
    result = equivariance(image, FILTERS["Vertical edge"], dx)
    assert result["max_error"] < 1e-14
    assert np.isnan(result["difference"][~result["valid"]]).all()
    assert not result["valid"][0].any() and not result["valid"][-1].any()


@pytest.mark.parametrize("layer,row,col", [(1, 0, 0), (2, 4, 5), (3, 0, 0), (4, 5, 5), (4, 9, 9)])
def test_structural_receptive_mask_matches_gradient_support_in_linear_control(layer, row, col):
    image = torch.ones(1, 1, 20, 20, dtype=torch.float64, requires_grad=True)
    value = image
    for index in range(1, layer + 1):
        if index == 2:
            # Average pooling activates all structural paths. Max pooling's
            # sample-dependent gradient routes would only expose a subset.
            value = F.avg_pool2d(value, 2)
        else:
            value = F.conv2d(value, torch.ones(1, 1, 3, 3, dtype=torch.float64), padding=1)
    value[0, 0, row, col].backward()
    np.testing.assert_array_equal(receptive_mask(layer, row, col), image.grad[0, 0].numpy() != 0)


def test_nominal_receptive_fields_and_border_clipping():
    assert [row["span"] for row in rf_table()] == [1, 3, 4, 8, 12]
    assert [row["jump"] for row in rf_table()] == [1, 1, 2, 2, 2]
    assert receptive_mask(4, 5, 5).sum() == 144
    assert receptive_mask(4, 0, 0).sum() < 144
    with pytest.raises(ValueError):
        receptive_mask(4, 10, 0)


def test_cnn_architecture_shapes_and_parameters_are_actual():
    cnn = make_model("cnn", 59)
    mlp = make_model("mlp", 59)
    inputs = torch.zeros(2, 1, 32, 32)
    first, second, pooled = cnn.features(inputs)
    assert [tuple(value.shape[1:]) for value in (first, second, pooled)] == [(8, 32, 32), (16, 16, 16), (16, 8, 8)]
    assert cnn(inputs).shape == mlp(inputs).shape == (2, 2)
    assert sum(p.numel() for p in cnn.parameters()) == sum(row[2] for row in architecture_table()) == 1282
    assert sum(p.numel() for p in mlp.parameters()) == 32866


def test_checkpoint_roundtrip_and_test_metrics_are_executed(tiny_report):
    report = tiny_report
    assert report["interpretation_status"] == "author review required"
    run = report["runs"][0]
    images, labels = bars(8, run["dataset_seeds"][2])
    assert [c["epoch"] for c in run["models"]["cnn"]["checkpoints"]] == [0, 1, 2]
    for kind in ("cnn", "mlp"):
        model_data = run["models"][kind]
        restored = load_model(kind, model_data["checkpoints"][-1]["state"])
        assert evaluate(restored, images, labels) == model_data["test"]
        assert len(model_data["history"]) == 3
        assert model_data["shift_sweep"][4]["accuracy"] == model_data["test"]["accuracy"]
        assert model_data["noise_sweep"][0]["accuracy"] == model_data["test"]["accuracy"]
        matrix = np.asarray(model_data["test"]["confusion_true_rows"])
        assert matrix.sum() == 8
    before, after = run["models"]["cnn"]["checkpoints"][0], run["models"]["cnn"]["checkpoints"][-1]
    assert before["state"]["conv1.weight"] != after["state"]["conv1.weight"]


def test_experiment_repeats_and_seeds_do_not_share_split_seeds(tiny_report):
    repeated = run_comparison(seeds=(5,), epochs=2, train_size=16, val_size=8, test_size=8)
    assert repeated["runs"] == tiny_report["runs"]
    two = run_comparison(seeds=(5, 6), epochs=1, train_size=4, val_size=2, test_size=2)
    split_seeds = [seed for run in two["runs"] for seed in run["dataset_seeds"]]
    assert len(set(split_seeds)) == 6


def test_translation_gallery_selects_by_label_and_recomputes_actual_probabilities(tiny_report):
    result = translation_predictions(tiny_report)
    assert result["label"] == 0
    run = tiny_report["runs"][0]
    images, labels = bars(8, run["dataset_seeds"][2])
    assert result["sample"] == int(torch.nonzero(labels == 0)[0, 0])
    for kind in ("cnn", "mlp"):
        model = load_model(kind, run["models"][kind]["checkpoints"][-1]["state"])
        with torch.no_grad():
            expected = torch.softmax(model(images[result["sample"]:result["sample"] + 1]), 1)[0, 0].item()
        assert result["results"][kind][4]["true_class_probability"] == expected


@pytest.mark.rendering
def test_all_real_export_formats_are_valid_in_temporary_directory(tmp_path, tiny_report):
    from visualizations import EXPECTED, generate_collection
    output, _ = generate_collection(tmp_path / "exports", report=tiny_report, frame_limit=2)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["artifacts"]) == 14
    for name in EXPECTED:
        path = output / name
        assert path.is_file() and path.stat().st_size > 1000
        if path.suffix in (".gif", ".png"):
            with Image.open(path) as image:
                assert image.width >= 900 and image.height >= 400
                if path.suffix == ".gif":
                    assert image.n_frames >= 2
                image.verify()
        else:
            html = path.read_text(encoding="utf-8")
            assert "Plotly.newPlot" in html and "plotly.js" in html
            assert '<script src="http' not in html
    with pytest.raises(FileExistsError):
        generate_collection(output, report=tiny_report, frame_limit=2)


@pytest.mark.integration
def test_dashboard_sections_widgets_and_invalid_geometry(tiny_report):
    from streamlit.testing.v1 import AppTest
    path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(path), default_timeout=60)
    app.session_state["measured_report"] = tiny_report
    app.run()
    assert not app.exception
    sections = app.selectbox(key="section").options
    for section in sections:
        app.selectbox(key="section").set_value(section).run()
        assert not app.exception, section
    app.selectbox(key="section").set_value(sections[3]).run()
    app.slider(key="input_size").set_value(4)
    app.selectbox(key="kernel_size").set_value(7)
    app.selectbox(key="padding").set_value(0).run()
    assert not app.exception and app.error
    app.selectbox(key="padding").set_value(2).run()
    assert not app.exception and not app.error
    app.selectbox(key="section").set_value(sections[5]).run()
    app.selectbox(key="rf_layer").set_value(1).run()
    app.slider(key="rf_row_1").set_value(0).run()
    assert not app.exception


@pytest.mark.integration
def test_training_progress_cache_replays_across_two_sessions(monkeypatch):
    import experiments
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    calls = []
    actual_run = experiments.run_comparison

    def bounded_run(*, seeds, epochs, progress):
        calls.append((seeds, epochs))
        return actual_run(seeds=seeds, epochs=1, train_size=8, val_size=4, test_size=4, progress=progress)

    monkeypatch.setattr(experiments, "run_comparison", bounded_run)
    path = str(Path(__file__).resolve().parents[1] / "app.py")
    first = AppTest.from_file(path, default_timeout=60).run()
    first.button(key="run_training").click().run()
    assert not first.exception
    second = AppTest.from_file(path, default_timeout=60).run()
    second.button(key="run_training").click().run()
    assert not second.exception
    assert calls == [((59, 60), 20)]
    assert first.session_state["measured_report"] == second.session_state["measured_report"]
