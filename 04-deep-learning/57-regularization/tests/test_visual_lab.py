"""Independent mathematical oracles, real checkpoints, exported media and app interactions."""

import copy
import re
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import train_visuals, visual_data
from visual_math import (CENTER, MATRIX, METHODS, batchnorm_trace, decay_paths, dropout_moments,
                         dropout_pass, regularized, stopping_replay)
from visual_render import (bn_html_figure, decay_surface, lambda_html_figure,
                           run_by_name, state_at, stopping_html_figure)
from visualizations import export_lab, verify_outputs


def test_dropout_moment_formula_mask_scaling_and_inference_case():
    moments = dropout_moments(.4, 100_000)
    np.testing.assert_allclose(moments["mean"], [.5, 1, 2], atol=.01)
    np.testing.assert_allclose(moments["variance"], .4/.6*np.array([.5, 1, 2])**2, atol=.02)
    first, second = dropout_pass(.4, 0), dropout_pass(.4, 1)
    assert any(not np.array_equal(a, b) for a, b in zip(first["active"][1:3], second["active"][1:3]))
    for raw, mask, post in zip(first["raw"][1:3], first["active"][1:3], first["post"][1:3]):
        np.testing.assert_allclose(post, raw*mask/.6)
    assert first["post"][-1].sum() == pytest.approx(1)
    zero = dropout_pass(0)
    for raw, post in zip(zero["raw"], zero["post"]):
        np.testing.assert_array_equal(raw, post)


@pytest.mark.parametrize("decay", [0., .3, 1.])
def test_quadratic_paths_match_pytorch_optimizers(decay):
    study = decay_paths(decay, steps=12)
    for name in METHODS:
        w = torch.nn.Parameter(torch.tensor([-2., 2.], dtype=torch.float64))
        if name.startswith("SGD"):
            optimizer = torch.optim.SGD([w], lr=.12, weight_decay=decay if name == "SGD + L2" else 0, foreach=False)
        else:
            cls = torch.optim.AdamW if name == "AdamW" else torch.optim.Adam
            optimizer = cls([w], lr=.08, weight_decay=decay, betas=(.9, .999), eps=1e-8, foreach=False)
        for step in range(1, 13):
            w.grad = torch.tensor(MATRIX @ (w.detach().numpy()-CENTER))
            optimizer.step()
            np.testing.assert_allclose(w.detach().numpy(), study["paths"][name]["points"][step], atol=2e-12)
    np.testing.assert_allclose((MATRIX+decay*np.eye(2))@study["minimum"], MATRIX@CENTER, atol=1e-14)
    if decay:
        assert not np.allclose(study["paths"]["Adam + L2"]["points"], study["paths"]["AdamW"]["points"])
    else:
        np.testing.assert_array_equal(study["paths"]["Adam + L2"]["points"], study["paths"]["AdamW"]["points"])


@pytest.mark.parametrize("batch_size", [2, 16, 128])
def test_batchnorm_full_stream_and_inference_match_pytorch(batch_size):
    trace = batchnorm_trace(batch_size, gamma=1.5, beta=-.5)
    layer = torch.nn.BatchNorm1d(3, eps=1e-5, momentum=.1).double()
    with torch.no_grad():
        layer.weight.fill_(1.5)
        layer.bias.fill_(-.5)
    for i, batch in enumerate(trace["raw"]):
        layer.train()
        actual = layer(torch.tensor(batch)).detach().numpy()
        np.testing.assert_allclose(actual, trace["affine"][i], atol=2e-12)
        np.testing.assert_allclose(layer.running_mean.numpy(), trace["running_mean"][i], atol=2e-12)
        np.testing.assert_allclose(layer.running_var.numpy(), trace["running_var"][i], atol=2e-12)
        layer.eval()
        np.testing.assert_allclose(layer(torch.tensor(batch)).detach().numpy(), trace["inference"][i], atol=2e-12)
    zero = batchnorm_trace(batch_size, gamma=0, beta=2)
    np.testing.assert_array_equal(zero["affine"], np.full_like(zero["affine"], 2))


def test_small_batch_mean_variation_is_larger_in_this_fixed_simulation():
    small, large = batchnorm_trace(2), batchnorm_trace(128)
    assert np.all(small["batch_mean"].std(0) > large["batch_mean"].std(0))


def test_stopping_rule_excludes_unseen_future_and_tracks_absolute_minimum():
    history = [{"epoch": i, "validation": {"loss": loss}} for i, loss in enumerate([1., .95, .94, .5], 1)]
    replay = stopping_replay(history, patience=2, min_delta=.1)
    assert replay["stop_epoch"] == 3 and replay["best_epoch"] == 3
    assert replay["best_loss"] == .94
    assert stopping_replay(history, patience=3, min_delta=.1)["best_epoch"] == 4
    assert stopping_replay(history, patience=3, maximum=2)["best_epoch"] == 2


@pytest.fixture(scope="module")
def small_study():
    return train_visuals(epochs=6, n=100, grid=15)


def test_actual_training_snapshots_selection_and_no_test_queries(small_study):
    study = small_study
    assert study["test_evaluations"] == 0 and study["final_test"] is None
    assert len({run["initial_linear_sha256"] for run in study["runs"]}) == 1
    assert study["configuration"]["splits"] == {"train": 60, "validation": 20, "test": 20}
    for run in study["runs"]:
        assert run["snapshots"][0]["epoch"] == 0
        selected = run["selected"]
        original = run["snapshots"][selected["epoch"]]
        np.testing.assert_array_equal(selected["probability"], original["probability"])
        np.testing.assert_array_equal(selected["weights"], original["weights"])
        assert selected["validation"] == original["validation"]
        assert selected["probability"].shape == (15, 15)
        assert np.all((selected["probability"] >= 0) & (selected["probability"] <= 1))
        assert np.isfinite(selected["weights"]).all()
    first, early = study["runs"][0], study["runs"][4]
    assert first["history"][:len(early["history"])] == early["history"]
    expected = min(study["runs"], key=lambda run: run["selected"]["validation"]["loss"])
    assert expected["name"] == study["selected_configuration"]


def test_visual_scaler_is_train_only_and_context_restores_rng(small_study):
    from sklearn.datasets import make_moons
    data = visual_data(n=100, grid=15)
    raw, _ = make_moons(n_samples=100, noise=.30, random_state=157)
    np.testing.assert_allclose(data["scaler_mean"], raw[data["ids"]["train"]].mean(0))
    before = torch.get_rng_state().clone()
    repeat = train_visuals(epochs=6, n=100, grid=15)
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
    for left, right in zip(repeat["runs"], small_study["runs"]):
        assert left["history"] == right["history"]
        np.testing.assert_array_equal(left["selected"]["probability"], right["selected"]["probability"])


def test_explicit_single_test_evaluation_after_selection():
    study = train_visuals(epochs=2, n=100, grid=15, evaluate_test=True)
    assert study["test_evaluations"] == 1
    assert 0 <= study["final_test"]["accuracy"] <= 1


def test_surface_heights_and_lambda_slider_are_mathematically_aligned():
    fig = decay_surface(.3, 10)
    assert fig.data[-1].z[0] == 0
    assert fig.data[-1].z[1] == pytest.approx(regularized(decay_paths(.3, 10)["minimum"], .3))
    for index in range(2, 6):
        trace = fig.data[index]
        points = np.column_stack((trace.x, trace.y))
        np.testing.assert_allclose(trace.z, regularized(points, .3))
    slider = lambda_html_figure(10)
    assert len(slider.frames) == 11
    assert slider.frames[0].name == "0.0" and slider.frames[-1].name == "1.0"
    frame = slider.frames[-1]
    np.testing.assert_allclose(frame.data[-1].x[1], decay_paths(1, 10)["minimum"][0])
    assert "AdamW is not gradient descent on J" in slider.layout.annotations[0].text


def test_animation_axes_do_not_rescale_the_same_experiment(small_study):
    from visual_render import bn_figure, boundary_figure
    pairs = [(boundary_figure(small_study, 0), boundary_figure(small_study, 6)),
             (bn_figure(batchnorm_trace(), 0), bn_figure(batchnorm_trace(), 39))]
    for first, last in pairs:
        for left, right in zip(first.axes, last.axes):
            np.testing.assert_allclose(left.get_xlim(), right.get_xlim())
            np.testing.assert_allclose(left.get_ylim(), right.get_ylim())
        plt.close(first)
        plt.close(last)


@pytest.mark.parametrize("function,arguments", [
    (dropout_pass, {"p": 1}), (decay_paths, {"decay": -1}),
    (batchnorm_trace, {"batch_size": 1}), (batchnorm_trace, {"batch_size": 129}),
    (batchnorm_trace, {"eps": 0}), (stopping_replay, {"history": []}),
    (train_visuals, {"epochs": 201}), (visual_data, {"n": 101}),
])
def test_invalid_inputs_fail_explicitly(function, arguments):
    with pytest.raises(ValueError):
        function(**arguments)


@pytest.mark.rendering
def test_real_exports_decode_and_embed_offline_runtime(tmp_path, small_study, monkeypatch):
    import visualizations
    monkeypatch.setattr(visualizations, "train_visuals", lambda *args, **kwargs: small_study)
    report = export_lab(tmp_path, epochs=6, n=100, grid=15, frames=8)
    assert len(report["artifacts"]) == 14
    gifs = [item for item in report["artifacts"] if item["file"].endswith(".gif")]
    assert len(gifs) == 5 and all(item["frames"] >= 2 for item in gifs)
    assert sum(item["offline_runtime"] for item in report["artifacts"] if item["file"].endswith(".html")) == 3
    assert (tmp_path/"manifest.json").exists()
    for path in tmp_path.glob("*.html"):
        html = path.read_text(encoding="utf-8")
        startup = html[html.rfind("Plotly.newPlot("):]
        assert not re.search(r"\.then\(function\(\)\s*\{\s*Plotly\.animate", startup)
    with pytest.raises(ValueError, match="execution record"):
        export_lab(tmp_path, epochs=6, n=100, frames=8)
    assert not plt.get_fignums()


def test_dashboard_pages_and_controls_without_automatic_training(small_study):
    from streamlit.testing.v1 import AppTest
    app = Path(__file__).resolve().parents[1]/"app.py"
    at = AppTest.from_file(str(app), default_timeout=40).run()
    assert not at.exception and at.info
    assert "study" not in at.session_state
    at.selectbox(key="page").select("Dropout").run()
    at.slider(key="drop_p").set_value(.8).run()
    assert not at.exception
    assert float(at.metric[0].value) < .3
    at.selectbox(key="page").select("Batch normalization").run()
    at.slider(key="bn_gamma").set_value(0.).run()
    at.slider(key="bn_beta").set_value(2.).run()
    assert not at.exception
    table = at.dataframe[0].value
    assert table.loc[2, "Mean"] == 2 and table.loc[2, "Biased variance"] == 0
    at.selectbox(key="page").select("Weight decay and optimization").run()
    at.slider(key="decay").set_value(1.).run()
    assert not at.exception
    at.session_state["study"] = copy.deepcopy(small_study)
    for page in ("Generalization and overfitting", "Early stopping", "Integrated comparison"):
        at.selectbox(key="page").select(page).run()
        assert not at.exception
    at.selectbox(key="page").select("Early stopping").run()
    at.slider(key="es_patience").set_value(1).run()
    at.slider(key="es_max").set_value(1).run()
    assert not at.exception
    assert at.metric[0].value == "1"
