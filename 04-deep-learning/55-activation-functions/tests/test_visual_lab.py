"""Mechanisms, actual CPU autograd/training, sliders and temporary exports."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from from_scratch import activation_and_derivative, softmax
from interactive_3d import export_3d, simplex_figure, surface_figure
from visual_lab import build_lab, verify_outputs
from visual_math import (
    attention, autograd_checks, dying_neuron, entropy, gradient_chains,
    require_torch, simplex_samples, train_moons,
)
from visual_render import Renderer, activation_explorer, softmax_explorer


def test_real_autograd_and_exact_gelu_agreement():
    report = autograd_checks()
    assert max(report["max_absolute_derivative_errors"].values()) < 1e-14


def test_activation_bounds_on_plot_domain():
    x = np.linspace(-6, 6, 601)
    sigmoid = activation_and_derivative(x, "sigmoid")[0]
    tanh = activation_and_derivative(x, "tanh")[0]
    assert np.all((sigmoid > 0) & (sigmoid < 1))
    assert np.all((tanh > -1) & (tanh < 1))
    np.testing.assert_array_equal(activation_and_derivative([-2, 0, 2], "relu")[0], [0, 0, 2])
    np.testing.assert_allclose(activation_and_derivative([-2, 2], "leaky_relu", .1)[1], [.1, 1])


def test_gradient_products_include_weights_and_start_at_one():
    chains = gradient_chains()
    for c in chains:
        assert c["magnitude"][0] == 1
        np.testing.assert_allclose(c["magnitude"][1:] / c["magnitude"][:-1],
                                   abs(c["local_derivative"] * c["weight"]))
    assert chains[0]["magnitude"][-1] < 1e-30
    np.testing.assert_array_equal(chains[2]["magnitude"], np.ones(21))
    assert chains[3]["magnitude"][-1] < 1
    assert chains[4]["magnitude"][-1] > 1


def test_inactive_neuron_actual_gradient_and_update_equations():
    data = dying_neuron()
    for name, history in data["traces"].items():
        initial = history[0]
        assert np.all(initial["z"] < 0)
        for current, following in zip(history[:-1], history[1:]):
            residual = current["output"] - data["target"]
            dw = np.mean(residual * current["slope"] * data["x"])
            db = np.mean(residual * current["slope"])
            assert following["w"] == pytest.approx(current["w"] - .5 * dw)
            assert following["b"] == pytest.approx(current["b"] - .5 * db)
        if name == "relu":
            assert all(s["gradient_norm"] == 0 for s in history)
            assert history[-1]["loss"] == initial["loss"]
        else:
            assert initial["gradient_norm"] > 0
            assert history[-1]["loss"] < initial["loss"] / 100


def test_attention_and_simplex_use_correct_axes():
    scores = attention()["scores"]
    q, k = attention()["queries"], attention()["keys"]
    np.testing.assert_allclose(scores, q @ k.T / np.sqrt(2))
    for t in (.5, 1., 2.):
        weights = attention(t)["weights"]
        np.testing.assert_allclose(weights, softmax(scores, axis=1, temperature=t))
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(3))
    data = simplex_samples()
    np.testing.assert_allclose(data["probabilities"].sum(axis=1), 1)
    np.testing.assert_allclose(data["trajectory"].sum(axis=1), 1)
    assert np.all(data["trajectory"].argmax(axis=1) == 0)
    assert entropy(data["trajectory"])[0] < entropy(data["trajectory"])[-1]
    assert entropy([0., 1., 0.]) == 0


@pytest.mark.parametrize("p", [[.2, .2], [-.1, 1.1], [np.nan, 1], []])
def test_entropy_rejects_invalid_distributions(p):
    with pytest.raises(ValueError):
        entropy(p)


@pytest.fixture(scope="module")
def small_study():
    return train_moons(epochs=8, grid=15)


def test_seeded_training_is_repeatable_and_validation_separate(small_study):
    torch = require_torch()
    threads = torch.get_num_threads()
    rng = torch.get_rng_state().clone()
    other = train_moons(epochs=8, grid=15)
    assert torch.get_num_threads() == threads
    assert torch.equal(torch.get_rng_state(), rng)
    np.testing.assert_array_equal(other["x"], small_study["x"])
    assert set(small_study["train_idx"]).isdisjoint(small_study["valid_idx"])
    np.testing.assert_allclose(small_study["x"][small_study["train_idx"]].mean(axis=0), 0, atol=1e-15)
    hashes = set()
    for name, model in small_study["models"].items():
        hashes.add(model["initial_parameter_sha256"])
        np.testing.assert_array_equal(model["logits"], other["models"][name]["logits"])
        np.testing.assert_allclose(model["probability"], 1 / (1 + np.exp(-model["logits"])), atol=1e-7)
        assert 0 <= model["metrics"]["validation_accuracy"] <= 1
        assert model["metrics"]["training_seconds"] > 0
        assert model["snapshots"][-1]["epoch"] == 8
        np.testing.assert_array_equal(model["probability"], model["snapshots"][-1]["probability"])
    assert len(hashes) == 1
    # Composition of affine layers has zero grid curvature in logit space.
    z = small_study["models"]["identity"]["logits"]
    # Float32 affine layers accumulate CPU-dependent roundoff, amplified by
    # second differences. Keep the bound relative to dtype and logit scale.
    curvature_atol = 16 * np.finfo(z.dtype).eps * max(1.0, float(np.max(np.abs(z))))
    np.testing.assert_allclose(np.diff(z, n=2, axis=0), 0, rtol=0, atol=curvature_atol)
    np.testing.assert_allclose(np.diff(z, n=2, axis=1), 0, rtol=0, atol=curvature_atol)


def test_activation_slider_updates_actual_function_and_zero_convention():
    view = activation_explorer()
    view["sliders"]["alpha"].set_val(.2)
    view["sliders"]["input"].set_val(0)
    assert "x = +0.000" in view["readout"].get_text()
    assert "not differentiable" in view["readout"].get_text()
    values = view["forward_lines"]["leaky_relu"].get_ydata()
    assert values[0] == pytest.approx(-1.2)
    plt.close(view["figure"])


def test_softmax_sliders_update_probabilities_jacobian_and_shift_invariance():
    view = softmax_explorer()
    view["sliders"]["logit1"].set_val(4)
    view["sliders"]["temperature"].set_val(.3)
    # A redraw invokes the same callback as a native slider interaction.
    before = view["state"]["probabilities"].copy()
    view["sliders"]["offset"].set_val(10000)
    view["figure"].canvas.draw()
    np.testing.assert_allclose(view["state"]["probabilities"], before, atol=1e-12)
    np.testing.assert_allclose(before, softmax([2, 4, -1], temperature=.3), atol=1e-12)
    assert view["state"]["entropy"] < .02
    plt.close(view["figure"])


@pytest.mark.rendering
def test_all_static_and_gif_exports_decode_in_temp_directory(tmp_path):
    renderer = Renderer(tmp_path, frames=12)
    paths = [renderer.activations(), renderer.gradients(), renderer.dying(), renderer.gelu(),
             renderer.temperature(), renderer.jacobian(), renderer.attention(), renderer.transformation(),
             renderer.gradients(True), renderer.dying(True), renderer.temperature(True)]
    records = verify_outputs(paths)
    assert len(records) == 11
    gifs = [r for r in records if r["file"].endswith(".gif")]
    assert len(gifs) == 4
    assert all(r["decoded_frames"] >= 2 and r["duration_ms"] > 0 and r["loop"] == 0 for r in gifs)


@pytest.mark.rendering
def test_actual_offline_3d_exports_and_checkpoint_data(tmp_path, small_study):
    renderer = Renderer(tmp_path, frames=12)
    records = verify_outputs([renderer.boundaries(small_study), *export_3d(tmp_path, small_study)])
    assert len(records) == 4
    assert all(r["embedded_runtime"] for r in records if r["file"].endswith(".html"))
    for path in tmp_path.glob("*.html"):
        html = path.read_text(encoding="utf-8")
        startup = html[html.rfind("Plotly.newPlot("):]
        assert "Plotly.animate(" not in startup, "opening the HTML must not play checkpoints automatically"
    figure = surface_figure(small_study)
    np.testing.assert_array_equal(figure.data[0].z, small_study["models"]["relu"]["snapshots"][0]["probability"])
    np.testing.assert_array_equal(figure.frames[-1].data[0].z, small_study["models"]["relu"]["probability"])
    simplex = simplex_figure()
    np.testing.assert_allclose(np.asarray(simplex.data[2].x) + simplex.data[2].y + simplex.data[2].z, 1)
    assert simplex.frames[-1].name == "59"
    logits = surface_figure(small_study, probability=False)
    assert logits.layout.scene.zaxis.range == logits.layout.scene4.zaxis.range
    assert logits.data[0].cmin == logits.data[3].cmin
    assert logits.data[0].cmax == logits.data[3].cmax


@pytest.mark.parametrize("arguments", [{"mode": "invalid"}, {"frames": 2}, {"epochs": 0}, {"grid": 5}])
def test_cli_bounds_before_rendering(tmp_path, arguments):
    with pytest.raises(ValueError):
        build_lab(tmp_path, **arguments)
