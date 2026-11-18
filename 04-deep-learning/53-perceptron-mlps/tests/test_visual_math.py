"""Geometry, derivatives and provenance invariants behind the exported figures."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from from_scratch import fit_perceptron, initialize_parameters, loss_and_gradients, predict_proba
from visual_math import (
    ACTIVATIONS, XOR_X, XOR_Y, activation, boundary_segment, check_gradients,
    fixed_forward, grid_points, loss_slice, perceptron_trace, relu_approximation,
    relu_features, target_function, training_trace,
)
from visualize_day53 import build_lab, main, verify_outputs


@pytest.mark.parametrize("name", [name for name in ACTIVATIONS if name != "step"])
def test_activation_derivatives_match_finite_differences(name):
    z = np.array([-3.0, -0.7, 0.4, 2.0])
    eps = 1e-6
    numerical = (activation(name, z+eps)-activation(name, z-eps))/(2*eps)
    np.testing.assert_allclose(activation(name, z, derivative=True), numerical, atol=1e-8)


def test_nondifferentiable_points_and_extreme_sigmoid():
    for name in ("relu", "step"):
        assert np.isnan(activation(name, 0, derivative=True))
    np.testing.assert_array_equal(activation("sigmoid", [-1000, 1000]), [0, 1])
    assert activation("gelu", 0) == 0
    assert activation("gelu", 0, derivative=True) == 0.5
    assert activation("sigmoid", 6, derivative=True) < 0.003
    assert activation("tanh", 6, derivative=True) < 0.0001


@pytest.mark.parametrize("weights,bias", [([1, 1], 0), ([2, -1], 0), ([1, 1], -1), ([1, 0], 0.3), ([0, 1], -0.7)])
def test_boundary_endpoints_obey_actual_neuron_equation(weights, bias):
    segment = boundary_segment(weights, bias, -1.5, 1.5)
    assert segment.shape == (2, 2)
    np.testing.assert_allclose(segment @ weights + bias, 0, atol=1e-12)
    assert np.all(np.abs(segment) <= 1.5+1e-10)


def test_degenerate_and_invisible_boundaries_are_empty():
    for weights, bias in (([0, 0], 0), ([0, 0], 2), ([1, 0], 4)):
        assert boundary_segment(weights, bias, -1, 1).shape == (0, 2)


def test_explicit_hidden_transformation_and_lifting():
    hidden = relu_features(XOR_X)
    np.testing.assert_array_equal(hidden, [[0, 0], [1, 0], [1, 0], [2, 1]])
    np.testing.assert_array_equal(hidden @ [1, -2], XOR_Y)
    lift = XOR_X.prod(axis=1)
    np.testing.assert_array_equal((XOR_X.sum(axis=1)-2*lift) > 0.5, XOR_Y)
    np.testing.assert_array_equal(lift < (XOR_X.sum(axis=1)-0.5)/2, XOR_Y)


def test_perceptron_checkpoints_match_core_implementation():
    trace = perceptron_trace()
    w, b, mistakes = fit_perceptron(trace["X"], trace["y"])
    final = trace["states"][-1]
    assert final["update"] == sum(mistakes)
    np.testing.assert_allclose(final["weights"], w)
    assert final["bias"] == b and trace["certified"]
    assert final["training_errors"] == final["margin_violations"] == 0
    for previous, current in zip(trace["states"], trace["states"][1:]):
        if current["update"] == previous["update"]: continue
        difference = current["weights"]-previous["weights"]
        bias_difference = current["bias"]-previous["bias"]
        assert np.any(np.all(np.isclose(trace["y"][:, None]*trace["X"], difference), axis=1) & (trace["y"] == bias_difference))


def test_mlp_checkpoints_are_exact_gradient_descent_states():
    trace = training_trace(epochs=25, frames=9)
    saved = {state["epoch"]: state for state in trace["states"]}
    parameters = initialize_parameters(2, 8)
    assert trace["states"][0]["epoch"] == 0 and trace["states"][-1]["epoch"] == 25
    for epoch in range(26):
        loss, grads = loss_and_gradients(XOR_X, XOR_Y, parameters)
        assert loss == pytest.approx(trace["losses"][epoch])
        if epoch in saved:
            for key in parameters:
                np.testing.assert_array_equal(parameters[key], saved[epoch]["parameters"][key])
            np.testing.assert_allclose(predict_proba(XOR_X, parameters), saved[epoch]["probabilities"])
        parameters = {key: value-0.5*grads[key] for key, value in parameters.items()}


def test_gradient_and_slice_match_direct_objective():
    parameters = initialize_parameters(2, 8)
    assert check_gradients(parameters) < 1e-7
    da, db, losses = loss_slice(parameters, size=7)
    for i, j in ((0, 6), (3, 3), (5, 2)):
        changed = {key: value.copy() for key, value in parameters.items()}
        changed["W2"][0, 0] += da[i, j]
        changed["b2"][0] += db[i, j]
        assert losses[i, j] == pytest.approx(loss_and_gradients(XOR_X, XOR_Y, changed)[0])


def test_fixed_forward_is_exact_matrix_composition_with_bias():
    network = fixed_forward()
    assert network["parameters"] == 37
    assert [len(values) for values in network["values"]] == [2, 4, 4, 1]
    for layer, (w, b) in enumerate(zip(network["weights"], network["biases"])):
        z = network["values"][layer] @ w + b
        np.testing.assert_allclose(z, network["logits"][layer])
        np.testing.assert_allclose(activation("sigmoid" if layer == 2 else "tanh", z), network["values"][layer+1])


@pytest.mark.parametrize("width", [2, 4, 8, 16])
def test_relu_construction_is_the_linear_interpolant(width):
    knots = np.linspace(0, 1, width+1)
    x = np.linspace(0, 1, 1001)
    np.testing.assert_allclose(relu_approximation(knots, width), target_function(knots), atol=1e-14)
    np.testing.assert_allclose(relu_approximation(x, width), np.interp(x, knots, target_function(knots)), atol=1e-14)


@pytest.mark.parametrize("call", [lambda: activation("unknown", 1), lambda: activation("relu", np.nan),
                                  lambda: grid_points(size=1), lambda: relu_features([[1, 2, 3]]),
                                  lambda: relu_approximation([1.1], 4), lambda: relu_approximation([0.5], 1),
                                  lambda: training_trace(epochs=0), lambda: loss_slice(initialize_parameters(2), size=1)])
def test_invalid_numerical_inputs_raise(call):
    with pytest.raises(ValueError): call()


def test_cli_invalid_mode_and_seed(tmp_path):
    with pytest.raises(SystemExit): main(["--mode", "invalid"])
    with pytest.raises(ValueError): build_lab(tmp_path, seed=-1)


@pytest.mark.rendering
def test_real_png_gif_html_exports_in_temporary_directory(tmp_path):
    from visual_render import Renderer
    renderer = Renderer(tmp_path, quick=True)
    renderer.size = 7
    renderer.training(training_trace(epochs=8, frames=3))
    renderer.neuron()
    records = verify_outputs(renderer.files)
    assert {Path(record["file"]).suffix for record in records} == {".png", ".gif", ".html"}
    assert next(record for record in records if record["file"].endswith(".gif"))["frames"] >= 3
    assert all(record["bytes"] > 100 for record in records)


def test_output_verification_rejects_external_script(tmp_path):
    page = tmp_path / "bad.html"
    page.write_text('<script src="https://cdn.example/plotly.js"></script>Plotly.newPlot', encoding="utf-8")
    with pytest.raises(ValueError, match="embed Plotly"): verify_outputs([page])


def test_missing_plotly_has_actionable_installation_message(monkeypatch):
    import builtins
    from visual_render import Renderer
    original = builtins.__import__
    def without_plotly(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("simulated missing optional renderer")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_plotly)
    with pytest.raises(RuntimeError, match=r"pip install.*dev"):
        Renderer.plotly()
