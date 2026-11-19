"""Numerical invariants, finite differences and paired framework diagnostics."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from example import DeepReLU, diagnose, run_study
from from_scratch import initialization_std, paired_demo, propagate_relu


@pytest.mark.parametrize("method,mode,variance", [
    ("xavier", "fan_in", 2 / 12), ("he", "fan_in", 2 / 3), ("he", "fan_out", 2 / 9),
])
def test_rectangular_fan_formulas_and_gain(method, mode, variance):
    assert initialization_std(3, 9, method, mode=mode)**2 == pytest.approx(variance)
    assert initialization_std(3, 9, method, mode=mode, gain=2)**2 == pytest.approx(4 * variance)


@pytest.mark.parametrize("args,kwargs", [
    ((0, 4), {}), ((4, -1), {}), ((True, 4), {}), ((2.5, 4), {}),
    ((4, 4, "unknown"), {}), ((4, 4), {"mode": "both"}),
    ((4, 4), {"gain": 0}), ((4, 4), {"gain": np.nan}),
])
def test_invalid_initialization_inputs(args, kwargs):
    with pytest.raises(ValueError):
        initialization_std(*args, **kwargs)


def test_relu_hand_oracle_second_moment_is_not_variance():
    inputs = np.array([[-2., 1.], [2., -1.]])
    actual = propagate_relu(inputs, [np.eye(2)])
    np.testing.assert_allclose(actual, [2.5, 1.25])
    output = np.maximum(inputs, 0)
    assert output.var() != pytest.approx(actual[-1])
    assert output.var() + output.mean()**2 == pytest.approx(actual[-1])


@pytest.mark.parametrize("inputs,weights", [
    ([], [np.eye(2)]), ([[1., np.nan]], [np.eye(2)]), ([[1., 2.]], []),
    ([[1., 2.]], [np.ones((2, 3))]), ([[1., 2.]], [np.array([[np.inf, 1.]])]),
])
def test_propagation_rejects_bad_data(inputs, weights):
    with pytest.raises(ValueError):
        propagate_relu(inputs, weights)


def test_propagation_reports_overflow():
    with pytest.raises(FloatingPointError):
        propagate_relu([[1e200]], [np.eye(1)])


def test_paired_scale_identity_at_every_layer():
    results = paired_demo(width=8, depth=5, batch=7, seed=2)
    np.testing.assert_allclose(results["xavier"], results["he"] * 0.5**np.arange(6), rtol=2e-14)
    repeat = paired_demo(width=8, depth=5, batch=7, seed=2)
    np.testing.assert_array_equal(results["he"], repeat["he"])


def test_torch_initializer_pairing_numpy_forward_and_rng_isolation():
    state = torch.random.get_rng_state().clone()
    he = DeepReLU(6, 3, "he", 4)
    xavier = DeepReLU(6, 3, "xavier", 4)
    normalized = DeepReLU(6, 3, "he_layernorm", 4)
    torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
    for h, x, n in zip(he.linears, xavier.linears, normalized.linears):
        torch.testing.assert_close(h.weight, x.weight * np.sqrt(2), rtol=2e-14, atol=1e-15)
        torch.testing.assert_close(h.weight, n.weight, rtol=0, atol=0)
    inputs = torch.tensor(np.random.default_rng(1).normal(size=(4, 6)), dtype=torch.float64)
    actual = np.array([float(h.detach().square().mean()) for h in he(inputs)])
    expected = propagate_relu(inputs.numpy(), [layer.weight.detach().numpy() for layer in he.linears])
    np.testing.assert_allclose(actual, expected, rtol=2e-14)


def test_torch_official_rectangular_initializers_match_formula():
    for method in ("xavier", "he"):
        generator = torch.Generator().manual_seed(3)
        actual = torch.empty((9, 3), dtype=torch.float64)
        if method == "xavier":
            nn.init.xavier_normal_(actual, generator=generator)
        else:
            nn.init.kaiming_normal_(actual, mode="fan_out", nonlinearity="relu", generator=generator)
        base = torch.randn((9, 3), dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(actual, base * initialization_std(3, 9, method, mode="fan_out"))


@pytest.mark.parametrize("strategy", ["he", "he_layernorm"])
def test_input_and_weight_gradients_against_finite_differences(strategy):
    model = DeepReLU(4, 2, strategy, 8)
    inputs = torch.tensor([[0.3, -0.7, 0.9, 1.2], [0.6, 0.4, -0.2, -0.8]], dtype=torch.float64,
                          requires_grad=True)
    output = model(inputs)[-1]
    loss = output.square().mean()
    input_grad, weight_grad = torch.autograd.grad(loss, (inputs, model.linears[0].weight))
    eps = 1e-6
    for parameter, gradient in ((inputs, input_grad), (model.linears[0].weight, weight_grad)):
        for index in np.ndindex(tuple(parameter.shape)):
            original = float(parameter[index].detach())
            with torch.no_grad():
                parameter[index] = original + eps
            plus = float(model(inputs)[-1].detach().square().mean())
            with torch.no_grad():
                parameter[index] = original - eps
            minus = float(model(inputs)[-1].detach().square().mean())
            with torch.no_grad():
                parameter[index] = original
            assert float(gradient[index]) == pytest.approx((plus - minus) / (2 * eps), abs=2e-8, rel=2e-5)


def test_two_adjoint_definitions_and_no_parameter_mutation():
    model = DeepReLU(4, 2, "he", 2)
    inputs = torch.ones((3, 4), dtype=torch.float64)
    probe = torch.arange(1., 13., dtype=torch.float64).reshape(3, 4)
    before = [p.detach().clone() for p in model.parameters()]
    report = diagnose(model, inputs, probe)
    output = model(inputs)[-1].detach()
    expected_loss_rms = float((2 * output / output.numel()).square().mean().sqrt())
    assert report["layers"][-1]["loss_activation_gradient_rms"] == pytest.approx(expected_loss_rms)
    assert report["layers"][-1]["probe_activation_gradient_rms"] == pytest.approx(1)
    # Independent manual backpropagation of the fixed output adjoint.
    activations = model(inputs)
    adjoint = probe / probe.square().mean().sqrt()
    for index in reversed(range(len(model.linears))):
        adjoint = (adjoint * (activations[index + 1] > 0)) @ model.linears[index].weight
    assert report["layers"][0]["probe_activation_gradient_rms"] == pytest.approx(float(adjoint.detach().square().mean().sqrt()))
    for actual, expected in zip(model.parameters(), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.grad is None


def test_layernorm_population_variance_and_batch_independence():
    model = DeepReLU(5, 2, "he_layernorm", 1)
    inputs = torch.tensor([[1., 2., 3., 5., 8.], [-1., 4., 2., 7., 0.]], dtype=torch.float64)
    z = model.linears[0](inputs)
    variance = z.var(dim=-1, unbiased=False, keepdim=True)
    expected = (z - z.mean(dim=-1, keepdim=True)) / (variance + 1e-5).sqrt()
    torch.testing.assert_close(model.norms[0](z), expected)
    torch.testing.assert_close(model.norms[0](z).var(-1, unbiased=False), (variance / (variance + 1e-5)).ravel())
    together = model(inputs)[-1]
    alone = model(inputs[:1])[-1]
    torch.testing.assert_close(alone, together[:1])
    model.eval()
    torch.testing.assert_close(model(inputs)[-1], together)


@pytest.mark.parametrize("kwargs", [
    {"width": 0}, {"depth": -1}, {"strategy": "unknown"}, {"seed": -1}, {"seed": True},
])
def test_model_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        DeepReLU(**kwargs)


def test_diagnostic_rejects_invalid_input_and_probe():
    model = DeepReLU(3, 1)
    with pytest.raises(ValueError):
        model(torch.ones(2, 3))  # Deliberate float32 input.
    with pytest.raises(ValueError):
        diagnose(model, torch.ones(2, 3, dtype=torch.float64), torch.zeros(2, 3, dtype=torch.float64))


def test_small_study_all_strategies_and_deterministic_metrics():
    first = run_study(width=8, depth=3, batch=5, seed=6)
    second = run_study(width=8, depth=3, batch=5, seed=6)
    assert first["results"] == second["results"]
    assert first["configuration"]["optimizer_steps"] == 0
    assert len(first["results"]) == 5
    for result in first["results"].values():
        assert len(result["layers"]) == 4
        for layer in result["layers"]:
            assert layer["second_moment"] == pytest.approx(layer["variance"] + layer["mean"]**2)
        assert result["layers"][-1]["probe_activation_gradient_rms"] == pytest.approx(1)
