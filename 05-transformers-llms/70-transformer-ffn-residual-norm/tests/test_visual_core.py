"""Independent numerical checks for geometric and empirical visual mechanisms."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example import matched_stacks
from from_scratch import layer_norm
from visual_core import (Config, activation, activation_derivative, backward_decomposition,
                         batch_norm, block_trace, dense_diagnostics, geometry_fixture,
                         normalization_interventions, norm_steps, numerical_checks,
                         operation_counts, parameter_counts, rms_norm, scalar_gradient,
                         swiglu_fixture, transformer_diagnostics)

torch.set_num_threads(1)


@pytest.mark.parametrize("name", ["identity", "relu", "gelu", "silu"])
def test_activation_derivative_finite_differences(name):
    x = np.array([-4., -1., -.1, .1, 1., 4.])
    h = 1e-6
    numeric = (activation(x + h, name) - activation(x - h, name)) / (2 * h)
    np.testing.assert_allclose(activation_derivative(x, name), numeric, rtol=1e-6, atol=1e-8)
    assert activation_derivative([0.], "relu")[0] == 0
    assert np.isfinite(activation(np.array([-1000., 0., 1000.]), "silu")).all()


def test_layer_norm_plane_circle_affine_and_near_constant():
    x = np.random.default_rng(1).normal(size=(30, 3))
    states = norm_steps(x, eps=1e-12)
    np.testing.assert_allclose(states["centered"].sum(axis=-1), 0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(states["normalized"], axis=-1), np.sqrt(3), atol=1e-9)
    transformed = norm_steps(x, gamma=[2, 1, .5], beta=[1, 1, 1])["output"]
    assert np.max(np.abs(transformed.sum(axis=-1))) > 1
    np.testing.assert_array_equal(norm_steps(np.ones((2, 3)))["output"], np.zeros((2, 3)))
    near = norm_steps(1 + x * 1e-9)["normalized"]
    assert np.max(np.linalg.norm(near, axis=-1)) < 1e-5
    np.testing.assert_allclose(rms_norm(x), nn.functional.rms_norm(torch.tensor(x), (3,), eps=1e-5).numpy(), atol=1e-12)


def test_batch_norm_layout_and_running_variance_convention():
    x = np.random.default_rng(2).normal(size=(3, 4, 5)) + 2
    model = nn.BatchNorm1d(5, affine=False, momentum=.1).double()
    actual = model(torch.tensor(x).transpose(1, 2)).transpose(1, 2).detach().numpy()
    np.testing.assert_allclose(batch_norm(x), actual, atol=1e-12)
    np.testing.assert_allclose(model.running_mean.numpy(), .1 * x.mean(axis=(0, 1)))
    np.testing.assert_allclose(model.running_var.numpy(), .9 + .1 * x.var(axis=(0, 1), ddof=1))
    model.eval()
    actual = model(torch.tensor(x).transpose(1, 2)).transpose(1, 2).detach().numpy()
    np.testing.assert_allclose(batch_norm(x, False, model.running_mean.numpy(), model.running_var.numpy()), actual, atol=1e-12)


def test_normalization_interventions_couple_only_training_batchnorm():
    f = normalization_interventions(3)
    changes = f["unmodified_sample_max_change"]
    assert changes["LayerNorm"] == changes["RMSNorm"] == changes["BatchNorm eval"] == 0
    assert changes["BatchNorm train"] > .01
    v = f["variants"]
    np.testing.assert_allclose(v["base"]["LayerNorm"], v["offset +3"]["LayerNorm"], atol=1e-12)
    assert not np.allclose(v["base"]["RMSNorm"], v["offset +3"]["RMSNorm"])
    np.testing.assert_allclose(v["base"]["LayerNorm"], v["added sample"]["LayerNorm"][:2], atol=1e-12)
    assert not np.allclose(v["base"]["BatchNorm train"], v["added sample"]["BatchNorm train"][:2])
    np.testing.assert_allclose(v["base"]["BatchNorm eval"][:1], v["single sample"]["BatchNorm eval"])


@pytest.mark.parametrize("a,depth,alpha", [(.1, 24, 1.), (-1., 48, 1.), (-2., 6, .5), (0., 12, 2.)])
def test_scalar_gradient_matches_actual_autograd(a, depth, alpha):
    for residual in (False, True):
        x = torch.tensor(1., dtype=torch.float64, requires_grad=True)
        y = x
        for _ in range(depth):
            y = y + alpha * a * y if residual else a * y
        actual = abs(torch.autograd.grad(y, x)[0].item())
        result = scalar_gradient(a, depth, alpha, residual)
        # Exponentiation and repeated backward additions accumulate different rounding.
        # Relative tolerance also protects the tiny nonzero plain-network gradients.
        assert result["value"] == pytest.approx(actual, rel=2e-13, abs=0.)
    assert scalar_gradient(0., 0)["value"] == 1


def test_scalar_extremes_report_zero_overflow_and_underflow():
    assert scalar_gradient(-1., 48, residual=True)["status"] == "exact zero"
    assert scalar_gradient(100., 1000)["status"] == "float64 overflow"
    assert scalar_gradient(.001, 1000)["status"] == "float64 underflow"
    assert scalar_gradient(100., 1000)["value"] is None
    assert scalar_gradient(.001, 1000)["value"] == 0


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("gated", [True, False])
def test_parameter_counts_match_real_modules(bias, gated):
    d, h = 8, 21
    linear_modules = [nn.Linear(d, h, bias=bias), nn.Linear(h, d, bias=bias)]
    if gated:
        linear_modules.append(nn.Linear(d, h, bias=bias))
    ffn = sum(p.numel() for m in linear_modules for p in m.parameters())
    mha = nn.MultiheadAttention(d, 2, bias=bias)
    norms = nn.ModuleList([nn.LayerNorm(d), nn.LayerNorm(d)])
    actual = ffn + sum(p.numel() for p in mha.parameters()) + sum(p.numel() for p in norms.parameters())
    result = parameter_counts(d, h, gated=gated, bias=bias)
    assert result["total"] == actual
    assert result["ffn"] == ffn
    assert parameter_counts(d, h, layers=3, gated=gated, bias=bias)["total"] == 3 * actual


def test_macs_small_manual_oracle_and_cached_workload():
    # d=4,h=8,T=3: projections 4*3*4*4=192; QK+AV=2*9*4=72.
    assert operation_counts(4, 8, 3) == {"attention_macs": 264, "ffn_macs": 192, "total_macs": 456}
    assert operation_counts(4, 8, 3, cached=True)["attention_macs"] == 4*4*4 + 2*3*4
    assert operation_counts(4, 8, 3, gated=True)["ffn_macs"] == 3*3*4*8


def test_shared_geometry_and_swiglu_actual_signed_features():
    geometry = geometry_fixture(4)
    assert geometry["affine_error"] < 1e-12
    np.testing.assert_array_equal(geometry["variants"]["identity"][1], geometry["variants"]["gelu"][1])
    assert geometry["variants"]["gelu"][2].shape == (96, 3)
    f = swiglu_fixture(4)
    np.testing.assert_allclose(f["gated_features"], f["activated_gate"] * f["features"])
    assert f["output"].shape == (4, 8)
    assert abs(f["dense_count"] - f["gated_count"]) <= 3 * 8 + 2
    assert activation(np.array([4.]), "silu")[0] > 1
    assert activation(np.array([-1.]), "silu")[0] < 0


def test_complete_trace_matches_block_and_gradient_vector_sums():
    for order in ("pre", "post"):
        for residual in ("residual", "no_residual"):
            block = matched_stacks(5, 1, 8, 2, 32)[order + "_" + residual][0]
            x = torch.arange(32, dtype=torch.float64).reshape(1, 4, 8) / 20
            torch.testing.assert_close(block_trace(block, x)["output"], block(x))
    f = backward_decomposition(5)
    assert f["decomposition_max_abs_error"] < 1e-12
    c = list(f["contributions"].values())
    np.testing.assert_allclose(c[0] + c[1], c[2], atol=1e-12)
    np.testing.assert_allclose(c[2] + c[3], c[4], atol=1e-12)
    assert all(np.isfinite(v).all() for v in f["gradients"].values())


def test_actual_profiles_finite_and_final_norm_leaves_states_identical():
    config = Config(seed=6, depth=2, d_model=8, expansion=2, quick=True)
    rows = transformer_diagnostics(config)
    assert all(r["finite"] and r["parameters_matched"] for r in rows)
    for r in rows:
        if r["condition"] == "pre":
            other = next(s for s in rows if s["condition"] == "pre+final-LN" and s["depth"] == r["depth"] and s["seed"] == r["seed"] and s["layer"] == r["layer"])
            assert r["activation_rms"] == other["activation_rms"]
            assert r["gradient_rms"] == pytest.approx(r["gradient_l2"] / np.sqrt(2 * 8 * 8))
    dense = dense_diagnostics(config)
    assert all(r["finite"] for r in dense)
    assert {r["condition"] for r in dense} == {"plain", "residual"}
    for r in dense:
        if r["layer"] == r["depth"]:
            assert r["relative_gradient"] == pytest.approx(1.)
    assert numerical_checks(6)["ffn_other_tokens_change"] == 0


@pytest.mark.parametrize("changes", [{"depth": 0}, {"d_model": 7}, {"expansion": float("nan")},
                                    {"seed": -1}, {"device": "tpu"}])
def test_invalid_configuration_rejected(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes).validate()
