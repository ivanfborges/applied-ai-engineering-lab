"""Numerical oracles, structural controls and actual autograd on synthetic data."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import feed_forward, gelu, layer_norm, layer_norm_vjp, residual_sublayer
from example import TransformerBlock, matched_stacks, numpy_parity, profile_stack, run_diagnostics

torch.set_num_threads(1)


def test_layer_norm_population_variance_and_affine_oracle():
    x = np.array([[[2., 4., 6., 8.], [5., 5., 5., 5.]]])
    gamma, beta = np.array([1., -2., 3., 0.]), np.array([2., 1., -3., 4.])
    expected = np.array([-3., -1., 1., 3.]) / np.sqrt(5. + 1e-5) * gamma + beta
    actual = layer_norm(x, gamma, beta)
    np.testing.assert_allclose(actual[0, 0], expected, atol=1e-14)
    np.testing.assert_array_equal(actual[0, 1], beta)
    np.testing.assert_array_equal(x[0, 0], [2., 4., 6., 8.])


def test_layer_norm_token_batch_independence_shift_and_epsilon():
    x = np.random.default_rng(1).normal(size=(2, 3, 5))
    actual = layer_norm(x, np.ones(5), np.zeros(5))
    changed = x.copy()
    changed[1, 2] += 100
    np.testing.assert_allclose(layer_norm(changed, np.ones(5), np.zeros(5)), actual, atol=1e-13)
    for b in range(2):
        for t in range(3):
            np.testing.assert_allclose(layer_norm(x[b, t], np.ones(5), np.zeros(5)), actual[b, t])
    np.testing.assert_allclose(actual.var(axis=-1), x.var(axis=-1) / (x.var(axis=-1) + 1e-5))
    tiny = np.array([1e-7, -1e-7])
    assert not np.allclose(layer_norm(tiny, [1, 1], [0, 0]), layer_norm(tiny * 100, [1, 1], [0, 0]))


@pytest.mark.parametrize("width", [1, 2, 7])
def test_layer_norm_vjp_matches_finite_difference_and_autograd(width):
    rng = np.random.default_rng(width)
    x, upstream = rng.normal(size=(2, width)), rng.normal(size=(2, width))
    gamma, beta = rng.normal(size=width), rng.normal(size=width)
    actual = layer_norm_vjp(x, gamma, upstream, eps=0.01)
    expected = np.empty_like(x)
    step = 1e-6
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += step
        minus[index] -= step
        expected[index] = np.sum((layer_norm(plus, gamma, beta, 0.01) - layer_norm(minus, gamma, beta, 0.01)) * upstream) / (2 * step)
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-7)
    tx = torch.tensor(x, requires_grad=True)
    output = nn.functional.layer_norm(tx, (width,), torch.tensor(gamma), torch.tensor(beta), 0.01)
    oracle = torch.autograd.grad(output, tx, torch.tensor(upstream))[0]
    np.testing.assert_allclose(actual, oracle.numpy(), atol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=-1), 0, atol=1e-12)


def test_layer_norm_constant_vector_backward_is_finite():
    actual = layer_norm_vjp(np.ones(3), np.ones(3), [1, 2, 3])
    np.testing.assert_allclose(actual, np.array([-1, 0, 1]) / np.sqrt(1e-5))


def test_numpy_torch_forward_and_backward_agreement():
    for value in numpy_parity().values():
        assert value < 1e-11
    x = np.linspace(-10, 10, 101)
    np.testing.assert_allclose(gelu(x), nn.functional.gelu(torch.tensor(x), approximate="tanh").numpy(), atol=1e-14)


def test_ffn_positionwise_loop_and_single_token_intervention():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 3, 4))
    params = [rng.normal(size=s) for s in [(4, 9), (9,), (9, 4), (4,)]]
    expected = np.stack([np.stack([feed_forward(token, *params) for token in row]) for row in x])
    np.testing.assert_allclose(feed_forward(x, *params), expected, atol=1e-13)
    changed = x.copy()
    changed[0, 1] += 20
    difference = feed_forward(changed, *params) - expected
    assert np.linalg.norm(difference[0, 1]) > 0
    difference[0, 1] = 0
    np.testing.assert_allclose(difference, 0, atol=1e-13)


@pytest.mark.parametrize("order", ["pre", "post"])
def test_zero_branch_and_forbidden_residual_broadcast(order):
    x = np.array([[1., 2., 4.], [4., 3., 2.]])
    actual = residual_sublayer(x, np.zeros_like, np.ones(3), np.zeros(3), order)
    expected = x if order == "pre" else layer_norm(x, np.ones(3), np.zeros(3))
    np.testing.assert_allclose(actual, expected)
    with pytest.raises(ValueError, match="exactly"):
        residual_sublayer(x, lambda z: z[:1], np.ones(3), np.zeros(3), order)


@pytest.mark.parametrize("eps", [0, -1, np.nan, np.inf, True, [1e-5], "small", 1j])
def test_invalid_layer_norm_epsilon(eps):
    with pytest.raises(ValueError, match="eps"):
        layer_norm([1., 2.], [1, 1], [0, 0], eps)


@pytest.mark.parametrize("x", [[], 3, [np.nan], [np.inf], [1j], ["text"], [True]])
def test_invalid_layer_norm_input(x):
    with pytest.raises(ValueError):
        layer_norm(x, [1], [0])


def test_invalid_shapes_and_order():
    with pytest.raises(ValueError, match="gamma"):
        layer_norm([1, 2], [[1, 1]], [0, 0])
    with pytest.raises(ValueError, match="upstream"):
        layer_norm_vjp([[1, 2]], [1, 1], [1, 1])
    with pytest.raises(ValueError, match="w1"):
        feed_forward([1, 2], np.ones((3, 4)), np.ones(4), np.ones((4, 2)), np.ones(2))
    with pytest.raises(ValueError, match="parameters"):
        feed_forward([1, 2], np.ones((2, 4)), np.ones(3), np.ones((4, 2)), np.ones(2))
    with pytest.raises(ValueError, match="order"):
        residual_sublayer([1, 2], lambda z: z, [1, 1], [0, 0], "other")


@pytest.mark.parametrize("order", ["pre", "post"])
def test_block_matches_official_transformer_encoder_layer(order):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        block = TransformerBlock(8, 2, 16, order).double()
        reference = nn.TransformerEncoderLayer(
            8, 2, 16, dropout=0., batch_first=True, norm_first=order == "pre",
            activation=lambda x: nn.functional.gelu(x, approximate="tanh"),
        ).double()
        reference.self_attn.load_state_dict(block.attention.state_dict())
        reference.linear1.load_state_dict(block.ffn[0].state_dict())
        reference.linear2.load_state_dict(block.ffn[2].state_dict())
        reference.norm1.load_state_dict(block.norm1.state_dict())
        reference.norm2.load_state_dict(block.norm2.state_dict())
        x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    actual, expected = block(x), reference(x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    probe = torch.arange(actual.numel(), dtype=torch.float64).reshape_as(actual)
    ga = torch.autograd.grad(actual, x, probe)[0]
    ge = torch.autograd.grad(expected, x, probe)[0]
    torch.testing.assert_close(ga, ge, atol=1e-11, rtol=1e-11)


def test_zero_sublayers_pre_identity_and_post_not_identity():
    x = torch.tensor([[[1., 2., 5., 7.]]], dtype=torch.float64, requires_grad=True)
    stacks = matched_stacks(seed=8, depth=1, d_model=4, n_heads=2, d_ff=8)
    for blocks in stacks.values():
        with torch.no_grad():
            for module in (blocks[0].attention, blocks[0].ffn):
                for parameter in module.parameters():
                    parameter.zero_()
    pre = stacks["pre_residual"][0](x)
    post = stacks["post_residual"][0](x)
    torch.testing.assert_close(pre, x)
    torch.testing.assert_close(torch.autograd.grad(pre.sum(), x)[0], torch.ones_like(x))
    assert not torch.allclose(post, x)
    torch.testing.assert_close(stacks["pre_no_residual"][0](x), torch.zeros_like(x))


def test_matched_parameters_distinct_storage_layers_and_unchanged_rng():
    before = torch.random.get_rng_state().clone()
    stacks = matched_stacks(9, 2, 8, 2, 16)
    assert torch.equal(before, torch.random.get_rng_state())
    base = stacks["pre_residual"]
    for name, stack in stacks.items():
        for key, value in base.state_dict().items():
            assert torch.equal(value, stack.state_dict()[key])
        if name != "pre_residual":
            assert next(stack.parameters()).data_ptr() != next(base.parameters()).data_ptr()
    assert not torch.equal(base[0].ffn[0].weight, base[1].ffn[0].weight)


@pytest.mark.parametrize("final_norm", [False, True])
def test_small_diagnostics_reproducibility_profiles_and_no_updates(final_norm):
    report = run_diagnostics([10], 3, 8, 2, 16, final_norm)
    repeated = run_diagnostics([10], 3, 8, 2, 16, final_norm)
    assert report["runs"] == repeated["runs"]
    assert report["configuration"]["optimizer_steps"] == 0
    for result in report["runs"][0]["conditions"].values():
        assert len(result["state_rms"]) == len(result["state_gradient_l2"]) == 4
        assert len(result["ffn_expansion_weight_gradient_l2"]) == 3
        assert np.isfinite(result["state_gradient_l2"]).all()
    stack = matched_stacks(10, 2, 8, 2, 16)["pre_residual"]
    before = {key: value.clone() for key, value in stack.state_dict().items()}
    x = torch.arange(48, dtype=torch.float64).reshape(2, 3, 8) / 50
    profile_stack(stack, x, -x, final_norm)
    assert all(torch.equal(value, stack.state_dict()[key]) for key, value in before.items())


@pytest.mark.parametrize("kwargs", [
    {"depth": 0}, {"depth": True}, {"seed": -1}, {"seed": True},
    {"d_model": 7, "n_heads": 2}, {"d_ff": 0},
])
def test_invalid_stack_configuration(kwargs):
    with pytest.raises(ValueError):
        matched_stacks(**kwargs)


def test_invalid_block_and_target():
    block = TransformerBlock(4, 2, 8).double()
    with pytest.raises(ValueError, match="shape"):
        block(torch.ones(2, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        block(torch.full((1, 2, 4), float("nan"), dtype=torch.float64))
    with pytest.raises(ValueError, match="target"):
        profile_stack(nn.ModuleList([block]), torch.ones(1, 2, 4), torch.ones(1, 1, 4))


def test_cli_new_record_and_refuses_overwrite(tmp_path):
    script = Path(__file__).resolve().parents[1] / "example.py"
    output = tmp_path / "diagnostics.json"
    command = [sys.executable, str(script), "--depth", "1", "--seeds", "70", "--output", str(output)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(result.stdout) == report
    assert report["runs"][0]["parameters_matched"]
    original = output.read_bytes()
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0
    assert "output exists" in again.stderr
    assert output.read_bytes() == original
