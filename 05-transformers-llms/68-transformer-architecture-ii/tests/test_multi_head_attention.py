"""Independent head-loop oracle, controlled routing, masks and framework parity."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import MultiHeadAttention
from example import entropy, run, torch_reference


@pytest.mark.parametrize("heads", [1, 2, 4, 8])
def test_vectorized_matches_explicit_head_loop(heads):
    x = np.random.default_rng(1).normal(size=(2, 5, 8))
    layer = MultiHeadAttention(8, heads)
    actual, maps = layer.forward(x)
    expected_heads = []
    for head in range(heads):
        columns = slice(head * layer.d_head, (head + 1) * layer.d_head)
        q, k, v = (x @ weight[:, columns] for weight in (layer.wq, layer.wk, layer.wv))
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(layer.d_head)
        weights = np.exp(scores - scores.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        np.testing.assert_allclose(maps[:, head], weights, atol=1e-14)
        expected_heads.append(weights @ v)
    np.testing.assert_allclose(actual, np.concatenate(expected_heads, axis=-1) @ layer.wo, atol=1e-14)
    assert actual.shape == x.shape and maps.shape == (2, heads, 5, 5)


def test_uniform_maps_can_have_distinct_value_contributions_and_output_mixing():
    layer = MultiHeadAttention(4, 2)
    layer.wq = layer.wk = np.zeros((4, 4))
    layer.wv = np.eye(4)
    layer.wo = np.array([[1., 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]])
    x = np.array([[[2., 4, 10, 20], [6, 8, 30, 40]]])
    output, maps = layer.forward(x)
    np.testing.assert_array_equal(maps, .5)
    np.testing.assert_allclose(output, [[[24, 36, 6, 4], [24, 36, 6, 4]]])
    # Identical maps do not imply identical values or interchangeable heads.
    assert not np.array_equal(x[..., :2].mean(1), x[..., 2:].mean(1))


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("heads", [1, 4])
def test_numpy_pytorch_parity_with_batch_specific_padding_and_head_masks(causal, heads):
    x = np.random.default_rng(2).normal(size=(2, 5, 8))
    layer = MultiHeadAttention(8, heads)
    mask = np.zeros((2, heads, 5, 5), dtype=bool)
    mask[1, -1, 2:, 1] = True
    padding = np.zeros((2, 5), dtype=bool)
    padding[0, -1] = True
    kwargs = dict(causal=causal, blocked_mask=mask, key_padding_mask=padding)
    actual, weights = layer.forward(x, **kwargs)
    expected, expected_weights = torch_reference(layer, x, **kwargs)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(weights, expected_weights, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-14)
    np.testing.assert_array_equal(weights[0, :, :, -1], 0)
    np.testing.assert_array_equal(weights[1, -1, 2:, 1], 0)
    if causal:
        np.testing.assert_array_equal(np.triu(weights, 1), 0)


def test_causal_prefix_invariance_and_unmasked_positive_control():
    x = np.random.default_rng(3).normal(size=(2, 6, 8))
    changed = x.copy()
    changed[:, 3:] += 5
    layer = MultiHeadAttention(8, 2)
    first, _ = layer.forward(x, causal=True)
    second, _ = layer.forward(changed, causal=True)
    np.testing.assert_allclose(first[:, :3], second[:, :3], atol=1e-13)
    full, _ = layer.forward(x)
    other, _ = layer.forward(changed)
    assert np.max(abs(full[:, :3] - other[:, :3])) > 1e-5


def test_blocked_key_perturbation_and_padded_queries_still_produce_outputs():
    x = np.random.default_rng(4).normal(size=(1, 4, 8))
    changed = x.copy()
    changed[:, -1] += 100
    layer = MultiHeadAttention(8, 2)
    padding = np.array([[False, False, False, True]])
    first, maps = layer.forward(x, key_padding_mask=padding)
    second, _ = layer.forward(changed, key_padding_mask=padding)
    np.testing.assert_allclose(first[:, :-1], second[:, :-1], atol=1e-13)
    np.testing.assert_allclose(maps[:, :, -1].sum(-1), 1)
    assert np.isfinite(first[:, -1]).all()


def test_shared_2d_mask_and_hand_computed_single_visible_key():
    layer = MultiHeadAttention(4, 2)
    layer.wv = layer.wo = np.eye(4)
    x = np.arange(24.).reshape(2, 3, 4)
    mask = np.tile([True, False, True], (3, 1))
    output, maps = layer.forward(x, blocked_mask=mask)
    np.testing.assert_array_equal(maps[..., 1], 1)
    np.testing.assert_array_equal(output, np.repeat(x[:, 1:2], 3, axis=1))


def test_seed_fixed_weights_parameter_count_and_repeated_calls():
    one, many = MultiHeadAttention(8, 1, 5), MultiHeadAttention(8, 4, 5)
    for name in ("wq", "wk", "wv", "wo"):
        np.testing.assert_array_equal(getattr(one, name), getattr(many, name))
    assert sum(getattr(many, name).size for name in ("wq", "wk", "wv", "wo")) == 4 * 8 ** 2
    x = np.ones((1, 2, 8))
    before = many.wq.copy()
    np.testing.assert_array_equal(many.forward(x)[0], many.forward(x)[0])
    np.testing.assert_array_equal(before, many.wq)


def test_large_finite_logits_stay_normalized_and_overflow_is_explicit():
    layer = MultiHeadAttention(2, 1)
    layer.wq = layer.wk = layer.wv = layer.wo = np.eye(2)
    output, maps = layer.forward(np.array([[[100., 0], [-100, 0]]]))
    assert np.isfinite(output).all()
    np.testing.assert_allclose(maps.sum(-1), 1)
    with pytest.raises(ValueError, match="float64"):
        layer.forward(np.full((1, 2, 2), 1e308))


@pytest.mark.parametrize("width,heads", [(0, 1), (-8, 2), (8, 0), (8, 3), (True, 1), (8, False), (8., 2)])
def test_invalid_configuration(width, heads):
    with pytest.raises(ValueError):
        MultiHeadAttention(width, heads)


@pytest.mark.parametrize("x", [np.ones((2, 8)), np.ones((1, 2, 7)), np.ones((1, 0, 8)), np.ones((0, 2, 8)), np.full((1, 2, 8), np.nan), np.full((1, 2, 8), np.inf), np.ones((1, 2, 8), dtype=complex), np.full((1, 2, 8), "1")])
def test_invalid_input(x):
    with pytest.raises(ValueError):
        MultiHeadAttention(8, 2).forward(x)


@pytest.mark.parametrize("kwargs", [
    {"blocked_mask": np.zeros((3, 3))},
    {"blocked_mask": np.zeros((1, 3, 3), dtype=bool)},
    {"blocked_mask": np.zeros((2, 2), dtype=bool)},
    {"key_padding_mask": np.zeros((3,), dtype=bool)},
    {"key_padding_mask": np.zeros((1, 3))},
    {"blocked_mask": np.ones((3, 3), dtype=bool)},
    {"key_padding_mask": np.ones((1, 3), dtype=bool)},
    {"causal": True, "key_padding_mask": np.array([[True, False, False]])},
    {"causal": 1}, {"scaled": "yes"},
])
def test_invalid_masks_and_flags(kwargs):
    with pytest.raises(ValueError):
        MultiHeadAttention(8, 2).forward(np.ones((1, 3, 8)), **kwargs)


def test_entropy_and_unscaled_control():
    assert entropy(np.array([[[[1., 0]]]])) == 0
    assert entropy(np.full((1, 1, 2, 4), .25)) == pytest.approx(np.log(4))
    layer = MultiHeadAttention(8, 2)
    x = np.random.default_rng(8).normal(size=(1, 4, 8))
    _, scaled = layer.forward(x)
    _, unscaled = layer.forward(x, scaled=False)
    assert entropy(unscaled) < entropy(scaled)


def test_experiment_record_and_deterministic_measurements():
    first, second = run(), run()
    for experiment in ("parity", "causal_control", "head_and_scaling_sweep"):
        assert first[experiment] == second[experiment]
        assert {"hypothesis", "result", "interpretation_candidate", "limitation"} <= first[experiment].keys()
    assert first["review_status"] == "author review required"
    assert {row["projection_parameters"] for row in first["head_and_scaling_sweep"]["result"]} == {16384}


def test_cli_writes_new_record_and_preserves_existing_file(tmp_path):
    output = tmp_path / "record.json"
    script = Path(__file__).resolve().parents[1] / "example.py"
    command = [sys.executable, str(script), "--output", str(output)]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    original = output.read_bytes()
    assert json.loads(original)["configuration"]["seed"] == 68
    duplicate = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert duplicate.returncode != 0 and "FileExistsError" in duplicate.stderr
    assert output.read_bytes() == original
