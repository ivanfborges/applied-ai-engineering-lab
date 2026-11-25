"""Independent small-matrix oracles and architectural controls on synthetic data."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import absolute_position_lookup, apply_rope, causal_alibi_bias, sinusoidal_encoding
from example import cache_offset_control, permutation_controls


def test_sinusoidal_scalar_formula_and_shift_rotation():
    positions = np.array([0, 1, 7, 100])
    dim, base, offset = 8, 100.0, 3
    actual = sinusoidal_encoding(positions, dim, base)
    expected = np.empty_like(actual)
    shifted = np.empty_like(actual)
    for row, position in enumerate(positions):
        for pair in range(dim // 2):
            frequency = base ** (-2 * pair / dim)
            expected[row, 2 * pair] = np.sin(position * frequency)
            expected[row, 2 * pair + 1] = np.cos(position * frequency)
            angle = offset * frequency
            transform = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            shifted[row, 2 * pair:2 * pair + 2] = transform @ expected[row, 2 * pair:2 * pair + 2]
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    np.testing.assert_array_equal(actual[0], [0, 1, 0, 1, 0, 1, 0, 1])
    np.testing.assert_allclose(sinusoidal_encoding(positions + offset, dim, base), shifted, atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=-1)**2, dim / 2, atol=1e-13)


def test_pure_sinusoidal_dot_products_depend_on_displacement():
    positions = np.array([0, 5, 100, 105])
    pe = sinusoidal_encoding(positions, 16)
    expected = sum(np.cos(5 * 10000.0 ** (-2 * pair / 16)) for pair in range(8))
    assert pe[0] @ pe[1] == pytest.approx(expected, abs=1e-13)
    assert pe[2] @ pe[3] == pytest.approx(expected, abs=1e-13)


def test_table_lookup_repeats_positions_and_does_not_alias_or_extrapolate():
    table = np.arange(24).reshape(6, 4)
    selected = absolute_position_lookup(table, [3, 0, 3])
    np.testing.assert_array_equal(selected, table[[3, 0, 3]])
    selected[0, 0] = -10
    assert table[3, 0] == 12
    with pytest.raises(ValueError, match="exceeds"):
        absolute_position_lookup(table, [6])


def test_rope_against_block_diagonal_rotation_and_no_input_mutation():
    x = np.random.default_rng(1).normal(size=(3, 8))
    original = x.copy()
    positions = np.array([0, 3, 27])
    expected = np.empty_like(x)
    for row, position in enumerate(positions):
        rotation = np.zeros((8, 8))
        for pair in range(4):
            angle = position / 10000.0 ** (2 * pair / 8)
            block = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            rotation[2*pair:2*pair+2, 2*pair:2*pair+2] = block
        expected[row] = rotation @ x[row]
    actual = apply_rope(x, positions)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-13)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(actual[0], x[0])


@pytest.mark.parametrize("shift", [0, 3, 1000, 10000, 1000000])
def test_rope_relative_dot_product_with_fixed_content(shift):
    q, k = np.random.default_rng(2).normal(size=(2, 1, 16))
    actual = (apply_rope(q, [10 + shift]) @ apply_rope(k, [15 + shift]).T).item()
    relative = (q @ apply_rope(k, [5]).T).item()
    assert actual == pytest.approx(relative, abs=1e-9, rel=0)


def test_rope_is_not_a_monotonic_recency_penalty():
    q = np.array([[1., 0.]])
    near = (apply_rope(q, [0]) @ apply_rope(q, [3]).T).item()
    farther = (apply_rope(q, [0]) @ apply_rope(q, [6]).T).item()
    assert farther > near


def test_alibi_sign_mask_head_slopes_and_rectangular_cache_offsets():
    query, keys, slopes = [3, 4], [0, 2, 3, 4], [.25, 1.]
    bias = causal_alibi_bias(query, keys, slopes)
    assert bias.shape == (2, 2, 4)
    for head, slope in enumerate(slopes):
        for row, position in enumerate(query):
            for column, key in enumerate(keys):
                if key > position:
                    assert bias[head, row, column] == -np.inf
                else:
                    assert bias[head, row, column] == -slope * (position - key)
    full = causal_alibi_bias(np.arange(5), np.arange(5), slopes)
    np.testing.assert_array_equal(bias, full[:, [3, 4]][:, :, [0, 2, 3, 4]])
    shifted = causal_alibi_bias(np.array(query) + 100, np.array(keys) + 100, slopes)
    np.testing.assert_array_equal(shifted, bias)


def test_alibi_flat_content_distribution_matches_geometric_series():
    slope, length = .5, 16
    bias = causal_alibi_bias([15], np.arange(length), [slope])[0, 0]
    actual = np.exp(bias) / np.exp(bias).sum()
    ratio = np.exp(-slope)
    expected = (1 - ratio) * ratio ** np.arange(15, -1, -1) / (1 - ratio ** length)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert np.all(np.diff(actual) > 0)
    assert actual.sum() == pytest.approx(1)


def test_pytorch_fixed_and_joint_position_permutations_and_mask_control():
    metrics = permutation_controls()
    assert metrics["no_position_equivariance_max_error"] < 1e-12
    assert metrics["joint_token_position_permutation_max_error"] < 1e-12
    assert metrics["numpy_torch_sinusoidal_max_error"] < 1e-13
    assert metrics["fixed_position_permutation_max_difference"] > 1e-3
    assert metrics["fixed_causal_mask_permutation_max_difference"] > 1e-3


def test_correct_cache_offsets_and_intentionally_reset_ids():
    metrics = cache_offset_control()
    assert metrics["correct_cache_offset_max_error"] < 1e-12
    assert metrics["reset_new_position_ids_max_difference"] > 1e-3


@pytest.mark.parametrize("positions", [[], [[0, 1]], [0., 1.], [-1, 0], [True], [2**53], [np.nan]])
def test_invalid_positions_rejected(positions):
    with pytest.raises(ValueError):
        sinusoidal_encoding(positions, 4)


@pytest.mark.parametrize("dim", [0, -2, 3, 4., True])
def test_invalid_dimensions_rejected(dim):
    with pytest.raises(ValueError):
        sinusoidal_encoding([0], dim)


@pytest.mark.parametrize("base", [1, 0, -1, np.inf, np.nan, True, [10000]])
def test_invalid_bases_rejected(base):
    with pytest.raises(ValueError):
        sinusoidal_encoding([0], 4, base)


@pytest.mark.parametrize("x, positions", [([[1., 2.]], [0, 1]), ([[1., 2., 3.]], [0]),
                                          ([[np.inf, 0]], [0]), ([1., 2.], [0]),
                                          ([[1j, 0]], [0]), (np.empty((0, 4)), [0])])
def test_invalid_rope_inputs_rejected(x, positions):
    with pytest.raises(ValueError):
        apply_rope(x, positions)


@pytest.mark.parametrize("slopes", [[], [0], [-1], [np.nan], [np.inf], [[.1]], [True]])
def test_invalid_alibi_slopes_rejected(slopes):
    with pytest.raises(ValueError):
        causal_alibi_bias([1], [0, 1], slopes)


def test_alibi_rejects_fully_masked_query_and_overflow():
    with pytest.raises(ValueError, match="visible"):
        causal_alibi_bias([0], [1, 2], [.1])
    with pytest.raises(ValueError, match="overflow"):
        causal_alibi_bias([100], [0], [1e308])


def test_cli_writes_finite_json_outside_working_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / "example.py"
    record = tmp_path / "run.json"
    completed = subprocess.run([sys.executable, str(script), "--output", str(record)],
                               cwd=tmp_path, capture_output=True, text=True, check=True)
    result = json.loads(record.read_text(encoding="utf-8"))
    assert json.loads(completed.stdout) == result
    assert result["interpretation_status"].startswith("author review required")
    assert result["attention_configuration"]["training"] is False


def test_cli_preserves_existing_dated_record(tmp_path):
    script = Path(__file__).resolve().parents[1] / "example.py"
    record = tmp_path / "existing.json"
    original = '{"historical": true}\n'
    record.write_text(original, encoding="utf-8")
    completed = subprocess.run([sys.executable, str(script), "--output", str(record)],
                               cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "preserve the record" in completed.stderr
    assert record.read_text(encoding="utf-8") == original
