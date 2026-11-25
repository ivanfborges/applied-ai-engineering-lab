"""Information-flow, padding and teacher-forcing contracts for Day 71."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import attention, shift_right, visibility
from example import causal_block_mask, run_experiment


@pytest.mark.parametrize("n,m", [(1, 1), (3, 5), (4, 20)])
def test_cross_attention_uniform_scores_average_only_valid_values(n, m):
    valid = np.arange(m) % 2 == 0
    values = np.arange(m * 2).reshape(m, 2)
    output, weights = attention(np.zeros((n, 4)), np.ones((m, 4)), values,
                                visibility(n, m, key_valid=valid))
    np.testing.assert_allclose(output, np.broadcast_to(values[valid].mean(axis=0), (n, 2)))
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert np.all(weights[:, ~valid] == 0)


def test_causal_prefix_matches_truncated_forward_and_ignores_future_changes():
    rng = np.random.default_rng(71)
    q, k, v = (rng.normal(size=(5, 4)) for _ in range(3))
    output, weights = attention(q, k, v, visibility(5, 5, causal=True))
    changed_q, changed_k, changed_v = q.copy(), k.copy(), v.copy()
    for matrix in (changed_q, changed_k, changed_v):
        matrix[-1] += 100
    changed, _ = attention(changed_q, changed_k, changed_v, visibility(5, 5, causal=True))
    np.testing.assert_allclose(output[:-1], changed[:-1], atol=1e-12, rtol=0)
    for i in range(5):
        prefix, _ = attention(q[i:i+1], k[:i+1], v[:i+1], visibility(1, i+1))
        np.testing.assert_allclose(output[i], prefix[0], atol=1e-12)
    assert np.all(weights[np.triu_indices(5, 1)] == 0)


def test_masked_keys_and_values_cannot_change_cross_output():
    rng = np.random.default_rng(72)
    q, k, v = rng.normal(size=(3, 4)), rng.normal(size=(5, 4)), rng.normal(size=(5, 6))
    allowed = visibility(3, 5, key_valid=[True, True, True, True, False])
    before, _ = attention(q, k, v, allowed)
    k[-1] += 200
    v[-1] -= 200
    after, _ = attention(q, k, v, allowed)
    np.testing.assert_allclose(before, after, atol=1e-12, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_numpy_matches_real_sdpa_with_permitted_boolean_mask(causal):
    rng = np.random.default_rng(3)
    q, k, v = (rng.normal(size=(5, 4)) for _ in range(3))
    allowed = visibility(5, 5, causal=causal, key_valid=[True, True, True, True, False])
    expected, _ = attention(q, k, v, allowed)
    actual = F.scaled_dot_product_attention(
        *[torch.from_numpy(matrix)[None, None] for matrix in (q, k, v)],
        attn_mask=torch.from_numpy(allowed), dropout_p=0.0,
    )
    np.testing.assert_allclose(actual[0, 0].numpy(), expected, atol=1e-12, rtol=0)


def test_extreme_finite_logits_are_stable():
    output, weights = attention([[1000.0]], [[1000.0], [-1000.0]], [[3.0], [9.0]], visibility(1, 2))
    np.testing.assert_array_equal(weights, [[1, 0]])
    np.testing.assert_array_equal(output, [[3]])


@pytest.mark.parametrize("kwargs", [
    {"query_length": 0, "key_length": 2},
    {"query_length": True, "key_length": 2},
    {"query_length": 1.5, "key_length": 2},
    {"query_length": 2, "key_length": 3, "causal": True},
    {"query_length": 2, "key_length": 2, "causal": "yes"},
    {"query_length": 2, "key_length": 2, "key_valid": [1, 0]},
    {"query_length": 2, "key_length": 2, "key_valid": [True]},
    {"query_length": 2, "key_length": 2, "key_valid": [False, False]},
    {"query_length": 2, "key_length": 2, "causal": True, "key_valid": [False, True]},
])
def test_invalid_visibility_is_explicit(kwargs):
    with pytest.raises(ValueError):
        visibility(**kwargs)


@pytest.mark.parametrize("q,k,v,mask", [
    ([], [[1]], [[1]], [[True]]),
    ([[np.nan]], [[1]], [[1]], [[True]]),
    ([[1]], [[np.inf]], [[1]], [[True]]),
    ([[1]], [[1]], [[1j]], [[True]]),
    ([[1, 2]], [[1]], [[1]], [[True]]),
    ([[1]], [[1]], [[1], [2]], [[True]]),
    ([[1]], [[1]], [[1]], [[1]]),
    ([[1]], [[1]], [[1]], [[False]]),
    ([[1]], [[1]], [[1]], [[True, True]]),
    ([[1e308]], [[1e308]], [[1]], [[True]]),
])
def test_invalid_attention_is_explicit(q, k, v, mask):
    with pytest.raises(ValueError):
        attention(q, k, v, mask)


def test_shifted_labels_do_not_expose_current_target_and_preserve_eos():
    labels = np.array([4, 7, 2])
    shifted = shift_right(labels, 0)
    np.testing.assert_array_equal(shifted, [0, 4, 7])
    np.testing.assert_array_equal(labels, [4, 7, 2])
    np.testing.assert_array_equal(shift_right(np.array([2]), 0), [0])
    np.testing.assert_array_equal(shift_right(np.array([4, 2], dtype=np.uint8), 1000), [1000, 4])
    large = 2**53 + 1
    np.testing.assert_array_equal(shift_right(np.array([large, 2], dtype=np.uint64), 0), [0, large])


@pytest.mark.parametrize("labels,start", [([], 0), ([1.5], 0), ([-1], 0), ([[1]], 0), ([True], 0), ([1], -1), ([1], True), ([1], 1.5), ([1], 2**64)])
def test_invalid_shift_is_explicit(labels, start):
    with pytest.raises(ValueError):
        shift_right(labels, start)


@pytest.fixture(scope="module")
def report():
    return run_experiment()


def test_real_blocks_enforce_both_causal_and_padding_boundaries(report):
    deltas = report["positionwise_l2_deltas"]
    assert all(report["checks"].values())
    assert report["output_shapes"] == {"encoder": [1, 5, 16], "causal": [1, 5, 16], "seq2seq": [1, 3, 16]}
    assert max(deltas["causal_after_source_change"][:-1]) <= 1e-12
    assert max(deltas["seq2seq_after_target_change"][:-1]) <= 1e-12
    # Positive controls make a disconnected or constant-output implementation fail.
    assert min(deltas["encoder_after_source_change"][:-1]) > 1e-6
    assert min(deltas["seq2seq_after_source_change"]) > 1e-6
    assert deltas["causal_after_source_change"][-1] > 1e-6
    assert deltas["seq2seq_after_target_change"][-1] > 1e-6
    np.testing.assert_array_equal(causal_block_mask(3).numpy(), np.triu(np.ones((3, 3), dtype=bool), 1))


def test_same_seed_reproduces_numerical_report(report):
    rerun = run_experiment()
    assert rerun["positionwise_l2_deltas"] == report["positionwise_l2_deltas"]
    assert rerun["numpy_sdpa_max_abs_error"] == report["numpy_sdpa_max_abs_error"]


@pytest.mark.parametrize("seed", [-1, True, 3.5, 2**32])
def test_invalid_seed_is_explicit(seed):
    with pytest.raises(ValueError):
        run_experiment(seed)
