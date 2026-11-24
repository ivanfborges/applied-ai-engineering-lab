"""Numerical contracts and a real CPU PyTorch reference; no assets or downloads."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import scaled_dot_product_attention, softmax


def fixture():
    q = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    return q, q.copy(), np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])


def test_hand_calculation_and_convex_bounds():
    q, k, v = fixture()
    output, weights = scaled_dot_product_attention(q[:1], k, v)
    denominator = np.exp(1 / np.sqrt(2)) + 1 + np.exp(-1 / np.sqrt(2))
    expected = np.array([np.exp(1 / np.sqrt(2)), 1, np.exp(-1 / np.sqrt(2))]) / denominator
    np.testing.assert_allclose(weights[0], expected, atol=1e-14)
    np.testing.assert_allclose(output[0], [2 * expected[0] + expected[2], 2 * expected[1] + expected[2]])
    assert np.all(output >= v.min(axis=0)) and np.all(output <= v.max(axis=0))


def test_cross_attention_shapes_and_uniform_queries():
    output, weights = scaled_dot_product_attention(np.zeros((2, 4)), np.ones((5, 4)), np.arange(15).reshape(5, 3))
    assert output.shape == (2, 3) and weights.shape == (2, 5)
    np.testing.assert_allclose(weights, 0.2)
    np.testing.assert_allclose(output, [[6, 7, 8], [6, 7, 8]])


def test_mask_normalization_and_value_independence():
    q, k, v = fixture()
    mask = np.array([True, False, True])
    output, weights = scaled_dot_product_attention(q, k, v, mask)
    shifted, other_weights = scaled_dot_product_attention(q, k, v + [3, -2], mask)
    np.testing.assert_array_equal(weights[:, 1], 0)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert (weights >= 0).all()
    np.testing.assert_array_equal(weights, other_weights)
    np.testing.assert_allclose(shifted - output, np.tile([3, -2], (3, 1)))


def test_causal_future_invariance_and_unmasked_sensitivity():
    q, k, v = fixture()
    mask = np.tri(3, dtype=bool)
    before, weights = scaled_dot_product_attention(q, k, v, mask)
    full_before, _ = scaled_dot_product_attention(q, k, v)
    k[-1], v[-1] = [20, -20], [100, -100]
    after, _ = scaled_dot_product_attention(q, k, v, mask)
    full_after, _ = scaled_dot_product_attention(q, k, v)
    np.testing.assert_array_equal(before[:2], after[:2])
    np.testing.assert_array_equal(weights[np.triu_indices(3, 1)], 0)
    assert not np.allclose(full_before[:2], full_after[:2])


def test_unmasked_permutation_equivariance():
    rng = np.random.default_rng(66)
    x = rng.normal(size=(5, 6))
    q, k, v = [x @ rng.normal(size=(6, d)) for d in (3, 3, 4)]
    order = [3, 0, 4, 1, 2]
    before, _ = scaled_dot_product_attention(q, k, v)
    after, _ = scaled_dot_product_attention(q[order], k[order], v[order])
    np.testing.assert_allclose(after, before[order], atol=1e-14)


def test_stable_softmax_extremes_shift_and_masked_maximum():
    np.testing.assert_allclose(softmax([[10000, 9999]]), softmax([[1, 0]]))
    np.testing.assert_allclose(softmax([[1e308, -1e308]]), [[1, 0]])
    np.testing.assert_allclose(softmax([[1e308, 0, 0]], [[False, True, True]]), [[0, 0.5, 0.5]])


@pytest.mark.parametrize("mask", [np.zeros((3, 3), bool), [[True, False, False], [False, False, False], [True, True, True]], [1, 0, 1], np.ones((2, 2), bool)])
def test_invalid_masks(mask):
    with pytest.raises(ValueError):
        scaled_dot_product_attention(*fixture(), allowed=mask)


@pytest.mark.parametrize("array", [[], [1, 2], np.zeros((0, 2)), np.zeros((1, 2, 3)), [[np.nan, 0]], [[np.inf, 0]], [[1j, 0]], [["1", "2"]]])
def test_invalid_matrices(array):
    _, k, v = fixture()
    with pytest.raises(ValueError):
        scaled_dot_product_attention(array, k, v)


@pytest.mark.parametrize("q,k,v", [(np.ones((2, 3)), np.ones((3, 2)), np.ones((3, 1))), (np.ones((2, 3)), np.ones((3, 3)), np.ones((2, 1))), (np.full((2, 2), 1e308), np.full((3, 2), 1e308), np.ones((3, 1)))])
def test_incompatible_shapes_or_overflow(q, k, v):
    with pytest.raises(ValueError):
        scaled_dot_product_attention(q, k, v)


@pytest.mark.integration
@pytest.mark.parametrize("n,m,dk,dv", [(3, 3, 2, 2), (2, 5, 4, 3)])
def test_pytorch_sdpa_reference(n, m, dk, dv):
    import torch
    from torch.nn.functional import scaled_dot_product_attention as sdpa

    torch.set_num_threads(1)
    rng = np.random.default_rng(66)
    q, k, v = rng.normal(size=(n, dk)), rng.normal(size=(m, dk)), rng.normal(size=(m, dv))
    for mask in (None, np.arange(m)[None, :] <= np.arange(n)[:, None]):
        expected, _ = scaled_dot_product_attention(q, k, v, mask)
        tensors = [torch.from_numpy(x)[None, None] for x in (q, k, v)]
        with torch.inference_mode():
            actual = sdpa(*tensors, attn_mask=None if mask is None else torch.from_numpy(mask), dropout_p=0.0)
        np.testing.assert_allclose(actual.numpy()[0, 0], expected, rtol=1e-12, atol=1e-12)
