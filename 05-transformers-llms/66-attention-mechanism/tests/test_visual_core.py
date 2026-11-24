"""Contracts for the visual lab, independent of image appearance."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_lab import (
    cache_fixture, context_fixture, cross_fixture, geometry_state, leakage_fixture,
    mask_fixture, memory_estimates, multihead_fixture, projection_fixture,
    rag_fixture, scaling_experiment, softmax_state, weighted_values,
)


def test_projection_products_and_dimensions_are_actual_multiplications():
    x, matrices, t = projection_fixture()
    assert x.shape == (4, 3)
    for w, actual in zip(matrices, (t.query, t.key, t.value)):
        np.testing.assert_allclose(actual, x@w)
    assert t.output.shape == (4, 2)
    assert not np.allclose(t.query, t.key)


def test_direction_vs_magnitude_and_exact_orthogonality():
    state = geometry_state(90, 0)
    assert state["raw"][0] == pytest.approx(0, abs=1e-14)
    original, changed = geometry_state(), geometry_state(key_gain=3)
    np.testing.assert_allclose(original["cosine"], changed["cosine"])
    assert changed["raw"][0] == pytest.approx(3*original["raw"][0])
    assert not np.allclose(original["weights"], changed["weights"])
    assert np.linalg.norm(original["query"]) == pytest.approx(1)
    np.testing.assert_allclose(np.linalg.norm(original["keys"], axis=1), 1)


@pytest.mark.parametrize("kwargs", [dict(query_norm=0), dict(key_gain=-1), dict(yaw=np.nan)])
def test_geometry_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        geometry_state(**kwargs)


def test_softmax_shift_temperature_and_measured_jacobian():
    original, shifted = softmax_state(), softmax_state([10002, 10001, 10000, 9999])
    np.testing.assert_allclose(original["weights"], shifted["weights"])
    assert softmax_state(temperature=.1)["entropy"] < softmax_state(temperature=5)["entropy"]
    assert np.all(original["exponentials"] <= 1)
    step = 1e-5
    numerical = []
    raw = original["raw"]
    for i in range(4):
        delta = np.eye(4)[i]*step
        # d softmax / d incoming normalized logits = d softmax / d raw × √d_k.
        numerical.append((softmax_state(raw+delta)["weights"]-softmax_state(raw-delta)["weights"])/(2*step)*2)
    assert np.linalg.norm(numerical) == pytest.approx(original["jacobian_norm"], rel=1e-8)


@pytest.mark.parametrize("temperature", [0, -1, np.inf])
def test_temperature_invalid_inputs(temperature):
    with pytest.raises(ValueError):
        softmax_state(temperature=temperature)


def test_values_normalize_and_keep_output_in_convex_bounds():
    weights, weighted, output = weighted_values([4, 3, 2, 1])
    np.testing.assert_allclose(weights, [.4, .3, .2, .1])
    np.testing.assert_allclose(output, weighted.sum(axis=0))
    _, _, shifted = weighted_values(weights, np.array([[-1.5, -.8, 0], [1.5, -.8, .2], [0, 1.6, .3], [0, 0, 2]])+[3, -2, 1])
    np.testing.assert_allclose(shifted-output, [3, -2, 1])


@pytest.mark.parametrize("weights", [[0, 0, 0, 0], [1, -1, 2, 0], [np.nan, 1, 1, 1], [1, 2]])
def test_values_reject_invalid_masses(weights):
    with pytest.raises(ValueError):
        weighted_values(weights)


def test_scaling_statistics_and_jacobian_formula():
    results = scaling_experiment(quick=True)
    assert [r["dimension"] for r in results] == [4, 16, 64, 256, 1024]
    for row in results:
        np.testing.assert_allclose(row["raw_scores"]/np.sqrt(row["dimension"]), row["scaled_scores"])
        assert row["metrics"]["scaled"]["variance"] == pytest.approx(1, abs=.2)
        assert row["metrics"]["raw"]["theory_variance"] == row["dimension"]
        from from_scratch import softmax
        p = softmax(row["raw_scores"])
        direct_norms = [np.linalg.norm(np.diag(w)-np.outer(w, w)) for w in p]
        assert np.mean(direct_norms) == pytest.approx(row["metrics"]["raw"]["jacobian_norm"], abs=1e-12)


def test_causal_and_padding_masks_and_future_control():
    cases = mask_fixture()
    np.testing.assert_array_equal(cases["Padding"].weights[:, 4:], 0)
    np.testing.assert_array_equal(cases["Causal"].weights[np.triu_indices(6, 1)], 0)
    before, after = leakage_fixture()
    np.testing.assert_array_equal(before["Causal"].output[:5], after["Causal"].output[:5])
    assert not np.allclose(before["Full"].output[:5], after["Full"].output[:5])


def test_engineered_context_same_bank_input_different_output():
    river, finance = context_fixture(False), context_fixture(True)
    np.testing.assert_array_equal(river["x"][river["bank"]], finance["x"][finance["bank"]])
    a, b = river["attention"].output[river["bank"]], finance["attention"].output[finance["bank"]]
    assert a[0] > a[1] and b[1] > b[0]


def test_rectangular_sources_and_multihead_concatenation():
    f = cross_fixture()
    assert f["self_attention"].weights.shape == (5, 5)
    assert f["cross_attention"].weights.shape == (3, 5)
    assert f["cross_attention"].output.shape == (3, 2)
    h = multihead_fixture()
    np.testing.assert_array_equal(h["concatenated"], np.concatenate([t.output for t in h["heads"]], axis=1))
    np.testing.assert_allclose(h["output"], h["concatenated"]@h["wo"])
    assert not np.allclose(h["heads"][0].weights, h["heads"][1].weights)


def test_memory_exact_byte_counts_and_no_large_allocation():
    f = memory_estimates([2048, 16384])
    np.testing.assert_array_equal(f["interactions"], [4194304, 268435456])
    np.testing.assert_allclose(f["fp32_mib"], [16, 1024])
    np.testing.assert_allclose(f["fp16_mib"], [8, 512])


def test_rag_cosine_selection_and_incremental_cache_equivalence():
    f = rag_fixture()
    assert f["selected"].tolist() == np.argsort(-f["cosine"])[:2].tolist()
    for state in cache_fixture()["states"]:
        np.testing.assert_allclose(state["cache"].output[0], state["full_output"], rtol=1e-12, atol=1e-12)
        assert state["cached_projection_rows"] == 1
        assert state["recomputed_prefix_rows"] == state["index"]+1
