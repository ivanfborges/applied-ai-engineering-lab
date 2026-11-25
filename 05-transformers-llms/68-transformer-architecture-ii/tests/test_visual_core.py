"""Scientific invariants for intermediate tensors, geometry and theoretical counts."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import MultiHeadAttention
from visual_core import (architecture_counts, complexity_counts, constructed_routing,
                         default_case, diversity_case, kv_bytes, numerical_report,
                         online_attention, pair_geometry, pca_view, principal_angles,
                         scaling_samples, subspace_cases)
from visual_html import serializable


def test_trace_does_not_change_forward_and_preserves_every_stage():
    layer, traces = default_case()
    for mode, trace in traces.items():
        output, weights = layer.forward(trace["x"], causal=mode == "causal")
        np.testing.assert_array_equal(trace["output"], output)
        np.testing.assert_array_equal(trace["attention"], weights)
        for kind in ("q", "k", "v"):
            projected = trace[kind].transpose(0, 2, 1, 3).reshape(1, 6, 16)
            np.testing.assert_allclose(projected, trace["x"] @ getattr(layer, "w" + kind), atol=1e-14)
        np.testing.assert_allclose(trace["raw"], trace["q"] @ trace["k"].swapaxes(-1, -2))
        np.testing.assert_allclose(trace["scaled"], trace["raw"] / 2)
        np.testing.assert_array_equal(np.isneginf(trace["masked"]), trace["blocked"])
        np.testing.assert_allclose(trace["heads"], trace["attention"] @ trace["v"])
        np.testing.assert_allclose(trace["output"], trace["concat"] @ layer.wo)
        blocks = sum(trace["heads"][:, h] @ layer.wo[h*4:(h+1)*4] for h in range(4))
        np.testing.assert_allclose(blocks, output, atol=1e-14)
    with pytest.raises(ValueError, match="return_trace"):
        layer.forward(traces["causal"]["x"], return_trace=1)


def test_projection_uses_every_input_dimension_and_fused_columns_agree():
    layer, traces = default_case()
    x = traces["bidirectional"]["x"]
    for head in range(4):
        weight = layer.wq[:, head*4:(head+1)*4]
        assert np.all(np.linalg.norm(weight, axis=1) > 0)
        np.testing.assert_allclose(x @ weight, traces["bidirectional"]["q"][:, head], atol=1e-14)


def test_progressive_prefix_rows_normalize_and_causal_outputs_match_full_prefix():
    layer, traces = default_case()
    x = traces["causal"]["x"]
    for length in range(1, 7):
        prefix, maps = layer.forward(x[:, :length], causal=True)
        np.testing.assert_allclose(maps.sum(-1), 1, atol=1e-14)
        np.testing.assert_allclose(prefix, traces["causal"]["output"][:, :length], atol=1e-13)


def test_exact_pair_geometry_and_undefined_zero_vector_angle():
    distances, angles = pair_geometry(np.array([[1., 0], [3., 0], [0, 1], [0, 0]]))
    assert distances[0, 1] == 2
    assert angles[0, 1] == 0 and angles[0, 2] == 90
    assert np.isnan(angles[0, 3])


def test_pca_full_rank_distances_and_rank_reduction_warning_evidence():
    x = np.random.default_rng(0).normal(size=(10, 3))
    projected, fraction = pca_view(x)
    np.testing.assert_allclose(pair_geometry(x)[0], pair_geometry(projected)[0], atol=1e-14)
    assert fraction == pytest.approx(1)
    higher = np.random.default_rng(1).normal(size=(20, 8))
    _, retained = pca_view(higher)
    assert 0 < retained < 1
    for name, case in subspace_cases().items():
        assert len(case["entries"]) == 12
        for entry in case["entries"]:
            np.testing.assert_allclose(case["x"] @ entry["weight"], entry["projected"])


def test_principal_angles_overlap_and_orthogonal_constructed_spans():
    eye = np.eye(6)
    np.testing.assert_allclose(principal_angles(eye[:, :2], eye[:, :2]), 0, atol=1e-6)
    np.testing.assert_allclose(principal_angles(eye[:, :2], eye[:, 2:4]), 90)
    angles = principal_angles(eye[:, :2], eye[:, [1, 2]])
    np.testing.assert_allclose(angles, [0, 90])


def test_paired_iid_dot_product_variance_has_expected_scaling():
    first, second = scaling_samples(samples=8000), scaling_samples(samples=8000)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a["raw"], b["raw"])
        np.testing.assert_allclose(a["raw"] / np.sqrt(a["dimension"]), a["scaled"])
        # Broad Monte Carlo tolerance; this is a numerical property, not a model ranking.
        assert abs(a["scaled_variance"] - 1) < .12
        assert a["raw_variance"] == pytest.approx(a["scaled_variance"] * a["dimension"])


def test_constructed_independent_channels_are_outside_fixed_shared_mixture():
    case = constructed_routing()
    assert case["routed"][0] > .999 and case["routed"][1] > 9.999
    np.testing.assert_allclose(case["shared_curve"][:, 0] + case["shared_curve"][:, 1] / 10, 1)
    assert case["routed"][0] + case["routed"][1] / 10 > 1.99


def test_identical_maps_distinct_values_and_ablation_contribution_identity():
    data = diversity_case()
    assert data["identical_map_error"] == 0
    assert data["distinct_value_output_difference"] > 1e-3
    np.testing.assert_allclose(data["contributions"].sum(0), data["trace"]["output"][0], atol=1e-14)
    np.testing.assert_allclose(data["map_similarity"][:2, :2], 1)
    assert data["output_similarity"][0, 1] < .99


@pytest.mark.parametrize("kvheads", [1, 2, 4, 8])
def test_shared_kv_architecture_counts_and_assignment(kvheads):
    counts = architecture_counts(8, kvheads, 4)
    assert counts["d_model"] == 32
    assert counts["parameters"] == 2 * 32**2 + 2 * 32 * kvheads * 4
    assert np.bincount(counts["assignment"]).tolist() == [8 // kvheads] * kvheads
    if kvheads == 8:
        assert counts["parameters"] == 4 * 32**2


@pytest.mark.parametrize("bits", [4, 8, 16, 32])
def test_cache_formula_precision_and_scaling(bits):
    expected = 2 * 2 * 3 * 16 * 4 * 8 * bits // 8
    assert kv_bytes(2, 3, 16, 4, 8, bits) == expected
    assert kv_bytes(4, 3, 16, 4, 8, bits) == 2 * expected
    assert kv_bytes(1, 32, 8192, 32, 128, 16) == 4 * 2**30


def test_prefill_vs_decode_and_map_storage_growth():
    a, b = complexity_counts(128, 64, 4), complexity_counts(256, 64, 4)
    assert b["attention_macs"] == 4 * a["attention_macs"]
    assert b["map_bytes"] == 4 * a["map_bytes"]
    assert b["projection_macs"] == 2 * a["projection_macs"]
    assert b["decode_attention_macs"] == 2 * a["decode_attention_macs"]


@pytest.mark.parametrize("block_size", [1, 2, 3, 5, 20])
def test_online_softmax_matches_dense_for_rectangular_queries_and_large_logits(block_size):
    rng = np.random.default_rng(9)
    q, k, v = rng.normal(size=(3, 4)) * 20, rng.normal(size=(7, 4)) * 20, rng.normal(size=(7, 5))
    scores = q @ k.T / 2
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    actual, stages = online_attention(q, k, v, block_size)
    np.testing.assert_allclose(actual, weights @ v, atol=1e-13)
    assert stages[-1]["end"] == 7


@pytest.mark.parametrize("operation", [
    lambda: architecture_counts(8, 3, 4), lambda: architecture_counts(True, 1, 4),
    lambda: kv_bytes(0, 3, 16, 4, 8), lambda: kv_bytes(1, 3, 16, 4, 8, 3),
    lambda: complexity_counts(10, 7, 2), lambda: online_attention(np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2)), 0),
])
def test_invalid_count_and_tiling_configurations(operation):
    with pytest.raises(ValueError):
        operation()


def test_executed_numerical_report_and_strict_json_mask_encoding():
    report = numerical_report(samples=1000)
    assert max(row["output_max_abs_error"] for row in report["parity"]) < 1e-12
    assert report["causal_prefix_max_abs_change"] < 1e-12
    assert report["bidirectional_prefix_max_abs_change"] > 1e-3
    assert report["online_softmax_max_abs_error"] < 1e-12
    assert serializable(np.array([1, -np.inf, np.nan])) == [1, None, None]
