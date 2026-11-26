"""Numerical and experiment protocols; no rendering or external resources."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import decoding_distribution, entropy_bits, generate, softmax
from visual_core import (GROUPS, LabConfig, PRESETS, beam_search, binary_logits, compute_group,
                         exhaustive_search, kv_budget, kv_equivalence, landscape, sample_path,
                         sequence_agreement, serializable, stages)


def test_softmax_rank_and_entropy_temperature_identity():
    logits = PRESETS["Moderate"]
    temperatures = np.linspace(0.1, 3, 80)
    entropies = []
    for temperature in temperatures:
        probabilities = softmax(logits, float(temperature))
        assert probabilities.sum() == pytest.approx(1)
        assert np.all(probabilities >= 0)
        assert list(np.argsort(-probabilities)) == list(np.argsort(-logits))
        entropies.append(entropy_bits(probabilities))
    assert np.all(np.diff(entropies) > 0)
    np.testing.assert_allclose(softmax([1000, 1001]), [1 / (1 + np.e), np.e / (1 + np.e)])
    np.testing.assert_allclose(softmax(logits), decoding_distribution(logits, top_k=10, top_p=1))


@pytest.mark.parametrize("values", [[0, 0], [-0.1, 1.1], [np.nan, 1], [[1]], []])
def test_invalid_entropy_inputs(values):
    with pytest.raises(ValueError):
        entropy_bits(values)


def test_stages_show_correct_raw_and_normalized_mass():
    state = stages(np.log([0.5, 0.3, 0.2]), top_k=2, top_p=0.6)
    np.testing.assert_allclose(state["raw_k"], [0.5, 0.3, 0])
    np.testing.assert_allclose(state["after_k"], [0.625, 0.375, 0])
    np.testing.assert_allclose(state["raw_p"], [0.625, 0, 0])
    np.testing.assert_array_equal(state["final"], [1, 0, 0])
    assert state["retained_original_mass"] == pytest.approx(0.5)
    assert state["effective"] == pytest.approx(1)
    assert stages([1, 2], temperature=0)["scaled_logits"] is None


def test_landscape_matches_individual_reference_at_every_grid_point():
    temperatures, thresholds = [0.1, 1, 3], [0.05, 0.9, 1]
    grid = landscape(PRESETS["Moderate"], temperatures, thresholds, 3)
    for row, p in enumerate(thresholds):
        for column, t in enumerate(temperatures):
            probabilities = decoding_distribution(PRESETS["Moderate"], temperature=t, top_k=3, top_p=p)
            assert grid["entropy"][row, column] == pytest.approx(entropy_bits(probabilities))
            assert grid["candidates"][row, column] == np.count_nonzero(probabilities)
    assert np.all(grid["candidates"] >= 1)
    assert np.all(grid["candidates"] <= 3)


def test_greedy_counterexample_exact_and_beam_probabilities():
    exact = exhaustive_search(depth=2)
    assert exact[0]["ids"] == (1, 0)
    assert np.exp(exact[0]["log_probability"]) == pytest.approx(0.38)
    greedy = sample_path(depth=2, temperature=0)
    assert greedy["ids"] == (0, 0)
    assert np.exp(greedy["log_probability"]) == pytest.approx(0.33)
    assert beam_search(depth=2, width=1)["best"]["ids"] == greedy["ids"]
    assert beam_search(depth=2, width=2)["best"]["ids"] == exact[0]["ids"]
    assert sum(np.exp(row["log_probability"]) for row in exhaustive_search(3)) == pytest.approx(1)
    assert beam_search(3, 8)["best"] == exhaustive_search(3)[0]


def test_model_probabilities_and_policy_probabilities_are_distinct():
    result = sample_path(seed=275, temperature=0)
    assert result["policy_log_probability"] == 0
    assert result["log_probability"] < 0
    assert sample_path(seed=275)["ids"] == sample_path(seed=275)["ids"]
    assert not np.allclose(binary_logits((0,)), binary_logits((1,)))


@pytest.mark.parametrize("function,options", [
    (exhaustive_search, {"depth": 0}), (exhaustive_search, {"depth": 9}),
    (beam_search, {"width": 0}), (beam_search, {"width": True}),
    (sample_path, {"depth": 9}), (sample_path, {"temperature": -1}),
    (LabConfig, {"samples": 0}), (LabConfig, {"seed": -1}),
    (LabConfig, {"beam_width": 17}), (kv_budget, {"layers": 0}),
])
def test_invalid_visual_parameters(function, options):
    with pytest.raises(ValueError):
        function(**options)


def test_shared_rng_and_reset_rng_are_different_protocols():
    report = compute_group("reproducibility", LabConfig())
    assert report["Greedy"]["unique"] == 1
    assert report["Reset same seed"]["exact_agreement"] == 1
    assert report["Shared RNG"]["unique"] > 1
    assert report["Different seeds"]["unique"] > 1
    rng = np.random.default_rng(275)
    fixture = lambda context: [0.0, 0.0, -10.0]
    first = generate(fixture, (), ("A", "B", "EOS"), eos_id=2, rng=rng, max_new_tokens=10)
    replay = generate(fixture, (), ("A", "B", "EOS"), eos_id=2, seed=275, max_new_tokens=10)
    assert first == replay
    with pytest.raises(ValueError):
        generate(fixture, (), ("A", "B", "EOS"), eos_id=2, rng="invalid")


def test_pairwise_agreement_excludes_double_padding_and_includes_eos():
    report = sequence_agreement([(0, 2), (0, 1, 2), (0, 2)])
    assert report["exact_agreement"] == pytest.approx(1 / 3)
    assert report["token_agreement"] == pytest.approx((1 + 1 / 3 + 1 / 3) / 3)
    assert report["unique"] == 2
    assert sequence_agreement([(), ()])["token_agreement"] == 1
    with pytest.raises(ValueError):
        sequence_agreement([(0,)])


def test_stopping_traces_and_json_completeness():
    report = compute_group("stopping", LabConfig())
    eos = report["runs"]["Natural EOS"]
    stop = report["runs"]["Cross-token stop"]
    budget = report["runs"]["Output budget"]
    assert eos["result"]["finish_reason"] == "eos"
    assert stop["result"]["finish_reason"] == "stop_sequence"
    assert len(stop["result"]["sampled_ids"]) == 5
    assert stop["trace"][-2]["text"].endswith("<EN")
    assert stop["result"]["text"] == eos["result"]["text"]
    assert json.loads(eos["result"]["text"]) == {"ok": True}
    assert json.loads(report["complete_budget"]["result"]["text"]) == {"ok": True}
    with pytest.raises(json.JSONDecodeError):
        json.loads(budget["result"]["text"])


def test_sampling_tv_is_computed_from_actual_counts():
    report = compute_group("sampling", LabConfig(samples=10_000, quick=True))
    for row in report:
        snapshot = row["snapshots"][-1]
        assert snapshot["samples"] == 10_000
        frequencies = np.asarray(snapshot["empirical"])
        np.testing.assert_allclose(frequencies * 10_000, np.round(frequencies * 10_000))
        assert snapshot["total_variation"] == pytest.approx(0.5 * np.abs(frequencies - row["probabilities"]).sum())
        assert snapshot["total_variation"] < 0.03


def test_cache_counts_and_actual_fixed_layer_equivalence():
    report = kv_budget(prompt=4, output=3, layers=2, kv_heads=2, head_dim=8, bytes_per_value=4)
    np.testing.assert_array_equal(report["lengths"], [4, 5, 6])
    np.testing.assert_array_equal(report["new_positions"], [4, 1, 1])
    np.testing.assert_array_equal(report["reused_positions"], [0, 4, 5])
    np.testing.assert_array_equal(report["cached_kv_bytes"], 2 * 2 * 2 * 8 * 4 * np.array([4, 5, 6]))
    np.testing.assert_array_equal(report["cached_attention_pairs"], [10, 5, 6])
    assert kv_equivalence()["max_absolute_error"] < 1e-12


@pytest.mark.parametrize("group", GROUPS)
def test_measurements_serialize_without_nan(group):
    encoded = json.dumps(serializable(compute_group(group, LabConfig(quick=True))), allow_nan=False)
    assert len(encoded) > 50
