"""Real CPU instrumentation, checkpoint and geometry contracts for the visual lab."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example import Config, Seq2SeqTransformer, make_split, reversal_targets
from transformer_lab import (capture_attention, capture_model, default_trace, equivariance_demo,
    generate_prefix, masked_softmax, memory_bytes, model_at, playground_cases, shared_pca, train_comparison)


@pytest.fixture
def config():
    return Config(seed=67, vocab_size=12, sequence_length=3, d_model=8, heads=2,
                  layers=2, d_ff=16, steps=2, train_size=12, test_size=4, batch_size=4)


@pytest.mark.parametrize("pre_norm", [False, True])
def test_instrumented_forward_matches_original_and_captures_individual_heads(config, pre_norm):
    torch.manual_seed(4)
    model = Seq2SeqTransformer(config).eval()
    for layer in list(model.transformer.encoder.layers) + list(model.transformer.decoder.layers):
        layer.norm_first = pre_norm
    train, _ = make_split(config)
    target, _ = reversal_targets(train[:1])
    trace = capture_model(model, train[:1], target)
    assert trace["instrumentation_max_error"] < 1e-5
    assert len(trace["attention"]) == 6
    for attention in trace["attention"]:
        probability = attention["probability"]
        assert probability.shape[0] == config.heads
        np.testing.assert_allclose(probability.sum(-1), 1, atol=2e-7)
        if attention["mechanism"] == "decoder":
            np.testing.assert_array_equal(probability[:, np.triu_indices(4, 1)[0], np.triu_indices(4, 1)[1]], 0)
        if attention["mechanism"] == "cross":
            assert probability.shape == (2, 4, 3)
    assert not np.allclose(trace["attention"][0]["probability"][0], trace["attention"][0]["probability"][1])


def test_causal_invariance_and_unmasked_sensitivity_in_instrumented_model(config):
    model, _ = default_trace(config)
    train, _ = make_split(config)
    decoder, _ = reversal_targets(train[:1])
    changed = decoder.clone()
    changed[:, 2:] = torch.tensor([[8, 9]])
    first = capture_model(model, train[:1], decoder)
    other = capture_model(model, train[:1], changed)
    np.testing.assert_allclose(first["logits"][:2], other["logits"][:2], atol=1e-6)
    unmasked = capture_model(model, train[:1], decoder, causal=False)
    other = capture_model(model, train[:1], changed, causal=False)
    assert np.max(abs(unmasked["logits"][:2] - other["logits"][:2])) > 1e-5


def test_shared_pca_uses_one_basis_for_all_stages(config):
    _, trace = default_trace(config)
    projection = shared_pca(trace["stages"])
    for i, stage in enumerate(trace["stages"]):
        expected = (stage["values"] - projection["mean"]) @ projection["components"].T
        np.testing.assert_allclose(projection["coordinates"][i], expected, atol=1e-12)
    assert 0 < sum(projection["variance_ratio"]) <= 1 + 1e-12


def test_equivariance_control_and_position_change():
    result = equivariance_demo()
    assert result["without_positions_error"] < 1e-12
    assert result["with_positions_change"] > 1e-5


def test_masked_softmax_all_blocked_handling_and_rectangular_shape():
    values = masked_softmax(np.zeros((4, 3, 5)), np.array([[False] * 5, [False, True, True, True, True], [False] * 5]))
    np.testing.assert_allclose(values.sum(-1), 1)
    np.testing.assert_array_equal(values[:, 1, 1:], 0)
    for bad in (np.ones((3, 5), dtype=bool), np.zeros((5, 3), dtype=bool), np.zeros((3, 5))):
        with pytest.raises(ValueError):
            masked_softmax(np.zeros((4, 3, 5)), bad)


def test_theoretical_bytes_no_matrix_allocation():
    result = memory_bytes(32768, 16384, heads=8, batch=4, bytes_per_value=2)
    assert result["cross"] == 4 * 8 * 32768 * 16384 * 2
    assert memory_bytes(256, 128)["encoder"] == 4 * memory_bytes(128, 128)["encoder"]
    with pytest.raises(ValueError):
        memory_bytes(0, 3)


def test_matched_initialization_disjoint_split_cache_and_generation(config, tmp_path):
    comparison, path, reused = train_comparison(config, tmp_path)
    assert not reused and path.is_file()
    assert comparison["config"]["steps"] == 2
    for key in comparison["states"]["masked"][0]:
        assert torch.equal(comparison["states"]["masked"][0][key], comparison["states"]["unmasked"][0][key])
    loaded, same_path, reused = train_comparison(config, tmp_path, skip_training=True)
    assert reused and same_path == path
    assert loaded["history"] == comparison["history"]
    missing, _, _ = train_comparison(replace(config, seed=68), tmp_path, skip_training=True)
    assert missing is None
    train, test = make_split(config)
    assert set(map(tuple, train.tolist())).isdisjoint(set(map(tuple, test.tolist())))
    model = model_at(comparison)
    generated, probability = generate_prefix(model, test)
    assert torch.equal(generated, model.generate(test))
    np.testing.assert_allclose(probability.sum(-1), 1, atol=2e-7)
    second, _, _ = train_comparison(config, tmp_path / "repeat")
    assert second["history"] == comparison["history"]


def test_playground_cases_are_computed_and_explicitly_counterfactual(config):
    cases = playground_cases(config)
    assert len(cases) == 8
    assert {case["norm"] for case in cases} == {"pre", "post"}
    assert all(case["instrumentation_max_error"] < 1e-5 for case in cases)
    assert any(case["counterfactual"] for case in cases)
    assert any(not case["counterfactual"] for case in cases)


def test_capture_requires_eval_and_fixed_mask(config):
    model = Seq2SeqTransformer(config)
    train, _ = make_split(config)
    decoder, _ = reversal_targets(train[:1])
    with pytest.raises(ValueError):
        capture_model(model, train[:1], decoder)
    model.eval()
    x = model.embed(train[:1])
    with pytest.raises(ValueError, match="fully masked"):
        capture_attention(model.transformer.encoder.layers[0].self_attn, x, x, torch.ones((3, 3), dtype=torch.bool))
