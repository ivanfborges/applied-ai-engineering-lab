"""Architectural invariants and small CPU integration checks; no convergence threshold."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import Attention, EncoderDecoderBlock, layer_norm, sinusoidal_positions
from example import (Config, Seq2SeqTransformer, causal_mask, make_split,
                     reversal_targets, run, sequence_metrics)


@pytest.fixture
def tiny_config():
    return Config(vocab_size=11, sequence_length=3, train_size=12, test_size=4,
                  d_model=8, heads=2, layers=1, d_ff=16, dropout=0, steps=2, batch_size=4)


def test_positions_even_and_odd_widths():
    for width in (1, 5, 8):
        actual = sinusoidal_positions(3, width)
        np.testing.assert_array_equal(actual[0, 0::2], 0)
        np.testing.assert_array_equal(actual[0, 1::2], 1)
        np.testing.assert_allclose(actual[1, 0], np.sin(1))
        if width > 1:
            np.testing.assert_allclose(actual[1, 1], np.cos(1))
        assert actual.shape == (3, width)


def test_layer_norm_feature_axis_and_constant_rows():
    x = np.array([[1., 2., 5.], [7., 7., 7.]])
    actual = layer_norm(x)
    np.testing.assert_allclose(actual[0], (x[0] - x[0].mean()) / np.sqrt(x[0].var() + 1e-5))
    np.testing.assert_allclose(actual.mean(axis=1), 0, atol=1e-15)
    np.testing.assert_array_equal(actual[1], 0)


def test_attention_hand_result_rectangular_and_blocked_keys():
    attention = Attention(2, np.random.default_rng(0))
    attention.wq = attention.wk = attention.wv = attention.wo = np.eye(2)
    q, memory = np.zeros((2, 2)), np.array([[2., 0.], [0., 4.], [8., 8.]])
    mask = np.array([[False, False, True], [True, False, True]])
    output, weights = attention(q, memory, mask)
    np.testing.assert_allclose(weights, [[.5, .5, 0], [0, 1, 0]])
    np.testing.assert_allclose(output, [[1, 2], [0, 4]])


def test_numpy_fixed_parameters_shapes_and_causal_invariance():
    rng = np.random.default_rng(5)
    source, target = rng.normal(size=(5, 8)), rng.normal(size=(4, 8))
    block = EncoderDecoderBlock()
    memory, decoded, weights = block(source, target)
    _, repeated, _ = block(source, target)
    np.testing.assert_array_equal(decoded, repeated)
    assert memory.shape == (5, 8) and decoded.shape == (4, 8)
    assert weights["cross"].shape == (4, 5)
    for attention in weights.values():
        np.testing.assert_allclose(attention.sum(-1), 1, atol=1e-14)
    np.testing.assert_array_equal(weights["decoder"][np.triu_indices(4, 1)], 0)
    changed = target.copy()
    changed[2:] += rng.normal(size=(2, 8)) * 3
    _, other, _ = block(source, changed)
    np.testing.assert_allclose(decoded[:2], other[:2], atol=1e-13)
    _, unmasked, _ = block(source, target, causal=False)
    _, unmasked_changed, _ = block(source, changed, causal=False)
    assert np.max(abs(unmasked[:2] - unmasked_changed[:2])) > 1e-5
    changed_source = source.copy()
    changed_source[0] += rng.normal(size=8)
    _, different_memory, _ = block(changed_source, target)
    assert np.max(abs(decoded - different_memory)) > 1e-5


@pytest.mark.parametrize("query,memory,mask", [
    (np.ones((1, 2)), np.ones((3, 2)), np.ones((1, 3), dtype=bool)),
    (np.ones((1, 2)), np.ones((3, 2)), np.zeros((1, 3))),
    (np.ones((1, 2)), np.ones((3, 2)), np.zeros((3, 1), dtype=bool)),
    (np.ones((1, 3)), np.ones((3, 2)), None),
    (np.ones((0, 2)), np.ones((3, 2)), None),
    (np.full((1, 2), np.nan), np.ones((3, 2)), None),
])
def test_numpy_attention_rejects_invalid_inputs(query, memory, mask):
    with pytest.raises(ValueError):
        Attention(2, np.random.default_rng(0))(query, memory, mask)


@pytest.mark.parametrize("kwargs", [{"length": 0, "width": 8}, {"length": 2, "width": True}])
def test_invalid_positions(kwargs):
    with pytest.raises(ValueError):
        sinusoidal_positions(**kwargs)


def test_numpy_other_invalid_inputs():
    with pytest.raises(ValueError):
        layer_norm(np.ones((2, 3)), eps=0)
    with pytest.raises(ValueError):
        EncoderDecoderBlock()(np.ones((2, 7)), np.ones((3, 8)))


def test_unique_disjoint_deterministic_split(tiny_config):
    train, test = make_split(tiny_config)
    train_again, test_again = make_split(tiny_config)
    assert torch.equal(train, train_again) and torch.equal(test, test_again)
    rows = [tuple(row) for row in torch.cat((train, test)).tolist()]
    assert len(set(rows)) == len(rows)
    assert min(min(row) for row in rows) >= 2
    assert max(max(row) for row in rows) < tiny_config.vocab_size


def test_shifted_targets_and_mask_orientation():
    decoder, target = reversal_targets(torch.tensor([[7, 4, 9]]))
    assert decoder.tolist() == [[0, 9, 4, 7]]
    assert target.tolist() == [[9, 4, 7, 1]]
    assert causal_mask(3).tolist() == [[False, True, True], [False, False, True], [False, False, False]]


def test_torch_causal_logits_source_sensitivity_and_unmasked_control(tiny_config):
    torch.manual_seed(11)
    model = Seq2SeqTransformer(tiny_config).eval()
    source = torch.tensor([[2, 4, 6]])
    decoder = torch.tensor([[0, 6, 4, 2]])
    changed = decoder.clone()
    changed[:, 2:] = torch.tensor([[7, 8]])
    with torch.no_grad():
        logits = model(source, decoder)
        other = model(source, changed)
        assert logits.shape == (1, 4, tiny_config.vocab_size)
        torch.testing.assert_close(logits[:, :2], other[:, :2], atol=1e-6, rtol=1e-6)
        unmasked = model(source, decoder, causal=False)
        unmasked_changed = model(source, changed, causal=False)
        assert (unmasked[:, :2] - unmasked_changed[:, :2]).abs().max() > 1e-5
        changed_source = model(torch.tensor([[8, 4, 6]]), decoder)
        assert (changed_source - logits).abs().max() > 1e-5


def test_registered_parameters_receive_finite_gradients(tiny_config):
    torch.manual_seed(13)
    model = Seq2SeqTransformer(tiny_config)
    train, _ = make_split(tiny_config)
    decoder, target = reversal_targets(train[:4])
    loss = torch.nn.functional.cross_entropy(model(train[:4], decoder).flatten(0, 1), target.flatten())
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.embedding.weight.grad.abs().sum() > 0
    assert model.transformer.decoder.layers[0].multihead_attn.in_proj_weight.grad.abs().sum() > 0


def test_generation_matches_slow_prefix_reference_and_uses_no_targets(tiny_config):
    torch.manual_seed(7)
    model = Seq2SeqTransformer(tiny_config).eval()
    source = torch.tensor([[2, 4, 6], [3, 5, 7]])
    generated = model.generate(source)
    prefix = torch.zeros((2, 1), dtype=torch.long)
    with torch.no_grad():
        for _ in range(4):
            selected = model(source, prefix)[:, -1].argmax(-1, keepdim=True)
            prefix = torch.cat((prefix, selected), dim=1)
    assert torch.equal(generated, prefix[:, 1:])
    assert generated.shape == (2, 4)
    assert not generated.requires_grad


def test_metrics_include_eos_and_penalize_one_wrong_token():
    target = torch.tensor([[3, 2, 1], [4, 2, 1]])
    prediction = torch.tensor([[3, 2, 1], [4, 2, 0]])
    actual = sequence_metrics(prediction, target)
    assert actual["token_accuracy"] == pytest.approx(5 / 6)
    assert actual["exact_match"] == .5
    with pytest.raises(ValueError):
        sequence_metrics(prediction[:, :1], target)


def test_small_training_run_repeats_without_convergence_claim(tiny_config):
    first, second = run(tiny_config), run(tiny_config)
    assert first == second
    assert np.isfinite(first["train_eval_cross_entropy_after"])
    assert 0 <= first["test_autoregressive"]["token_accuracy"] <= 1
    assert first["interpretation_status"] == "author review required"


@pytest.mark.parametrize("kwargs", [
    {"steps": 0}, {"heads": 3}, {"dropout": 1}, {"learning_rate": float("nan")},
    {"vocab_size": 2}, {"sequence_length": 0}, {"seed": -1}, {"seed": True},
    {"batch_size": True}, {"vocab_size": 3, "sequence_length": 1},
])
def test_invalid_config(tiny_config, kwargs):
    with pytest.raises(ValueError):
        replace(tiny_config, **kwargs)


def test_invalid_token_and_generation_inputs(tiny_config):
    model = Seq2SeqTransformer(tiny_config)
    source = torch.tensor([[2, 4, 6]])
    with pytest.raises(ValueError, match="eval"):
        model.generate(source)
    model.eval()
    for bad in (source.float(), torch.tensor([[11, 2, 3]]), torch.empty((0, 3), dtype=torch.long),
                torch.ones((1, 7), dtype=torch.long)):
        with pytest.raises(ValueError):
            model.embed(bad)
    with pytest.raises(ValueError):
        reversal_targets(torch.tensor([[0, 3]]))
    with pytest.raises(ValueError):
        model.generate(source[:, :2])
    with pytest.raises(ValueError):
        model(source, torch.tensor([[0], [0]]))
    with pytest.raises(ValueError):
        causal_mask(0)
