"""Objective alignment, information boundaries, explicit gradients and CPU training."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example import (Config, TinyLanguageModel, clm_batch, evaluate, fixed_mlm_batch,
                     make_corpus, mlm_batch, run_experiment, token_loss)
from from_scratch import bigram_loss_gradient


def test_split_is_disjoint_and_train_contains_every_content_token():
    train, val, vocab = make_corpus()
    assert train.shape == (12, 6) and val.shape == (4, 6)
    assert not set(map(tuple, train.tolist())) & set(map(tuple, val.tolist()))
    assert set(val.flatten().tolist()) <= set(train.flatten().tolist())
    assert vocab.index("<mask>") not in train


def test_clm_shift_keeps_eos_and_does_not_mutate_source():
    tokens = torch.tensor([[0, 4, 5, 1], [0, 6, 7, 1]])
    inputs, targets = clm_batch(tokens)
    assert inputs.tolist() == [[0, 4, 5], [0, 6, 7]]
    assert targets.tolist() == [[4, 5, 1], [6, 7, 1]]
    inputs[0, 0] = 9
    assert tokens[0, 0] == 0


def test_shortest_clm_pair_can_be_scored_with_a_single_input_token():
    batch = clm_batch(torch.tensor([[0, 1]]))
    model = TinyLanguageModel(3)
    logits = model(batch[0], True)
    assert logits.shape == (1, 1, 3)
    assert torch.isfinite(token_loss(logits, batch[1]))


def test_real_block_blocks_future_and_bidirectional_control_responds():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        model = TinyLanguageModel(13).eval()
        tokens = torch.tensor([[0, 4, 5, 6, 1]])
        changed = tokens.clone()
        changed[:, -1] = 8
        with torch.inference_mode():
            before = model(tokens, True)
            after = model(changed, True)
            torch.testing.assert_close(before[:, :-1], after[:, :-1], atol=1e-6, rtol=0)
            torch.testing.assert_close(before[:, :3], model(tokens[:, :3], True), atol=1e-6, rtol=0)
            assert torch.linalg.vector_norm(before[:, -1] - after[:, -1]) > 1e-4
            assert torch.linalg.vector_norm(model(tokens, False)[:, 0] - model(changed, False)[:, 0]) > 1e-4


@pytest.mark.parametrize("masked,count", [(1, 4), (2, 6), (4, 1)])
def test_fixed_masks_cover_all_combinations_without_bos_eos(masked, count):
    train, _, vocab = make_corpus()
    original = train.clone()
    mask_id = vocab.index("<mask>")
    inputs, targets = fixed_mlm_batch(train, mask_id, masked)
    selected = targets != -100
    assert inputs.shape == (12 * count, 6)
    assert (selected.sum(1) == masked).all()
    assert not selected[:, [0, -1]].any()
    assert (inputs[selected] == mask_id).all()
    expanded = train.repeat_interleave(count, dim=0)
    torch.testing.assert_close(targets[selected], expanded[selected])
    torch.testing.assert_close(inputs[~selected], expanded[~selected])
    torch.testing.assert_close(train, original)


def test_unselected_mlm_logits_affect_neither_loss_nor_gradient():
    logits = torch.arange(24, dtype=torch.float64).reshape(1, 6, 4).requires_grad_()
    targets = torch.tensor([[-100, 2, -100, 1, -100, -100]])
    loss = token_loss(logits, targets)
    expected = F.cross_entropy(logits[0, [1, 3]], torch.tensor([2, 1]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert (logits.grad[targets == -100] == 0).all()
    altered = logits.detach().clone()
    altered[targets == -100] = torch.tensor([1000., -1000., 0., 0.], dtype=torch.float64)
    torch.testing.assert_close(token_loss(altered, targets), loss)


def test_evaluation_restores_mode_and_changes_neither_weights_nor_gradients():
    train, _, vocab = make_corpus()
    model = TinyLanguageModel(len(vocab)).train()
    batch = clm_batch(train)
    token_loss(model(batch[0], True), batch[1]).backward()
    weights = deepcopy(model.state_dict())
    grads = [p.grad.clone() for p in model.parameters()]
    a, b = evaluate(model, batch, True), evaluate(model, batch, True)
    assert a == b and model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
    for parameter, grad in zip(model.parameters(), grads):
        torch.testing.assert_close(parameter.grad, grad, rtol=0, atol=0)


def test_bigram_repeated_row_gradient_matches_autograd_and_finite_difference():
    weights = np.random.default_rng(73).normal(size=(4, 4))
    current, targets = np.array([0, 0, 1, 0]), np.array([1, 2, 3, 2])
    loss, gradient = bigram_loss_gradient(weights, current, targets)
    tensor = torch.tensor(weights, requires_grad=True)
    expected = F.cross_entropy(tensor[current], torch.tensor(targets))
    expected.backward()
    assert loss == pytest.approx(expected.item(), abs=1e-12)
    np.testing.assert_allclose(gradient, tensor.grad.numpy(), atol=1e-12)
    epsilon = 1e-6
    for index in np.ndindex(weights.shape):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (bigram_loss_gradient(plus, current, targets)[0]
                     - bigram_loss_gradient(minus, current, targets)[0]) / (2 * epsilon)
        assert gradient[index] == pytest.approx(numerical, abs=1e-9)
    np.testing.assert_allclose(gradient.sum(1), 0, atol=1e-16)
    np.testing.assert_array_equal(gradient[2:], 0)
    probabilities = torch.softmax(tensor.detach(), dim=-1).numpy()
    np.testing.assert_allclose(probabilities.sum(1), 1)


def test_extreme_logits_use_logsumexp_instead_of_log_of_underflowed_probability():
    loss, gradient = bigram_loss_gradient([[1000., -1000.], [0., 0.]], [0], [1])
    assert loss == 2000 and np.isfinite(gradient).all()


@pytest.mark.parametrize("objective", ["clm", "mlm"])
def test_cpu_training_is_finite_reproducible_and_budget_is_explicit(objective):
    config = Config(objective=objective, steps=25)
    report, rerun = run_experiment(config), run_experiment(config)
    for key in ("initial_train", "initial_validation", "final_train", "final_validation"):
        assert report[key] == rerun[key]
        assert np.isfinite(report[key]["nll"])
    assert report["final_train"]["nll"] < report["initial_train"]["nll"]
    assert report["budget"]["processed_input_tokens"] == 25 * 12 * (5 if objective == "clm" else 6)
    assert report["budget"]["supervised_tokens"] == 25 * 12 * (5 if objective == "clm" else 1)
    assert ("perplexity" in report["final_validation"]) == (objective == "clm")
    assert report["budget"]["unique_train_sequences"] == 12


@pytest.mark.parametrize("kwargs", [
    {"objective": "other"}, {"steps": 0}, {"steps": True}, {"width": 3},
    {"heads": 0}, {"seed": -1}, {"seed": True}, {"seed": 2**32},
    {"learning_rate": float("nan")}, {"learning_rate": -1}, {"learning_rate": True},
    {"masked_tokens": 0}, {"masked_tokens": 5},
])
def test_invalid_config_is_explicit(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


@pytest.mark.parametrize("tokens", [torch.empty(0, 6, dtype=torch.long),
                                         torch.tensor([[1]]), torch.tensor([[1., 2.]]),
                                         torch.tensor([[-1, 2]])])
def test_invalid_sequences_are_explicit(tokens):
    with pytest.raises(ValueError):
        clm_batch(tokens)


@pytest.mark.parametrize("positions", [[], [0], [5]])
def test_invalid_mlm_selection_is_explicit(positions):
    train, _, vocab = make_corpus()
    selected = torch.zeros_like(train, dtype=torch.bool)
    selected[:, positions] = True
    with pytest.raises(ValueError):
        mlm_batch(train, selected, vocab.index("<mask>"))


@pytest.mark.parametrize("weights,current,targets", [
    ([[0., 0.]], [0], [1]), ([[np.nan, 0], [0, 0]], [0], [1]),
    (np.eye(2), [], []), (np.eye(2), [0.5], [1]),
    (np.eye(2), [0], [2]), (np.eye(2), [0], [-1]), (np.eye(2), [0, 1], [1]),
])
def test_invalid_bigram_inputs_are_explicit(weights, current, targets):
    with pytest.raises(ValueError):
        bigram_loss_gradient(weights, current, targets)
