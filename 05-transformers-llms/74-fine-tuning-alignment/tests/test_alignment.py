"""DPO derivatives, token scoring, LoRA invariants and deterministic CPU stages."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example import (Config, LoRALinear, dpo_loss, make_data, pair_logps,
                     response_logps, run_experiment)
from from_scratch import dpo_loss_gradient


def test_numpy_gradient_matches_finite_differences_and_autograd():
    policy = np.array([[-2.0, -4.0], [-3.0, -2.0], [-1.0, -5.0]])
    reference = np.array([[-3.0, -4.0], [-2.0, -3.0], [-2.0, -4.0]])
    loss, gradient = dpo_loss_gradient(policy, reference, 0.7)
    finite_difference = np.empty_like(policy)
    for index in np.ndindex(policy.shape):
        plus, minus = policy.copy(), policy.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        finite_difference[index] = (dpo_loss_gradient(plus, reference, 0.7)[0]
                                    - dpo_loss_gradient(minus, reference, 0.7)[0]) / 2e-6
    np.testing.assert_allclose(gradient, finite_difference, atol=1e-9, rtol=1e-7)
    tensor = torch.tensor(policy, requires_grad=True)
    frozen = torch.tensor(reference, requires_grad=True)
    torch_loss = dpo_loss(tensor, frozen, 0.7)
    torch_loss.backward()
    assert float(torch_loss.detach()) == pytest.approx(loss)
    np.testing.assert_allclose(tensor.grad.numpy(), gradient, atol=1e-12)
    assert frozen.grad is None


def test_category_dpo_gradient_cancels_softmax_normalizer():
    logits = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64, requires_grad=True)
    pairs = pair_logps(logits, torch.tensor([0]), torch.tensor([2]))
    reference = torch.tensor([[-2., -2.]], dtype=torch.float64)
    loss = dpo_loss(pairs, reference, 0.5)
    loss.backward()
    _, gradient = dpo_loss_gradient(pairs.detach().numpy(), reference.numpy(), 0.5)
    expected = np.zeros((1, 4))
    expected[0, [0, 2]] = gradient[0]
    np.testing.assert_allclose(logits.grad.numpy(), expected, atol=1e-14)


def test_matching_margins_give_log_two_and_batch_duplication_keeps_mean():
    policy = np.array([[-2., -4.], [-3., -2.]])
    reference = policy - 3
    loss, gradient = dpo_loss_gradient(policy, reference)
    assert loss == pytest.approx(np.log(2))
    loss2, gradient2 = dpo_loss_gradient(np.tile(policy, (2, 1)), np.tile(reference, (2, 1)))
    assert loss2 == loss
    np.testing.assert_allclose(gradient2[:2], gradient / 2)


def test_extreme_margins_stay_finite_and_have_correct_gradient_signs():
    policy = np.array([[0., -10000.], [-10000., 0.]])
    loss, gradient = dpo_loss_gradient(policy, np.full((2, 2), -1.))
    assert np.isfinite(loss) and np.isfinite(gradient).all()
    assert loss == pytest.approx(2500.)
    assert (gradient[:, 0] <= 0).all() and (gradient[:, 1] >= 0).all()
    assert float(dpo_loss(torch.tensor(policy), torch.full((2, 2), -1.))) == loss


def test_relative_preference_can_improve_while_both_probabilities_drop():
    reference = np.log([[0.4, 0.3]])
    policy = np.log([[0.3, 0.1]])
    assert (np.exp(policy) < np.exp(reference)).all()
    assert dpo_loss_gradient(policy, reference)[0] < np.log(2)


@pytest.mark.parametrize("policy,reference,beta", [
    ([], [], 0.5), ([[1., -1.]], [[-1., -1.]], 0.5),
    ([[-1., -2.]], [[-1.]], 0.5), ([[-1., float("nan")]], [[-1., -1.]], 0.5),
    ([[-1., -2.]], [[-1., -1.]], 0), ([[-1., -2.]], [[-1., -1.]], True),
    ([[-1., -2.]], [[-1., -1.]], float("inf")),
])
def test_dpo_rejects_invalid_values_in_both_implementations(policy, reference, beta):
    with pytest.raises(ValueError):
        dpo_loss_gradient(policy, reference, beta)
    with pytest.raises(ValueError):
        dpo_loss(torch.tensor(policy), torch.tensor(reference), beta)


def test_response_scoring_sums_shifted_tokens_and_masks_prompt_and_padding():
    logits = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4).requires_grad_()
    targets = torch.tensor([[-100, 2, 1], [3, -100, -100]])
    mask = targets != -100
    scores = response_logps(logits, targets, mask)
    losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    torch.testing.assert_close(scores, -losses.sum(-1))
    torch.testing.assert_close(-scores.sum() / mask.sum(),
                               F.cross_entropy(logits.transpose(1, 2), targets))
    (-scores.mean()).backward()
    assert torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
    changed = logits.detach().clone()
    changed[~mask] = -900
    torch.testing.assert_close(response_logps(changed, targets, mask), scores.detach())


@pytest.mark.parametrize("targets,mask", [
    (torch.tensor([[0, 1]]), torch.tensor([[False, False]])),
    (torch.tensor([[0, 4]]), torch.tensor([[True, True]])),
    (torch.tensor([[0., 1.]]), torch.tensor([[True, True]])),
    (torch.tensor([[0, 1]]), torch.tensor([[1, 1]])),
])
def test_response_scoring_rejects_empty_or_invalid_targets(targets, mask):
    with pytest.raises(ValueError):
        response_logps(torch.zeros(1, 2, 4), targets, mask)


def test_lora_initialization_first_gradients_frozen_base_and_merge():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(74)
        base = nn.Linear(8, 4).double()
        layer = LoRALinear(base, rank=1, alpha=2)
        x = torch.randn(5, 8, dtype=torch.float64)
        snapshot = deepcopy(layer.base.state_dict())
        torch.testing.assert_close(layer(x), base(x), atol=0, rtol=0)
        assert sum(p.numel() for p in layer.parameters() if p.requires_grad) == 12
        layer(x).square().mean().backward()
        assert torch.count_nonzero(layer.A.grad) == 0
        assert torch.linalg.vector_norm(layer.B.grad) > 0
        assert all(p.grad is None and not p.requires_grad for p in layer.base.parameters())
        with torch.no_grad():
            layer.B.add_(-0.1 * layer.B.grad)
        layer.zero_grad(set_to_none=True)
        layer(x).square().mean().backward()
        assert torch.linalg.vector_norm(layer.A.grad) > 0
        torch.testing.assert_close(layer(x), layer.merged()(x), atol=1e-12, rtol=1e-12)
        assert all(torch.equal(v, snapshot[k]) for k, v in layer.base.state_dict().items())
        assert torch.linalg.matrix_rank(layer.B @ layer.A) <= 1


@pytest.mark.parametrize("rank,alpha", [(0, 2), (5, 2), (True, 2), (1, 0), (1, float("nan"))])
def test_lora_rejects_invalid_rank_and_scale(rank, alpha):
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(8, 4), rank, alpha)


def test_synthetic_splits_are_repeatable_and_features_are_disjoint():
    splits = [make_data(32, seed) for seed in range(43, 47)]
    for left, right in zip(splits[0], make_data(32, 43)):
        assert torch.equal(left, right)
    for i, first in enumerate(splits):
        assert first[0][:, :2].sum().item() == 32
        assert (first[2] // 2 == first[0][:, :2].argmax(-1)).all()
        for second in splits[i + 1:]:
            assert not set(map(tuple, first[0].tolist())) & set(map(tuple, second[0].tolist()))


@pytest.mark.parametrize("kwargs", [{"rank": 5}, {"seed": -1}, {"base_steps": 0},
                                    {"beta": 0}, {"sft_lr": float("nan")}])
def test_config_rejects_invalid_training_parameters(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_cpu_pipeline_is_repeatable_keeps_reference_and_base_and_restores_rng():
    state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    config = Config(base_steps=20, sft_steps=10, dpo_steps=10)
    first, second = run_experiment(config), run_experiment(config)
    assert first == second
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    assert first["invariants"]["base_unchanged"]
    assert first["invariants"]["reference_unchanged"]
    assert first["invariants"]["merged_max_abs_error"] < 2e-6
    assert first["stages"]["sft_lora"]["dpo_loss_vs_sft_reference"] == pytest.approx(np.log(2))
    assert first["stages"]["sft_lora"]["kl_policy_to_sft_reference"] == 0
    for metrics in first["stages"].values():
        assert np.isfinite(list(metrics.values())).all()
