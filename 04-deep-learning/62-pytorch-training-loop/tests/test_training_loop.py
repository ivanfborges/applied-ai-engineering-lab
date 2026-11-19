"""Checks for gradient/update boundaries, aggregation and holdout isolation."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import example as study
from from_scratch import loss_and_gradients, sgd_epoch


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def toy_dataset():
    return TensorDataset(torch.tensor([[1., -1.], [0., 2.], [-3., 1.], [1., 4.], [2., 2.]]),
                         torch.tensor([[0.], [1.], [0.], [1.], [0.]]))


def test_manual_gradient_matches_finite_differences_and_autograd():
    x, y = (t.numpy().astype(float) for t in toy_dataset().tensors)
    y = y.ravel()
    w, b = np.array([0.2, -0.3]), 0.1
    loss, grad_w, grad_b = loss_and_gradients(x, y, w, b)
    eps = 1e-6
    for index in range(2):
        delta = np.eye(2)[index] * eps
        numerical = (loss_and_gradients(x, y, w + delta, b)[0]
                     - loss_and_gradients(x, y, w - delta, b)[0]) / (2 * eps)
        assert grad_w[index] == pytest.approx(numerical, abs=1e-9)
    numerical_b = (loss_and_gradients(x, y, w, b + eps)[0]
                   - loss_and_gradients(x, y, w, b - eps)[0]) / (2 * eps)
    assert grad_b == pytest.approx(numerical_b, abs=1e-9)
    tw = torch.tensor(w, requires_grad=True)
    tb = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    reference = nn.functional.binary_cross_entropy_with_logits(torch.tensor(x) @ tw + tb, torch.tensor(y))
    reference.backward()
    assert loss == pytest.approx(reference.item(), abs=1e-14)
    np.testing.assert_allclose(grad_w, tw.grad.numpy(), atol=1e-14)
    assert grad_b == pytest.approx(tb.grad.item(), abs=1e-14)


def test_stable_loss_handles_extreme_logits():
    x = np.array([[-1000.], [1000.]])
    with np.errstate(over="raise", invalid="raise"):
        loss, grad, bias_grad = loss_and_gradients(x, np.array([1., 0.]), np.ones(1), 0.)
    assert loss == pytest.approx(1000.)
    np.testing.assert_allclose(grad, [1000.])
    assert bias_grad == pytest.approx(0.)


def test_backward_accumulates_and_only_step_changes_parameters():
    model = nn.Linear(2, 1)
    x, y = toy_dataset().tensors
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    study.batch_loss(model, x, y)[1].backward()
    first_gradient = model.weight.grad.clone()
    study.batch_loss(model, x, y)[1].backward()
    torch.testing.assert_close(model.weight.grad, 2 * first_gradient)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    optimizer.step()
    assert not torch.equal(model.weight, before["weight"])
    optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters())


def test_epoch_clears_stale_gradients_and_matches_independent_sgd_steps():
    model = nn.Linear(2, 1)
    reference = copy.deepcopy(model)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 100.)
    loader = study.make_loader(toy_dataset(), 3)
    expected_total = 0.
    for x, y in loader:
        loss = study.batch_loss(reference, x, y)[1]
        gradients = torch.autograd.grad(loss, tuple(reference.parameters()))
        expected_total += loss.item() * len(x)
        with torch.no_grad():
            for parameter, gradient in zip(reference.parameters(), gradients):
                parameter.add_(gradient, alpha=-0.1)
    result = study.train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.1))
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
    assert result == pytest.approx({"online_loss": expected_total / 5, "samples": 5, "steps": 2})


def test_eval_loss_is_sample_weighted_and_keeps_singleton_tail():
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1., -2.]]))
        model.bias.fill_(0.3)
    dataset = toy_dataset()
    loader = study.make_loader(dataset, 2)
    result = study.evaluate(model, loader)
    x, y = dataset.tensors
    direct = study.batch_loss(model, x, y)[1].item()
    wrong_mean = np.mean([study.batch_loss(model, xb, yb)[1].item() for xb, yb in loader])
    assert result["loss"] == pytest.approx(direct, abs=1e-6)
    assert abs(wrong_mean - direct) > 0.01
    assert result["samples"] == 5
    assert sum(result[k] for k in ("tp", "fp", "tn", "fn")) == 5


def test_evaluation_disables_gradients_and_preserves_parameters_buffers_and_grads():
    class Probe(nn.Sequential):
        def forward(self, x):
            assert not torch.is_grad_enabled()
            assert torch.is_inference_mode_enabled()
            assert not self.training
            return super().forward(x)

    model = Probe(nn.BatchNorm1d(2), nn.Dropout(0.5), nn.Linear(2, 1))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = copy.deepcopy(model.state_dict())
    loader = study.make_loader(toy_dataset(), 2)
    first = study.evaluate(model, loader)
    second = study.evaluate(model, loader)
    assert first == second
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())


def test_data_splits_are_disjoint_reproducible_and_standardization_uses_train_only():
    partitions, preprocessing, indices = study.prepare_data()
    other, other_preprocessing, other_indices = study.prepare_data()
    all_indices = torch.cat(list(indices.values()))
    assert len(torch.unique(all_indices)) == 1800
    assert sorted(all_indices.tolist()) == list(range(1800))
    for name in indices:
        assert torch.equal(indices[name], other_indices[name])
        assert all(torch.equal(a, b) for a, b in zip(partitions[name].tensors, other[name].tensors))
    assert preprocessing == other_preprocessing
    raw = torch.randn(1800, 2, generator=torch.Generator().manual_seed(42))
    mean, scale = study.fit_standardizer(raw[indices["train"]])
    np.testing.assert_allclose(mean.numpy(), preprocessing["mean"])
    np.testing.assert_allclose(scale.numpy(), preprocessing["scale"])
    train = partitions["train"].tensors[0]
    torch.testing.assert_close(train.mean(0), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(train.std(0, unbiased=False), torch.ones(2))


def test_shuffle_has_no_drops_and_uses_independent_seed():
    data = TensorDataset(torch.arange(11))
    one = study.make_loader(data, 4, shuffle=True, seed=9)
    two = study.make_loader(data, 4, shuffle=True, seed=9)
    order = torch.cat([batch[0] for batch in one])
    torch.testing.assert_close(order, torch.cat([batch[0] for batch in two]))
    assert sorted(order.tolist()) == list(range(11))


def test_best_checkpoint_is_a_snapshot_and_restored_instead_of_last_epoch(monkeypatch):
    model = nn.Linear(2, 1)
    epochs = iter([1., 2., 3.])
    losses = iter([0.4, 0.2, 0.8])

    def fake_train(model, loader, optimizer):
        with torch.no_grad():
            model.weight.fill_(next(epochs))
        return {"online_loss": 0.5, "steps": 2, "samples": 5}

    monkeypatch.setattr(study, "train_one_epoch", fake_train)
    monkeypatch.setattr(study, "evaluate", lambda model, loader: {"loss": next(losses)})
    loader = study.make_loader(toy_dataset(), 3)
    result = study.fit(model, loader, loader, epochs=3, learning_rate=0.001)
    assert result["best_epoch"] == 2
    assert result["optimizer_steps"] == 6
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 2.))


def test_manual_unequal_microbatch_gradients_need_sample_weights():
    x, y = (t.numpy().astype(float) for t in toy_dataset().tensors)
    y = y.ravel()
    w = np.array([0.3, -0.2])
    _, full, full_bias = loss_and_gradients(x, y, w, 0.1)
    left = loss_and_gradients(x[:3], y[:3], w, 0.1)
    right = loss_and_gradients(x[3:], y[3:], w, 0.1)
    np.testing.assert_allclose((3 * left[1] + 2 * right[1]) / 5, full)
    assert (3 * left[2] + 2 * right[2]) / 5 == pytest.approx(full_bias)
    assert not np.allclose((left[1] + right[1]) / 2, full)


def test_numpy_epoch_uses_all_rows_without_mutating_initial_weights():
    x, y = (t.numpy().astype(float) for t in toy_dataset().tensors)
    initial = np.array([0.2, -0.1])
    expected = initial.copy()
    expected_bias, total = 0., 0.
    order = np.random.default_rng(8).permutation(5)
    for idx in (order[:3], order[3:]):
        loss, grad, grad_bias = loss_and_gradients(x[idx], y[idx].ravel(), expected, expected_bias)
        expected -= 0.1 * grad
        expected_bias -= 0.1 * grad_bias
        total += loss * len(idx)
    weights, bias, online = sgd_epoch(x, y.ravel(), initial, 0., learning_rate=0.1,
                                     batch_size=3, rng=np.random.default_rng(8))
    np.testing.assert_allclose(weights, expected)
    np.testing.assert_array_equal(initial, [0.2, -0.1])
    assert bias == pytest.approx(expected_bias)
    assert online == pytest.approx(total / 5)


@pytest.mark.integration
def test_small_real_training_repeats_and_only_selected_model_sees_test(monkeypatch):
    partitions, preprocessing, indices = study.prepare_data()
    monkeypatch.setattr(study, "prepare_data", lambda seed: (partitions, preprocessing, indices))
    actual_evaluate = study.evaluate
    test_calls = []

    def counted_evaluation(model, loader):
        if loader.dataset is partitions["test"]:
            test_calls.append(model)
        return actual_evaluate(model, loader)

    monkeypatch.setattr(study, "evaluate", counted_evaluation)
    result = study.run_study(epochs=3)
    assert len(test_calls) == 1
    expected = min(result["candidates"], key=lambda key: result["candidates"][key]["validation"]["loss"])
    assert result["selected_model"] == expected
    assert result["selected_test"]["samples"] == 300
    for candidate in result["candidates"].values():
        assert candidate["optimizer_steps"] == 3 * 19
        assert len(candidate["history"]) == 3
    repeat = study.run_study(epochs=3)
    assert repeat["candidates"] == result["candidates"]
    assert repeat["selected_test"] == result["selected_test"]


@pytest.mark.parametrize("x,y", [
    (torch.empty(0, 2), torch.empty(0, 1)),
    (torch.ones(3, 2), torch.ones(3)),
    (torch.ones(3, 2, dtype=torch.float64), torch.ones(3, 1)),
    (torch.ones(3, 2), torch.full((3, 1), 0.5)),
    (torch.full((3, 2), float("nan")), torch.ones(3, 1)),
])
def test_invalid_batches_raise(x, y):
    with pytest.raises(ValueError):
        study.validate_batch(x, y)


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"batch_size": 0}, {"seed": -1},
                                    {"learning_rate": float("nan")}, {"batch_size": True}])
def test_invalid_run_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        study.run_study(**kwargs)


@pytest.mark.parametrize("targets", [np.ones((5, 1)), np.full(5, 0.3), np.full(5, np.nan)])
def test_invalid_manual_targets_raise(targets):
    with pytest.raises(ValueError):
        loss_and_gradients(np.ones((5, 2)), targets, np.zeros(2), 0.)


def test_empty_loader_wrong_output_shape_and_constant_feature():
    empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, 1)))
    with pytest.raises(ValueError, match="no samples"):
        study.evaluate(nn.Linear(2, 1), empty)
    with pytest.raises(ValueError, match="logits"):
        study.evaluate(nn.Linear(2, 2), study.make_loader(toy_dataset(), 2))
    mean, scale = study.fit_standardizer(torch.ones(3, 2))
    assert torch.isfinite((torch.ones(3, 2) - mean) / scale).all()
