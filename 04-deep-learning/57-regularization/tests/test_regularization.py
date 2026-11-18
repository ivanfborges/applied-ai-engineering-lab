"""Mechanism oracles, checkpoint buffers, split boundaries and a small CPU run."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import example
from example import (CONFIGURATIONS, Configuration, EarlyStopping, MLP, batch_indices,
                     evaluate, linear_parameters, make_data, make_optimizer, run_comparison)
from from_scratch import batch_norm_training, inverted_dropout, sgd_l2_pair


def test_dropout_moments_and_eval_identity():
    samples = inverted_dropout(np.full(200_000, 2.0), 0.25, np.random.default_rng(57))
    np.testing.assert_allclose(samples.mean(), 2, atol=0.015)
    np.testing.assert_allclose(samples.var(), 4 / 3, atol=0.025)
    assert set(np.unique(samples)) == {0, 2 / 0.75}
    rng = np.random.default_rng(2)
    state = copy.deepcopy(rng.bit_generator.state)
    values = np.arange(8.0)
    np.testing.assert_array_equal(inverted_dropout(values, 0.9, rng, training=False), values)
    np.testing.assert_array_equal(inverted_dropout(values, 0, rng), values)
    assert rng.bit_generator.state == state


@pytest.mark.parametrize("p", [-0.1, 1, np.nan, np.inf])
def test_dropout_rejects_invalid_probability(p):
    with pytest.raises(ValueError):
        inverted_dropout([1], p, np.random.default_rng(1))


def test_batchnorm_matches_framework_and_variance_conventions():
    x = np.array([[1., 4., 7.], [3., 4., 1.], [5., 4., 4.], [7., 4., 0.]])
    gamma, beta = np.array([2., 3., 0.5]), np.array([1., -1., 2.])
    layer = nn.BatchNorm1d(3, eps=1e-5, momentum=0.2).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(gamma))
        layer.bias.copy_(torch.tensor(beta))
    actual = layer(torch.tensor(x)).detach().numpy()
    np.testing.assert_allclose(batch_norm_training(x, gamma, beta), actual, atol=1e-12)
    np.testing.assert_allclose(layer.running_mean.numpy(), 0.2 * x.mean(0))
    np.testing.assert_allclose(layer.running_var.numpy(), 0.8 + 0.2 * x.var(0, ddof=1))
    standardized = batch_norm_training(x, np.ones(3), np.zeros(3))
    np.testing.assert_allclose(standardized.var(0), x.var(0) / (x.var(0) + 1e-5))
    assert np.isfinite(standardized).all()  # Includes a constant feature.


@pytest.mark.parametrize("values,gamma,beta,eps", [
    ([[1, 2]], [1, 1], [0, 0], 1e-5),
    ([1, 2], [1], [0], 1e-5),
    ([[1, 2], [3, 4]], [1], [0], 1e-5),
    ([[1, 2], [3, 4]], [1, 1], [0, 0], 0),
    ([[1, np.nan], [3, 4]], [1, 1], [0, 0], 1e-5),
])
def test_batchnorm_rejects_invalid_inputs(values, gamma, beta, eps):
    with pytest.raises(ValueError):
        batch_norm_training(values, gamma, beta, eps=eps)


def test_vanilla_sgd_l2_hand_oracle():
    coupled, decoupled = sgd_l2_pair([2., -1.], [0.5, -0.2], lr=0.1, decay=0.2)
    np.testing.assert_allclose(coupled, [1.91, -0.96])
    np.testing.assert_allclose(decoupled, coupled, atol=1e-15)


@pytest.mark.parametrize("weights,gradient,lr,decay", [
    ([], [], 0.1, 0.1), ([1], [1, 2], 0.1, 0.1),
    ([np.inf], [1], 0.1, 0.1), ([1], [1], 0, 0.1), ([1], [1], 0.1, -1),
])
def test_sgd_rejects_invalid_inputs(weights, gradient, lr, decay):
    with pytest.raises(ValueError):
        sgd_l2_pair(weights, gradient, lr=lr, decay=decay)


def test_full_checkpoint_is_copied_and_restores_batchnorm_buffers():
    model = MLP(Configuration("bn", batch_norm=True), 2)
    model.train()
    model(torch.randn(8, 20))
    monitor = EarlyStopping(patience=2, min_delta=0.01)
    assert not monitor.observe(0.5, 1, model)
    expected = copy.deepcopy(model.state_dict())
    model(torch.randn(8, 20) + 9)
    with torch.no_grad():
        model.first.weight.add_(5)
    assert not monitor.observe(0.6, 2, model)
    assert monitor.observe(0.7, 3, model)
    monitor.restore(model)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert model.norm1.num_batches_tracked.item() == 1
    assert monitor.best_epoch == 1


def test_min_delta_resets_patience_separately_from_absolute_best():
    model = nn.Linear(1, 1)
    monitor = EarlyStopping(patience=2, min_delta=0.1)
    assert not monitor.observe(1., 1, model)
    assert not monitor.observe(0.95, 2, model)
    assert monitor.observe(0.94, 3, model)
    assert monitor.best_epoch == 3
    assert monitor.best_loss == 0.94
    # Cumulative improvement relative to the last progress anchor resets patience.
    other = EarlyStopping(patience=3, min_delta=0.1)
    for epoch, loss in enumerate([1., 0.95, 0.89], 1):
        assert not other.observe(loss, epoch, model)
    assert other.bad_checks == 0


def test_checkpoint_errors():
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)
    with pytest.raises(ValueError):
        EarlyStopping(min_delta=np.nan)
    monitor, model = EarlyStopping(), nn.Linear(1, 1)
    with pytest.raises(ValueError):
        monitor.restore(model)
    with pytest.raises(ValueError):
        monitor.observe(np.inf, 1, model)


def test_eval_disables_dropout_and_preserves_running_statistics_and_rng():
    model = MLP(CONFIGURATIONS[-1], 12)
    x, y = torch.randn(12, 20), torch.arange(12) % 2
    model.train()
    model(x)
    state = copy.deepcopy(model.state_dict())
    rng_state = torch.get_rng_state().clone()
    first, second = evaluate(model, x, y), evaluate(model, x, y)
    assert first == second
    assert not model.training
    torch.testing.assert_close(rng_state, torch.get_rng_state(), rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())


def test_initial_linear_weights_match_and_decay_excludes_biases_and_norm():
    reference = linear_parameters(MLP(CONFIGURATIONS[0], 60))
    for config in CONFIGURATIONS:
        model = MLP(config, 60)
        torch.testing.assert_close(reference, linear_parameters(model), rtol=0, atol=0)
    model = MLP(CONFIGURATIONS[-1], 60)
    optimizer = make_optimizer(model, decay=0.2, lr=0.1)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for name, parameter in model.named_parameters():
        expected = before[name] * (0.98 if parameter.ndim == 2 else 1)
        torch.testing.assert_close(parameter, expected)


@pytest.mark.parametrize("n,size", [(2, 2), (3, 2), (9, 4), (65, 64), (129, 64), (10, 99)])
def test_batching_keeps_all_examples_without_singletons(n, size):
    order = torch.arange(n)
    batches = batch_indices(order, size)
    torch.testing.assert_close(torch.cat(batches), order)
    assert min(map(len, batches)) >= 2


def test_splits_disjoint_balanced_and_scaler_train_only():
    from sklearn.datasets import make_moons
    data, ids, scaler = make_data(200, 57)
    assert {key: len(value) for key, value in ids.items()} == {"train": 120, "validation": 40, "test": 40}
    assert len(set(np.concatenate(list(ids.values())))) == 200
    signal, _ = make_moons(n_samples=200, noise=0.3, random_state=57)
    raw = np.column_stack((signal, np.random.default_rng(58).normal(size=(200, 18))))
    np.testing.assert_allclose(scaler.mean_, raw[ids["train"]].mean(0))
    np.testing.assert_allclose(data["train"][0].mean(0).numpy(), 0, atol=1e-7)
    for x, y in data.values():
        assert x.dtype == torch.float32 and y.dtype == torch.long
        assert torch.bincount(y).tolist() == [len(y) // 2] * 2


@pytest.mark.parametrize("options", [
    {"n": 99}, {"n": 101}, {"seed": -1}, {"epochs": 0}, {"epochs": True},
    {"batch_size": 1}, {"patience": 0}, {"min_delta": -1},
])
def test_experiment_rejects_invalid_configuration(options):
    with pytest.raises(ValueError):
        run_comparison(**options)


def test_small_experiment_is_deterministic_and_test_is_used_only_after_selection(monkeypatch):
    original_make_data, original_evaluate = example.make_data, example.evaluate
    accesses, protected = [], []

    def capture_data(*args):
        bundle = original_make_data(*args)
        protected.append(bundle[0]["test"][0])
        return bundle

    def capture_evaluation(model, features, labels):
        if features is protected[-1]:
            accesses.append("test")
        else:
            accesses.append("development")
        return original_evaluate(model, features, labels)

    monkeypatch.setattr(example, "make_data", capture_data)
    monkeypatch.setattr(example, "evaluate", capture_evaluation)
    rng = torch.get_rng_state().clone()
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    first = run_comparison(epochs=4, n=100)
    assert accesses.count("test") == 1 and accesses[-1] == "test"
    second = run_comparison(epochs=4, n=100)
    assert accesses.count("test") == 2 and accesses[-1] == "test"
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert first["final_test"] == second["final_test"]
    assert first["selected_configuration"] == second["selected_configuration"]
    assert len({run["initial_linear_sha256"] for run in first["runs"]}) == 1
    for left, right in zip(first["runs"], second["runs"]):
        assert left["history"] == right["history"]
        checkpoint = left["history"][left["selected_epoch"] - 1]
        assert left["selected_validation"] == checkpoint["validation"]
        assert left["selected_train"] == checkpoint["train"]
        assert left["steps"] == 4
    baseline, early = first["runs"][0], first["runs"][4]
    assert baseline["history"] == early["history"]
    expected = min(first["runs"], key=lambda run: run["selected_validation"]["loss"])
    assert first["selected_configuration"] == expected["configuration"]["name"]
