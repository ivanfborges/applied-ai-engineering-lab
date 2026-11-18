"""Independent update oracles, framework agreement and matched experiment controls."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from from_scratch import METHODS, Optimizer, quadratic
from example import batch_orders, make_data, make_optimizer, run_comparison


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("decay", [0.0, 0.2])
@pytest.mark.parametrize("eps", [1e-8, 0.05])
def test_multistep_dense_float64_against_pytorch(method, decay, eps):
    initial = np.array([[2.0, -1.0, 0.0], [0.1, -0.7, 3.0]])
    educational = Optimizer(initial, method, 0.03, weight_decay=decay, eps=eps)
    parameter = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float64))
    options = dict(lr=0.03, weight_decay=decay, foreach=False)
    if method in ("sgd", "momentum"):
        reference = torch.optim.SGD([parameter], momentum=0.9 if method == "momentum" else 0,
                                    dampening=0, nesterov=False, **options)
    elif method == "rmsprop":
        reference = torch.optim.RMSprop([parameter], alpha=0.99, momentum=0,
                                        centered=False, eps=eps, **options)
    else:
        cls = torch.optim.Adam if method == "adam" else torch.optim.AdamW
        reference = cls([parameter], betas=(0.9, 0.999), amsgrad=False, eps=eps, **options)
    gradients = np.random.default_rng(56).normal(size=(25, 2, 3))
    gradients[3] = 0
    gradients[7] *= 1e-10
    for step, gradient in enumerate(gradients, start=1):
        parameter.grad = torch.tensor(gradient, dtype=torch.float64)
        reference.step()
        np.testing.assert_allclose(educational.step(gradient), parameter.detach().numpy(),
                                   rtol=2e-12, atol=2e-13)
        assert educational.steps == step
    state = reference.state[parameter]
    if method == "momentum":
        np.testing.assert_allclose(educational.state["first"], state["momentum_buffer"].numpy())
    if method == "rmsprop":
        np.testing.assert_allclose(educational.state["second"], state["square_avg"].numpy())
    if method in ("adam", "adamw"):
        np.testing.assert_allclose(educational.state["first"], state["exp_avg"].numpy())
        np.testing.assert_allclose(educational.state["second"], state["exp_avg_sq"].numpy())


def test_hand_calculated_sgd_and_momentum_history():
    optimizer = Optimizer([1.0], "sgd", 0.1, weight_decay=0.2)
    np.testing.assert_allclose(optimizer.step([2.0]), [0.78])
    momentum = Optimizer([1.0], "momentum", 0.1, momentum=0.5)
    np.testing.assert_allclose(momentum.step([2.0]), [0.8])
    np.testing.assert_allclose(momentum.step([-1.0]), [0.8])
    np.testing.assert_allclose(momentum.step([0.0]), [0.8])


def test_adam_bias_correction_and_rmsprop_epsilon_placement():
    adam = Optimizer([1.0, -2.0], "adam", 0.1, eps=0.5)
    np.testing.assert_allclose(adam.step([2.0, -4.0]), [0.92, -2 + 0.4 / 4.5])
    rmsprop = Optimizer([1.0], "rmsprop", 0.1, alpha=0.75, eps=0.5)
    np.testing.assert_allclose(rmsprop.step([2.0]), [1 - 0.2 / 1.5])


def test_adamw_zero_gradient_decay_bypasses_moments():
    coupled = Optimizer([2.0], "adam", 0.1, weight_decay=0.2)
    decoupled = Optimizer([2.0], "adamw", 0.1, weight_decay=0.2)
    np.testing.assert_allclose(decoupled.step([0.0]), [1.96])
    np.testing.assert_array_equal(decoupled.state["first"], [0.0])
    np.testing.assert_array_equal(decoupled.state["second"], [0.0])
    np.testing.assert_allclose(coupled.step([0.0]), [2 - 0.04 / (0.4 + 1e-8)])
    assert coupled.state["first"][0] > 0
    assert coupled.state["second"][0] > 0


def test_adam_adamw_zero_decay_and_residual_momentum():
    adam = Optimizer([2.0, -1.0], "adam", 0.1)
    adamw = Optimizer([2.0, -1.0], "adamw", 0.1)
    for gradient in ([1.0, -0.1], [-2.0, 0.5], [0.0, 0.0]):
        np.testing.assert_array_equal(adam.step(gradient), adamw.step(gradient))
    assert np.any(adam.state["first"] != 0)
    before = adam.parameters
    assert np.any(adam.step([0.0, 0.0]) != before)


@pytest.mark.parametrize("method,count", list(zip(METHODS, [0, 1, 1, 2, 2])))
def test_only_needed_state_is_allocated_and_copies_are_isolated(method, count):
    initial = np.array([2.0, -1.0])
    optimizer = Optimizer(initial, method, 0.1)
    assert len(optimizer.state) == count
    initial[:] = 999
    exposed = optimizer.parameters
    exposed[:] = -999
    for state in optimizer.state.values():
        state[:] = 999
    np.testing.assert_array_equal(optimizer.parameters, [2, -1])
    for state in optimizer.state.values():
        np.testing.assert_array_equal(state, [0, 0])
    gradient = np.array([1.0, 1.0])
    result = optimizer.step(gradient)
    result[:] = 999
    np.testing.assert_array_equal(gradient, [1, 1])
    assert np.all(optimizer.parameters != 999)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", [[1.0], [[1.0, 2.0]], [np.nan, 0], [0, np.inf],
                                 [1j, 0], [True, False], ["1", "2"], []])
def test_invalid_gradient_is_rejected_without_advancing_state(method, bad):
    optimizer = Optimizer([2.0, -1.0], method, 0.1)
    optimizer.step([1.0, 0.5])
    before, state, steps = optimizer.parameters, optimizer.state, optimizer.steps
    with pytest.raises(ValueError):
        optimizer.step(bad)
    np.testing.assert_array_equal(optimizer.parameters, before)
    assert optimizer.steps == steps
    for name in state:
        np.testing.assert_array_equal(optimizer.state[name], state[name])


@pytest.mark.parametrize("key,value", [("lr", 0), ("lr", -1), ("lr", np.inf),
    ("lr", True), ("lr", "0.1"), ("weight_decay", -1), ("weight_decay", np.nan),
    ("momentum", 1), ("momentum", -0.1), ("alpha", 1), ("alpha", -0.1),
    ("beta1", 1), ("beta2", 1), ("beta1", -0.1), ("eps", 0), ("eps", np.inf)])
def test_invalid_hyperparameters(key, value):
    options = {"lr": 0.1, key: value}
    with pytest.raises(ValueError):
        Optimizer([1.0], "adam", **options)


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf], [1j], [True], ["1"]])
def test_invalid_parameters(bad):
    with pytest.raises(ValueError):
        Optimizer(bad, "sgd", 0.1)


def test_unknown_method_and_overflow_are_explicit_and_atomic():
    with pytest.raises(ValueError):
        Optimizer([1.0], "adma", 0.1)
    optimizer = Optimizer([1.0], "adam", 0.1)
    with pytest.raises(ValueError, match="overflowed"):
        optimizer.step([1e308])
    assert optimizer.steps == 0
    np.testing.assert_array_equal(optimizer.parameters, [1.0])
    np.testing.assert_array_equal(optimizer.state["first"], [0.0])


def test_quadratic_gradient_minimum_and_sgd_stability_boundary():
    point = np.array([0.7, -1.3])
    _, gradient = quadratic(point)
    finite_difference = []
    for axis in np.eye(2):
        finite_difference.append((quadratic(point + 1e-5 * axis)[0]
                                  - quadratic(point - 1e-5 * axis)[0]) / 2e-5)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-10)
    assert quadratic([0.0, 0.0])[0] == 0
    stable = Optimizer([2.0, -1.0], "sgd", 0.06)
    unstable = Optimizer([2.0, -1.0], "sgd", 0.21)
    for _ in range(50):
        stable.step(quadratic(stable.parameters)[1])
        unstable.step(quadratic(unstable.parameters)[1])
    assert quadratic(stable.parameters)[0] < 1e-5
    assert quadratic(unstable.parameters)[0] > 21
    with pytest.raises(ValueError):
        quadratic([1.0])


def test_data_and_batches_are_reproducible_with_complete_epoch_coverage():
    first, second = make_data(100, 56), make_data(100, 56)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert first[0].shape == (80, 2) and first[2].shape == (20, 2)
    assert sorted(torch.cat([first[1], first[3]]).tolist()) == [0] * 50 + [1] * 50
    orders = batch_orders(80, 3, 57)
    for order in orders:
        assert torch.equal(order.sort().values, torch.arange(80))
        assert sum(len(batch) for batch in order.split(17)) == 80
    assert not torch.equal(orders[0], orders[1])


def test_small_real_training_has_matched_initialization_budget_and_repeatable_results():
    first = run_comparison(epochs=2, batch_size=17, n=100, seed=56)
    second = run_comparison(epochs=2, batch_size=17, n=100, seed=56)
    assert first["configuration"] == second["configuration"]
    assert len(first["runs"]) == 7
    for a, b in zip(first["runs"], second["runs"]):
        assert a["steps"] == 10
        assert a["initial_parameters_sha256"] == first["configuration"]["initial_parameters_sha256"]
        assert a["final_parameters_sha256"] == b["final_parameters_sha256"]
        assert a["history"] == b["history"]
        assert np.isfinite(a["final_validation"]["loss"])
        assert 0 <= a["final_validation"]["accuracy"] <= 1
    assert first["runs"][3]["final_parameters_sha256"] == first["runs"][4]["final_parameters_sha256"]
    assert first["runs"][5]["final_parameters_sha256"] != first["runs"][6]["final_parameters_sha256"]


@pytest.mark.parametrize("options", [{"epochs": 0}, {"batch_size": 0}, {"n": 19},
                                      {"n": 21}, {"seed": -1}, {"epochs": True}])
def test_invalid_experiment_configuration(options):
    with pytest.raises(ValueError):
        run_comparison(**options)


def test_framework_factory_rejects_unknown_method():
    with pytest.raises(ValueError):
        make_optimizer([torch.nn.Parameter(torch.ones(1))], "adma", 0.1, 0)
