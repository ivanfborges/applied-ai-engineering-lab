"""Numerical contracts, graph branching and parameter update semantics."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from from_scratch import (
    finite_difference_gradients, initialize_parameters, loss_and_gradients,
    loss_value, sgd_step, shared_path_trace,
)


@pytest.fixture
def problem():
    X = np.array([[-0.8, 0.2], [0.3, -0.1], [0.6, 0.9]])
    y = np.array([[0.2], [-0.4], [0.7]])
    p = initialize_parameters(2, 3, seed=9)
    p["b1"][:] = [[0.1, -0.2, 0.3]]
    p["b2"][:] = 0.15
    return X, y, p


@pytest.mark.parametrize("epsilon", [1e-4, 1e-5, 1e-6])
def test_every_parameter_against_finite_differences(problem, epsilon):
    X, y, p = problem
    before = {name: value.copy() for name, value in p.items()}
    _, gradients = loss_and_gradients(X, y, p)
    numerical = finite_difference_gradients(X, y, p, epsilon)
    for name in p:
        assert gradients[name].shape == p[name].shape
        np.testing.assert_allclose(gradients[name], numerical[name], rtol=2e-5, atol=1e-8)
        np.testing.assert_array_equal(before[name], p[name])


def test_known_linear_output_and_bias_reduction():
    p = {"W1": np.array([[0.0]]), "b1": np.array([[0.0]]),
         "W2": np.array([[2.0]]), "b2": np.array([[1.0]])}
    loss, g = loss_and_gradients([[1.0], [3.0]], [[0.0], [0.0]], p)
    assert loss == 0.5
    for name, expected in (("W1", 4.0), ("b1", 2.0), ("W2", 0.0), ("b2", 1.0)):
        np.testing.assert_allclose(g[name], [[expected]])


def test_branch_contributions_are_added():
    trace = shared_path_trace()
    z = 0.5
    residual = np.tanh(z) + z - 0.4
    expected_dz = residual * (2 - np.tanh(z)**2)
    assert trace["dL_dz"] == pytest.approx(expected_dz)
    assert trace["dL_dw"] == pytest.approx(3 * expected_dz)
    for name, value, gradient in (("w", 0.2, trace["dL_dw"]),
                                  ("b", -0.1, trace["dL_db"]),
                                  ("x", 3.0, trace["dL_dx"])):
        plus = shared_path_trace(**{name: value + 1e-5})["loss"]
        minus = shared_path_trace(**{name: value - 1e-5})["loss"]
        assert gradient == pytest.approx((plus - minus) / 2e-5, rel=1e-7)


def test_mean_loss_is_invariant_to_batch_replication(problem):
    X, y, p = problem
    loss, g = loss_and_gradients(X, y, p)
    repeated_loss, repeated = loss_and_gradients(np.tile(X, (3, 1)), np.tile(y, (3, 1)), p)
    assert loss == pytest.approx(repeated_loss)
    for name in p:
        np.testing.assert_allclose(g[name], repeated[name], atol=1e-15)


def test_unequal_microbatches_need_sample_weights(problem):
    X, y, p = problem
    _, full = loss_and_gradients(X, y, p)
    _, first = loss_and_gradients(X[:1], y[:1], p)
    _, rest = loss_and_gradients(X[1:], y[1:], p)
    for name in p:
        np.testing.assert_allclose(full[name], first[name] / 3 + 2 * rest[name] / 3, atol=1e-15)
    assert not np.allclose(full["b2"], (first["b2"] + rest["b2"]) / 2)


def test_backward_is_read_only_and_step_is_simultaneous(problem):
    X, y, p = problem
    before = {name: v.copy() for name, v in p.items()}
    loss, g = loss_and_gradients(X, y, p)
    new = sgd_step(p, g, 0.01)
    for name in p:
        np.testing.assert_array_equal(p[name], before[name])
        np.testing.assert_array_equal(new[name], before[name] - 0.01 * g[name])
        assert not np.shares_memory(new[name], p[name])
    assert loss_value(X, y, new) < loss


@pytest.mark.parametrize("X,y", [([], []), ([[1, 2]], [1]),
                                     ([[1]], [[1]]), ([[np.nan, 0]], [[0]]),
                                     ([[1, 0]], [[np.inf]])])
def test_invalid_data(problem, X, y):
    with pytest.raises(ValueError):
        loss_and_gradients(X, y, problem[2])


@pytest.mark.parametrize("name,value", [("W1", [[np.inf, 1, 2], [0, 0, 0]]),
                                          ("b1", [0, 0, 0]), ("W2", [[1, 2, 3]]),
                                          ("b2", [[1, 2]])])
def test_invalid_parameter_shapes(problem, name, value):
    p = {**problem[2], name: np.array(value)}
    with pytest.raises(ValueError):
        loss_value(problem[0], problem[1], p)


@pytest.mark.parametrize("value", [0, -0.1, np.inf, np.nan])
def test_invalid_step_and_perturbation(problem, value):
    X, y, p = problem
    _, g = loss_and_gradients(X, y, p)
    with pytest.raises(ValueError):
        sgd_step(p, g, value)
    with pytest.raises(ValueError):
        finite_difference_gradients(X, y, p, value)


def test_invalid_gradient_cannot_silently_broadcast(problem):
    X, y, p = problem
    _, g = loss_and_gradients(X, y, p)
    g["W1"] = np.zeros((1, 3))
    with pytest.raises(ValueError):
        sgd_step(p, g, 0.1)


@pytest.mark.parametrize("features,hidden", [(0, 2), (2, -1), (True, 2), (2, 1.5)])
def test_invalid_architecture(features, hidden):
    with pytest.raises(ValueError):
        initialize_parameters(features, hidden)


def test_large_step_can_increase_a_correct_loss(problem):
    X, y, p = problem
    loss, g = loss_and_gradients(X, y, p)
    assert loss_value(X, y, sgd_step(p, g, 100)) > loss
