"""Local derivatives, coupled Jacobians, stable losses and input contracts."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf, expit, logsumexp, softmax as scipy_softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from from_scratch import (
    ELEMENTWISE, activation_and_derivative, cross_entropy_with_logits,
    finite_difference_jacobian, softmax, softmax_jacobian,
)


@pytest.mark.parametrize("name", ELEMENTWISE)
@pytest.mark.parametrize("epsilon", [1e-4, 1e-5, 1e-6])
def test_elementwise_derivatives_away_from_kinks(name, epsilon):
    x = np.array([-4.0, -1.2, -0.2, 0.3, 1.7, 4.0])
    _, derivative = activation_and_derivative(x, name)
    numerical = finite_difference_jacobian(
        lambda v: activation_and_derivative(v, name)[0], x, epsilon
    )
    np.testing.assert_allclose(numerical, np.diag(derivative), rtol=2e-6, atol=2e-9)


def test_reference_values_and_saturation():
    x = np.array([-1000.0, -8.0, 0.0, 8.0, 1000.0])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        sigmoid, derivative = activation_and_derivative(x, "sigmoid")
        np.testing.assert_allclose(sigmoid, expit(x), atol=1e-16)
        assert derivative[2] == 0.25
        assert derivative[1] == derivative[-2]
        for name in ELEMENTWISE:
            values, slopes = activation_and_derivative(x, name)
            assert np.all(np.isfinite(values)) and np.all(np.isfinite(slopes))
        np.testing.assert_allclose(np.tanh(x / 2), 2 * sigmoid - 1, atol=1e-15)
        gelu, _ = activation_and_derivative(x, "gelu")
        np.testing.assert_allclose(gelu, x * 0.5 * (1 + erf(x / np.sqrt(2))), atol=1e-12)


def test_kink_conventions_are_not_ordinary_derivatives():
    assert activation_and_derivative(0.0, "relu")[1] == 0
    assert activation_and_derivative(0.0, "leaky_relu", 0.1)[1] == 0.1
    central = finite_difference_jacobian(
        lambda x: activation_and_derivative(x, "relu")[0], [0.0]
    )
    assert central[0, 0] == pytest.approx(0.5)


def test_leaky_slope_and_gelu_are_distinct_from_relu():
    value, derivative = activation_and_derivative([-2.0, 2.0], "leaky_relu", 0.1)
    np.testing.assert_allclose(value, [-0.2, 2.0])
    np.testing.assert_allclose(derivative, [0.1, 1.0])
    assert activation_and_derivative(-2.0, "gelu")[1] < 0
    assert activation_and_derivative(2.0, "gelu")[1] > 1


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_softmax_axis_reference_shift_and_no_mutation(axis, temperature):
    z = np.arange(24.0).reshape(2, 3, 4)
    before = z.copy()
    p = softmax(z, axis=axis, temperature=temperature)
    np.testing.assert_allclose(p, scipy_softmax(z / temperature, axis=axis), atol=1e-15)
    np.testing.assert_allclose(p.sum(axis=axis), 1, atol=1e-15)
    np.testing.assert_allclose(p, softmax(z + 10000, axis, temperature), atol=1e-15)
    np.testing.assert_array_equal(z, before)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_softmax_dense_jacobian(temperature):
    z = np.array([2.0, 1.0, -1.0])
    j = softmax_jacobian(z, temperature)
    numerical = finite_difference_jacobian(lambda v: softmax(v, temperature=temperature), z)
    np.testing.assert_allclose(j, numerical, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(j, j.T, atol=1e-15)
    np.testing.assert_allclose(j @ np.ones(3), 0, atol=1e-15)
    assert np.all(j[np.triu_indices(3, 1)] < 0)


def test_temperature_preserves_argmax_and_changes_entropy():
    z = [2.0, 1.0, -1.0]
    entropies = []
    for t in [0.5, 1, 2]:
        p = softmax(z, temperature=t)
        assert p.argmax() == 0
        entropies.append(-np.sum(p * np.log(p)))
    assert entropies[0] < entropies[1] < entropies[2]
    np.testing.assert_allclose(softmax([4.0, 4.0, 4.0]), 1 / 3)
    np.testing.assert_allclose(softmax([7.0]), [1.0])


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("soft_targets", [False, True])
def test_mean_cross_entropy_and_every_logit_gradient(temperature, soft_targets):
    z = np.array([[2.0, 1.0, -1.0], [-1.0, 0.0, 3.0]])
    y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    if soft_targets:
        y = 0.8 * y + 0.2 / 3
    before_z, before_y = z.copy(), y.copy()
    loss, gradient = cross_entropy_with_logits(z, y, temperature)
    reference = np.mean(logsumexp(z / temperature, axis=1) - np.sum(y * z / temperature, axis=1))
    assert loss == pytest.approx(reference, abs=1e-14)
    for index in np.ndindex(z.shape):
        plus, minus = z.copy(), z.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        numerical = (cross_entropy_with_logits(plus, y, temperature)[0]
                     - cross_entropy_with_logits(minus, y, temperature)[0]) / 2e-5
        assert gradient[index] == pytest.approx(numerical, rel=1e-6, abs=1e-9)
    repeated_loss, repeated_gradient = cross_entropy_with_logits(
        np.tile(z, (2, 1)), np.tile(y, (2, 1)), temperature
    )
    assert repeated_loss == pytest.approx(loss)
    np.testing.assert_allclose(repeated_gradient[:2], gradient / 2, atol=1e-15)
    np.testing.assert_array_equal(z, before_z)
    np.testing.assert_array_equal(y, before_y)


def test_extreme_logits_and_common_offset_before_temperature():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        p = softmax([[1000.0, -1000.0]])
        np.testing.assert_array_equal(p, [[1.0, 0.0]])
        loss, gradient = cross_entropy_with_logits([[1000.0, -1000.0]], [[0.0, 1.0]])
        assert loss == 2000
        np.testing.assert_array_equal(gradient, [[1.0, -1.0]])
        np.testing.assert_allclose(softmax([1e308, 1e308], temperature=0.5), [0.5, 0.5])


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [-np.inf], [1j], ["1"]])
def test_invalid_numeric_inputs(values):
    for function in (softmax, softmax_jacobian,
                     lambda x: activation_and_derivative(x, "relu")):
        with pytest.raises(ValueError):
            function(values)


@pytest.mark.parametrize("temperature", [0, -1, np.inf, np.nan])
def test_invalid_temperatures(temperature):
    with pytest.raises(ValueError):
        softmax([1, 2], temperature=temperature)
    with pytest.raises(ValueError):
        cross_entropy_with_logits([[1, 2]], [[0, 1]], temperature)


@pytest.mark.parametrize("axis", [None, 2, -3, 0.5, True])
def test_invalid_axes(axis):
    with pytest.raises(ValueError):
        softmax([[1, 2]], axis=axis)


@pytest.mark.parametrize("alpha", [0, -0.1, 1, 2, np.inf, np.nan])
def test_invalid_leaky_slopes(alpha):
    with pytest.raises(ValueError):
        activation_and_derivative([1.0], "leaky_relu", alpha)


@pytest.mark.parametrize("z,y", [([1, 2], [0, 1]), ([[1, 2]], [[1]]),
                                  ([[1, 2]], [[0.2, 0.2]]),
                                  ([[1, 2]], [[-0.1, 1.1]]),
                                  ([[1, 2]], [[np.nan, 1]])])
def test_invalid_loss_inputs(z, y):
    with pytest.raises(ValueError):
        cross_entropy_with_logits(z, y)


def test_invalid_derivative_requests_and_unrepresentable_scales():
    with pytest.raises(ValueError):
        activation_and_derivative([1], "softmax")
    with pytest.raises(ValueError):
        softmax(1.0)
    with pytest.raises(ValueError):
        softmax_jacobian([[1, 2]])
    with pytest.raises(ValueError):
        softmax([1e308, -1e308])
    for epsilon in [0, -1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            finite_difference_jacobian(softmax, [1, 2], epsilon)
    with pytest.raises(ValueError):
        finite_difference_jacobian(softmax, [1e30, 1e30])
    with pytest.raises(ValueError):
        finite_difference_jacobian(lambda x: x.sum(), [1, 2])
    with pytest.raises(ValueError):
        finite_difference_jacobian(softmax, [[1, 2]])
