"""Numerical invariants and small real-library integration checks."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from example import make_noisy_xor, run_comparison
from from_scratch import (
    constructive_xor, fit_perceptron, initialize_parameters,
    loss_and_gradients, predict_proba, train_mlp,
)

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([0, 1, 1, 0])


def test_perceptron_separates_and_but_never_certifies_xor():
    y = np.array([-1, -1, -1, 1])
    w, b, mistakes = fit_perceptron(XOR_X, y)
    assert mistakes[-1] == 0
    assert np.all(y * (XOR_X @ w + b) > 0)
    _, _, mistakes = fit_perceptron(XOR_X, 2 * XOR_Y - 1, max_epochs=30)
    assert len(mistakes) == 30 and all(value > 0 for value in mistakes)


def test_constructive_xor():
    np.testing.assert_array_equal(constructive_xor(XOR_X), XOR_Y)


def test_affine_layers_collapse():
    rng = np.random.default_rng(7)
    X, A, B = rng.normal(size=(5, 2)), rng.normal(size=(2, 4)), rng.normal(size=(4, 1))
    a, b = rng.normal(size=4), rng.normal(size=1)
    np.testing.assert_allclose((X @ A + a) @ B + b, X @ (A @ B) + a @ B + b)


def test_all_parameter_gradients_match_central_differences():
    X = np.array([[-0.4, 0.3], [0.2, -0.1], [0.7, 0.9]])
    y = np.array([0, 1, 1])
    parameters = initialize_parameters(2, 3, seed=9)
    _, gradients = loss_and_gradients(X, y, parameters)
    epsilon = 1e-6
    for name, array in parameters.items():
        numerical = np.empty_like(array)
        for index in np.ndindex(array.shape):
            plus = {key: value.copy() for key, value in parameters.items()}
            minus = {key: value.copy() for key, value in parameters.items()}
            plus[name][index] += epsilon
            minus[name][index] -= epsilon
            numerical[index] = (loss_and_gradients(X, y, plus)[0] - loss_and_gradients(X, y, minus)[0]) / (2 * epsilon)
        np.testing.assert_allclose(gradients[name], numerical, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize("logit", [-1000.0, 1000.0])
def test_extreme_logits_have_finite_loss_and_gradients(logit):
    parameters = initialize_parameters(2, 2)
    parameters["W2"][:] = 0
    parameters["b2"][:] = logit
    loss, gradients = loss_and_gradients(XOR_X, XOR_Y, parameters)
    assert loss == pytest.approx(500.0)
    assert all(np.isfinite(value).all() for value in gradients.values())
    assert np.isfinite(predict_proba(XOR_X, parameters)).all()


def test_seeded_mlp_fits_truth_table_and_keeps_shapes():
    parameters, history = train_mlp(XOR_X, XOR_Y)
    assert history.shape == (10001,)
    assert np.isfinite(history).all() and history[-1] < 0.02 < history[0]
    assert sum(value.size for value in parameters.values()) == 33
    np.testing.assert_array_equal(predict_proba(XOR_X, parameters) >= 0.5, XOR_Y)


def test_loss_averages_over_batch_and_does_not_mutate_parameters():
    parameters = initialize_parameters(2)
    before = {key: value.copy() for key, value in parameters.items()}
    loss, grads = loss_and_gradients(XOR_X, XOR_Y, parameters)
    repeated_loss, repeated_grads = loss_and_gradients(np.tile(XOR_X, (2, 1)), np.tile(XOR_Y, 2), parameters)
    assert loss == pytest.approx(repeated_loss)
    for key in parameters:
        np.testing.assert_allclose(grads[key], repeated_grads[key])
        np.testing.assert_array_equal(parameters[key], before[key])


@pytest.mark.parametrize("X,y", [([], []), ([[np.nan, 0]], [0]), ([[1, 0]], [2]), ([[1, 0]], [[0]])])
def test_invalid_training_data(X, y):
    with pytest.raises(ValueError):
        train_mlp(X, y, epochs=1)


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"epochs": 1.5}, {"n_hidden": 0},
                                       {"learning_rate": 0}, {"learning_rate": np.nan}])
def test_invalid_training_configuration(kwargs):
    with pytest.raises(ValueError):
        train_mlp(XOR_X, XOR_Y, **kwargs)


def test_invalid_parameters_and_constructive_inputs():
    parameters = initialize_parameters(2)
    parameters["b2"] = np.zeros(2)
    with pytest.raises(ValueError, match="shapes"):
        predict_proba(XOR_X, parameters)
    with pytest.raises(ValueError, match="binary"):
        constructive_xor([[0.5, 1]])
    with pytest.raises(ValueError):
        fit_perceptron(XOR_X, XOR_Y)


@pytest.mark.parametrize("kwargs", [{"n_samples": 3}, {"n_samples": 8.5}, {"noise": -1}, {"noise": np.inf}])
def test_invalid_generator_configuration(kwargs):
    with pytest.raises(ValueError):
        make_noisy_xor(**kwargs)


def test_generator_is_deterministic_with_latent_corner_labels():
    X, y = make_noisy_xor(noise=0)
    X2, y2 = make_noisy_xor(noise=0)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y, y2)
    np.testing.assert_array_equal(y, np.logical_xor(X[:, 0], X[:, 1]))


@pytest.mark.integration
def test_comparison_runs_real_pipelines_with_finite_metrics():
    result = run_comparison()
    assert (result["train_samples"], result["test_samples"]) == (600, 200)
    assert set(result["models"]) == {"logistic", "mlp_identity", "mlp_relu"}
    for metrics in result["models"].values():
        assert 0 <= metrics["test_accuracy"] <= 1
        assert 0 <= metrics["test_balanced_accuracy"] <= 1
        assert np.isfinite(metrics["test_log_loss"])
        assert metrics["iterations"] > 0
