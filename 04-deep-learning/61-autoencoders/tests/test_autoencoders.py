"""Numerical oracles, calibration boundaries and synthetic study integration."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from from_scratch import fit_linear_autoencoder, linear_loss_gradients, matrix
from example import (Autoencoder, ae_scores, calibrate_threshold, evaluate_scores,
                     fit_autoencoder, make_data, reconstruction_scores, run_study)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_all_linear_gradients_against_finite_differences_and_autograd():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(5, 3))
    W = rng.normal(size=(3, 2))
    V = rng.normal(size=(2, 3))
    loss, grad_W, grad_V = linear_loss_gradients(X, W, V)
    for parameter, gradient in ((W, grad_W), (V, grad_V)):
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + 1e-6
            plus = linear_loss_gradients(X, W, V)[0]
            parameter[index] = original - 1e-6
            minus = linear_loss_gradients(X, W, V)[0]
            parameter[index] = original
            assert gradient[index] == pytest.approx((plus - minus) / 2e-6, abs=1e-8)
    wt = torch.tensor(W, requires_grad=True)
    vt = torch.tensor(V, requires_grad=True)
    xt = torch.tensor(X)
    reference = ((xt @ wt @ vt - xt)**2).mean()
    reference.backward()
    assert loss == pytest.approx(reference.item())
    np.testing.assert_allclose(grad_W, wt.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(grad_V, vt.grad.numpy(), atol=1e-12)


def test_update_is_simultaneous():
    X = np.array([[1., -1.], [-1., 1.]])
    rng = np.random.default_rng(4)
    W, V = rng.normal(0, 0.1, (2, 1)), rng.normal(0, 0.1, (1, 2))
    _, dw, dv = linear_loss_gradients(X, W, V)
    actual_W, actual_V, losses = fit_linear_autoencoder(X, 1, steps=1, seed=4)
    np.testing.assert_allclose(actual_W, W - 0.4 * dw)
    np.testing.assert_allclose(actual_V, V - 0.4 * dv)
    assert len(losses) == 2


def test_linear_fit_approaches_svd_bound_and_latent_coordinates_are_not_identified():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(150, 2)) @ rng.normal(size=(2, 5)) + 0.05 * rng.normal(size=(150, 5))
    X -= X.mean(axis=0)
    X /= X.std(axis=0)
    W, V, losses = fit_linear_autoencoder(X)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    optimum = np.mean((X - X @ vt[:2].T @ vt[:2])**2)
    assert losses[-1] >= optimum - 1e-12
    assert losses[-1] < optimum + 2e-5
    transform = np.array([[2., 1.], [0., 0.5]])
    np.testing.assert_allclose(X @ W @ V, X @ (W @ transform) @ (np.linalg.inv(transform) @ V), atol=1e-12)


@pytest.mark.parametrize("bad", [[], [1, 2], [[np.nan]], [[np.inf]], np.empty((0, 2))])
def test_matrix_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError):
        matrix(bad)


@pytest.mark.parametrize("kwargs", [{"latent_dim": 0}, {"latent_dim": 4}, {"steps": 0},
                                   {"steps": 1.5}, {"learning_rate": 0}, {"learning_rate": np.nan}])
def test_linear_fit_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        fit_linear_autoencoder(np.ones((3, 3)), **kwargs)


def test_reconstruction_scores_are_per_observation_and_reject_broadcasting():
    np.testing.assert_allclose(reconstruction_scores([[1, 2], [3, 4]], [[1, 0], [1, 4]]), [2, 2])
    with pytest.raises(ValueError):
        reconstruction_scores(np.ones((2, 3)), np.ones((1, 3)))


def test_quantile_ties_and_strict_alert_rule():
    threshold = calibrate_threshold([0, 1, 1, 2], 0.5)
    assert threshold == 1
    metrics = evaluate_scores([0, 0, 1, 1], [0, 1, 1, 2], threshold)
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 0, 1, 2)
    assert metrics["recall"] == 0.5 and metrics["precision"] == 1
    assert evaluate_scores([0, 1], [0, 1], 2)["precision"] == 0


@pytest.mark.parametrize("scores,q", [([], .99), ([np.nan], .99), ([[1]], .99),
                                    ([1], 0), ([1], 1), ([1], np.nan)])
def test_invalid_calibration(scores, q):
    with pytest.raises(ValueError):
        calibrate_threshold(scores, q)


@pytest.mark.parametrize("labels,scores,threshold", [([0, 0], [1, 2], 1), ([0, 2], [1, 2], 1),
                                                    ([0, 1], [1], 1), ([0, 1], [1, 2], np.inf)])
def test_invalid_evaluation(labels, scores, threshold):
    with pytest.raises(ValueError):
        evaluate_scores(labels, scores, threshold)


def test_generator_is_deterministic_disjoint_and_injects_documented_mechanisms():
    data, repeat = make_data(), make_data()
    np.testing.assert_array_equal(data.test, repeat.test)
    assert data.train.shape == (1800, 12) and data.calibration.shape == (800, 12)
    assert np.sum(data.labels) == 40 and len(data.labels) == 400
    assert not np.shares_memory(data.train, data.test)
    # No repeated rows across splits; shared generating geometry is intentional.
    combined = np.vstack((data.train, data.calibration, data.test))
    assert len(np.unique(combined, axis=0)) == len(combined)
    assert (data.test[data.kinds == "off_subspace"] @ data.basis[:, 2]).mean() > 3.8
    assert (data.test[data.kinds == "on_subspace"] @ data.basis[:, 0]).mean() > 5
    with pytest.raises(ValueError):
        make_data(n_train=0)


def test_framework_fit_shapes_loss_and_determinism():
    X = np.random.default_rng(1).normal(size=(48, 4))
    model, history = fit_autoencoder(X, epochs=10)
    repeated, repeated_history = fit_autoencoder(X, epochs=10)
    assert model.encoder(torch.tensor(X, dtype=torch.float32)).shape == (48, 2)
    assert history[-1] < history[0]
    assert history == repeated_history
    np.testing.assert_array_equal(ae_scores(model, X), ae_scores(repeated, X))
    assert not model.training
    with pytest.raises(ValueError):
        ae_scores(model, np.ones((2, 3)))
    with pytest.raises(ValueError):
        Autoencoder(4, 0)
    with pytest.raises(ValueError):
        fit_autoencoder(X, epochs=0)


def test_study_training_and_calibration_are_unchanged_by_test_perturbation(monkeypatch):
    import example

    data = make_data(n_train=100, n_calibration=40, n_normal=20, n_each_anomaly=5)
    monkeypatch.setattr(example, "make_data", lambda seed: data)
    original = run_study(epochs=2)
    data.test = data.test * 1000
    perturbed = run_study(epochs=2)
    assert original["ae_train_mse"] == perturbed["ae_train_mse"]
    for name in original["results"]:
        assert original["results"][name]["threshold"] == perturbed["results"][name]["threshold"]


def test_small_end_to_end_study_restores_execution_settings():
    threads = torch.get_num_threads()
    determinism = torch.are_deterministic_algorithms_enabled()
    report = run_study(epochs=2)
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == determinism
    assert report["interpretation_status"] == "author review required"
    for result in report["results"].values():
        assert result["tp"] + result["fn"] == 40
        assert result["fp"] + result["tn"] == 360
        assert 0 <= result["average_precision"] <= 1
        assert set(result["recall_by_kind"]) == {"off_subspace", "on_subspace"}
