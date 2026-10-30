"""Probability invariants, exact gradients, neighborhood metrics and API smoke."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits


TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load_module("day40_scratch", "from_scratch.py")
example = load_module("day40_example", "example.py")


def test_affinities_match_hand_calculation():
    X = np.array([[0.0], [1.0], [3.0]])
    weights = np.exp(-np.array([[0, 1, 9], [1, 0, 4], [9, 4, 0]]) / 2)
    np.fill_diagonal(weights, 0)
    conditional = weights / weights.sum(axis=1, keepdims=True)
    P = scratch.gaussian_affinities(X, sigma=1)
    np.testing.assert_allclose(P, (conditional + conditional.T) / 6)
    np.testing.assert_array_equal(np.diag(P), 0)
    assert P.sum() == pytest.approx(1)


def test_duplicate_points_and_small_bandwidth_have_valid_probabilities():
    uniform = scratch.gaussian_affinities(np.zeros((4, 3)))
    np.testing.assert_allclose(uniform[~np.eye(4, dtype=bool)], 1 / 12)
    sharp = scratch.gaussian_affinities([[0], [1000], [3000]], sigma=0.001)
    assert np.isfinite(sharp).all()
    np.testing.assert_allclose(sharp, sharp.T)
    assert sharp.sum() == pytest.approx(1)


def test_two_point_objective_is_zero_for_any_finite_layout():
    P = scratch.gaussian_affinities([[0], [2]])
    loss, gradient = scratch.loss_and_gradient([[3, 2], [-1, 5]], P)
    assert loss == pytest.approx(0, abs=1e-14)
    np.testing.assert_allclose(gradient, 0, atol=1e-14)


@pytest.mark.parametrize("seed", [0, 42])
def test_gradient_matches_central_finite_differences(seed):
    rng = np.random.default_rng(seed)
    P = scratch.gaussian_affinities(rng.normal(size=(7, 4)), sigma=0.8)
    Y = rng.normal(size=(7, 2))
    _, analytic = scratch.loss_and_gradient(Y, P)
    numerical = np.zeros_like(Y)
    epsilon = 1e-6
    for index in np.ndindex(Y.shape):
        plus, minus = Y.copy(), Y.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (
            scratch.loss_and_gradient(plus, P)[0] - scratch.loss_and_gradient(minus, P)[0]
        ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-5, atol=2e-9)


def test_translation_rotation_and_zero_net_gradient():
    rng = np.random.default_rng(13)
    P = scratch.gaussian_affinities(rng.normal(size=(8, 3)))
    Y = rng.normal(size=(8, 2))
    rotation = np.array([[0, -1], [1, 0]])
    loss, gradient = scratch.loss_and_gradient(Y, P)
    moved_loss, moved_gradient = scratch.loss_and_gradient(Y @ rotation + 11, P)
    assert moved_loss == pytest.approx(loss)
    np.testing.assert_allclose(moved_gradient, gradient @ rotation, atol=1e-14)
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=1e-14)


def test_optimizer_is_repeatable_and_reduces_toy_objective():
    X = np.random.default_rng(7).normal(size=(12, 3))
    Y, history = scratch.optimize(X, steps=80)
    repeated, repeated_history = scratch.optimize(X, steps=80)
    np.testing.assert_array_equal(Y, repeated)
    np.testing.assert_array_equal(history, repeated_history)
    assert history.shape == (81,)
    assert history[-1] < history[0]
    np.testing.assert_allclose(Y.mean(axis=0), 0, atol=1e-14)


@pytest.mark.parametrize("X", [[], [[1]], [[1], [np.nan]], [[0], [np.inf]], [1, 2]])
def test_invalid_data_rejected(X):
    with pytest.raises(ValueError):
        scratch.gaussian_affinities(X)


@pytest.mark.parametrize("sigma", [0, -1, np.nan, np.inf])
def test_invalid_bandwidth_rejected(sigma):
    with pytest.raises(ValueError, match="sigma"):
        scratch.gaussian_affinities([[0], [1]], sigma)


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"steps": 1.2}, {"steps": True},
                                        {"learning_rate": 0}, {"learning_rate": np.nan}])
def test_invalid_optimizer_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        scratch.optimize([[0], [1]], **kwargs)


@pytest.mark.parametrize("P", [np.ones((3, 3)), [[0, 1], [0, 0]],
                                [[0.1, 0.4], [0.4, 0.1]], [[0, -0.5], [-0.5, 0]],
                                [[0, np.nan], [np.nan, 0]]])
def test_invalid_joint_distribution_rejected(P):
    with pytest.raises(ValueError, match="P must"):
        scratch.loss_and_gradient([[0, 0], [1, 1]], P)


def test_neighbors_exclude_self_and_handle_duplicates_and_ties():
    neighbors = example.neighbor_indices([[0], [0], [1], [3]], 2)
    np.testing.assert_array_equal(neighbors, [[1, 2], [0, 2], [0, 1], [2, 0]])
    assert example.neighborhood_overlap(neighbors, neighbors) == 1
    # Every row shares exactly one of its two neighbors.
    A = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])
    B = np.array([[2, 3], [2, 3], [1, 3], [1, 2]])
    assert example.neighborhood_overlap(A, B) == 0.5


@pytest.mark.parametrize("k", [0, 3, 1.5, True])
def test_invalid_neighbor_count_rejected(k):
    with pytest.raises(ValueError):
        example.neighbor_indices([[0], [1], [2]], k)


@pytest.mark.parametrize("B", [np.array([[1], [0]]), np.array([[0, 1], [0, 1], [0, 1]]),
                                np.array([[1, 1], [0, 2], [0, 1]]),
                                np.array([[1, 3], [0, 2], [0, 1]]),
                                np.array([[1., 2.], [0., 2.], [0., 1.]])])
def test_invalid_neighbor_lists_rejected(B):
    with pytest.raises(ValueError):
        example.neighborhood_overlap(np.array([[1, 2], [0, 2], [0, 1]]), B)


@pytest.mark.parametrize("n", [60, 1798, 800.5, True])
def test_invalid_sample_size_rejected_before_fitting(n):
    with pytest.raises(ValueError, match="n_samples"):
        example.run_comparison(n)


def test_library_embeddings_smoke_with_required_optional_extra():
    # No importorskip: CI must install the manifold extra exercised here.
    methods = example.make_methods()
    X = np.random.default_rng(8).normal(size=(65, 6))
    with threadpool_limits(limits=1):
        for key in ("pca", "tsne_p30_s42", "umap_md0.1_s42"):
            model = methods[key]
            if key.startswith("umap"):
                model.set_params(n_epochs=30)
            Y = model.fit_transform(X)
            assert Y.shape == (65, 2)
            assert np.isfinite(Y).all()
            neighbors = example.neighbor_indices(Y, 10)
            recall = example.neighborhood_overlap(example.neighbor_indices(X, 10), neighbors)
            assert 0 <= recall <= 1
