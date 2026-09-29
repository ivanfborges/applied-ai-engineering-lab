"""Numerical checks for the Day 24 visual experiments."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors


TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


visuals = load_module("day24_visualizations", "visualizations.py")
three_d = load_module("day24_interactive_3d", "interactive_3d.py")


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_neighbor_selection_matches_sklearn(metric):
    X = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, 0.5], [2.0, 1.0]])
    query = np.array([0.08, 0.05])
    selected, distances = visuals.nearest_indices(X, query, 3, metric=metric)
    expected_distances, expected_indices = NearestNeighbors(
        n_neighbors=3, metric=metric
    ).fit(X).kneighbors(query.reshape(1, -1))
    np.testing.assert_array_equal(selected, expected_indices[0])
    np.testing.assert_allclose(distances[selected], expected_distances[0])


def test_invalid_neighborhood_inputs():
    with pytest.raises(ValueError, match="finite"):
        visuals.nearest_indices([[np.nan, 0.0]], [0.0, 0.0], 1)
    with pytest.raises(ValueError, match="between"):
        visuals.nearest_indices([[0.0, 0.0]], [0.0, 0.0], 2)
    with pytest.raises(ValueError, match="metric"):
        visuals.nearest_indices([[0.0, 0.0]], [0.0, 0.0], 1, metric="cosine")


def test_simulations_are_finite_and_reproducible():
    first, distributions = visuals.simulate_distance_concentration()
    second, _ = visuals.simulate_distance_concentration()
    np.testing.assert_array_equal(first, second)
    assert first.shape == (len(visuals.DIMENSIONS), 6)
    assert np.isfinite(first).all()
    assert all(np.isfinite(values).all() for values in distributions.values())
    irrelevant = visuals.experiment_irrelevant_features()
    assert irrelevant.shape == (len(visuals.NOISE_DIMENSIONS), 2)
    assert np.isfinite(irrelevant).all()
    assert np.all((irrelevant[:, 1] >= 0) & (irrelevant[:, 1] <= 1))


def test_three_dimensional_neighbors_are_exact():
    X, y, query = three_d.make_three_feature_data()
    fig, selected, distances = three_d.build_figure(X, y, query, k=5)
    expected = np.argsort(np.linalg.norm(X - query, axis=1), kind="stable")[:5]
    np.testing.assert_array_equal(selected, expected)
    assert np.isfinite(distances).all()
    assert len(fig.data) == 2 + 5 + 2
