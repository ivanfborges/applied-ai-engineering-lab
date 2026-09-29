"""Behavioral checks for the educational classifier and synthetic generator."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier


TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load_module("day24_knn", "from_scratch.py")
example = load_module("day24_example", "example.py")


def test_parity_without_distance_or_vote_ties():
    X = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, 1.0], [1.1, 0.9]])
    y = np.array([0, 0, 1, 1])
    queries = np.array([[0.05, 0.02], [1.05, 0.95]])
    expected = KNeighborsClassifier(n_neighbors=3).fit(X, y).predict(queries)
    np.testing.assert_array_equal(scratch.KNNClassifier(3).fit(X, y).predict(queries), expected)


def test_vote_tie_uses_sorted_label():
    model = scratch.KNNClassifier(2).fit([[0.0], [2.0]], ["z", "a"])
    assert model.predict([[1.0]]).tolist() == ["a"]


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        scratch.KNNClassifier(k)


def test_invalid_training_or_query_data():
    with pytest.raises(ValueError, match="one label"):
        scratch.KNNClassifier().fit([[0.0], [1.0]], [0])
    with pytest.raises(ValueError, match="cannot exceed"):
        scratch.KNNClassifier(3).fit([[0.0], [1.0]], [0, 1])
    model = scratch.KNNClassifier(1).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="feature count"):
        model.predict([[0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        model.predict([[np.nan]])


def test_synthetic_data_is_deterministic_and_labeled():
    X1, y1 = example.make_synthetic_data()
    X2, y2 = example.make_synthetic_data()
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    assert X1.shape == (400, 2)
    assert set(y1) == {0, 1}
