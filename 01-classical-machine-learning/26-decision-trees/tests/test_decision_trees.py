"""Focused numerical and behavioral checks for the Day 26 tree."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scratch = load_module("day26_scratch", "from_scratch.py")
example = load_module("day26_example", "example.py")


def test_impurity_extremes_and_manual_weighted_gain():
    assert scratch.gini([0, 0, 0]) == 0
    assert scratch.entropy([0, 0, 0]) == 0
    assert scratch.gini([0, 1]) == pytest.approx(0.5)
    assert scratch.entropy([0, 1]) == pytest.approx(1.0)
    parent = scratch.gini([1] * 6 + [0] * 4)
    weighted = (6 * scratch.gini([1] * 5 + [0]) + 4 * scratch.gini([1, 0, 0, 0])) / 10
    assert parent == pytest.approx(0.48)
    assert weighted == pytest.approx(19 / 60)
    assert parent - weighted == pytest.approx(49 / 300)


def test_split_threshold_gain_and_leaf_constraint():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    assert scratch.best_gini_split(X, y) == pytest.approx((0, 1.5, 0.5))
    assert scratch.best_gini_split(X, y, min_samples_leaf=3) is None
    assert scratch.best_gini_split(X, np.zeros(4, dtype=int)) is None


def test_tree_predictions_and_depth_boundary():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = scratch.GiniTreeClassifier(max_depth=1).fit(X, y)
    np.testing.assert_array_equal(model.predict([[0.5], [2.5]]), [0, 1])
    expected = DecisionTreeClassifier(max_depth=1, random_state=26).fit(X, y).predict([[0.5], [2.5]])
    np.testing.assert_array_equal(model.predict([[0.5], [2.5]]), expected)
    assert model.root_.left.feature is None
    assert model.root_.right.feature is None


def test_majority_tie_and_invalid_inputs():
    model = scratch.GiniTreeClassifier(max_depth=1, min_samples_leaf=2).fit(
        [[0.0], [1.0]], ["z", "a"]
    )
    assert model.predict([[0.2]]).tolist() == ["a"]
    with pytest.raises(ValueError, match="fit"):
        scratch.GiniTreeClassifier().predict([[0.0]])
    with pytest.raises(ValueError, match="finite"):
        model.predict([[np.nan]])
    with pytest.raises(ValueError, match="feature count"):
        model.predict([[0.0, 1.0]])
    with pytest.raises(ValueError, match="one label per row"):
        model.fit([[0.0], [1.0]], [0])
    with pytest.raises(ValueError, match="positive integer"):
        scratch.GiniTreeClassifier(max_depth=0)
    with pytest.raises(ValueError, match="positive integer"):
        scratch.best_gini_split([[0.0]], [0], min_samples_leaf=0)
    with pytest.raises(ValueError, match="nonempty"):
        scratch.gini([])
    with pytest.raises(ValueError, match="finite labels"):
        scratch.GiniTreeClassifier().fit([[0.0], [1.0]], [0.0, np.nan])


def test_synthetic_generator_and_pruning_selection():
    X, y = example.make_synthetic_data()
    X_again, y_again = example.make_synthetic_data()
    np.testing.assert_array_equal(X, X_again)
    np.testing.assert_array_equal(y, y_again)
    assert X.shape == (600, 2)
    assert set(np.unique(y)) == {0, 1}
    with pytest.raises(ValueError, match="at least 100"):
        example.make_synthetic_data(50)
    alpha, score, count = example.select_pruning_alpha(X[:200], y[:200], X[200:250], y[200:250])
    assert alpha >= 0
    assert 0 <= score <= 1
    assert count >= 1
