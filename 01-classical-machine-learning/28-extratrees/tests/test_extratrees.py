"""Mechanism, numerical, and validation checks for Day 28."""

import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

TOPIC = Path(__file__).resolve().parents[1]


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scratch = load_module("from_scratch.py", "day28_scratch")
example = load_module("example.py", "day28_example")
ExtraTreesBinary = scratch.ExtraTreesBinary


def test_gini_known_values():
    assert scratch.gini([0, 0, 0]) == 0.0
    assert scratch.gini([0, 1]) == 0.5
    assert scratch.gini([0, 0, 0, 1]) == 0.375


class FixedProposals:
    """Prescribe two candidates whose split qualities are known by hand."""

    def choice(self, n_features, size, replace):
        assert (n_features, size, replace) == (2, 2, False)
        return np.array([0, 1])

    def random(self):
        return 0.5


def test_selects_best_random_candidate():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model = ExtraTreesBinary(max_depth=1, max_features=2)
    root = model._grow(X, y, depth=0, k=2, rng=FixedProposals())
    # Feature 0 yields two pure leaves; feature 1 leaves both children mixed.
    assert root.feature == 0
    assert root.threshold == 0.5
    assert root.left.probability == 0
    assert root.right.probability == 1


def test_zero_gain_split_can_recover_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    shallow = ExtraTreesBinary(n_estimators=5, max_depth=1, max_features=2).fit(X, y)
    deep = ExtraTreesBinary(n_estimators=5, max_depth=2, max_features=2).fit(X, y)
    np.testing.assert_allclose(shallow.predict_proba(X)[:, 1], 0.5)
    np.testing.assert_array_equal(deep.predict(X), y)
    assert len({root.threshold for root in deep.roots_}) > 1


def test_probability_mean_is_not_hard_majority_vote():
    model = ExtraTreesBinary(n_estimators=3)
    model.n_features_in_ = 1
    model.roots_ = [scratch.Node(p, 10) for p in [0.49, 0.49, 0.99]]
    np.testing.assert_allclose(model.predict_proba([[0]]), [[1 - 1.97 / 3, 1.97 / 3]])
    assert model.predict([[0]]).tolist() == [1]


def test_leaf_frequencies_and_tie_policy():
    X = [[0], [1], [2], [3]]
    model = ExtraTreesBinary(n_estimators=3, max_depth=0).fit(X, [0, 0, 0, 1])
    np.testing.assert_allclose(model.predict_proba(X), np.tile([0.75, 0.25], (4, 1)))
    tied = ExtraTreesBinary(max_depth=0).fit([[0], [1]], [0, 1])
    assert tied.predict([[2]]).tolist() == [0]
    pure = ExtraTreesBinary().fit(X, [1, 1, 1, 1])
    np.testing.assert_allclose(pure.predict_proba(X)[:, 1], 1)


def test_tree_constraints_and_leaf_probabilities():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = ExtraTreesBinary(n_estimators=8, max_depth=4, min_samples_leaf=4).fit(X, y)

    def inspect(node, rows, depth):
        assert node.n_samples == len(rows)
        assert node.probability == pytest.approx(y[rows].mean())
        assert depth <= 4
        if node.feature is None:
            assert len(rows) >= 4
            return
        values = X[rows, node.feature]
        assert values.min() <= node.threshold < values.max()
        left = values <= node.threshold
        inspect(node.left, rows[left], depth + 1)
        inspect(node.right, rows[~left], depth + 1)

    for root in model.roots_:
        # Every root sees all rows, without bootstrap duplication or omission.
        inspect(root, np.arange(len(X)), 0)


def test_constant_features_and_invalid_proposals_stop():
    model = ExtraTreesBinary(n_estimators=4).fit(np.ones((6, 2)), [0, 1, 0, 1, 0, 1])
    assert all(root.feature is None for root in model.roots_)
    np.testing.assert_allclose(model.predict_proba([[1, 1]]), [[0.5, 0.5]])
    # Both random thresholds are valid numerically, but one child is too small.
    X = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 1]], dtype=float)
    model = ExtraTreesBinary(min_samples_leaf=2, max_features=2)
    root = model._grow(X, np.array([0, 1, 0, 1, 1]), 0, 2, FixedProposals())
    assert root.feature is None


def test_seeded_fit_and_refit_are_reproducible():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    first = ExtraTreesBinary(n_estimators=6, random_state=12).fit(X, y)
    second = ExtraTreesBinary(n_estimators=6, random_state=12).fit(X, y)
    assert first.roots_ == second.roots_
    before = first.tree_probabilities(X)
    first.fit(X, y)
    np.testing.assert_array_equal(first.tree_probabilities(X), before)
    assert first.roots_ == second.roots_
    np.testing.assert_allclose(first.predict_proba(X).sum(axis=1), 1)


@pytest.mark.parametrize("kwargs", [
    {"n_estimators": 0}, {"n_estimators": True}, {"max_depth": -1},
    {"min_samples_leaf": 0}, {"max_features": 0}, {"max_features": 1.5},
    {"random_state": -1}, {"random_state": None},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ExtraTreesBinary(**kwargs)


@pytest.mark.parametrize("X,y", [
    ([], []), ([0, 1], [0, 1]), ([[np.nan], [1]], [0, 1]),
    ([[0], [np.inf]], [0, 1]), ([[0], [1]], [0]),
    ([[0], [1]], [[0], [1]]), ([[0], [1]], [0, 2]),
    ([[0], [1]], [0, np.nan]),
])
def test_invalid_training_data(X, y):
    with pytest.raises(ValueError):
        ExtraTreesBinary().fit(X, y)


def test_invalid_prediction_and_feature_limit():
    with pytest.raises(RuntimeError):
        ExtraTreesBinary().predict([[0]])
    with pytest.raises(ValueError):
        ExtraTreesBinary(max_features=2).fit([[0], [1]], [0, 1])
    fitted = ExtraTreesBinary().fit([[0], [1]], [0, 1])
    for X in [[], [[0, 1]], [[np.inf]]]:
        with pytest.raises(ValueError):
            fitted.predict(X)


def test_disagreement_matches_direct_pair_enumeration():
    predictions = np.array([[0, 0, 1, 1], [0, 1, 1, 0], [1, 0, 1, 0]])
    direct = np.mean([np.mean(a != b) for a, b in combinations(predictions, 2)])
    assert example.pairwise_disagreement(predictions) == pytest.approx(direct)
    assert example.pairwise_disagreement(predictions) == pytest.approx(
        example.pairwise_disagreement(1 - predictions[::-1])
    )
    assert example.pairwise_disagreement(np.zeros((3, 4))) == 0
    assert example.pairwise_disagreement([[0, 1], [1, 0]]) == 1


@pytest.mark.parametrize("predictions", [
    [[0, 1]], np.empty((2, 0)), [0, 1], [[0, 2], [1, 0]],
    [[0, np.nan], [0, 1]],
])
def test_invalid_disagreement_inputs(predictions):
    with pytest.raises(ValueError):
        example.pairwise_disagreement(predictions)


def test_library_comparison_controls_and_seeded_predictions():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 8, dtype=float)
    y = np.array([0, 1, 1, 0] * 8)
    first = example.build_models(n_estimators=5)
    second = example.build_models(n_estimators=5)
    assert len(first) == 4
    for name, model in first.items():
        for key, value in example.MODEL_CONFIG.items():
            if key != "n_estimators":
                assert model.get_params()[key] == value
        assert model.bootstrap == name.endswith("_true")
        assert model.estimator.splitter == ("random" if name.startswith("ET") else "best")
        model.fit(X, y)
        second[name].fit(X, y)
        np.testing.assert_array_equal(model.predict_proba(X), second[name].predict_proba(X))
