"""Focused checks for the educational randomized-stump ensemble."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "from_scratch.py"
SPEC = importlib.util.spec_from_file_location("day27_from_scratch", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
RandomStumpForest = module.RandomStumpForest


def sample_data():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, size=(40, 4))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_bootstrap_features_and_oob_votes():
    X, y = sample_data()
    forest = RandomStumpForest(n_estimators=20, max_features=2, random_state=9).fit(X, y)
    assert len(forest.stumps_) == 20
    assert all(len(indices) == len(X) for indices in forest.bootstrap_indices_)
    assert any(len(np.unique(indices)) < len(X) for indices in forest.bootstrap_indices_)
    assert all(len(set(features)) == 2 for features in forest.feature_subsets_)
    assert len({tuple(sorted(features)) for features in forest.feature_subsets_}) > 1

    for row in range(len(X)):
        votes = [
            int(stump.predict(X[row : row + 1])[0])
            for stump, indices in zip(forest.stumps_, forest.bootstrap_indices_)
            if row not in indices
        ]
        assert forest.oob_counts_[row] == len(votes)
        if votes:
            assert forest.oob_predictions_[row] == int(2 * sum(votes) >= len(votes))
        else:
            assert np.isnan(forest.oob_predictions_[row])
    covered = forest.oob_counts_ > 0
    assert forest.oob_coverage_ == pytest.approx(np.mean(covered))
    assert forest.oob_accuracy_ == pytest.approx(np.mean(forest.oob_predictions_[covered] == y[covered]))


def test_prediction_vote_and_repeatability():
    X, y = sample_data()
    first = RandomStumpForest(n_estimators=15, max_features=2, random_state=5).fit(X, y)
    second = RandomStumpForest(n_estimators=15, max_features=2, random_state=5).fit(X, y)
    votes = np.stack([stump.predict(X) for stump in first.stumps_])
    expected = (2 * votes.sum(axis=0) >= len(first.stumps_)).astype(int)
    np.testing.assert_array_equal(first.predict(X), expected)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    np.testing.assert_array_equal(first.oob_predictions_, second.oob_predictions_)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_estimators": 0},
        {"n_estimators": True},
        {"max_features": 0},
        {"random_state": -1},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        RandomStumpForest(**kwargs)


def test_invalid_data_and_unfitted_prediction():
    X, y = sample_data()
    with pytest.raises(RuntimeError):
        RandomStumpForest().predict(X)
    with pytest.raises(ValueError):
        RandomStumpForest(max_features=5).fit(X, y)
    with pytest.raises(ValueError):
        RandomStumpForest().fit(X, np.zeros(len(X)))
    with pytest.raises(ValueError):
        RandomStumpForest().fit(X[:-1], y)
    with pytest.raises(ValueError):
        RandomStumpForest().fit(np.full_like(X, np.nan), y)
    with pytest.raises(ValueError):
        RandomStumpForest().fit(X, y).predict(X[:, :2])

