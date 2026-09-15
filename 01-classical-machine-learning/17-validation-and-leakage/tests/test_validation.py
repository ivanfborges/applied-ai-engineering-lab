"""Test partition invariants and preprocessing isolation, not metric snapshots."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import kfold_indices
from example import classification_pipeline, group_demo, temporal_folds


@pytest.mark.parametrize("n,k", [(11, 3), (10, 5), (2, 2), (7, 7)])
def test_partition_invariants(n, k):
    folds = list(kfold_indices(n, k))
    assert len(folds) == k
    np.testing.assert_array_equal(
        np.sort(np.concatenate([valid for _, valid in folds])), np.arange(n)
    )
    sizes = [len(valid) for _, valid in folds]
    assert max(sizes) - min(sizes) <= 1
    for train, valid in folds:
        assert not np.intersect1d(train, valid).size
        np.testing.assert_array_equal(np.sort(np.r_[train, valid]), np.arange(n))


def test_unshuffled_matches_sklearn():
    for actual, expected in zip(
        kfold_indices(11, 3, shuffle=False), KFold(3).split(np.zeros(11))
    ):
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a, e)


def test_seed_is_deterministic_and_local():
    np.random.seed(9)
    expected = np.random.random(3)
    np.random.seed(9)
    first = list(kfold_indices(20, seed=12))
    np.testing.assert_array_equal(np.random.random(3), expected)
    for a, b in zip(first, kfold_indices(20, seed=12)):
        np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(first[0][1], list(kfold_indices(20, seed=13))[0][1])


@pytest.mark.parametrize("kwargs", [
    {"n_samples": 1}, {"n_samples": 10, "n_splits": 1},
    {"n_samples": 3, "n_splits": 4}, {"n_samples": True},
    {"n_samples": 10.5}, {"n_samples": 10, "n_splits": 2.5},
    {"n_samples": 10, "shuffle": "yes"}, {"n_samples": 10, "seed": -1},
    {"n_samples": 10, "seed": True},
])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        list(kfold_indices(**kwargs))


def test_temporal_label_availability():
    previous = set()
    for train, valid in temporal_folds():
        assert set(train) >= previous
        assert train.max() + 2 < valid.min()
        assert valid.max() < 30
        assert not np.intersect1d(train, valid).size
        previous = set(train)


def test_group_boundary():
    result = group_demo()
    assert result["group_overlapping_groups"] == [0] * 5
    assert all(count > 0 for count in result["random_overlapping_groups"])


def test_scaler_fits_only_training_rows():
    X = np.arange(80, dtype=float).reshape(40, 2)
    y = np.tile([0, 1], 20)
    for train, valid in StratifiedKFold(4).split(X, y):
        fitted = classification_pipeline().fit(X[train], y[train])
        np.testing.assert_allclose(
            fitted.named_steps["scale"].mean_, X[train].mean(axis=0)
        )
        altered = X.copy()
        altered[valid] += 1e6
        refitted = clone(fitted).fit(altered[train], y[train])
        np.testing.assert_allclose(
            fitted.named_steps["model"].coef_, refitted.named_steps["model"].coef_
        )
