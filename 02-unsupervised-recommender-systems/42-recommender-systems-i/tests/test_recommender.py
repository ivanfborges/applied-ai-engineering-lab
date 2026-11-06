"""Numerical examples, cold-start policy and frozen-time information boundaries."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scratch = load("from_scratch", "from_scratch.py")
example = load("day42_example", "example.py")
pearson = scratch.overlap_pearson
Model = scratch.UserNeighborhood


def test_pearson_agrees_with_numpy_on_overlap_and_shrinkage():
    a = np.array([5., 4., np.nan, 1., 3.])
    b = np.array([4., 5., 2., 1., np.nan])
    expected = np.corrcoef([5., 4., 1.], [4., 5., 1.])[0, 1] * 3 / 8
    assert pearson(a, b) == pytest.approx(expected)
    assert pearson(b, a) == pytest.approx(expected)
    assert pearson(a + 10, b - 5) == pytest.approx(expected)
    assert pearson(a, b, shrink=0) == pytest.approx(expected * 8 / 3)


@pytest.mark.parametrize("a,b,expected", [
    ([1, 2, 3], [3, 2, 1], -1),
    ([1, 1, 1], [1, 2, 3], 0),
    ([np.nan] * 3, [1, 2, 3], 0),
    ([1, np.nan, np.nan], [np.nan, 2, 3], 0),
    ([1, 2, np.nan], [1, 2, 3], 0),
])
def test_undefined_and_opposing_patterns(a, b, expected):
    assert pearson(a, b, shrink=0) == pytest.approx(expected)


@pytest.mark.parametrize("a,b", [([], []), ([1, 2], [1]), ([[1, 2]], [[1, 2]]),
                                         ([1, np.inf], [1, 2]), ([1j, 2j], [1, 2]),
                                         (["1", "2"], [1, 2])])
def test_invalid_vectors(a, b):
    with pytest.raises(ValueError):
        pearson(a, b)


@pytest.mark.parametrize("kwargs", [
    {"min_common": 1}, {"min_common": True}, {"min_common": 2.5},
    {"shrink": -1}, {"shrink": np.inf}, {"shrink": np.nan}, {"shrink": 1j},
    {"shrink": "5"}, {"shrink": True}, {"shrink": [5]},
    {"max_neighbors": 0}, {"max_neighbors": True},
    {"rating_bounds": [1, 1]}, {"rating_bounds": [np.nan, 5]},
])
def test_invalid_model_controls(kwargs):
    with pytest.raises(ValueError):
        Model(**kwargs)


@pytest.mark.parametrize("ratings", [[], [[np.nan]], [[0, 5]], [[1, 6]],
                                    [[1, np.inf]], [[1j, 2j]], [[True, False]]])
def test_invalid_training_matrices(ratings):
    with pytest.raises(ValueError):
        Model().fit(ratings)


def test_prediction_matches_hand_calculation_and_excludes_self():
    ratings = np.array([[1, 2, 3, np.nan], [1, 2, 3, 4], [2, 3, 4, 5.]])
    model = Model(shrink=0, max_neighbors=2).fit(ratings)
    # Target mean=2; neighbor means=2.5 and 3.5; both deviations=1.5.
    result = model.predict(0, 3)
    assert result == {"score": pytest.approx(3.5), "source": "neighborhood", "neighbors": 2}
    np.testing.assert_array_equal(np.diag(model.similarities_), 0)
    assert model.predict(0, 0)["neighbors"] == 2
    np.testing.assert_array_equal(ratings, [[1, 2, 3, np.nan], [1, 2, 3, 4], [2, 3, 4, 5]])
    ratings[1, 3] = 1
    assert model.predict(0, 3)["score"] == pytest.approx(3.5)


def test_neighbor_selection_is_item_specific_and_ties_use_row_order():
    ratings = [[1, 2, 3, np.nan], [1, 2, 3, np.nan],
               [1, 2, 3, 4], [1, 2, 3, 2.]]
    model = Model(shrink=0, max_neighbors=1).fit(ratings)
    assert model.predict(0, 3) == {
        "score": pytest.approx(3.5), "source": "neighborhood", "neighbors": 1,
    }


def test_prediction_clips_to_explicit_scale():
    model = Model(shrink=0).fit([[4, 4, 5, np.nan], [1, 1, 2, 5.]])
    assert model.predict(0, 3)["score"] == 5


def test_fallback_ladder_handles_both_cold_dimensions():
    model = Model().fit([[1, 3, np.nan], [3, 5, np.nan], [np.nan] * 3])
    assert model.predict(2, 0) == {"score": 2., "source": "item_mean", "neighbors": 0}
    assert model.predict(None, 0) == model.predict(2, 0)
    assert model.predict(0, 2) == {"score": 2., "source": "user_mean", "neighbors": 0}
    assert model.predict(None, 2) == {"score": 3., "source": "global_mean", "neighbors": 0}
    assert model.predict(0, 1)["source"] == "item_mean"  # No supported neighbor.


def test_negative_neighbors_are_not_used():
    model = Model(shrink=0).fit([[1, 2, 3, np.nan], [3, 2, 1, 5.]])
    assert model.similarities_[0, 1] < 0
    assert model.predict(0, 3) == {"score": 5., "source": "item_mean", "neighbors": 0}


def test_recommendations_exclude_seen_unsupported_and_ineligible_items():
    model = Model().fit([[5, np.nan, np.nan, np.nan], [4, 3, 3, np.nan]])
    assert [r["item"] for r in model.recommend(0, k=9)] == [1, 2]
    assert [r["item"] for r in model.recommend(None, k=9)] == [0, 1, 2]
    assert [r["item"] for r in model.recommend(0, eligible=[True, False, True, True])] == [2]
    assert model.recommend(0, eligible=[False] * 4) == []
    assert Model().fit([[1, 2, 3]]).recommend(0) == []
    with pytest.raises(ValueError):
        model.recommend(0, eligible=[1, 0, 1, 1])
    with pytest.raises(ValueError):
        model.recommend(0, k=0)


@pytest.mark.parametrize("user,item", [(-1, 0), (8, 0), (True, 0), (0, -1), (0, 1.5)])
def test_invalid_indices(user, item):
    with pytest.raises(ValueError):
        Model().fit([[1., 2.]]).predict(user, item)


def test_unfitted_model_reports_explicit_error():
    with pytest.raises(ValueError, match="fit"):
        Model().predict(0, 0)


def test_future_labels_cannot_change_training_statistics_or_predictions():
    ratings, times = example.synthetic_snapshot()
    original = ratings.copy()
    train, pairs = example.temporal_holdout(ratings, times, example.CUTOFF)
    model = Model().fit(train)
    changed = ratings.copy()
    changed[times > example.CUTOFF] = 1
    train2, pairs2 = example.temporal_holdout(changed, times, example.CUTOFF)
    other = Model().fit(train2)
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(pairs, pairs2)
    np.testing.assert_array_equal(model.user_means_, other.user_means_)
    np.testing.assert_array_equal(model.item_means_, other.item_means_)
    np.testing.assert_array_equal(model.similarities_, other.similarities_)
    for u, i in pairs:
        assert np.isnan(train[u, i])
        assert model.predict(int(u), int(i)) == other.predict(int(u), int(i))
    np.testing.assert_array_equal(ratings, original)


def test_cutoff_is_global_and_inclusive():
    train, pairs = example.temporal_holdout([[1., 2., 3.]], [[1., 2., 3.]], 2)
    np.testing.assert_array_equal(train, [[1, 2, np.nan]])
    np.testing.assert_array_equal(pairs, [[0, 2]])


@pytest.mark.parametrize("ratings,times,cutoff", [
    ([[1]], [[np.nan]], 1), ([[1, 2]], [[1]], 1),
    ([[1, 2]], [[1, np.inf]], 1), ([[1, 2]], [[1, 2]], np.nan),
    ([[1, 2]], [[1, 2]], 3), ([[1, 2]], [[1, 2]], 0),
    ([[1, 2]], [[1, 2]], "1"), ([[1, 2]], [[1, 2]], True),
])
def test_invalid_temporal_inputs(ratings, times, cutoff):
    with pytest.raises(ValueError):
        example.temporal_holdout(ratings, times, cutoff)


def test_experiment_accounting_and_determinism():
    result = example.run_experiment()
    second = example.run_experiment()
    result.pop("executed_at_utc")
    second.pop("executed_at_utc")
    assert result == second
    assert result["configuration"]["future_ratings"] == 8
    assert [r["n"] for r in result["summary"]] == [8, 4, 2, 1, 1]
    for row in result["summary"]:
        assert 0 <= row["mae"] <= row["rmse"] + 1e-12 <= 4 + 1e-12
    cold = [r for r in result["predictions"] if r["segment"] != "warm"]
    assert all(r["source"] != "neighborhood" and r["neighbors"] == 0 for r in cold)
    assert all(r["score"] == r["baseline"] for r in cold)
