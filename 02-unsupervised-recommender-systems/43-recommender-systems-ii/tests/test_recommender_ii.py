"""Numerical, eligibility and temporal-boundary checks for Day 43."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import example
from from_scratch import (
    bpr_triplet_loss_gradient, candidate_union, fit_bpr, rank_items,
    single_positive_metrics, standardize_eligible,
)


def test_triplet_gradient_matches_finite_differences():
    vectors = np.array([[0.3, -0.2], [0.8, 0.4], [-0.5, 0.1]])
    _, gradients = bpr_triplet_loss_gradient(*vectors, reg=0.07)
    numerical = np.zeros_like(vectors)
    for index in np.ndindex(vectors.shape):
        plus, minus = vectors.copy(), vectors.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (bpr_triplet_loss_gradient(*plus, reg=0.07)[0]
                            - bpr_triplet_loss_gradient(*minus, reg=0.07)[0]) / 2e-6
    np.testing.assert_allclose(gradients, numerical, atol=1e-9)


@pytest.mark.parametrize("margin", [-1000.0, 1000.0])
def test_extreme_triplet_margin_is_stable(margin):
    loss, gradients = bpr_triplet_loss_gradient([1.0], [margin], [0.0], reg=0)
    assert np.isfinite(loss)
    assert np.isfinite(gradients).all()


def test_one_sgd_step_uses_simultaneous_old_vectors_and_unseen_negative():
    rng = np.random.default_rng(9)
    p = rng.normal(0, 0.1, (1, 2))
    q = rng.normal(0, 0.1, (2, 2))
    _, gradients = bpr_triplet_loss_gradient(p[0], q[0], q[1], reg=0.03)
    expected_p = p[0] - 0.04 * gradients[0]
    expected_q = q - 0.04 * np.array(gradients[1:])
    actual_p, actual_q = fit_bpr([[1, 0]], factors=2, epochs=1, lr=0.04, reg=0.03, seed=9)
    np.testing.assert_allclose(actual_p[0], expected_p, atol=1e-14)
    np.testing.assert_allclose(actual_q, expected_q, atol=1e-14)


def test_fixed_training_triplets_improve_and_seed_repeats():
    matrix = np.array([[1, 0, 0], [0, 1, 0]])
    first = fit_bpr(matrix, factors=3, epochs=1, seed=43)
    final = fit_bpr(matrix, factors=3, epochs=100, seed=43)
    repeat = fit_bpr(matrix, factors=3, epochs=100, seed=43)
    def mean_loss(factors):
        p, q = factors
        return np.mean([bpr_triplet_loss_gradient(p[u], q[i], q[j], reg=0)[0]
                        for u, i in np.argwhere(matrix == 1)
                        for j in np.flatnonzero(matrix[u] == 0)])
    assert mean_loss(final) < mean_loss(first)
    for actual, expected in zip(final, repeat):
        np.testing.assert_array_equal(actual, expected)


def test_empty_history_user_has_zero_scores():
    p, q = fit_bpr([[1, 0, 0], [0, 0, 0]], epochs=2)
    np.testing.assert_array_equal(p[1] @ q.T, np.zeros(3))


@pytest.mark.parametrize("matrix", [[], [[0, 0]], [[1, 1]], [[1, -1]],
                                    [[1, 0.2]], [[1, np.nan]], [[1, np.inf]], [[1j, 0]]])
def test_invalid_training_matrices_are_rejected(matrix):
    with pytest.raises(ValueError):
        fit_bpr(matrix)


@pytest.mark.parametrize("kwargs", [{"factors": 0}, {"factors": 1.5}, {"epochs": True},
                                    {"epochs": 0}, {"lr": 0}, {"lr": np.nan},
                                    {"reg": -1}, {"reg": np.inf}])
def test_invalid_hyperparameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_bpr([[1, 0]], **kwargs)


def test_seen_filter_ties_and_candidate_subset():
    scores = [100, 2, 2, 3]
    seen = np.array([True, False, False, False])
    np.testing.assert_array_equal(rank_items(scores, seen, 10), [3, 1, 2])
    np.testing.assert_array_equal(rank_items(scores, seen, 2, [2, 0, 1]), [1, 2])
    assert len(rank_items(scores, seen, 2, np.array([], dtype=int))) == 0


@pytest.mark.parametrize("pool", [[1, 1], [-1], [4], [1.1]])
def test_invalid_candidate_ids_are_rejected(pool):
    with pytest.raises(ValueError):
        rank_items([1, 2, 3, 4], np.zeros(4, dtype=bool), 2, pool)


def test_normalization_ignores_seen_and_is_stable_across_candidate_budgets():
    seen = np.array([[True, False, False, False], [False, True, False, False]])
    scores = np.array([[100, 1, 2, 3], [7, -100, 7, 7]])
    z = standardize_eligible(scores, seen)
    assert z[0, ~seen[0]].mean() == pytest.approx(0)
    assert z[0, ~seen[0]].std() == pytest.approx(1)
    np.testing.assert_array_equal(z[1], 0)
    altered = scores.copy()
    altered[seen] = -999
    np.testing.assert_array_equal(standardize_eligible(altered, seen), z)


def test_one_positive_metrics_have_known_positions_and_full_denominator():
    result = single_positive_metrics([np.array([2, 1]), np.array([2, 3]),
                                      np.array([], dtype=int)], [1, 9, 7], 2)
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["ndcg"] == pytest.approx(1 / (3 * np.log2(3)))


def test_duplicate_ranking_is_invalid():
    with pytest.raises(ValueError):
        single_positive_metrics([[1, 1]], [1], 2)


def test_content_only_uses_observed_profile_and_handles_empty_history():
    features = np.eye(2)[[0, 1, 0]]
    values = example.content_scores([[1, 0, 0], [0, 0, 0]], features)
    np.testing.assert_allclose(values, [[1, 0, 1], [0, 0, 0]])


def test_content_source_can_admit_metadata_only_item():
    matrix = np.array([[1, 0]])
    features = np.eye(2)
    probe = example.cold_item_probe(matrix, features, [[0.8, 0.5]], per_source=1)
    assert probe["collaborative_fallback_score"] == 0
    assert probe["content_score"] == pytest.approx(1)
    assert probe["in_content_candidates"] and probe["in_union"]
    np.testing.assert_array_equal(candidate_union([0.8, 0.5, 0], [1, 0, 1],
                                                 np.array([True, False, False]), 1), [1, 2])


def test_temporal_matrices_exclude_future_targets():
    sequences, features = example.synthetic_history()
    repeated, _ = example.synthetic_history()
    np.testing.assert_array_equal(sequences, repeated)
    train = example.history_matrix(sequences, len(features), 14)
    refit = example.history_matrix(sequences, len(features), 15)
    users = np.arange(80)
    assert (train[users, sequences[:, 14:16].T] == 0).all()
    assert (refit[users, sequences[:, 14]] == 1).all()
    assert (refit[users, sequences[:, 15]] == 0).all()
    np.testing.assert_array_equal(train.sum(axis=1), 14)
    np.testing.assert_array_equal(refit.sum(axis=1), 15)


def test_validation_weight_selection_uses_ndcg_and_stable_ties():
    seen = np.array([[True, False, False]])
    weight, rows = example.select_weight([[100, 0, 1]], [[100, 1, 0]], seen, [1], 1)
    assert weight == 0
    assert rows[0]["ndcg"] == 1
    assert rows[-1]["ndcg"] == 0


def test_end_to_end_refits_after_selection_and_respects_retrieval_ceiling(monkeypatch):
    original = example.fit_scores
    fitted = []
    def record(matrix, *args):
        fitted.append(matrix.copy())
        return original(matrix, *args)
    monkeypatch.setattr(example, "fit_scores", record)
    result = example.run_experiment(factors=3, epochs=2)
    assert len(fitted) == 2
    assert fitted[0].sum() == 80 * 14
    assert fitted[1].sum() == 80 * 15
    assert len(result["validation"]) == 5
    assert result["interpretation_status"] == "author review required"
    for row in result["retrieval"]:
        assert 0 <= row["ndcg"] <= row["recall"] <= row["candidate_recall"] <= 1
        assert row["mean_candidates"] <= 2 * row["per_source"]
    assert result["retrieval"][0]["candidate_recall"] <= result["retrieval"][1]["candidate_recall"]
    json.dumps(result, allow_nan=False)
