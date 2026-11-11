"""Arithmetic, library agreement and evaluation-boundary tests for Day 44."""

from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, ndcg_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import example
from from_scratch import evaluate_queries, evaluate_query


JUDGMENTS = dict(zip("abcdef", [3, 0, 2, 1, 0, 2]))


def test_hand_calculated_query():
    actual = evaluate_query(list("abcdef"), JUDGMENTS, 5)
    dcg = 7 + 3 / 2 + 1 / np.log2(5)
    idcg = 7 + 3 / np.log2(3) + 3 / 2 + 1 / np.log2(5)
    assert actual == pytest.approx({"precision": 3 / 5, "recall": 3 / 4,
                                    "ap": (1 + 2 / 3 + 3 / 4) / 4,
                                    "rr": 1, "ndcg": dcg / idcg})


def test_macro_map_and_mrr_are_per_query_means():
    judgments = {"a": {"x": 1}, "b": {"x": 0, "y": 1, "z": 1}}
    result = evaluate_queries({"a": ["x"], "b": ["x", "y"]}, judgments, 2)
    assert result["macro"] == pytest.approx({"precision": 0.5, "recall": 0.75,
                                             "map": 0.625, "mrr": 0.75,
                                             "ndcg": (1 + (1 / np.log2(3)) /
                                                      (1 + 1 / np.log2(3))) / 2})


def test_short_and_empty_rankings_use_fixed_k_and_full_universe():
    short = evaluate_query(["a"], JUDGMENTS, 10)
    assert short["precision"] == 0.1
    assert short["recall"] == short["ap"] == 0.25
    assert short["ndcg"] < 1
    assert set(evaluate_query([], JUDGMENTS, 5).values()) == {0.0}


def test_no_hit_within_cutoff_and_missing_positive():
    assert set(evaluate_query(list("be"), JUDGMENTS, 2).values()) == {0.0}
    assert evaluate_query(list("bea"), JUDGMENTS, 2)["rr"] == 0


def test_ap_denominator_conventions_differ_when_r_exceeds_k():
    assert evaluate_query(list("ac"), JUDGMENTS, 2)["ap"] == 0.5
    assert evaluate_query(list("ac"), JUDGMENTS, 2, ap_denominator="capped")["ap"] == 1


def test_no_relevant_requires_explicit_policy_and_zero_queries_remain_in_mean():
    with pytest.raises(ValueError, match="No relevant"):
        evaluate_query(["x"], {"x": 0}, 1)
    result = evaluate_queries({"a": ["x"], "b": []},
                              {"a": {"x": 1}, "b": {"x": 0}}, 1, no_relevant="zero")
    assert set(result["macro"].values()) == {0.5}


@pytest.mark.parametrize("gain", ["linear", "exponential"])
@pytest.mark.parametrize("k", [1, 3, 6, 8])
def test_ndcg_matches_sklearn_with_aligned_gains_and_distinct_scores(gain, k):
    rng = np.random.default_rng(44)
    for _ in range(12):
        grades = rng.integers(0, 4, 6)
        grades[0] = 3
        judgments = dict(zip("abcdef", grades))
        order = list(rng.permutation(list(judgments)))
        labels = np.array([[judgments[item] for item in order]], dtype=float)
        compatible = labels if gain == "linear" else np.exp2(labels) - 1
        expected = ndcg_score(compatible, [[6, 5, 4, 3, 2, 1]], k=k, ignore_ties=True)
        assert evaluate_query(order, judgments, k, gain=gain)["ndcg"] == pytest.approx(expected)


def test_full_list_ap_matches_classifier_ap_only_under_aligned_protocol():
    actual = evaluate_query(list("abcdef"), JUDGMENTS, 6)["ap"]
    labels = np.array(list(JUDGMENTS.values())) > 0
    assert actual == pytest.approx(average_precision_score(labels, [6, 5, 4, 3, 2, 1]))


def test_bounds_monotonic_recall_and_total_ap_over_all_small_permutations():
    judgments = dict(zip("abcd", [3, 1, 0, 2]))
    for ranking in permutations(judgments):
        rows = [evaluate_query(ranking, judgments, k) for k in range(1, 7)]
        assert all(-1e-14 <= value <= 1 + 1e-14 for row in rows for value in row.values())
        for metric in ("recall", "ap", "rr"):
            assert np.all(np.diff([row[metric] for row in rows]) >= -1e-14)
    assert evaluate_query(list("adbc"), judgments, 3)["ndcg"] == pytest.approx(1)


def test_tail_permutation_preserves_denominators_and_top_k_metrics():
    assert evaluate_query(list("abcdef"), JUDGMENTS, 3) == evaluate_query(list("abcfed"), JUDGMENTS, 3)


def test_input_order_is_explicit_tie_policy():
    judgments = {"a": 3, "b": 0}
    assert evaluate_query(["a", "b"], judgments, 1)["rr"] == 1
    assert evaluate_query(["b", "a"], judgments, 1)["rr"] == 0


def test_candidate_only_normalization_cannot_establish_universe_recall():
    labels = {"a": 3, "b": 0, "c": 1}
    overall = evaluate_query(["c", "b"], labels, 2)
    conditional = evaluate_query(["c", "b"], {"b": 0, "c": 1}, 2)
    assert overall["recall"] == 0.5
    assert overall["ndcg"] < conditional["ndcg"] == 1


@pytest.mark.parametrize("k", [0, -1, 1.5, True, np.bool_(True), "5", None])
def test_invalid_cutoffs(k):
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_query(["a"], JUDGMENTS, k)


@pytest.mark.parametrize("grade", [-1, float("nan"), float("inf"), "1", True, 1j, None])
def test_invalid_grades(grade):
    with pytest.raises(ValueError, match="Grades"):
        evaluate_query(["a"], {"a": grade}, 1)


@pytest.mark.parametrize("ranking,judgments", [
    (["a", "a"], {"a": 1}), (["a", "b", "b"], {"a": 1, "b": 0}),
    (["unknown"], {"a": 1}), (["a", "unknown"], {"a": 1}),
    ("a", {"a": 1}), ([1], {"a": 1}), ([[]], {"a": 1}),
    ([""], {"": 1}), ([], {}), ([], {1: 1}), ([], [1]),
])
def test_invalid_ids_duplicates_and_judgment_containers(ranking, judgments):
    with pytest.raises(ValueError):
        evaluate_query(ranking, judgments, 1)


@pytest.mark.parametrize("options", [{"gain": "other"}, {"ap_denominator": "hits"},
                                         {"no_relevant": "skip"}])
def test_invalid_conventions(options):
    with pytest.raises(ValueError):
        evaluate_query(["a"], JUDGMENTS, 1, **options)


@pytest.mark.parametrize("rankings,judgments", [({}, {}), ({"q": []}, {}),
    ({"q": []}, {"q": {"a": 1}, "missing": {"a": 1}}),
    ({"": []}, {"": {"a": 1}})])
def test_query_population_cannot_be_silently_dropped(rankings, judgments):
    with pytest.raises(ValueError):
        evaluate_queries(rankings, judgments, 1)


def test_gain_overflow_is_explicit_and_large_linear_grades_remain_finite():
    with pytest.raises(ValueError, match="overflow"):
        evaluate_query(["a"], {"a": 2000}, 1)
    assert evaluate_query(["a", "b"], {"a": 1e308, "b": 1e308}, 2,
                          gain="linear")["ndcg"] == pytest.approx(1)
    assert evaluate_query(["a"], {"a": 1e-20}, np.int64(1))["ndcg"] == 1


def test_experiment_preserves_top_k_membership_and_labels_oracle():
    result = example.run_experiment()
    before, after = result["baseline"]["macro"], result["reordered"]["macro"]
    assert before["precision"] == after["precision"] == 0.6
    assert before["recall"] == after["recall"] == 0.75
    assert after["map"] > before["map"]
    assert after["mrr"] > before["mrr"]
    probe = result["candidate_probe"]
    assert probe["candidate_recall"] == probe["end_to_end"]["recall"] == 0.5
    assert probe["conditional_on_candidates"]["ndcg"] == 1
    assert result["interpretation_status"] == "author review required"
