"""Numerical contracts and valid states behind the visual explanations."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics as lab


def test_reference_query_and_unscaled_dcg_arithmetic():
    order, labels = lab.fixture()
    details = lab.dcg_details(order, labels, 5)
    assert details["metrics"] == pytest.approx({"precision": 0.6, "recall": 0.75,
        "ap": 0.6041666666666666, "rr": 1, "ndcg": 0.8251217038604399})
    assert details["dcg"] == pytest.approx(8.930676558073393)
    assert details["idcg"] == pytest.approx(10.823465818787767)
    assert details["dcg"] / details["idcg"] == pytest.approx(details["metrics"]["ndcg"])
    assert details["rows"][5]["dcg_contribution"] == 0
    assert details["rows"][5]["grade"] == 2


def test_ap_trace_adds_only_at_relevant_selected_ranks():
    order, labels = lab.fixture([1, 0, 1, 1, 0, 1])
    rows = lab.trace(order, labels, 5)
    assert [row["hits_so_far"] for row in rows[:5]] == [1, 1, 2, 3, 3]
    assert [row["ap_contribution"] for row in rows] == pytest.approx([1, 0, 2/3, 3/4, 0, 0])
    assert rows[4]["ap_numerator"] == pytest.approx(29/12)
    assert rows[4]["running_ap"] == pytest.approx(29/48)


@pytest.mark.parametrize("gain", ["linear", "exponential"])
def test_ideal_score_bounds_tail_invariance_and_cutoff_monotonicity(gain):
    order, labels = lab.fixture()
    ideal = lab.reorder(order, labels, "ideal")
    assert lab.evaluate_query(ideal, labels, 5, gain=gain)["ndcg"] == pytest.approx(1)
    assert lab.evaluate_query(order[:3]+order[3:][::-1], labels, 3, gain=gain) == lab.evaluate_query(order, labels, 3, gain=gain)
    for name, grades in lab.SCENARIOS.items():
        ranking, qrels = lab.fixture(grades)
        rows = lab.cutoff_curve(ranking, qrels, gain)
        assert all(0 <= value <= 1+1e-12 for row in rows for key, value in row.items() if key != "k")
        assert np.all(np.diff([row["recall"] for row in rows]) >= 0), name
        assert np.all(np.diff([row["ap"] for row in rows]) >= -1e-12), name


@pytest.mark.parametrize("position", range(6))
def test_rr_first_positions_and_macro(position):
    order, labels = lab.first_hit_order(position)
    expected = 1/position if position else 0
    assert lab.evaluate_query(order, labels, 5)["rr"] == expected
    result = lab.evaluate_queries({"one": order, "two": order}, {"one": labels, "two": labels}, 5)
    assert result["macro"]["mrr"] == expected


def test_short_empty_and_no_positive_policies():
    order, labels = lab.fixture()
    assert lab.evaluate_query(order[:1], labels, 10)["precision"] == 0.1
    assert set(lab.evaluate_query([], labels, 5).values()) == {0}
    assert set(lab.evaluate_query(order[1:2], labels, 1).values()) == {0}
    with pytest.raises(ValueError, match="No relevant"):
        lab.evaluate_query(["x"], {"x": 0}, 1)
    assert set(lab.evaluate_query(["x"], {"x": 0}, 1, no_relevant="zero").values()) == {0}


@pytest.mark.parametrize("ranking,labels,k", [(["x"], {"x": -1}, 1), (["x"], {"x": np.inf}, 1),
    (["x"], {"x": np.nan}, 1), (["x"], {"x": True}, 1), (["x"], {"x": 1}, True),
    (["x"], {"x": 1}, 0), (["x"], {"x": 1}, 1.5), (["x", "x"], {"x": 1}, 1),
    (["unjudged"], {"x": 1}, 1)])
def test_invalid_metric_inputs_raise(ranking, labels, k):
    with pytest.raises(ValueError):
        lab.evaluate_query(ranking, labels, k)


def test_seeded_shuffle_is_repeatable_and_scores_use_id_ties():
    order, labels = lab.fixture()
    first = lab.reorder(order, labels, "random", 44)
    assert first == lab.reorder(order[::-1], labels, "random", 44)
    assert set(first) == set(order)
    assert lab.order_scores({"b": 0.7, "a": 0.7, "c": 0.9}) == ["c", "a", "b"]
    with pytest.raises(ValueError, match="finite"):
        lab.order_scores({"a": np.nan})


@pytest.mark.parametrize("gain", ["linear", "exponential"])
def test_landscape_enumerates_only_valid_stable_permutations(gain):
    rows, labels = lab.landscape(5, gain)
    assert len(rows) == 8 * 7
    assert len({tuple(row["ranking"]) for row in rows}) == 56
    rest = list(labels)[2:]
    for row in rows:
        ranking = row["ranking"]
        assert row["high_position"] != row["medium_position"]
        assert ranking[row["high_position"] - 1] == "D01"
        assert ranking[row["medium_position"] - 1] == "D02"
        assert [item for item in ranking if item not in ("D01", "D02")] == rest
        assert row["ndcg"] == lab.evaluate_query(ranking, labels, 5, gain=gain)["ndcg"]


def test_comparison_scenarios_really_express_the_named_disagreements():
    pairs = {}
    for name in lab.COMPARISONS:
        a,b,labels,k = lab.comparison(name)
        assert set(a) == set(b) == set(labels)
        pairs[name] = lab.evaluate_query(a, labels, k), lab.evaluate_query(b, labels, k)
    a,b = pairs["Same P/R, different AP"]
    assert a["precision"] == b["precision"] and a["recall"] == b["recall"] and a["ap"] < b["ap"]
    a,b = pairs["Same RR, different recall"]
    assert a["rr"] == b["rr"] and a["recall"] < b["recall"]
    a,b = pairs["Same binary sequence, different grades"]
    assert all(a[key] == b[key] for key in ("precision", "recall", "ap", "rr")) and a["ndcg"] > b["ndcg"]
    a,b = pairs["Better first hit, worse overall order"]
    assert a["rr"] > b["rr"] and a["ap"] < b["ap"] and a["ndcg"] < b["ndcg"]
    a,b = pairs["Better recall, worse top utility"]
    assert a["recall"] > b["recall"] and a["ndcg"] < b["ndcg"]


@pytest.mark.parametrize("n", [1, 8, 20])
@pytest.mark.parametrize("budget", [180, 900, 3600])
def test_rag_keeps_exact_pool_and_respects_context_budget(n, budget):
    result = lab.rag_simulation(n=n, budget=budget)
    assert result == lab.rag_simulation(n=n, budget=budget)
    assert set(result["candidates"]) == set(result["reranked"])
    assert set(result["excluded"]).isdisjoint(result["reranked"])
    assert result["after"]["recall"] <= result["candidate_recall"]
    assert result["context_tokens"] <= budget
    assert result["effective_k"] == min(5, n, budget//180)
    assert len(result["judgments"]) == 20


def test_changes_are_named_without_calling_a_query_score_mrr():
    _, labels = lab.fixture()
    initial = lab.evaluate_query(list(labels), labels, 5)
    ideal = lab.evaluate_query(lab.reorder(list(labels), labels, "ideal"), labels, 5)
    explanation = lab.explain_change(initial, ideal)
    assert "Changed:" in explanation and "Unchanged:" in explanation
    assert "MRR" not in explanation
