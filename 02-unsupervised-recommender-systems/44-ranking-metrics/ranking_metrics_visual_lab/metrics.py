"""Shared Day 44 metrics plus small, deterministic visual experiments.

The original educational NumPy evaluator remains the single source of truth.
Traces expose its arithmetic; no black-box metric library is used here.
"""

import importlib.util
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np


_spec = importlib.util.spec_from_file_location(
    "_day44_metrics_core", Path(__file__).resolve().parents[1] / "from_scratch.py"
)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)
evaluate_query = _core.evaluate_query
evaluate_queries = _core.evaluate_queries

COLORS = {0: "#9AA4B2", 1: "#E5AE38", 2: "#3987D7", 3: "#23866F"}
GRADE_NAMES = {0: "Irrelevant", 1: "Slightly relevant", 2: "Relevant", 3: "Highly relevant"}
SCENARIOS = {
    "Scattered evidence": [3, 0, 2, 1, 0, 2, 0, 1, 0, 0],
    "Relevant results early": [3, 2, 2, 1, 1, 0, 0, 0, 0, 0],
    "Relevant results late": [0, 0, 0, 0, 0, 1, 2, 1, 2, 3],
    "Few relevant documents": [0, 0, 3, 0, 0, 0, 0, 1, 0, 0],
}


def fixture(grades=(3, 0, 2, 1, 0, 2)):
    """Return stable document identities and fully judged synthetic labels."""
    labels = {f"D{i:02d}": grade for i, grade in enumerate(grades, 1)}
    return list(labels), labels


def order_scores(scores):
    """Descending finite synthetic scores; exact ties use ascending item ID."""
    if not isinstance(scores, Mapping) or not scores or any(not isinstance(item, str) or not item for item in scores):
        raise ValueError("Scores need nonempty string item IDs.")
    if not all(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) for value in scores.values()):
        raise ValueError("Scores must be finite real numbers.")
    values = np.asarray(list(scores.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Scores must be finite real numbers.")
    return sorted(scores, key=lambda item: (-scores[item], item))


def reorder(ranking, judgments, mode, seed=44):
    """Preserve the item set and judgments while choosing a new order."""
    if mode == "ideal":
        return sorted(ranking, key=lambda item: (-judgments[item], item))
    if mode == "poor":
        return sorted(ranking, key=lambda item: (judgments[item], item))
    if mode == "random":
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("Seed must be a nonnegative integer.")
        # Start from stable IDs so repeated clicks with a seed give the same order.
        return list(np.random.default_rng(seed).permutation(sorted(ranking)))
    raise ValueError("Order must be ideal, poor or random.")


def trace(ranking, judgments, k, gain="exponential"):
    """Expose every rank's binary AP and graded DCG arithmetic, including the tail."""
    evaluate_query(ranking, judgments, k, gain=gain)
    total = sum(grade > 0 for grade in judgments.values())
    rows, hits, numerator = [], 0, 0.0
    for rank, item in enumerate(ranking, 1):
        grade = float(judgments[item])
        relevant = int(grade > 0)
        hits += relevant
        precision = hits / rank
        selected = rank <= k
        ap_contribution = precision * relevant if selected else 0.0
        numerator += ap_contribution
        utility = float(np.expm1(grade * np.log(2))) if gain == "exponential" else grade
        discount = 1 / np.log2(rank + 1)
        rows.append({"rank": rank, "document": item, "grade": grade,
                     "relevant": relevant, "inside_k": selected, "hits_so_far": hits,
                     "precision_at_rank": precision, "ap_contribution": ap_contribution,
                     "ap_numerator": numerator, "running_ap": numerator / total,
                     "gain": utility, "discount": discount,
                     "dcg_contribution": utility * discount if selected else 0.0})
    return rows


def dcg_details(ranking, judgments, k, gain="exponential"):
    rows = trace(ranking, judgments, k, gain)
    ideal = reorder(list(judgments), judgments, "ideal")
    ideal_rows = trace(ideal, judgments, k, gain)
    return {"rows": rows, "ideal_rows": ideal_rows, "ideal_order": ideal,
            "dcg": sum(row["dcg_contribution"] for row in rows),
            "idcg": sum(row["dcg_contribution"] for row in ideal_rows),
            "metrics": evaluate_query(ranking, judgments, k, gain=gain)}


def cutoff_curve(ranking, judgments, gain="exponential"):
    return [{"k": k, **evaluate_query(ranking, judgments, k, gain=gain)}
            for k in range(1, len(judgments) + 1)]


def first_hit_order(position):
    """Move one relevant ID through five positions; zero means no retrieved hit."""
    ranking, labels = fixture([0, 0, 0, 0, 3])
    if isinstance(position, bool) or not isinstance(position, Integral) or position not in range(6):
        raise ValueError("First-hit position must be an integer from zero to five.")
    ranking.remove("D05")
    if position:
        ranking.insert(position - 1, "D05")
    return ranking, labels


def landscape(k=5, gain="exponential"):
    """Enumerate discrete positions for grades 3/2; keep the other IDs in order."""
    original, labels = fixture([3, 2, 1, 0, 1, 0, 0, 0])
    rest = original[2:]
    rows = []
    for high in range(1, 9):
        for medium in range(1, 9):
            if high == medium:
                continue
            remaining = iter(rest)
            ranking = ["D01" if rank == high else "D02" if rank == medium else next(remaining)
                       for rank in range(1, 9)]
            rows.append({"high_position": high, "medium_position": medium,
                         "ndcg": evaluate_query(ranking, labels, k, gain=gain)["ndcg"],
                         "ranking": ranking})
    return rows, labels


COMPARISONS = {
    "Same P/R, different AP": ("abcdef", "acdbef", 5),
    "Same RR, different recall": ("abecdf", "acdfbe", 5),
    "Same binary sequence, different grades": ("acdfbe", "dcafbe", 4),
    "Better first hit, worse overall order": ("dbeacf", "bacfde", 5),
    "Better recall, worse top utility": ("dcfabe", "abecdf", 3),
}


def comparison(name):
    a, b, reference_k = COMPARISONS[name]
    labels = dict(zip("abcdef", [3, 0, 2, 1, 0, 2]))
    return list(a), list(b), labels, reference_k


def rag_simulation(seed=44, n=8, k=5, budget=900, gain="exponential"):
    """Synthetic two-stage scores, no trained retriever or generated answer.

    Scores deliberately correlate with known grades plus noise. D01 is given
    a low retrieval score to illustrate missing evidence. Every document costs
    a hypothetical 180 tokens, so before/after comparison uses the same K.
    """
    for name, value, lower, upper in (("seed", seed, 0, 2**32 - 1),
                                      ("n", n, 1, 20), ("k", k, 1, 20),
                                      ("budget", budget, 180, 3600)):
        if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer between {lower} and {upper}.")
    rng = np.random.default_rng(seed)
    _, labels = fixture([3, 0, 2, 1, 0, 2, 0, 3, 0, 1, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0])
    retrieval = {item: float(0.25 * grade + rng.normal(0, 0.6)) for item, grade in labels.items()}
    retrieval["D01"] = -1.8
    reranking = {item: float(0.7 * grade + rng.normal(0, 0.45)) for item, grade in labels.items()}
    corpus_order = order_scores(retrieval)
    candidates = corpus_order[:n]
    reranked = order_scores({item: reranking[item] for item in candidates})
    effective_k = min(k, n, budget // 180)
    return {"judgments": labels, "retriever_scores": retrieval, "reranker_scores": reranking,
            "candidates": candidates, "reranked": reranked,
            "excluded": corpus_order[n:], "requested_k": k, "effective_k": effective_k,
            "context_tokens": effective_k * 180, "budget": budget,
            "candidate_recall": evaluate_query(candidates, labels, n)["recall"],
            "before": evaluate_query(candidates, labels, effective_k, gain=gain),
            "after": evaluate_query(reranked, labels, effective_k, gain=gain)}


def explain_change(previous, current):
    """Describe observed metric changes without attributing causality to one metric."""
    changed = [name for name in current if not np.isclose(current[name], previous[name], atol=1e-12, rtol=0)]
    fixed = [name for name in current if name not in changed]
    reasons = {
        "precision": "P@K counts hits and divides by K",
        "recall": "Recall@K divides hits by all relevant judged items",
        "ap": "AP sums precision at relevant ranks and divides by the full relevant count",
        "rr": "RR uses only the first relevant rank",
        "ndcg": "NDCG depends on gain, positions and the ideal ordering",
    }
    parts = ["Changed: " + (", ".join(changed) if changed else "none") + ".",
             "Unchanged: " + (", ".join(fixed) if fixed else "none") + "."]
    parts.extend(reasons[name] + "." for name in changed)
    if "precision" in fixed and "recall" in fixed:
        parts.append("The hit fractions are unchanged; these two metrics ignore order within K.")
    if "rr" in fixed:
        parts.append("The first-hit reward is unchanged; later hits cannot affect RR.")
    if "ap" in fixed and changed:
        parts.append("AP's precision-at-hit sum divided by the full relevant count is unchanged.")
    if "ndcg" in fixed and changed:
        parts.append("The discounted-gain to ideal-gain ratio is unchanged.")
    if not changed:
        parts.append("The operation preserved all five summaries at this cutoff; inspect document identities as well.")
    return " ".join(parts)
