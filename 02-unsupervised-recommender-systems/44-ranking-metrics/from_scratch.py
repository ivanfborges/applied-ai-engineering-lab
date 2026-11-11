"""Educational top-K metrics over explicit item IDs and complete judgments."""

from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np


def evaluate_query(
    ranking: Sequence[str],
    judgments: Mapping[str, float],
    k: int,
    *,
    gain: str = "exponential",
    ap_denominator: str = "total",
    no_relevant: str = "error",
) -> dict[str, float]:
    """Evaluate an ordered, possibly incomplete ranking against a judged universe.

    Binary relevance is grade > 0. Missing ranks contribute zero; precision
    always divides by k. Recall and ideal DCG use all judgments, including
    items absent from ranking. Unknown or repeated IDs are errors.

    AP divides by R (total) or min(R, k) (capped). Zero-positive queries raise
    by default; no_relevant='zero' explicitly assigns all five metrics zero.
    Input order resolves score ties upstream; scores are not accepted here.
    """
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer.")
    if gain not in ("linear", "exponential"):
        raise ValueError("gain must be 'linear' or 'exponential'.")
    if ap_denominator not in ("total", "capped"):
        raise ValueError("ap_denominator must be 'total' or 'capped'.")
    if no_relevant not in ("error", "zero"):
        raise ValueError("no_relevant must be 'error' or 'zero'.")
    if not isinstance(judgments, Mapping) or not judgments:
        raise ValueError("judgments must be a nonempty mapping of item IDs to grades.")
    if any(not isinstance(item, str) or not item for item in judgments):
        raise ValueError("Judged item IDs must be nonempty strings.")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
           for value in judgments.values()):
        raise ValueError("Grades must be real numbers, excluding booleans.")
    try:
        grades = np.asarray(list(judgments.values()), dtype=float)
    except (ValueError, OverflowError) as error:
        raise ValueError("Grades must be representable as finite floats.") from error
    if not np.isfinite(grades).all() or (grades < 0).any():
        raise ValueError("Grades must be finite and nonnegative.")
    if isinstance(ranking, (str, bytes)) or not isinstance(ranking, Sequence):
        raise ValueError("ranking must be a sequence of item IDs.")
    if any(not isinstance(item, str) or not item for item in ranking):
        raise ValueError("Ranked item IDs must be nonempty strings.")
    if len(set(ranking)) != len(ranking):
        raise ValueError("Duplicate item IDs are not allowed, even below k.")
    if any(item not in judgments for item in ranking):
        raise ValueError("Every ranked item must have a relevance judgment.")

    total_relevant = int(np.count_nonzero(grades > 0))
    if total_relevant == 0:
        if no_relevant == "error":
            raise ValueError("No relevant items: choose an explicit no-answer policy.")
        return dict.fromkeys(("precision", "recall", "ap", "rr", "ndcg"), 0.0)

    with np.errstate(over="ignore", invalid="ignore"):
        gains = np.expm1(grades * np.log(2.0)) if gain == "exponential" else grades
    if not np.isfinite(gains).all():
        raise ValueError("Grades overflow the chosen gain function.")
    # A common scale cancels in NDCG and prevents overflow in discounted sums.
    gains = gains / gains.max()
    gain_by_id = dict(zip(judgments, gains))
    top = list(ranking[:k])
    relevant = np.array([judgments[item] > 0 for item in top], dtype=float)
    ranks = np.arange(1, len(top) + 1, dtype=float)
    hits = float(relevant.sum())
    numerator = float(np.sum(np.cumsum(relevant) / ranks * relevant))
    denominator = total_relevant if ap_denominator == "total" else min(total_relevant, k)
    first_hits = np.flatnonzero(relevant)
    dcg = float(np.sum([gain_by_id[item] for item in top] / np.log2(ranks + 1)))
    ideal = np.sort(gains)[::-1][:k]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return {
        "precision": hits / k,
        "recall": hits / total_relevant,
        "ap": numerator / denominator,
        "rr": 1.0 / (int(first_hits[0]) + 1) if first_hits.size else 0.0,
        "ndcg": dcg / idcg,
    }


def evaluate_queries(
    rankings: Mapping[str, Sequence[str]],
    judgments: Mapping[str, Mapping[str, float]],
    k: int,
    *,
    gain: str = "exponential",
    ap_denominator: str = "total",
    no_relevant: str = "error",
) -> dict:
    """Return per-query scores and equal-query macro means (AP -> MAP, RR -> MRR).

    Query sets must match exactly: an empty result list is explicit, rather
    than silently dropping a failed query from the aggregate.
    """
    if (not isinstance(rankings, Mapping) or not isinstance(judgments, Mapping)
            or not rankings or set(rankings) != set(judgments)):
        raise ValueError("Nonempty rankings and judgments must have identical query IDs.")
    if any(not isinstance(query, str) or not query for query in rankings):
        raise ValueError("Query IDs must be nonempty strings.")
    rows = {
        query: evaluate_query(ranking, judgments[query], k, gain=gain,
                              ap_denominator=ap_denominator, no_relevant=no_relevant)
        for query, ranking in rankings.items()
    }
    names = {"precision": "precision", "recall": "recall", "ap": "map",
             "rr": "mrr", "ndcg": "ndcg"}
    macro = {name: float(np.mean([row[key] for row in rows.values()]))
             for key, name in names.items()}
    return {"per_query": rows, "macro": macro}
