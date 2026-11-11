"""Deterministic synthetic ranking evaluation; no fitted retrieval system."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.metrics import ndcg_score

from from_scratch import evaluate_queries, evaluate_query


def run_experiment() -> dict:
    """Compare order, cutoff, gain and candidate-universe choices on fixed labels."""
    judgments = {
        "policy": dict(zip("abcdef", [3, 0, 2, 1, 0, 2])),
        "returns": dict(zip("abcdef", [0, 0, 2, 1, 3, 1])),
    }
    rankings = {query: list("abcdef") for query in judgments}
    baseline = evaluate_queries(rankings, judgments, 5)
    reordered = {"policy": list("acdbef"), "returns": list("ecdabf")}

    # The reranker can only reorder its four candidates; f and a are absent.
    candidates = list("bcde")
    oracle_order = sorted(candidates, key=lambda item: -judgments["policy"][item])
    conditional_labels = {item: judgments["policy"][item] for item in candidates}

    library_check = {}
    labels = np.array([list(judgments["policy"].values())], dtype=float)
    scores = np.array([[6, 5, 4, 3, 2, 1]], dtype=float)
    for gain in ("linear", "exponential"):
        compatible_labels = labels if gain == "linear" else np.exp2(labels) - 1
        library_check[gain] = {
            "educational": evaluate_query(rankings["policy"], judgments["policy"], 5,
                                          gain=gain)["ndcg"],
            "sklearn": float(ndcg_score(compatible_labels, scores, k=5, ignore_ties=True)),
        }
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Order-sensitive metrics can change at fixed P@K/R@K; candidate-only normalization can conceal missing evidence.",
        "configuration": {"data": "two hand-constructed synthetic queries, six fully judged IDs each",
                          "judgments": judgments, "rankings": rankings, "reordered": reordered,
                          "k": 5, "binary_relevance": "grade > 0", "gain": "exponential",
                          "ap_denominator": "total relevant items", "aggregation": "equal-query macro",
                          "no_relevant": "error", "short_lists": "missing ranks contribute zero",
                          "randomness": "none; no seed needed"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "platform": platform.platform()},
        "baseline": baseline,
        "reordered": evaluate_queries(reordered, judgments, 5),
        "cutoff_sweep_policy": {str(k): evaluate_query(rankings["policy"], judgments["policy"], k)
                                for k in range(1, 7)},
        "ap_conventions_at_2": {mode: evaluate_query(list("ac"), judgments["policy"], 2,
                                                   ap_denominator=mode)["ap"]
                                for mode in ("total", "capped")},
        "candidate_probe": {
            "candidates": candidates, "oracle_order": oracle_order, "final_k": 3,
            "candidate_recall": evaluate_query(candidates, judgments["policy"], 4)["recall"],
            "end_to_end": evaluate_query(oracle_order, judgments["policy"], 3),
            "conditional_on_candidates": evaluate_query(oracle_order, conditional_labels, 3),
            "oracle_warning": "Uses labels to order a fixed pool: diagnostic upper bound, not a learned reranker.",
        },
        "ndcg_library_check": library_check,
        "interpretation_candidate": "Inspect the unchanged P@5/R@5 versus MAP/MRR/NDCG changes and the candidate-conditioned NDCG=1 alongside incomplete universe recall.",
        "interpretation_status": "author review required",
        "limitations": "Two constructed queries, complete judgments, no uncertainty estimate, real retriever, traffic, latency or downstream answer measurement. Oracle ordering is label-informed by design.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="save a unique dated ignored JSON record")
    args = parser.parse_args()
    result = run_experiment()
    serialized = json.dumps(result, indent=2, allow_nan=False)
    print(serialized)
    if args.save:
        output = Path(__file__).resolve().parent / "outputs"
        output.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = output / f"run_{stamp}.json"
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
        print(f"Saved ignored run record: {path}")


if __name__ == "__main__":
    main()
