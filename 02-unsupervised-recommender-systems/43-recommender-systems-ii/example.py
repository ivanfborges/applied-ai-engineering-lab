"""Synthetic, globally time-split retrieval/ranking comparison. No downloads."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from from_scratch import (
    binary_interactions, candidate_union, finite_array, fit_bpr, positive_integer,
    rank_items, single_positive_metrics, standardize_eligible,
)


def synthetic_history(seed=43, n_users=80, n_items=100, n_genres=8, rounds=16):
    """One distinct item per user per global round; metadata exists at round 0."""
    for value, name in ((n_users, "n_users"), (n_items, "n_items"),
                        (n_genres, "n_genres"), (rounds, "rounds")):
        positive_integer(value, name)
    if not 3 <= rounds < n_items:
        raise ValueError("require 3 <= rounds < n_items")
    rng = np.random.default_rng(seed)
    genres = rng.integers(n_genres, size=n_items)
    favorites = rng.integers(n_genres, size=n_users)
    sequences = []
    for favorite in favorites:
        weights = np.where(genres == favorite, 6.0, 1.0)
        sequences.append(rng.choice(n_items, rounds, replace=False, p=weights / weights.sum()))
    return np.asarray(sequences), np.eye(n_genres)[genres]


def history_matrix(sequences, n_items, stop):
    sequences = np.asarray(sequences)
    n_items = positive_integer(n_items, "n_items")
    stop = positive_integer(stop, "stop")
    if (sequences.ndim != 2 or sequences.size == 0 or sequences.dtype.kind not in "iu"
            or stop > sequences.shape[1] or np.any(sequences < 0)
            or np.any(sequences >= n_items)):
        raise ValueError("sequences/stop must describe valid item histories")
    if any(len(np.unique(row)) != len(row) for row in sequences):
        raise ValueError("this example requires distinct interactions per user")
    matrix = np.zeros((len(sequences), n_items))
    matrix[np.arange(len(sequences))[:, None], sequences[:, :stop]] = 1
    return matrix


def content_scores(interactions, features):
    matrix = binary_interactions(interactions)
    features = finite_array(features, "features", 2)
    if len(features) != matrix.shape[1]:
        raise ValueError("one feature row is required per item")
    # Only past interactions enter the profile; empty histories yield zero cosine.
    profiles = matrix @ features / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    return cosine_similarity(profiles, features)


def fit_scores(matrix, features, factors, epochs, seed):
    svd = TruncatedSVD(n_components=factors, n_iter=7, random_state=seed)
    user_svd = svd.fit_transform(matrix)
    p, q = fit_bpr(matrix, factors=factors, epochs=epochs, seed=seed)
    return {
        "Popularity": np.broadcast_to(matrix.sum(axis=0), matrix.shape).copy(),
        "SVD": user_svd @ svd.components_,
        "BPR": p @ q.T,
        "Content": content_scores(matrix, features),
    }


def evaluate(scores, seen, targets, k):
    return single_positive_metrics(
        [rank_items(row, mask, k) for row, mask in zip(scores, seen)], targets, k)


def select_weight(collaborative, content, seen, validation_targets, k):
    cf_z = standardize_eligible(collaborative, seen)
    content_z = standardize_eligible(content, seen)
    rows = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        metrics = evaluate(weight * cf_z + (1 - weight) * content_z,
                           seen, validation_targets, k)
        rows.append({"collaborative_weight": weight, **metrics})
    # Stable ties favor the first (smaller collaborative) weight in the grid.
    best = max(rows, key=lambda row: row["ndcg"])
    return best["collaborative_weight"], rows


def cold_item_probe(matrix, features, collaborative, per_source=20):
    """Append one metadata-only item for user 0; this is not a relevance test."""
    u = 0
    favorite_feature = int(np.argmax(matrix[u] @ features))
    new_features = np.vstack([features, np.eye(features.shape[1])[favorite_feature]])
    extended = np.pad(matrix, ((0, 0), (0, 1)))
    # An appended ID has no fitted embedding: use an explicit zero-score policy.
    cf = np.append(collaborative[u], 0.0)
    content = content_scores(extended, new_features)[u]
    new_id = matrix.shape[1]
    seen = extended[u].astype(bool)
    content_candidates = rank_items(content, seen, per_source)
    union = candidate_union(cf, content, seen, per_source)
    return {"user": u, "new_item": new_id, "training_interactions": 0,
            "collaborative_fallback_score": float(cf[new_id]),
            "content_score": float(content[new_id]),
            "in_content_candidates": bool(new_id in content_candidates),
            "in_union": bool(new_id in union), "per_source": per_source}


def run_experiment(seed=43, factors=12, epochs=60, k=10):
    factors = positive_integer(factors, "factors")
    epochs = positive_integer(epochs, "epochs")
    k = positive_integer(k, "k")
    if factors >= 80 or k > 85:
        raise ValueError("require factors < 80 and k <= 85 for this fixed data setup")
    sequences, features = synthetic_history(seed)
    train = history_matrix(sequences, len(features), stop=14)
    validation = sequences[:, 14]
    scores_validation = fit_scores(train, features, factors, epochs, seed)
    weight, selection = select_weight(scores_validation["BPR"], scores_validation["Content"],
                                      train.astype(bool), validation, k)
    # Refit with the validation round only after selecting the weight. Test stays hidden.
    development = history_matrix(sequences, len(features), stop=15)
    targets = sequences[:, 15]
    scores = fit_scores(development, features, factors, epochs, seed)
    seen = development.astype(bool)
    hybrid = (weight * standardize_eligible(scores["BPR"], seen)
              + (1 - weight) * standardize_eligible(scores["Content"], seen))
    metrics = {name: evaluate(values, seen, targets, k) for name, values in scores.items()}
    metrics["Hybrid full catalog"] = evaluate(hybrid, seen, targets, k)
    retrieval = []
    for per_source in (5, 20):
        candidates = [candidate_union(cf, content, mask, per_source)
                      for cf, content, mask in zip(scores["BPR"], scores["Content"], seen)]
        rankings = [rank_items(row, mask, k, pool)
                    for row, mask, pool in zip(hybrid, seen, candidates)]
        retrieval.append({"per_source": per_source,
                          "mean_candidates": float(np.mean([len(pool) for pool in candidates])),
                          "candidate_recall": float(np.mean([t in pool for t, pool in zip(targets, candidates)])),
                          **single_positive_metrics(rankings, targets, k)})
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "A candidate budget can discard held-out positives before hybrid ranking; content may complement collaborative scores.",
        "configuration": {"seed": seed, "users": 80, "items": 100, "genres": 8,
                          "rounds": 16, "train_rounds": [0, 13], "validation_round": 14,
                          "test_round": 15, "factors": factors, "epochs": epochs,
                          "lr": 0.05, "reg": 0.01, "k": k, "svd_n_iter": 7,
                          "favorite_genre_weight": 6, "other_genre_weight": 1,
                          "weight_grid": [0, 0.25, 0.5, 0.75, 1],
                          "selection_metric": "validation NDCG@K",
                          "refit": "rounds 0..14, same fixed model hyperparameters"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "platform": platform.platform()},
        "validation": selection, "selected_collaborative_weight": weight,
        "test": metrics, "retrieval": retrieval,
        "cold_item_probe": cold_item_probe(development, features, scores["BPR"]),
        "interpretation_candidate": "Inspect full-catalog versus candidate-limited results and the content-only baseline before attributing value to fusion.",
        "interpretation_status": "author review required",
        "limitations": "One synthetic seed; genre-driven positives favor content. One unseen target per user, no impressions or online outcomes. Exhaustive dense retrieval is not a scalability benchmark. Cold-item probe has no held-out relevance label.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--factors", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--save", action="store_true", help="save a dated ignored JSON beside this script")
    args = parser.parse_args()
    result = run_experiment(args.seed, args.factors, args.epochs, args.k)
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
