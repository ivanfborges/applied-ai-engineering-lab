"""Reproducible numerical states for the Day 43 visual laboratory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))
from example import content_scores, history_matrix, synthetic_history
from from_scratch import (
    binary_interactions, candidate_union, fit_bpr, positive_integer, rank_items,
    single_positive_metrics, standardize_eligible,
)


def trace_bpr(matrix, factors=3, epochs=30, seed=43, frames=24, lr=0.05, reg=0.01):
    """Instrument the study's SGD schedule; snapshots contain actual old/new vectors."""
    matrix = binary_interactions(matrix)
    for value, name in ((factors, "factors"), (epochs, "epochs"), (frames, "frames")):
        positive_integer(value, name)
    if not np.isfinite([lr, reg]).all() or lr <= 0 or reg < 0:
        raise ValueError("lr must be positive and reg nonnegative, both finite")
    positives = np.argwhere(matrix == 1)
    if not len(positives) or np.any(matrix.sum(axis=1) == matrix.shape[1]):
        raise ValueError("require positives and an unseen sampling pool per user")
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.1, (matrix.shape[0], factors))
    q = rng.normal(0, 0.1, (matrix.shape[1], factors))
    pools = [np.flatnonzero(row == 0) for row in matrix]
    u0, i0 = positives[0]
    j0 = pools[u0][0]
    records, history, states = [], [], []
    keep = set(np.linspace(0, epochs * len(positives) - 1, frames, dtype=int))

    def checkpoint(epoch):
        scores = p @ q.T
        losses = []
        for u, row in enumerate(matrix):
            pos = np.flatnonzero(row == 1)
            if len(pos):
                margins = scores[u, pos, None] - scores[u, pools[u]][None, :]
                losses.extend(np.logaddexp(0, -margins).ravel())
        history.append({"epoch": epoch, "pair_loss": float(np.mean(losses)),
                        "positive_score": float(scores[u0, i0]),
                        "unobserved_score": float(scores[u0, j0])})
        states.append({"epoch": epoch, "p": p.copy(), "q": q.copy()})

    checkpoint(0)
    step = 0
    for epoch in range(1, epochs + 1):
        for index in rng.permutation(len(positives)):
            u, i = positives[index]
            j = int(rng.choice(pools[u]))
            old = np.array([p[u], q[i], q[j]])
            margin = old[0] @ (old[1] - old[2])
            g = np.exp(-np.logaddexp(0.0, margin))
            p[u] -= lr * (-g * (old[1] - old[2]) + reg * old[0])
            q[i] -= lr * (-g * old[0] + reg * old[1])
            q[j] -= lr * (g * old[0] + reg * old[2])
            if step in keep:
                new = np.array([p[u], q[i], q[j]])
                before_scores = [float(old[0] @ old[1]), float(old[0] @ old[2])]
                after_scores = [float(new[0] @ new[1]), float(new[0] @ new[2])]
                records.append({"step": step + 1, "epoch": epoch, "user": int(u),
                                "positive": int(i), "unobserved": j, "before": old,
                                "after": new, "before_scores": before_scores,
                                "after_scores": after_scores,
                                "pair_loss": float(np.logaddexp(0, -(after_scores[0] - after_scores[1])))})
            step += 1
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("nonfinite BPR training state")
        checkpoint(epoch)
    p[matrix.sum(axis=1) == 0] = 0
    return {"p": p, "q": q, "steps": records, "history": history,
            "states": states, "tracked_pair": [int(u0), int(i0), int(j0)]}


def shared_projection(p, q):
    """Use one shared PCA basis when d>3; return exact coordinates for d=3."""
    p, q = np.asarray(p), np.asarray(q)
    if p.ndim != 2 or q.ndim != 2 or p.shape[1] != q.shape[1] or p.shape[1] < 2:
        raise ValueError("user and item factors must share at least two dimensions")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("factors must be finite")
    if p.shape[1] > 3:
        combined = np.vstack([p, q])
        pca = PCA(n_components=3, svd_solver="full").fit(combined)
        coordinates = pca.transform(combined)
        return coordinates[:len(p)], coordinates[len(p):], {
            "mode": "Shared PCA (centered; displayed dot products are not original scores)",
            "variance_retained": float(pca.explained_variance_ratio_.sum()),
            "mean": pca.mean_, "components": pca.components_}
    pad = ((0, 0), (0, 3 - p.shape[1]))
    return np.pad(p, pad), np.pad(q, pad), {"mode": "Original factors (zero-padded if d=2)",
                                           "variance_retained": 1.0}


@dataclass
class LabData:
    seed: int
    factors: int
    epochs: int
    sequences: np.ndarray
    features: np.ndarray
    matrix: np.ndarray
    targets: np.ndarray
    trace: dict
    svd_p: np.ndarray
    svd_q: np.ndarray
    scores: dict
    cf_z: np.ndarray
    content_z: np.ndarray

    @property
    def seen(self):
        return self.matrix.astype(bool)

    def hybrid(self, alpha=0.5):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        return alpha * self.cf_z + (1 - alpha) * self.content_z


def build_lab(seed=43, factors=3, epochs=30, frames=24):
    positive_integer(factors, "factors")
    if not 2 <= factors < 80:
        raise ValueError("require 2 <= factors < 80")
    sequences, features = synthetic_history(seed)
    matrix = history_matrix(sequences, len(features), 15)
    targets = sequences[:, 15].copy()
    trace = trace_bpr(matrix, factors, epochs, seed, frames)
    svd = TruncatedSVD(n_components=factors, n_iter=7, random_state=seed)
    svd_p = svd.fit_transform(matrix)
    svd_q = svd.components_.T
    scores = {"Popularity": np.broadcast_to(matrix.sum(axis=0), matrix.shape).copy(),
              "SVD": svd_p @ svd_q.T, "BPR": trace["p"] @ trace["q"].T,
              "Content": content_scores(matrix, features)}
    lab = LabData(seed, factors, epochs, sequences, features, matrix, targets,
                  trace, svd_p, svd_q, scores,
                  standardize_eligible(scores["BPR"], matrix.astype(bool)),
                  standardize_eligible(scores["Content"], matrix.astype(bool)))
    if not all(np.isfinite(value).all() for value in scores.values()):
        raise ValueError("nonfinite model scores")
    return lab


def model_metrics(lab, ks=(1, 3, 5, 10, 20), alpha=0.5):
    result = {}
    for name, scores in {**lab.scores, "Hybrid": lab.hybrid(alpha)}.items():
        rankings = [rank_items(row, mask, max(ks)) for row, mask in zip(scores, lab.seen)]
        result[name] = {str(k): single_positive_metrics(rankings, lab.targets, k) for k in ks}
    return result


def candidate_analysis(lab, budgets=(5, 10, 20, 50), k=10, alpha=0.5):
    rows = []
    hybrid = lab.hybrid(alpha)
    full = [rank_items(row, seen, k) for row, seen in zip(hybrid, lab.seen)]
    for budget in budgets:
        pools = [candidate_union(cf, content, seen, budget)
                 for cf, content, seen in zip(lab.scores["BPR"], lab.scores["Content"], lab.seen)]
        ranked = [rank_items(row, seen, k, pool) for row, seen, pool in zip(hybrid, lab.seen, pools)]
        rows.append({"budget": budget, "pools": pools, "rankings": ranked,
                     "full_rankings": full,
                     "candidate_recall": float(np.mean([t in pool for t, pool in zip(lab.targets, pools)])),
                     "mean_candidates": float(np.mean([len(pool) for pool in pools])),
                     "metrics": single_positive_metrics(ranked, lab.targets, k),
                     "full_metrics": single_positive_metrics(full, lab.targets, k),
                     "excluded_users": [u for u, (t, pool) in enumerate(zip(lab.targets, pools)) if t not in pool]})
    return rows


def objective_losses(logits, labels):
    """Toy judged labels: BCE, all positive-negative pairs, top-one ListNet CE."""
    scores, labels = np.asarray(logits, float), np.asarray(labels, float)
    if (scores.ndim != 1 or scores.shape != labels.shape or not len(scores)
            or not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all()
            or not (labels == 1).any() or not (labels == 0).any()):
        raise ValueError("require finite logits and both positive and negative binary judgments")
    point = float(np.mean(np.logaddexp(0, scores) - labels * scores))
    margins = scores[labels == 1, None] - scores[labels == 0][None, :]
    pair = float(np.logaddexp(0, -margins).mean())
    log_probability = scores - np.logaddexp.reduce(scores)
    target_probability = np.exp(labels - np.logaddexp.reduce(labels))
    return {"pointwise": point, "pairwise": pair,
            "listwise": float(-target_probability @ log_probability)}


def ranking_contributions(relevance):
    relevance = np.asarray(relevance, float)
    if relevance.ndim != 1 or not len(relevance) or not np.isfinite(relevance).all() or np.any(relevance < 0):
        raise ValueError("require a nonempty nonnegative finite relevance vector")
    discounts = 1 / np.log2(np.arange(len(relevance)) + 2)
    dcg = np.cumsum((2 ** relevance - 1) * discounts)
    ideal = np.cumsum((2 ** np.sort(relevance)[::-1] - 1) * discounts)
    return {"discounts": discounts, "dcg": dcg,
            "ndcg": np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)}


def cold_states(lab, user=0, steps=(0, 1, 3, 6, 15)):
    """Fold in a new user's vector against frozen items; append a metadata-only item."""
    positive_integer(user + 1, "user+1")
    if user >= len(lab.matrix):
        raise ValueError("user out of range")
    # Remove this user's history from item training: a genuine new-user probe.
    other_users = lab.matrix.copy()
    other_users[user] = 0
    _, q = fit_bpr(other_users, factors=lab.factors, epochs=lab.epochs, seed=lab.seed)
    # Fixed metadata, independent of this user's later onboarding interactions.
    new_genre = 0
    features = np.vstack([lab.features, np.eye(lab.features.shape[1])[new_genre]])
    q = np.vstack([q, np.zeros(lab.factors)])
    rows = []
    for count in steps:
        if not 0 <= count <= 15:
            raise ValueError("history counts must be between 0 and 15")
        history = lab.sequences[user, :count]
        observed = np.zeros((1, len(features)))
        observed[0, history] = 1
        p = np.zeros(lab.factors)
        rng = np.random.default_rng(lab.seed)
        pool = np.flatnonzero(observed[0] == 0)
        for _ in range(30 if count else 0):
            for i in rng.permutation(history):
                j = int(rng.choice(pool))
                g = np.exp(-np.logaddexp(0, p @ (q[i] - q[j])))
                # Fold-in updates only the new user's factors; item factors stay frozen.
                p -= 0.05 * (-g * (q[i] - q[j]) + 0.01 * p)
        cf = (p @ q.T)[None, :]
        content = content_scores(observed, features)
        seen = observed.astype(bool)
        fused = 0.5 * standardize_eligible(cf, seen) + 0.5 * standardize_eligible(content, seen)
        union = candidate_union(cf[0], content[0], seen[0], 20)
        top = rank_items(fused[0], seen[0], 10, union)
        rows.append({"count": count, "cf": cf[0], "content": content[0],
                     "hybrid": fused[0], "top": top, "pool": union,
                     "seen": seen[0], "new_item": len(features) - 1,
                     "new_item_in_pool": bool(len(features) - 1 in union),
                     "content_candidates": rank_items(content[0], seen[0], 20),
                     "profile": observed @ features})
    return rows


def exposure_simulation(seed=43, rounds=20, n_items=12, impressions=120):
    """Two declared policies share stationary hidden CTRs and start with equal counts."""
    positive_integer(rounds, "rounds")
    positive_integer(n_items, "n_items")
    positive_integer(impressions, "impressions")
    if n_items < 2:
        raise ValueError("need at least two items")
    rng = np.random.default_rng(seed)
    true_ctr = rng.uniform(0.05, 0.35, n_items)
    clicks = np.ones((2, n_items), dtype=int)
    exposure = np.full((2, n_items), 10, dtype=int)
    states = []
    for time in range(rounds + 1):
        states.append({"round": time, "clicks": clicks.copy(), "exposure": exposure.copy(),
                       "true_ctr": true_ctr.copy(), "observed_ctr": clicks / exposure})
        if time == rounds:
            break
        for policy in range(2):
            probability = np.ones(n_items) / n_items if policy == 0 else (
                0.1 / n_items + 0.9 * clicks[policy] ** 2 / np.sum(clicks[policy] ** 2))
            new_exposure = rng.multinomial(impressions, probability)
            exposure[policy] += new_exposure
            clicks[policy] += rng.binomial(new_exposure, true_ctr)
    return states
