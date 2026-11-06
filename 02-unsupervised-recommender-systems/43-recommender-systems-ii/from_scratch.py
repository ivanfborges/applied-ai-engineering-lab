"""Educational BPR matrix factorization and small-catalog ranking helpers."""

from __future__ import annotations

import numpy as np


def finite_array(values, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "biuf" or raw.ndim != ndim or raw.size == 0:
        raise ValueError(f"{name} must be a nonempty real {ndim}D array")
    result = raw.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def binary_interactions(values) -> np.ndarray:
    matrix = finite_array(values, "interactions", 2)
    if not np.isin(matrix, [0, 1]).all():
        raise ValueError("interactions must be binary; zero means unobserved")
    return matrix


def bpr_triplet_loss_gradient(user, positive, negative, reg: float = 0.01):
    """Return softplus(-margin) + reg/2 * squared norms and its gradients."""
    p = finite_array(user, "user", 1)
    qi = finite_array(positive, "positive", 1)
    qj = finite_array(negative, "negative", 1)
    if p.shape != qi.shape or p.shape != qj.shape:
        raise ValueError("all three embeddings must have the same shape")
    if not np.isscalar(reg) or not np.isreal(reg) or not np.isfinite(reg) or reg < 0:
        raise ValueError("reg must be finite and nonnegative")
    margin = float(p @ (qi - qj))
    # exp(-logaddexp(0, margin)) is sigmoid(-margin) without overflow.
    weight = np.exp(-np.logaddexp(0.0, margin))
    loss = np.logaddexp(0.0, -margin) + 0.5 * reg * (p @ p + qi @ qi + qj @ qj)
    return float(loss), (-weight * (qi - qj) + reg * p,
                         -weight * p + reg * qi, weight * p + reg * qj)


def fit_bpr(interactions, factors=12, epochs=60, lr=0.05, reg=0.01, seed=43):
    """Visit every positive once per epoch; draw one uniform unseen item per visit.

    Per-triplet regularization is frequency-weighted by this sampling schedule.
    Empty-history users get a zero embedding; saturated users are rejected.
    This dense scalar-SGD implementation is for teaching, not serving at scale.
    """
    matrix = binary_interactions(interactions)
    factors = positive_integer(factors, "factors")
    epochs = positive_integer(epochs, "epochs")
    for value, name, allow_zero in ((lr, "lr", False), (reg, "reg", True)):
        if (not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value)
                or value < 0 or (not allow_zero and value == 0)):
            raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    positives = np.argwhere(matrix == 1)
    if len(positives) == 0:
        raise ValueError("at least one observed positive is required")
    if np.any(matrix.sum(axis=1) == matrix.shape[1]):
        raise ValueError("each user must have an unobserved item for sampling")
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.1, (matrix.shape[0], factors))
    q = rng.normal(0, 0.1, (matrix.shape[1], factors))
    pools = [np.flatnonzero(row == 0) for row in matrix]
    for _ in range(epochs):
        for index in rng.permutation(len(positives)):
            u, i = positives[index]
            j = rng.choice(pools[u])
            # All three gradients use the old vectors (simultaneous SGD).
            old_p, old_i, old_j = p[u].copy(), q[i].copy(), q[j].copy()
            margin = old_p @ (old_i - old_j)
            weight = np.exp(-np.logaddexp(0.0, margin))
            p[u] -= lr * (-weight * (old_i - old_j) + reg * old_p)
            q[i] -= lr * (-weight * old_p + reg * old_i)
            q[j] -= lr * (weight * old_p + reg * old_j)
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("BPR diverged; reduce lr or rescale the configuration")
    p[matrix.sum(axis=1) == 0] = 0
    return p, q


def rank_items(scores, seen, k, candidates=None) -> np.ndarray:
    """Rank eligible item IDs by descending score, then ascending ID for ties."""
    values = finite_array(scores, "scores", 1)
    mask = np.asarray(seen)
    k = positive_integer(k, "k")
    if mask.dtype != bool or mask.shape != values.shape:
        raise ValueError("seen must be a boolean array matching scores")
    if candidates is None:
        ids = np.arange(len(values))
    else:
        ids = np.asarray(candidates)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or np.any(ids >= len(values))
                or len(np.unique(ids)) != len(ids)):
            raise ValueError("candidates must be unique valid integer item IDs")
    ids = ids[~mask[ids]]
    return ids[np.lexsort((ids, -values[ids]))][:k]


def standardize_eligible(scores, seen) -> np.ndarray:
    """Normalize per user over the complete eligible catalog; constants map to 0."""
    values = finite_array(scores, "scores", 2)
    mask = np.asarray(seen)
    if mask.dtype != bool or mask.shape != values.shape:
        raise ValueError("seen must be a boolean matrix matching scores")
    result = np.zeros_like(values)
    for u, row in enumerate(values):
        eligible = ~mask[u]
        if not eligible.any():
            raise ValueError("each user must have at least one eligible item")
        scale = row[eligible].std()
        if scale > 1e-12:
            result[u, eligible] = (row[eligible] - row[eligible].mean()) / scale
    return result


def candidate_union(collaborative, content, seen, per_source):
    return np.union1d(rank_items(collaborative, seen, per_source),
                      rank_items(content, seen, per_source))


def single_positive_metrics(rankings, targets, k):
    """Macro Recall@K (= hit rate) and binary NDCG@K for one target per user."""
    k = positive_integer(k, "k")
    targets = np.asarray(targets)
    if (targets.ndim != 1 or targets.dtype.kind not in "iu" or len(targets) == 0
            or np.any(targets < 0) or len(rankings) != len(targets)):
        raise ValueError("targets must contain one nonnegative integer per ranking")
    hits, discounts = [], []
    for ranking, target in zip(rankings, targets):
        ids = np.asarray(ranking)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0)
                or len(np.unique(ids)) != len(ids)):
            raise ValueError("rankings must contain unique nonnegative integer IDs")
        positions = np.flatnonzero(ids[:k] == target)
        hits.append(float(len(positions) > 0))
        discounts.append(float(1 / np.log2(positions[0] + 2)) if len(positions) else 0.0)
    return {"recall": float(np.mean(hits)), "ndcg": float(np.mean(discounts))}
