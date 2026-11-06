"""Deterministic teaching data and calculations, independent of plotting."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from from_scratch import UserNeighborhood, overlap_pearson
from factorization import fit_factorization


SEED = 42
USERS = tuple(f"U{i:02d}" for i in range(12))
ITEMS = ("Orbit", "Robots", "Mars", "Future", "Kitchen", "Garden", "Art", "Drama", "Travel", "Nature")


def make_data(seed=SEED):
    rng = np.random.default_rng(seed)
    templates = np.array([[5, 5, 4, 4, 1, 1, 2, 2, 3, 3],
                          [1, 2, 1, 2, 5, 5, 4, 4, 3, 3],
                          [3, 2, 4, 3, 2, 3, 1, 2, 5, 5]])
    truth = np.clip(np.repeat(templates, 4, axis=0)
                    + rng.choice([-1, 0, 1], (12, 10), p=[.15, .7, .15]), 1, 5).astype(float)
    mask = rng.random(truth.shape) < .68
    mask[:4, :5] = True
    mask[0, 5:] = False
    test_mask = ~mask & (rng.random(truth.shape) < .65)
    # Both masks are fixed before either model sees any ratings.
    train = np.where(mask, truth, np.nan)
    impressions = rng.integers(0, 16, truth.shape)
    views = rng.binomial(impressions, .65)
    clicks = rng.binomial(views, .4)
    purchases = rng.binomial(clicks, .2)
    return dict(truth=truth, train=train, train_mask=mask, test_mask=test_mask,
                frame=pd.DataFrame(train, index=USERS, columns=ITEMS),
                impressions=impressions, views=views, clicks=clicks, purchases=purchases)


def similarity_matrices(ratings, min_common=3, shrink=5.):
    values = np.asarray(ratings, dtype=float)
    observed = np.isfinite(values)
    counts = observed.sum(axis=1)
    means = np.divide(np.nansum(values, axis=1), counts,
                      out=np.zeros(len(values)), where=counts > 0)
    raw = cosine_similarity(np.nan_to_num(values, nan=0.))
    centered = cosine_similarity(np.where(observed, values - means[:, None], 0.))
    pearson = np.zeros((len(values), len(values)))
    shrunk = pearson.copy()
    for u in range(len(values)):
        for v in range(u + 1):
            pearson[u, v] = pearson[v, u] = overlap_pearson(values[u], values[v], min_common, 0)
            shrunk[u, v] = shrunk[v, u] = overlap_pearson(values[u], values[v], min_common, shrink)
    return dict(raw_cosine=raw, centered_cosine=centered, pearson=pearson, shrunk=shrunk,
                overlap=observed.astype(int) @ observed.astype(int).T)


def prediction_trace(model, user, item):
    result = model.predict(user, item)
    weights = model.similarities_[user]
    candidates = np.flatnonzero((weights > 0) & np.isfinite(model.ratings_[:, item]))
    candidates = candidates[candidates != user]
    neighbors = candidates[np.argsort(-weights[candidates], kind="stable")][:model.max_neighbors]
    rows = []
    total = float(weights[neighbors].sum())
    for v in neighbors:
        deviation = float(model.ratings_[v, item] - model.user_means_[v])
        rows.append(dict(user=int(v), rating=float(model.ratings_[v, item]),
                         mean=float(model.user_means_[v]), weight=float(weights[v]),
                         deviation=deviation, contribution=float(weights[v] * deviation / total)))
    return dict(user=user, item=item, baseline=float(model.user_means_[user]),
                weight_sum=total, rows=rows, result=result)


def implicit_targets(counts, alpha=2.):
    counts = np.asarray(counts)
    if (counts.dtype.kind not in "iuf" or not np.isfinite(counts).all()
            or np.any(counts < 0) or not np.isscalar(alpha)
            or not isinstance(alpha, (int, float, np.integer, np.floating))
            or not np.isfinite(alpha) or alpha < 0):
        raise ValueError("counts and alpha must be finite and nonnegative")
    return (counts > 0).astype(int), 1. + alpha * counts


def ranking_metrics(order, relevant, k, candidates):
    order, relevant, candidates = list(order), set(relevant), set(candidates)
    if (isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1
            or k > len(order) or len(order) != len(set(order))
            or set(order) != candidates or not relevant <= candidates or not relevant):
        raise ValueError("provide a full unique candidate ranking, nonempty relevant subset and valid k")
    gains = np.array([float(item in relevant) for item in order[:k]])
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.ones(min(k, len(relevant))) @ discounts[:min(k, len(relevant))]
    return dict(precision=float(gains.mean()), recall=float(gains.sum() / len(relevant)),
                ndcg=float(gains @ discounts / ideal), gains=gains,
                discounted_gains=gains * discounts)


def cold_start_states(train):
    """Add a synthetic user; onboarding ratings reuse U00's training observations only."""
    known = np.flatnonzero(np.isfinite(train[0]))
    matrix = np.vstack((train, np.full(train.shape[1], np.nan)))
    states = []
    for step in range(len(known) + 1):
        if step:
            i = known[step - 1]
            matrix[-1, i] = train[0, i]
        model = UserNeighborhood().fit(matrix)
        predictions = model.recommend(len(matrix) - 1, k=4)
        # A distinct count-popularity fallback, not an item-mean score.
        if step == 0:
            popularity = np.argsort(-model.item_counts_, kind="stable")[:4]
            predictions = [dict(item=int(i), score=float(model.item_counts_[i]),
                                source="popularity_count", neighbors=0) for i in popularity]
        states.append(dict(step=step, predictions=predictions,
                           positive_neighbors=int(np.sum(model.similarities_[-1] > 0)),
                           personalized=sum(p["source"] == "neighborhood" for p in predictions)))
    return states


def sparsity_sweep(data, repeats=8, seed=SEED):
    """Paired perturbations of training observations; evaluate a fixed held-out set."""
    rng = np.random.default_rng(seed)
    query = np.argwhere(data["test_mask"])
    records = []
    for keep in (1., .8, .6, .4, .2):
        scores, coverages, supports, sparsities = [], [], [], []
        for _ in range(repeats):
            mask = data["train_mask"] & (rng.random(data["train"].shape) < keep)
            train = np.where(mask, data["train"], np.nan)
            if not mask.any():
                raise ValueError("sparsity simulation removed every training rating")
            model = UserNeighborhood().fit(train)
            predictions = [model.predict(int(u), int(i)) for u, i in query]
            scores.append([p["score"] for p in predictions])
            coverages.append(np.mean([p["source"] == "neighborhood" for p in predictions]))
            supports.append(np.mean([p["neighbors"] for p in predictions]))
            sparsities.append(1 - mask.mean())
        records.append(dict(keep=keep, sparsity=float(np.mean(sparsities)),
                            coverage=float(np.mean(coverages)), neighbors=float(np.mean(supports)),
                            stability_sd=float(np.std(scores, axis=0).mean())))
    return records


def exposure_simulation(rounds=30, seed=SEED):
    """Equal click propensities; policy-dependent exposure creates unequal counts."""
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    records = {}
    for policy in ("popularity", "explore_25pct"):
        rng = np.random.default_rng(seed)
        prior = np.array([8., 2., 1., 1., 1., 1.])
        clicks, impressions = np.zeros(6, dtype=int), np.zeros(6, dtype=int)
        history = []
        for time in range(rounds + 1):
            share = (prior + clicks) / (prior + clicks).sum()
            history.append(dict(round=time, clicks=clicks.copy(), impressions=impressions.copy(),
                                share=share.copy(), hhi=float(np.sum(share ** 2))))
            if time == rounds:
                break
            probabilities = share if policy == "popularity" else .75 * share + .25 / 6
            shown = rng.choice(6, size=80, p=probabilities)
            clicked = rng.random(80) < .25  # Same intrinsic propensity for every item.
            impressions += np.bincount(shown, minlength=6)
            clicks += np.bincount(shown[clicked], minlength=6)
        records[policy] = history
    return records


def evaluation(data, factor):
    mask = data["test_mask"]
    actual = data["truth"][mask]
    predicted = factor["predictions"][mask]
    model = UserNeighborhood().fit(data["train"])
    cf = np.array([model.predict(int(u), int(i))["score"] for u, i in np.argwhere(mask)])
    errors = {name: dict(mae=float(np.mean(np.abs(values - actual))),
                        rmse=float(np.sqrt(np.mean((values - actual) ** 2))))
              for name, values in (("factorization_unclipped", predicted), ("neighborhood_clipped", cf))}
    # This is a separately labeled full-information synthetic catalog evaluation.
    user = 0
    candidates = list(np.flatnonzero(~data["train_mask"][user]))
    relevant = {i for i in candidates if data["truth"][user, i] >= 3}
    order = sorted(candidates, key=lambda i: (-factor["predictions"][user, i], i))
    metrics = ranking_metrics(order, relevant, 3, candidates)
    return dict(errors=errors, test_pairs=int(mask.sum()), user=user, candidates=candidates,
                relevant=sorted(relevant), order=order, metrics=metrics)


def build_context(fast=False):
    data = make_data()
    # Fast changes saved animation states and perturbation repeats, not fitting.
    factor = fit_factorization(data["train"], snapshots=6 if fast else 14)
    return dict(data=data, factor=factor, model=UserNeighborhood().fit(data["train"]),
                fast=fast, frames=8 if fast else 24)
