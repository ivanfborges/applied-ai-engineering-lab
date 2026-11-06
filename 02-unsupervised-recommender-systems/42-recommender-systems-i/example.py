"""Synthetic fixed-time explicit-rating experiment; no downloads or output files."""

from datetime import datetime, timezone
import json
from numbers import Real
import platform

import numpy as np

from from_scratch import UserNeighborhood


USERS = ("Ana", "Bruno", "Carla", "Davi", "Eva", "Fabi", "NewUser")
ITEMS = ("SciFi", "Robots", "Mystery", "Cooking", "Art", "Garden", "NewItem")
CUTOFF = 99


def synthetic_snapshot():
    """Hand-authored values and artificial time ticks, not actual human behavior."""
    nan = np.nan
    ratings = np.array([
        [5, 4, nan, 1, nan, nan, nan],
        [4, 5, 4, 1, nan, nan, nan],
        [5, 4, 5, 2, 2, 1, nan],
        [1, 2, 1, 5, 4, 5, nan],
        [1, 1, 2, 5, 5, 4, nan],
        [2, 1, nan, 4, 5, 5, nan],
        [nan, nan, nan, nan, nan, nan, nan],
    ])
    times = np.where(np.isnan(ratings), np.nan, 1.0)
    # All future labels are hidden together before computing any model statistics.
    future = [(0, 2, 5), (1, 5, 1), (5, 2, 2), (0, 4, 1),
              (6, 2, 4), (6, 4, 3), (0, 6, 4), (6, 6, 2)]
    for tick, (user, item, rating) in enumerate(future, start=100):
        ratings[user, item] = rating
        times[user, item] = tick
    return ratings, times


def temporal_holdout(ratings, times, cutoff):
    """Single frozen snapshot: one event per pair; future labels never enter fit."""
    ratings, times = np.asarray(ratings), np.asarray(times)
    if (ratings.ndim != 2 or ratings.size == 0 or ratings.shape != times.shape
            or ratings.dtype.kind not in "iuf" or times.dtype.kind not in "iuf"):
        raise ValueError("ratings and times must be matching nonempty real numeric matrices")
    if np.isinf(ratings).any() or np.isinf(times).any():
        raise ValueError("ratings and times must not contain infinity")
    if not np.array_equal(np.isnan(ratings), np.isnan(times)):
        raise ValueError("each observed rating must have exactly one timestamp")
    if (isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, Real)
            or not np.isfinite(cutoff)):
        raise ValueError("cutoff must be finite")
    train_mask = np.isfinite(times) & (times <= cutoff)
    future_mask = np.isfinite(times) & (times > cutoff)
    if not train_mask.any() or not future_mask.any():
        raise ValueError("the cutoff must leave both training and future observations")
    return np.where(train_mask, ratings, np.nan), np.argwhere(future_mask)


def segment(model, user, item):
    cold_user = model.user_counts_[user] == 0
    cold_item = model.item_counts_[item] == 0
    if cold_user and cold_item:
        return "both_cold"
    return "new_user" if cold_user else "new_item" if cold_item else "warm"


def run_experiment():
    ratings, times = synthetic_snapshot()
    train, pairs = temporal_holdout(ratings, times, CUTOFF)
    model = UserNeighborhood(min_common=3, shrink=5.0, max_neighbors=3).fit(train)
    predictions = []
    for u, i in pairs:
        result = model.predict(int(u), int(i))
        predictions.append({
            "user": USERS[u], "item": ITEMS[i], "segment": segment(model, u, i),
            "actual": float(ratings[u, i]), **result,
            "baseline": model.baseline(int(u), int(i))["score"],
        })
    summary = []
    for group in ("all", "warm", "new_user", "new_item", "both_cold"):
        rows = [row for row in predictions if group == "all" or row["segment"] == group]
        errors = np.array([row["score"] - row["actual"] for row in rows])
        baseline_errors = np.array([row["baseline"] - row["actual"] for row in rows])
        summary.append({
            "segment": group, "n": len(rows),
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "baseline_mae": float(np.abs(baseline_errors).mean()),
            "neighborhood_predictions": sum(row["source"] == "neighborhood" for row in rows),
        })
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "configuration": {"cutoff": CUTOFF, "min_common": 3, "shrink": 5.0, "max_neighbors": 3,
                          "rating_bounds": [1, 5], "training_ratings": int(np.isfinite(train).sum()),
                          "future_ratings": len(pairs), "randomness": "none; hand-authored data"},
        "predictions": predictions, "summary": summary,
        "ana_unseen_supported_items": [
            {**row, "item": ITEMS[row["item"]]} for row in model.recommend(0)
        ],
        "new_user_fallback_items": [
            {**row, "item": ITEMS[row["item"]]} for row in model.recommend(None)
        ],
        "interpretation_status": "author review required",
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2, allow_nan=False))
