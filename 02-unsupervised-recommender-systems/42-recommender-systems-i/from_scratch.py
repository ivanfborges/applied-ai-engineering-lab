"""Educational explicit-rating user neighborhoods; dense arrays, not a serving library."""

from numbers import Real

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _real_array(values, ndim, name):
    array = np.asarray(values)
    if array.ndim != ndim or array.size == 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty {ndim}D real numeric array")
    array = array.astype(float, copy=True)
    if np.isinf(array).any():
        raise ValueError(f"{name} may contain NaN for missing entries, but not infinity")
    return array


def overlap_pearson(a, b, min_common=3, shrink=5.0):
    """Pearson on co-rated entries, multiplied by n / (n + shrink).

    Undefined/insufficient-support correlations return zero, not NaN.
    Negative correlations are retained here; prediction uses positive ones only.
    """
    a = _real_array(a, 1, "a")
    b = _real_array(b, 1, "b")
    if a.shape != b.shape:
        raise ValueError("rating vectors must have matching shapes")
    _positive_integer(min_common, "min_common")
    if min_common < 2:
        raise ValueError("min_common must be at least 2")
    if (isinstance(shrink, (bool, np.bool_)) or not isinstance(shrink, Real)
            or not np.isfinite(shrink) or shrink < 0):
        raise ValueError("shrink must be finite and nonnegative")
    common = ~np.isnan(a) & ~np.isnan(b)
    count = int(common.sum())
    if count < min_common:
        return 0.0
    x = a[common] - a[common].mean()
    y = b[common] - b[common].mean()
    # Normalize separately to avoid multiplying two large vector norms.
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 0.0
    correlation = np.clip(np.dot(x / nx, y / ny), -1.0, 1.0)
    return float(correlation * count / (count + shrink))


class UserNeighborhood:
    """Positive-neighbor deviations with explicit, training-only fallback reasons.

    Rows/columns are fixed user/item IDs; NaN means unobserved. Use user=None
    for an unknown user. An all-NaN column represents a known catalog item
    with no training ratings. Scores are clipped to the supplied rating scale.
    """

    def __init__(self, min_common=3, shrink=5.0, max_neighbors=3, rating_bounds=(1.0, 5.0)):
        # Validate similarity controls even if fit later receives only one user.
        overlap_pearson([1., 2.], [1., 2.], min_common, shrink)
        _positive_integer(max_neighbors, "max_neighbors")
        bounds = _real_array(rating_bounds, 1, "rating_bounds")
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("rating_bounds must contain two finite increasing values")
        self.min_common = min_common
        self.shrink = shrink
        self.max_neighbors = max_neighbors
        self.rating_bounds = tuple(bounds)

    def fit(self, ratings):
        ratings = _real_array(ratings, 2, "ratings")
        observed = ~np.isnan(ratings)
        if not observed.any():
            raise ValueError("at least one training rating is required")
        lo, hi = self.rating_bounds
        if np.any((ratings[observed] < lo) | (ratings[observed] > hi)):
            raise ValueError("observed ratings must lie within rating_bounds")
        self.ratings_ = ratings
        self.user_counts_ = observed.sum(axis=1)
        self.item_counts_ = observed.sum(axis=0)
        self.global_mean_ = float(ratings[observed].mean())
        self.user_means_ = np.divide(
            np.nansum(ratings, axis=1), self.user_counts_,
            out=np.full(ratings.shape[0], np.nan), where=self.user_counts_ > 0,
        )
        self.item_means_ = np.divide(
            np.nansum(ratings, axis=0), self.item_counts_,
            out=np.full(ratings.shape[1], np.nan), where=self.item_counts_ > 0,
        )
        self.similarities_ = np.zeros((len(ratings), len(ratings)))
        for u in range(len(ratings)):
            for v in range(u):
                score = overlap_pearson(ratings[u], ratings[v], self.min_common, self.shrink)
                self.similarities_[u, v] = self.similarities_[v, u] = score
        return self

    def _check_user(self, user):
        if not hasattr(self, "ratings_"):
            raise ValueError("fit must be called before prediction")
        if user is not None:
            self._check_index(user, self.ratings_.shape[0], "user")

    @staticmethod
    def _check_index(value, size, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < size:
            raise ValueError(f"{name} must be an integer index in [0, {size})")

    def baseline(self, user, item):
        """Item mean -> user mean -> global mean; no collaborative evidence claimed."""
        self._check_user(user)
        self._check_index(item, self.ratings_.shape[1], "item")
        if self.item_counts_[item] > 0:
            score, source = self.item_means_[item], "item_mean"
        elif user is not None and self.user_counts_[user] > 0:
            score, source = self.user_means_[user], "user_mean"
        else:
            score, source = self.global_mean_, "global_mean"
        return {"score": float(score), "source": source, "neighbors": 0}

    def predict(self, user, item):
        fallback = self.baseline(user, item)
        if user is None or self.user_counts_[user] == 0:
            return fallback
        weights = self.similarities_[user]
        candidates = np.flatnonzero((weights > 0) & ~np.isnan(self.ratings_[:, item]))
        candidates = candidates[candidates != user]
        # Stable ordering breaks equal-similarity ties by row ID.
        neighbors = candidates[np.argsort(-weights[candidates], kind="stable")][:self.max_neighbors]
        if len(neighbors) == 0:
            return fallback
        deviations = self.ratings_[neighbors, item] - self.user_means_[neighbors]
        score = self.user_means_[user] + np.average(deviations, weights=weights[neighbors])
        return {
            "score": float(np.clip(score, *self.rating_bounds)),
            "source": "neighborhood", "neighbors": len(neighbors),
        }

    def recommend(self, user, k=3, eligible=None):
        """Rank unseen supported catalog items; caller may restrict eligibility further."""
        self._check_user(user)
        _positive_integer(k, "k")
        allowed = self.item_counts_ > 0
        if eligible is not None:
            eligible = np.asarray(eligible)
            if eligible.dtype.kind != "b" or eligible.shape != allowed.shape:
                raise ValueError("eligible must be a boolean mask with one entry per item")
            allowed &= eligible
        if user is not None:
            allowed &= np.isnan(self.ratings_[user])
        results = [{"item": int(i), **self.predict(user, int(i))} for i in np.flatnonzero(allowed)]
        return sorted(results, key=lambda row: (-row["score"], row["item"]))[:k]


if __name__ == "__main__":
    a = [5., 4., np.nan, 1.]
    b = [4., 5., 4., 1.]
    print(f"Overlap Pearson (min_common=3, shrink=5): {overlap_pearson(a, b):.6f}")
