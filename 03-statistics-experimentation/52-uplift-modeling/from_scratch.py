"""Educational stratified uplift and known-propensity policy evaluation."""

from dataclasses import dataclass

import numpy as np


def vector(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty, finite one-dimensional array")
    return result


def binary(values, name):
    result = vector(values, name)
    if not np.isin(result, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return result


def same_length(*arrays):
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("arrays must have the same length")


def nonnegative(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 0 or not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return float(result)


def ipw_effect(outcome, treatment, propensity=0.5):
    """Return a noisy pseudo-outcome, never an individual causal label."""
    y, t = binary(outcome, "outcome"), binary(treatment, "treatment")
    same_length(y, t)
    e = np.asarray(propensity, dtype=float)
    if e.ndim == 0:
        e = np.full(y.size, e)
    if e.shape != y.shape or not np.isfinite(e).all() or np.any((e <= 0) | (e >= 1)):
        raise ValueError("propensity must be scalar or aligned, strictly between 0 and 1")
    return t * y / e - (1 - t) * y / (1 - e)


def gain_curve(scores, pseudo_outcome):
    """Return population-normalized gain at complete score-tie blocks."""
    scores, z = vector(scores, "scores"), vector(pseudo_outcome, "pseudo_outcome")
    same_length(scores, z)
    order = np.argsort(-scores, kind="stable")
    end = np.r_[np.flatnonzero(np.diff(scores[order]) != 0) + 1, len(scores)]
    fractions = np.r_[0.0, end / len(scores)]
    gains = np.r_[0.0, np.cumsum(z[order])[end - 1] / len(scores)]
    # Linear interpolation within ties represents random ordering within a tie.
    return fractions, gains


def ranking_metrics(scores, pseudo_outcome):
    fractions, gains = gain_curve(scores, pseudo_outcome)
    auuc = float(np.sum(np.diff(fractions) * (gains[1:] + gains[:-1]) / 2))
    return {"auuc": auuc, "qini_area": auuc - float(gains[-1]) / 2}


def top_fraction(scores, fraction):
    """Select floor(n*fraction) rows; ties keep input order, independent of Y/T."""
    scores = vector(scores, "scores")
    fraction = nonnegative(fraction, "fraction")
    if fraction > 1:
        raise ValueError("fraction must be at most 1")
    selected = np.zeros(scores.size, dtype=bool)
    selected[np.argsort(-scores, kind="stable")[:int(scores.size * fraction)]] = True
    return selected


def cost_aware_policy(scores, value=100.0, cost=5.0, capacity=0.3):
    """Equal-cost, additive policy: positive predicted net value, capped capacity."""
    scores = vector(scores, "scores")
    value, cost = nonnegative(value, "value"), nonnegative(cost, "cost")
    return top_fraction(scores, capacity) & (value * scores > cost)


def policy_value(selected, pseudo_outcome, value=100.0, cost=5.0):
    """Incremental net value per eligible person relative to treating nobody."""
    selected, z = binary(selected, "selected"), vector(pseudo_outcome, "pseudo_outcome")
    same_length(selected, z)
    value, cost = nonnegative(value, "value"), nonnegative(cost, "cost")
    return float(np.mean(selected * (value * z - cost)))


@dataclass
class BinnedUplift:
    """One-feature training quantiles with separate treatment/control means."""

    cuts: np.ndarray
    effects: np.ndarray
    counts: np.ndarray

    def predict(self, feature):
        return self.effects[np.searchsorted(self.cuts, vector(feature, "feature"), side="right")]


def fit_binned_uplift(feature, treatment, outcome, n_bins=6):
    x = vector(feature, "feature")
    t, y = binary(treatment, "treatment"), binary(outcome, "outcome")
    same_length(x, t, y)
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    cuts = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
    cuts = cuts[(cuts > x.min()) & (cuts < x.max())]
    groups = np.searchsorted(cuts, x, side="right")
    effects, counts = [], []
    for group in range(len(cuts) + 1):
        control, treated = (groups == group) & (t == 0), (groups == group) & (t == 1)
        if not control.any() or not treated.any():
            raise ValueError(f"bin {group} needs observations in both treatment arms")
        effects.append(y[treated].mean() - y[control].mean())
        counts.append([control.sum(), treated.sum()])
    return BinnedUplift(cuts, np.array(effects), np.array(counts))


if __name__ == "__main__":
    model = fit_binned_uplift([-2, -1, 1, 2], [0, 1, 0, 1], [0, 1, 1, 0], n_bins=2)
    print("Hand-check strata effects (+1, -1):", model.effects)
    z = ipw_effect([1, 0, 1, 0], [1, 0, 0, 1])
    print("IPW pseudo-outcomes:", z)
    print("Gain fractions and values:", gain_curve([4, 3, 2, 1], z))
