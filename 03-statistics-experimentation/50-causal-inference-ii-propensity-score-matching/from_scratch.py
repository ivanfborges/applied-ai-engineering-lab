"""Educational matching, weighting and balance diagnostics; no inference engine."""

from dataclasses import dataclass

import numpy as np


def vector(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite vector.")
    return result


def treatment_vector(treatment):
    treatment = vector(treatment, "treatment")
    if not np.isin(treatment, [0, 1]).all() or np.unique(treatment).size != 2:
        raise ValueError("treatment must be binary with both groups present.")
    return treatment.astype(int)


def propensity_vector(scores, treatment):
    scores = vector(scores, "scores")
    if scores.shape != treatment.shape or np.any((scores <= 0) | (scores >= 1)):
        raise ValueError("scores must align with treatment and lie strictly inside (0, 1).")
    return scores


def analysis_weights(weights, treatment):
    weights = vector(weights, "weights")
    if weights.shape != treatment.shape or np.any(weights < 0):
        raise ValueError("weights must align with treatment and be nonnegative.")
    if any(not np.any(weights[treatment == arm] > 0) for arm in (0, 1)):
        raise ValueError("Each group needs positive total weight.")
    return weights


@dataclass(frozen=True)
class Matches:
    treated: np.ndarray
    control: np.ndarray
    distance: np.ndarray
    unmatched: np.ndarray


def match_propensity(scores, treatment, caliper=None):
    """1:1 logit-score matching with replacement; ties use first control row.

    Caliper is an inclusive absolute logit distance, not an SD multiplier.
    One distance vector at a time keeps auxiliary memory linear in sample size.
    """
    treatment = treatment_vector(treatment)
    scores = propensity_vector(scores, treatment)
    if caliper is not None and (not np.isfinite(caliper) or caliper < 0):
        raise ValueError("caliper must be finite and nonnegative, or None.")
    treated = np.flatnonzero(treatment == 1)
    controls = np.flatnonzero(treatment == 0)
    logits = np.log(scores) - np.log1p(-scores)
    nearest = np.array([
        np.argmin(np.abs(logits[controls] - logits[index])) for index in treated
    ])
    matched = controls[nearest]
    distances = np.abs(logits[treated] - logits[matched])
    keep = np.ones(treated.size, dtype=bool) if caliper is None else distances <= caliper
    return Matches(treated[keep], matched[keep], distances[keep], treated[~keep])


def matching_weights(matches, treatment):
    """Count reused controls as frequency weights, preserving pair multiplicity."""
    treatment = treatment_vector(treatment)
    if matches.treated.size == 0:
        raise ValueError("No treated units retained by matching.")
    # Matches must come from match_propensity for this same row ordering.
    weights = np.bincount(matches.treated, minlength=treatment.size).astype(float)
    weights += np.bincount(matches.control, minlength=treatment.size)
    return analysis_weights(weights, treatment)


def ipw_weights(scores, treatment, estimand="ate"):
    """Observed-arm ATE weights or ATT odds weights, without clipping/trimming."""
    treatment = treatment_vector(treatment)
    scores = propensity_vector(scores, treatment)
    if estimand == "ate":
        return np.where(treatment == 1, 1 / scores, 1 / (1 - scores))
    if estimand == "att":
        return np.where(treatment == 1, 1.0, scores / (1 - scores))
    raise ValueError("estimand must be 'ate' or 'att'.")


def weighted_effect(outcome, treatment, weights):
    """Difference of separately normalized arm means (Hajek style for IPW)."""
    treatment = treatment_vector(treatment)
    outcome = vector(outcome, "outcome")
    if outcome.shape != treatment.shape:
        raise ValueError("outcome must align with treatment.")
    weights = analysis_weights(weights, treatment)
    means = []
    for arm in (0, 1):
        selected = treatment == arm
        arm_weights = weights[selected] / weights[selected].max()
        means.append(np.average(outcome[selected], weights=arm_weights))
    return float(means[1] - means[0])


def effective_sample_size(weights):
    weights = vector(weights, "weights")
    if np.any(weights < 0) or weights.max() <= 0:
        raise ValueError("weights must be nonnegative with positive mass.")
    scaled = weights / weights.max()
    return float(scaled.sum() ** 2 / np.dot(scaled, scaled))


def standardized_mean_difference(covariates, treatment, weights=None):
    """Signed SMD with original pooled sample SD fixed before/after adjustment.

    Zero reference variance gives 0 for equal means, signed infinity otherwise.
    Weights include matching frequencies, so reused controls are counted fully.
    """
    treatment = treatment_vector(treatment)
    covariates = np.asarray(covariates, dtype=float)
    if (covariates.ndim != 2 or covariates.shape[0] != treatment.size
            or covariates.shape[1] == 0 or not np.isfinite(covariates).all()):
        raise ValueError("covariates must be a finite nonempty n-by-p matrix.")
    if min(np.bincount(treatment)) < 2:
        raise ValueError("SMD needs at least two original observations per arm.")
    weights = analysis_weights(
        np.ones(treatment.size) if weights is None else weights, treatment
    )
    groups = [covariates[treatment == arm] for arm in (0, 1)]
    scale = np.sqrt((groups[0].var(axis=0, ddof=1) + groups[1].var(axis=0, ddof=1)) / 2)
    means = []
    for arm in (0, 1):
        arm_weights = weights[treatment == arm]
        means.append(np.average(groups[arm], axis=0, weights=arm_weights / arm_weights.max()))
    difference = means[1] - means[0]
    smd = np.zeros_like(difference)
    np.divide(difference, scale, out=smd, where=scale > 0)
    zero_scale_difference = (scale == 0) & (difference != 0)
    smd[zero_scale_difference] = np.sign(difference[zero_scale_difference]) * np.inf
    return smd
