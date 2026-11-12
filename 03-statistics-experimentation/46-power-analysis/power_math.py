"""Transparent, two-sided independent-proportion planning for the visual lab."""

from math import ceil, sqrt
from numbers import Integral

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

try:
    from statsmodels.stats.proportion import power_proportions_2indep
except ImportError as error:
    raise ImportError('Install the lab from the repository root: python -m pip install -e ".[dev,experimentation]"') from error

from from_scratch import finite_number, probability


def positive_int(value, name, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum:,}.")
    return int(value)


def validate_design(baseline, effect, alpha):
    baseline = probability(baseline, "Baseline probability")
    effect = finite_number(effect, "Absolute effect")
    treatment = probability(baseline + effect, "Treatment probability")
    alpha = probability(alpha, "Alpha")
    if alpha / 2 == 0:
        raise ValueError("Alpha is too small for floating-point quantiles.")
    return baseline, treatment, alpha


def theoretical_power(baseline, effect, n, alpha=0.05):
    """Balanced normal approximation from statsmodels; fractional n for roots."""
    baseline, treatment, alpha = validate_design(baseline, effect, alpha)
    n = finite_number(n, "Sample size per arm")
    if n <= 0:
        raise ValueError("Sample size per arm must be positive.")
    # statsmodels sample 1 is treatment; sample 2 is control. Ratio is one here.
    return float(power_proportions_2indep(
        treatment - baseline, baseline, n, ratio=1, alpha=alpha,
        alternative="two-sided", return_results=False,
    ))


def required_n(baseline, effect, alpha=0.05, target=0.80):
    validate_design(baseline, effect, alpha)
    target = probability(target, "Target power")
    if effect == 0 or target <= max(0.5, alpha):
        raise ValueError("Use a nonzero effect and target power above both alpha and 0.5.")
    objective = lambda n: theoretical_power(baseline, effect, n, alpha) - target
    if objective(1_000_000_000) < 0:
        raise ValueError("Required n exceeds the one-billion-per-arm search cap.")
    root = 1 if objective(1) >= 0 else brentq(objective, 1, 1_000_000_000)
    n = ceil(root)
    while theoretical_power(baseline, effect, n, alpha) < target:
        n += 1
    return n


def detectable_effect(baseline, n, alpha=0.05, target=0.80):
    """Positive (improvement) MDE; None when unattainable in the interior domain."""
    baseline = probability(baseline, "Baseline probability")
    n = finite_number(n, "Sample size per arm")
    alpha = probability(alpha, "Alpha")
    target = probability(target, "Target power")
    if n <= 0 or target <= max(0.5, alpha):
        raise ValueError("Use positive n and target power above both alpha and 0.5.")
    upper = (1 - baseline) * (1 - 1e-10)
    objective = lambda delta: theoretical_power(baseline, delta, n, alpha) - target
    if objective(upper) < 0:
        return None
    return float(brentq(objective, 0, upper, xtol=1e-12))


def sampling_model(baseline, effect, n, alpha=0.05):
    baseline, treatment, alpha = validate_design(baseline, effect, alpha)
    n = finite_number(n, "Sample size per arm")
    if n <= 0:
        raise ValueError("Sample size per arm must be positive.")
    pooled = (baseline + treatment) / 2
    se_null = sqrt(2 * pooled * (1 - pooled) / n)
    se_alt = sqrt((baseline * (1 - baseline) + treatment * (1 - treatment)) / n)
    critical = norm.isf(alpha / 2) * se_null
    power = float(norm.cdf((-critical - effect) / se_alt) + norm.sf((critical - effect) / se_alt))
    actual_alpha = float(2 * norm.sf(critical / se_null))
    return dict(se_null=se_null, se_alt=se_alt, critical=critical,
                alpha=actual_alpha, power=power, beta=1 - power, null_rate=pooled)


def approximation_warning(baseline, effect, n):
    baseline, treatment, _ = validate_design(baseline, effect, 0.05)
    n = finite_number(n, "Sample size per arm")
    if n <= 0:
        raise ValueError("Sample size per arm must be positive.")
    smallest = min(n * baseline, n * (1 - baseline), n * treatment, n * (1 - treatment))
    if smallest < 10:
        return (f"Normal approximation may be unreliable: smallest expected success/failure count "
                f"is {smallest:.2f} per arm (<10). Consider exact inference or design-specific simulation.")
    return ""


def cluster_information(users, conversations, icc):
    users = positive_int(users, "Users")
    conversations = positive_int(conversations, "Conversations per user")
    icc = finite_number(icc, "ICC")
    if not 0 <= icc <= 1:
        raise ValueError("ICC must be between zero and one for this illustration.")
    raw = users * conversations
    design_effect = 1 + (conversations - 1) * icc
    return raw, design_effect, raw / design_effect


def traffic_plan(n_required, daily_potential, eligibility, enrollment_share, delay, window):
    """Balanced enrollment; share is total traffic admitted, not treatment ratio.

    delay is a fixed observation window. All assigned units eventually have an
    outcome; recent assignments are immature rather than dropped or filtered.
    """
    n_required = positive_int(n_required, "Required sample per arm")
    daily_potential = finite_number(daily_potential, "Daily potential users")
    eligibility = finite_number(eligibility, "Eligibility rate")
    enrollment_share = finite_number(enrollment_share, "Experiment traffic share")
    window = positive_int(window, "Calendar window", 730)
    if isinstance(delay, bool) or not isinstance(delay, Integral) or delay < 0 or delay > 730:
        raise ValueError("Outcome delay must be an integer between 0 and 730 days.")
    if daily_potential <= 0 or not 0 < eligibility <= 1 or not 0 < enrollment_share <= 1:
        raise ValueError("Traffic must be positive; eligibility and traffic share must be in (0, 1].")
    per_day = daily_potential * eligibility * enrollment_share / 2
    reach_day = int(ceil(np.nextafter(n_required / per_day, -np.inf))) + delay
    days = np.arange(window + 1)
    assigned = days * per_day
    mature = np.maximum(days - delay, 0) * per_day
    return dict(days=days, assigned=assigned, mature=mature, reach_day=reach_day,
                available_n=int(np.floor(mature[-1])), sufficient=reach_day <= window)
