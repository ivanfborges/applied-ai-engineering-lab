"""Seeded, bounded binomial experiments and pooled two-sided Z inference."""

import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

from from_scratch import probability
from power_math import positive_int, theoretical_power, validate_design


def pooled_z_test(success_control, success_treatment, n_control, n_treatment, alpha=0.05):
    """Vectorized actual pooled Z-test; all-zero/all-one draws do not reject."""
    n_control = positive_int(n_control, "Control sample size")
    n_treatment = positive_int(n_treatment, "Treatment sample size")
    alpha = probability(alpha, "Alpha")
    xc, xt = np.broadcast_arrays(success_control, success_treatment)
    for counts, total in ((xc, n_control), (xt, n_treatment)):
        if counts.dtype.kind not in "iu" or np.any(counts < 0) or np.any(counts > total):
            raise ValueError("Success counts must be integers between zero and arm size.")
    pc, pt = xc / n_control, xt / n_treatment
    pooled = (xc + xt) / (n_control + n_treatment)
    se_null = np.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    effect = pt - pc
    z = np.divide(effect, se_null, out=np.zeros_like(effect, dtype=float), where=se_null > 0)
    p_values = 2 * norm.sf(np.abs(z))
    se_alt = np.sqrt(pc * (1 - pc) / n_control + pt * (1 - pt) / n_treatment)
    margin = norm.isf(alpha / 2) * se_alt
    return dict(control_rate=pc, treatment_rate=pt, effect=effect, p_value=p_values,
                reject=p_values < alpha, degenerate=se_null == 0,
                ci_low=effect - margin, ci_high=effect + margin)


def simulate_experiments(baseline=0.10, effect=0.01, n=14751, alpha=0.05,
                         repetitions=5000, seed=46, mode="alternative"):
    baseline, treatment, alpha = validate_design(baseline, effect, alpha)
    n = positive_int(n, "Sample size per arm", 10_000_000)
    repetitions = positive_int(repetitions, "Repetitions", 100_000)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32).")
    if mode not in ("null", "alternative"):
        raise ValueError("Mode must be 'null' or 'alternative'.")
    actual_effect = 0.0 if mode == "null" else effect
    treatment = baseline + actual_effect
    rng = np.random.default_rng(seed)
    xc = rng.binomial(n, baseline, repetitions)
    xt = rng.binomial(n, treatment, repetitions)
    result = pooled_z_test(xc, xt, n, n, alpha)
    rejections = int(result["reject"].sum())
    interval = proportion_confint(rejections, repetitions, alpha=0.05, method="wilson")
    return dict(
        config=dict(baseline=baseline, effect=actual_effect, n=n, alpha=alpha,
                    repetitions=repetitions, seed=int(seed), mode=mode),
        outcomes=result, rejections=rejections, rejection_rate=rejections / repetitions,
        ci_low=float(interval[0]), ci_high=float(interval[1]),
        analytical=theoretical_power(baseline, actual_effect, n, alpha),
        degenerate_draws=int(result["degenerate"].sum()),
    )


def synthetic_intervals(baseline, effect, sample_sizes, alpha=0.05, seed=46):
    """One seeded synthetic experiment per n, not selected for significance."""
    baseline, treatment, alpha = validate_design(baseline, effect, alpha)
    rng = np.random.default_rng(seed)
    rows = []
    for n in sample_sizes:
        n = positive_int(n, "Sample size")
        xc, xt = rng.binomial(n, baseline), rng.binomial(n, treatment)
        result = pooled_z_test(int(xc), int(xt), n, n, alpha)
        rows.append(dict(n=n, **{key: float(result[key]) for key in ("effect", "ci_low", "ci_high", "p_value")}))
    return rows
