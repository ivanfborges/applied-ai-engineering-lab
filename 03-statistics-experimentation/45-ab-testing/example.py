"""Synthetic user-level A/B test and fixed-horizon calibration simulations."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import binomtest, chi2_contingency

from from_scratch import (
    sample_size_per_arm, two_proportion_test, validate_count, validate_probability,
)


def analyze_users(user_ids, assignment, outcomes, alpha=0.05, expected_share_b=0.5):
    """Analyze all assigned users; missing outcomes must be resolved upstream."""
    user_ids, assignment, outcomes = map(np.asarray, (user_ids, assignment, outcomes))
    if any(array.ndim != 1 for array in (user_ids, assignment, outcomes)):
        raise ValueError("Inputs must be one-dimensional.")
    if not (user_ids.size == assignment.size == outcomes.size) or not user_ids.size:
        raise ValueError("Inputs must be nonempty with equal lengths.")
    if user_ids.dtype.kind not in "iu" or np.unique(user_ids).size != user_ids.size:
        raise ValueError("This synthetic analyzer requires unique integer user IDs.")
    if not np.isin(assignment, [0, 1]).all() or not np.isin(outcomes, [0, 1]).all():
        raise ValueError("Assignment and observed outcomes must be binary, without missing values.")
    expected_share_b = validate_probability(expected_share_b, "expected_share_b")
    b_mask = assignment == 1
    n_b = int(b_mask.sum())
    n_a = int(assignment.size - n_b)
    x_a, x_b = int(outcomes[~b_mask].sum()), int(outcomes[b_mask].sum())
    result = two_proportion_test(x_a, n_a, x_b, n_b, alpha)
    reference = chi2_contingency([[x_a, n_a - x_a], [x_b, n_b - x_b]], correction=False)
    if not np.isclose(result.z_statistic ** 2, reference.statistic, rtol=1e-10, atol=1e-12):
        raise AssertionError("Pooled Z-squared and uncorrected Pearson chi-square disagree.")
    if not np.isclose(result.p_value, reference.pvalue, rtol=1e-10, atol=1e-12):
        raise AssertionError("From-scratch and SciPy p-values disagree.")
    return {
        "n_a": n_a, "n_b": n_b, "successes_a": x_a, "successes_b": x_b,
        **asdict(result), "scipy_pearson_p_value": float(reference.pvalue),
        "srm_p_value": float(binomtest(n_b, n_a + n_b, expected_share_b).pvalue),
    }


def simulate_users(n_users=30000, p_a=0.10, p_b=0.11, seed=45):
    n_users = validate_count(n_users, "n_users", 2)
    p_a, p_b = validate_probability(p_a, "p_a"), validate_probability(p_b, "p_b")
    seed = validate_count(seed, "seed")
    rng = np.random.default_rng(seed)
    user_ids = np.arange(n_users)
    assignment = rng.integers(0, 2, size=n_users)
    outcomes = rng.random(n_users) < np.where(assignment == 0, p_a, p_b)
    return user_ids, assignment, outcomes


def repeated_experiments(n_per_arm, p_a, p_b, repetitions=2000, seed=4501, alpha=0.05):
    """Independent binomial counts equal sums of independent user Bernoullis.

    Each repetition has exactly n_per_arm users in each arm and one final test.
    This checks inference under an ideal generator, not a live assignment system.
    """
    n_per_arm = validate_count(n_per_arm, "n_per_arm", 10)
    repetitions = validate_count(repetitions, "repetitions", 1)
    seed = validate_count(seed, "seed")
    p_a, p_b = validate_probability(p_a, "p_a"), validate_probability(p_b, "p_b")
    alpha = validate_probability(alpha, "alpha")
    rng = np.random.default_rng(seed)
    counts_a = rng.binomial(n_per_arm, p_a, size=repetitions)
    counts_b = rng.binomial(n_per_arm, p_b, size=repetitions)
    rejections, covered = 0, 0
    for x_a, x_b in zip(counts_a, counts_b):
        # Sparse repetitions raise rather than silently being dropped or assigned p=1.
        result = two_proportion_test(x_a, n_per_arm, x_b, n_per_arm, alpha)
        rejections += result.p_value < alpha
        covered += result.ci_low <= p_b - p_a <= result.ci_high
    interval = binomtest(rejections, repetitions).proportion_ci(method="exact")
    return {
        "n_per_arm": n_per_arm, "p_a": p_a, "p_b": p_b, "seed": seed,
        "repetitions": repetitions, "alpha": alpha, "rejections": rejections,
        "rejection_rate": rejections / repetitions,
        "rejection_rate_mc_se": float(np.sqrt((rejections / repetitions)
                                               * (1 - rejections / repetitions) / repetitions)),
        "rejection_rate_mc_ci95": [float(interval.low), float(interval.high)],
        "wald_coverage": covered / repetitions,
    }


def run_study(seed=45, repetitions=2000):
    single = analyze_users(*simulate_users(seed=seed))
    return {
        "data": "Synthetic only; no production users or external dataset.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "design": {"n_users": 30000, "p_a": 0.10, "p_b": 0.11, "seed": seed,
                   "share_b": 0.5, "alpha": 0.05, "srm_alpha": 0.001,
                   "minimum_worthwhile_difference": 0.005,
                   "stopping": "Fixed horizon; one binary observation per assigned user"},
        "single_experiment": single,
        "srm_investigation_required": single["srm_p_value"] < 0.001,
        "planned_users_per_arm": sample_size_per_arm(0.10, 0.11),
        "aa": repeated_experiments(15000, 0.10, 0.10, repetitions, seed * 100 + 1),
        "power_sweep": [repeated_experiments(n, 0.10, 0.11, repetitions, seed * 100 + i)
                        for i, n in enumerate((1000, 5000, 15000), start=2)],
        "interpretation_status": "Author review required; guardrails are not simulated.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing records are preserved.")
    args = parser.parse_args()
    try:
        report = run_study(args.seed, args.repetitions)
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
