"""Prospective two-proportion planning with SciPy; no data or plots generated."""

from math import ceil, nextafter, sqrt
import platform

import scipy
from scipy.optimize import brentq
from scipy.stats import norm

from from_scratch import finite_number, probability, sample_size_per_arm


def normal_power(
    p_control: float, p_treatment: float, n_control: float,
    n_treatment: float, alpha: float = 0.05,
) -> float:
    """Both rejection tails, pooled null SE, unpooled alternative SE.

    Positive real sample sizes allow continuous root finding. This is an
    asymptotic planning model, not finite-sample power of an exact test.
    """
    p_control = probability(p_control, "p_control")
    p_treatment = probability(p_treatment, "p_treatment")
    alpha = probability(alpha, "alpha")
    n_control = finite_number(n_control, "n_control")
    n_treatment = finite_number(n_treatment, "n_treatment")
    if min(n_control, n_treatment) <= 0:
        raise ValueError("Sample sizes must be positive.")
    if alpha / 2 == 0:
        raise ValueError("alpha is too small for floating-point quantiles.")

    pooled = (n_control * p_control + n_treatment * p_treatment) / (n_control + n_treatment)
    se_null = sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    se_alternative = sqrt(
        p_control * (1 - p_control) / n_control
        + p_treatment * (1 - p_treatment) / n_treatment
    )
    threshold = norm.isf(alpha / 2) * se_null
    effect = p_treatment - p_control
    return float(
        norm.sf((threshold - effect) / se_alternative)
        + norm.cdf((-threshold - effect) / se_alternative)
    )


def required_sample_sizes(
    p_control: float, p_treatment: float, alpha: float = 0.05,
    target_power: float = 0.80, ratio: float = 1.0,
) -> tuple[int, int]:
    """Solve continuous n, round up both arms and recheck achieved power.

    ratio = n_treatment / n_control. The search caps control at 1e9 units;
    rounded unequal allocations need not be the globally smallest integer plan.
    """
    p_control = probability(p_control, "p_control")
    p_treatment = probability(p_treatment, "p_treatment")
    alpha = probability(alpha, "alpha")
    target_power = probability(target_power, "target_power")
    ratio = finite_number(ratio, "ratio")
    if ratio <= 0:
        raise ValueError("ratio must be positive.")
    if p_control == p_treatment:
        raise ValueError("A nonzero assumed effect is required.")
    if target_power <= max(0.5, alpha):
        raise ValueError("target_power must exceed both 0.5 and alpha.")

    def objective(n):
        return normal_power(p_control, p_treatment, n, ratio * n, alpha) - target_power

    lower = max(1.0, 1.0 / ratio)
    upper = 1_000_000_000
    if lower > upper or objective(upper) < 0:
        raise ValueError("Target power is not reached within the 1e9-control search cap.")
    root = lower if objective(lower) >= 0 else brentq(objective, lower, upper)
    n_control, n_treatment = ceil(root), ceil(ratio * root)
    # Independent rounding slightly changes allocation; check the resulting plan.
    while normal_power(p_control, p_treatment, n_control, n_treatment, alpha) < target_power:
        n_control += 1
        n_treatment = ceil(ratio * n_control)
        if n_control > upper:
            raise ValueError("Rounded plan exceeds the 1e9-control search cap.")
    return n_control, n_treatment


def calendar_days(
    n_control: float, n_treatment: float, daily_unique_units: float,
    eligibility: float, treatment_share: float = 0.5,
    outcome_delay_days: int = 0, minimum_enrollment_days: int = 0,
) -> int:
    """Expected traffic accumulation plus final outcome maturation.

    Counts refer to distinct, once-enrolled units. Random allocation variability,
    seasonal traffic, attrition and missing outcomes are not modeled.
    """
    n_control = finite_number(n_control, "n_control")
    n_treatment = finite_number(n_treatment, "n_treatment")
    daily_unique_units = finite_number(daily_unique_units, "daily_unique_units")
    eligibility = finite_number(eligibility, "eligibility")
    treatment_share = probability(treatment_share, "treatment_share")
    if min(n_control, n_treatment, daily_unique_units) <= 0 or not 0 < eligibility <= 1:
        raise ValueError("Counts and traffic must be positive; eligibility must be in (0, 1].")
    for value in (outcome_delay_days, minimum_enrollment_days):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Day constraints must be nonnegative integers.")
    daily_eligible = daily_unique_units * eligibility
    expected_days = max(
        n_control / (daily_eligible * (1 - treatment_share)),
        n_treatment / (daily_eligible * treatment_share),
    )
    # Division can land one floating-point step above an exact integer day.
    enrollment = max(1, ceil(nextafter(expected_days, -float("inf"))))
    return max(enrollment, minimum_enrollment_days) + outcome_delay_days


def main() -> None:
    print(f"Python {platform.python_version()} | SciPy {scipy.__version__}")
    print("HYPOTHETICAL planning assumptions; no sampled outcomes or observed business effects.")
    print("scenario                control treatment total  achieved_power")
    scenarios = [
        ("effect +0.5 pp", 0.105, 0.05, 0.80, 1.0),
        ("baseline +1 pp", 0.11, 0.05, 0.80, 1.0),
        ("effect +2 pp", 0.12, 0.05, 0.80, 1.0),
        ("alpha 0.01", 0.11, 0.01, 0.80, 1.0),
        ("target power 0.90", 0.11, 0.05, 0.90, 1.0),
        ("target power 0.95", 0.11, 0.05, 0.95, 1.0),
        ("treatment:control 2:1", 0.11, 0.05, 0.80, 2.0),
    ]
    for label, treatment, alpha, target, ratio in scenarios:
        nc, nt = required_sample_sizes(0.10, treatment, alpha, target, ratio)
        power = normal_power(0.10, treatment, nc, nt, alpha)
        print(f"{label:23} {nc:7d} {nt:9d} {nc + nt:6d} {power:.6f}")

    nc, nt = required_sample_sizes(0.10, 0.11)
    print(f"Balanced dominant-tail formula: {sample_size_per_arm(0.10, 0.11):,}/arm")
    for traffic in (500, 2500, 10000):
        days = calendar_days(nc, nt, traffic, 0.40, outcome_delay_days=2,
                             minimum_enrollment_days=7)
        print(f"{traffic:,} new units/day, 40% eligible, 7-day floor + 2-day delay: {days} days")
    # Fourteen calendar days minus two days of maturation leaves twelve to enroll.
    available_per_arm = (14 - 2) * 2500 * 0.40 / 2
    available_power = normal_power(0.10, 0.11, available_per_arm, available_per_arm)
    print(f"14-day deadline, 2,500 units/day: {available_per_arm:,.0f}/arm; power={available_power:.6f}")
    print(f"Hypothetical extra treatment cost at 0.03 currency units/unit: {nt * 0.03:.2f}")
    print("Interpretation candidates require author review; these are analytical calculations.")


if __name__ == "__main__":
    main()
