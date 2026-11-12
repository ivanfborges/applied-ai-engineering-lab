"""Educational balanced, two-sided sample-size approximation (standard library)."""

from math import ceil, isfinite, sqrt
from numbers import Real
from statistics import NormalDist


def finite_number(value: float, name: str) -> float:
    """Reject nonnumeric, boolean and nonfinite planning inputs explicitly."""
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def probability(value: float, name: str) -> float:
    value = finite_number(value, name)
    if not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between zero and one.")
    return value


def sample_size_per_arm(
    p_control: float, p_treatment: float, alpha: float = 0.05,
    target_power: float = 0.80,
) -> int:
    """Round up the dominant-tail approximation; not exact binomial sizing.

    This closed form supports target power above both 0.5 and alpha. It assumes
    equal allocation and omits the remote rejection tail of a two-sided test.
    """
    p_control = probability(p_control, "p_control")
    p_treatment = probability(p_treatment, "p_treatment")
    alpha = probability(alpha, "alpha")
    target_power = probability(target_power, "target_power")
    if p_control == p_treatment:
        raise ValueError("A nonzero assumed effect is required.")
    if target_power <= max(0.5, alpha):
        raise ValueError("target_power must exceed both 0.5 and alpha.")
    if alpha / 2 == 0:
        raise ValueError("alpha is too small for floating-point quantiles.")

    normal = NormalDist()
    z_alpha = -normal.inv_cdf(alpha / 2)
    z_power = normal.inv_cdf(target_power)
    pooled = (p_control + p_treatment) / 2
    null_variance = 2 * pooled * (1 - pooled)
    alternative_variance = p_control * (1 - p_control) + p_treatment * (1 - p_treatment)
    standardized = (
        z_alpha * sqrt(null_variance) + z_power * sqrt(alternative_variance)
    ) / abs(p_treatment - p_control)
    if not isfinite(standardized * standardized):
        raise ValueError("The assumed effect requires an unrepresentable sample size.")
    return ceil(standardized * standardized)


if __name__ == "__main__":
    n = sample_size_per_arm(0.10, 0.11)
    print(f"Hypothetical 10% -> 11%, two-sided alpha=0.05, power=0.80: {n:,}/arm")
    print("Dominant-tail normal approximation; no experimental outcomes generated.")
