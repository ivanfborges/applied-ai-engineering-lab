"""Educational fixed-horizon inference for two independent binary samples."""

from dataclasses import asdict, dataclass
from math import ceil, erfc, isfinite, sqrt
from numbers import Integral, Real
from statistics import NormalDist


def validate_count(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def validate_probability(value: float, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not isfinite(value) or not 0 < value < 1):
        raise ValueError(f"{name} must be finite and strictly between 0 and 1.")
    return float(value)


@dataclass(frozen=True)
class ProportionResult:
    rate_a: float
    rate_b: float
    difference: float
    relative_lift: float
    se_null: float
    se_difference: float
    z_statistic: float
    p_value: float
    ci_low: float
    ci_high: float


def two_proportion_test(
    successes_a: int, n_a: int, successes_b: int, n_b: int, alpha: float = 0.05,
) -> ProportionResult:
    """Pooled two-sided Z-test and separate unpooled Wald difference interval.

    Difference is B minus A. Refuse sparse tables: at least five observed
    successes and failures in each arm is a heuristic, not a validity proof.
    """
    n_a = validate_count(n_a, "n_a", 1)
    n_b = validate_count(n_b, "n_b", 1)
    successes_a = validate_count(successes_a, "successes_a")
    successes_b = validate_count(successes_b, "successes_b")
    alpha = validate_probability(alpha, "alpha")
    if successes_a > n_a or successes_b > n_b:
        raise ValueError("Success counts must not exceed their sample sizes.")
    if min(successes_a, n_a - successes_a, successes_b, n_b - successes_b) < 5:
        raise ValueError("Sparse table: use an appropriate score or exact method; "
                         "this example requires >= 5 successes and failures per arm.")
    rate_a, rate_b = successes_a / n_a, successes_b / n_b
    difference = rate_b - rate_a
    pooled = (successes_a + successes_b) / (n_a + n_b)
    se_null = sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    se_difference = sqrt(rate_a * (1 - rate_a) / n_a
                         + rate_b * (1 - rate_b) / n_b)
    z_statistic = difference / se_null
    # erfc avoids subtracting a near-one CDF in the normal tail.
    p_value = erfc(abs(z_statistic) / sqrt(2))
    critical = -NormalDist().inv_cdf(alpha / 2)
    return ProportionResult(
        rate_a, rate_b, difference, difference / rate_a, se_null, se_difference,
        z_statistic, p_value, difference - critical * se_difference,
        difference + critical * se_difference,
    )


def sample_size_per_arm(
    p_a: float, p_b: float, alpha: float = 0.05, power: float = 0.8,
) -> int:
    """Balanced normal planning approximation; not an exact power guarantee.

    Uses pooled variance for the null threshold and alternative variance for
    the power quantile. The smaller rejection tail is neglected when solving n.
    """
    p_a = validate_probability(p_a, "p_a")
    p_b = validate_probability(p_b, "p_b")
    alpha = validate_probability(alpha, "alpha")
    power = validate_probability(power, "power")
    if p_a == p_b or power <= max(0.5, alpha):
        raise ValueError("Planning needs distinct probabilities and power > max(0.5, alpha).")
    pooled = (p_a + p_b) / 2
    z_alpha = -NormalDist().inv_cdf(alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    numerator = (z_alpha * sqrt(2 * pooled * (1 - pooled))
                 + z_power * sqrt(p_a * (1 - p_a) + p_b * (1 - p_b)))
    return ceil((numerator / abs(p_b - p_a)) ** 2)


if __name__ == "__main__":
    import json

    print("Assumed counts only: 1500/15000 vs 1650/15000; not observed data.")
    print(json.dumps(asdict(two_proportion_test(1500, 15000, 1650, 15000)), indent=2))
    print("Approximate users per arm:", sample_size_per_arm(0.10, 0.11))
