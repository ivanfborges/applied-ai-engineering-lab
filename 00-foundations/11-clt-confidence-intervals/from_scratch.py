"""Educational first-principles helpers for confidence intervals."""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import NormalDist


def _finite_sample(
    values: Iterable[float],
    *,
    minimum_size: int = 1,
) -> list[float]:
    sample = [float(value) for value in values]
    if len(sample) < minimum_size:
        raise ValueError(f"values must contain at least {minimum_size} observations.")
    if not all(math.isfinite(value) for value in sample):
        raise ValueError("values must contain only finite observations.")
    return sample


def _validate_confidence(confidence: float) -> float:
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and strictly between 0 and 1.")
    return confidence


def sample_mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean of a finite, non-empty sample."""
    sample = _finite_sample(values)
    return math.fsum(sample) / len(sample)


def sample_variance(values: Iterable[float]) -> float:
    """Return the sample variance with Bessel's correction."""
    sample = _finite_sample(values, minimum_size=2)
    mean = math.fsum(sample) / len(sample)
    return math.fsum((value - mean) ** 2 for value in sample) / (len(sample) - 1)


def standard_error_mean(values: Iterable[float]) -> float:
    """Estimate the standard error of an IID sample mean as s / sqrt(n)."""
    sample = _finite_sample(values, minimum_size=2)
    variance = sample_variance(sample)
    return math.sqrt(variance / len(sample))


def normal_mean_confidence_interval(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return an educational normal-approximation interval for a mean.

    The population standard deviation is estimated from the sample. A
    Student-t interval is generally preferable for small normal samples when
    that variance is unknown; this helper intentionally exposes the simpler
    asymptotic construction.
    """
    sample = _finite_sample(values, minimum_size=2)
    confidence = _validate_confidence(confidence)
    mean = sample_mean(sample)
    standard_error = standard_error_mean(sample)
    critical_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    margin = critical_value * standard_error
    return mean - margin, mean + margin


def wilson_score_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer.")
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer.")
    if trials <= 0:
        raise ValueError("trials must be positive.")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials.")
    confidence = _validate_confidence(confidence)

    proportion = successes / trials
    critical_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = critical_value**2
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    half_width = (
        critical_value
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials**2)
        )
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == trials else min(1.0, center + half_width)
    return lower, upper


def main() -> None:
    """Print deterministic examples of the educational formulas."""
    response_times = [12.1, 11.8, 12.4, 11.9, 12.3, 12.0, 12.5, 11.7]
    mean_interval = normal_mean_confidence_interval(response_times)
    proportion_interval = wilson_score_interval(successes=164, trials=200)

    print("Educational normal approximation for a mean")
    print(f"Sample mean: {sample_mean(response_times):.4f}")
    print(f"Estimated SE: {standard_error_mean(response_times):.4f}")
    print(f"95% interval: [{mean_interval[0]:.4f}, {mean_interval[1]:.4f}]")
    print()
    print("Wilson interval for 164 successes in 200 trials")
    print(f"Estimated proportion: {164 / 200:.4f}")
    print(
        f"95% interval: [{proportion_interval[0]:.4f}, "
        f"{proportion_interval[1]:.4f}]"
    )


if __name__ == "__main__":
    main()
