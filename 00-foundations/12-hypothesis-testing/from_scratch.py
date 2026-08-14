"""Educational first-principles helpers for paired hypothesis tests."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


ALTERNATIVES = {"two-sided", "greater", "less"}


@dataclass(frozen=True)
class PairedStatistics:
    """Descriptive statistics for paired differences."""

    sample_size: int
    mean_difference: float
    sample_standard_deviation: float
    standard_error: float
    t_statistic: float
    cohens_dz: float


@dataclass(frozen=True)
class SignFlipResult:
    """Result of an exact or Monte Carlo paired sign-flip test."""

    observed_mean: float
    p_value: float
    samples_evaluated: int
    exact: bool
    alternative: str


def _finite_differences(values: Iterable[float]) -> np.ndarray:
    differences = np.asarray(list(values), dtype=float)
    if differences.ndim != 1 or differences.size < 2:
        raise ValueError("differences must contain at least two observations.")
    if not np.all(np.isfinite(differences)):
        raise ValueError("differences must contain only finite observations.")
    return differences


def _validate_alternative(alternative: str) -> str:
    if alternative not in ALTERNATIVES:
        allowed = ", ".join(sorted(ALTERNATIVES))
        raise ValueError(f"alternative must be one of: {allowed}.")
    return alternative


def paired_statistics(differences: Iterable[float]) -> PairedStatistics:
    """Calculate the paired t statistic and Cohen's dz from first principles."""
    sample = _finite_differences(differences)
    sample_size = int(sample.size)
    mean_difference = math.fsum(float(value) for value in sample) / sample_size
    squared_deviations = math.fsum(
        (float(value) - mean_difference) ** 2 for value in sample
    )
    sample_variance = squared_deviations / (sample_size - 1)
    sample_standard_deviation = math.sqrt(sample_variance)
    if sample_standard_deviation == 0.0:
        raise ValueError("differences must have non-zero sample variance.")
    standard_error = sample_standard_deviation / math.sqrt(sample_size)

    return PairedStatistics(
        sample_size=sample_size,
        mean_difference=mean_difference,
        sample_standard_deviation=sample_standard_deviation,
        standard_error=standard_error,
        t_statistic=mean_difference / standard_error,
        cohens_dz=mean_difference / sample_standard_deviation,
    )


def _is_extreme(statistic: float, observed: float, alternative: str) -> bool:
    tolerance = 1e-12
    if alternative == "two-sided":
        return abs(statistic) >= abs(observed) - tolerance
    if alternative == "greater":
        return statistic >= observed - tolerance
    return statistic <= observed + tolerance


def paired_sign_flip_test(
    differences: Iterable[float],
    *,
    alternative: str = "two-sided",
    permutations: int = 20_000,
    seed: int = 12,
    exact_max_pairs: int = 16,
) -> SignFlipResult:
    """Test a paired mean by flipping difference signs under the null.

    Every sign assignment is enumerated for at most ``exact_max_pairs`` pairs.
    Larger samples use a seeded Monte Carlo approximation and the plus-one
    correction. Validity requires the paired differences to be exchangeable
    with their negatives under the null; this procedure is not assumption-free.
    """
    sample = _finite_differences(differences)
    alternative = _validate_alternative(alternative)
    if isinstance(permutations, bool) or not isinstance(permutations, int):
        raise TypeError("permutations must be an integer.")
    if permutations <= 0:
        raise ValueError("permutations must be positive.")
    if isinstance(exact_max_pairs, bool) or not isinstance(exact_max_pairs, int):
        raise TypeError("exact_max_pairs must be an integer.")
    if exact_max_pairs < 0:
        raise ValueError("exact_max_pairs must be non-negative.")

    observed = float(np.mean(sample))
    extreme_count = 0

    if sample.size <= exact_max_pairs:
        assignments = itertools.product((-1.0, 1.0), repeat=sample.size)
        samples_evaluated = 0
        for signs in assignments:
            statistic = float(np.mean(sample * np.asarray(signs)))
            extreme_count += _is_extreme(statistic, observed, alternative)
            samples_evaluated += 1
        p_value = extreme_count / samples_evaluated
        exact = True
    else:
        rng = np.random.default_rng(seed)
        remaining = permutations
        samples_evaluated = permutations
        while remaining:
            batch_size = min(remaining, 10_000)
            signs = rng.choice((-1.0, 1.0), size=(batch_size, sample.size))
            statistics = np.mean(signs * sample, axis=1)
            if alternative == "two-sided":
                extreme_count += int(
                    np.count_nonzero(np.abs(statistics) >= abs(observed) - 1e-12)
                )
            elif alternative == "greater":
                extreme_count += int(
                    np.count_nonzero(statistics >= observed - 1e-12)
                )
            else:
                extreme_count += int(
                    np.count_nonzero(statistics <= observed + 1e-12)
                )
            remaining -= batch_size
        p_value = (extreme_count + 1) / (samples_evaluated + 1)
        exact = False

    return SignFlipResult(
        observed_mean=observed,
        p_value=p_value,
        samples_evaluated=samples_evaluated,
        exact=exact,
        alternative=alternative,
    )


def main() -> None:
    """Print an exact sign-flip example for a small synthetic paired sample."""
    synthetic_differences = [0.04, 0.03, 0.01, 0.05, -0.01, 0.02, 0.04, 0.03]
    statistics = paired_statistics(synthetic_differences)
    result = paired_sign_flip_test(synthetic_differences)

    print("Educational paired analysis (synthetic differences)")
    print(f"Pairs: {statistics.sample_size}")
    print(f"Mean difference: {statistics.mean_difference:.4f}")
    print(f"Standard error: {statistics.standard_error:.4f}")
    print(f"Paired t statistic: {statistics.t_statistic:.4f}")
    print(f"Cohen's dz: {statistics.cohens_dz:.4f}")
    print(f"Exact sign-flip p-value: {result.p_value:.6f}")
    print(f"Sign assignments evaluated: {result.samples_evaluated}")


if __name__ == "__main__":
    main()
