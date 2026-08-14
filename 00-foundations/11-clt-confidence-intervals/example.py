"""Run deterministic synthetic CLT and confidence-interval experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import t


@dataclass(frozen=True)
class SamplingSummary:
    """Summary of a simulated sampling distribution of the mean."""

    sample_size: int
    mean_of_means: float
    empirical_standard_error: float
    theoretical_standard_error: float
    skewness: float


@dataclass(frozen=True)
class CoverageSummary:
    """Summary of a repeated Student-t interval experiment."""

    sample_size: int
    experiments: int
    confidence: float
    empirical_coverage: float
    mean_interval_width: float


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _confidence_level(confidence: float) -> float:
    confidence = float(confidence)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and strictly between 0 and 1.")
    return confidence


def _skewness(values: np.ndarray) -> float:
    centered = values - np.mean(values)
    standard_deviation = np.std(values, ddof=0)
    if standard_deviation == 0.0:
        return 0.0
    return float(np.mean((centered / standard_deviation) ** 3))


def simulate_clt(
    sample_sizes: Iterable[int] = (5, 30, 100),
    *,
    simulations: int = 10_000,
    population_mean: float = 2.0,
    seed: int = 42,
) -> list[SamplingSummary]:
    """Simulate means from an exponential population with finite variance."""
    sizes = list(sample_sizes)
    if not sizes:
        raise ValueError("sample_sizes must contain at least one value.")
    for size in sizes:
        _positive_integer(size, name="sample size")
    _positive_integer(simulations, name="simulations")
    population_mean = float(population_mean)
    if not np.isfinite(population_mean) or population_mean <= 0.0:
        raise ValueError("population_mean must be finite and positive.")

    child_seeds = np.random.SeedSequence(seed).spawn(len(sizes))
    summaries: list[SamplingSummary] = []
    for sample_size, child_seed in zip(sizes, child_seeds, strict=True):
        rng = np.random.default_rng(child_seed)
        samples = rng.exponential(
            scale=population_mean,
            size=(simulations, sample_size),
        )
        means = np.mean(samples, axis=1)
        summaries.append(
            SamplingSummary(
                sample_size=sample_size,
                mean_of_means=float(np.mean(means)),
                empirical_standard_error=float(np.std(means, ddof=0)),
                theoretical_standard_error=population_mean / np.sqrt(sample_size),
                skewness=_skewness(means),
            )
        )
    return summaries


def simulate_t_interval_coverage(
    *,
    sample_size: int = 50,
    experiments: int = 10_000,
    population_mean: float = 2.0,
    confidence: float = 0.95,
    seed: int = 2024,
) -> CoverageSummary:
    """Estimate Student-t interval coverage for exponential sample means."""
    _positive_integer(sample_size, name="sample_size")
    if sample_size < 2:
        raise ValueError("sample_size must be at least two.")
    _positive_integer(experiments, name="experiments")
    population_mean = float(population_mean)
    if not np.isfinite(population_mean) or population_mean <= 0.0:
        raise ValueError("population_mean must be finite and positive.")
    confidence = _confidence_level(confidence)

    rng = np.random.default_rng(seed)
    samples = rng.exponential(
        scale=population_mean,
        size=(experiments, sample_size),
    )
    means = np.mean(samples, axis=1)
    standard_errors = np.std(samples, axis=1, ddof=1) / np.sqrt(sample_size)
    critical_value = float(t.ppf(0.5 + confidence / 2.0, df=sample_size - 1))
    margins = critical_value * standard_errors
    covered = (means - margins <= population_mean) & (
        population_mean <= means + margins
    )
    return CoverageSummary(
        sample_size=sample_size,
        experiments=experiments,
        confidence=confidence,
        empirical_coverage=float(np.mean(covered)),
        mean_interval_width=float(np.mean(2.0 * margins)),
    )


def main() -> None:
    """Run and print both deterministic synthetic experiments."""
    summaries = simulate_clt()
    coverage = simulate_t_interval_coverage()

    print("Synthetic exponential population (educational; no public dataset)")
    print("Population mean and standard deviation: 2.0")
    print("CLT configuration: seed=42, 10,000 simulations per sample size")
    print()
    print(f"{'n':>5} {'mean(means)':>13} {'empirical SE':>14} {'theory SE':>12} {'skewness':>10}")
    for summary in summaries:
        print(
            f"{summary.sample_size:>5} "
            f"{summary.mean_of_means:>13.4f} "
            f"{summary.empirical_standard_error:>14.4f} "
            f"{summary.theoretical_standard_error:>12.4f} "
            f"{summary.skewness:>10.4f}"
        )

    print()
    print("Student-t interval coverage configuration: seed=2024")
    print(
        f"n={coverage.sample_size}, experiments={coverage.experiments:,}, "
        f"nominal confidence={coverage.confidence:.1%}"
    )
    print(f"Empirical coverage: {coverage.empirical_coverage:.2%}")
    print(f"Mean interval width: {coverage.mean_interval_width:.4f}")


if __name__ == "__main__":
    main()
