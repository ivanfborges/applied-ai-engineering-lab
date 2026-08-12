"""Compare sampling designs on a deterministic synthetic population."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from from_scratch import estimator_statistics, weighted_mean


REGULAR: Final = "regular"
PREMIUM: Final = "premium"


@dataclass(frozen=True)
class SyntheticPopulation:
    """A finite synthetic population used only for education."""

    segments: np.ndarray
    spend: np.ndarray

    def __post_init__(self) -> None:
        if self.segments.ndim != 1 or self.spend.ndim != 1:
            raise ValueError("segments and spend must be one-dimensional.")
        if len(self.segments) != len(self.spend) or len(self.spend) < 2:
            raise ValueError("segments and spend must have the same length >= 2.")
        if not np.isfinite(self.spend).all():
            raise ValueError("spend must contain only finite values.")
        observed = set(np.unique(self.segments))
        if observed != {REGULAR, PREMIUM}:
            raise ValueError("population must contain regular and premium segments.")

    @property
    def size(self) -> int:
        return len(self.spend)

    @property
    def true_mean(self) -> float:
        return float(np.mean(self.spend))


def _validate_positive_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def generate_synthetic_population(
    population_size: int = 50_000,
    *,
    premium_share: float = 0.10,
    seed: int = 42,
) -> SyntheticPopulation:
    """Generate customer segments and spend with no external data source."""
    _validate_positive_integer(population_size, name="population_size")
    if population_size < 2:
        raise ValueError("population_size must be at least two.")
    premium_share = float(premium_share)
    if not np.isfinite(premium_share) or not 0.0 < premium_share < 1.0:
        raise ValueError("premium_share must be finite and strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    segments = rng.choice(
        np.array([REGULAR, PREMIUM]),
        size=population_size,
        p=[1.0 - premium_share, premium_share],
    )
    regular_spend = rng.normal(100.0, 20.0, size=population_size)
    premium_spend = rng.normal(300.0, 50.0, size=population_size)
    spend = np.where(segments == REGULAR, regular_spend, premium_spend)
    return SyntheticPopulation(segments=segments, spend=spend)


def simple_random_mean(
    population: SyntheticPopulation,
    sample_size: int,
    rng: np.random.Generator,
) -> float:
    """Estimate the population mean using sampling without replacement."""
    _validate_positive_integer(sample_size, name="sample_size")
    if sample_size > population.size:
        raise ValueError("sample_size cannot exceed the population size.")
    indices = rng.choice(population.size, size=sample_size, replace=False)
    return float(np.mean(population.spend[indices]))


def selection_biased_mean(
    population: SyntheticPopulation,
    sample_size: int,
    rng: np.random.Generator,
    *,
    premium_selection_odds: float = 8.0,
) -> float:
    """Estimate the mean when premium rows are more likely to be selected."""
    _validate_positive_integer(sample_size, name="sample_size")
    if sample_size > population.size:
        raise ValueError("sample_size cannot exceed the population size.")
    premium_selection_odds = float(premium_selection_odds)
    if not np.isfinite(premium_selection_odds) or premium_selection_odds <= 0.0:
        raise ValueError("premium_selection_odds must be finite and positive.")

    weights = np.where(
        population.segments == PREMIUM,
        premium_selection_odds,
        1.0,
    )
    probabilities = weights / weights.sum()
    indices = rng.choice(
        population.size,
        size=sample_size,
        replace=False,
        p=probabilities,
    )
    return float(np.mean(population.spend[indices]))


def stratified_mean(
    population: SyntheticPopulation,
    regular_sample_size: int,
    premium_sample_size: int,
    rng: np.random.Generator,
) -> float:
    """Oversample premium rows, then restore empirical population shares."""
    _validate_positive_integer(regular_sample_size, name="regular_sample_size")
    _validate_positive_integer(premium_sample_size, name="premium_sample_size")
    segment_means: list[float] = []
    population_shares: list[float] = []

    for segment, size in (
        (REGULAR, regular_sample_size),
        (PREMIUM, premium_sample_size),
    ):
        segment_indices = np.flatnonzero(population.segments == segment)
        if size > len(segment_indices):
            raise ValueError(f"sample size exceeds the {segment} stratum size.")
        sampled = rng.choice(segment_indices, size=size, replace=False)
        segment_means.append(float(np.mean(population.spend[sampled])))
        population_shares.append(len(segment_indices) / population.size)

    return weighted_mean(segment_means, population_shares)


def run_sampling_experiment(
    population: SyntheticPopulation,
    *,
    sample_size: int = 400,
    trials: int = 500,
    premium_selection_odds: float = 8.0,
    seed: int = 2024,
) -> dict[str, dict[str, float]]:
    """Construct empirical sampling distributions for three designs."""
    _validate_positive_integer(sample_size, name="sample_size")
    _validate_positive_integer(trials, name="trials")
    if sample_size % 2:
        raise ValueError("sample_size must be even for the equal-allocation design.")
    if sample_size > population.size:
        raise ValueError("sample_size cannot exceed the population size.")

    random_rng, biased_rng, stratified_rng = (
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed).spawn(3)
    )
    estimates: dict[str, list[float]] = {
        "simple_random": [],
        "selection_biased": [],
        "stratified_weighted": [],
    }
    per_stratum = sample_size // 2
    for _ in range(trials):
        estimates["simple_random"].append(
            simple_random_mean(population, sample_size, random_rng)
        )
        estimates["selection_biased"].append(
            selection_biased_mean(
                population,
                sample_size,
                biased_rng,
                premium_selection_odds=premium_selection_odds,
            )
        )
        estimates["stratified_weighted"].append(
            stratified_mean(
                population,
                regular_sample_size=per_stratum,
                premium_sample_size=per_stratum,
                rng=stratified_rng,
            )
        )

    return {
        design: estimator_statistics(values, population.true_mean)
        for design, values in estimates.items()
    }


def main() -> None:
    """Run and report the deterministic sampling experiment."""
    population = generate_synthetic_population()
    summaries = run_sampling_experiment(population)
    premium_share = float(np.mean(population.segments == PREMIUM))

    print("Synthetic finite population (educational; no public dataset)")
    print(f"Population size: {population.size:,}")
    print(f"Empirical premium share: {premium_share:.4f}")
    print(f"True population mean spend: {population.true_mean:.4f}")
    print("Configuration: 500 trials, n=400, without replacement")
    print("Stratified allocation: 200 regular + 200 premium, population-weighted")
    print("Biased design: premium selection odds multiplier = 8.0")
    print()
    print(f"{'design':<23} {'mean':>10} {'bias':>10} {'variance':>12} {'mse':>12}")
    for design, statistics in summaries.items():
        print(
            f"{design:<23} "
            f"{statistics['expected_estimate']:>10.4f} "
            f"{statistics['bias']:>10.4f} "
            f"{statistics['variance']:>12.4f} "
            f"{statistics['mse']:>12.4f}"
        )


if __name__ == "__main__":
    main()
