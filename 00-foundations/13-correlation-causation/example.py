"""Run deterministic synthetic examples of association and causal traps."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from from_scratch import ols_coefficients, pearson_correlation


@dataclass(frozen=True)
class ConfoundingResult:
    """Summary of one synthetic treatment-outcome scenario."""

    label: str
    true_effect: float
    correlation: float
    naive_coefficient: float
    adjusted_coefficient: float


@dataclass(frozen=True)
class ColliderResult:
    """Correlations before and after selection on a common effect."""

    population_correlation: float
    selected_correlation: float
    selected_count: int


def generate_confounding_data(
    *,
    sample_size: int = 5_000,
    true_effect: float = 2.0,
    assignment_strength: float = 1.5,
    intent_effect: float = 5.0,
    randomized_exposure: bool = False,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic intent, ad exposure, and purchase value.

    The observational graph is ``intent -> exposure -> purchase`` plus
    ``intent -> purchase``. With randomized exposure, the first edge is cut.
    """
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer.")
    if sample_size < 3:
        raise ValueError("sample_size must be at least three.")
    parameters = (true_effect, assignment_strength, intent_effect)
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("simulation coefficients must be finite.")

    rng = np.random.default_rng(seed)
    intent = rng.normal(0.0, 1.0, sample_size)
    exposure_noise = rng.normal(0.0, 1.0, sample_size)
    if randomized_exposure:
        ad_exposure = exposure_noise
    else:
        ad_exposure = assignment_strength * intent + exposure_noise
    purchase_value = (
        true_effect * ad_exposure
        + intent_effect * intent
        + rng.normal(0.0, 2.0, sample_size)
    )
    return intent, ad_exposure, purchase_value


def analyze_confounding(
    *,
    label: str,
    true_effect: float,
    randomized_exposure: bool,
    seed: int = 42,
) -> ConfoundingResult:
    """Compare an unadjusted coefficient with observed-confounder adjustment."""
    intent, exposure, purchase = generate_confounding_data(
        true_effect=true_effect,
        randomized_exposure=randomized_exposure,
        seed=seed,
    )
    naive = ols_coefficients(exposure, purchase)
    adjusted = ols_coefficients(np.column_stack((exposure, intent)), purchase)
    return ConfoundingResult(
        label=label,
        true_effect=true_effect,
        correlation=pearson_correlation(exposure, purchase),
        naive_coefficient=float(naive[1]),
        adjusted_coefficient=float(adjusted[1]),
    )


def nonlinear_dependence() -> float:
    """Return Pearson correlation for the deterministic relation y = x squared."""
    positive = np.linspace(0.001, 3.0, 2_500)
    x = np.concatenate((-positive[::-1], positive))
    return pearson_correlation(x, x**2)


def analyze_collider(*, sample_size: int = 20_000, seed: int = 42) -> ColliderResult:
    """Show selection bias after conditioning on a synthetic hiring collider."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer.")
    if sample_size < 100:
        raise ValueError("sample_size must be at least 100.")

    rng = np.random.default_rng(seed)
    technical_skill = rng.normal(0.0, 1.0, sample_size)
    communication_skill = rng.normal(0.0, 1.0, sample_size)
    hiring_score = (
        technical_skill
        + communication_skill
        + rng.normal(0.0, 0.5, sample_size)
    )
    selected = hiring_score >= np.quantile(hiring_score, 0.75)
    return ColliderResult(
        population_correlation=pearson_correlation(
            technical_skill, communication_skill
        ),
        selected_correlation=pearson_correlation(
            technical_skill[selected], communication_skill[selected]
        ),
        selected_count=int(np.sum(selected)),
    )


def main() -> None:
    """Execute all synthetic demonstrations and print their summaries."""
    scenarios = (
        analyze_confounding(
            label="observational, true effect = 2",
            true_effect=2.0,
            randomized_exposure=False,
        ),
        analyze_confounding(
            label="observational, true effect = 0",
            true_effect=0.0,
            randomized_exposure=False,
        ),
        analyze_confounding(
            label="randomized exposure, true effect = 2",
            true_effect=2.0,
            randomized_exposure=True,
        ),
    )

    print("Correlation and causation experiments (synthetic data)")
    print(
        f"{'Scenario':<39} {'True':>7} {'Corr':>8} "
        f"{'Naive b':>10} {'Adjusted b':>12}"
    )
    for result in scenarios:
        print(
            f"{result.label:<39} {result.true_effect:>7.3f} "
            f"{result.correlation:>8.3f} {result.naive_coefficient:>10.3f} "
            f"{result.adjusted_coefficient:>12.3f}"
        )

    collider = analyze_collider()
    print("\nNonlinear deterministic dependence")
    print(f"Pearson corr(x, x^2): {nonlinear_dependence():.6f}")
    print("\nCollider conditioning")
    print(f"Population skill correlation: {collider.population_correlation:.3f}")
    print(
        f"Selected-only skill correlation: {collider.selected_correlation:.3f} "
        f"(n={collider.selected_count})"
    )
    print("\nAuthor review required before adopting causal interpretations.")


if __name__ == "__main__":
    main()
