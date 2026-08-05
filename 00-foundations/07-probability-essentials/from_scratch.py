"""Small probability utilities implemented from first principles.

These functions are educational rather than production-grade. They use Python
loops to keep each formula visible and should not replace tested numerical
libraries for performance-sensitive or high-stakes work.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _validate_probability(value: float, name: str) -> None:
    """Validate one finite value in the closed interval [0, 1]."""
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1.")


def _validate_distribution(
    values: Sequence[float],
    probabilities: Sequence[float],
) -> None:
    """Validate a finite discrete probability distribution."""
    if not values:
        raise ValueError("values and probabilities cannot be empty.")
    if len(values) != len(probabilities):
        raise ValueError("values and probabilities must have the same length.")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("values must contain only finite numbers.")

    for probability in probabilities:
        _validate_probability(float(probability), "each probability")
    if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-12):
        raise ValueError("probabilities must sum to 1.")


def expected_value(
    values: Sequence[float],
    probabilities: Sequence[float],
) -> float:
    """Return the probability-weighted mean of a discrete random variable."""
    _validate_distribution(values, probabilities)
    return math.fsum(
        float(value) * float(probability)
        for value, probability in zip(values, probabilities, strict=True)
    )


def variance(
    values: Sequence[float],
    probabilities: Sequence[float],
) -> float:
    """Return population variance E[(X - E[X])**2]."""
    mean = expected_value(values, probabilities)
    return math.fsum(
        (float(value) - mean) ** 2 * float(probability)
        for value, probability in zip(values, probabilities, strict=True)
    )


def conditional_probability(
    event_a: Sequence[bool],
    event_b: Sequence[bool],
) -> float:
    """Estimate P(A | B) from aligned Boolean observations."""
    if not event_a:
        raise ValueError("events cannot be empty.")
    if len(event_a) != len(event_b):
        raise ValueError("events must contain the same number of observations.")

    count_b = sum(bool(b) for b in event_b)
    if count_b == 0:
        raise ValueError("P(B) cannot be estimated because B never occurred.")

    count_a_and_b = sum(
        bool(a) and bool(b)
        for a, b in zip(event_a, event_b, strict=True)
    )
    return count_a_and_b / count_b


def binary_bayes_update(
    prior: float,
    probability_evidence_given_hypothesis: float,
    probability_evidence_given_not_hypothesis: float,
) -> float:
    """Return P(H | E) for a binary hypothesis using Bayes' theorem."""
    _validate_probability(prior, "prior")
    _validate_probability(
        probability_evidence_given_hypothesis,
        "probability_evidence_given_hypothesis",
    )
    _validate_probability(
        probability_evidence_given_not_hypothesis,
        "probability_evidence_given_not_hypothesis",
    )

    probability_evidence = (
        probability_evidence_given_hypothesis * prior
        + probability_evidence_given_not_hypothesis * (1.0 - prior)
    )
    if probability_evidence == 0.0:
        raise ValueError("the marginal probability of evidence must be positive.")

    return (
        probability_evidence_given_hypothesis
        * prior
        / probability_evidence
    )


def main() -> None:
    """Run small deterministic examples as a smoke test."""
    die_values = [1, 2, 3, 4, 5, 6]
    die_probabilities = [1 / 6] * 6

    print(f"Expected fair-die value: {expected_value(die_values, die_probabilities):.4f}")
    print(f"Fair-die variance:       {variance(die_values, die_probabilities):.4f}")

    fraud_posterior = binary_bayes_update(
        prior=0.01,
        probability_evidence_given_hypothesis=0.90,
        probability_evidence_given_not_hypothesis=0.05,
    )
    print(f"P(Fraud | Alert):        {fraud_posterior:.4f}")

    is_fraud = [True, False, True, False, False]
    has_alert = [True, True, False, False, True]
    empirical = conditional_probability(is_fraud, has_alert)
    print(f"Empirical P(Fraud | Alert): {empirical:.4f}")


if __name__ == "__main__":
    main()

