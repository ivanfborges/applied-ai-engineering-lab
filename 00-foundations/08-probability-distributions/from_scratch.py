"""Educational maximum-likelihood estimators for common distributions.

The estimators expose the formulas with standard-library operations. They are
not production-grade replacements for statistical libraries, which provide
more complete diagnostics, confidence intervals, numerical methods, and
support for censoring or weighted data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributionEstimate:
    """Hold the name and fitted parameters of one distribution."""

    name: str
    parameters: dict[str, float]


def _finite_values(
    samples: Sequence[float],
    sample_name: str,
) -> list[float]:
    """Convert a non-empty sequence to validated finite floats."""
    if len(samples) == 0:
        raise ValueError(f"{sample_name} cannot be empty.")
    values = [float(value) for value in samples]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{sample_name} must contain only finite values.")
    return values


def _mean(values: Sequence[float]) -> float:
    """Compute an arithmetic mean with accurate floating-point summation."""
    return math.fsum(values) / len(values)


def _mle_variance(values: Sequence[float], mean: float) -> float:
    """Compute the population-form variance used by the Gaussian MLE."""
    squared_deviations = ((value - mean) ** 2 for value in values)
    return math.fsum(squared_deviations) / len(values)


def fit_bernoulli(samples: Sequence[float]) -> DistributionEstimate:
    """Estimate Bernoulli p as the observed fraction of ones."""
    values = _finite_values(samples, "Bernoulli samples")
    if any(value not in (0.0, 1.0) for value in values):
        raise ValueError("Bernoulli samples must contain only 0 and 1.")
    return DistributionEstimate("Bernoulli", {"p": _mean(values)})


def fit_binomial(
    counts: Sequence[float],
    number_of_trials: int,
) -> DistributionEstimate:
    """Estimate Binomial p when the number of trials is known."""
    if (
        isinstance(number_of_trials, bool)
        or not isinstance(number_of_trials, int)
        or number_of_trials <= 0
    ):
        raise ValueError("number_of_trials must be a positive integer.")

    values = _finite_values(counts, "Binomial counts")
    if any(
        not value.is_integer() or not 0.0 <= value <= number_of_trials
        for value in values
    ):
        raise ValueError(
            "Binomial counts must be integers from 0 to number_of_trials."
        )
    return DistributionEstimate(
        "Binomial",
        {"n": float(number_of_trials), "p": _mean(values) / number_of_trials},
    )


def fit_poisson(counts: Sequence[float]) -> DistributionEstimate:
    """Estimate the Poisson rate as the mean observed count."""
    values = _finite_values(counts, "Poisson counts")
    if any(not value.is_integer() or value < 0.0 for value in values):
        raise ValueError("Poisson counts must be non-negative integers.")
    return DistributionEstimate("Poisson", {"lambda": _mean(values)})


def fit_exponential(
    waiting_times: Sequence[float],
) -> DistributionEstimate:
    """Estimate the Exponential rate as one over mean waiting time."""
    values = _finite_values(waiting_times, "Exponential waiting times")
    if any(value < 0.0 for value in values):
        raise ValueError("Exponential waiting times cannot be negative.")
    mean_wait = _mean(values)
    if mean_wait == 0.0:
        raise ValueError("Mean waiting time must be greater than zero.")
    return DistributionEstimate("Exponential", {"lambda": 1.0 / mean_wait})


def fit_normal(samples: Sequence[float]) -> DistributionEstimate:
    """Estimate Normal mean and MLE variance (denominator n, not n-1)."""
    values = _finite_values(samples, "Normal samples")
    mean = _mean(values)
    variance = _mle_variance(values, mean)
    return DistributionEstimate(
        "Normal",
        {"mu": mean, "variance": variance, "sigma": math.sqrt(variance)},
    )


def fit_lognormal(samples: Sequence[float]) -> DistributionEstimate:
    """Fit Normal MLE parameters to log(samples)."""
    values = _finite_values(samples, "Log-normal samples")
    if any(value <= 0.0 for value in values):
        raise ValueError("Log-normal samples must be strictly positive.")

    log_values = [math.log(value) for value in values]
    log_mean = _mean(log_values)
    log_variance = _mle_variance(log_values, log_mean)
    original_scale_mean = math.exp(log_mean + log_variance / 2.0)
    return DistributionEstimate(
        "Log-normal",
        {
            "log_mu": log_mean,
            "log_variance": log_variance,
            "log_sigma": math.sqrt(log_variance),
            "estimated_original_scale_mean": original_scale_mean,
        },
    )


def main() -> None:
    """Fit small deterministic synthetic samples as an executable smoke test."""
    synthetic_datasets = [
        fit_bernoulli([1, 1, 0, 1, 0, 1, 1, 1]),
        fit_binomial([6, 8, 7, 5, 9, 6], number_of_trials=20),
        fit_poisson([2, 5, 4, 3, 6, 4, 5, 3]),
        fit_exponential([0.4, 1.8, 2.2, 0.9, 3.1, 1.6]),
        fit_normal([7.2, 9.4, 10.1, 11.3, 12.0]),
        fit_lognormal([3.2, 4.1, 5.8, 8.7, 13.5, 21.0]),
    ]

    print("Dataset: small deterministic synthetic observations")
    for estimate in synthetic_datasets:
        print(f"\n{estimate.name}")
        for parameter, value in estimate.parameters.items():
            print(f"  {parameter}: {value:.4f}")


if __name__ == "__main__":
    main()
