"""Educational first-principles helpers for sampling estimators."""

from __future__ import annotations

import math
from collections.abc import Iterable


def _finite_values(
    values: Iterable[float],
    *,
    name: str,
) -> list[float]:
    converted = [float(value) for value in values]
    if not converted:
        raise ValueError(f"{name} must contain at least one value.")
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain only finite values.")
    return converted


def estimator_statistics(
    estimates: Iterable[float],
    true_parameter: float,
) -> dict[str, float]:
    """Return empirical bias, population variance, and MSE.

    The repeated estimates are treated as the complete simulated sampling
    distribution, so variance uses denominator ``n``. This makes the empirical
    MSE decomposition exact up to floating-point precision.
    """
    values = _finite_values(estimates, name="estimates")
    true_parameter = float(true_parameter)
    if not math.isfinite(true_parameter):
        raise ValueError("true_parameter must be finite.")

    expected_estimate = math.fsum(values) / len(values)
    bias = expected_estimate - true_parameter
    variance = math.fsum(
        (estimate - expected_estimate) ** 2 for estimate in values
    ) / len(values)
    mse = math.fsum(
        (estimate - true_parameter) ** 2 for estimate in values
    ) / len(values)
    bias_squared = bias**2
    return {
        "expected_estimate": expected_estimate,
        "bias": bias,
        "variance": variance,
        "bias_squared": bias_squared,
        "mse": mse,
        "variance_plus_bias_squared": variance + bias_squared,
    }


def weighted_mean(
    values: Iterable[float],
    weights: Iterable[float],
) -> float:
    """Return a normalized non-negative weighted mean."""
    finite_values = _finite_values(values, name="values")
    finite_weights = _finite_values(weights, name="weights")
    if len(finite_values) != len(finite_weights):
        raise ValueError("values and weights must have the same length.")
    if any(weight < 0.0 for weight in finite_weights):
        raise ValueError("weights must be non-negative.")
    weight_sum = math.fsum(finite_weights)
    if weight_sum <= 0.0:
        raise ValueError("at least one weight must be positive.")
    return math.fsum(
        value * weight
        for value, weight in zip(finite_values, finite_weights, strict=True)
    ) / weight_sum


def effective_sample_size(weights: Iterable[float]) -> float:
    """Return Kish's effective sample size approximation for weights."""
    finite_weights = _finite_values(weights, name="weights")
    if any(weight < 0.0 for weight in finite_weights):
        raise ValueError("weights must be non-negative.")
    weight_sum = math.fsum(finite_weights)
    if weight_sum <= 0.0:
        raise ValueError("at least one weight must be positive.")
    squared_weight_sum = math.fsum(weight**2 for weight in finite_weights)
    return weight_sum**2 / squared_weight_sum


def sample_mean_variance(
    population_variance: float,
    sample_size: int,
    *,
    population_size: int | None = None,
) -> float:
    """Return the variance of a simple-random-sample mean.

    If ``population_size`` is supplied, sampling is assumed to be without
    replacement and the finite population correction is applied.
    """
    population_variance = float(population_variance)
    if not math.isfinite(population_variance) or population_variance < 0.0:
        raise ValueError("population_variance must be finite and non-negative.")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    variance = population_variance / sample_size
    if population_size is None:
        return variance
    if isinstance(population_size, bool) or not isinstance(population_size, int):
        raise TypeError("population_size must be an integer.")
    if population_size <= 1:
        raise ValueError("population_size must be greater than one.")
    if sample_size > population_size:
        raise ValueError("sample_size cannot exceed population_size.")
    correction = (population_size - sample_size) / (population_size - 1)
    return variance * correction


def main() -> None:
    """Print a small deterministic demonstration of the formulas."""
    estimates = [9.0, 10.0, 10.0, 11.0]
    statistics = estimator_statistics(estimates, true_parameter=10.0)
    equal_weights = [1.0, 1.0, 1.0, 1.0]
    unequal_weights = [1.0, 1.0, 1.0, 9.0]

    print("Repeated estimates:", estimates)
    for name, value in statistics.items():
        print(f"{name}: {value:.6f}")
    print(f"Equal-weight ESS: {effective_sample_size(equal_weights):.3f}")
    print(f"Unequal-weight ESS: {effective_sample_size(unequal_weights):.3f}")
    print(
        "Mean variance with finite population correction: "
        f"{sample_mean_variance(25.0, 20, population_size=100):.6f}"
    )


if __name__ == "__main__":
    main()
