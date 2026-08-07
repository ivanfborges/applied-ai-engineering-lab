"""Educational implementations of descriptive statistics.

The functions make definitions and edge cases explicit. They are useful for
study and small checks, not replacements for well-tested statistical libraries
that provide weighted estimators, missing-value policies, and bias corrections.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _finite_values(
    data: Iterable[float],
    *,
    name: str = "data",
    minimum_size: int = 1,
) -> list[float]:
    """Return validated finite floats from an iterable."""
    if isinstance(data, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of numbers.")
    try:
        values = [float(value) for value in data]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an iterable of numbers.") from error
    if len(values) < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} value(s).")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _validate_ddof(ddof: int, sample_size: int) -> None:
    """Validate a delta degrees of freedom for a denominator n - ddof."""
    if isinstance(ddof, bool) or not isinstance(ddof, int):
        raise ValueError("ddof must be an integer.")
    if not 0 <= ddof < sample_size:
        raise ValueError("ddof must satisfy 0 <= ddof < number of values.")


def mean(data: Iterable[float]) -> float:
    """Compute the arithmetic mean."""
    values = _finite_values(data)
    return math.fsum(values) / len(values)


def median(data: Iterable[float]) -> float:
    """Compute the median, averaging the middle pair for an even sample."""
    values = sorted(_finite_values(data))
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def quantile(data: Iterable[float], probability: float) -> float:
    """Compute a linearly interpolated quantile at probability in [0, 1].

    This matches NumPy and pandas' default linear method for one-dimensional
    unweighted data. Other valid quantile conventions can return different
    values for small samples.
    """
    values = sorted(_finite_values(data))
    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and between 0 and 1.")

    position = (len(values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return values[lower_index]
    weight = position - lower_index
    return values[lower_index] * (1.0 - weight) + values[upper_index] * weight


def variance(data: Iterable[float], *, ddof: int = 0) -> float:
    """Compute variance with denominator n - ddof."""
    values = _finite_values(data)
    _validate_ddof(ddof, len(values))
    center = math.fsum(values) / len(values)
    squared_deviations = ((value - center) ** 2 for value in values)
    return math.fsum(squared_deviations) / (len(values) - ddof)


def standard_deviation(data: Iterable[float], *, ddof: int = 0) -> float:
    """Compute the square root of variance."""
    return math.sqrt(variance(data, ddof=ddof))


def interquartile_range(data: Iterable[float]) -> float:
    """Return Q3 - Q1 using linear quantiles."""
    values = _finite_values(data)
    return quantile(values, 0.75) - quantile(values, 0.25)


def median_absolute_deviation(data: Iterable[float]) -> float:
    """Return the unscaled median absolute deviation from the median."""
    values = _finite_values(data)
    center = median(values)
    return median(abs(value - center) for value in values)


def iqr_bounds(
    data: Iterable[float],
    *,
    multiplier: float = 1.5,
) -> tuple[float, float]:
    """Return Tukey-style lower and upper fences for potential outliers."""
    values = _finite_values(data)
    multiplier = float(multiplier)
    if not math.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("multiplier must be a finite non-negative number.")
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    spread = q3 - q1
    return q1 - multiplier * spread, q3 + multiplier * spread


def covariance(
    first: Iterable[float],
    second: Iterable[float],
    *,
    ddof: int = 1,
) -> float:
    """Compute covariance with denominator n - ddof."""
    x = _finite_values(first, name="first", minimum_size=2)
    y = _finite_values(second, name="second", minimum_size=2)
    if len(x) != len(y):
        raise ValueError("first and second must have the same length.")
    _validate_ddof(ddof, len(x))
    x_mean = math.fsum(x) / len(x)
    y_mean = math.fsum(y) / len(y)
    cross_deviations = (
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y, strict=True)
    )
    return math.fsum(cross_deviations) / (len(x) - ddof)


def pearson_correlation(
    first: Iterable[float],
    second: Iterable[float],
) -> float:
    """Compute Pearson's linear correlation coefficient."""
    x = _finite_values(first, name="first", minimum_size=2)
    y = _finite_values(second, name="second", minimum_size=2)
    if len(x) != len(y):
        raise ValueError("first and second must have the same length.")

    x_mean = math.fsum(x) / len(x)
    y_mean = math.fsum(y) / len(y)
    x_deviations = [value - x_mean for value in x]
    y_deviations = [value - y_mean for value in y]
    numerator = math.fsum(
        x_value * y_value
        for x_value, y_value in zip(x_deviations, y_deviations, strict=True)
    )
    denominator = math.sqrt(
        math.fsum(value**2 for value in x_deviations)
        * math.fsum(value**2 for value in y_deviations)
    )
    if denominator == 0.0:
        raise ValueError("correlation is undefined for a constant input.")
    return numerator / denominator


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Assign one-based average ranks, including tied observations."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        stop = start + 1
        while stop < len(indexed) and indexed[stop][1] == indexed[start][1]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[indexed[position][0]] = average_rank
        start = stop
    return ranks


def spearman_correlation(
    first: Iterable[float],
    second: Iterable[float],
) -> float:
    """Compute Spearman correlation as Pearson correlation of average ranks."""
    x = _finite_values(first, name="first", minimum_size=2)
    y = _finite_values(second, name="second", minimum_size=2)
    if len(x) != len(y):
        raise ValueError("first and second must have the same length.")
    return pearson_correlation(_average_ranks(x), _average_ranks(y))


def population_skewness(data: Iterable[float]) -> float:
    """Compute the standardized third population central moment."""
    values = _finite_values(data, minimum_size=2)
    center = math.fsum(values) / len(values)
    second_moment = math.fsum((value - center) ** 2 for value in values) / len(
        values
    )
    if second_moment == 0.0:
        raise ValueError("skewness is undefined for a constant input.")
    third_moment = math.fsum((value - center) ** 3 for value in values) / len(
        values
    )
    return third_moment / second_moment**1.5


def population_excess_kurtosis(data: Iterable[float]) -> float:
    """Compute the fourth standardized population moment minus three."""
    values = _finite_values(data, minimum_size=2)
    center = math.fsum(values) / len(values)
    second_moment = math.fsum((value - center) ** 2 for value in values) / len(
        values
    )
    if second_moment == 0.0:
        raise ValueError("kurtosis is undefined for a constant input.")
    fourth_moment = math.fsum((value - center) ** 4 for value in values) / len(
        values
    )
    return fourth_moment / second_moment**2 - 3.0


def main() -> None:
    """Print a small deterministic demonstration of the definitions."""
    values = [1.0, 2.0, 3.0, 4.0, 20.0]
    monotonic_x = [0.0, 1.0, 2.0, 3.0, 4.0]
    monotonic_y = [math.exp(value) for value in monotonic_x]

    print("Dataset: small deterministic synthetic observations")
    print(f"Values: {values}")
    print(f"Mean: {mean(values):.4f}")
    print(f"Median: {median(values):.4f}")
    print(f"Population variance (ddof=0): {variance(values):.4f}")
    print(f"Sample variance (ddof=1): {variance(values, ddof=1):.4f}")
    print(f"Sample standard deviation: {standard_deviation(values, ddof=1):.4f}")
    print(f"IQR: {interquartile_range(values):.4f}")
    print(f"MAD (unscaled): {median_absolute_deviation(values):.4f}")
    print(f"Population skewness: {population_skewness(values):.4f}")
    print(
        "Population excess kurtosis: "
        f"{population_excess_kurtosis(values):.4f}"
    )
    print("\nMonotonic nonlinear relationship: y = exp(x)")
    print(f"Pearson correlation: {pearson_correlation(monotonic_x, monotonic_y):.4f}")
    print(
        f"Spearman correlation: {spearman_correlation(monotonic_x, monotonic_y):.4f}"
    )


if __name__ == "__main__":
    main()
