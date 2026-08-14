"""Validated numerical utilities for the Day 12 visual laboratory."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


ALTERNATIVES = {"two-sided", "greater", "less"}


@dataclass(frozen=True)
class MeanTestResult:
    """Normal-reference one-sample mean-test result."""

    standard_error: float
    statistic: float
    p_value: float
    critical_lower: float
    critical_upper: float
    reject: bool
    alternative: str


@dataclass(frozen=True)
class ErrorRates:
    """One-sided normal-reference Type I, Type II, and power values."""

    standard_error: float
    critical_mean: float
    alpha: float
    beta: float
    power: float


@dataclass(frozen=True)
class PairedVisualSummary:
    """Paired and independent analyses of the same score vectors."""

    sample_size: int
    baseline_mean: float
    candidate_mean: float
    mean_difference: float
    difference_sd: float
    paired_standard_error: float
    independent_standard_error: float
    confidence_interval: tuple[float, float]
    paired_t_statistic: float
    paired_p_value: float
    independent_t_statistic: float
    independent_p_value: float
    cohens_dz: float


@dataclass(frozen=True)
class SignFlipDistribution:
    """Monte Carlo sign-flip result with its simulated null statistics."""

    observed_mean: float
    p_value: float
    extreme_count: int
    null_statistics: np.ndarray


@dataclass(frozen=True)
class MultipleTestingSummary:
    """All-null multiple-testing simulation and one displayed family."""

    example_p_values: np.ndarray
    example_uncorrected: np.ndarray
    example_bonferroni: np.ndarray
    example_bh: np.ndarray
    familywise_rates: dict[str, float]
    mean_discoveries: dict[str, float]


@dataclass(frozen=True)
class ConfidenceTestConnection:
    """Matching normal-reference confidence interval and two-sided test."""

    confidence_interval: tuple[float, float]
    statistic: float
    p_value: float
    reject: bool
    alpha: float


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be finite and strictly between zero and one.")
    return value


def _validate_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _validate_integer(
    value: int,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            raise ValueError(f"{name} must be at least {minimum}.")
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _validate_alternative(alternative: str) -> str:
    if alternative not in ALTERNATIVES:
        allowed = ", ".join(sorted(ALTERNATIVES))
        raise ValueError(f"alternative must be one of: {allowed}.")
    return alternative


def _finite_vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must contain at least two observations.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite observations.")
    return array


def normal_mean_test(
    observed_effect: float,
    sample_size: int,
    standard_deviation: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> MeanTestResult:
    """Return a known-SD normal-reference test for ``H0: mean = 0``."""
    observed_effect = float(observed_effect)
    if not math.isfinite(observed_effect):
        raise ValueError("observed_effect must be finite.")
    sample_size = _validate_integer(sample_size, "sample_size", minimum=2)
    standard_deviation = _validate_positive(standard_deviation, "standard_deviation")
    alpha = _validate_probability(alpha, "alpha")
    alternative = _validate_alternative(alternative)

    standard_error = standard_deviation / math.sqrt(sample_size)
    statistic = observed_effect / standard_error
    if alternative == "two-sided":
        critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
        p_value = float(2.0 * stats.norm.sf(abs(statistic)))
        critical_lower, critical_upper = -critical, critical
    elif alternative == "greater":
        critical = float(stats.norm.ppf(1.0 - alpha))
        p_value = float(stats.norm.sf(statistic))
        critical_lower, critical_upper = critical, math.inf
    else:
        critical = float(stats.norm.ppf(alpha))
        p_value = float(stats.norm.cdf(statistic))
        critical_lower, critical_upper = -math.inf, critical

    return MeanTestResult(
        standard_error=standard_error,
        statistic=statistic,
        p_value=p_value,
        critical_lower=critical_lower,
        critical_upper=critical_upper,
        reject=p_value < alpha,
        alternative=alternative,
    )


def normal_error_rates(
    effect: float,
    sample_size: int,
    standard_deviation: float,
    alpha: float = 0.05,
) -> ErrorRates:
    """Calculate one-sided normal-reference errors for ``H1: mean > 0``."""
    effect = float(effect)
    if not math.isfinite(effect) or effect < 0.0:
        raise ValueError("effect must be finite and non-negative.")
    sample_size = _validate_integer(sample_size, "sample_size", minimum=2)
    standard_deviation = _validate_positive(standard_deviation, "standard_deviation")
    alpha = _validate_probability(alpha, "alpha")

    standard_error = standard_deviation / math.sqrt(sample_size)
    critical_mean = float(stats.norm.ppf(1.0 - alpha) * standard_error)
    beta = float(stats.norm.cdf(critical_mean, loc=effect, scale=standard_error))
    power = 1.0 - beta
    return ErrorRates(
        standard_error=standard_error,
        critical_mean=critical_mean,
        alpha=alpha,
        beta=beta,
        power=power,
    )


def one_sample_t_power(
    sample_size: int,
    standardized_effect: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """Calculate one-sample t-test power using a noncentral t distribution."""
    sample_size = _validate_integer(sample_size, "sample_size", minimum=2)
    standardized_effect = float(standardized_effect)
    if not math.isfinite(standardized_effect):
        raise ValueError("standardized_effect must be finite.")
    alpha = _validate_probability(alpha, "alpha")
    alternative = _validate_alternative(alternative)

    degrees_of_freedom = sample_size - 1
    noncentrality = standardized_effect * math.sqrt(sample_size)
    if alternative == "two-sided":
        critical = stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom)
        power = stats.nct.sf(critical, degrees_of_freedom, -noncentrality)
        power += stats.nct.sf(critical, degrees_of_freedom, noncentrality)
    elif alternative == "greater":
        critical = stats.t.ppf(1.0 - alpha, degrees_of_freedom)
        power = stats.nct.sf(critical, degrees_of_freedom, noncentrality)
    else:
        critical = stats.t.ppf(alpha, degrees_of_freedom)
        power = stats.nct.cdf(critical, degrees_of_freedom, noncentrality)
    return float(np.clip(power, 0.0, 1.0))


def power_curve(
    sample_sizes: Sequence[int],
    standardized_effect: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> np.ndarray:
    """Calculate a noncentral-t power value for each sample size."""
    sizes = np.asarray(sample_sizes)
    if sizes.ndim != 1 or sizes.size == 0:
        raise ValueError("sample_sizes must be a non-empty one-dimensional sequence.")
    return np.asarray(
        [
            one_sample_t_power(int(size), standardized_effect, alpha, alternative)
            for size in sizes
        ],
        dtype=float,
    )


def power_surface(
    sample_sizes: Sequence[int],
    standardized_effects: Sequence[float],
    alpha: float = 0.05,
) -> np.ndarray:
    """Return two-sided one-sample t-test power on an effect-by-size grid."""
    sizes = np.asarray(sample_sizes, dtype=int)
    effects = np.asarray(standardized_effects, dtype=float)
    if sizes.ndim != 1 or sizes.size == 0 or np.any(sizes < 2):
        raise ValueError("sample_sizes must be one-dimensional with values >= 2.")
    if effects.ndim != 1 or effects.size == 0 or not np.all(np.isfinite(effects)):
        raise ValueError("standardized_effects must be a finite one-dimensional array.")
    if np.any(effects < 0.0):
        raise ValueError("standardized_effects must be non-negative.")
    alpha = _validate_probability(alpha, "alpha")

    degrees_of_freedom = sizes - 1
    critical = stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom)
    noncentrality = effects[:, None] * np.sqrt(sizes)[None, :]
    lower = stats.nct.sf(
        critical[None, :], degrees_of_freedom[None, :], -noncentrality
    )
    upper = stats.nct.sf(
        critical[None, :], degrees_of_freedom[None, :], noncentrality
    )
    return np.clip(lower + upper, 0.0, 1.0)


def simulate_t_experiments(
    *,
    true_mean: float,
    sample_size: int,
    standard_deviation: float,
    simulations: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    seed: int = 42,
) -> pd.DataFrame:
    """Vectorize repeated one-sample t tests against a zero null mean."""
    true_mean = float(true_mean)
    if not math.isfinite(true_mean):
        raise ValueError("true_mean must be finite.")
    sample_size = _validate_integer(sample_size, "sample_size", minimum=2, maximum=500)
    simulations = _validate_integer(simulations, "simulations", minimum=1, maximum=10_000)
    standard_deviation = _validate_positive(standard_deviation, "standard_deviation")
    alpha = _validate_probability(alpha, "alpha")
    alternative = _validate_alternative(alternative)

    rng = np.random.default_rng(seed)
    samples = rng.normal(
        loc=true_mean,
        scale=standard_deviation,
        size=(simulations, sample_size),
    )
    sample_means = np.mean(samples, axis=1)
    sample_sds = np.std(samples, axis=1, ddof=1)
    standard_errors = sample_sds / math.sqrt(sample_size)
    statistics = sample_means / standard_errors
    degrees_of_freedom = sample_size - 1
    if alternative == "two-sided":
        p_values = 2.0 * stats.t.sf(np.abs(statistics), degrees_of_freedom)
    elif alternative == "greater":
        p_values = stats.t.sf(statistics, degrees_of_freedom)
    else:
        p_values = stats.t.cdf(statistics, degrees_of_freedom)

    return pd.DataFrame(
        {
            "experiment": np.arange(1, simulations + 1),
            "sample_mean": sample_means,
            "test_statistic": statistics,
            "p_value": p_values,
            "reject": p_values < alpha,
        }
    )


def paired_visual_summary(
    baseline: Iterable[float],
    candidate: Iterable[float],
    *,
    alpha: float = 0.05,
) -> PairedVisualSummary:
    """Compare paired inference with an intentionally unpaired analysis."""
    baseline_array = _finite_vector(baseline, "baseline")
    candidate_array = _finite_vector(candidate, "candidate")
    if baseline_array.size != candidate_array.size:
        raise ValueError("baseline and candidate must have the same length.")
    alpha = _validate_probability(alpha, "alpha")

    differences = candidate_array - baseline_array
    sample_size = int(differences.size)
    mean_difference = float(np.mean(differences))
    difference_sd = float(np.std(differences, ddof=1))
    if difference_sd == 0.0:
        raise ValueError("paired differences must have non-zero sample variance.")
    paired_standard_error = difference_sd / math.sqrt(sample_size)
    critical_t = float(stats.t.ppf(1.0 - alpha / 2.0, sample_size - 1))
    confidence_interval = (
        mean_difference - critical_t * paired_standard_error,
        mean_difference + critical_t * paired_standard_error,
    )
    paired_result = stats.ttest_rel(candidate_array, baseline_array)
    independent_result = stats.ttest_ind(
        candidate_array, baseline_array, equal_var=False
    )
    independent_standard_error = math.sqrt(
        np.var(candidate_array, ddof=1) / sample_size
        + np.var(baseline_array, ddof=1) / sample_size
    )

    return PairedVisualSummary(
        sample_size=sample_size,
        baseline_mean=float(np.mean(baseline_array)),
        candidate_mean=float(np.mean(candidate_array)),
        mean_difference=mean_difference,
        difference_sd=difference_sd,
        paired_standard_error=paired_standard_error,
        independent_standard_error=independent_standard_error,
        confidence_interval=confidence_interval,
        paired_t_statistic=float(paired_result.statistic),
        paired_p_value=float(paired_result.pvalue),
        independent_t_statistic=float(independent_result.statistic),
        independent_p_value=float(independent_result.pvalue),
        cohens_dz=mean_difference / difference_sd,
    )


def sign_flip_distribution(
    differences: Iterable[float],
    *,
    permutations: int = 20_000,
    seed: int = 42,
) -> SignFlipDistribution:
    """Construct a two-sided Monte Carlo null distribution by sign flipping."""
    sample = _finite_vector(differences, "differences")
    permutations = _validate_integer(
        permutations, "permutations", minimum=1, maximum=100_000
    )
    observed_mean = float(np.mean(sample))
    rng = np.random.default_rng(seed)
    null_statistics = np.empty(permutations, dtype=float)

    start = 0
    while start < permutations:
        stop = min(start + 5_000, permutations)
        signs = rng.choice((-1.0, 1.0), size=(stop - start, sample.size))
        null_statistics[start:stop] = np.mean(signs * sample, axis=1)
        start = stop

    extreme_count = int(
        np.count_nonzero(np.abs(null_statistics) >= abs(observed_mean) - 1e-12)
    )
    p_value = (extreme_count + 1) / (permutations + 1)
    return SignFlipDistribution(
        observed_mean=observed_mean,
        p_value=p_value,
        extreme_count=extreme_count,
        null_statistics=null_statistics,
    )


def benjamini_hochberg(p_values: Iterable[float], alpha: float = 0.05) -> np.ndarray:
    """Return Benjamini-Hochberg rejection decisions in original order."""
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and between zero and one.")
    alpha = _validate_probability(alpha, "alpha")

    order = np.argsort(values)
    sorted_values = values[order]
    thresholds = alpha * np.arange(1, values.size + 1) / values.size
    passing = np.flatnonzero(sorted_values <= thresholds)
    rejected = np.zeros(values.size, dtype=bool)
    if passing.size:
        cutoff = sorted_values[passing[-1]]
        rejected = values <= cutoff
    return rejected


def simulate_multiple_testing(
    *,
    hypotheses: int,
    repetitions: int,
    alpha: float = 0.05,
    seed: int = 42,
) -> MultipleTestingSummary:
    """Simulate independent valid p-values when every null is true."""
    hypotheses = _validate_integer(hypotheses, "hypotheses", minimum=1, maximum=500)
    repetitions = _validate_integer(repetitions, "repetitions", minimum=1, maximum=10_000)
    alpha = _validate_probability(alpha, "alpha")
    rng = np.random.default_rng(seed)
    p_values = rng.uniform(size=(repetitions, hypotheses))

    uncorrected = p_values < alpha
    bonferroni = p_values < alpha / hypotheses
    bh = np.vstack([benjamini_hochberg(row, alpha) for row in p_values])
    methods = {
        "No correction": uncorrected,
        "Bonferroni": bonferroni,
        "Benjamini-Hochberg": bh,
    }
    return MultipleTestingSummary(
        example_p_values=p_values[0],
        example_uncorrected=uncorrected[0],
        example_bonferroni=bonferroni[0],
        example_bh=bh[0],
        familywise_rates={
            name: float(np.mean(np.any(decisions, axis=1)))
            for name, decisions in methods.items()
        },
        mean_discoveries={
            name: float(np.mean(np.sum(decisions, axis=1)))
            for name, decisions in methods.items()
        },
    )


def confidence_test_connection(
    estimate: float,
    standard_error: float,
    confidence: float = 0.95,
) -> ConfidenceTestConnection:
    """Match a normal confidence interval with a two-sided zero-null test."""
    estimate = float(estimate)
    if not math.isfinite(estimate):
        raise ValueError("estimate must be finite.")
    standard_error = _validate_positive(standard_error, "standard_error")
    confidence = _validate_probability(confidence, "confidence")
    alpha = 1.0 - confidence
    critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
    interval = (
        estimate - critical * standard_error,
        estimate + critical * standard_error,
    )
    statistic = estimate / standard_error
    p_value = float(2.0 * stats.norm.sf(abs(statistic)))
    return ConfidenceTestConnection(
        confidence_interval=interval,
        statistic=statistic,
        p_value=p_value,
        reject=p_value < alpha,
        alpha=alpha,
    )


def simulate_confidence_intervals(
    *,
    true_mean: float,
    sample_size: int,
    intervals: int,
    confidence: float = 0.95,
    standard_deviation: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate Student-t intervals for repeated normal samples."""
    true_mean = float(true_mean)
    if not math.isfinite(true_mean):
        raise ValueError("true_mean must be finite.")
    sample_size = _validate_integer(sample_size, "sample_size", minimum=2, maximum=500)
    intervals = _validate_integer(intervals, "intervals", minimum=1, maximum=500)
    confidence = _validate_probability(confidence, "confidence")
    standard_deviation = _validate_positive(standard_deviation, "standard_deviation")

    rng = np.random.default_rng(seed)
    samples = rng.normal(
        loc=true_mean,
        scale=standard_deviation,
        size=(intervals, sample_size),
    )
    means = np.mean(samples, axis=1)
    standard_errors = np.std(samples, axis=1, ddof=1) / math.sqrt(sample_size)
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, sample_size - 1))
    lower = means - critical * standard_errors
    upper = means + critical * standard_errors
    covered = (lower <= true_mean) & (true_mean <= upper)
    return pd.DataFrame(
        {
            "interval": np.arange(1, intervals + 1),
            "estimate": means,
            "lower": lower,
            "upper": upper,
            "covered": covered,
        }
    )


def practical_significance_scenarios(
    practical_threshold: float,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Return four explicitly synthetic significance scenarios."""
    practical_threshold = _validate_positive(
        practical_threshold, "practical_threshold"
    )
    confidence = _validate_probability(confidence, "confidence")
    estimates = np.array(
        [2.5 * practical_threshold, 0.25 * practical_threshold,
         2.5 * practical_threshold, 0.25 * practical_threshold]
    )
    standard_errors = np.array(
        [0.6 * practical_threshold, 0.05 * practical_threshold,
         1.5 * practical_threshold, 0.5 * practical_threshold]
    )
    critical = float(stats.norm.ppf(0.5 + confidence / 2.0))
    p_values = 2.0 * stats.norm.sf(np.abs(estimates / standard_errors))
    return pd.DataFrame(
        {
            "scenario": [
                "A · large, detectable",
                "B · tiny, detectable",
                "C · large, uncertain",
                "D · tiny, uncertain",
            ],
            "estimate": estimates,
            "standard_error": standard_errors,
            "lower": estimates - critical * standard_errors,
            "upper": estimates + critical * standard_errors,
            "p_value": p_values,
            "statistically_significant": p_values < 1.0 - confidence,
            "point_estimate_exceeds_threshold": estimates >= practical_threshold,
        }
    )
