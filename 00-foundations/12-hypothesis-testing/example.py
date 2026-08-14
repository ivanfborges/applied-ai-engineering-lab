"""Compare paired AI-system scores with complementary hypothesis tests."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from from_scratch import paired_sign_flip_test, paired_statistics


@dataclass(frozen=True)
class PairedEvaluation:
    """Statistical summary of a paired baseline-candidate comparison."""

    sample_size: int
    baseline_mean: float
    candidate_mean: float
    mean_difference: float
    confidence_interval: tuple[float, float]
    t_statistic: float
    t_p_value: float
    sign_flip_p_value: float
    cohens_dz: float
    alpha: float


def generate_synthetic_scores(
    *,
    sample_size: int = 80,
    mean_improvement: float = 0.018,
    difference_sd: float = 0.04,
    seed: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate bounded paired scores for an illustrative AI evaluation."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer.")
    if sample_size < 2:
        raise ValueError("sample_size must be at least two.")
    if not math.isfinite(mean_improvement):
        raise ValueError("mean_improvement must be finite.")
    if not math.isfinite(difference_sd) or difference_sd <= 0.0:
        raise ValueError("difference_sd must be finite and positive.")

    rng = np.random.default_rng(seed)
    baseline = np.clip(rng.normal(0.72, 0.08, sample_size), 0.0, 1.0)
    differences = rng.normal(mean_improvement, difference_sd, sample_size)
    candidate = np.clip(baseline + differences, 0.0, 1.0)
    return baseline, candidate


def _paired_arrays(
    baseline: Iterable[float], candidate: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    baseline_array = np.asarray(list(baseline), dtype=float)
    candidate_array = np.asarray(list(candidate), dtype=float)
    if baseline_array.ndim != 1 or candidate_array.ndim != 1:
        raise ValueError("baseline and candidate must be one-dimensional.")
    if baseline_array.size != candidate_array.size:
        raise ValueError("baseline and candidate must have the same length.")
    if baseline_array.size < 2:
        raise ValueError("at least two paired observations are required.")
    if not np.all(np.isfinite(baseline_array)) or not np.all(
        np.isfinite(candidate_array)
    ):
        raise ValueError("scores must contain only finite observations.")
    return baseline_array, candidate_array


def analyze_paired_scores(
    baseline: Iterable[float],
    candidate: Iterable[float],
    *,
    alpha: float = 0.05,
    sign_flip_permutations: int = 50_000,
    seed: int = 12,
) -> PairedEvaluation:
    """Analyze a two-sided paired mean difference with effect uncertainty."""
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between zero and one.")
    baseline_array, candidate_array = _paired_arrays(baseline, candidate)
    differences = candidate_array - baseline_array
    descriptive = paired_statistics(differences)

    t_result = stats.ttest_rel(candidate_array, baseline_array)
    critical_t = stats.t.ppf(1.0 - alpha / 2.0, descriptive.sample_size - 1)
    margin = critical_t * descriptive.standard_error
    confidence_interval = (
        descriptive.mean_difference - margin,
        descriptive.mean_difference + margin,
    )
    sign_flip = paired_sign_flip_test(
        differences,
        permutations=sign_flip_permutations,
        seed=seed,
        exact_max_pairs=16,
    )

    return PairedEvaluation(
        sample_size=descriptive.sample_size,
        baseline_mean=float(np.mean(baseline_array)),
        candidate_mean=float(np.mean(candidate_array)),
        mean_difference=descriptive.mean_difference,
        confidence_interval=confidence_interval,
        t_statistic=float(t_result.statistic),
        t_p_value=float(t_result.pvalue),
        sign_flip_p_value=sign_flip.p_value,
        cohens_dz=descriptive.cohens_dz,
        alpha=alpha,
    )


def main() -> None:
    """Run the deterministic synthetic paired-system experiment."""
    minimum_practical_improvement = 0.02
    baseline, candidate = generate_synthetic_scores()
    result = analyze_paired_scores(baseline, candidate)
    lower, upper = result.confidence_interval

    print("Paired AI-system evaluation (synthetic data)")
    print("H0: mean(candidate - baseline) = 0")
    print("H1: mean(candidate - baseline) != 0")
    print(f"Evaluation pairs: {result.sample_size}")
    print(f"Baseline mean: {result.baseline_mean:.4f}")
    print(f"Candidate mean: {result.candidate_mean:.4f}")
    print(f"Mean paired difference: {result.mean_difference:.4f}")
    print(f"95% CI for mean difference: [{lower:.4f}, {upper:.4f}]")
    print(f"Paired t statistic: {result.t_statistic:.4f}")
    print(f"Paired t-test p-value: {result.t_p_value:.6f}")
    print(f"Sign-flip p-value: {result.sign_flip_p_value:.6f}")
    print(f"Cohen's dz: {result.cohens_dz:.4f}")
    print(f"Illustrative practical threshold: {minimum_practical_improvement:.4f}")
    print("Author review required: assess uncertainty, assumptions, and trade-offs.")


if __name__ == "__main__":
    main()
