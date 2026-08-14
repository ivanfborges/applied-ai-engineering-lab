"""Applied uncertainty experiments for dependence and model decisions."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from .visual_utils import (
    BLUE,
    GRAY,
    GREEN,
    INK,
    LIGHT_BLUE,
    LIGHT_GRAY,
    LIGHT_ORANGE,
    ORANGE,
    RED,
    AssetResult,
    save_figure,
)


@dataclass(frozen=True)
class DependenceResult:
    """Sampling variability under equal row counts but different dependence."""

    independent_means: np.ndarray
    clustered_means: np.ndarray
    naive_se: float
    independent_empirical_se: float
    clustered_empirical_se: float
    cluster_aware_se: float
    intraclass_correlation: float


@dataclass(frozen=True)
class ModelComparisonResult:
    """Independent mean estimates and their difference interval."""

    mean_a: float
    mean_b: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    difference: float
    difference_ci: tuple[float, float]


@dataclass(frozen=True)
class PracticalSignificanceResult:
    """A precisely estimated small synthetic effect."""

    sample_size_per_group: int
    difference: float
    confidence_interval: tuple[float, float]
    practical_threshold: float


def calculate_dependence_results(
    *,
    trials: int = 2_000,
    users: int = 100,
    observations_per_user: int = 50,
    user_sd: float = 2.0,
    residual_sd: float = 1.0,
    seed: int = 11101,
) -> DependenceResult:
    """Simulate means with identical row marginals and different dependence."""
    rows = users * observations_per_user
    marginal_sd = float(np.sqrt(user_sd**2 + residual_sd**2))
    naive_se = marginal_sd / np.sqrt(rows)
    cluster_aware_se = float(
        np.sqrt(user_sd**2 / users + residual_sd**2 / rows)
    )
    rng = np.random.default_rng(seed)
    independent_means = rng.normal(0.0, naive_se, size=trials)
    clustered_means = (
        np.mean(rng.normal(0.0, user_sd, size=(trials, users)), axis=1)
        + rng.normal(0.0, residual_sd / np.sqrt(rows), size=trials)
    )
    return DependenceResult(
        independent_means=independent_means,
        clustered_means=clustered_means,
        naive_se=naive_se,
        independent_empirical_se=float(np.std(independent_means, ddof=0)),
        clustered_empirical_se=float(np.std(clustered_means, ddof=0)),
        cluster_aware_se=cluster_aware_se,
        intraclass_correlation=user_sd**2 / (user_sd**2 + residual_sd**2),
    )


def generate_dependence_demo(*, quick: bool = False) -> AssetResult:
    """Show why a nominal row count can exaggerate independent information."""
    result = calculate_dependence_results(trials=700 if quick else 2_000)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    common_bins = np.linspace(-0.75, 0.75, 70)
    axes[0].hist(
        result.independent_means,
        bins=common_bins,
        density=True,
        color=BLUE,
        alpha=0.75,
        label="5,000 independent rows",
    )
    axes[0].hist(
        result.clustered_means,
        bins=common_bins,
        density=True,
        histtype="step",
        color=ORANGE,
        linewidth=2.3,
        label="100 users × 50 correlated rows",
    )
    axes[0].axvline(0.0, color=INK, linestyle="--")
    axes[0].set(
        xlabel="Mean across 5,000 rows",
        ylabel="Density",
        title="Same row count, different sampling variability",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.6)

    labels = ("Naive IID\nformula", "Independent\nempirical", "Clustered\nempirical", "Cluster-aware\nformula")
    values = (
        result.naive_se,
        result.independent_empirical_se,
        result.clustered_empirical_se,
        result.cluster_aware_se,
    )
    bars = axes[1].bar(labels, values, color=(GRAY, BLUE, ORANGE, GREEN))
    for bar, value in zip(bars, values, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.4f}", ha="center")
    axes[1].set(ylabel="Standard error of the mean", title="Naive IID SE misses covariance")
    axes[1].grid(axis="y", alpha=0.65)
    axes[1].text(
        0.03,
        0.94,
        f"Configured ICC = {result.intraclass_correlation:.2f}",
        transform=axes[1].transAxes,
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT_ORANGE},
    )
    fig.suptitle("5,000 rows do not necessarily provide 5,000 independent observations", fontsize=15)
    fig.tight_layout()
    path = save_figure(fig, "11_independence_violation.png", quick=quick)
    return AssetResult(
        "Violation of independence",
        path,
        (
            f"Naive SE={result.naive_se:.4f}; independent empirical SE={result.independent_empirical_se:.4f}",
            f"Clustered empirical SE={result.clustered_empirical_se:.4f}; cluster-aware SE={result.cluster_aware_se:.4f}; ICC={result.intraclass_correlation:.4f}",
        ),
    )


def _welch_interval(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, tuple[float, float]]:
    """Return mean(B)-mean(A) and a Welch Student-t interval."""
    n_a, n_b = len(sample_a), len(sample_b)
    variance_a = float(np.var(sample_a, ddof=1))
    variance_b = float(np.var(sample_b, ddof=1))
    term_a, term_b = variance_a / n_a, variance_b / n_b
    standard_error = float(np.sqrt(term_a + term_b))
    degrees = (term_a + term_b) ** 2 / (
        term_a**2 / (n_a - 1) + term_b**2 / (n_b - 1)
    )
    critical = float(t.ppf(0.5 + confidence / 2.0, df=degrees))
    difference = float(np.mean(sample_b) - np.mean(sample_a))
    return difference, (
        difference - critical * standard_error,
        difference + critical * standard_error,
    )


def calculate_model_comparison(seed: int = 12101) -> ModelComparisonResult:
    """Simulate two independent continuous evaluation-score samples."""
    rng = np.random.default_rng(seed)
    sample_a = rng.normal(loc=75.0, scale=8.0, size=120)
    sample_b = rng.normal(loc=76.0, scale=8.0, size=120)
    means = (float(np.mean(sample_a)), float(np.mean(sample_b)))
    intervals: list[tuple[float, float]] = []
    for sample, mean in zip((sample_a, sample_b), means, strict=True):
        se = float(np.std(sample, ddof=1) / np.sqrt(len(sample)))
        critical = float(t.ppf(0.975, df=len(sample) - 1))
        intervals.append((mean - critical * se, mean + critical * se))
    difference, difference_ci = _welch_interval(sample_a, sample_b)
    return ModelComparisonResult(
        mean_a=means[0],
        mean_b=means[1],
        ci_a=intervals[0],
        ci_b=intervals[1],
        difference=difference,
        difference_ci=difference_ci,
    )


def generate_model_comparison(*, quick: bool = False) -> AssetResult:
    """Plot model estimates and the interval for their direct difference."""
    result = calculate_model_comparison()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    model_means = (result.mean_a, result.mean_b)
    model_intervals = (result.ci_a, result.ci_b)
    for row, (label, mean, interval, color) in enumerate(
        zip(("Model A", "Model B"), model_means, model_intervals, (BLUE, ORANGE), strict=True)
    ):
        axes[0].errorbar(
            mean,
            row,
            xerr=[[mean - interval[0]], [interval[1] - mean]],
            fmt="o",
            color=color,
            capsize=7,
            linewidth=2.6,
            markersize=9,
        )
        axes[0].text(interval[1] + 0.12, row, f"{mean:.2f} [{interval[0]:.2f}, {interval[1]:.2f}]", va="center")
    axes[0].set(
        yticks=(0, 1),
        yticklabels=("Model A", "Model B"),
        xlabel="Mean synthetic quality score with 95% CI",
        title="Point estimates do not show comparison uncertainty",
    )
    axes[0].grid(axis="x", alpha=0.7)

    lower, upper = result.difference_ci
    axes[1].errorbar(
        result.difference,
        0,
        xerr=[[result.difference - lower], [upper - result.difference]],
        fmt="D",
        color=GREEN,
        capsize=8,
        linewidth=3,
        markersize=8,
    )
    axes[1].axvline(0.0, color=INK, linestyle="--", label="No mean difference")
    axes[1].set(
        yticks=[],
        xlabel="Mean(B) − Mean(A)",
        title="Estimate the difference directly",
        ylim=(-0.35, 0.35),
    )
    axes[1].text(
        result.difference,
        0.13,
        f"difference={result.difference:.2f}\n95% CI [{lower:.2f}, {upper:.2f}]",
        ha="center",
    )
    axes[1].grid(axis="x", alpha=0.7)
    axes[1].legend(loc="lower right")
    fig.suptitle("Model B's higher point estimate is not the complete evidence", fontsize=15)
    fig.tight_layout()
    path = save_figure(fig, "12_model_comparison_ci.png", quick=quick)
    return AssetResult(
        "Two-model comparison with uncertainty",
        path,
        (
            f"Model A mean={result.mean_a:.4f}; Model B mean={result.mean_b:.4f}",
            f"Difference={result.difference:.4f}; 95% CI=[{lower:.4f}, {upper:.4f}]",
        ),
    )


def calculate_practical_significance(
    *,
    sample_size_per_group: int = 80_000,
    seed: int = 13101,
) -> PracticalSignificanceResult:
    """Simulate a detectable effect smaller than an illustrative threshold."""
    rng = np.random.default_rng(seed)
    baseline = rng.normal(loc=0.0, scale=5.0, size=sample_size_per_group)
    treatment = rng.normal(loc=0.15, scale=5.0, size=sample_size_per_group)
    difference, interval = _welch_interval(baseline, treatment)
    return PracticalSignificanceResult(
        sample_size_per_group=sample_size_per_group,
        difference=difference,
        confidence_interval=interval,
        practical_threshold=0.50,
    )


def generate_practical_significance_demo(*, quick: bool = False) -> AssetResult:
    """Separate statistical detectability from an illustrative value threshold."""
    sample_size = 80_000
    result = calculate_practical_significance(sample_size_per_group=sample_size)
    lower, upper = result.confidence_interval
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.axvspan(
        -result.practical_threshold,
        result.practical_threshold,
        color=LIGHT_GRAY,
        label="Below illustrative practical-magnitude threshold",
    )
    ax.axvline(0.0, color=INK, linestyle="--", label="Zero effect")
    ax.axvline(
        result.practical_threshold,
        color=ORANGE,
        linestyle=":",
        linewidth=2,
        label="Illustrative practical significance threshold (+0.50)",
    )
    ax.errorbar(
        result.difference,
        0,
        xerr=[[result.difference - lower], [upper - result.difference]],
        fmt="D",
        color=BLUE,
        capsize=9,
        linewidth=3,
        markersize=9,
        label="Estimated effect and 95% CI",
    )
    ax.text(
        result.difference,
        0.16,
        f"effect={result.difference:.3f}\n95% CI [{lower:.3f}, {upper:.3f}]",
        ha="center",
    )
    ax.set(
        xlim=(-0.7, 0.8),
        ylim=(-0.35, 0.38),
        yticks=[],
        xlabel="Synthetic treatment effect",
        title=f"A large sample (n={sample_size:,} per group) can resolve a very small effect",
    )
    ax.grid(axis="x", alpha=0.7)
    ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Statistically detectable does not mean practically valuable", fontsize=15)
    fig.tight_layout()
    path = save_figure(fig, "13_practical_vs_statistical_significance.png", quick=quick)
    return AssetResult(
        "Statistical vs practical significance",
        path,
        (
            f"n/group={sample_size}; effect={result.difference:.4f}; 95% CI=[{lower:.4f}, {upper:.4f}]",
            f"Illustrative practical threshold={result.practical_threshold:.2f}",
        ),
    )
