"""Generate the static and standalone interactive assets for the Day 8 lab."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.integrate import trapezoid
from scipy import stats

from distribution_utils import (
    COLORS,
    SEED,
    calculate_empirical_statistics,
    ensure_output_directories,
    write_plotly_surfaces,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOPIC_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = TOPIC_DIRECTORY / "outputs"


def configure_matplotlib() -> None:
    """Apply a restrained, consistent technical-portfolio style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 190,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 16,
        }
    )


def _save_figure(figure: plt.Figure, output_path: Path) -> Path:
    """Save and close one Matplotlib figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved: {output_path}")
    return output_path


def _moment_annotation(
    theoretical_mean: float,
    theoretical_variance: float,
    samples: np.ndarray,
) -> str:
    """Format theoretical and empirical moments for chart annotations."""
    empirical = calculate_empirical_statistics(samples)
    return (
        f"Theory: μ={theoretical_mean:.3f}, σ²={theoretical_variance:.3f}\n"
        f"Sample: μ={empirical.mean:.3f}, σ²={empirical.variance:.3f}"
    )


def create_distribution_overview(
    output_path: Path,
    sample_size: int,
    seed: int,
) -> Path:
    """Compare empirical samples with all six theoretical PMFs or PDFs."""
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))

    p = 0.70
    samples = rng.binomial(1, p, sample_size)
    axes[0, 0].hist(
        samples,
        bins=[-0.5, 0.5, 1.5],
        density=True,
        rwidth=0.72,
        alpha=0.65,
        color=COLORS["empirical"],
        label="Empirical frequency",
    )
    bernoulli_mass = stats.bernoulli.pmf([0, 1], p)
    axes[0, 0].vlines(
        [0, 1],
        0,
        bernoulli_mass,
        color=COLORS["theoretical"],
        linewidth=2,
    )
    axes[0, 0].plot(
        [0, 1],
        bernoulli_mass,
        "o",
        color=COLORS["theoretical"],
        label="Theoretical PMF",
    )
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set(
        title=f"Bernoulli (discrete, p={p})",
        xlabel="Outcome",
        ylabel="Probability mass",
    )
    axes[0, 0].text(
        0.97,
        0.76,
        _moment_annotation(p, p * (1 - p), samples),
        transform=axes[0, 0].transAxes,
        va="top",
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    n, p = 20, 0.35
    samples = rng.binomial(n, p, sample_size)
    support = np.arange(n + 1)
    axes[0, 1].hist(
        samples,
        bins=np.arange(-0.5, n + 1.5),
        density=True,
        rwidth=0.75,
        alpha=0.65,
        color=COLORS["empirical"],
        label="Empirical frequency",
    )
    axes[0, 1].plot(
        support,
        stats.binom.pmf(support, n, p),
        "o",
        color=COLORS["theoretical"],
        label="Theoretical PMF",
    )
    axes[0, 1].set(
        title=f"Binomial (discrete, n={n}, p={p})",
        xlabel="Success count",
        ylabel="Probability mass",
    )
    axes[0, 1].text(
        0.03,
        0.95,
        _moment_annotation(n * p, n * p * (1 - p), samples),
        transform=axes[0, 1].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    rate = 4.0
    samples = rng.poisson(rate, sample_size)
    support = np.arange(0, 16)
    axes[0, 2].hist(
        samples,
        bins=np.arange(-0.5, 16.5),
        density=True,
        rwidth=0.75,
        alpha=0.65,
        color=COLORS["empirical"],
        label="Empirical frequency",
    )
    axes[0, 2].plot(
        support,
        stats.poisson.pmf(support, rate),
        "o",
        color=COLORS["theoretical"],
        label="Theoretical PMF",
    )
    axes[0, 2].set(
        title=f"Poisson (discrete, λ={rate})",
        xlabel="Event count",
        ylabel="Probability mass",
    )
    axes[0, 2].text(
        0.97,
        0.95,
        _moment_annotation(rate, rate, samples),
        transform=axes[0, 2].transAxes,
        va="top",
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    rate = 0.5
    samples = rng.exponential(1 / rate, sample_size)
    upper = float(np.quantile(samples, 0.995))
    x = np.linspace(0, upper, 500)
    axes[1, 0].hist(
        samples,
        bins=90,
        density=True,
        alpha=0.62,
        color=COLORS["empirical"],
        label="Empirical density",
    )
    axes[1, 0].plot(
        x,
        stats.expon.pdf(x, scale=1 / rate),
        color=COLORS["theoretical"],
        linewidth=2.3,
        label="Theoretical PDF",
    )
    axes[1, 0].set(
        title=f"Exponential (continuous, λ={rate})",
        xlabel="Waiting time",
        ylabel="Probability density",
        xlim=(0, upper),
    )
    axes[1, 0].text(
        0.97,
        0.95,
        _moment_annotation(1 / rate, 1 / rate**2, samples),
        transform=axes[1, 0].transAxes,
        va="top",
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    mean, std = 10.0, 2.0
    samples = rng.normal(mean, std, sample_size)
    x = np.linspace(mean - 4 * std, mean + 4 * std, 500)
    axes[1, 1].hist(
        samples,
        bins=90,
        density=True,
        alpha=0.62,
        color=COLORS["empirical"],
        label="Empirical density",
    )
    axes[1, 1].plot(
        x,
        stats.norm.pdf(x, mean, std),
        color=COLORS["theoretical"],
        linewidth=2.3,
        label="Theoretical PDF",
    )
    axes[1, 1].set(
        title=f"Normal (continuous, μ={mean}, σ={std})",
        xlabel="Value",
        ylabel="Probability density",
    )
    axes[1, 1].text(
        0.03,
        0.95,
        _moment_annotation(mean, std**2, samples),
        transform=axes[1, 1].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    log_mean, log_std = 2.0, 0.6
    samples = rng.lognormal(log_mean, log_std, sample_size)
    upper = float(np.quantile(samples, 0.995))
    x = np.linspace(0.001, upper, 500)
    theoretical_mean = float(np.exp(log_mean + log_std**2 / 2))
    theoretical_variance = float(
        (np.exp(log_std**2) - 1)
        * np.exp(2 * log_mean + log_std**2)
    )
    axes[1, 2].hist(
        samples,
        bins=100,
        density=True,
        alpha=0.62,
        color=COLORS["empirical"],
        label="Empirical density",
    )
    axes[1, 2].plot(
        x,
        stats.lognorm.pdf(x, s=log_std, scale=np.exp(log_mean)),
        color=COLORS["theoretical"],
        linewidth=2.3,
        label="Theoretical PDF",
    )
    axes[1, 2].set(
        title=f"Log-normal (continuous, log μ={log_mean}, log σ={log_std})",
        xlabel="Positive value",
        ylabel="Probability density",
        xlim=(0, upper),
    )
    axes[1, 2].text(
        0.97,
        0.95,
        _moment_annotation(theoretical_mean, theoretical_variance, samples),
        transform=axes[1, 2].transAxes,
        va="top",
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )

    for axis in axes.flat:
        axis.legend(loc="best")
    figure.suptitle("Distribution overview: theory versus synthetic samples")
    figure.tight_layout()
    return _save_figure(figure, output_path)


def create_bernoulli_sensitivity(output_path: Path) -> Path:
    """Show probability mass moving between zero and one."""
    probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]
    x = np.arange(len(probabilities))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        x - width / 2,
        [1 - p for p in probabilities],
        width,
        label="P(X=0)",
        color=COLORS["empirical"],
    )
    axis.bar(
        x + width / 2,
        probabilities,
        width,
        label="P(X=1)",
        color=COLORS["theoretical"],
    )
    for index, p in enumerate(probabilities):
        axis.text(
            index,
            1.03,
            f"E[X]={p:.1f}\nVar={p * (1 - p):.2f}",
            ha="center",
            fontsize=9,
        )
    axis.set(
        title="Bernoulli parameter sensitivity",
        xlabel="Success probability p",
        ylabel="Probability mass",
        xticks=x,
        xticklabels=[str(p) for p in probabilities],
        ylim=(0, 1.18),
    )
    axis.legend()
    figure.tight_layout()
    return _save_figure(figure, output_path)


def create_binomial_sensitivity(output_path: Path) -> Path:
    """Compare Binomial shape, center, spread, and skew across n and p."""
    trial_counts = [10, 30, 100]
    probabilities = [0.2, 0.5, 0.8]
    figure, axes = plt.subplots(3, 3, figsize=(17, 12))
    for row, n in enumerate(trial_counts):
        for column, p in enumerate(probabilities):
            support = np.arange(n + 1)
            mass = stats.binom.pmf(support, n, p)
            mean = n * p
            variance = n * p * (1 - p)
            axes[row, column].bar(
                support,
                mass,
                color=COLORS["empirical"],
                width=max(0.7, n / 130),
            )
            axes[row, column].axvline(
                mean,
                color=COLORS["warning"],
                linestyle="--",
                label=f"E[X]={mean:.1f}",
            )
            axes[row, column].set(
                title=f"n={n}, p={p} | Var={variance:.1f}",
                xlabel="Success count",
                ylabel="Probability mass",
            )
            axes[row, column].legend()
    figure.suptitle(
        "Binomial sensitivity: n controls scale; p controls center and skew"
    )
    figure.tight_layout()
    return _save_figure(figure, output_path)


def create_poisson_sensitivity(output_path: Path) -> Path:
    """Show how a Poisson PMF changes as its rate increases."""
    rates = [1, 3, 5, 10, 20]
    support = np.arange(0, 46)
    figure, axis = plt.subplots(figsize=(12, 6.5))
    for rate in rates:
        axis.plot(
            support,
            stats.poisson.pmf(support, rate),
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=f"λ={rate}: mean=variance={rate}",
        )
    axis.set(
        title="Poisson parameter sensitivity",
        xlabel="Event count k",
        ylabel="P(X = k)",
        xlim=(-0.5, 45),
    )
    axis.legend(ncol=2)
    figure.tight_layout()
    return _save_figure(figure, output_path)


def create_exponential_sensitivity(output_path: Path) -> Path:
    """Compare Exponential PDF, CDF, survival, and constant hazard."""
    rates = [0.25, 0.5, 1.0, 2.0]
    x = np.linspace(0, 12, 600)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for rate in rates:
        scale = 1 / rate
        axes[0, 0].plot(
            x,
            stats.expon.pdf(x, scale=scale),
            label=f"λ={rate}, E[T]={scale:.2f}",
        )
        axes[0, 1].plot(x, stats.expon.cdf(x, scale=scale), label=f"λ={rate}")
        axes[1, 0].plot(x, stats.expon.sf(x, scale=scale), label=f"λ={rate}")
        axes[1, 1].plot(x, np.full_like(x, rate), label=f"λ={rate}")
    labels = [
        ("PDF: probability density", "Density"),
        ("CDF: P(T ≤ t)", "Cumulative probability"),
        ("Survival: P(T > t)", "Survival probability"),
        ("Hazard: h(t) = λ is constant", "Instantaneous hazard"),
    ]
    for axis, (title, ylabel) in zip(axes.flat, labels, strict=True):
        axis.set(title=title, xlabel="Waiting time t", ylabel=ylabel)
        axis.legend()
    figure.suptitle("Exponential parameter sensitivity and memoryless hazard")
    figure.tight_layout()
    return _save_figure(figure, output_path)


def create_normal_sensitivity(output_path: Path) -> Path:
    """Separate location and scale effects for the Normal distribution."""
    x = np.linspace(-10, 10, 800)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for mean in [-4, -2, 0, 2, 4]:
        density = stats.norm.pdf(x, loc=mean, scale=1.0)
        area = trapezoid(density, x)
        axes[0].plot(x, density, label=f"μ={mean}, area≈{area:.3f}")
    axes[0].set(
        title="Changing μ translates the curve",
        xlabel="Value x",
        ylabel="Density",
    )
    for std in [0.5, 1.0, 2.0, 3.0]:
        density = stats.norm.pdf(x, loc=0.0, scale=std)
        area = trapezoid(density, x)
        axes[1].plot(x, density, label=f"σ={std}, area≈{area:.3f}")
    axes[1].set(
        title="Changing σ controls spread and peak height",
        xlabel="Value x",
        ylabel="Density",
    )
    for axis in axes:
        axis.legend()
    figure.suptitle("Normal parameter sensitivity: total probability stays one")
    figure.tight_layout()
    return _save_figure(figure, output_path)


def create_lognormal_sensitivity(output_path: Path) -> Path:
    """Show location and right-tail changes for Log-normal parameters."""
    x = np.geomspace(0.03, 80, 900)
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    for log_mean in [-0.5, 0.0, 0.5, 1.0]:
        log_std = 0.45
        axes[0].plot(
            x,
            stats.lognorm.pdf(x, s=log_std, scale=np.exp(log_mean)),
            label=f"log μ={log_mean}",
        )
    axes[0].set(
        title="Changing log μ shifts the positive scale",
        xlabel="Positive value",
        ylabel="Density",
        xlim=(0, 12),
    )
    for log_std in [0.2, 0.45, 0.8, 1.1]:
        log_mean = 0.5
        mean = np.exp(log_mean + log_std**2 / 2)
        median = np.exp(log_mean)
        axes[1].plot(
            x,
            stats.lognorm.pdf(x, s=log_std, scale=np.exp(log_mean)),
            label=f"log σ={log_std}: mean={mean:.2f}, median={median:.2f}",
        )
    axes[1].set(
        title="Increasing log σ increases skew and the upper tail",
        xlabel="Positive value (log axis)",
        ylabel="Density",
        xscale="log",
        yscale="log",
        xlim=(0.05, 80),
        ylim=(1e-6, 4),
    )
    for axis in axes:
        axis.legend()
    figure.suptitle("Log-normal parameter sensitivity")
    figure.tight_layout()
    return _save_figure(figure, output_path)


def create_location_comparison(output_path: Path) -> Path:
    """Compare mean, median, and mode for Normal and Log-normal variables."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    mean, std = 0.0, 1.0
    x_normal = np.linspace(-4, 4, 600)
    axes[0].plot(
        x_normal,
        stats.norm.pdf(x_normal, mean, std),
        color=COLORS["empirical"],
        linewidth=2.4,
    )
    for label, value, color in [
        ("Mean", mean, COLORS["warning"]),
        ("Median", mean, COLORS["accent"]),
        ("Mode", mean, COLORS["theoretical"]),
    ]:
        axes[0].axvline(value, color=color, linewidth=2.2, label=f"{label}={value:.1f}")
    axes[0].set(
        title="Normal: mean = median = mode",
        xlabel="Value",
        ylabel="Density",
    )

    log_mean, log_std = 0.7, 0.7
    x_lognormal = np.linspace(0.01, 12, 700)
    lognormal_mean = np.exp(log_mean + log_std**2 / 2)
    lognormal_median = np.exp(log_mean)
    lognormal_mode = np.exp(log_mean - log_std**2)
    axes[1].plot(
        x_lognormal,
        stats.lognorm.pdf(
            x_lognormal,
            s=log_std,
            scale=np.exp(log_mean),
        ),
        color=COLORS["empirical"],
        linewidth=2.4,
    )
    for label, value, color in [
        ("Mode", lognormal_mode, COLORS["theoretical"]),
        ("Median", lognormal_median, COLORS["accent"]),
        ("Mean", lognormal_mean, COLORS["warning"]),
    ]:
        axes[1].axvline(value, color=color, linewidth=2.2, label=f"{label}={value:.2f}")
    axes[1].set(
        title="Log-normal: mode < median < mean",
        xlabel="Positive value",
        ylabel="Density",
    )
    for axis in axes:
        axis.legend()
    figure.suptitle("Location summaries reveal distribution asymmetry")
    figure.tight_layout()
    return _save_figure(figure, output_path)


def create_tail_behavior_comparison(
    output_path: Path,
    sample_size: int,
    seed: int,
) -> Path:
    """Compare central behavior and upper tails for normalized latency candidates."""
    rng = np.random.default_rng(seed)
    normal_samples = rng.normal(1.0, 0.22, sample_size)
    lognormal_samples = rng.lognormal(0.0, 0.55, sample_size)

    # Normalize both distributions to median one for a fair central comparison.
    normal_samples /= np.median(normal_samples)
    lognormal_samples /= np.median(lognormal_samples)
    normal_stats = calculate_empirical_statistics(normal_samples)
    lognormal_stats = calculate_empirical_statistics(lognormal_samples)

    upper = float(np.quantile(np.concatenate([normal_samples, lognormal_samples]), 0.995))
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes[0, 0].hist(
        normal_samples,
        bins=100,
        density=True,
        alpha=0.52,
        color=COLORS["empirical"],
        label="Normal candidate",
    )
    axes[0, 0].hist(
        lognormal_samples,
        bins=100,
        density=True,
        alpha=0.52,
        color=COLORS["theoretical"],
        label="Log-normal candidate",
    )
    axes[0, 0].set(
        title="Comparable medians, different shapes",
        xlabel="Median-normalized latency",
        ylabel="Density",
        xlim=(0, upper),
    )

    for samples, label, color in [
        (normal_samples, "Normal", COLORS["empirical"]),
        (lognormal_samples, "Log-normal", COLORS["theoretical"]),
    ]:
        sorted_values = np.sort(samples)
        cumulative = np.arange(1, sample_size + 1) / sample_size
        axes[0, 1].plot(sorted_values, cumulative, label=label, color=color)
        survival = 1 - cumulative + 1 / sample_size
        axes[1, 0].plot(sorted_values, survival, label=label, color=color)
    axes[0, 1].set(
        title="Empirical cumulative distribution",
        xlabel="Median-normalized latency",
        ylabel="P(X ≤ x)",
        xlim=(0, upper),
    )
    axes[1, 0].set(
        title="Upper-tail survival (log probability scale)",
        xlabel="Median-normalized latency",
        ylabel="P(X > x)",
        xlim=(0, upper),
        yscale="log",
        ylim=(1 / sample_size, 1),
    )

    quantile_labels = ["p50", "p90", "p95", "p99"]
    normal_quantiles = np.quantile(normal_samples, [0.5, 0.9, 0.95, 0.99])
    lognormal_quantiles = np.quantile(lognormal_samples, [0.5, 0.9, 0.95, 0.99])
    positions = np.arange(len(quantile_labels))
    width = 0.38
    axes[1, 1].bar(
        positions - width / 2,
        normal_quantiles,
        width,
        color=COLORS["empirical"],
        label=f"Normal mean={normal_stats.mean:.2f}",
    )
    axes[1, 1].bar(
        positions + width / 2,
        lognormal_quantiles,
        width,
        color=COLORS["theoretical"],
        label=f"Log-normal mean={lognormal_stats.mean:.2f}",
    )
    axes[1, 1].set(
        title="Tail percentiles diverge despite equal medians",
        xlabel="Percentile",
        ylabel="Median-normalized latency",
        xticks=positions,
        xticklabels=quantile_labels,
    )

    for axis in axes.flat:
        axis.legend()
    figure.suptitle(
        "Why average latency is insufficient: shape and tail behavior matter"
    )
    figure.tight_layout()
    return _save_figure(figure, output_path)


def parse_args() -> argparse.Namespace:
    """Parse static asset generation options."""
    parser = argparse.ArgumentParser(
        description="Generate high-resolution PNG and Plotly HTML assets."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30_000,
        help="Synthetic samples for empirical comparisons (default: 30000).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Skip standalone Plotly surface generation.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate every static PNG and standalone interactive HTML file."""
    args = parse_args()
    if args.sample_size < 1_000:
        raise ValueError("sample-size must be at least 1000 for stable visuals.")
    configure_matplotlib()
    directories = ensure_output_directories(OUTPUT_DIRECTORY)
    static_directory = directories["static"]

    create_distribution_overview(
        static_directory / "distribution_overview.png",
        args.sample_size,
        args.seed,
    )
    create_bernoulli_sensitivity(
        static_directory / "bernoulli_parameter_sensitivity.png"
    )
    create_binomial_sensitivity(
        static_directory / "binomial_parameter_sensitivity.png"
    )
    create_poisson_sensitivity(
        static_directory / "poisson_parameter_sensitivity.png"
    )
    create_exponential_sensitivity(
        static_directory / "exponential_functions_and_hazard.png"
    )
    create_normal_sensitivity(
        static_directory / "normal_parameter_sensitivity.png"
    )
    create_lognormal_sensitivity(
        static_directory / "lognormal_parameter_sensitivity.png"
    )
    create_location_comparison(
        static_directory / "mean_median_mode_comparison.png"
    )
    create_tail_behavior_comparison(
        static_directory / "normal_lognormal_tail_behavior.png",
        args.sample_size,
        args.seed + 1,
    )

    if not args.skip_html:
        for output_path in write_plotly_surfaces(directories["html"]):
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
