"""Static visual experiments for the Day 11 laboratory."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t

from .visual_utils import (
    BLUE,
    GRAY,
    GREEN,
    INK,
    LIGHT_BLUE,
    LIGHT_GRAY,
    LIGHT_ORANGE,
    ORANGE,
    PURPLE,
    RED,
    AssetResult,
    exponential_sample_means,
    save_figure,
    standardized_sample_means,
)


def generate_population_vs_sampling(*, quick: bool = False) -> AssetResult:
    """Contrast skewed observations with their sample-mean distribution."""
    simulations = 3_000 if quick else 10_000
    population_size = 30_000 if quick else 100_000
    sample_size = 30
    scale = 2.0
    rng = np.random.default_rng(1101)
    population = rng.exponential(scale=scale, size=population_size)
    means = exponential_sample_means(
        sample_size,
        simulations,
        seed=1102,
        scale=scale,
    )
    population_sd = float(np.std(population, ddof=0))
    empirical_se = float(np.std(means, ddof=0))
    theoretical_se = scale / np.sqrt(sample_size)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(population, bins=80, density=True, color=ORANGE, alpha=0.78)
    axes[0].axvline(scale, color=INK, linestyle="--", label=f"True mean = {scale:.1f}")
    axes[0].set(xlim=(0, 12), xlabel="Observation value", ylabel="Density")
    axes[0].set_title("Population: individual observations stay skewed")
    axes[0].legend()

    axes[1].hist(means, bins=55, density=True, color=BLUE, alpha=0.78)
    x = np.linspace(max(0.0, means.min()), means.max(), 400)
    axes[1].plot(
        x,
        norm.pdf(x, loc=scale, scale=theoretical_se),
        color=INK,
        linewidth=2,
        linestyle="--",
        label="CLT normal approximation",
    )
    axes[1].axvline(scale, color=INK, linestyle=":", label="True mean")
    axes[1].set(xlabel="Sample mean", ylabel="Density")
    axes[1].set_title(f"Sampling distribution of the mean (n={sample_size})")
    axes[1].legend(loc="upper left")
    axes[1].text(
        0.98,
        0.95,
        f"Population SD: {population_sd:.3f}\n"
        f"Theoretical SE: {theoretical_se:.3f}\n"
        f"Empirical SD of means: {empirical_se:.3f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )
    fig.suptitle("The CLT changes the estimator's distribution, not the source data", fontsize=15)
    fig.tight_layout()
    path = save_figure(fig, "01_population_vs_sampling.png", quick=quick)
    return AssetResult(
        "Population vs sampling distribution",
        path,
        (
            f"Theoretical SE={theoretical_se:.4f}",
            f"Empirical SD of means={empirical_se:.4f}",
        ),
    )


def generate_standard_error_plot(*, quick: bool = False) -> AssetResult:
    """Show the square-root law and its diminishing returns."""
    sample_sizes = np.arange(1, 2_001)
    sigmas = (1.0, 2.0, 5.0)
    colors = (GREEN, BLUE, ORANGE)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for sigma, color in zip(sigmas, colors, strict=True):
        ax.plot(
            sample_sizes,
            sigma / np.sqrt(sample_sizes),
            color=color,
            linewidth=2.2,
            label=rf"$\sigma={sigma:g}$",
        )

    selected = np.array([25, 100, 400, 1600])
    selected_se = 2.0 / np.sqrt(selected)
    ax.scatter(selected, selected_se, color=BLUE, edgecolor="white", s=55, zorder=4)
    for n, se in zip(selected, selected_se, strict=True):
        ax.annotate(
            f"n={n:,}\nSE={se:.3f}",
            (n, se),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize=9,
            ha="center",
        )
    ax.annotate(
        "4x more observations\n≈ half the standard error",
        xy=(400, selected_se[2]),
        xytext=(145, 0.95),
        arrowprops={"arrowstyle": "->", "color": INK},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT_BLUE},
    )
    ax.set(
        xlim=(1, 2_000),
        ylim=(0, 5.1),
        xlabel="Independent sample size n (log scale)",
        ylabel="Standard error of the mean",
        title=r"Standard error shrinks as $\sigma/\sqrt{n}$ — with diminishing returns",
    )
    ax.set_xscale("log")
    ax.set_xticks((1, 25, 100, 400, 1_600))
    ax.set_xticklabels(("1", "25", "100", "400", "1,600"))
    ax.grid(alpha=0.8)
    ax.legend(title="Population variability")
    fig.tight_layout()
    path = save_figure(fig, "03_standard_error_vs_n.png", quick=quick)
    return AssetResult(
        "Standard error vs sample size",
        path,
        ("For sigma=2: SE(25)=0.4000; SE(100)=0.2000; SE(400)=0.1000",),
    )


def generate_sd_vs_se(*, quick: bool = False) -> AssetResult:
    """Show SD and SE as spreads of two different random quantities."""
    simulations = 3_000 if quick else 10_000
    sample_size = 50
    scale = 2.0
    rng = np.random.default_rng(5101)
    observations = rng.exponential(scale=scale, size=50_000)
    means = exponential_sample_means(sample_size, simulations, seed=5102, scale=scale)
    observation_sd = float(np.std(observations, ddof=0))
    estimator_se = float(np.std(means, ddof=0))

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True)
    bins = np.linspace(0, 10, 75)
    axes[0].hist(observations, bins=bins, density=True, color=ORANGE, alpha=0.8)
    axes[0].hlines(0.16, scale - observation_sd, scale + observation_sd, color=INK, linewidth=4)
    axes[0].plot([scale - observation_sd, scale + observation_sd], [0.16, 0.16], "|", color=INK, markersize=15)
    axes[0].text(scale, 0.185, f"one population SD ≈ {observation_sd:.3f}", ha="center")
    axes[0].set(ylabel="Density", title="SD: variability among individual observations")

    axes[1].hist(means, bins=bins, density=True, color=BLUE, alpha=0.8)
    axes[1].hlines(0.65, scale - estimator_se, scale + estimator_se, color=INK, linewidth=4)
    axes[1].plot([scale - estimator_se, scale + estimator_se], [0.65, 0.65], "|", color=INK, markersize=15)
    axes[1].text(scale, 0.82, f"one estimator SE ≈ {estimator_se:.3f}", ha="center")
    axes[1].set(xlabel="Value in the same units", ylabel="Density", title=f"SE: variability among repeated sample means (n={sample_size})")
    axes[1].set_xlim(0, 10)
    for ax in axes:
        ax.axvline(scale, color=GRAY, linestyle="--", linewidth=1.5)
        ax.grid(axis="y", alpha=0.7)
    fig.suptitle("SD and SE measure the spread of different random quantities", fontsize=15)
    fig.tight_layout()
    path = save_figure(fig, "05_sd_vs_se.png", quick=quick)
    return AssetResult(
        "Standard deviation vs standard error",
        path,
        (f"Population SD={observation_sd:.4f}", f"Estimator SE={estimator_se:.4f}"),
    )


def _confidence_interval_sample() -> np.ndarray:
    """Return the common normal sample used by interval-construction plots."""
    return np.random.default_rng(6101).normal(loc=50.0, scale=8.0, size=24)


def generate_ci_construction(*, quick: bool = False) -> AssetResult:
    """Decompose one Student-t confidence interval into its ingredients."""
    sample = _confidence_interval_sample()
    sample_size = len(sample)
    mean = float(np.mean(sample))
    sample_sd = float(np.std(sample, ddof=1))
    standard_error = sample_sd / np.sqrt(sample_size)
    critical = float(t.ppf(0.975, df=sample_size - 1))
    margin = critical * standard_error
    lower, upper = mean - margin, mean + margin

    fig = plt.figure(figsize=(11, 6.2))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.15, 1.0))
    ax_steps = fig.add_subplot(grid[0])
    ax_ci = fig.add_subplot(grid[1])
    ax_steps.axis("off")
    labels = (
        ("Sample", f"n = {sample_size}\nmean = {mean:.3f}\ns = {sample_sd:.3f}"),
        ("Uncertainty", f"SE = s / √n\n= {standard_error:.3f}"),
        ("Confidence", f"t* (df={sample_size - 1})\n= {critical:.3f}"),
        ("Margin", f"t* × SE\n= {margin:.3f}"),
    )
    x_positions = np.linspace(0.12, 0.88, len(labels))
    for index, (heading, body) in enumerate(labels):
        x = x_positions[index]
        ax_steps.text(
            x,
            0.55,
            f"{heading}\n{body}",
            transform=ax_steps.transAxes,
            ha="center",
            va="center",
            fontsize=10.5,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": LIGHT_BLUE if index % 2 == 0 else LIGHT_ORANGE,
                "edgecolor": "#A7AFB5",
            },
        )
        if index < len(labels) - 1:
            ax_steps.annotate(
                "",
                xy=(x_positions[index + 1] - 0.09, 0.55),
                xytext=(x + 0.09, 0.55),
                xycoords=ax_steps.transAxes,
                arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 1.5},
            )
    ax_steps.set_title("Build the interval from the sample—not from a memorized endpoint", pad=8)

    ax_ci.errorbar(
        mean,
        0,
        xerr=margin,
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=8,
        markersize=9,
        linewidth=3,
    )
    ax_ci.axvline(mean, color=INK, linestyle="--", linewidth=1.2)
    ax_ci.text(lower, 0.12, f"lower\n{lower:.3f}", ha="center")
    ax_ci.text(mean, -0.16, f"estimate\n{mean:.3f}", ha="center")
    ax_ci.text(upper, 0.12, f"upper\n{upper:.3f}", ha="center")
    ax_ci.set(ylim=(-0.35, 0.35), yticks=[], xlabel="Mean quality score")
    ax_ci.set_title(r"95% CI = estimate $\pm$ critical value $\times$ standard error")
    ax_ci.grid(axis="x", alpha=0.7)
    fig.suptitle("How a Student-t confidence interval is assembled", fontsize=15)
    fig.tight_layout()
    path = save_figure(fig, "06_ci_construction.png", quick=quick)
    return AssetResult(
        "Confidence interval construction",
        path,
        (
            f"mean={mean:.4f}; sample SD={sample_sd:.4f}; SE={standard_error:.4f}",
            f"t*={critical:.4f}; margin={margin:.4f}; CI=[{lower:.4f}, {upper:.4f}]",
        ),
    )


def generate_confidence_width_comparison(*, quick: bool = False) -> AssetResult:
    """Hold the sample fixed while confidence level changes interval width."""
    sample = _confidence_interval_sample()
    sample_size = len(sample)
    mean = float(np.mean(sample))
    standard_error = float(np.std(sample, ddof=1) / np.sqrt(sample_size))
    levels = np.array([0.80, 0.90, 0.95, 0.99])
    critical_values = t.ppf(0.5 + levels / 2.0, df=sample_size - 1)
    margins = critical_values * standard_error

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    y = np.arange(len(levels))
    for row, level, critical, margin in zip(
        y, levels, critical_values, margins, strict=True
    ):
        ax.errorbar(
            mean,
            row,
            xerr=margin,
            fmt="o",
            color=BLUE,
            ecolor=BLUE,
            capsize=7,
            markersize=8,
            linewidth=2.6,
        )
        ax.text(mean + margin + 0.08, row, f"t*={critical:.3f}  width={2 * margin:.3f}", va="center")
    ax.axvline(mean, color=INK, linestyle="--")
    ax.set(
        yticks=y,
        yticklabels=[f"{level:.0%}" for level in levels],
        xlabel="Confidence interval for the same population mean",
        ylabel="Confidence level",
        title="Higher confidence requires a wider interval",
    )
    ax.grid(axis="x", alpha=0.8)
    ax.set_xlim(46.8, 57.2)
    fig.text(
        0.5,
        0.01,
        f"Same sample: n={sample_size}, estimate={mean:.3f}, SE={standard_error:.3f}. "
        "Only the requested long-run coverage changes.",
        ha="center",
        color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = save_figure(fig, "08_confidence_level_width.png", quick=quick)
    return AssetResult(
        "Confidence level vs interval width",
        path,
        tuple(
            f"{level:.0%}: t*={critical:.4f}, width={2 * margin:.4f}"
            for level, critical, margin in zip(
                levels, critical_values, margins, strict=True
            )
        ),
    )


def generate_t_vs_normal(*, quick: bool = False) -> AssetResult:
    """Compare Student-t tails and critical values with the standard normal."""
    x = np.linspace(-5, 5, 1_200)
    degrees = (3, 5, 10, 30)
    colors = (RED, ORANGE, PURPLE, BLUE)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.plot(x, norm.pdf(x), color=INK, linewidth=2.8, label="Normal(0, 1)")
    for df, color in zip(degrees, colors, strict=True):
        ax.plot(x, t.pdf(x, df=df), color=color, linewidth=1.9, label=f"t(df={df})")

    normal_critical = float(norm.ppf(0.975))
    t3_critical = float(t.ppf(0.975, df=3))
    ax.axvline(normal_critical, color=INK, linestyle="--", linewidth=1.4)
    ax.axvline(t3_critical, color=RED, linestyle=":", linewidth=1.8)
    ax.annotate(
        f"Normal 95% critical value\n{normal_critical:.3f}",
        (normal_critical, norm.pdf(normal_critical)),
        xytext=(0.55, 0.30),
        arrowprops={"arrowstyle": "->", "color": INK},
    )
    ax.annotate(
        f"t(df=3) critical value\n{t3_critical:.3f}",
        (t3_critical, t.pdf(t3_critical, df=3)),
        xytext=(3.25, 0.17),
        arrowprops={"arrowstyle": "->", "color": RED},
    )
    ax.text(
        0.02,
        0.96,
        "Small df → more uncertainty about σ\n"
        "→ heavier tails → larger critical value → wider CI",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": LIGHT_ORANGE},
    )
    ax.set(
        xlim=(-5, 5),
        ylim=(0, 0.43),
        xlabel="Standardized value",
        ylabel="Probability density",
        title=r"Student-t approaches Normal(0, 1) as degrees of freedom increase",
    )
    ax.grid(alpha=0.65)
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    path = save_figure(fig, "09_z_vs_t_distribution.png", quick=quick)
    return AssetResult(
        "Normal vs Student-t",
        path,
        (
            f"Normal 95% critical value={normal_critical:.4f}",
            f"t(df=3) 95% critical value={t3_critical:.4f}",
        ),
    )


def generate_clt_distribution_comparison(*, quick: bool = False) -> AssetResult:
    """Show that CLT approximation speed depends on source-distribution shape."""
    distributions = ("Normal", "Uniform", "Exponential", "Lognormal")
    sample_sizes = (5, 30, 100)
    simulations = 1_500 if quick else 5_000
    child_seeds = np.random.SeedSequence(10101).spawn(
        len(distributions) * len(sample_sizes)
    )
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 11), sharex=True, sharey=True)
    normal_x = np.linspace(-3.5, 7.0, 500)
    seed_index = 0
    skewness_values: dict[tuple[str, int], float] = {}
    for row, distribution in enumerate(distributions):
        for column, sample_size in enumerate(sample_sizes):
            seed = int(child_seeds[seed_index].generate_state(1)[0])
            seed_index += 1
            standardized = standardized_sample_means(
                distribution,
                sample_size,
                simulations,
                seed=seed,
            )
            centered = standardized - np.mean(standardized)
            sd = float(np.std(standardized, ddof=0))
            skewness = float(np.mean((centered / sd) ** 3))
            skewness_values[(distribution, sample_size)] = skewness
            ax = axes[row, column]
            ax.hist(
                standardized,
                bins=np.linspace(-3.5, 7.0, 60),
                density=True,
                color=BLUE,
                alpha=0.73,
            )
            ax.plot(normal_x, norm.pdf(normal_x), color=INK, linestyle="--", linewidth=1.4)
            ax.text(
                0.96,
                0.88,
                f"skew={skewness:.2f}",
                transform=ax.transAxes,
                ha="right",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            if row == 0:
                ax.set_title(f"n = {sample_size}")
            if column == 0:
                ax.set_ylabel(f"{distribution}\nDensity")
            if row == len(distributions) - 1:
                ax.set_xlabel("Standardized sample mean")
            ax.grid(alpha=0.5)

    fig.suptitle(
        "There is no universal n=30 rule: source shape changes CLT convergence",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.015,
        "Dashed curve: standard normal reference. All panels use exact source mean and variance for standardization.",
        ha="center",
        color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    path = save_figure(fig, "10_skewness_and_sample_size.png", quick=quick)
    return AssetResult(
        "CLT convergence by source shape",
        path,
        tuple(
            f"{distribution}, n={sample_size}: standardized-mean skewness={skewness_values[(distribution, sample_size)]:.4f}"
            for distribution in distributions
            for sample_size in sample_sizes
        ),
    )
