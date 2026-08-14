"""Animated repeated-sampling experiments for the Day 11 laboratory."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from scipy.stats import norm

from .visual_utils import (
    BLUE,
    GREEN,
    INK,
    LIGHT_BLUE,
    ORANGE,
    RED,
    AssetResult,
    ensure_asset_directory,
    exponential_sample_means,
    simulate_normal_t_intervals,
)


CLT_SAMPLE_SIZES = (1, 2, 5, 10, 20, 30, 50, 100, 200)


def generate_clt_animation(*, quick: bool = False) -> AssetResult:
    """Animate changing shape and spread of exponential sample means."""
    sample_sizes = (1, 5, 30, 100) if quick else CLT_SAMPLE_SIZES
    simulations = 1_200 if quick else 4_000
    scale = 2.0
    seeds = np.random.SeedSequence(2101).spawn(len(sample_sizes))
    means_by_size: dict[int, np.ndarray] = {}
    for sample_size, child_seed in zip(sample_sizes, seeds, strict=True):
        means_by_size[sample_size] = exponential_sample_means(
            sample_size,
            simulations,
            seed=int(child_seed.generate_state(1)[0]),
            scale=scale,
        )

    bins = np.linspace(0, 10, 72)
    x = np.linspace(0, 10, 500)
    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    def update(frame: int) -> None:
        ax.clear()
        sample_size = sample_sizes[frame]
        means = means_by_size[sample_size]
        theoretical_se = scale / np.sqrt(sample_size)
        empirical_se = float(np.std(means, ddof=0))
        histogram_peak = float(np.max(np.histogram(means, bins=bins, density=True)[0]))
        normal_peak = float(norm.pdf(scale, loc=scale, scale=theoretical_se))
        ax.hist(
            means,
            bins=bins,
            density=True,
            color=BLUE,
            alpha=0.78,
            label="Simulated sample means",
        )
        ax.plot(
            x,
            norm.pdf(x, loc=scale, scale=theoretical_se),
            color=INK,
            linestyle="--",
            linewidth=2.2,
            label="CLT normal approximation",
        )
        ax.axvline(scale, color=ORANGE, linestyle=":", linewidth=2, label="True mean")
        ax.set(
            xlim=(0, 10),
            ylim=(0, max(0.6, 1.18 * histogram_peak, 1.18 * normal_peak)),
            xlabel="Sample mean (fixed x-axis across frames)",
            ylabel="Density",
            title=f"Sample size n = {sample_size}",
        )
        ax.grid(alpha=0.55)
        ax.legend(loc="upper right")
        ax.text(
            0.98,
            0.72,
            f"Theoretical SE = {theoretical_se:.4f}\n"
            f"Empirical SD = {empirical_se:.4f}\n"
            f"Simulations = {simulations:,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.92},
        )
        fig.suptitle(
            "CLT convergence: increasing n changes shape and shrinks uncertainty",
            fontsize=14,
        )

    animation = FuncAnimation(fig, update, frames=len(sample_sizes), interval=900, repeat=True)
    output = ensure_asset_directory() / "02_clt_convergence.gif"
    animation.save(output, writer=PillowWriter(fps=1.25), dpi=75 if quick else 95)
    plt.close(fig)
    final_size = sample_sizes[-1]
    final_empirical = float(np.std(means_by_size[final_size], ddof=0))
    final_theoretical = scale / np.sqrt(final_size)
    return AssetResult(
        "CLT convergence animation",
        output,
        (
            f"Frames={len(sample_sizes)}; simulations/frame={simulations}",
            f"Final n={final_size}: theoretical SE={final_theoretical:.4f}; empirical SD={final_empirical:.4f}",
        ),
    )


def generate_ci_coverage_animation(*, quick: bool = False) -> AssetResult:
    """Animate the repeated-sampling meaning of a 95% confidence procedure."""
    interval_count = 20 if quick else 60
    sample_size = 25
    population_mean = 10.0
    lower, means, upper, covered = simulate_normal_t_intervals(
        sample_size=sample_size,
        intervals=interval_count,
        confidence=0.95,
        population_mean=population_mean,
        population_sd=3.0,
        seed=7101,
    )
    x_padding = 0.25
    x_limits = (float(np.min(lower) - x_padding), float(np.max(upper) + x_padding))
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    legend_handles = (
        Line2D([0], [0], color=GREEN, marker="o", linestyle="-", label="Contains μ"),
        Line2D([0], [0], color=RED, marker="x", linestyle="--", label="Misses μ"),
    )

    def update(frame: int) -> None:
        ax.clear()
        shown = frame + 1
        for index in range(shown):
            if covered[index]:
                color, marker, linestyle = GREEN, "o", "-"
            else:
                color, marker, linestyle = RED, "x", "--"
            ax.plot(
                [lower[index], upper[index]],
                [index + 1, index + 1],
                color=color,
                marker=marker,
                linestyle=linestyle,
                markevery=(0, 1),
                linewidth=1.6,
                markersize=4.5,
            )
            ax.plot(means[index], index + 1, marker="D", color=color, markersize=3.5)
        covering = int(np.sum(covered[:shown]))
        empirical = covering / shown
        ax.axvline(population_mean, color=INK, linestyle=":", linewidth=2, label="True mean μ")
        ax.set(
            xlim=x_limits,
            ylim=(interval_count + 1, 0),
            xlabel="Student-t confidence interval for the mean",
            ylabel="Repeated sample",
            title="Each interval is random; the population mean is fixed",
        )
        ax.grid(axis="x", alpha=0.55)
        ax.legend(handles=(*legend_handles, ax.lines[-1]), loc="lower right")
        ax.text(
            0.02,
            0.97,
            f"Intervals generated: {shown}\n"
            f"Intervals covering μ: {covering}\n"
            f"Empirical coverage: {empirical:.1%}\n"
            "Nominal coverage: 95%",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": LIGHT_BLUE, "alpha": 0.94},
        )
        fig.suptitle("Frequentist confidence belongs to the repeated procedure", fontsize=14)

    animation = FuncAnimation(fig, update, frames=interval_count, interval=190, repeat=True)
    output = ensure_asset_directory() / "07_ci_coverage.gif"
    animation.save(output, writer=PillowWriter(fps=5), dpi=72 if quick else 88)
    plt.close(fig)
    final_coverage = float(np.mean(covered))
    return AssetResult(
        "Confidence interval coverage animation",
        output,
        (
            f"Nominal coverage=95.00%; empirical coverage={final_coverage:.2%}",
            f"Intervals={interval_count}; covered={int(np.sum(covered))}; sample size={sample_size}",
        ),
    )
