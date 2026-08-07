"""Visualize dispersion, squared deviations, and Bessel's correction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visual_utils import (
    COLORS,
    GIF_DIRECTORY,
    IMAGE_DIRECTORY,
    configure_matplotlib,
    ensure_output_directories,
    save_animation,
    save_figure,
)

import matplotlib.pyplot as plt


def create_variance_animation(
    output_path: Path = GIF_DIRECTORY / "variance_and_std.gif",
) -> Path:
    """Spread observations around a fixed mean and update variance and SD."""
    configure_matplotlib()
    standardized_offsets = np.array(
        [-2.4, -1.8, -1.25, -0.8, -0.35, 0.35, 0.8, 1.25, 1.8, 2.4]
    )
    scales = np.concatenate([np.linspace(1.0, 13.0, 25), np.full(5, 13.0)])
    mean = 50.0
    figure, (number_axis, metric_axis) = plt.subplots(1, 2, figsize=(13, 5.8))

    def update(frame: int) -> None:
        scale = float(scales[frame])
        values = mean + scale * standardized_offsets
        population_variance = float(np.var(values, ddof=0))
        population_std = float(np.std(values, ddof=0))
        number_axis.clear()
        metric_axis.clear()

        for index, value in enumerate(values):
            color = COLORS["orange"] if value > mean else COLORS["blue"]
            number_axis.plot(
                [mean, value],
                [index, index],
                color=color,
                alpha=0.65,
                linewidth=2,
            )
            number_axis.scatter(value, index, color=color, s=45, zorder=3)
        number_axis.axvline(
            mean,
            color=COLORS["dark"],
            linewidth=2.2,
            label=f"Fixed mean = {mean:.0f}",
        )
        number_axis.set(
            title="Each segment is a deviation from the mean",
            xlabel="Observation value",
            ylabel="Observation index",
            xlim=(15, 85),
            ylim=(-1, len(values)),
        )
        number_axis.legend(loc="upper left")

        metric_axis.bar(
            ["Population variance\n(squared units)", "Population SD\n(original units)"],
            [population_variance, population_std],
            color=[COLORS["purple"], COLORS["green"]],
        )
        metric_axis.set(
            title="Same center, increasing dispersion",
            ylabel="Statistic value",
            ylim=(0, 1_050),
        )
        metric_axis.text(
            0,
            population_variance + 25,
            f"σ² = {population_variance:.1f}",
            ha="center",
        )
        metric_axis.text(
            1,
            population_std + 25,
            f"σ = {population_std:.1f}",
            ha="center",
        )
        metric_axis.text(
            0.03,
            0.95,
            "Variance grows with squared distance;\n"
            "standard deviation returns to the data's unit.",
            transform=metric_axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )

        figure.suptitle(
            "Dispersion changes while the mean remains fixed\n"
            f"scale={scale:.1f}   μ={mean:.1f}   σ²={population_variance:.1f}   "
            f"σ={population_std:.1f}",
            y=0.98,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.86))

    return save_animation(
        figure,
        update,
        len(scales),
        output_path,
        fps=6,
        dpi=88,
    )


def create_squared_deviations_figure(
    output_path: Path = IMAGE_DIRECTORY / "why_squared_deviations.png",
) -> Path:
    """Show cancellation of signed deviations and the effect of squaring."""
    configure_matplotlib()
    values = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    mean = float(np.mean(values))
    deviations = values - mean
    squared = deviations**2
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    colors = [
        COLORS["blue"] if deviation < 0 else COLORS["orange"]
        for deviation in deviations
    ]
    axes[0].scatter(values, np.zeros_like(values), s=100, c=colors, zorder=3)
    for value in values:
        axes[0].plot([mean, value], [0, 0], linewidth=3, alpha=0.65)
    axes[0].axvline(
        mean,
        color=COLORS["dark"],
        linewidth=2,
        label=f"Mean = {mean:.1f}",
    )
    axes[0].set(
        title="1. Measure distance from the mean",
        xlabel="Observation value",
        yticks=[],
        xlim=(0, 12),
        ylim=(-0.35, 0.55),
    )
    axes[0].legend()

    axes[1].bar(
        [str(int(value)) for value in values],
        deviations,
        color=colors,
    )
    axes[1].axhline(0, color=COLORS["dark"], linewidth=1)
    axes[1].set(
        title="2. Signed deviations cancel",
        xlabel="Observation",
        ylabel=r"$x_i - \bar{x}$",
        ylim=(-5, 5),
    )
    axes[1].text(
        0.5,
        0.92,
        f"Σ deviations = {np.sum(deviations):.1f}",
        transform=axes[1].transAxes,
        ha="center",
        bbox={"facecolor": "white", "edgecolor": "0.8"},
    )

    axes[2].bar(
        [str(int(value)) for value in values],
        squared,
        color=COLORS["purple"],
    )
    axes[2].set(
        title="3. Squaring preserves magnitude",
        xlabel="Observation",
        ylabel=r"$(x_i - \bar{x})^2$",
        ylim=(0, 19),
    )
    axes[2].text(
        0.5,
        0.92,
        f"Σ squared = {np.sum(squared):.1f}\n"
        f"Population variance = {np.mean(squared):.1f}",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.8"},
    )

    figure.suptitle(
        "Why variance uses squared deviations",
        y=1.02,
    )
    figure.tight_layout()
    return save_figure(figure, output_path)


def create_bessel_correction_figure(
    output_path: Path = IMAGE_DIRECTORY / "bessel_correction.png",
    *,
    repetitions: int = 4_000,
    sample_size: int = 5,
) -> Path:
    """Compare repeated variance estimates using n and n-1 denominators."""
    configure_matplotlib()
    rng = np.random.default_rng(42)
    population = rng.normal(100.0, 15.0, 200_000)
    true_variance = float(np.var(population, ddof=0))
    samples = rng.choice(population, size=(repetitions, sample_size), replace=True)
    estimates_n = np.var(samples, axis=1, ddof=0)
    estimates_n_minus_1 = np.var(samples, axis=1, ddof=1)
    running_n = np.cumsum(estimates_n) / np.arange(1, repetitions + 1)
    running_n_minus_1 = np.cumsum(estimates_n_minus_1) / np.arange(
        1, repetitions + 1
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    bins = np.linspace(0, np.quantile(estimates_n_minus_1, 0.99), 55)
    axes[0].hist(
        estimates_n,
        bins=bins,
        alpha=0.62,
        color=COLORS["red"],
        label=f"Divide by n: mean={np.mean(estimates_n):.1f}",
    )
    axes[0].hist(
        estimates_n_minus_1,
        bins=bins,
        alpha=0.55,
        color=COLORS["green"],
        label=f"Divide by n−1: mean={np.mean(estimates_n_minus_1):.1f}",
    )
    axes[0].axvline(
        true_variance,
        color=COLORS["dark"],
        linewidth=2.3,
        label=f"Population variance={true_variance:.1f}",
    )
    axes[0].set(
        title=f"Distribution of {repetitions:,} estimates (sample n={sample_size})",
        xlabel="Estimated variance",
        ylabel="Repeated samples",
    )
    axes[0].legend()

    draws = np.arange(1, repetitions + 1)
    axes[1].plot(
        draws,
        running_n,
        color=COLORS["red"],
        label="Running mean, denominator n",
    )
    axes[1].plot(
        draws,
        running_n_minus_1,
        color=COLORS["green"],
        label="Running mean, denominator n−1",
    )
    axes[1].axhline(
        true_variance,
        color=COLORS["dark"],
        linewidth=2.2,
        linestyle="--",
        label="True population variance",
    )
    axes[1].set(
        title="Average estimate across repeated samples",
        xlabel="Number of repeated samples",
        ylabel="Running mean of variance estimates",
        xlim=(1, repetitions),
    )
    axes[1].legend()

    figure.suptitle(
        "Bessel's correction addresses variance-estimator bias\n"
        "IID samples with replacement from a fixed synthetic population",
        y=1.02,
    )
    figure.tight_layout()
    return save_figure(figure, output_path)


def generate_dispersion_visuals() -> list[Path]:
    """Generate all dispersion-related assets."""
    ensure_output_directories()
    return [
        create_variance_animation(),
        create_squared_deviations_figure(),
        create_bessel_correction_figure(),
    ]


def main() -> None:
    """Generate this module's outputs independently."""
    generated = generate_dispersion_visuals()
    print(f"Generated {len(generated)} dispersion visualizations.")


if __name__ == "__main__":
    main()
