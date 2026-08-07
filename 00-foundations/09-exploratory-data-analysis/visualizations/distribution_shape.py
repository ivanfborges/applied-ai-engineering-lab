"""Visualize skewness, kurtosis, and production tail percentiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visual_utils import (
    COLORS,
    GIF_DIRECTORY,
    IMAGE_DIRECTORY,
    configure_matplotlib,
    ensure_output_directories,
    sample_excess_kurtosis,
    sample_skewness,
    save_animation,
    save_figure,
    synthetic_ai_workload,
)

import matplotlib.pyplot as plt


def _standardize(values: np.ndarray) -> np.ndarray:
    """Center and scale one sample using population SD for visual comparison."""
    return (values - np.mean(values)) / np.std(values, ddof=0)


def create_skewness_comparison(
    output_path: Path = IMAGE_DIRECTORY / "skewness_comparison.png",
) -> Path:
    """Compare left-skewed, symmetric, and right-skewed samples."""
    configure_matplotlib()
    rng = np.random.default_rng(42)
    positive = _standardize(rng.lognormal(0.0, 0.58, 4_000))
    symmetric = _standardize(rng.normal(0.0, 1.0, 4_000))
    negative = -positive
    datasets = [
        ("Negative skew: tail points left", negative, COLORS["purple"]),
        ("Approximately symmetric", symmetric, COLORS["blue"]),
        ("Positive skew: tail points right", positive, COLORS["orange"]),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    bins = np.linspace(-5, 5, 60)
    for axis, (title, values, color) in zip(axes, datasets, strict=True):
        sample_mean = float(np.mean(values))
        sample_median = float(np.median(values))
        skew = sample_skewness(values)
        axis.hist(values, bins=bins, color=color, alpha=0.72)
        axis.axvline(
            sample_mean,
            color=COLORS["red"],
            linestyle="--",
            linewidth=2,
            label=f"Mean {sample_mean:.2f}",
        )
        axis.axvline(
            sample_median,
            color=COLORS["green"],
            linewidth=2,
            label=f"Median {sample_median:.2f}",
        )
        axis.set(
            title=title,
            xlabel="Standardized value",
            xlim=(-5, 5),
        )
        axis.text(
            0.04,
            0.93,
            f"Sample skewness = {skew:.2f}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )
        axis.legend(loc="upper right")
    axes[0].set_ylabel("Count")
    figure.suptitle(
        "Skewness describes direction of asymmetry, not normality",
        y=1.02,
    )
    figure.tight_layout()
    return save_figure(figure, output_path)


def create_skewness_animation(
    output_path: Path = GIF_DIRECTORY / "skewness_animation.gif",
) -> Path:
    """Transform a fixed symmetric sample into a right-skewed sample."""
    configure_matplotlib()
    rng = np.random.default_rng(42)
    base = rng.normal(0.0, 1.0, 900)
    strengths = np.concatenate([np.linspace(0.0, 1.55, 24), np.full(5, 1.55)])
    figure, axis = plt.subplots(figsize=(9.5, 5.8))

    def update(frame: int) -> None:
        strength = float(strengths[frame])
        # Only positive values receive an increasing quadratic displacement,
        # so the right tail lengthens while the central ordering changes less.
        values = base + strength * np.maximum(base, 0.0) ** 2
        current_mean = float(np.mean(values))
        current_median = float(np.median(values))
        skew = sample_skewness(values)
        axis.clear()
        axis.hist(
            values,
            bins=np.linspace(-4, 14, 65),
            color=COLORS["blue"],
            alpha=0.74,
        )
        axis.axvline(
            current_mean,
            color=COLORS["red"],
            linestyle="--",
            linewidth=2.3,
            label=f"Mean = {current_mean:.2f}",
        )
        axis.axvline(
            current_median,
            color=COLORS["green"],
            linewidth=2.3,
            label=f"Median = {current_median:.2f}",
        )
        axis.annotate(
            "Increasing right tail",
            xy=(min(12.0, np.quantile(values, 0.995)), 7),
            xytext=(8.0, 55),
            arrowprops={"arrowstyle": "->", "color": COLORS["orange"]},
            color=COLORS["orange"],
            ha="center",
        )
        axis.set(
            title="A longer right tail pulls the mean and raises skewness",
            xlabel="Value",
            ylabel="Count",
            xlim=(-4, 14),
            ylim=(0, 105),
        )
        axis.text(
            0.98,
            0.94,
            f"shape strength={strength:.2f}\n"
            f"skewness={skew:.2f}\n"
            f"mean−median={current_mean-current_median:.2f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )
        axis.legend(loc="upper left")
        figure.tight_layout()

    return save_animation(
        figure,
        update,
        len(strengths),
        output_path,
        fps=6,
        dpi=90,
    )


def _draw_tail_histogram(
    axis: plt.Axes,
    values: np.ndarray,
    title: str,
    color: str,
) -> None:
    """Draw one variance-standardized distribution and emphasize tail points."""
    standardized = _standardize(values)
    kurtosis = sample_excess_kurtosis(standardized)
    axis.hist(
        standardized,
        bins=np.linspace(-8, 8, 65),
        color=color,
        alpha=0.70,
    )
    beyond_two = standardized[np.abs(standardized) > 2]
    beyond_three = standardized[np.abs(standardized) > 3]
    axis.scatter(
        beyond_two,
        np.full_like(beyond_two, -9.0),
        marker="|",
        s=75,
        color=COLORS["orange"],
        label=f"|z|>2: {len(beyond_two)}",
    )
    axis.scatter(
        beyond_three,
        np.full_like(beyond_three, -16.0),
        marker="|",
        s=85,
        color=COLORS["red"],
        label=f"|z|>3: {len(beyond_three)}",
    )
    axis.set(
        title=f"{title}\nexcess kurtosis={kurtosis:.2f}",
        xlabel="Value in sample SD units",
        ylabel="Count",
        xlim=(-8, 8),
    )
    axis.legend(loc="upper right", fontsize=8)


def create_kurtosis_animation(
    output_path: Path = GIF_DIRECTORY / "kurtosis_tail_behavior.gif",
) -> Path:
    """Compare tail families and animate kurtosis reacting to rare extremes."""
    configure_matplotlib()
    rng = np.random.default_rng(42)
    sample_size = 1_600
    normal = rng.normal(0.0, 1.0, sample_size)
    uniform = rng.uniform(-np.sqrt(3), np.sqrt(3), sample_size)
    student_t = rng.standard_t(df=5, size=sample_size) * np.sqrt(3 / 5)
    base = rng.normal(0.0, 1.0, sample_size)
    magnitudes = np.concatenate([np.linspace(2.0, 12.0, 23), np.full(5, 12.0)])
    figure, axes = plt.subplots(2, 2, figsize=(13, 9.2))

    def update(frame: int) -> None:
        for axis in axes.flat:
            axis.clear()
        _draw_tail_histogram(axes[0, 0], uniform, "Uniform: light tails", COLORS["green"])
        _draw_tail_histogram(axes[0, 1], normal, "Normal reference", COLORS["blue"])
        _draw_tail_histogram(
            axes[1, 0],
            student_t,
            "Student-t (df=5): heavy tails",
            COLORS["purple"],
        )

        magnitude = float(magnitudes[frame])
        contaminated = base.copy()
        contaminated[:4] = [-magnitude, -0.85 * magnitude, 0.85 * magnitude, magnitude]
        standardized = _standardize(contaminated)
        kurtosis = sample_excess_kurtosis(standardized)
        axes[1, 1].hist(
            standardized,
            bins=np.linspace(-12, 12, 75),
            color=COLORS["cyan"],
            alpha=0.72,
        )
        tail = standardized[np.abs(standardized) > 3]
        axes[1, 1].scatter(
            tail,
            np.full_like(tail, -10.0),
            marker="|",
            s=90,
            color=COLORS["red"],
            label=f"|z|>3: {len(tail)}",
        )
        axes[1, 1].set(
            title="Four rare observations move outward\n"
            f"excess kurtosis={kurtosis:.2f}",
            xlabel="Value in current sample SD units",
            ylabel="Count",
            xlim=(-12, 12),
        )
        axes[1, 1].legend(loc="upper right")
        axes[1, 1].text(
            0.03,
            0.93,
            f"Raw extreme magnitude: ±{magnitude:.1f}",
            transform=axes[1, 1].transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )

        figure.suptitle(
            "Kurtosis is about tail contribution, not peak height alone\n"
            "All reference samples are centered and scaled to variance 1",
            y=0.98,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.90))

    return save_animation(
        figure,
        update,
        len(magnitudes),
        output_path,
        fps=5,
        dpi=82,
    )


def create_latency_percentiles_figure(
    output_path: Path = IMAGE_DIRECTORY / "latency_percentiles.png",
) -> Path:
    """Show why mean latency does not characterize user tail experience."""
    configure_matplotlib()
    workload = synthetic_ai_workload()
    latency = workload["latency_ms"].to_numpy()
    statistics = {
        "Mean": float(np.mean(latency)),
        "P50": float(np.quantile(latency, 0.50)),
        "P90": float(np.quantile(latency, 0.90)),
        "P95": float(np.quantile(latency, 0.95)),
        "P99": float(np.quantile(latency, 0.99)),
    }
    colors = {
        "Mean": COLORS["dark"],
        "P50": COLORS["green"],
        "P90": COLORS["cyan"],
        "P95": COLORS["orange"],
        "P99": COLORS["red"],
    }
    upper = float(np.quantile(latency, 0.998))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    clipped = latency[latency <= upper]
    axes[0].hist(clipped, bins=60, color=COLORS["blue"], alpha=0.72)
    for label, value in statistics.items():
        axes[0].axvline(
            value,
            color=colors[label],
            linewidth=2.0,
            linestyle="--" if label == "Mean" else "-",
            label=f"{label}={value:.0f} ms",
        )
    axes[0].set(
        title="Most requests cluster well below the tail",
        xlabel="Latency (ms; axis clipped at P99.8)",
        ylabel="Synthetic request count",
        xlim=(0, upper),
    )
    axes[0].legend(loc="upper right")

    sorted_latency = np.sort(latency)
    percentile_rank = 100 * np.arange(1, len(latency) + 1) / len(latency)
    axes[1].plot(
        percentile_rank,
        sorted_latency,
        color=COLORS["blue"],
        linewidth=2.2,
    )
    axes[1].fill_between(
        percentile_rank,
        sorted_latency,
        where=percentile_rank >= 95,
        color=COLORS["red"],
        alpha=0.22,
        label="Slowest 5% of requests",
    )
    label_offsets = {
        50: (-10, 14),
        90: (-28, 18),
        95: (8, 27),
        99: (-10, 14),
    }
    for percentile in (50, 90, 95, 99):
        value = statistics[f"P{percentile}"]
        axes[1].scatter(
            percentile,
            value,
            s=58,
            color=colors[f"P{percentile}"],
            zorder=3,
        )
        axes[1].annotate(
            f"P{percentile}\n{value:.0f} ms",
            (percentile, value),
            xytext=label_offsets[percentile],
            textcoords="offset points",
            ha="right" if percentile != 95 else "left",
        )
    axes[1].set(
        title="Tail latency accelerates near the highest percentiles",
        xlabel="Percentile rank",
        ylabel="Latency (ms)",
        xlim=(0, 100),
        yscale="log",
    )
    axes[1].legend(loc="upper left")

    figure.suptitle(
        "Synthetic educational AI workload: mean latency can hide tail latency\n"
        "Most users are fast; a small fraction experience much slower responses",
        y=1.02,
    )
    figure.tight_layout()
    return save_figure(figure, output_path)


def generate_distribution_shape_visuals() -> list[Path]:
    """Generate skewness, kurtosis, and latency-tail assets."""
    ensure_output_directories()
    return [
        create_skewness_comparison(),
        create_skewness_animation(),
        create_kurtosis_animation(),
        create_latency_percentiles_figure(),
    ]


def main() -> None:
    """Generate this module's outputs independently."""
    generated = generate_distribution_shape_visuals()
    print(f"Generated {len(generated)} distribution-shape visualizations.")


if __name__ == "__main__":
    main()
