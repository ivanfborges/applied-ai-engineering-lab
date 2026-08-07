"""Animate center and robust-spread statistics under extreme observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visual_utils import (
    COLORS,
    GIF_DIRECTORY,
    configure_matplotlib,
    ensure_output_directories,
    save_animation,
    unscaled_mad,
)

import matplotlib.pyplot as plt


def create_mean_vs_median_animation(
    output_path: Path = GIF_DIRECTORY / "mean_vs_median_outlier.gif",
) -> Path:
    """Show one observation pulling the mean while the median stays stable."""
    configure_matplotlib()
    base = np.arange(10.0, 21.0)
    moving_values = np.concatenate(
        [np.geomspace(20.0, 1_000.0, 27), np.full(6, 1_000.0)]
    )
    figure, (zoom_axis, full_axis) = plt.subplots(
        1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.05, 1.6]}
    )

    def update(frame: int) -> None:
        outlier = float(moving_values[frame])
        values = np.append(base, outlier)
        current_mean = float(np.mean(values))
        current_median = float(np.median(values))

        for axis in (zoom_axis, full_axis):
            axis.clear()

        zoom_axis.scatter(
            base,
            np.zeros_like(base),
            s=58,
            color=COLORS["blue"],
            label="Stable central observations",
            zorder=3,
        )
        zoom_axis.axvline(
            current_median,
            color=COLORS["green"],
            linewidth=2.5,
            label=f"Median = {current_median:.2f}",
        )
        zoom_axis.axvline(
            current_mean,
            color=COLORS["red"],
            linewidth=2.2,
            linestyle="--",
            label=f"Mean = {current_mean:.2f}",
        )
        zoom_axis.set(
            title="Zoom: the ordered center barely changes",
            xlabel="Value",
            yticks=[],
            xlim=(8, 26),
            ylim=(-0.3, 0.55),
        )
        zoom_axis.legend(loc="upper left")

        full_axis.scatter(
            base,
            np.zeros_like(base),
            s=50,
            color=COLORS["blue"],
            zorder=3,
        )
        full_axis.scatter(
            [outlier],
            [0],
            s=95,
            marker="D",
            color=COLORS["orange"],
            label=f"Moving observation = {outlier:.1f}",
            zorder=4,
        )
        full_axis.axvline(
            current_median,
            color=COLORS["green"],
            linewidth=2.5,
            label="Median",
        )
        full_axis.axvline(
            current_mean,
            color=COLORS["red"],
            linewidth=2.2,
            linestyle="--",
            label="Mean",
        )
        full_axis.annotate(
            "The mean uses magnitude",
            xy=(current_mean, 0.0),
            xytext=(current_mean, 0.55),
            arrowprops={"arrowstyle": "->", "color": COLORS["red"]},
            color=COLORS["red"],
            ha="center",
        )
        full_axis.set(
            title="Full range: one magnitude becomes increasingly extreme",
            xlabel="Value",
            yticks=[],
            xlim=(0, 1_030),
            ylim=(-0.3, 0.85),
        )
        full_axis.legend(loc="upper right")

        figure.suptitle(
            "Mean vs median under one extreme observation\n"
            f"mean={current_mean:.2f}   median={current_median:.2f}   "
            f"difference={current_mean - current_median:.2f}",
            y=0.98,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.86))

    return save_animation(
        figure,
        update,
        len(moving_values),
        output_path,
        fps=6,
        dpi=88,
    )


def create_robust_statistics_animation(
    output_path: Path = GIF_DIRECTORY / "robust_statistics.gif",
) -> Path:
    """Compare sensitive and robust summaries as one value moves outward."""
    configure_matplotlib()
    rng = np.random.default_rng(42)
    base = rng.normal(100.0, 10.0, 260)
    baseline = {
        "mean": float(np.mean(base)),
        "median": float(np.median(base)),
        "std": float(np.std(base, ddof=1)),
        "iqr": float(np.quantile(base, 0.75) - np.quantile(base, 0.25)),
        "mad": unscaled_mad(base),
    }
    extremes = np.concatenate(
        [np.linspace(110.0, 650.0, 25), np.full(5, 650.0)]
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.7))

    def update(frame: int) -> None:
        extreme = float(extremes[frame])
        values = np.append(base, extreme)
        current = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)),
            "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "mad": unscaled_mad(values),
        }
        for axis in axes:
            axis.clear()

        axes[0].hist(
            values,
            bins=np.linspace(60, 670, 50),
            color=COLORS["blue"],
            alpha=0.72,
        )
        axes[0].axvline(
            current["mean"],
            color=COLORS["red"],
            linewidth=2,
            linestyle="--",
            label=f"Mean {current['mean']:.1f}",
        )
        axes[0].axvline(
            current["median"],
            color=COLORS["green"],
            linewidth=2,
            label=f"Median {current['median']:.1f}",
        )
        axes[0].set(
            title=f"One injected value = {extreme:.0f}",
            xlabel="Synthetic metric value",
            ylabel="Count",
            xlim=(60, 670),
        )
        axes[0].legend(loc="upper right")

        center_names = ["Mean", "Median"]
        center_change = [
            (current["mean"] - baseline["mean"]) / baseline["std"],
            (current["median"] - baseline["median"]) / baseline["std"],
        ]
        axes[1].bar(
            center_names,
            center_change,
            color=[COLORS["red"], COLORS["green"]],
        )
        axes[1].axhline(0, color=COLORS["dark"], linewidth=1)
        axes[1].set(
            title="Center displacement",
            ylabel="Change in baseline standard deviations",
            ylim=(-0.05, 0.24),
        )
        for index, value in enumerate(center_change):
            axes[1].text(index, value + 0.008, f"{value:+.3f}", ha="center")

        spread_names = ["Std", "IQR", "MAD"]
        spread_ratio = [
            current["std"] / baseline["std"],
            current["iqr"] / baseline["iqr"],
            current["mad"] / baseline["mad"],
        ]
        axes[2].bar(
            spread_names,
            spread_ratio,
            color=[COLORS["red"], COLORS["cyan"], COLORS["purple"]],
        )
        axes[2].axhline(
            1.0,
            color=COLORS["dark"],
            linewidth=1.3,
            linestyle="--",
            label="Baseline",
        )
        axes[2].set(
            title="Spread relative to baseline",
            ylabel="Current / baseline",
            ylim=(0, 4.2),
        )
        axes[2].legend(loc="upper right")
        for index, value in enumerate(spread_ratio):
            axes[2].text(index, value + 0.08, f"{value:.2f}×", ha="center")

        figure.suptitle(
            "Robust statistics resist extreme magnitudes\n"
            "Median, IQR, and MAD depend mainly on order and the central data",
            y=0.98,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.86))

    return save_animation(
        figure,
        update,
        len(extremes),
        output_path,
        fps=6,
        dpi=88,
    )


def generate_central_tendency_visuals() -> list[Path]:
    """Generate all center-and-robustness animations."""
    ensure_output_directories()
    return [
        create_mean_vs_median_animation(),
        create_robust_statistics_animation(),
    ]


def main() -> None:
    """Generate this module's outputs independently."""
    generated = generate_central_tendency_visuals()
    print(f"Generated {len(generated)} central-tendency visualizations.")


if __name__ == "__main__":
    main()
