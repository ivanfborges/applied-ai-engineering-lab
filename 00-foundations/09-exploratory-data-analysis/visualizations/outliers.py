"""Visualize IQR anatomy and distribution-sensitive outlier heuristics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visual_utils import (
    COLORS,
    IMAGE_DIRECTORY,
    configure_matplotlib,
    ensure_output_directories,
    save_figure,
)

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def create_iqr_boxplot_anatomy(
    output_path: Path = IMAGE_DIRECTORY / "iqr_boxplot_anatomy.png",
) -> Path:
    """Label quartiles, fences, whiskers, and potential outliers directly."""
    configure_matplotlib()
    values = np.array(
        [41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 59, 71, 78],
        dtype=float,
    )
    q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    inside = values[(values >= lower_fence) & (values <= upper_fence)]
    outliers = values[(values < lower_fence) | (values > upper_fence)]
    lower_whisker = float(np.min(inside))
    upper_whisker = float(np.max(inside))

    figure, axis = plt.subplots(figsize=(13, 5.4))
    axis.hlines(0, lower_whisker, upper_whisker, color=COLORS["dark"], linewidth=2)
    axis.vlines(
        [lower_whisker, upper_whisker],
        -0.12,
        0.12,
        color=COLORS["dark"],
        linewidth=2,
    )
    axis.add_patch(
        Rectangle(
            (q1, -0.27),
            iqr,
            0.54,
            facecolor=COLORS["blue"],
            edgecolor=COLORS["dark"],
            alpha=0.34,
            linewidth=2,
        )
    )
    axis.vlines(median, -0.27, 0.27, color=COLORS["green"], linewidth=3)
    axis.vlines(
        [lower_fence, upper_fence],
        -0.44,
        0.44,
        color=COLORS["orange"],
        linestyle="--",
        linewidth=2,
    )
    axis.scatter(
        inside,
        np.full_like(inside, -0.47),
        color=COLORS["gray"],
        s=35,
        label="Observed value inside fences",
    )
    axis.scatter(
        outliers,
        np.full_like(outliers, 0.0),
        color=COLORS["red"],
        marker="D",
        s=75,
        label="Potential statistical outlier",
        zorder=4,
    )

    labels = [
        (q1, "Q1", -0.75),
        (median, "Median", 0.72),
        (q3, "Q3", -0.75),
        (lower_fence, "Lower fence\nQ1 − 1.5×IQR", 0.78),
        (upper_fence, "Upper fence\nQ3 + 1.5×IQR", 0.78),
    ]
    for x_value, label, y_text in labels:
        axis.annotate(
            f"{label}\n{x_value:.1f}",
            xy=(x_value, 0.0),
            xytext=(x_value, y_text),
            arrowprops={"arrowstyle": "->", "color": COLORS["dark"]},
            ha="center",
            va="center",
        )
    axis.annotate(
        f"IQR = Q3 − Q1 = {iqr:.1f}",
        xy=((q1 + q3) / 2, 0.27),
        xytext=((q1 + q3) / 2, 1.05),
        arrowprops={"arrowstyle": "-[", "color": COLORS["blue"]},
        ha="center",
        color=COLORS["blue"],
        fontweight="bold",
    )
    axis.set(
        title="IQR boxplot anatomy: a fence flags observations; it does not delete them",
        xlabel="Synthetic metric value",
        yticks=[],
        xlim=(30, 83),
        ylim=(-1.05, 1.35),
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    return save_figure(figure, output_path)


def _plot_detection_method(
    axis: plt.Axes,
    values: np.ndarray,
    flagged: np.ndarray,
    bounds: tuple[float, float],
    title: str,
) -> None:
    """Draw a deterministic jittered strip and detection thresholds."""
    rng = np.random.default_rng(7)
    jitter = rng.normal(0.0, 0.035, len(values))
    axis.scatter(
        values[~flagged],
        jitter[~flagged],
        s=16,
        color=COLORS["blue"],
        alpha=0.50,
        label="Not flagged",
    )
    axis.scatter(
        values[flagged],
        jitter[flagged],
        s=28,
        color=COLORS["red"],
        alpha=0.82,
        label="Flagged",
    )
    for bound in bounds:
        axis.axvline(
            bound,
            color=COLORS["orange"],
            linestyle="--",
            linewidth=2,
        )
    axis.set(
        title=f"{title}\nflagged {int(np.sum(flagged))} / {len(values)}",
        xlabel="Observed value",
        yticks=[],
        ylim=(-0.18, 0.22),
    )
    axis.legend(loc="upper right")


def create_zscore_vs_iqr_figure(
    output_path: Path = IMAGE_DIRECTORY / "zscore_vs_iqr.png",
) -> Path:
    """Compare Z-score and IQR flags under Gaussian and skewed samples."""
    configure_matplotlib()
    rng = np.random.default_rng(42)
    datasets = [
        ("Approximately Gaussian", rng.normal(0.0, 1.0, 700)),
        ("Strongly right-skewed log-normal", rng.lognormal(0.0, 0.85, 700)),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(14, 8.0))

    for row, (distribution_name, values) in enumerate(datasets):
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        z_bounds = (mean - 3 * std, mean + 3 * std)
        z_flagged = (values < z_bounds[0]) | (values > z_bounds[1])

        q1, q3 = np.quantile(values, [0.25, 0.75])
        iqr = q3 - q1
        iqr_bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        iqr_flagged = (values < iqr_bounds[0]) | (values > iqr_bounds[1])

        _plot_detection_method(
            axes[row, 0],
            values,
            z_flagged,
            z_bounds,
            f"{distribution_name}: |z| > 3",
        )
        _plot_detection_method(
            axes[row, 1],
            values,
            iqr_flagged,
            iqr_bounds,
            f"{distribution_name}: 1.5×IQR fences",
        )

    figure.suptitle(
        "Outlier heuristics encode different assumptions\n"
        "Mean/SD thresholds are distorted by skew; neither flag is an automatic removal rule",
        y=1.01,
    )
    figure.tight_layout()
    return save_figure(figure, output_path)


def generate_outlier_visuals() -> list[Path]:
    """Generate the IQR and outlier-rule assets."""
    ensure_output_directories()
    return [
        create_iqr_boxplot_anatomy(),
        create_zscore_vs_iqr_figure(),
    ]


def main() -> None:
    """Generate this module's outputs independently."""
    generated = generate_outlier_visuals()
    print(f"Generated {len(generated)} outlier visualizations.")


if __name__ == "__main__":
    main()
