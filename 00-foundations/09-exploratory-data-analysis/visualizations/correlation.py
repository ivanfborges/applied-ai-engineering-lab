"""Visualize covariance, correlation, dependence, leverage, and aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visual_utils import (
    COLORS,
    GIF_DIRECTORY,
    IMAGE_DIRECTORY,
    anscombe_quartet,
    configure_matplotlib,
    ensure_output_directories,
    save_animation,
    save_figure,
    simpsons_paradox_data,
)

import matplotlib.pyplot as plt


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return least-squares slope and intercept for a visual trend line."""
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def create_covariance_geometry(
    output_path: Path = IMAGE_DIRECTORY / "covariance_geometry.png",
) -> Path:
    """Show positive and negative cross-deviation contributions by quadrant."""
    configure_matplotlib()
    rng = np.random.default_rng(42)
    x = np.linspace(-3.0, 3.0, 80)
    datasets = [
        ("Positive covariance", x + rng.normal(0, 0.8, len(x))),
        ("Approximately zero covariance", rng.normal(0, 1.7, len(x))),
        ("Negative covariance", -x + rng.normal(0, 0.8, len(x))),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)

    for axis, (title, y) in zip(axes, datasets, strict=True):
        mean_x = float(np.mean(x))
        mean_y = float(np.mean(y))
        contributions = (x - mean_x) * (y - mean_y)
        positive = contributions >= 0
        covariance = float(np.cov(x, y, ddof=1)[0, 1])
        axis.scatter(
            x[positive],
            y[positive],
            color=COLORS["green"],
            s=34,
            alpha=0.75,
            label="Same signs → positive term",
        )
        axis.scatter(
            x[~positive],
            y[~positive],
            color=COLORS["red"],
            s=34,
            alpha=0.75,
            label="Opposite signs → negative term",
        )
        axis.axvline(mean_x, color=COLORS["dark"], linewidth=1.5)
        axis.axhline(mean_y, color=COLORS["dark"], linewidth=1.5)
        axis.scatter(
            mean_x,
            mean_y,
            marker="X",
            s=95,
            color=COLORS["purple"],
            label=r"$(\bar{x}, \bar{y})$",
            zorder=4,
        )
        axis.set(
            title=f"{title}\nsample covariance={covariance:.2f}",
            xlabel="x",
            ylabel="y",
            xlim=(-3.5, 3.5),
            ylim=(-4.5, 4.5),
        )
        axis.legend(loc="upper left", fontsize=7.7)

    figure.suptitle(
        r"Covariance averages cross-products $(x_i-\bar{x})(y_i-\bar{y})$",
        y=1.02,
    )
    figure.tight_layout()
    return save_figure(figure, output_path)


def _orthogonal_noise(x: np.ndarray, seed: int) -> np.ndarray:
    """Create centered unit-variance noise with zero sample correlation to x."""
    rng = np.random.default_rng(seed)
    centered_x = x - np.mean(x)
    noise = rng.normal(0.0, 1.0, len(x))
    noise -= np.mean(noise)
    projection = np.dot(noise, centered_x) / np.dot(centered_x, centered_x)
    noise -= projection * centered_x
    return noise / np.std(noise, ddof=0)


def create_pearson_noise_levels(
    output_path: Path = IMAGE_DIRECTORY / "pearson_noise_levels.png",
) -> Path:
    """Map exact target correlations to their scatter geometry."""
    configure_matplotlib()
    x = np.linspace(-2.5, 2.5, 180)
    x_standardized = (x - np.mean(x)) / np.std(x, ddof=0)
    noise = _orthogonal_noise(x_standardized, seed=42)
    targets = [0.99, 0.80, 0.50, 0.00, -0.80, -0.99]
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 9.0), sharex=True, sharey=True)

    for axis, target in zip(axes.flat, targets, strict=True):
        direction = 1.0 if target >= 0 else -1.0
        y = (
            direction * abs(target) * x_standardized
            + np.sqrt(1 - target**2) * noise
        )
        actual = float(np.corrcoef(x_standardized, y)[0, 1])
        axis.scatter(
            x_standardized,
            y,
            s=20,
            color=COLORS["blue"] if target >= 0 else COLORS["orange"],
            alpha=0.65,
        )
        slope, intercept = _fit_line(x_standardized, y)
        axis.plot(
            x_standardized,
            slope * x_standardized + intercept,
            color=COLORS["red"],
            linewidth=2,
        )
        axis.set(
            title=f"Pearson r = {actual:.2f}",
            xlabel="Standardized x",
            ylabel="Standardized y",
            xlim=(-2.2, 2.2),
            ylim=(-3.4, 3.4),
        )

    figure.suptitle(
        "Pearson correlation measures how tightly points follow a line\n"
        "Noise expands the cloud; sign controls its direction",
        y=1.01,
    )
    figure.tight_layout()
    return save_figure(figure, output_path)


def create_pearson_vs_spearman(
    output_path: Path = IMAGE_DIRECTORY / "pearson_vs_spearman.png",
) -> Path:
    """Compare linear and monotonic nonlinear associations."""
    configure_matplotlib()
    rng = np.random.default_rng(42)
    linear_x = np.linspace(-2.0, 2.0, 220)
    linear_y = 2.0 * linear_x + rng.normal(0.0, 0.7, len(linear_x))
    exponential_x = np.linspace(0.0, 4.0, 220)
    exponential_y = np.exp(exponential_x) + rng.normal(
        0.0, 0.35, len(exponential_x)
    )
    logarithmic_x = np.geomspace(0.1, 30.0, 220)
    logarithmic_y = np.log(logarithmic_x) + rng.normal(
        0.0, 0.08, len(logarithmic_x)
    )
    datasets = [
        ("Linear: y = 2x + noise", linear_x, linear_y),
        ("Monotonic nonlinear: y = exp(x)", exponential_x, exponential_y),
        ("Monotonic saturating: y = log(x)", logarithmic_x, logarithmic_y),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.1))

    for axis, (title, x, y) in zip(axes, datasets, strict=True):
        pearson = float(np.corrcoef(x, y)[0, 1])
        ranks_x = np.argsort(np.argsort(x))
        ranks_y = np.argsort(np.argsort(y))
        spearman = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
        axis.scatter(x, y, s=21, color=COLORS["blue"], alpha=0.62)
        axis.set(title=title, xlabel="x", ylabel="y")
        axis.text(
            0.04,
            0.94,
            f"Pearson r = {pearson:.3f}\nSpearman ρ = {spearman:.3f}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "0.8"},
        )

    figure.suptitle(
        "Pearson asks 'how linear?'; Spearman asks 'how monotonic?'",
        y=1.02,
    )
    figure.tight_layout()
    return save_figure(figure, output_path)


def create_nonlinear_dependence(
    output_path: Path = IMAGE_DIRECTORY / "nonlinear_dependence.png",
) -> Path:
    """Show deterministic dependence with zero Pearson correlation."""
    configure_matplotlib()
    x = np.linspace(-10.0, 10.0, 501)
    y = x**2
    angle = np.linspace(0.0, 2 * np.pi, 500, endpoint=False)
    circle_x = np.cos(angle)
    circle_y = np.sin(angle)
    datasets = [
        ("Deterministic function: y = x²", x, y),
        ("Deterministic constraint: x² + y² = 1", circle_x, circle_y),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for axis, (title, current_x, current_y) in zip(
        axes, datasets, strict=True
    ):
        pearson = float(np.corrcoef(current_x, current_y)[0, 1])
        axis.scatter(
            current_x,
            current_y,
            s=20,
            color=COLORS["purple"],
            alpha=0.70,
        )
        axis.set(
            title=f"{title}\nPearson r = {pearson:.6f}",
            xlabel="x",
            ylabel="y",
        )
        axis.text(
            0.5,
            0.06,
            "Low Pearson correlation ≠ independence",
            transform=axis.transAxes,
            ha="center",
            color=COLORS["red"],
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )

    figure.suptitle(
        "Pearson can miss perfect nonlinear dependence",
        y=1.02,
    )
    figure.tight_layout()
    return save_figure(figure, output_path)


def create_outlier_correlation_animation(
    output_path: Path = GIF_DIRECTORY / "outlier_effect_on_correlation.gif",
) -> Path:
    """Move one high-leverage point and update Pearson correlation."""
    configure_matplotlib()
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, 90)
    y = 0.25 * x + rng.normal(0.0, 1.0, 90)
    baseline_r = float(np.corrcoef(x, y)[0, 1])
    positions = np.concatenate([np.linspace(0.0, 11.0, 25), np.full(5, 11.0)])
    figure, axis = plt.subplots(figsize=(8.5, 6.2))

    def update(frame: int) -> None:
        position = float(positions[frame])
        current_x = np.append(x, position)
        current_y = np.append(y, position)
        current_r = float(np.corrcoef(current_x, current_y)[0, 1])
        axis.clear()
        axis.scatter(
            x,
            y,
            s=30,
            color=COLORS["blue"],
            alpha=0.62,
            label=f"Base cloud: r={baseline_r:.2f}",
        )
        axis.scatter(
            position,
            position,
            marker="D",
            s=105,
            color=COLORS["red"],
            label="One moving high-leverage point",
            zorder=4,
        )
        slope, intercept = _fit_line(current_x, current_y)
        line_x = np.array([-3.5, 11.5])
        axis.plot(
            line_x,
            slope * line_x + intercept,
            color=COLORS["orange"],
            linewidth=2.2,
            label=f"Current fit: r={current_r:.2f}",
        )
        axis.set(
            title="One observation can rotate the fitted relationship",
            xlabel="x",
            ylabel="y",
            xlim=(-3.5, 11.5),
            ylim=(-3.5, 11.5),
        )
        axis.text(
            0.03,
            0.96,
            f"r before point = {baseline_r:.3f}\n"
            f"r with point = {current_r:.3f}\n"
            f"point = ({position:.1f}, {position:.1f})",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "0.8"},
        )
        axis.legend(loc="lower right")
        figure.tight_layout()

    return save_animation(
        figure,
        update,
        len(positions),
        output_path,
        fps=6,
        dpi=90,
    )


def create_correlation_causation_figure(
    output_path: Path = IMAGE_DIRECTORY / "correlation_causation.png",
) -> Path:
    """Use a common synthetic cause to create an observed association."""
    configure_matplotlib()
    rng = np.random.default_rng(42)
    system_load = rng.uniform(0.0, 1.0, 500)
    requests = 180 + 750 * system_load + rng.normal(0, 65, len(system_load))
    latency = 120 + 520 * system_load + rng.normal(0, 55, len(system_load))
    observed_r = float(np.corrcoef(requests, latency)[0, 1])
    residual_requests = requests - (
        np.polyfit(system_load, requests, 1)[0] * system_load
        + np.polyfit(system_load, requests, 1)[1]
    )
    residual_latency = latency - (
        np.polyfit(system_load, latency, 1)[0] * system_load
        + np.polyfit(system_load, latency, 1)[1]
    )
    residual_r = float(np.corrcoef(residual_requests, residual_latency)[0, 1])
    figure, axes = plt.subplots(
        1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [0.9, 1.4]}
    )

    axes[0].axis("off")
    boxes = {
        "System load\n(hidden/common cause)": (0.50, 0.78),
        "Number of\nrequests": (0.22, 0.28),
        "Latency": (0.78, 0.28),
    }
    for label, (x_position, y_position) in boxes.items():
        axes[0].text(
            x_position,
            y_position,
            label,
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#EFF6FF",
                "edgecolor": COLORS["blue"],
                "linewidth": 2,
            },
        )
    for target in ((0.22, 0.36), (0.78, 0.36)):
        axes[0].annotate(
            "",
            xy=target,
            xytext=(0.50, 0.70),
            xycoords=axes[0].transAxes,
            textcoords=axes[0].transAxes,
            arrowprops={"arrowstyle": "->", "linewidth": 2, "color": COLORS["dark"]},
        )
    axes[0].text(
        0.50,
        0.08,
        "The graph encodes the data-generating story.\n"
        "The requests↔latency edge is not identified by correlation alone.",
        transform=axes[0].transAxes,
        ha="center",
        color=COLORS["dark"],
    )

    scatter = axes[1].scatter(
        requests,
        latency,
        c=system_load,
        cmap="viridis",
        s=28,
        alpha=0.70,
    )
    axes[1].set(
        title=f"Observed requests–latency correlation: r={observed_r:.2f}",
        xlabel="Synthetic requests per interval",
        ylabel="Synthetic latency (ms)",
    )
    axes[1].text(
        0.04,
        0.95,
        f"Correlation after linear load adjustment: {residual_r:.2f}",
        transform=axes[1].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )
    figure.colorbar(scatter, ax=axes[1], label="Synthetic system load")
    figure.suptitle(
        "Correlation does not identify the causal mechanism",
        y=1.02,
    )
    figure.tight_layout()
    return save_figure(figure, output_path)


def create_simpsons_paradox_figure(
    output_path: Path = IMAGE_DIRECTORY / "simpsons_paradox.png",
) -> Path:
    """Contrast the aggregate trend with conditional group trends."""
    configure_matplotlib()
    data = simpsons_paradox_data()
    overall_r = float(data[["x", "y"]].corr().iloc[0, 1])
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True, sharey=True)

    axes[0].scatter(
        data["x"],
        data["y"],
        color=COLORS["gray"],
        alpha=0.62,
        s=28,
    )
    slope, intercept = _fit_line(data["x"].to_numpy(), data["y"].to_numpy())
    line_x = np.array([data["x"].min(), data["x"].max()])
    axes[0].plot(
        line_x,
        slope * line_x + intercept,
        color=COLORS["red"],
        linewidth=2.5,
    )
    axes[0].set(
        title=f"Aggregated: positive association (r={overall_r:.2f})",
        xlabel="x",
        ylabel="y",
    )

    palette = {"A": COLORS["blue"], "B": COLORS["green"], "C": COLORS["orange"]}
    group_text: list[str] = []
    for group, frame in data.groupby("group", sort=True):
        x_values = frame["x"].to_numpy()
        y_values = frame["y"].to_numpy()
        correlation = float(np.corrcoef(x_values, y_values)[0, 1])
        group_text.append(f"{group}: r={correlation:.2f}")
        axes[1].scatter(
            x_values,
            y_values,
            color=palette[group],
            alpha=0.72,
            s=30,
            label=f"Group {group}: r={correlation:.2f}",
        )
        slope, intercept = _fit_line(x_values, y_values)
        group_line = np.array([x_values.min(), x_values.max()])
        axes[1].plot(
            group_line,
            slope * group_line + intercept,
            color=palette[group],
            linewidth=2.2,
        )
    axes[1].set(
        title="Conditioned on group: every association is negative",
        xlabel="x",
        ylabel="y",
    )
    axes[1].legend(loc="upper right")

    figure.suptitle(
        "Simpson's paradox: aggregation can reverse the visible relationship",
        y=1.02,
    )
    figure.tight_layout()
    return save_figure(figure, output_path)


def create_anscombe_figure(
    output_path: Path = IMAGE_DIRECTORY / "anscombes_quartet.png",
) -> Path:
    """Render Anscombe's Quartet with its nearly identical summaries."""
    configure_matplotlib()
    datasets = anscombe_quartet()
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)

    for axis, (title, x, y) in zip(axes.flat, datasets, strict=True):
        slope, intercept = _fit_line(x, y)
        correlation = float(np.corrcoef(x, y)[0, 1])
        axis.scatter(x, y, s=58, color=COLORS["blue"], alpha=0.82)
        line_x = np.array([3.0, 20.0])
        axis.plot(
            line_x,
            slope * line_x + intercept,
            color=COLORS["red"],
            linewidth=2,
        )
        axis.set(
            title=title,
            xlabel="x",
            ylabel="y",
            xlim=(3, 20),
            ylim=(2, 14),
        )
        axis.text(
            0.04,
            0.94,
            f"x̄={np.mean(x):.2f}, ȳ={np.mean(y):.2f}\n"
            f"Var(x)={np.var(x, ddof=1):.2f}, Var(y)={np.var(y, ddof=1):.2f}\n"
            f"r={correlation:.3f}, y≈{intercept:.2f}+{slope:.2f}x",
            transform=axis.transAxes,
            va="top",
            fontsize=8.7,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )

    figure.suptitle(
        "Anscombe's Quartet: similar summaries, radically different structure\n"
        "Summary statistics are not a substitute for visualization",
        y=1.01,
    )
    figure.tight_layout()
    return save_figure(figure, output_path)


def generate_correlation_visuals() -> list[Path]:
    """Generate all association and dependence assets."""
    ensure_output_directories()
    return [
        create_covariance_geometry(),
        create_pearson_noise_levels(),
        create_pearson_vs_spearman(),
        create_nonlinear_dependence(),
        create_outlier_correlation_animation(),
        create_correlation_causation_figure(),
        create_simpsons_paradox_figure(),
        create_anscombe_figure(),
    ]


def main() -> None:
    """Generate this module's outputs independently."""
    generated = generate_correlation_visuals()
    print(f"Generated {len(generated)} correlation visualizations.")


if __name__ == "__main__":
    main()
