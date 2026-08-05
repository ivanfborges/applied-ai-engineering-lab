"""Generate deterministic Probability Essentials PNG and GIF assets.

The script uses only synthetic data, a non-interactive Matplotlib backend, and
PillowWriter. It does not require ImageMagick, FFmpeg, network access, or an
external dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from visualizations import (
    EVENT_CATEGORY_ORDER,
    EVENT_COLORS,
    FRAUD_CATEGORY_ORDER,
    FRAUD_COLORS,
    allocate_integer_counts,
    bayes_posterior,
    bayes_surface_values,
    bernoulli_variance,
    conditional_population,
    die_monte_carlo,
    event_operation_mask,
    event_sample_space,
    expected_decision_threshold,
    fraud_outcome_counts,
    fraud_outcome_probabilities,
    joint_probability_table,
)


TOPIC_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = TOPIC_DIR / "outputs"
GIF_FPS = 8
GIF_DPI = 90
PNG_DPI = 140

EVENT_MARKERS = {
    "Neither": "o",
    "A only": "s",
    "B only": "D",
    "A ∩ B": "*",
}
FRAUD_MARKERS = {
    "True positive": "*",
    "False positive": "x",
    "False negative": "D",
    "True negative": "o",
}


def _style_axis(axis: Axes, title: str) -> None:
    """Apply consistent labels and grid styling to a Matplotlib axis."""
    axis.set_title(title, fontsize=12, fontweight="bold")
    axis.grid(alpha=0.20)


def _draw_event_grid(
    axis: Axes,
    operation: str,
    *,
    annotate_outcomes: bool,
) -> None:
    """Draw one event operation with marker shape and opacity encodings."""
    data = event_sample_space(100)
    selected = event_operation_mask(data, operation)
    for category in EVENT_CATEGORY_ORDER:
        subset = data.loc[data["category"] == category]
        selected_subset = selected.loc[subset.index].to_numpy()
        axis.scatter(
            subset["x"],
            subset["y"],
            s=np.where(selected_subset, 76, 34),
            c=EVENT_COLORS[category],
            marker=EVENT_MARKERS[category],
            alpha=np.where(selected_subset, 1.0, 0.16),
            edgecolors="#111827" if EVENT_MARKERS[category] not in {"x", "*"} else None,
            linewidths=0.6,
            label=category,
        )
        if annotate_outcomes:
            for row_index, row in subset.iterrows():
                axis.text(
                    row["x"],
                    row["y"],
                    str(int(row["outcome"])),
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="#111827",
                    alpha=1.0 if selected.loc[row_index] else 0.28,
                )
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(
        f"{operation}: {int(selected.sum())}/100 outcomes, P={selected.mean():.2f}",
        fontsize=10,
        fontweight="bold",
    )


def generate_event_operations(output_path: Path) -> None:
    """Generate a six-panel event-operations PNG."""
    operations = ("A", "B", "A ∪ B", "A ∩ B", "Aᶜ", "Bᶜ")
    figure, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    for axis, operation in zip(axes.ravel(), operations, strict=True):
        _draw_event_grid(axis, operation, annotate_outcomes=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Event operations on Ω={1,…,100}: A is divisible by 2; B by 5",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "Marker shape identifies membership; size and opacity identify the selected event.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    figure.savefig(output_path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(figure)


def generate_conditional_probability_animation(output_path: Path) -> None:
    """Animate the denominator change from Ω to B and then A∩B."""
    data = conditional_population(100, count_a=50, count_b=20, count_intersection=10)
    frames = 27
    figure, axis = plt.subplots(figsize=(8.5, 5.5))

    def draw(frame: int) -> None:
        axis.clear()
        phase = frame / (frames - 1)
        if phase < 1 / 3:
            stage = "1/3 — Start with the complete sample space Ω (denominator=100)"
            outside_opacity = 1.0
            intersection_scale = 1.0
        elif phase < 2 / 3:
            progress = (phase - 1 / 3) * 3
            stage = "2/3 — Condition on B: outcomes outside B fade away (denominator=20)"
            outside_opacity = max(0.08, 1.0 - 0.92 * progress)
            intersection_scale = 1.0
        else:
            progress = (phase - 2 / 3) * 3
            stage = "3/3 — Count A within B: A∩B is the numerator (10 of 20)"
            outside_opacity = 0.08
            intersection_scale = 1.0 + 1.0 * progress

        for category in EVENT_CATEGORY_ORDER:
            subset = data.loc[data["category"] == category]
            is_in_b = bool(subset["event_b"].iloc[0]) if not subset.empty else False
            opacity = 1.0 if is_in_b else outside_opacity
            size = 105 if category == "A ∩ B" else 64
            size *= intersection_scale if category == "A ∩ B" else 1.0
            axis.scatter(
                subset["x"],
                subset["y"],
                s=size,
                c=EVENT_COLORS[category],
                marker=EVENT_MARKERS[category],
                alpha=opacity,
                edgecolors=(
                    "#111827"
                    if EVENT_MARKERS[category] not in {"x", "*"}
                    else None
                ),
                linewidths=1.0,
                label=category,
            )

        axis.set_title(
            "Conditional probability changes the denominator\n" + stage,
            fontsize=12,
            fontweight="bold",
        )
        axis.text(
            0.02,
            0.02,
            "P(A | B) = P(A∩B) / P(B) = 10/20 = 0.50",
            transform=axis.transAxes,
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "#111827", "alpha": 0.92},
        )
        axis.text(
            0.98,
            0.98,
            f"Frame {frame + 1}/{frames}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#475569",
        )
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=4,
            frameon=False,
        )

    animation = FuncAnimation(figure, draw, frames=frames, interval=125)
    animation.save(output_path, writer=PillowWriter(fps=GIF_FPS), dpi=GIF_DPI)
    plt.close(figure)


def generate_independence_comparison(output_path: Path) -> None:
    """Generate negative, independent, and positive joint-distribution panels."""
    probability_a = 0.45
    probability_b = 0.35
    independent = probability_a * probability_b
    intersections = (0.06, independent, 0.30)
    labels = ("Negative dependence", "Independence", "Positive dependence")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.4))

    for axis, intersection, label in zip(
        axes,
        intersections,
        labels,
        strict=True,
    ):
        table = joint_probability_table(
            probability_a,
            probability_b,
            intersection,
        )
        matrix = table.to_numpy()
        image = axis.imshow(matrix, cmap="Blues", vmin=0.0, vmax=0.60)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.3f}\n({matrix[row, column]:.1%})",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#111827",
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
                )
        difference = intersection - independent
        axis.set_title(
            f"{label}\nP(A∩B)−P(A)P(B)={difference:+.4f}",
            fontsize=10,
            fontweight="bold",
        )
        axis.set_xticks([0, 1], ["B=False", "B=True"])
        axis.set_yticks([0, 1], ["A=False", "A=True"])
        axis.set_xlabel("Event B")
        axis.set_ylabel("Event A")

    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.03, label="Joint probability")
    figure.suptitle(
        "Dependence changes the 2×2 joint distribution while marginals stay fixed",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.02,
        "Independence means P(A∩B)=P(A)P(B); it is not mutual exclusivity.",
        ha="center",
        fontsize=10,
    )
    figure.subplots_adjust(left=0.07, right=0.91, bottom=0.18, top=0.78, wspace=0.35)
    figure.savefig(output_path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(figure)


def _draw_fraud_population(
    axis: Axes,
    prior: float,
    true_positive_rate: float,
    false_positive_rate: float,
    population_size: int,
) -> None:
    """Draw deterministic fraud outcome composition for one animation frame."""
    probabilities = fraud_outcome_probabilities(
        prior,
        true_positive_rate,
        false_positive_rate,
    )
    counts = allocate_integer_counts(list(probabilities.values()), population_size)
    categories = np.concatenate(
        [
            np.repeat(category, count)
            for category, count in zip(probabilities, counts, strict=True)
        ]
    )
    columns = int(np.ceil(np.sqrt(population_size)))
    index = np.arange(population_size)
    for category in FRAUD_CATEGORY_ORDER:
        mask = categories == category
        axis.scatter(
            index[mask] % columns,
            columns - 1 - index[mask] // columns,
            c=FRAUD_COLORS[category],
            marker=FRAUD_MARKERS[category],
            s=30 if category == "True negative" else 48,
            linewidths=0.8,
            edgecolors=(
                "#111827"
                if FRAUD_MARKERS[category] not in {"x", "*"}
                else None
            ),
            label=category,
        )
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def generate_bayes_base_rate_animation(output_path: Path) -> None:
    """Animate posterior changes as prior prevalence increases."""
    true_positive_rate = 0.90
    false_positive_rate = 0.05
    population_size = 400
    priors = np.linspace(0.0025, 0.15, 30)
    figure, axis = plt.subplots(figsize=(9, 6))

    def draw(frame: int) -> None:
        axis.clear()
        prior = float(priors[frame])
        _draw_fraud_population(
            axis,
            prior,
            true_positive_rate,
            false_positive_rate,
            population_size,
        )
        counts = fraud_outcome_counts(
            prior,
            true_positive_rate,
            false_positive_rate,
            population_size,
        )
        posterior = bayes_posterior(
            prior,
            true_positive_rate,
            false_positive_rate,
        )
        axis.set_title(
            "Bayes base-rate effect — detector behavior stays fixed\n"
            f"Prior={prior:.2%} | TP={counts['True positive']} | "
            f"FP={counts['False positive']} | "
            f"P(Fraud|Alert)={posterior:.2%}",
            fontsize=12,
            fontweight="bold",
        )
        axis.text(
            0.02,
            0.02,
            "TPR=90%, FPR=5%. Posterior denominator = true positives + false positives.",
            transform=axis.transAxes,
            fontsize=9.5,
            bbox={"facecolor": "white", "edgecolor": "#111827", "alpha": 0.92},
        )
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=2,
            frameon=False,
        )

    animation = FuncAnimation(
        figure,
        draw,
        frames=len(priors),
        interval=125,
    )
    animation.save(output_path, writer=PillowWriter(fps=GIF_FPS), dpi=GIF_DPI)
    plt.close(figure)


def generate_bayes_surface(output_path: Path) -> None:
    """Generate a static Matplotlib 3D Bayes surface."""
    true_positive_rate = 0.90
    priors = np.linspace(0.001, 0.25, 70)
    false_positive_rates = np.linspace(0.001, 0.30, 70)
    prior_grid, fpr_grid = np.meshgrid(priors, false_positive_rates)
    posterior_grid = bayes_surface_values(
        prior_grid,
        fpr_grid,
        true_positive_rate,
    )

    figure = plt.figure(figsize=(10, 7))
    axis = figure.add_subplot(111, projection="3d")
    surface = axis.plot_surface(
        prior_grid,
        fpr_grid,
        posterior_grid,
        cmap="viridis",
        alpha=0.88,
        linewidth=0,
        antialiased=True,
    )
    selected_prior = 0.01
    selected_fpr = 0.05
    selected_posterior = bayes_posterior(
        selected_prior,
        true_positive_rate,
        selected_fpr,
    )
    axis.scatter(
        [selected_prior],
        [selected_fpr],
        [selected_posterior],
        color="#DC2626",
        marker="D",
        s=70,
        edgecolor="#111827",
        label=f"Selected scenario: posterior={selected_posterior:.1%}",
    )
    axis.set(
        xlabel="Prior P(Fraud)",
        ylabel="False-positive rate P(Alert | not fraud)",
        zlabel="Posterior P(Fraud | Alert)",
        zlim=(0.0, 1.0),
    )
    axis.set_title(
        "Bayes surface: prevalence and false positives control alert precision\n"
        "True-positive rate fixed at 90%",
        fontsize=13,
        fontweight="bold",
    )
    axis.legend(loc="upper left")
    figure.colorbar(surface, ax=axis, shrink=0.65, pad=0.12, label="Posterior")
    figure.tight_layout()
    figure.savefig(output_path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(figure)


def generate_expected_value_balance(output_path: Path) -> None:
    """Generate a fair-die PMF and balance-point interpretation."""
    values = np.arange(1, 7, dtype=float)
    probabilities = np.full(6, 1 / 6)
    contributions = values * probabilities
    mean = float(np.sum(contributions))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9, 6.5),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    bars = axes[0].bar(
        values,
        probabilities,
        color="#2563EB",
        edgecolor="#111827",
        hatch="//",
    )
    axes[0].axvline(
        mean,
        color="#DC2626",
        linestyle="--",
        linewidth=2.5,
        label=f"E[X]={mean:.1f}",
    )
    axes[0].bar_label(
        bars,
        labels=[f"x·p={value:.3f}" for value in contributions],
        padding=4,
        fontsize=8,
    )
    axes[0].set(
        xlabel="Die outcome x",
        ylabel="Probability P(X=x)",
        ylim=(0.0, 0.24),
    )
    _style_axis(
        axes[0],
        "Fair-die expectation is the sum of weighted contributions",
    )
    axes[0].legend()

    axes[1].hlines(0.0, 1.0, 6.0, color="#334155", linewidth=5)
    axes[1].scatter(
        values,
        np.zeros_like(values),
        s=probabilities * 900,
        color="#2563EB",
        marker="o",
        edgecolor="#111827",
        label="Equal probability masses",
    )
    axes[1].scatter(
        [mean],
        [-0.12],
        marker="^",
        s=250,
        color="#DC2626",
        edgecolor="#111827",
        label="Balance point E[X]",
    )
    axes[1].set(
        xlim=(0.6, 6.4),
        ylim=(-0.20, 0.12),
        xlabel="Outcome value",
        yticks=[],
    )
    axes[1].set_title(
        "The expected value is the distribution's balance point—not an observable die face",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=2)
    figure.suptitle(
        "Expected value as a probability-weighted balance point",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.94))
    figure.savefig(output_path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(figure)


def generate_variance_spread_animation(output_path: Path) -> None:
    """Animate increasing spread while holding the mean fixed."""
    standard_deviations = np.linspace(0.45, 3.0, 30)
    x = np.linspace(-9.0, 9.0, 500)
    rng = np.random.default_rng(42)
    standardized_sample = rng.normal(size=160)
    standardized_sample -= standardized_sample.mean()
    figure, axis = plt.subplots(figsize=(8.5, 5.2))

    def draw(frame: int) -> None:
        axis.clear()
        sigma = float(standard_deviations[frame])
        density = np.exp(-0.5 * (x / sigma) ** 2) / (
            sigma * np.sqrt(2.0 * np.pi)
        )
        sample = standardized_sample * sigma
        axis.plot(
            x,
            density,
            color="#2563EB",
            linewidth=3,
            label="Gaussian density",
        )
        axis.scatter(
            sample,
            np.full_like(sample, -0.008),
            marker="|",
            s=45,
            color="#D97706",
            alpha=0.50,
            label="Fixed synthetic draws, rescaled",
        )
        axis.axvline(
            0.0,
            color="#DC2626",
            linestyle="--",
            linewidth=2.5,
            label="Mean μ=0",
        )
        axis.set(
            xlim=(-9.0, 9.0),
            ylim=(-0.02, 0.95),
            xlabel="Outcome",
            ylabel="Probability density",
        )
        axis.set_title(
            "Variance changes spread, not central location\n"
            f"μ=0.00 | σ={sigma:.2f} | Var(X)={sigma**2:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        axis.grid(alpha=0.20)
        axis.legend(loc="upper right")

    animation = FuncAnimation(
        figure,
        draw,
        frames=len(standard_deviations),
        interval=125,
    )
    animation.save(output_path, writer=PillowWriter(fps=GIF_FPS), dpi=GIF_DPI)
    plt.close(figure)


def generate_bernoulli_variance(output_path: Path) -> None:
    """Generate the Bernoulli variance curve and uncertainty maximum."""
    probability = np.linspace(0.0, 1.0, 401)
    variance = bernoulli_variance(probability)
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(probability, variance, color="#2563EB", linewidth=3)
    axis.fill_between(
        probability,
        variance,
        color="#93C5FD",
        alpha=0.35,
        hatch="//",
    )
    axis.scatter(
        [0.0, 0.5, 1.0],
        [0.0, 0.25, 0.0],
        color=["#64748B", "#DC2626", "#64748B"],
        marker="D",
        s=[55, 95, 55],
        edgecolor="#111827",
        zorder=3,
    )
    axis.annotate(
        "Maximum uncertainty\np=0.5, Var=0.25",
        xy=(0.5, 0.25),
        xytext=(0.62, 0.21),
        arrowprops={"arrowstyle": "->", "color": "#111827"},
        fontsize=10,
    )
    axis.annotate(
        "Near-certain outcomes\nvariance approaches 0",
        xy=(0.98, 0.0196),
        xytext=(0.66, 0.07),
        arrowprops={"arrowstyle": "->", "color": "#111827"},
        fontsize=9,
    )
    axis.set(
        title="Bernoulli variance: p(1−p)",
        xlabel="Success probability p",
        ylabel="Variance",
        xlim=(0.0, 1.0),
        ylim=(0.0, 0.28),
    )
    axis.grid(alpha=0.20)
    figure.tight_layout()
    figure.savefig(output_path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(figure)


def generate_monte_carlo_convergence_animation(output_path: Path) -> None:
    """Animate fair-die empirical mean convergence and absolute error."""
    max_sample_size = 20_000
    data = die_monte_carlo(max_sample_size, seed=42)
    frame_sizes = np.unique(
        np.geomspace(10, max_sample_size, num=32).astype(int)
    )
    figure, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    def draw(frame: int) -> None:
        for axis in axes:
            axis.clear()
        current_size = int(frame_sizes[frame])
        subset = data.iloc[:current_size]
        current_estimate = float(subset["estimate"].iloc[-1])
        current_error = float(subset["absolute_error"].iloc[-1])

        axes[0].plot(
            subset["sample_size"],
            subset["estimate"],
            color="#2563EB",
            linewidth=2,
            label="Empirical mean",
        )
        axes[0].axhline(
            3.5,
            color="#DC2626",
            linestyle="--",
            linewidth=2,
            label="Theoretical E[X]=3.5",
        )
        axes[0].scatter(
            [current_size],
            [current_estimate],
            color="#111827",
            marker="D",
            s=45,
            zorder=3,
        )
        axes[0].set_ylabel("Estimated mean")
        axes[0].set_title(
            "Monte Carlo convergence for a fair die\n"
            f"n={current_size:,} | estimate={current_estimate:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].legend(loc="upper right")
        axes[0].grid(alpha=0.20)

        axes[1].plot(
            subset["sample_size"],
            subset["absolute_error"],
            color="#D97706",
            linewidth=2,
            label="Absolute error",
        )
        axes[1].scatter(
            [current_size],
            [current_error],
            color="#111827",
            marker="x",
            s=55,
            zorder=3,
        )
        axes[1].set(
            xlabel="Sample size",
            ylabel="|Empirical mean − 3.5|",
            xlim=(1, max_sample_size),
        )
        axes[1].set_title(
            f"Remaining random fluctuation: absolute error={current_error:.4f}",
            fontsize=10.5,
            fontweight="bold",
        )
        axes[1].grid(alpha=0.20)
        axes[1].legend(loc="upper right")

    animation = FuncAnimation(
        figure,
        draw,
        frames=len(frame_sizes),
        interval=125,
    )
    animation.save(output_path, writer=PillowWriter(fps=GIF_FPS), dpi=GIF_DPI)
    plt.close(figure)


def generate_expected_cost(output_path: Path) -> None:
    """Generate a static expected-cost decision-threshold chart."""
    review_cost = 5.0
    missed_event_cost = 100.0
    threshold = expected_decision_threshold(review_cost, missed_event_cost)
    probability = np.linspace(0.0, 1.0, 301)
    no_review_cost = probability * missed_event_cost
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(
        probability,
        np.full_like(probability, review_cost),
        color="#2563EB",
        linestyle="--",
        linewidth=3,
        label="Review: Cᵣ = 5",
    )
    axis.plot(
        probability,
        no_review_cost,
        color="#DC2626",
        linewidth=3,
        label="Do not review: p × Cₘ",
    )
    axis.scatter(
        [threshold],
        [review_cost],
        color="#7C3AED",
        marker="D",
        s=90,
        edgecolor="#111827",
        zorder=3,
        label=f"Threshold p*={threshold:.2%}",
    )
    axis.fill_between(
        probability,
        0,
        np.minimum(review_cost, no_review_cost),
        color="#94A3B8",
        alpha=0.18,
        hatch="//",
        label="Lower-cost action envelope",
    )
    axis.annotate(
        "Below p*: do not review",
        xy=(0.025, 2.5),
        xytext=(0.12, 18),
        arrowprops={"arrowstyle": "->", "color": "#111827"},
    )
    axis.annotate(
        "Above p*: review",
        xy=(0.30, review_cost),
        xytext=(0.42, 25),
        arrowprops={"arrowstyle": "->", "color": "#111827"},
    )
    axis.set(
        title="Expected-cost threshold: p* = Cᵣ / Cₘ",
        xlabel="Posterior probability of fraud p",
        ylabel="Expected cost",
        xlim=(0.0, 1.0),
        ylim=(0.0, 105.0),
    )
    axis.grid(alpha=0.20)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(figure)


def generate_all_assets(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Generate every required asset and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generators: tuple[tuple[str, Callable[[Path], None]], ...] = (
        ("event_operations.png", generate_event_operations),
        ("conditional_probability.gif", generate_conditional_probability_animation),
        ("independence_comparison.png", generate_independence_comparison),
        ("bayes_base_rate.gif", generate_bayes_base_rate_animation),
        ("bayes_surface.png", generate_bayes_surface),
        ("expected_value_balance.png", generate_expected_value_balance),
        ("variance_spread.gif", generate_variance_spread_animation),
        ("bernoulli_variance.png", generate_bernoulli_variance),
        ("monte_carlo_convergence.gif", generate_monte_carlo_convergence_animation),
        ("expected_cost.png", generate_expected_cost),
    )
    generated_paths: list[Path] = []
    for filename, generator in generators:
        path = output_dir / filename
        generator(path)
        generated_paths.append(path)
        print(f"Generated: {path}")
    return generated_paths


if __name__ == "__main__":
    generate_all_assets()
