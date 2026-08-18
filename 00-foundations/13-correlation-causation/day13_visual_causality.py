"""Generate the Day 13 visual learning lab from deterministic synthetic data."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import plotly.graph_objects as go
from scipy import stats

from example import generate_confounding_data
from from_scratch import ols_coefficients, pearson_correlation


TOPIC_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIRECTORY = TOPIC_DIRECTORY / "visuals"

BLUE = "#2563EB"
CYAN = "#0891B2"
GREEN = "#16A34A"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"
RED = "#DC2626"
SLATE = "#334155"
LIGHT = "#F8FAFC"
GRID = "#CBD5E1"


@dataclass(frozen=True)
class VisualResult:
    """A generated artifact and diagnostics observed during generation."""

    path: Path
    label: str
    metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfoundingExperiment:
    """Synthetic confounding arrays and their fitted exposure coefficients."""

    intent: np.ndarray
    exposure: np.ndarray
    purchase: np.ndarray
    true_effect: float
    correlation: float
    naive_coefficient: float
    adjusted_coefficient: float
    adjusted_coefficients: np.ndarray


@dataclass(frozen=True)
class SimpsonExperiment:
    """Synthetic grouped data and aggregate/subgroup slopes."""

    treatment: np.ndarray
    outcome: np.ndarray
    group: np.ndarray
    aggregate_slope: float
    low_intent_slope: float
    high_intent_slope: float


@dataclass(frozen=True)
class ColliderExperiment:
    """Synthetic collider data before and after selection."""

    technical_skill: np.ndarray
    communication_skill: np.ndarray
    selection_score: np.ndarray
    selected: np.ndarray
    population_correlation: float
    selected_correlation: float


@dataclass(frozen=True)
class RagExperiment:
    """Observational and randomized synthetic RAG telemetry."""

    observational_top_k: np.ndarray
    observational_quality: np.ndarray
    randomized_top_k: np.ndarray
    randomized_quality: np.ndarray
    query_complexity: np.ndarray
    observational_slope: float
    randomized_slope: float
    randomized_assignment_correlation: float
    true_top_k_effect: float


def configure_plot_style() -> None:
    """Apply one restrained visual style across all Matplotlib assets."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": LIGHT,
            "axes.edgecolor": GRID,
            "axes.labelcolor": SLATE,
            "axes.titlecolor": "#0F172A",
            "axes.titleweight": "bold",
            "grid.color": GRID,
            "grid.alpha": 0.35,
            "font.size": 10,
            "legend.frameon": False,
            "text.color": SLATE,
        }
    )


def _warn_unless(condition: bool, message: str) -> None:
    """Emit a clear warning if a designed pedagogical property is absent."""
    if not condition:
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def _save_static(
    figure: plt.Figure,
    path: Path,
    *,
    show: bool,
    dpi: int = 150,
    tight_layout: bool = True,
) -> None:
    """Save a figure and optionally retain it for interactive display."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if show:
        figure.show()
    else:
        plt.close(figure)


def _save_animation(
    animation: FuncAnimation,
    figure: plt.Figure,
    path: Path,
    *,
    fps: int,
) -> None:
    """Save a bounded GIF with Pillow and close its source figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Animation was not created correctly: {path}")


def _line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    coefficients = ols_coefficients(x, y)
    return float(coefficients[0]), float(coefficients[1])


def _plot_regression_line(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    label: str,
    linewidth: float = 2.5,
) -> float:
    intercept, slope = _line_fit(x, y)
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 150)
    axis.plot(
        x_line,
        intercept + slope * x_line,
        color=color,
        linewidth=linewidth,
        label=label,
    )
    return slope


def correlation_relationships(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Construct six relationships that distinguish correlation measures."""
    rng = np.random.default_rng(seed)
    n = 180
    x_positive = rng.normal(0.0, 1.0, n)
    x_negative = rng.normal(0.0, 1.0, n)
    x_monotonic = rng.uniform(-2.2, 2.2, n)
    x_nonmonotonic = rng.normal(0.0, 1.0, n)
    x_independent = rng.normal(0.0, 1.0, n)
    x_outlier = np.linspace(0.0, 10.0, n)
    outlier_relation = x_outlier + rng.normal(0.0, 0.7, n)
    return {
        "A. Positive linear": (
            x_positive,
            1.8 * x_positive + rng.normal(0.0, 0.45, n),
        ),
        "B. Negative linear": (
            x_negative,
            -1.7 * x_negative + rng.normal(0.0, 0.5, n),
        ),
        "C. Nonlinear monotonic": (
            x_monotonic,
            np.exp(0.9 * x_monotonic) + rng.normal(0.0, 0.25, n),
        ),
        "D. Non-monotonic": (
            x_nonmonotonic,
            x_nonmonotonic**2 + rng.normal(0.0, 0.15, n),
        ),
        "E. Independent noise": (x_independent, rng.normal(0.0, 1.0, n)),
        "F. Influential outlier": (
            np.append(x_outlier, 12.0),
            np.append(outlier_relation, -28.0),
        ),
    }


def generate_correlation_types(
    output_dir: Path, *, seed: int, show: bool
) -> VisualResult:
    """Compare Pearson and Spearman across six relationship shapes."""
    relationships = correlation_relationships(seed)
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    metric_lines: list[str] = []
    for axis, (title, (x, y)) in zip(axes.flat, relationships.items(), strict=True):
        pearson = pearson_correlation(x, y)
        spearman = float(stats.spearmanr(x, y).statistic)
        metric_lines.append(f"{title}: Pearson={pearson:.3f}, Spearman={spearman:.3f}")
        axis.scatter(x, y, s=22, alpha=0.62, color=BLUE, edgecolors="none")
        axis.set(title=title, xlabel="X", ylabel="Y")
        axis.grid(True)
        axis.text(
            0.04,
            0.94,
            f"Pearson = {pearson:.2f}\nSpearman = {spearman:.2f}",
            transform=axis.transAxes,
            va="top",
            bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.9),
        )
    figure.suptitle(
        "Correlation measures association shape, not causal direction",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Pearson: linear association · Spearman: monotonic rank association",
        ha="center",
    )
    path = output_dir / "01_correlation_types.png"
    _save_static(figure, path, show=show)
    return VisualResult(path, "Correlation types", tuple(metric_lines))


def generate_nonlinear_animation(
    output_dir: Path, *, seed: int, quick: bool
) -> VisualResult:
    """Animate a transition from a linear relation to a quadratic relation."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, 800)
    noise = rng.normal(0.0, 0.12, x.size)
    blends = np.linspace(0.0, 1.0, 8 if quick else 22)
    correlations = np.array(
        [
            pearson_correlation(
                x,
                (1.0 - blend) * x + blend * (x**2 - 1.0) + noise,
            )
            for blend in blends
        ]
    )
    _warn_unless(
        abs(correlations[-1]) < 0.12,
        "The nonlinear frame did not produce a small Pearson correlation.",
    )
    figure, axis = plt.subplots(figsize=(8.2, 5.4), dpi=90 if quick else 110)

    def update(frame: int) -> None:
        axis.clear()
        blend = float(blends[frame])
        y = (1.0 - blend) * x + blend * (x**2 - 1.0) + noise
        axis.scatter(x, y, s=18, alpha=0.42, color=PURPLE, edgecolors="none")
        axis.set(
            xlim=(-3.4, 3.4),
            ylim=(-3.0, 8.0),
            xlabel="X",
            ylabel="Y",
            title="From linear association to nonlinear dependence",
        )
        axis.grid(True)
        axis.text(
            0.04,
            0.94,
            f"quadratic blend = {blend:.2f}\nPearson = {correlations[frame]:.3f}",
            transform=axis.transAxes,
            va="top",
            bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.92),
        )
        if frame == len(blends) - 1:
            axis.text(
                0.5,
                0.08,
                "Near-zero linear correlation ≠ independence",
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                color=RED,
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=RED, alpha=0.9),
            )

    animation = FuncAnimation(
        figure, update, frames=len(blends), interval=500, repeat=True
    )
    path = output_dir / "02_zero_correlation_nonlinear.gif"
    _save_animation(animation, figure, path, fps=2 if quick else 4)
    return VisualResult(
        path,
        "Nonlinear dependence",
        (
            f"Initial Pearson correlation: {correlations[0]:.3f}",
            f"Final Pearson correlation: {correlations[-1]:.3f}",
        ),
    )


def build_confounding_experiment(
    *, seed: int, sample_size: int = 2_000
) -> ConfoundingExperiment:
    """Generate and fit the known synthetic confounding process."""
    true_effect = 2.0
    intent, exposure, purchase = generate_confounding_data(
        sample_size=sample_size,
        true_effect=true_effect,
        assignment_strength=1.5,
        intent_effect=5.0,
        randomized_exposure=False,
        seed=seed,
    )
    naive = ols_coefficients(exposure, purchase)
    adjusted = ols_coefficients(np.column_stack((exposure, intent)), purchase)
    experiment = ConfoundingExperiment(
        intent=intent,
        exposure=exposure,
        purchase=purchase,
        true_effect=true_effect,
        correlation=pearson_correlation(exposure, purchase),
        naive_coefficient=float(naive[1]),
        adjusted_coefficient=float(adjusted[1]),
        adjusted_coefficients=adjusted,
    )
    _warn_unless(
        abs(experiment.naive_coefficient - true_effect) > 1.0,
        "The naive coefficient is not materially biased in the confounding example.",
    )
    _warn_unless(
        abs(experiment.adjusted_coefficient - true_effect)
        < abs(experiment.naive_coefficient - true_effect),
        "Adjustment is not closer to the known effect in this synthetic run.",
    )
    return experiment


def generate_confounded_relationship(
    experiment: ConfoundingExperiment, output_dir: Path, *, show: bool
) -> VisualResult:
    """Show the 2D exposure-outcome association colored by the common cause."""
    figure, axis = plt.subplots(figsize=(9.2, 6.0))
    scatter = axis.scatter(
        experiment.exposure,
        experiment.purchase,
        c=experiment.intent,
        cmap="coolwarm",
        s=22,
        alpha=0.55,
        edgecolors="none",
    )
    _plot_regression_line(
        axis,
        experiment.exposure,
        experiment.purchase,
        color=SLATE,
        label=f"Naive fit: slope={experiment.naive_coefficient:.2f}",
    )
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Customer intent (confounder)")
    axis.set(
        xlabel="Ad exposure",
        ylabel="Purchase value",
        title="A hidden common cause steepens the observed relationship",
    )
    axis.grid(True)
    axis.legend(loc="upper left")
    axis.text(
        0.98,
        0.05,
        (
            f"raw correlation = {experiment.correlation:.2f}\n"
            f"naive coefficient = {experiment.naive_coefficient:.2f}\n"
            f"known simulated effect = {experiment.true_effect:.2f}"
        ),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.94),
    )
    axis.annotate(
        "Color reveals that customer intent\nshifts both exposure and outcome",
        xy=(
            np.quantile(experiment.exposure, 0.8),
            np.quantile(experiment.purchase, 0.75),
        ),
        xytext=(
            np.quantile(experiment.exposure, 0.05),
            np.quantile(experiment.purchase, 0.92),
        ),
        arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.8),
        color=RED,
        fontweight="bold",
    )
    path = output_dir / "03_confounded_relationship.png"
    _save_static(figure, path, show=show)
    return VisualResult(
        path,
        "Confounded relationship",
        (
            f"Raw correlation: {experiment.correlation:.3f}",
            f"Naive coefficient: {experiment.naive_coefficient:.3f}",
        ),
    )


def generate_naive_vs_adjusted(
    experiment: ConfoundingExperiment, output_dir: Path, *, show: bool
) -> VisualResult:
    """Compare the known, naive, and adjusted exposure coefficients."""
    values = np.array(
        [
            experiment.true_effect,
            experiment.naive_coefficient,
            experiment.adjusted_coefficient,
        ]
    )
    labels = ("Known simulated\neffect", "Naive\nY ~ X", "Adjusted\nY ~ X + Z")
    figure, axis = plt.subplots(figsize=(8.6, 5.6))
    bars = axis.bar(labels, values, color=(GREEN, RED, BLUE), width=0.62)
    axis.axhline(
        experiment.true_effect,
        color=GREEN,
        linestyle="--",
        linewidth=1.8,
        alpha=0.75,
    )
    axis.bar_label(bars, fmt="%.2f", padding=4, fontweight="bold")
    axis.set(
        ylabel="Exposure coefficient",
        title="Adjustment succeeds only under the constructed assumptions",
        ylim=(0.0, max(values) * 1.23),
    )
    axis.grid(axis="y")
    axis.text(
        0.5,
        0.96,
        (
            "Known DAG · observed confounder · appropriate linear form\n"
            "adequate overlap · no hidden confounding by construction"
        ),
        transform=axis.transAxes,
        ha="center",
        va="top",
        bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.94),
    )
    path = output_dir / "05_naive_vs_adjusted.png"
    _save_static(figure, path, show=show)
    return VisualResult(
        path,
        "Naive versus adjusted regression",
        (
            f"Known effect: {experiment.true_effect:.3f}",
            f"Naive coefficient: {experiment.naive_coefficient:.3f}",
            f"Adjusted coefficient: {experiment.adjusted_coefficient:.3f}",
        ),
    )


def generate_confounder_3d(
    experiment: ConfoundingExperiment,
    output_dir: Path,
    *,
    show: bool,
) -> VisualResult:
    """Create an interactive 3D confounder view and adjusted regression plane."""
    stride = max(1, experiment.intent.size // 850)
    exposure = experiment.exposure[::stride]
    intent = experiment.intent[::stride]
    purchase = experiment.purchase[::stride]
    exposure_grid, intent_grid = np.meshgrid(
        np.linspace(np.quantile(exposure, 0.03), np.quantile(exposure, 0.97), 22),
        np.linspace(np.quantile(intent, 0.03), np.quantile(intent, 0.97), 22),
    )
    coefficients = experiment.adjusted_coefficients
    purchase_grid = (
        coefficients[0]
        + coefficients[1] * exposure_grid
        + coefficients[2] * intent_grid
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=exposure,
            y=intent,
            z=purchase,
            mode="markers",
            name="Synthetic customers",
            marker=dict(
                size=3.5,
                color=intent,
                colorscale="RdBu",
                opacity=0.62,
                colorbar=dict(title="Intent"),
            ),
            hovertemplate=(
                "Exposure=%{x:.2f}<br>Intent=%{y:.2f}<br>"
                "Purchase=%{z:.2f}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Surface(
            x=exposure_grid,
            y=intent_grid,
            z=purchase_grid,
            name="Adjusted plane",
            colorscale=[[0.0, "#BFDBFE"], [1.0, BLUE]],
            opacity=0.45,
            showscale=False,
            hovertemplate=(
                "Exposure=%{x:.2f}<br>Intent=%{y:.2f}<br>"
                "Fitted purchase=%{z:.2f}<extra>Adjusted plane</extra>"
            ),
        )
    )
    figure.update_layout(
        title="Reveal the common cause: exposure, intent, and purchase",
        template="plotly_white",
        scene=dict(
            xaxis_title="Ad exposure",
            yaxis_title="Customer intent",
            zaxis_title="Purchase value",
            camera_eye=dict(x=1.55, y=1.45, z=0.9),
        ),
        margin=dict(l=10, r=10, t=95, b=20),
        annotations=[
            dict(
                text=(
                    "Looking only at exposure and purchase hides the structure "
                    "created by intent.<br>The plane is an adjusted association; "
                    "causal interpretation relies on the known synthetic DAG."
                ),
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="center",
            )
        ],
    )
    path = output_dir / "04_confounder_3d.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True, full_html=True)
    if show:
        figure.show()
    return VisualResult(
        path,
        "Interactive 3D confounder view",
        (f"Adjusted exposure coefficient: {experiment.adjusted_coefficient:.3f}",),
    )


def omitted_variable_bias_path(
    *, seed: int, quick: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate coefficient paths as treatment-confounder correlation grows."""
    rng = np.random.default_rng(seed)
    n = 1_000
    intent = rng.normal(0.0, 1.0, n)
    exposure_noise = rng.normal(0.0, 1.0, n)
    outcome_noise = rng.normal(0.0, 1.6, n)
    strengths = np.linspace(0.0, 0.9, 7 if quick else 20)
    naive_values: list[float] = []
    adjusted_values: list[float] = []
    exposures: list[np.ndarray] = []
    purchases: list[np.ndarray] = []
    for strength in strengths:
        exposure = (
            strength * intent + np.sqrt(1.0 - strength**2) * exposure_noise
        )
        purchase = 2.0 * exposure + 5.0 * intent + outcome_noise
        exposures.append(exposure)
        purchases.append(purchase)
        naive_values.append(float(ols_coefficients(exposure, purchase)[1]))
        adjusted_values.append(
            float(
                ols_coefficients(
                    np.column_stack((exposure, intent)), purchase
                )[1]
            )
        )
    return (
        strengths,
        np.asarray(naive_values),
        np.asarray(adjusted_values),
        np.asarray(exposures),
        np.asarray(purchases),
        intent,
    )


def generate_omitted_variable_bias_animation(
    output_dir: Path, *, seed: int, quick: bool
) -> VisualResult:
    """Animate omitted-variable bias as treatment-confounder association grows."""
    (
        strengths,
        naive_values,
        adjusted_values,
        exposures,
        purchases,
        intent,
    ) = omitted_variable_bias_path(seed=seed, quick=quick)
    _warn_unless(
        abs(naive_values[-1] - 2.0) > abs(naive_values[0] - 2.0) + 2.5,
        "The naive coefficient did not drift materially as confounding grew.",
    )
    _warn_unless(
        np.max(np.abs(adjusted_values - 2.0)) < 0.15,
        "The adjusted coefficient did not remain close to the known effect.",
    )
    figure, (scatter_axis, coefficient_axis) = plt.subplots(
        1, 2, figsize=(11.2, 5.2), dpi=85 if quick else 105
    )
    point_indices = np.arange(0, intent.size, 2)

    def update(frame: int) -> None:
        scatter_axis.clear()
        coefficient_axis.clear()
        exposure = exposures[frame]
        purchase = purchases[frame]
        scatter_axis.scatter(
            exposure[point_indices],
            purchase[point_indices],
            c=intent[point_indices],
            cmap="coolwarm",
            s=18,
            alpha=0.5,
            edgecolors="none",
        )
        _plot_regression_line(
            scatter_axis,
            exposure,
            purchase,
            color=SLATE,
            label=f"naive slope = {naive_values[frame]:.2f}",
        )
        scatter_axis.set(
            xlim=(-3.5, 3.5),
            ylim=(-17.0, 18.0),
            xlabel="Ad exposure X",
            ylabel="Purchase value Y",
            title=f"Corr(X, intent) target = {strengths[frame]:.2f}",
        )
        scatter_axis.grid(True)
        scatter_axis.legend(loc="upper left")

        visible = slice(0, frame + 1)
        coefficient_axis.plot(
            strengths[visible],
            naive_values[visible],
            color=RED,
            marker="o",
            linewidth=2.5,
            label="Naive estimate",
        )
        coefficient_axis.plot(
            strengths[visible],
            adjusted_values[visible],
            color=BLUE,
            marker="o",
            linewidth=2.5,
            label="Adjusted estimate",
        )
        coefficient_axis.axhline(
            2.0, color=GREEN, linestyle="--", linewidth=2.2, label="Known effect"
        )
        coefficient_axis.set(
            xlim=(-0.03, 0.93),
            ylim=(1.4, 7.2),
            xlabel="Treatment-confounder correlation",
            ylabel="Exposure coefficient",
            title="Omitting Z moves the coefficient",
        )
        coefficient_axis.grid(True)
        coefficient_axis.legend(loc="upper left")
        coefficient_axis.text(
            0.98,
            0.06,
            "β_naive = β_true + γ Cov(X,Z) / Var(X)\n"
            "Adjustment works here under synthetic assumptions.",
            transform=coefficient_axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.92),
        )
        figure.tight_layout()

    animation = FuncAnimation(
        figure, update, frames=len(strengths), interval=550, repeat=True
    )
    path = output_dir / "06_omitted_variable_bias.gif"
    _save_animation(animation, figure, path, fps=2 if quick else 4)
    return VisualResult(
        path,
        "Omitted-variable bias",
        (
            f"Naive coefficient: {naive_values[0]:.3f} -> {naive_values[-1]:.3f}",
            f"Adjusted coefficient: {adjusted_values[0]:.3f} -> {adjusted_values[-1]:.3f}",
        ),
    )


def build_simpson_experiment(*, seed: int, sample_size: int = 900) -> SimpsonExperiment:
    """Construct a stable aggregate/subgroup slope reversal."""
    if sample_size < 200:
        raise ValueError("sample_size must be at least 200.")
    rng = np.random.default_rng(seed)
    low_size = sample_size // 2
    high_size = sample_size - low_size
    low_treatment = rng.normal(2.0, 0.75, low_size)
    high_treatment = rng.normal(7.0, 0.75, high_size)
    low_outcome = 5.0 - 0.65 * low_treatment + rng.normal(0.0, 0.65, low_size)
    high_outcome = 13.0 - 0.65 * high_treatment + rng.normal(0.0, 0.65, high_size)
    treatment = np.concatenate((low_treatment, high_treatment))
    outcome = np.concatenate((low_outcome, high_outcome))
    group = np.concatenate(
        (
            np.repeat("Low intent", low_size),
            np.repeat("High intent", high_size),
        )
    )
    aggregate_slope = _line_fit(treatment, outcome)[1]
    low_slope = _line_fit(low_treatment, low_outcome)[1]
    high_slope = _line_fit(high_treatment, high_outcome)[1]
    experiment = SimpsonExperiment(
        treatment=treatment,
        outcome=outcome,
        group=group,
        aggregate_slope=aggregate_slope,
        low_intent_slope=low_slope,
        high_intent_slope=high_slope,
    )
    _warn_unless(
        aggregate_slope > 0.0 and low_slope < 0.0 and high_slope < 0.0,
        "The configured data did not produce the intended Simpson reversal.",
    )
    return experiment


def generate_simpsons_paradox(
    experiment: SimpsonExperiment, output_dir: Path, *, show: bool
) -> VisualResult:
    """Contrast the aggregate trend with group-conditioned trends."""
    figure, (aggregate_axis, grouped_axis) = plt.subplots(
        1, 2, figsize=(12.2, 5.5), sharex=True, sharey=True
    )
    aggregate_axis.scatter(
        experiment.treatment,
        experiment.outcome,
        color=SLATE,
        s=20,
        alpha=0.38,
        edgecolors="none",
    )
    _plot_regression_line(
        aggregate_axis,
        experiment.treatment,
        experiment.outcome,
        color=RED,
        label=f"Aggregate slope = {experiment.aggregate_slope:.2f}",
    )
    aggregate_axis.set_title("A. Aggregated view")
    aggregate_axis.legend(loc="upper left")

    masks = {
        "Low intent": (experiment.group == "Low intent", BLUE),
        "High intent": (experiment.group == "High intent", ORANGE),
    }
    for label, (mask, color) in masks.items():
        grouped_axis.scatter(
            experiment.treatment[mask],
            experiment.outcome[mask],
            color=color,
            s=20,
            alpha=0.42,
            edgecolors="none",
            label=label,
        )
        slope = _plot_regression_line(
            grouped_axis,
            experiment.treatment[mask],
            experiment.outcome[mask],
            color=color,
            label=f"{label} slope = {_line_fit(experiment.treatment[mask], experiment.outcome[mask])[1]:.2f}",
        )
        if slope >= 0.0:
            warnings.warn(f"{label} did not retain a negative slope.", RuntimeWarning)
    grouped_axis.set_title("B. Conditioned on intent group")
    grouped_axis.legend(loc="upper left")

    for axis in (aggregate_axis, grouped_axis):
        axis.set_xlabel("Treatment intensity")
        axis.set_ylabel("Outcome")
        axis.grid(True)
    figure.suptitle(
        "Simpson's paradox: aggregation can reverse the apparent relationship",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "The causal graph and estimand—not the paradox alone—determine which comparison is relevant.",
        ha="center",
    )
    path = output_dir / "07_simpsons_paradox.png"
    _save_static(figure, path, show=show)
    return VisualResult(
        path,
        "Simpson's paradox",
        (
            f"Aggregate slope: {experiment.aggregate_slope:.3f}",
            f"Low-intent slope: {experiment.low_intent_slope:.3f}",
            f"High-intent slope: {experiment.high_intent_slope:.3f}",
        ),
    )


def build_collider_experiment(
    *, seed: int, sample_size: int = 6_000, selection_quantile: float = 0.75
) -> ColliderExperiment:
    """Generate independent skills and select on their noisy common effect."""
    if sample_size < 500:
        raise ValueError("sample_size must be at least 500.")
    if not 0.0 < selection_quantile < 1.0:
        raise ValueError("selection_quantile must be between zero and one.")
    rng = np.random.default_rng(seed)
    technical = rng.normal(0.0, 1.0, sample_size)
    communication = rng.normal(0.0, 1.0, sample_size)
    selection_score = technical + communication + rng.normal(0.0, 0.5, sample_size)
    selected = selection_score >= np.quantile(selection_score, selection_quantile)
    experiment = ColliderExperiment(
        technical_skill=technical,
        communication_skill=communication,
        selection_score=selection_score,
        selected=selected,
        population_correlation=pearson_correlation(technical, communication),
        selected_correlation=pearson_correlation(
            technical[selected], communication[selected]
        ),
    )
    _warn_unless(
        abs(experiment.selected_correlation)
        > abs(experiment.population_correlation) + 0.25,
        "Conditioning on the collider did not materially increase association.",
    )
    return experiment


def generate_collider_bias(
    experiment: ColliderExperiment, output_dir: Path, *, show: bool
) -> VisualResult:
    """Show how selecting on hiring creates an association between skills."""
    figure, (population_axis, selected_axis) = plt.subplots(
        1, 2, figsize=(12.0, 5.4), sharex=True, sharey=True
    )
    population_axis.scatter(
        experiment.technical_skill,
        experiment.communication_skill,
        s=15,
        alpha=0.22,
        color=SLATE,
        edgecolors="none",
    )
    population_axis.set_title(
        f"A. Entire population · r={experiment.population_correlation:.2f}"
    )
    selected_axis.scatter(
        experiment.technical_skill[experiment.selected],
        experiment.communication_skill[experiment.selected],
        s=18,
        alpha=0.38,
        color=PURPLE,
        edgecolors="none",
    )
    selected_axis.set_title(
        f"B. Selected candidates · r={experiment.selected_correlation:.2f}"
    )
    for axis in (population_axis, selected_axis):
        axis.set_xlabel("Technical skill")
        axis.set_ylabel("Communication skill")
        axis.grid(True)
    figure.suptitle(
        "Conditioning on a collider can create an association",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Technical skill → Hired ← Communication skill",
        ha="center",
        color=RED,
        fontweight="bold",
    )
    path = output_dir / "08_collider_bias.png"
    _save_static(figure, path, show=show)
    return VisualResult(
        path,
        "Collider bias",
        (
            f"Population correlation: {experiment.population_correlation:.3f}",
            f"Selected correlation: {experiment.selected_correlation:.3f}",
            f"Selected observations: {int(np.sum(experiment.selected))}",
        ),
    )


def generate_collider_animation(
    experiment: ColliderExperiment, output_dir: Path, *, quick: bool
) -> VisualResult:
    """Animate the induced association as the selection threshold rises."""
    quantiles = np.linspace(0.35, 0.9, 7 if quick else 18)
    thresholds = np.quantile(experiment.selection_score, quantiles)
    correlations = np.array(
        [
            pearson_correlation(
                experiment.technical_skill[experiment.selection_score >= threshold],
                experiment.communication_skill[
                    experiment.selection_score >= threshold
                ],
            )
            for threshold in thresholds
        ]
    )
    _warn_unless(
        abs(correlations[-1]) > abs(correlations[0]) + 0.15,
        "A stricter collider threshold did not strengthen the selected association.",
    )
    figure, axis = plt.subplots(figsize=(8.2, 5.4), dpi=85 if quick else 105)

    def update(frame: int) -> None:
        axis.clear()
        selected = experiment.selection_score >= thresholds[frame]
        axis.scatter(
            experiment.technical_skill[~selected],
            experiment.communication_skill[~selected],
            s=10,
            alpha=0.07,
            color=SLATE,
            edgecolors="none",
            label="Not selected",
        )
        axis.scatter(
            experiment.technical_skill[selected],
            experiment.communication_skill[selected],
            s=18,
            alpha=0.42,
            color=PURPLE,
            edgecolors="none",
            label="Selected",
        )
        axis.set(
            xlim=(-3.7, 3.7),
            ylim=(-3.7, 3.7),
            xlabel="Technical skill",
            ylabel="Communication skill",
            title="A stricter hiring threshold strengthens collider bias",
        )
        axis.grid(True)
        axis.legend(loc="lower left")
        axis.text(
            0.98,
            0.95,
            (
                f"selected top {1.0 - quantiles[frame]:.0%}\n"
                f"selected n = {int(np.sum(selected))}\n"
                f"selected correlation = {correlations[frame]:.3f}"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.92),
        )

    animation = FuncAnimation(
        figure, update, frames=len(quantiles), interval=550, repeat=True
    )
    path = output_dir / "08_collider_bias.gif"
    _save_animation(animation, figure, path, fps=2 if quick else 4)
    return VisualResult(
        path,
        "Collider-threshold animation",
        (
            f"Selected correlation: {correlations[0]:.3f} -> {correlations[-1]:.3f}",
        ),
    )


def _draw_node(axis: plt.Axes, location: tuple[float, float], label: str, color: str) -> None:
    axis.text(
        *location,
        label,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.55", facecolor=color, edgecolor="white"),
        color="white",
        zorder=3,
    )


def _draw_edge(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    alpha: float = 1.0,
    color: str = SLATE,
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            linewidth=2.5,
            alpha=alpha,
            shrinkA=20,
            shrinkB=20,
        ),
    )


def generate_observation_vs_intervention(
    output_dir: Path, *, quick: bool
) -> VisualResult:
    """Animate the graph surgery represented by an intervention."""
    transition = np.linspace(0.0, 1.0, 6 if quick else 14)
    figure, axis = plt.subplots(figsize=(8.5, 5.2), dpi=90 if quick else 110)
    z_location = (0.22, 0.68)
    x_location = (0.5, 0.35)
    y_location = (0.78, 0.68)

    def update(frame: int) -> None:
        axis.clear()
        progress = float(transition[frame])
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.axis("off")
        _draw_node(axis, z_location, "Z\nIntent", PURPLE)
        _draw_node(axis, x_location, "X\nExposure" if progress < 0.55 else "do(X=x)", BLUE)
        _draw_node(axis, y_location, "Y\nPurchase", GREEN)
        _draw_edge(axis, x_location, y_location, color=BLUE)
        _draw_edge(axis, z_location, y_location, color=PURPLE)
        _draw_edge(
            axis,
            z_location,
            x_location,
            alpha=max(0.0, 1.0 - progress * 1.5),
            color=RED,
        )
        if progress < 0.55:
            heading = "Observational world: P(Y | X=x)"
            explanation = "X retains its natural causes."
        else:
            heading = "Interventional world: P(Y | do(X=x))"
            explanation = "The intervention removes the incoming edge Z → X."
        axis.set_title(heading, fontsize=15, pad=20)
        axis.text(
            0.5,
            0.1,
            explanation,
            ha="center",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.94),
        )
        axis.text(
            0.5,
            0.94,
            "Observing X=x is not the same as forcing X=x.",
            ha="center",
            fontweight="bold",
            color=RED,
        )

    animation = FuncAnimation(
        figure, update, frames=len(transition), interval=650, repeat=True
    )
    path = output_dir / "09_observation_vs_intervention.gif"
    _save_animation(animation, figure, path, fps=2 if quick else 3)
    return VisualResult(
        path,
        "Observation versus intervention",
        ("Incoming Z -> X edge removed under do(X=x)",),
    )


def build_rag_experiment(*, seed: int, sample_size: int = 2_500) -> RagExperiment:
    """Simulate confounded telemetry and randomized retrieval-depth assignment."""
    if sample_size < 500:
        raise ValueError("sample_size must be at least 500.")
    rng = np.random.default_rng(seed)
    complexity = rng.normal(0.0, 1.0, sample_size)
    observational_top_k = 5.0 + 1.5 * complexity + rng.normal(0.0, 0.7, sample_size)
    randomized_top_k = rng.uniform(2.0, 8.0, sample_size)
    true_top_k_effect = 1.2
    observational_quality = (
        70.0
        + true_top_k_effect * observational_top_k
        - 5.0 * complexity
        + rng.normal(0.0, 2.5, sample_size)
    )
    randomized_quality = (
        70.0
        + true_top_k_effect * randomized_top_k
        - 5.0 * complexity
        + rng.normal(0.0, 2.5, sample_size)
    )
    experiment = RagExperiment(
        observational_top_k=observational_top_k,
        observational_quality=observational_quality,
        randomized_top_k=randomized_top_k,
        randomized_quality=randomized_quality,
        query_complexity=complexity,
        observational_slope=_line_fit(
            observational_top_k, observational_quality
        )[1],
        randomized_slope=_line_fit(randomized_top_k, randomized_quality)[1],
        randomized_assignment_correlation=pearson_correlation(
            randomized_top_k, complexity
        ),
        true_top_k_effect=true_top_k_effect,
    )
    _warn_unless(
        abs(experiment.randomized_assignment_correlation) < 0.08,
        "Randomized top_k remained too associated with query complexity.",
    )
    _warn_unless(
        abs(experiment.randomized_slope - true_top_k_effect)
        < abs(experiment.observational_slope - true_top_k_effect),
        "Randomization was not closer to the known top_k effect in this run.",
    )
    return experiment


def generate_rag_observation_vs_intervention(
    experiment: RagExperiment, output_dir: Path, *, show: bool
) -> VisualResult:
    """Contrast confounded RAG telemetry with randomized synthetic assignment."""
    figure, (observed_axis, randomized_axis) = plt.subplots(
        1, 2, figsize=(12.4, 5.6), sharey=True
    )
    observed_scatter = observed_axis.scatter(
        experiment.observational_top_k,
        experiment.observational_quality,
        c=experiment.query_complexity,
        cmap="coolwarm",
        s=18,
        alpha=0.45,
        edgecolors="none",
    )
    _plot_regression_line(
        observed_axis,
        experiment.observational_top_k,
        experiment.observational_quality,
        color=RED,
        label=f"Observed slope = {experiment.observational_slope:.2f}",
    )
    randomized_axis.scatter(
        experiment.randomized_top_k,
        experiment.randomized_quality,
        c=experiment.query_complexity,
        cmap="coolwarm",
        s=18,
        alpha=0.45,
        edgecolors="none",
    )
    _plot_regression_line(
        randomized_axis,
        experiment.randomized_top_k,
        experiment.randomized_quality,
        color=GREEN,
        label=f"Randomized slope = {experiment.randomized_slope:.2f}",
    )
    observed_axis.set_title("A. Production-like observational telemetry")
    randomized_axis.set_title("B. Randomized top_k assignment")
    for axis in (observed_axis, randomized_axis):
        axis.set_xlabel("Retrieval top_k (synthetic)")
        axis.set_ylabel("Answer quality score (synthetic)")
        axis.grid(True)
        axis.legend(loc="upper right")
    colorbar = figure.colorbar(
        observed_scatter, ax=(observed_axis, randomized_axis), shrink=0.86
    )
    colorbar.set_label("Query complexity")
    figure.suptitle(
        "Prediction from telemetry is not the effect of changing top_k",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        (
            "Synthetic DAG: Complexity → top_k; Complexity → Quality; "
            "top_k → Quality · P(Y|X) ≠ P(Y|do(X)) in general"
        ),
        ha="center",
    )
    figure.subplots_adjust(left=0.07, right=0.88, top=0.86, bottom=0.14, wspace=0.2)
    path = output_dir / "10_rag_observation_vs_intervention.png"
    _save_static(figure, path, show=show, tight_layout=False)
    return VisualResult(
        path,
        "Applied AI observation versus intervention",
        (
            f"Known top_k effect: {experiment.true_top_k_effect:.3f}",
            f"Observational slope: {experiment.observational_slope:.3f}",
            f"Randomized slope: {experiment.randomized_slope:.3f}",
            (
                "Corr(randomized top_k, complexity): "
                f"{experiment.randomized_assignment_correlation:.3f}"
            ),
        ),
    )


def generate_visual_summary(output_dir: Path, *, show: bool) -> VisualResult:
    """Create a compact causal-reasoning map suitable for the topic README."""
    figure, axis = plt.subplots(figsize=(12.0, 7.0))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")
    _draw_node(axis, (0.5, 0.88), "Observed X ↔ Y", SLATE)
    axis.text(
        0.5,
        0.76,
        "What data-generating process could create this association?",
        ha="center",
        fontweight="bold",
        fontsize=12,
    )
    _draw_edge(axis, (0.5, 0.84), (0.5, 0.79), color=SLATE)

    structures = (
        ((0.12, 0.55), "X → Y\ncausal path", GREEN),
        ((0.37, 0.55), "X ← Z → Y\nconfounding", PURPLE),
        ((0.63, 0.55), "Y → X\nreverse causality", ORANGE),
        ((0.88, 0.55), "X → C ← Y\ncollider", RED),
    )
    for location, label, color in structures:
        _draw_node(axis, location, label, color)
        _draw_edge(axis, (0.5, 0.73), (location[0], 0.62), color=color)

    axis.text(
        0.5,
        0.35,
        "Define the intervention, estimand, assignment process, and valid adjustment set",
        ha="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#E2E8F0", edgecolor=GRID),
    )
    for location, _, color in structures:
        _draw_edge(axis, (location[0], 0.47), (0.5, 0.4), color=color, alpha=0.7)
    axis.text(
        0.28,
        0.17,
        "Prediction\nP(Y | X)",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=BLUE,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#DBEAFE", edgecolor=BLUE),
    )
    axis.text(
        0.72,
        0.17,
        "Causal question\nP(Y | do(X))",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=PURPLE,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#EDE9FE", edgecolor=PURPLE),
    )
    _draw_edge(axis, (0.5, 0.31), (0.3, 0.23), color=BLUE)
    _draw_edge(axis, (0.5, 0.31), (0.7, 0.23), color=PURPLE)
    axis.set_title(
        "Correlation is the start of the causal question—not its answer",
        fontsize=17,
        pad=18,
    )
    path = output_dir / "causal_inference_visual_summary.png"
    _save_static(figure, path, show=show, dpi=170)
    return VisualResult(path, "Causal inference visual summary")


EXPECTED_FILENAMES = (
    "01_correlation_types.png",
    "02_zero_correlation_nonlinear.gif",
    "03_confounded_relationship.png",
    "04_confounder_3d.html",
    "05_naive_vs_adjusted.png",
    "06_omitted_variable_bias.gif",
    "07_simpsons_paradox.png",
    "08_collider_bias.png",
    "08_collider_bias.gif",
    "09_observation_vs_intervention.gif",
    "10_rag_observation_vs_intervention.png",
    "causal_inference_visual_summary.png",
)


def generate_all(
    output_dir: Path,
    *,
    seed: int = 42,
    show: bool = False,
    quick: bool = False,
) -> list[VisualResult]:
    """Generate the complete visual story in causal-learning order."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    results = [generate_correlation_types(output_dir, seed=seed, show=show)]
    results.append(
        generate_nonlinear_animation(output_dir, seed=seed, quick=quick)
    )
    confounding = build_confounding_experiment(
        seed=seed, sample_size=900 if quick else 2_000
    )
    results.append(
        generate_confounded_relationship(confounding, output_dir, show=show)
    )
    results.append(generate_confounder_3d(confounding, output_dir, show=show))
    results.append(generate_naive_vs_adjusted(confounding, output_dir, show=show))
    results.append(
        generate_omitted_variable_bias_animation(
            output_dir, seed=seed, quick=quick
        )
    )
    simpson = build_simpson_experiment(
        seed=seed, sample_size=450 if quick else 900
    )
    results.append(generate_simpsons_paradox(simpson, output_dir, show=show))
    collider = build_collider_experiment(
        seed=seed, sample_size=2_000 if quick else 6_000
    )
    results.append(generate_collider_bias(collider, output_dir, show=show))
    results.append(
        generate_collider_animation(collider, output_dir, quick=quick)
    )
    results.append(generate_observation_vs_intervention(output_dir, quick=quick))
    rag = build_rag_experiment(
        seed=seed, sample_size=1_000 if quick else 2_500
    )
    results.append(
        generate_rag_observation_vs_intervention(rag, output_dir, show=show)
    )
    results.append(generate_visual_summary(output_dir, show=show))
    return results


def parse_arguments() -> argparse.Namespace:
    """Parse output, seed, interactive-display, and smoke-render options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for PNG, GIF, and HTML assets.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Synthetic-data seed.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display static Matplotlib figures and the Plotly figure after saving.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer animation frames and smaller samples for a smoke run.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate every visual and report diagnostics from the completed run."""
    args = parse_arguments()
    print("Day 13 — Correlation vs Causation Visual Lab")
    print("Synthetic data only; causal interpretations require author review.\n")
    results = generate_all(
        args.output_dir,
        seed=args.seed,
        show=args.show,
        quick=args.quick,
    )
    print("Generated:")
    for result in results:
        print(f"[OK] {result.label}: {result.path.name}")
        for metric in result.metrics:
            print(f"     {metric}")
    print(f"\nOutput directory:\n{args.output_dir.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
