"""Generate a standalone visual learning lab for information theory.

All trajectories and distributions are deterministic educational constructions.
They are not neural-network or language-model training runs.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation, PillowWriter
from plotly.subplots import make_subplots

from from_scratch import (
    cross_entropy,
    entropy,
    kl_divergence,
    softmax,
    validate_distribution,
)


TOPIC_DIR = Path(__file__).resolve().parent
DEFAULT_ASSET_DIR = TOPIC_DIR / "assets"
COLORS = {
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "green": "#16A34A",
    "orange": "#EA580C",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "light": "#E2E8F0",
    "dark": "#0F172A",
}


def _style_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.set_facecolor("#F8FAFC")
    axis.grid(axis=grid_axis, alpha=0.22, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _save_animation(
    animation: FuncAnimation, figure: plt.Figure, path: Path, *, fps: int = 10
) -> Path:
    animation.save(path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    return path


def binary_entropy(probabilities: np.ndarray | float) -> np.ndarray | float:
    """Return Bernoulli entropy in nats without evaluating log(0)."""
    values = np.asarray(probabilities, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be finite values in [0, 1].")
    result = np.zeros_like(values)
    interior = (values > 0.0) & (values < 1.0)
    p = values[interior]
    result[interior] = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(result) if result.ndim == 0 else result


def stable_softmax(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    """Return softmax(logits / temperature) with explicit validation."""
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    return softmax(np.asarray(logits, dtype=float) / temperature)


def distribution_path(
    start: np.ndarray, end: np.ndarray, *, steps: int = 28
) -> np.ndarray:
    """Interpolate between two categorical distributions."""
    start = validate_distribution(start, name="start")
    end = validate_distribution(end, name="end")
    if start.shape != end.shape:
        raise ValueError("start and end must have the same shape.")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer of at least 2.")
    weights = np.linspace(0.0, 1.0, steps)[:, np.newaxis]
    return (1.0 - weights) * start + weights * end


def simplex_grid(*, resolution: int = 40) -> np.ndarray:
    """Return strictly positive points on a three-class probability simplex."""
    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise TypeError("resolution must be an integer.")
    if resolution < 6:
        raise ValueError("resolution must be at least 6.")
    points = []
    for first in range(1, resolution - 1):
        for second in range(1, resolution - first):
            third = resolution - first - second
            if third >= 1:
                points.append([first, second, third])
    return np.asarray(points, dtype=float) / resolution


def normal_density(x: np.ndarray, *, mean: float, std: float) -> np.ndarray:
    """Evaluate a univariate Normal density."""
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("std must be finite and positive.")
    values = np.asarray(x, dtype=float)
    return np.exp(-0.5 * ((values - mean) / std) ** 2) / (
        std * math.sqrt(2.0 * math.pi)
    )


def mixture_density(x: np.ndarray) -> np.ndarray:
    """Evaluate the fixed two-mode target density used for KL intuition."""
    return 0.5 * normal_density(x, mean=-2.0, std=0.65) + 0.5 * normal_density(
        x, mean=2.0, std=0.65
    )


def continuous_kl(x: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """Numerically integrate KL(P || Q) for positive sampled densities."""
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if x.ndim != 1 or p.shape != x.shape or q.shape != x.shape:
        raise ValueError("x, p, and q must be aligned one-dimensional arrays.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)):
        raise ValueError("x, p, and q must contain only finite values.")
    if np.any(p <= 0.0) or np.any(q <= 0.0):
        raise ValueError("sampled densities must be strictly positive.")
    integrand = p * np.log(p / q)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(integrand, x))
    return float(np.trapz(integrand, x))


def plot_self_information(output_dir: Path) -> Path:
    probabilities = np.geomspace(0.005, 1.0, 600)
    information = -np.log(probabilities)
    examples = np.array([0.9, 0.5, 0.1, 0.01])

    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot(probabilities, information, color=COLORS["purple"], linewidth=2.6)
    axis.scatter(examples, -np.log(examples), color=COLORS["orange"], zorder=3)
    for probability in examples:
        value = -math.log(probability)
        axis.annotate(
            f"p={probability:g}\nI={value:.2f} nats",
            (probability, value),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set(xlabel="Event probability p", ylabel="Self-information -log(p) [nats]")
    axis.set_title("Rare events carry more self-information", fontweight="bold")
    axis.set_xlim(0.0, 1.0)
    _style_axis(axis, grid_axis="both")
    figure.tight_layout()
    return _save_figure(figure, output_dir / "self_information.png")


def plot_bernoulli_entropy(output_dir: Path) -> Path:
    probabilities = np.linspace(0.0, 1.0, 501)
    values = binary_entropy(probabilities)
    maximum = math.log(2.0)

    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot(probabilities, values, color=COLORS["blue"], linewidth=2.8)
    axis.scatter([0.5], [maximum], color=COLORS["orange"], s=55, zorder=3)
    axis.annotate(
        f"Maximum uncertainty\np=0.5, H=ln(2)={maximum:.3f} nats",
        (0.5, maximum),
        xytext=(0.64, 0.58),
        arrowprops={"arrowstyle": "->", "color": COLORS["gray"]},
        fontsize=10,
    )
    axis.set(
        xlabel="Bernoulli probability p",
        ylabel="Entropy H(p) [nats]",
        title="Bernoulli entropy measures uncertainty, not correctness",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 0.76)
    _style_axis(axis, grid_axis="both")
    figure.tight_layout()
    return _save_figure(figure, output_dir / "bernoulli_entropy.png")


def animate_entropy_evolution(output_dir: Path) -> Path:
    forward = distribution_path(
        np.array([1.0, 0.0, 0.0]), np.full(3, 1.0 / 3.0), steps=24
    )
    trajectory = np.vstack((forward, forward[-2:0:-1]))
    figure, axis = plt.subplots(figsize=(8, 5))

    def update(frame: int) -> None:
        axis.clear()
        current = trajectory[frame]
        value = entropy(current)
        axis.bar(["Class 0", "Class 1", "Class 2"], current, color=COLORS["blue"])
        axis.axhline(1.0 / 3.0, color=COLORS["gray"], linestyle="--", alpha=0.65)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Probability")
        axis.set_title(
            f"Concentrated <-> uniform: entropy = {abs(value):.3f} nats",
            fontweight="bold",
        )
        axis.text(
            0.5,
            0.91,
            "Uniformity increases uncertainty",
            transform=axis.transAxes,
            ha="center",
            color=COLORS["dark"],
        )
        _style_axis(axis)

    animation = FuncAnimation(figure, update, frames=len(trajectory), interval=90)
    return _save_animation(
        animation, figure, output_dir / "entropy_evolution.gif", fps=11
    )


def plot_cross_entropy_confidence(output_dir: Path) -> Path:
    probabilities = np.geomspace(0.005, 1.0, 700)
    losses = -np.log(probabilities)
    examples = np.array([0.99, 0.9, 0.5, 0.1, 0.01])

    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot(probabilities, losses, color=COLORS["red"], linewidth=2.8)
    axis.scatter(examples, -np.log(examples), color=COLORS["purple"], zorder=3)
    for probability in examples:
        loss = -math.log(probability)
        offset = (-50, 10) if probability > 0.85 else (8, 8)
        axis.annotate(
            f"p={probability:g}\nL={loss:.2f}",
            (probability, loss),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
        )
    axis.set(
        xlabel="Probability assigned to the true class",
        ylabel="Cross-entropy -log(p_true) [nats]",
        title="Cross-entropy heavily penalizes confident mistakes",
    )
    axis.set_xlim(0.0, 1.0)
    _style_axis(axis, grid_axis="both")
    figure.tight_layout()
    return _save_figure(figure, output_dir / "cross_entropy_confidence.png")


def animate_classification_learning(output_dir: Path) -> Path:
    trajectory = distribution_path(
        np.array([0.05, 0.90, 0.05]), np.array([0.95, 0.03, 0.02]), steps=30
    )
    figure, axis = plt.subplots(figsize=(8, 5))

    def update(frame: int) -> None:
        axis.clear()
        prediction = trajectory[frame]
        loss = -math.log(prediction[0])
        colors = [COLORS["green"], COLORS["gray"], COLORS["gray"]]
        axis.bar(["Class 0 (true)", "Class 1", "Class 2"], prediction, color=colors)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Predicted probability")
        axis.set_title(
            f"Educational probability interpolation | q_true={prediction[0]:.3f} | CE={loss:.3f}",
            fontweight="bold",
        )
        axis.text(
            0.5,
            0.91,
            "Conceptual direction of training - not a fitted model trajectory",
            transform=axis.transAxes,
            ha="center",
            fontsize=9,
        )
        _style_axis(axis)

    animation = FuncAnimation(figure, update, frames=len(trajectory), interval=90)
    return _save_animation(
        animation, figure, output_dir / "classification_learning.gif", fps=11
    )


def plot_ce_kl_decomposition(output_dir: Path) -> Path:
    target = np.array([0.7, 0.2, 0.1])
    candidates = {
        "Match": target,
        "Small mismatch": np.array([0.6, 0.3, 0.1]),
        "Uniform": np.full(3, 1.0 / 3.0),
        "Large mismatch": np.array([0.15, 0.2, 0.65]),
    }
    target_entropy = entropy(target)
    kl_values = np.array([kl_divergence(target, q) for q in candidates.values()])
    ce_values = np.array([cross_entropy(target, q) for q in candidates.values()])

    figure, axis = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(candidates))
    axis.bar(positions, target_entropy, color=COLORS["blue"], label="H(P), constant")
    axis.bar(
        positions,
        kl_values,
        bottom=target_entropy,
        color=COLORS["orange"],
        label="KL(P || Q)",
    )
    for position, value in zip(positions, ce_values, strict=True):
        axis.text(position, value + 0.04, f"CE={value:.2f}", ha="center", fontsize=9)
    axis.set_xticks(positions, candidates.keys())
    axis.set_ylabel("Information [nats]")
    axis.set_title("Cross-entropy = target entropy + distribution mismatch", fontweight="bold")
    axis.legend(frameon=False)
    _style_axis(axis)
    figure.tight_layout()
    return _save_figure(figure, output_dir / "ce_entropy_kl_decomposition.png")


def animate_kl_convergence(output_dir: Path) -> Path:
    target = np.array([0.7, 0.2, 0.1])
    trajectory = distribution_path(np.array([0.1, 0.2, 0.7]), target, steps=30)
    kl_history = np.array([kl_divergence(target, q) for q in trajectory])
    ce_history = np.array([cross_entropy(target, q) for q in trajectory])
    target_entropy = entropy(target)
    figure, (bars, curve) = plt.subplots(1, 2, figsize=(10, 4.8))
    positions = np.arange(3)

    def update(frame: int) -> None:
        bars.clear()
        curve.clear()
        prediction = trajectory[frame]
        width = 0.34
        bars.bar(positions - width / 2, target, width, label="Target P", color=COLORS["blue"])
        bars.bar(
            positions + width / 2,
            prediction,
            width,
            label="Prediction Q",
            color=COLORS["orange"],
        )
        bars.set_xticks(positions, ["Class 0", "Class 1", "Class 2"])
        bars.set_ylim(0.0, 0.82)
        bars.set_ylabel("Probability")
        bars.set_title(
            f"KL={kl_history[frame]:.3f} | CE={ce_history[frame]:.3f} | H(P)={target_entropy:.3f}"
        )
        bars.legend(frameon=False, loc="upper center")
        _style_axis(bars)

        curve.plot(kl_history, color=COLORS["purple"], alpha=0.28, linewidth=2)
        curve.plot(
            np.arange(frame + 1),
            kl_history[: frame + 1],
            color=COLORS["purple"],
            linewidth=2.5,
        )
        curve.scatter([frame], [kl_history[frame]], color=COLORS["red"], zorder=3)
        curve.set(xlabel="Interpolation step", ylabel="KL(P || Q) [nats]")
        curve.set_ylim(0.0, kl_history.max() * 1.08)
        curve.set_title("Mismatch approaches zero")
        _style_axis(curve, grid_axis="both")
        figure.suptitle(
            "As Q approaches P, cross-entropy approaches H(P)",
            fontweight="bold",
        )
        figure.tight_layout()

    animation = FuncAnimation(figure, update, frames=len(trajectory), interval=100)
    return _save_animation(animation, figure, output_dir / "kl_convergence.gif", fps=10)


def plot_kl_asymmetry(output_dir: Path) -> Path:
    examples = [
        ("Moderate", np.array([0.8, 0.1, 0.1]), np.array([0.4, 0.3, 0.3])),
        ("Extreme", np.array([0.97, 0.02, 0.01]), np.array([0.34, 0.33, 0.33])),
    ]
    forward = [kl_divergence(p, q) for _, p, q in examples]
    reverse = [kl_divergence(q, p) for _, p, q in examples]
    positions = np.arange(len(examples))
    width = 0.34

    figure, axis = plt.subplots(figsize=(9, 5.4))
    first = axis.bar(
        positions - width / 2,
        forward,
        width,
        color=COLORS["blue"],
        label="KL(P || Q)",
    )
    second = axis.bar(
        positions + width / 2,
        reverse,
        width,
        color=COLORS["orange"],
        label="KL(Q || P)",
    )
    axis.bar_label(first, fmt="%.3f", padding=3)
    axis.bar_label(second, fmt="%.3f", padding=3)
    axis.set_xticks(positions, [item[0] for item in examples])
    axis.set_ylabel("Divergence [nats]")
    axis.set_title("KL(P || Q) != KL(Q || P): KL is not a distance", fontweight="bold")
    axis.legend(frameon=False)
    _style_axis(axis)
    figure.tight_layout()
    return _save_figure(figure, output_dir / "kl_asymmetry.png")


def write_kl_simplex(output_dir: Path) -> Path:
    target = np.array([0.7, 0.2, 0.1])
    points = simplex_grid(resolution=40)
    values = np.array([kl_divergence(target, point) for point in points])
    figure = go.Figure()
    figure.add_trace(
        go.Scatterternary(
            a=points[:, 0],
            b=points[:, 1],
            c=points[:, 2],
            mode="markers",
            name="Candidate Q",
            marker={
                "size": 7,
                "color": values,
                "colorscale": "Viridis",
                "colorbar": {
                    "title": "KL(P||Q) [nats]",
                    "x": 0.86,
                    "y": 0.42,
                    "len": 0.64,
                },
                "showscale": True,
            },
            showlegend=False,
            customdata=values,
            hovertemplate=(
                "q1=%{a:.3f}<br>q2=%{b:.3f}<br>q3=%{c:.3f}"
                "<br>KL=%{customdata:.4f} nats<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatterternary(
            a=[target[0]],
            b=[target[1]],
            c=[target[2]],
            mode="markers",
            marker={"size": 13, "color": COLORS["red"], "symbol": "diamond"},
            name="Q = P (KL = 0)",
            hovertemplate="P=[0.7, 0.2, 0.1]<br>KL=0<extra></extra>",
        )
    )
    figure.update_layout(
        title="Forward-KL landscape on the three-class probability simplex",
        ternary={
            "sum": 1,
            "domain": {"x": [0.0, 0.72], "y": [0.0, 1.0]},
            "aaxis": {"title": "q1"},
            "baxis": {"title": "q2"},
            "caxis": {"title": "q3"},
        },
        legend={"x": 0.78, "y": 0.96, "xanchor": "left", "yanchor": "top"},
        template="plotly_white",
        width=920,
        height=700,
    )
    path = output_dir / "kl_probability_simplex.html"
    figure.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def plot_logits_to_softmax(output_dir: Path) -> Path:
    logits = np.array([3.0, 1.0, 0.0])
    probabilities = stable_softmax(logits)
    labels = ["Class 0", "Class 1", "Class 2"]
    colors = [COLORS["blue"], COLORS["cyan"], COLORS["purple"]]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4.8))
    left.bar(labels, logits, color=colors)
    left.set(ylabel="Logit (unbounded score)", title="Model logits")
    right.bar(labels, probabilities, color=colors)
    right.set(ylabel="Probability", title="Softmax probabilities", ylim=(0.0, 1.0))
    right.text(
        0.5,
        0.92,
        f"sum = {probabilities.sum():.1f}; ranking preserved",
        transform=right.transAxes,
        ha="center",
    )
    for axis in (left, right):
        _style_axis(axis)
    figure.suptitle("Softmax maps real-valued logits to a distribution", fontweight="bold")
    figure.tight_layout()
    return _save_figure(figure, output_dir / "logits_to_softmax.png")


def animate_temperature_entropy(output_dir: Path) -> Path:
    logits = np.array([3.0, 1.0, 0.0])
    temperatures = np.geomspace(0.1, 5.0, 36)
    figure, axis = plt.subplots(figsize=(8, 5))

    def update(frame: int) -> None:
        axis.clear()
        temperature = temperatures[frame]
        probabilities = stable_softmax(logits, temperature=temperature)
        value = entropy(probabilities)
        axis.bar(
            ["Class 0", "Class 1", "Class 2"],
            probabilities,
            color=[COLORS["blue"], COLORS["cyan"], COLORS["purple"]],
        )
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Probability")
        axis.set_title(
            f"softmax(z / T): T={temperature:.2f}, entropy={value:.3f} nats",
            fontweight="bold",
        )
        tendency = "Sharper / lower entropy" if temperature < 1.0 else "Flatter / higher entropy"
        axis.text(0.5, 0.91, tendency, transform=axis.transAxes, ha="center")
        axis.text(
            0.5,
            -0.17,
            "Inference sampling intuition; training-time temperature has a different role.",
            transform=axis.transAxes,
            ha="center",
            fontsize=9,
        )
        _style_axis(axis)
        figure.tight_layout()

    animation = FuncAnimation(figure, update, frames=len(temperatures), interval=100)
    return _save_animation(
        animation, figure, output_dir / "temperature_entropy.gif", fps=10
    )


def write_temperature_interactive(output_dir: Path) -> Path:
    logits = np.array([3.0, 1.0, 0.0])
    temperatures = np.geomspace(0.1, 5.0, 160)
    probabilities = np.vstack(
        [stable_softmax(logits, temperature=value) for value in temperatures]
    )
    entropies = np.array([entropy(row) for row in probabilities])
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    for index, label in enumerate(["Class 0", "Class 1", "Class 2"]):
        figure.add_trace(
            go.Scatter(
                x=temperatures,
                y=probabilities[:, index],
                mode="lines",
                name=f"P({label})",
                hovertemplate="T=%{x:.3f}<br>probability=%{y:.4f}<extra></extra>",
            ),
            secondary_y=False,
        )
    figure.add_trace(
        go.Scatter(
            x=temperatures,
            y=entropies,
            mode="lines",
            name="Entropy",
            line={"color": COLORS["dark"], "width": 4, "dash": "dash"},
            hovertemplate="T=%{x:.3f}<br>entropy=%{y:.4f} nats<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_xaxes(title="Temperature T", type="log")
    figure.update_yaxes(title="Class probability", range=[0, 1], secondary_y=False)
    figure.update_yaxes(title="Entropy [nats]", secondary_y=True)
    figure.update_layout(
        title="Temperature changes class probabilities and entropy",
        template="plotly_white",
        width=980,
        height=600,
        hovermode="x unified",
    )
    path = output_dir / "temperature_interactive.html"
    figure.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def plot_label_smoothing(output_dir: Path) -> Path:
    hard = np.array([1.0, 0.0, 0.0])
    smoothed = np.array([0.9, 0.05, 0.05])
    positions = np.arange(3)
    width = 0.34
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.bar(
        positions - width / 2,
        hard,
        width,
        label=f"Hard target, H={abs(entropy(hard)):.3f}",
        color=COLORS["blue"],
    )
    axis.bar(
        positions + width / 2,
        smoothed,
        width,
        label=f"Smoothed target, H={entropy(smoothed):.3f}",
        color=COLORS["orange"],
    )
    axis.set_xticks(positions, ["Class 0 (target)", "Class 1", "Class 2"])
    axis.set(ylabel="Target probability", ylim=(0.0, 1.08))
    axis.set_title("Label smoothing changes the training target", fontweight="bold")
    axis.legend(frameon=False)
    axis.text(
        0.5,
        -0.16,
        "It discourages an exactly one-hot target; performance and calibration effects are context-dependent.",
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
    )
    _style_axis(axis)
    figure.tight_layout()
    return _save_figure(figure, output_dir / "label_smoothing.png")


def plot_forward_reverse_kl(output_dir: Path) -> Path:
    x = np.linspace(-6.0, 6.0, 2400)
    target = mixture_density(x)
    covering = normal_density(x, mean=0.0, std=math.sqrt(2.0**2 + 0.65**2))
    mode = normal_density(x, mean=-2.0, std=0.65)
    forward_value = continuous_kl(x, target, covering)
    reverse_value = continuous_kl(x, mode, target)

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    left.plot(x, target, color=COLORS["dark"], linewidth=2.5, label="Target P: two modes")
    left.plot(x, covering, color=COLORS["blue"], linewidth=2.5, label="Broad Q")
    left.fill_between(x, covering, alpha=0.14, color=COLORS["blue"])
    left.set_title(f"Forward-KL intuition: cover mass\nKL(P||Q)={forward_value:.3f}")
    right.plot(x, target, color=COLORS["dark"], linewidth=2.5, label="Target P: two modes")
    right.plot(x, mode, color=COLORS["orange"], linewidth=2.5, label="One-mode Q")
    right.fill_between(x, mode, alpha=0.14, color=COLORS["orange"])
    right.set_title(f"Reverse-KL intuition: select a mode\nKL(Q||P)={reverse_value:.3f}")
    for axis in (left, right):
        axis.set(xlabel="x", ylabel="Density")
        axis.legend(frameon=False, fontsize=9)
        _style_axis(axis, grid_axis="both")
    figure.suptitle(
        "Illustrative Gaussian approximations - useful tendency, not a universal rule",
        fontweight="bold",
    )
    figure.tight_layout()
    return _save_figure(figure, output_dir / "forward_reverse_kl.png")


def plot_llm_next_token(output_dir: Path) -> Path:
    context = '"The capital of France is"'
    vocabulary = ["Paris", "London", "Berlin", "Madrid", "Rome"]
    logits = np.array([2.8, 0.8, 0.2, 0.0, -0.5])
    probabilities = stable_softmax(logits)
    loss = -math.log(probabilities[0])
    colors = [COLORS["green"]] + [COLORS["gray"]] * 4
    figure, (diagram, bars) = plt.subplots(
        1, 2, figsize=(12, 5.3), gridspec_kw={"width_ratios": [1.0, 1.35]}
    )
    diagram.axis("off")
    nodes = [
        (0.88, f"Context\n{context}"),
        (0.67, f"Vocabulary logits\n{np.array2string(logits, precision=1)}"),
        (0.46, "Stable softmax"),
        (0.25, "Predicted distribution Q"),
        (0.04, f"Observed token = Paris\nCE = -log P(Paris) = {loss:.3f} nats"),
    ]
    for y, label in nodes:
        diagram.text(
            0.5,
            y,
            label,
            ha="center",
            va="center",
            transform=diagram.transAxes,
            bbox={"boxstyle": "round,pad=0.45", "fc": "#F8FAFC", "ec": COLORS["blue"]},
        )
    for upper, lower in zip(nodes[:-1], nodes[1:], strict=True):
        diagram.annotate(
            "",
            xy=(0.5, lower[0] + 0.065),
            xytext=(0.5, upper[0] - 0.065),
            xycoords=diagram.transAxes,
            arrowprops={"arrowstyle": "->", "color": COLORS["dark"], "lw": 1.5},
        )
    bars.barh(vocabulary[::-1], probabilities[::-1], color=colors[::-1])
    bars.set_xlim(0.0, 1.0)
    bars.set_xlabel("Next-token probability")
    bars.set_title("Synthetic vocabulary distribution")
    for index, value in enumerate(probabilities[::-1]):
        bars.text(value + 0.015, index, f"{value:.3f}", va="center")
    _style_axis(bars, grid_axis="x")
    figure.suptitle("Next-token cross-entropy connects logits to an observed token", fontweight="bold")
    figure.tight_layout()
    return _save_figure(figure, output_dir / "llm_next_token_cross_entropy.png")


def animate_llm_token_learning(output_dir: Path) -> Path:
    vocabulary = ["Paris", "London", "Berlin", "Madrid", "Rome"]
    trajectory = distribution_path(
        np.array([0.10, 0.40, 0.25, 0.15, 0.10]),
        np.array([0.90, 0.025, 0.025, 0.025, 0.025]),
        steps=30,
    )
    figure, axis = plt.subplots(figsize=(9, 5))

    def update(frame: int) -> None:
        axis.clear()
        probabilities = trajectory[frame]
        loss = -math.log(probabilities[0])
        perplexity = math.exp(loss)
        axis.bar(
            vocabulary,
            probabilities,
            color=[COLORS["green"]] + [COLORS["gray"]] * 4,
        )
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Next-token probability")
        axis.set_title(
            f"Paris probability={probabilities[0]:.3f} | CE={loss:.3f} | single-token PPL={perplexity:.3f}",
            fontweight="bold",
        )
        axis.text(
            0.5,
            0.91,
            "Educational synthetic optimization trajectory - not real LLM training",
            transform=axis.transAxes,
            ha="center",
            fontsize=9,
        )
        _style_axis(axis)
        figure.tight_layout()

    animation = FuncAnimation(figure, update, frames=len(trajectory), interval=100)
    return _save_animation(
        animation, figure, output_dir / "llm_token_learning.gif", fps=10
    )


def plot_cross_entropy_perplexity(output_dir: Path) -> Path:
    losses = np.linspace(0.0, 5.0, 500)
    perplexities = np.exp(losses)
    examples = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.plot(losses, perplexities, color=COLORS["purple"], linewidth=2.8)
    axis.scatter(examples, np.exp(examples), color=COLORS["orange"], zorder=3)
    for loss in examples:
        axis.annotate(
            f"CE={loss:g}\nPPL={math.exp(loss):.1f}",
            (loss, math.exp(loss)),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set(
        xlabel="Mean cross-entropy / token NLL [nats]",
        ylabel="Perplexity exp(CE)",
        title="Perplexity grows exponentially with mean token cross-entropy",
    )
    axis.text(
        0.5,
        -0.17,
        "Compare only with compatible tokenizers, datasets, masking, and context treatment.",
        transform=axis.transAxes,
        ha="center",
        fontsize=9,
    )
    _style_axis(axis, grid_axis="both")
    figure.tight_layout()
    return _save_figure(figure, output_dir / "cross_entropy_vs_perplexity.png")


def write_information_playground(output_dir: Path) -> Path:
    examples = {
        "Perfect match": (np.array([0.7, 0.2, 0.1]), np.array([0.7, 0.2, 0.1])),
        "Small mismatch": (np.array([0.7, 0.2, 0.1]), np.array([0.6, 0.3, 0.1])),
        "Large mismatch": (np.array([0.7, 0.2, 0.1]), np.array([0.15, 0.2, 0.65])),
        "Confident wrong": (np.array([0.9, 0.05, 0.05]), np.array([0.01, 0.04, 0.95])),
        "Uniform prediction": (np.array([0.7, 0.2, 0.1]), np.full(3, 1.0 / 3.0)),
        "Peaked target": (np.array([0.98, 0.01, 0.01]), np.array([0.8, 0.1, 0.1])),
    }
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Target P and prediction Q", "Information quantities"),
    )
    for index, (name, (p, q)) in enumerate(examples.items()):
        visible = index == 0
        figure.add_trace(
            go.Bar(x=["Class 0", "Class 1", "Class 2"], y=p, name="P", visible=visible),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(x=["Class 0", "Class 1", "Class 2"], y=q, name="Q", visible=visible),
            row=1,
            col=1,
        )
        metrics = [entropy(p), cross_entropy(p, q), kl_divergence(p, q), kl_divergence(q, p)]
        figure.add_trace(
            go.Bar(
                x=["H(P)", "H(P,Q)", "KL(P||Q)", "KL(Q||P)"],
                y=metrics,
                name="Metrics",
                marker_color=COLORS["purple"],
                text=[f"{value:.3f}" for value in metrics],
                textposition="outside",
                visible=visible,
                hovertemplate="%{x}: %{y:.4f} nats<extra></extra>",
            ),
            row=1,
            col=2,
        )
    trace_count = 3 * len(examples)
    buttons = []
    for index, name in enumerate(examples):
        visibility = [False] * trace_count
        visibility[index * 3 : index * 3 + 3] = [True, True, True]
        buttons.append(
            {
                "label": name,
                "method": "update",
                "args": [{"visible": visibility}, {"title": f"Probability playground: {name}"}],
            }
        )
    figure.update_layout(
        title="Probability playground: Perfect match",
        barmode="group",
        template="plotly_white",
        width=1100,
        height=610,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.5,
                "xanchor": "center",
                "y": 1.16,
                "yanchor": "top",
            }
        ],
    )
    figure.update_yaxes(title="Probability", range=[0, 1.08], row=1, col=1)
    figure.update_yaxes(title="Information [nats]", row=1, col=2)
    path = output_dir / "information_theory_playground.html"
    figure.write_html(path, include_plotlyjs=True, full_html=True)
    return path


Generator = Callable[[Path], Path]
ASSET_GENERATORS: tuple[tuple[str, Generator], ...] = (
    ("entropy", plot_self_information),
    ("entropy", plot_bernoulli_entropy),
    ("entropy", animate_entropy_evolution),
    ("entropy", plot_cross_entropy_confidence),
    ("entropy", animate_classification_learning),
    ("kl", plot_ce_kl_decomposition),
    ("kl", animate_kl_convergence),
    ("kl", plot_kl_asymmetry),
    ("kl", write_kl_simplex),
    ("softmax", plot_logits_to_softmax),
    ("softmax", animate_temperature_entropy),
    ("softmax", write_temperature_interactive),
    ("softmax", plot_label_smoothing),
    ("kl", plot_forward_reverse_kl),
    ("llm", plot_llm_next_token),
    ("llm", animate_llm_token_learning),
    ("llm", plot_cross_entropy_perplexity),
    ("llm", write_information_playground),
)


def selected_generators(selection: str) -> tuple[Generator, ...]:
    """Return generators for one CLI group or the complete lab."""
    if selection not in {"all", "entropy", "kl", "softmax", "llm"}:
        raise ValueError("selection must be all, entropy, kl, softmax, or llm.")
    return tuple(
        generator
        for group, generator in ASSET_GENERATORS
        if selection == "all" or group == selection
    )


def expected_asset_names() -> tuple[str, ...]:
    """Return the complete ordered asset manifest."""
    return (
        "self_information.png",
        "bernoulli_entropy.png",
        "entropy_evolution.gif",
        "cross_entropy_confidence.png",
        "classification_learning.gif",
        "ce_entropy_kl_decomposition.png",
        "kl_convergence.gif",
        "kl_asymmetry.png",
        "kl_probability_simplex.html",
        "logits_to_softmax.png",
        "temperature_entropy.gif",
        "temperature_interactive.html",
        "label_smoothing.png",
        "forward_reverse_kl.png",
        "llm_next_token_cross_entropy.png",
        "llm_token_learning.gif",
        "cross_entropy_vs_perplexity.png",
        "information_theory_playground.html",
    )


def generate_assets(selection: str, output_dir: Path = DEFAULT_ASSET_DIR) -> list[Path]:
    """Generate the requested asset group and return successful paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for generator in selected_generators(selection):
        generated.append(generator(output_dir))
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "entropy", "kl", "softmax", "llm"),
        default="all",
        help="Generate one conceptual group or the complete lab.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help="Asset directory; defaults to the topic-local assets folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = generate_assets(args.only, args.output_dir)
    print("Generated visual assets:")
    for path in generated:
        try:
            display = path.relative_to(TOPIC_DIR)
        except ValueError:
            display = path
        print(display.as_posix())


if __name__ == "__main__":
    main()
