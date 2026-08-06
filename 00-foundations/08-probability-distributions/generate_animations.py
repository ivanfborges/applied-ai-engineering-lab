"""Generate educational probability-distribution GIFs with PillowWriter."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import trapezoid
from scipy import stats

from distribution_utils import (
    COLORS,
    SEED,
    ensure_output_directories,
    total_absolute_probability_difference,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOPIC_DIRECTORY = Path(__file__).resolve().parent
GIF_DIRECTORY = TOPIC_DIRECTORY / "outputs" / "gifs"


def configure_matplotlib() -> None:
    """Apply the same restrained style used by the static visualizations."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 105,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def _save_animation(
    figure: plt.Figure,
    update: Callable[[int], None],
    frame_count: int,
    output_path: Path,
    fps: int,
    dpi: int,
) -> Path:
    """Create and save one animation using Pillow's always-available writer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 / fps,
        repeat=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
    print(f"Saved: {output_path}")
    return output_path


def animate_bernoulli(
    output_path: Path,
    frame_count: int,
    fps: int,
    dpi: int,
) -> Path:
    """Animate Bernoulli probability mass as p moves from 0.01 to 0.99."""
    probabilities = np.linspace(0.01, 0.99, frame_count)
    figure, axis = plt.subplots(figsize=(8, 5.5))

    def update(frame: int) -> None:
        p = float(probabilities[frame])
        axis.clear()
        axis.bar(
            [0, 1],
            [1 - p, p],
            color=[COLORS["empirical"], COLORS["theoretical"]],
            width=0.55,
        )
        axis.set(
            title="Bernoulli: probability mass moves between 0 and 1",
            xlabel="Outcome",
            ylabel="Probability mass",
            xticks=[0, 1],
            ylim=(0, 1.12),
        )
        axis.text(
            0.5,
            1.04,
            f"p={p:.2f}   E[X]={p:.2f}   Var(X)={p * (1 - p):.3f}",
            ha="center",
            transform=axis.get_xaxis_transform(),
        )

    return _save_animation(
        figure, update, frame_count, output_path, fps, dpi
    )


def animate_binomial_probability(
    output_path: Path,
    frame_count: int,
    fps: int,
    dpi: int,
) -> Path:
    """Animate a Binomial PMF as p changes while n remains fixed."""
    n = 30
    probabilities = np.linspace(0.05, 0.95, frame_count)
    support = np.arange(n + 1)
    figure, axis = plt.subplots(figsize=(9, 5.8))

    def update(frame: int) -> None:
        p = float(probabilities[frame])
        mean = n * p
        variance = n * p * (1 - p)
        axis.clear()
        axis.bar(
            support,
            stats.binom.pmf(support, n, p),
            color=COLORS["empirical"],
            width=0.78,
        )
        axis.axvline(
            mean,
            color=COLORS["warning"],
            linestyle="--",
            linewidth=2,
            label=f"E[X]=np={mean:.1f}",
        )
        axis.set(
            title="Binomial: changing success probability",
            xlabel="Success count k",
            ylabel="P(X = k)",
            xlim=(-0.75, n + 0.75),
            ylim=(0, 0.36),
        )
        axis.text(
            0.02,
            0.95,
            f"n={n}, p={p:.2f}, Var(X)={variance:.2f}",
            transform=axis.transAxes,
            va="top",
        )
        axis.legend(loc="upper right")

    return _save_animation(
        figure, update, frame_count, output_path, fps, dpi
    )


def animate_binomial_to_normal(
    output_path: Path,
    frame_count: int,
    fps: int,
    dpi: int,
) -> Path:
    """Animate the continuity-corrected Normal approximation to Binomial."""
    n_values = np.unique(
        np.rint(np.geomspace(5, 200, frame_count)).astype(int)
    )
    p = 0.5
    figure, axis = plt.subplots(figsize=(10, 6))

    def update(frame: int) -> None:
        n = int(n_values[frame])
        support = np.arange(n + 1)
        mean = n * p
        variance = n * p * (1 - p)
        std = np.sqrt(variance)
        binomial_mass = stats.binom.pmf(support, n, p)
        # Each Normal point represents the mass inside [k-0.5, k+0.5].
        normal_mass = stats.norm.cdf(
            support + 0.5, loc=mean, scale=std
        ) - stats.norm.cdf(support - 0.5, loc=mean, scale=std)
        axis.clear()
        axis.bar(
            support,
            binomial_mass,
            color=COLORS["empirical"],
            alpha=0.60,
            width=0.8,
            label="Binomial PMF",
        )
        axis.plot(
            support,
            normal_mass,
            color=COLORS["theoretical"],
            linewidth=2.3,
            label="Normal approximation with continuity correction",
        )
        span = max(5.0, 4.5 * std)
        axis.set(
            title="Binomial approaching a Normal distribution",
            xlabel="Success count k",
            ylabel="Probability mass",
            xlim=(max(-0.5, mean - span), min(n + 0.5, mean + span)),
            ylim=(0, max(0.22, float(binomial_mass.max()) * 1.22)),
        )
        axis.text(
            0.02,
            0.95,
            f"n={n}, p={p:.1f}, μ={mean:.1f}, σ²={variance:.1f}",
            transform=axis.transAxes,
            va="top",
        )
        axis.legend(loc="upper right")

    return _save_animation(
        figure, update, len(n_values), output_path, fps, dpi
    )


def animate_binomial_to_poisson(
    output_path: Path,
    frame_count: int,
    fps: int,
    dpi: int,
) -> Path:
    """Animate the rare-event Binomial approximation to Poisson."""
    rate = 5.0
    n_values = np.unique(
        np.rint(np.geomspace(5, 1000, frame_count)).astype(int)
    )
    support = np.arange(0, 21)
    poisson_mass = stats.poisson.pmf(support, rate)
    figure, axis = plt.subplots(figsize=(10, 6))

    def update(frame: int) -> None:
        n = int(n_values[frame])
        p = rate / n
        binomial_mass = stats.binom.pmf(support, n, p)
        distance = total_absolute_probability_difference(
            binomial_mass, poisson_mass
        )
        axis.clear()
        axis.bar(
            support,
            binomial_mass,
            color=COLORS["empirical"],
            alpha=0.62,
            width=0.78,
            label="Binomial PMF",
        )
        axis.plot(
            support,
            poisson_mass,
            "o-",
            color=COLORS["theoretical"],
            linewidth=2,
            label="Poisson(λ=5) PMF",
        )
        axis.set(
            title="Binomial approaching Poisson for rare events",
            xlabel="Event count k",
            ylabel="Probability mass",
            xlim=(-0.75, 20.75),
            ylim=(0, 0.21),
        )
        axis.text(
            0.98,
            0.95,
            f"n={n}, p={p:.4f}, np={rate:.1f}\nL1 distance={distance:.4f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
        )
        axis.legend(loc="upper right", bbox_to_anchor=(1.0, 0.77))

    return _save_animation(
        figure, update, len(n_values), output_path, fps, dpi
    )


def animate_poisson_rate(
    output_path: Path,
    frame_count: int,
    fps: int,
    dpi: int,
) -> Path:
    """Animate Poisson event-count behavior as the rate grows."""
    rates = np.linspace(0.5, 20.0, frame_count)
    support = np.arange(0, 46)
    figure, axis = plt.subplots(figsize=(9.5, 5.8))

    def update(frame: int) -> None:
        rate = float(rates[frame])
        axis.clear()
        axis.bar(
            support,
            stats.poisson.pmf(support, rate),
            color=COLORS["empirical"],
            width=0.78,
        )
        axis.axvline(
            rate,
            color=COLORS["warning"],
            linestyle="--",
            label=f"E[X]=λ={rate:.2f}",
        )
        axis.set(
            title="Poisson: event rate changes center and spread",
            xlabel="Event count k",
            ylabel="P(X = k)",
            xlim=(-0.75, 45),
            ylim=(0, 0.63),
        )
        axis.text(
            0.98,
            0.95,
            f"Variance={rate:.2f}\nP(X=0)=exp(-λ)={np.exp(-rate):.4f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
        )
        axis.legend(loc="upper right", bbox_to_anchor=(1.0, 0.78))

    return _save_animation(
        figure, update, frame_count, output_path, fps, dpi
    )


def animate_exponential_rate(
    output_path: Path,
    frame_count: int,
    fps: int,
    dpi: int,
) -> Path:
    """Animate Exponential PDF and survival as the rate changes."""
    rates = np.linspace(0.2, 2.0, frame_count)
    x = np.linspace(0, 12, 600)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4))

    def update(frame: int) -> None:
        rate = float(rates[frame])
        scale = 1 / rate
        for axis in axes:
            axis.clear()
        axes[0].plot(
            x,
            stats.expon.pdf(x, scale=scale),
            color=COLORS["theoretical"],
            linewidth=2.4,
        )
        axes[0].axvline(
            scale,
            color=COLORS["warning"],
            linestyle="--",
            label=f"E[T]={scale:.2f}",
        )
        axes[0].set(
            title="Probability density",
            xlabel="Waiting time t",
            ylabel="f(t)",
            xlim=(0, 12),
            ylim=(0, 2.1),
        )
        axes[0].legend()
        axes[1].plot(
            x,
            stats.expon.sf(x, scale=scale),
            color=COLORS["empirical"],
            linewidth=2.4,
        )
        axes[1].set(
            title="Survival probability",
            xlabel="Waiting time t",
            ylabel="P(T > t)",
            xlim=(0, 12),
            ylim=(0, 1.02),
        )
        figure.suptitle(
            f"Exponential rate λ={rate:.2f}: higher rate means shorter waits"
        )

    return _save_animation(
        figure, update, frame_count, output_path, fps, dpi
    )


def animate_normal_mean(
    output_path: Path,
    frame_count: int,
    fps: int,
    dpi: int,
) -> Path:
    """Animate a Normal curve translating as its mean changes."""
    means = np.linspace(-4.0, 4.0, frame_count)
    std = 1.0
    x = np.linspace(-9, 9, 700)
    figure, axis = plt.subplots(figsize=(9, 5.5))

    def update(frame: int) -> None:
        mean = float(means[frame])
        axis.clear()
        axis.plot(
            x,
            stats.norm.pdf(x, mean, std),
            color=COLORS["theoretical"],
            linewidth=2.5,
        )
        axis.fill_between(
            x,
            stats.norm.pdf(x, mean, std),
            color=COLORS["empirical"],
            alpha=0.24,
        )
        axis.axvline(mean, color=COLORS["warning"], linestyle="--")
        axis.set(
            title=f"Normal location: μ={mean:.2f}, σ={std:.1f}",
            xlabel="Value x",
            ylabel="Density",
            xlim=(-9, 9),
            ylim=(0, 0.43),
        )

    return _save_animation(
        figure, update, frame_count, output_path, fps, dpi
    )


def animate_normal_std(
    output_path: Path,
    frame_count: int,
    fps: int,
    dpi: int,
) -> Path:
    """Animate Normal spread and peak height as sigma changes."""
    standard_deviations = np.linspace(0.35, 3.0, frame_count)
    mean = 0.0
    x = np.linspace(-10, 10, 800)
    figure, axis = plt.subplots(figsize=(9, 5.5))

    def update(frame: int) -> None:
        std = float(standard_deviations[frame])
        density = stats.norm.pdf(x, mean, std)
        area = trapezoid(density, x)
        axis.clear()
        axis.plot(
            x,
            density,
            color=COLORS["theoretical"],
            linewidth=2.5,
        )
        axis.fill_between(x, density, color=COLORS["empirical"], alpha=0.24)
        axis.set(
            title=f"Normal scale: μ={mean:.1f}, σ={std:.2f}, area≈{area:.3f}",
            xlabel="Value x",
            ylabel="Density",
            xlim=(-10, 10),
            ylim=(0, 1.2),
        )

    return _save_animation(
        figure, update, frame_count, output_path, fps, dpi
    )


def animate_normal_to_lognormal(
    output_path: Path,
    frame_count: int,
    fps: int,
    dpi: int,
    seed: int,
) -> Path:
    """Animate the transformation Y=exp(X) from symmetry to positive skew."""
    rng = np.random.default_rng(seed)
    log_mean, log_std = 0.5, 0.65
    normal_samples = rng.normal(log_mean, log_std, 12_000)
    lognormal_samples = np.exp(normal_samples)
    progress = np.linspace(0.0, 1.0, frame_count)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    def update(frame: int) -> None:
        alpha = float(progress[frame])
        transformed = (
            (1 - alpha) * normal_samples + alpha * lognormal_samples
        )
        for axis in axes:
            axis.clear()
        axes[0].hist(
            normal_samples,
            bins=70,
            density=True,
            color=COLORS["empirical"],
            alpha=0.72,
        )
        axes[0].set(
            title="Log space: X is symmetric",
            xlabel="X",
            ylabel="Density",
            xlim=(-2, 3),
        )
        axes[1].hist(
            transformed,
            bins=80,
            density=True,
            color=COLORS["theoretical"],
            alpha=0.72,
        )
        axes[1].set(
            title=f"Transformation progress α={alpha:.2f}",
            xlabel="(1-α)X + α exp(X)",
            ylabel="Density",
            xlim=(-2 if alpha < 0.5 else 0, 8),
        )
        figure.suptitle(
            "Normal to Log-normal: exponentiation creates positive right skew\n"
            "Final frame: Y = exp(X)"
        )

    return _save_animation(
        figure, update, frame_count, output_path, fps, dpi
    )


def parse_args() -> argparse.Namespace:
    """Parse animation generation options."""
    parser = argparse.ArgumentParser(
        description="Generate the Day 8 animated GIF collection."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 12 frames per animation for a fast validation run.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=105)
    parser.add_argument(
        "--only",
        choices=[
            "bernoulli",
            "binomial",
            "binomial-normal",
            "binomial-poisson",
            "poisson",
            "exponential",
            "normal-mean",
            "normal-std",
            "normal-lognormal",
        ],
        help="Generate only one named animation.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate all requested animations or one selected animation."""
    args = parse_args()
    if args.fps <= 0 or args.dpi <= 0:
        raise ValueError("fps and dpi must be positive.")
    configure_matplotlib()
    ensure_output_directories(TOPIC_DIRECTORY / "outputs")
    frame_count = 12 if args.quick else 36

    jobs: Sequence[tuple[str, Callable[[], Path]]] = [
        (
            "bernoulli",
            lambda: animate_bernoulli(
                GIF_DIRECTORY / "bernoulli_probability.gif",
                frame_count,
                args.fps,
                args.dpi,
            ),
        ),
        (
            "binomial",
            lambda: animate_binomial_probability(
                GIF_DIRECTORY / "binomial_probability.gif",
                frame_count,
                args.fps,
                args.dpi,
            ),
        ),
        (
            "binomial-normal",
            lambda: animate_binomial_to_normal(
                GIF_DIRECTORY / "binomial_to_normal.gif",
                frame_count,
                args.fps,
                args.dpi,
            ),
        ),
        (
            "binomial-poisson",
            lambda: animate_binomial_to_poisson(
                GIF_DIRECTORY / "binomial_to_poisson.gif",
                frame_count,
                args.fps,
                args.dpi,
            ),
        ),
        (
            "poisson",
            lambda: animate_poisson_rate(
                GIF_DIRECTORY / "poisson_rate.gif",
                frame_count,
                args.fps,
                args.dpi,
            ),
        ),
        (
            "exponential",
            lambda: animate_exponential_rate(
                GIF_DIRECTORY / "exponential_rate.gif",
                frame_count,
                args.fps,
                args.dpi,
            ),
        ),
        (
            "normal-mean",
            lambda: animate_normal_mean(
                GIF_DIRECTORY / "normal_mean.gif",
                frame_count,
                args.fps,
                args.dpi,
            ),
        ),
        (
            "normal-std",
            lambda: animate_normal_std(
                GIF_DIRECTORY / "normal_standard_deviation.gif",
                frame_count,
                args.fps,
                args.dpi,
            ),
        ),
        (
            "normal-lognormal",
            lambda: animate_normal_to_lognormal(
                GIF_DIRECTORY / "normal_to_lognormal.gif",
                frame_count,
                args.fps,
                args.dpi,
                SEED,
            ),
        ),
    ]

    for name, job in jobs:
        if args.only is None or args.only == name:
            job()


if __name__ == "__main__":
    main()
