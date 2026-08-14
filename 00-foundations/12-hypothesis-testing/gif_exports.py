"""Generate bounded GIF exports for the Day 12 hypothesis-testing lab."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats

from statistical_utils import normal_error_rates, simulate_t_experiments


TOPIC_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = TOPIC_DIRECTORY / "outputs"
SAMPLE_SIZES = (10, 20, 50, 100, 250, 500)

BLUE = "#2563EB"
RED = "#DC2626"
ORANGE = "#F59E0B"
GREEN = "#16A34A"
PURPLE = "#7C3AED"


@dataclass(frozen=True)
class GifResult:
    """Path and observed metrics returned by a completed GIF generator."""

    path: Path
    label: str
    metrics: tuple[str, ...]


def _figure_to_image(figure: plt.Figure) -> Image.Image:
    """Render a Matplotlib figure into an in-memory Pillow image."""
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba())
    image = Image.fromarray(rgba).convert("RGB")
    plt.close(figure)
    return image


def _save_gif(frames: list[Image.Image], path: Path, *, duration_ms: int) -> None:
    """Save non-empty frames as a looping GIF."""
    if not frames:
        raise ValueError("frames must not be empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"GIF was not created correctly: {path}")


def generate_sample_size_uncertainty(*, quick: bool = False) -> GifResult:
    """Animate the normal sampling distribution narrowing as n increases."""
    sample_sizes = SAMPLE_SIZES[::2] if quick else SAMPLE_SIZES
    x_values = np.linspace(-1.1, 1.1, 800)
    maximum_density = stats.norm.pdf(0.0, scale=1.0 / math.sqrt(max(sample_sizes)))
    frames: list[Image.Image] = []

    for sample_size in sample_sizes:
        standard_error = 1.0 / math.sqrt(sample_size)
        density = stats.norm.pdf(x_values, scale=standard_error)
        figure, axis = plt.subplots(figsize=(8, 4.8), dpi=90 if quick else 110)
        axis.plot(x_values, density, color=BLUE, linewidth=3)
        axis.fill_between(x_values, density, color=BLUE, alpha=0.25)
        axis.axvline(0.0, color=PURPLE, linestyle="--", linewidth=2)
        axis.set(
            xlim=(-1.1, 1.1),
            ylim=(0.0, maximum_density * 1.08),
            xlabel="Sample mean under H₀",
            ylabel="Probability density",
            title=f"Sampling uncertainty shrinks with n · n={sample_size}",
        )
        axis.text(
            0.98,
            0.92,
            f"SE = 1/√n = {standard_error:.4f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="#CBD5E1", alpha=0.9),
        )
        axis.grid(alpha=0.18)
        figure.tight_layout()
        frames.append(_figure_to_image(figure))

    path = OUTPUT_DIRECTORY / "sample_size_uncertainty.gif"
    _save_gif(frames, path, duration_ms=900)
    return GifResult(
        path=path,
        label="Sample size and shrinking uncertainty",
        metrics=(
            f"Frames: {len(frames)}",
            f"SE at n={sample_sizes[0]}: {1 / math.sqrt(sample_sizes[0]):.4f}",
            f"SE at n={sample_sizes[-1]}: {1 / math.sqrt(sample_sizes[-1]):.4f}",
        ),
    )


def generate_statistical_power(*, quick: bool = False) -> GifResult:
    """Animate beta shrinking and power growing with sample size."""
    sample_sizes = SAMPLE_SIZES[::2] if quick else SAMPLE_SIZES
    effect = 0.35
    standard_deviation = 1.0
    alpha = 0.05
    x_values = np.linspace(-1.0, 1.2, 900)
    maximum_density = stats.norm.pdf(
        0.0, scale=standard_deviation / math.sqrt(max(sample_sizes))
    )
    frames: list[Image.Image] = []
    final_power = 0.0

    for sample_size in sample_sizes:
        result = normal_error_rates(effect, sample_size, standard_deviation, alpha)
        final_power = result.power
        h0_density = stats.norm.pdf(x_values, loc=0.0, scale=result.standard_error)
        h1_density = stats.norm.pdf(x_values, loc=effect, scale=result.standard_error)
        rejection = x_values >= result.critical_mean
        figure, axis = plt.subplots(figsize=(8, 4.8), dpi=90 if quick else 110)
        axis.plot(x_values, h0_density, color=BLUE, linewidth=2.5, label="H₀")
        axis.plot(x_values, h1_density, color=PURPLE, linewidth=2.5, label="H₁")
        axis.fill_between(
            x_values,
            h1_density,
            where=~rejection,
            color=ORANGE,
            alpha=0.42,
            label="β",
        )
        axis.fill_between(
            x_values,
            h1_density,
            where=rejection,
            color=GREEN,
            alpha=0.36,
            label="power",
        )
        axis.axvline(
            result.critical_mean,
            color=RED,
            linestyle="--",
            linewidth=2,
            label="critical boundary",
        )
        axis.set(
            xlim=(-1.0, 1.2),
            ylim=(0.0, maximum_density * 1.08),
            xlabel="Sample mean",
            ylabel="Sampling density",
            title=f"Power increases as uncertainty shrinks · n={sample_size}",
        )
        axis.text(
            0.98,
            0.92,
            f"β = {result.beta:.3f}\npower = {result.power:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="#CBD5E1", alpha=0.9),
        )
        axis.legend(loc="upper left", ncols=2)
        axis.grid(alpha=0.18)
        figure.tight_layout()
        frames.append(_figure_to_image(figure))

    path = OUTPUT_DIRECTORY / "statistical_power.gif"
    _save_gif(frames, path, duration_ms=900)
    return GifResult(
        path=path,
        label="Increasing statistical power",
        metrics=(
            f"Frames: {len(frames)}",
            f"Configured effect: {effect:.2f} SD units",
            f"Power at final n={sample_sizes[-1]}: {final_power:.4f}",
        ),
    )


def generate_false_positive_simulation(*, quick: bool = False) -> GifResult:
    """Animate cumulative false positives across repeated true-null tests."""
    experiments = 120 if quick else 250
    data = simulate_t_experiments(
        true_mean=0.0,
        sample_size=30,
        standard_deviation=1.0,
        simulations=experiments,
        alpha=0.05,
        alternative="two-sided",
        seed=42,
    )
    endpoints = np.unique(
        np.linspace(10, experiments, 6 if quick else 10, dtype=int)
    )
    frames: list[Image.Image] = []

    for endpoint in endpoints:
        visible = data.iloc[:endpoint]
        false_positives = int(visible["reject"].sum())
        empirical_rate = false_positives / endpoint
        colors = np.where(visible["reject"], RED, BLUE)
        figure, axis = plt.subplots(figsize=(8, 4.8), dpi=90 if quick else 110)
        axis.scatter(
            visible["experiment"],
            visible["p_value"],
            c=colors,
            s=24,
            alpha=0.8,
        )
        axis.axhspan(0.0, 0.05, color=RED, alpha=0.12, label="p < α")
        axis.axhline(0.05, color=RED, linestyle="--", linewidth=2, label="α = 0.05")
        axis.set(
            xlim=(0, experiments + 5),
            ylim=(0.0, 1.0),
            xlabel="Experiment sequence",
            ylabel="p-value",
            title="Repeated experiments under H₀",
        )
        axis.text(
            0.98,
            0.92,
            (
                f"experiments = {endpoint}\n"
                f"false positives = {false_positives}\n"
                f"empirical rate = {empirical_rate:.2%}"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="#CBD5E1", alpha=0.9),
        )
        axis.legend(loc="upper left")
        axis.grid(alpha=0.18)
        figure.tight_layout()
        frames.append(_figure_to_image(figure))

    path = OUTPUT_DIRECTORY / "false_positive_simulation.gif"
    _save_gif(frames, path, duration_ms=750)
    false_positives = int(data["reject"].sum())
    return GifResult(
        path=path,
        label="Repeated experiments and false positives",
        metrics=(
            f"Frames: {len(frames)}",
            f"False positives: {false_positives} of {experiments}",
            f"Empirical false-positive rate: {false_positives / experiments:.4f}",
        ),
    )


GENERATORS = (
    generate_sample_size_uncertainty,
    generate_statistical_power,
    generate_false_positive_simulation,
)


def parse_arguments() -> argparse.Namespace:
    """Parse the optional reduced-cost rendering mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Render fewer frames at lower resolution for a smoke check.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate all GIFs and report only metrics from completed runs."""
    args = parse_arguments()
    print("Generating Day 12 hypothesis-testing GIFs...")
    if args.quick:
        print("Quick mode: reduced frames and resolution.")
    results = [generator(quick=args.quick) for generator in GENERATORS]
    for result in results:
        relative_path = result.path.relative_to(TOPIC_DIRECTORY).as_posix()
        print(f"[OK] {result.label}: {relative_path}")
        for metric in result.metrics:
            print(f"     {metric}")
    print("All outputs remain ignored until explicit author curation.")


if __name__ == "__main__":
    main()
