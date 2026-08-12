"""Shared simulation, validation, style, and output helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from example import PREMIUM, REGULAR, SyntheticPopulation, generate_synthetic_population
from from_scratch import effective_sample_size, estimator_statistics, weighted_mean


SEED = 42
TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = TOPIC_DIRECTORY / "outputs"
STATIC_DIRECTORY = OUTPUT_DIRECTORY / "static"
GIF_DIRECTORY = OUTPUT_DIRECTORY / "gifs"
INTERACTIVE_DIRECTORY = OUTPUT_DIRECTORY / "interactive"

COLORS = {
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "green": "#059669",
    "orange": "#EA580C",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "dark": "#0F172A",
    "light": "#E2E8F0",
    "yellow": "#CA8A04",
}


def ensure_output_directories() -> None:
    """Create every generated-output directory."""
    for directory in (STATIC_DIRECTORY, GIF_DIRECTORY, INTERACTIVE_DIRECTORY):
        directory.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    """Apply a restrained technical-portfolio style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 170,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "legend.fontsize": 9,
            "figure.titlesize": 16,
        }
    )


def save_figure(figure: plt.Figure, name: str) -> Path:
    """Save and close one static PNG."""
    ensure_output_directories()
    path = STATIC_DIRECTORY / name
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved: {path.relative_to(TOPIC_DIRECTORY)}")
    return path


def save_animation(
    figure: plt.Figure,
    update: Callable[[int], None],
    frame_count: int,
    name: str,
    *,
    fps: int = 6,
    dpi: int = 90,
) -> Path:
    """Save a bounded Pillow-backed GIF and close its figure."""
    ensure_output_directories()
    path = GIF_DIRECTORY / name
    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 / fps,
        repeat=True,
    )
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
    print(f"Saved: {path.relative_to(TOPIC_DIRECTORY)}")
    return path


def save_plotly(figure: object, name: str) -> Path:
    """Save a self-contained Plotly figure without requiring a server."""
    ensure_output_directories()
    path = INTERACTIVE_DIRECTORY / name
    figure.write_html(
        path,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )
    print(f"Saved: {path.relative_to(TOPIC_DIRECTORY)}")
    return path


def create_population(
    population_size: int = 50_000,
    *,
    premium_share: float = 0.10,
    seed: int = SEED,
) -> SyntheticPopulation:
    """Create the shared synthetic two-segment finite population."""
    return generate_synthetic_population(
        population_size=population_size,
        premium_share=premium_share,
        seed=seed,
    )


def _validate_sampling_configuration(sample_size: int, trials: int) -> None:
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer.")
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer.")
    if sample_size <= 0 or trials <= 0:
        raise ValueError("sample_size and trials must be positive.")


def repeated_sample_means(
    values: np.ndarray,
    *,
    sample_size: int,
    trials: int,
    rng: np.random.Generator,
    probabilities: np.ndarray | None = None,
    replace: bool = True,
    batch_size: int = 200,
) -> np.ndarray:
    """Return Monte Carlo sample means in bounded-memory batches."""
    _validate_sampling_configuration(sample_size, trials)
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite one-dimensional array.")
    if not replace and sample_size > values.size:
        raise ValueError("sample_size cannot exceed values size without replacement.")
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != values.shape:
            raise ValueError("probabilities must have the same shape as values.")
        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or probabilities.sum() <= 0.0
        ):
            raise ValueError("probabilities must be finite, non-negative, and nonzero.")
        probabilities = probabilities / probabilities.sum()

    means = np.empty(trials, dtype=float)
    for start in range(0, trials, batch_size):
        stop = min(start + batch_size, trials)
        indices = rng.choice(
            values.size,
            size=(stop - start, sample_size),
            replace=replace,
            p=probabilities,
        )
        means[start:stop] = values[indices].mean(axis=1)
    return means


def biased_selection_probabilities(
    population: SyntheticPopulation,
    *,
    premium_selection_weight: float = 8.0,
) -> np.ndarray:
    """Return normalized row probabilities for a visible selection mechanism."""
    premium_selection_weight = float(premium_selection_weight)
    if (
        not np.isfinite(premium_selection_weight)
        or premium_selection_weight <= 0.0
    ):
        raise ValueError("premium_selection_weight must be finite and positive.")
    weights = np.where(
        population.segments == PREMIUM,
        premium_selection_weight,
        1.0,
    )
    return weights / weights.sum()


def repeated_stratified_means(
    population: SyntheticPopulation,
    *,
    regular_sample_size: int,
    premium_sample_size: int,
    trials: int,
    rng: np.random.Generator,
    replace: bool = True,
) -> np.ndarray:
    """Return population-share-weighted estimates from both strata."""
    _validate_sampling_configuration(regular_sample_size, trials)
    _validate_sampling_configuration(premium_sample_size, trials)
    regular = population.spend[population.segments == REGULAR]
    premium = population.spend[population.segments == PREMIUM]
    regular_means = repeated_sample_means(
        regular,
        sample_size=regular_sample_size,
        trials=trials,
        rng=rng,
        replace=replace,
    )
    premium_means = repeated_sample_means(
        premium,
        sample_size=premium_sample_size,
        trials=trials,
        rng=rng,
        replace=replace,
    )
    regular_share = regular.size / population.size
    premium_share = premium.size / population.size
    return regular_share * regular_means + premium_share * premium_means


def density_histogram(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    bins: np.ndarray | int,
    color: str,
    label: str | None = None,
    alpha: float = 0.55,
) -> None:
    """Draw a consistent density histogram."""
    axis.hist(
        values,
        bins=bins,
        density=True,
        alpha=alpha,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        label=label,
    )


def calculate_estimator_statistics(
    estimates: Sequence[float],
    true_parameter: float,
) -> dict[str, float]:
    """Expose the tested Day 10 estimator summary to visual generators."""
    return estimator_statistics(estimates, true_parameter)


def validate_mse_identity(
    statistics: dict[str, float],
    *,
    tolerance: float = 1e-10,
) -> None:
    """Fail if an empirical summary violates its exact MSE convention."""
    if not np.isclose(
        statistics["mse"],
        statistics["variance_plus_bias_squared"],
        rtol=tolerance,
        atol=tolerance,
    ):
        raise RuntimeError("Empirical MSE does not equal variance plus squared bias.")


def validate_effective_sample_size(weights: Sequence[float]) -> float:
    """Return ESS after checking its positive-weight bounds."""
    array = np.asarray(weights, dtype=float)
    ess = effective_sample_size(array)
    positive_count = int(np.count_nonzero(array > 0.0))
    if ess > positive_count + 1e-10 or ess < 1.0 - 1e-10:
        raise RuntimeError("Effective sample size is outside valid weight bounds.")
    return ess


def population_weighted_estimate(
    segment_means: Sequence[float],
    population_shares: Sequence[float],
) -> float:
    """Return a checked population-share-weighted aggregate."""
    shares = np.asarray(population_shares, dtype=float)
    if not np.isclose(shares.sum(), 1.0):
        raise ValueError("population_shares must sum to one.")
    return weighted_mean(segment_means, shares)
