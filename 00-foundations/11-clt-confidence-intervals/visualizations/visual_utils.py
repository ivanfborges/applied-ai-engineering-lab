"""Shared numerical and rendering helpers for the Day 11 visual lab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
ASSET_DIRECTORY = TOPIC_DIRECTORY / "assets"

INK = "#17212B"
BLUE = "#2F6B9A"
ORANGE = "#D97732"
GREEN = "#2D7D66"
PURPLE = "#7556A4"
RED = "#B44747"
GRAY = "#69747C"
LIGHT_BLUE = "#DCEAF4"
LIGHT_ORANGE = "#F7E5D7"
LIGHT_GRAY = "#EEF1F3"


@dataclass(frozen=True)
class AssetResult:
    """One generated asset and the empirical checks reported for it."""

    label: str
    path: Path
    metrics: tuple[str, ...] = ()


def configure_matplotlib() -> None:
    """Apply one neutral, readable style to every Matplotlib asset."""
    matplotlib.use("Agg", force=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#A7AFB5",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "grid.color": "#D7DDE1",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def ensure_asset_directory() -> Path:
    """Create and return the topic-local generated-asset directory."""
    ASSET_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return ASSET_DIRECTORY


def save_figure(fig: plt.Figure, filename: str, *, quick: bool) -> Path:
    """Save a bounded PNG and close its Matplotlib figure."""
    output = ensure_asset_directory() / filename
    fig.savefig(output, dpi=110 if quick else 160, bbox_inches="tight")
    plt.close(fig)
    return output


def positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def exponential_sample_means(
    sample_size: int,
    simulations: int,
    *,
    seed: int,
    scale: float = 2.0,
    batch_size: int = 2_000,
) -> np.ndarray:
    """Draw exponential sample means in batches to bound memory use."""
    positive_integer(sample_size, name="sample_size")
    positive_integer(simulations, name="simulations")
    positive_integer(batch_size, name="batch_size")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive.")

    rng = np.random.default_rng(seed)
    means = np.empty(simulations, dtype=float)
    for start in range(0, simulations, batch_size):
        stop = min(start + batch_size, simulations)
        samples = rng.exponential(scale=scale, size=(stop - start, sample_size))
        means[start:stop] = np.mean(samples, axis=1)
    return means


def distribution_moments(name: str) -> tuple[float, float]:
    """Return the exact mean and standard deviation of a configured source."""
    if name in {"Normal", "Uniform"}:
        return 0.0, 1.0
    if name == "Exponential":
        return 1.0, 1.0
    if name == "Lognormal":
        shape = 0.9
        mean = np.exp(shape**2 / 2.0)
        variance = (np.exp(shape**2) - 1.0) * np.exp(shape**2)
        return float(mean), float(np.sqrt(variance))
    raise ValueError(f"Unsupported distribution: {name}")


def draw_source_distribution(
    name: str,
    rng: np.random.Generator,
    size: tuple[int, int],
) -> np.ndarray:
    """Draw from one source distribution used by the CLT comparison."""
    if name == "Normal":
        return rng.normal(size=size)
    if name == "Uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=size)
    if name == "Exponential":
        return rng.exponential(size=size)
    if name == "Lognormal":
        return rng.lognormal(mean=0.0, sigma=0.9, size=size)
    raise ValueError(f"Unsupported distribution: {name}")


def standardized_sample_means(
    distribution: str,
    sample_size: int,
    simulations: int,
    *,
    seed: int,
    batch_size: int = 2_000,
) -> np.ndarray:
    """Return sample means standardized by exact CLT location and scale."""
    positive_integer(sample_size, name="sample_size")
    positive_integer(simulations, name="simulations")
    mean, standard_deviation = distribution_moments(distribution)
    rng = np.random.default_rng(seed)
    standardized = np.empty(simulations, dtype=float)
    for start in range(0, simulations, batch_size):
        stop = min(start + batch_size, simulations)
        samples = draw_source_distribution(
            distribution, rng, (stop - start, sample_size)
        )
        sample_means = np.mean(samples, axis=1)
        standardized[start:stop] = (
            np.sqrt(sample_size) * (sample_means - mean) / standard_deviation
        )
    return standardized


def simulate_normal_t_intervals(
    *,
    sample_size: int,
    intervals: int,
    confidence: float = 0.95,
    population_mean: float = 0.0,
    population_sd: float = 1.0,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Student-t intervals under exact normal-data assumptions."""
    positive_integer(sample_size, name="sample_size")
    if sample_size < 2:
        raise ValueError("sample_size must be at least two.")
    positive_integer(intervals, name="intervals")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one.")
    if not np.isfinite(population_sd) or population_sd <= 0.0:
        raise ValueError("population_sd must be finite and positive.")

    rng = np.random.default_rng(seed)
    samples = rng.normal(
        loc=population_mean,
        scale=population_sd,
        size=(intervals, sample_size),
    )
    means = np.mean(samples, axis=1)
    standard_errors = np.std(samples, axis=1, ddof=1) / np.sqrt(sample_size)
    critical = float(t.ppf(0.5 + confidence / 2.0, df=sample_size - 1))
    lower = means - critical * standard_errors
    upper = means + critical * standard_errors
    covered = (lower <= population_mean) & (population_mean <= upper)
    return lower, means, upper, covered
