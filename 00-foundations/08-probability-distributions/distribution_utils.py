"""Shared numerical and visualization helpers for the Day 8 laboratory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from scipy import stats


SEED = 42
COLORS = {
    "empirical": "#4C78A8",
    "theoretical": "#F58518",
    "accent": "#54A24B",
    "warning": "#E45756",
    "neutral": "#7F7F7F",
}


@dataclass(frozen=True)
class EmpiricalStatistics:
    """Summary statistics used across static and interactive charts."""

    mean: float
    variance: float
    std: float
    skewness: float
    median: float
    minimum: float
    maximum: float
    p90: float
    p95: float
    p99: float


def validate_positive(value: float, name: str) -> None:
    """Raise a useful error when a parameter must be strictly positive."""
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero.")


def validate_probability(value: float, name: str = "p") -> None:
    """Raise a useful error when a parameter must be a probability."""
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1.")


def ensure_output_directories(base_directory: Path) -> dict[str, Path]:
    """Create and return the static, GIF, and HTML output directories."""
    directories = {
        "static": base_directory / "static",
        "gifs": base_directory / "gifs",
        "html": base_directory / "html",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def calculate_empirical_statistics(samples: np.ndarray) -> EmpiricalStatistics:
    """Calculate comparable empirical summaries for one-dimensional samples."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values.")

    return EmpiricalStatistics(
        mean=float(np.mean(values)),
        variance=float(np.var(values, ddof=0)),
        std=float(np.std(values, ddof=0)),
        skewness=float(stats.skew(values, bias=False)) if values.size > 2 else 0.0,
        median=float(np.median(values)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        p90=float(np.quantile(values, 0.90)),
        p95=float(np.quantile(values, 0.95)),
        p99=float(np.quantile(values, 0.99)),
    )


def total_absolute_probability_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return the L1 distance between two aligned discrete probability vectors."""
    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    if first_array.shape != second_array.shape:
        raise ValueError("probability vectors must have the same shape.")
    return float(np.sum(np.abs(first_array - second_array)))


def simulate_poisson_process(
    rate: float,
    duration: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate arrival and inter-arrival times for a homogeneous Poisson process."""
    validate_positive(rate, "rate")
    validate_positive(duration, "duration")

    arrival_times: list[float] = []
    inter_arrival_times: list[float] = []
    current_time = 0.0
    while current_time < duration:
        wait = float(rng.exponential(scale=1.0 / rate))
        current_time += wait
        if current_time <= duration:
            inter_arrival_times.append(wait)
            arrival_times.append(current_time)
    return np.asarray(arrival_times), np.asarray(inter_arrival_times)


def _surface_layout(
    title: str,
    x_title: str,
    y_title: str,
    z_title: str,
) -> dict[str, Any]:
    """Return a consistent Plotly layout for mathematical surfaces."""
    return {
        "title": title,
        "template": "plotly_white",
        "height": 650,
        "margin": {"l": 20, "r": 20, "t": 70, "b": 20},
        "scene": {
            "xaxis_title": x_title,
            "yaxis_title": y_title,
            "zaxis_title": z_title,
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.0}},
        },
    }


def create_normal_sigma_surface(mu: float = 0.0) -> go.Figure:
    """Create density(x, sigma) for a Normal distribution with fixed mean."""
    x = np.linspace(-8.0, 8.0, 180)
    sigma = np.linspace(0.35, 3.0, 90)
    x_grid, sigma_grid = np.meshgrid(x, sigma)
    density = stats.norm.pdf(x_grid, loc=mu, scale=sigma_grid)
    figure = go.Figure(
        go.Surface(
            x=x_grid,
            y=sigma_grid,
            z=density,
            colorscale="Viridis",
            colorbar={"title": "Density"},
        )
    )
    figure.update_layout(
        **_surface_layout(
            "Normal density as standard deviation changes",
            "Value x",
            "Standard deviation σ",
            "Probability density",
        )
    )
    return figure


def create_normal_mu_surface(sigma: float = 1.0) -> go.Figure:
    """Create density(x, mu) for a Normal distribution with fixed spread."""
    validate_positive(sigma, "sigma")
    x = np.linspace(-8.0, 8.0, 180)
    mu = np.linspace(-4.0, 4.0, 90)
    x_grid, mu_grid = np.meshgrid(x, mu)
    density = stats.norm.pdf(x_grid, loc=mu_grid, scale=sigma)
    figure = go.Figure(
        go.Surface(
            x=x_grid,
            y=mu_grid,
            z=density,
            colorscale="Cividis",
            colorbar={"title": "Density"},
        )
    )
    figure.update_layout(
        **_surface_layout(
            "Normal density as the mean changes",
            "Value x",
            "Mean μ",
            "Probability density",
        )
    )
    return figure


def create_lognormal_surface(log_mu: float = 0.0) -> go.Figure:
    """Create density(x, log_sigma) for a Log-normal distribution."""
    x = np.geomspace(0.03, 30.0, 190)
    log_sigma = np.linspace(0.15, 1.25, 90)
    x_grid, sigma_grid = np.meshgrid(x, log_sigma)
    density = stats.lognorm.pdf(x_grid, s=sigma_grid, scale=np.exp(log_mu))
    figure = go.Figure(
        go.Surface(
            x=x_grid,
            y=sigma_grid,
            z=density,
            colorscale="Plasma",
            colorbar={"title": "Density"},
        )
    )
    figure.update_layout(
        **_surface_layout(
            "Log-normal density as log-scale spread changes",
            "Positive value",
            "Log standard deviation",
            "Probability density",
        )
    )
    figure.update_scenes(xaxis_type="log")
    return figure


def create_binomial_probability_surface(n: int = 30) -> go.Figure:
    """Create P(X=k) over success count and probability for fixed n."""
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    k = np.arange(n + 1)
    probabilities = np.linspace(0.02, 0.98, 100)
    k_grid, p_grid = np.meshgrid(k, probabilities)
    mass = stats.binom.pmf(k_grid, n=n, p=p_grid)
    figure = go.Figure(
        go.Surface(
            x=k_grid,
            y=p_grid,
            z=mass,
            colorscale="Blues",
            colorbar={"title": "P(X=k)"},
        )
    )
    figure.update_layout(
        **_surface_layout(
            f"Binomial probability surface (n={n})",
            "Success count k",
            "Success probability p",
            "P(X = k)",
        )
    )
    return figure


def create_poisson_probability_surface() -> go.Figure:
    """Create P(X=k) over event count and Poisson rate."""
    k = np.arange(0, 41)
    rates = np.linspace(0.25, 20.0, 100)
    k_grid, rate_grid = np.meshgrid(k, rates)
    mass = stats.poisson.pmf(k_grid, mu=rate_grid)
    figure = go.Figure(
        go.Surface(
            x=k_grid,
            y=rate_grid,
            z=mass,
            colorscale="Turbo",
            colorbar={"title": "P(X=k)"},
        )
    )
    figure.update_layout(
        **_surface_layout(
            "Poisson probability surface",
            "Event count k",
            "Rate λ",
            "P(X = k)",
        )
    )
    return figure


def write_plotly_surfaces(output_directory: Path) -> list[Path]:
    """Write standalone interactive HTML files for all parameter surfaces."""
    output_directory.mkdir(parents=True, exist_ok=True)
    figures = {
        "normal_sigma_surface.html": create_normal_sigma_surface(),
        "normal_mu_surface.html": create_normal_mu_surface(),
        "lognormal_surface.html": create_lognormal_surface(),
        "binomial_probability_surface.html": create_binomial_probability_surface(),
        "poisson_probability_surface.html": create_poisson_probability_surface(),
    }
    output_paths: list[Path] = []
    for filename, figure in figures.items():
        output_path = output_directory / filename
        figure.write_html(
            output_path,
            include_plotlyjs="cdn",
            full_html=True,
            auto_open=False,
        )
        output_paths.append(output_path)
    return output_paths
