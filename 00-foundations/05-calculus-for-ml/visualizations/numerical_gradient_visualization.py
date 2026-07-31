"""Compare analytical derivatives with forward and centered finite differences."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .utils import (
        OutputPaths,
        RenderConfig,
        configure_plot_style,
        print_group_header,
        save_figure,
        topic_output_paths,
    )
except ImportError:
    from utils import (  # type: ignore[no-redef]
        OutputPaths,
        RenderConfig,
        configure_plot_style,
        print_group_header,
        save_figure,
        topic_output_paths,
    )


def forward_difference(x: np.ndarray | float, epsilon: float) -> np.ndarray:
    """Estimate d(sin(x))/dx using one forward perturbation."""
    x_array = np.asarray(x, dtype=float)
    return (np.sin(x_array + epsilon) - np.sin(x_array)) / epsilon


def centered_difference(x: np.ndarray | float, epsilon: float) -> np.ndarray:
    """Estimate d(sin(x))/dx using symmetric perturbations."""
    x_array = np.asarray(x, dtype=float)
    return (
        np.sin(x_array + epsilon) - np.sin(x_array - epsilon)
    ) / (2.0 * epsilon)


def validate_centered_difference() -> None:
    """Check a representative centered difference against cos(x)."""
    x0 = 1.0
    analytical = math.cos(x0)
    numerical = float(centered_difference(x0, 1e-5))
    if not math.isclose(analytical, numerical, rel_tol=1e-9, abs_tol=1e-10):
        raise RuntimeError(
            "Centered finite-difference validation failed for d(sin(x))/dx."
        )


def create_derivative_comparison(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> Path:
    """Compare derivative curves produced by three methods."""
    x = np.linspace(-2.0 * np.pi, 2.0 * np.pi, 700)
    epsilon = 0.25
    analytical = np.cos(x)
    forward = forward_difference(x, epsilon)
    centered = centered_difference(x, epsilon)

    figure, axis = plt.subplots(figsize=(11, 5.8))
    axis.plot(x, analytical, color="#222222", linewidth=3, label=r"$\cos(x)$")
    axis.plot(
        x,
        forward,
        color="#d95f02",
        linestyle="--",
        label=rf"Forward difference, $\epsilon={epsilon}$",
    )
    axis.plot(
        x,
        centered,
        color="#1f4e79",
        linestyle=":",
        linewidth=3,
        label=rf"Centered difference, $\epsilon={epsilon}$",
    )
    axis.set(
        title=r"Numerical and Analytical Derivatives of $f(x)=\sin(x)$",
        xlabel=r"$x$",
        ylabel="Derivative estimate",
        ylim=(-1.25, 1.25),
    )
    axis.legend(ncol=3)
    axis.text(
        0.03,
        0.06,
        "Centered differences cancel more first-order error\n"
        "than forward differences at the same epsilon.",
        transform=axis.transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    return save_figure(
        figure,
        output_paths.static / "numerical_vs_analytical_derivative.png",
        config,
    )


def create_error_figure(
    output_paths: OutputPaths,
    config: RenderConfig,
    x0: float = 1.0,
) -> Path:
    """Plot truncation and floating-point cancellation across epsilon."""
    epsilon_values = np.logspace(-16, -1, 150)
    analytical = math.cos(x0)
    forward_errors = np.abs(
        forward_difference(x0, epsilon_values) - analytical
    )
    centered_errors = np.abs(
        centered_difference(x0, epsilon_values) - analytical
    )
    positive_centered = np.maximum(centered_errors, np.finfo(float).tiny)
    best_index = int(np.argmin(positive_centered))

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.loglog(
        epsilon_values,
        forward_errors,
        color="#d95f02",
        label="Forward-difference error",
    )
    axis.loglog(
        epsilon_values,
        positive_centered,
        color="#1f4e79",
        label="Centered-difference error",
    )
    axis.scatter(
        [epsilon_values[best_index]],
        [positive_centered[best_index]],
        color="#1b9e77",
        zorder=5,
        label="Best centered epsilon in this simulation",
    )
    axis.annotate(
        "Too small:\nfloating-point cancellation",
        xy=(2e-15, float(centered_errors[12])),
        xytext=(2e-13, 2e-4),
        arrowprops={"arrowstyle": "->"},
    )
    axis.annotate(
        "Too large:\ntruncation error",
        xy=(5e-2, float(centered_errors[-5])),
        xytext=(2e-5, 2e-3),
        arrowprops={"arrowstyle": "->"},
    )
    axis.set(
        title=(
            "Finite-Difference Error Has Competing Numerical Effects "
            f"(x₀={x0})"
        ),
        xlabel=r"Perturbation $\epsilon$",
        ylabel="Absolute derivative error",
    )
    axis.legend()
    return save_figure(
        figure,
        output_paths.static / "finite_difference_error.png",
        config,
    )


def print_error_table(x0: float = 1.0) -> None:
    """Print a compact centered-difference table for selected epsilons."""
    analytical = math.cos(x0)
    print("\n  Centered finite-difference check at x=1.0")
    print("  epsilon       analytical      numerical       absolute error")
    for epsilon in (1e-1, 1e-3, 1e-5, 1e-8, 1e-12):
        numerical = float(centered_difference(x0, epsilon))
        absolute_error = abs(numerical - analytical)
        print(
            f"  {epsilon:10.1e}  {analytical:14.10f}  "
            f"{numerical:14.10f}  {absolute_error:14.3e}"
        )


def generate(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> list[Path]:
    """Generate numerical-differentiation figures and console diagnostics."""
    print_group_header("Numerical versus analytical differentiation")
    configure_plot_style()
    validate_centered_difference()
    print_error_table()
    return [
        create_derivative_comparison(output_paths, config),
        create_error_figure(output_paths, config),
    ]


def main() -> None:
    """Generate numerical-differentiation artifacts with default settings."""
    generate(topic_output_paths(), RenderConfig())


if __name__ == "__main__":
    main()
