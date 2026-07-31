"""Compare sigmoid, tanh, and ReLU with their analytical derivatives."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

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


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Return sigmoid(x) over the visualization's numerically safe range."""
    x_array = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x_array))


def sigmoid_derivative(x: np.ndarray | float) -> np.ndarray:
    """Return sigmoid(x) × (1 − sigmoid(x))."""
    activation = sigmoid(x)
    return activation * (1.0 - activation)


def tanh_derivative(x: np.ndarray | float) -> np.ndarray:
    """Return 1 − tanh²(x)."""
    activation = np.tanh(np.asarray(x, dtype=float))
    return 1.0 - activation**2


def relu(x: np.ndarray | float) -> np.ndarray:
    """Return max(0, x)."""
    return np.maximum(0.0, np.asarray(x, dtype=float))


def relu_derivative(x: np.ndarray | float) -> np.ndarray:
    """Use zero at x=0 as a practical convention for ReLU's derivative."""
    return (np.asarray(x, dtype=float) > 0.0).astype(float)


def _centered_difference(
    function: Callable[[float], float],
    x: float,
    epsilon: float = 1e-6,
) -> float:
    """Estimate a scalar derivative for independent validation."""
    return (function(x + epsilon) - function(x - epsilon)) / (2.0 * epsilon)


def validate_activation_derivatives() -> None:
    """Check sigmoid and tanh derivatives at a non-special point."""
    x0 = 0.7
    sigmoid_numeric = _centered_difference(
        lambda value: float(sigmoid(value)),
        x0,
    )
    tanh_numeric = _centered_difference(math.tanh, x0)
    if not math.isclose(
        float(sigmoid_derivative(x0)),
        sigmoid_numeric,
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Finite-difference check failed for sigmoid.")
    if not math.isclose(
        float(tanh_derivative(x0)),
        tanh_numeric,
        rel_tol=1e-7,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Finite-difference check failed for tanh.")


def create_activation_figure(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> Path:
    """Plot each activation beside its derivative and important regions."""
    x = np.linspace(-7.0, 7.0, 700)
    activations = [
        (
            "Sigmoid",
            sigmoid(x),
            sigmoid_derivative(x),
            "#1f4e79",
        ),
        (
            "tanh",
            np.tanh(x),
            tanh_derivative(x),
            "#1b9e77",
        ),
        (
            "ReLU",
            relu(x),
            relu_derivative(x),
            "#d95f02",
        ),
    ]

    figure, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for row, (name, activation, derivative, color) in enumerate(activations):
        axes[row, 0].plot(x, activation, color=color)
        axes[row, 0].axvline(0.0, color="#777777", linestyle=":", linewidth=1)
        axes[row, 0].set(
            title=f"{name} activation",
            ylabel="Activation value",
        )
        axes[row, 1].plot(x, derivative, color=color)
        axes[row, 1].axvline(0.0, color="#777777", linestyle=":", linewidth=1)
        axes[row, 1].set(
            title=f"{name} derivative",
            ylabel="Local derivative",
        )

    for axis in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.axvspan(-7.0, -3.0, color="#aaaaaa", alpha=0.12)
        axis.axvspan(3.0, 7.0, color="#aaaaaa", alpha=0.12)
    axes[0, 1].scatter([0.0], [0.25], color="#222222", zorder=5)
    axes[0, 1].annotate(
        "maximum derivative = 0.25",
        xy=(0.0, 0.25),
        xytext=(1.1, 0.18),
        arrowprops={"arrowstyle": "->"},
    )
    axes[1, 1].annotate(
        "tanh also saturates\nfor large |z|",
        xy=(4.0, float(tanh_derivative(4.0))),
        xytext=(1.0, 0.35),
        arrowprops={"arrowstyle": "->"},
    )
    axes[2, 0].axvspan(-7.0, 0.0, color="#d95f02", alpha=0.09)
    axes[2, 1].axvspan(-7.0, 0.0, color="#d95f02", alpha=0.09)
    axes[2, 1].scatter([0.0], [0.0], facecolors="white", edgecolors="#222222")
    axes[2, 1].annotate(
        "undefined mathematically at 0;\nthis code uses the convention 0",
        xy=(0.0, 0.0),
        xytext=(1.0, 0.35),
        arrowprops={"arrowstyle": "->"},
    )
    axes[2, 0].text(
        -5.8,
        3.7,
        "negative inputs map to zero",
        color="#9c3d00",
    )
    for axis in axes[-1, :]:
        axis.set_xlabel("Pre-activation z")
    figure.suptitle(
        "Activation Functions and Their Local Derivatives",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Saturation can shrink gradients, but initialization, architecture, "
        "normalization, and repeated Jacobians also affect stability.",
        ha="center",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.95))
    return save_figure(
        figure,
        output_paths.static / "activation_functions_and_derivatives.png",
        config,
    )


def generate(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> list[Path]:
    """Generate activation and derivative comparisons."""
    print_group_header("Activation functions and derivatives")
    configure_plot_style()
    validate_activation_derivatives()
    return [create_activation_figure(output_paths, config)]


def main() -> None:
    """Generate activation artifacts with default settings."""
    generate(topic_output_paths(), RenderConfig())


if __name__ == "__main__":
    main()
