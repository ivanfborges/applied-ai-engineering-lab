"""Visualize derivatives as tangent slopes and local linear approximations."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

try:
    from .utils import (
        OutputPaths,
        RenderConfig,
        configure_plot_style,
        print_group_header,
        save_animation,
        save_figure,
        topic_output_paths,
    )
except ImportError:
    from utils import (  # type: ignore[no-redef]
        OutputPaths,
        RenderConfig,
        configure_plot_style,
        print_group_header,
        save_animation,
        save_figure,
        topic_output_paths,
    )


def quadratic(x: np.ndarray | float) -> np.ndarray | float:
    """Return f(x) = x²."""
    return np.asarray(x) ** 2 if isinstance(x, np.ndarray) else x**2


def quadratic_derivative(x: np.ndarray | float) -> np.ndarray | float:
    """Return the analytical derivative f'(x) = 2x."""
    return 2.0 * x


def tangent_values(x: np.ndarray, x0: float) -> np.ndarray:
    """Evaluate the tangent line to f(x)=x² at x0."""
    return float(quadratic(x0)) + float(quadratic_derivative(x0)) * (x - x0)


def validate_quadratic_derivative(epsilon: float = 1e-6) -> None:
    """Check d(x²)/dx = 2x with a centered finite difference."""
    x0 = 1.4
    numerical = (
        float(quadratic(x0 + epsilon)) - float(quadratic(x0 - epsilon))
    ) / (2.0 * epsilon)
    if not math.isclose(
        float(quadratic_derivative(x0)),
        numerical,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Finite-difference check failed for d(x²)/dx.")


def create_local_sensitivity_figure(
    output_paths: OutputPaths,
    config: RenderConfig,
    x0: float = 1.4,
    delta_x: float = 0.35,
) -> Path:
    """Plot a tangent and compare a true change with its linear estimate."""
    x = np.linspace(-3.0, 3.0, 500)
    local_x = np.linspace(x0 - 0.65, x0 + 0.65, 100)
    true_next = float(quadratic(x0 + delta_x))
    linear_next = float(quadratic(x0)) + float(
        quadratic_derivative(x0)
    ) * delta_x

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(x, quadratic(x), color="#1f4e79", label=r"$f(x)=x^2$")
    axis.plot(
        local_x,
        tangent_values(local_x, x0),
        color="#d95f02",
        label=rf"Tangent: slope $f'({x0:.1f})={2*x0:.1f}$",
    )
    axis.axvspan(
        x0 - 0.65,
        x0 + 0.65,
        color="#f6c85f",
        alpha=0.18,
        label="Local interval",
    )
    axis.scatter(
        [x0, x0 + delta_x, x0 + delta_x],
        [quadratic(x0), true_next, linear_next],
        color=["#222222", "#1b9e77", "#d95f02"],
        zorder=5,
    )
    axis.plot(
        [x0 + delta_x, x0 + delta_x],
        [linear_next, true_next],
        color="#555555",
        linestyle=":",
    )
    axis.annotate(
        "Selected point",
        xy=(x0, quadratic(x0)),
        xytext=(x0 - 1.25, quadratic(x0) + 2.3),
        arrowprops={"arrowstyle": "->"},
    )
    axis.annotate(
        "True value",
        xy=(x0 + delta_x, true_next),
        xytext=(x0 + 0.75, true_next + 1.0),
        arrowprops={"arrowstyle": "->", "color": "#1b9e77"},
    )
    axis.annotate(
        "First-order approximation",
        xy=(x0 + delta_x, linear_next),
        xytext=(x0 + 0.65, linear_next - 1.5),
        arrowprops={"arrowstyle": "->", "color": "#d95f02"},
    )
    axis.text(
        0.03,
        0.96,
        r"$f(x_0+\Delta x)\approx f(x_0)+f'(x_0)\Delta x$"
        f"\ntrue={true_next:.3f}, linear={linear_next:.3f}",
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    axis.set(
        title="A Derivative Is a Local Slope",
        xlabel=r"Input $x$",
        ylabel=r"Output $f(x)$",
        xlim=(-3.0, 3.0),
        ylim=(-2.0, 9.5),
    )
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, 0.78))
    return save_figure(
        figure,
        output_paths.static / "derivative_local_sensitivity.png",
        config,
    )


def create_secant_comparison(
    output_paths: OutputPaths,
    config: RenderConfig,
    x0: float = 1.0,
) -> Path:
    """Show secant slopes approaching the tangent as h becomes small."""
    x = np.linspace(-0.5, 2.6, 400)
    h_values = [1.2, 0.65, 0.25, 0.08]
    colors = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(x, quadratic(x), color="#222222", label=r"$f(x)=x^2$")
    for h, color in zip(h_values, colors):
        secant_slope = (
            float(quadratic(x0 + h)) - float(quadratic(x0))
        ) / h
        secant = float(quadratic(x0)) + secant_slope * (x - x0)
        axis.plot(
            x,
            secant,
            color=color,
            alpha=0.9,
            label=rf"$h={h:.2f}$, slope={secant_slope:.2f}",
        )

    axis.plot(
        x,
        tangent_values(x, x0),
        color="#d95f02",
        linestyle="--",
        linewidth=3,
        label=r"Tangent slope $f'(1)=2$",
    )
    axis.scatter([x0], [quadratic(x0)], color="#222222", zorder=5)
    axis.set(
        title="A Secant Becomes the Tangent as h Approaches Zero",
        xlabel=r"Input $x$",
        ylabel=r"Output $f(x)$",
        xlim=(-0.5, 2.6),
        ylim=(-1.0, 7.0),
    )
    axis.legend(ncol=2)
    axis.text(
        0.03,
        0.05,
        r"$f'(x_0)=\lim_{h\to0}\frac{f(x_0+h)-f(x_0)}{h}$",
        transform=axis.transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    return save_figure(
        figure,
        output_paths.static / "secant_to_tangent.png",
        config,
    )


def create_tangent_animation(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> Path:
    """Animate a selected point and its tangent across the quadratic."""
    x = np.linspace(-3.0, 3.0, 500)
    x0_values = np.linspace(-2.4, 2.4, config.frames)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(x, quadratic(x), color="#1f4e79", label=r"$f(x)=x^2$")
    tangent_line, = axis.plot([], [], color="#d95f02", linewidth=2.5)
    selected_point, = axis.plot([], [], "o", color="#222222", markersize=7)
    slope_arrow = axis.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "->", "color": "#d95f02", "lw": 2},
    )
    explanation = axis.text(
        0.03,
        0.95,
        "",
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    axis.set(
        title="Moving Tangent: the Derivative Changes with Location",
        xlabel=r"Input $x$",
        ylabel=r"Output $f(x)$",
        xlim=(-3.0, 3.0),
        ylim=(-1.0, 9.5),
    )
    axis.legend(loc="upper center")

    def update(frame_index: int) -> tuple[object, ...]:
        x0 = float(x0_values[frame_index])
        slope = float(quadratic_derivative(x0))
        local_x = np.linspace(x0 - 0.8, x0 + 0.8, 80)
        tangent_line.set_data(local_x, tangent_values(local_x, x0))
        selected_point.set_data([x0], [quadratic(x0)])
        arrow_dx = 0.45
        slope_arrow.xy = (
            x0 + arrow_dx,
            float(quadratic(x0)) + slope * arrow_dx,
        )
        slope_arrow.set_position((x0, float(quadratic(x0))))
        explanation.set_text(
            rf"$x_0={x0:.2f}$"
            "\n"
            rf"$f'(x_0)=2x_0={slope:.2f}$"
            "\nThe arrow follows the local slope."
        )
        return tangent_line, selected_point, slope_arrow, explanation

    animation = FuncAnimation(
        figure,
        update,
        frames=config.frames,
        interval=1000 / config.fps,
        blit=False,
    )
    return save_animation(
        animation,
        figure,
        output_paths.animations / "derivative_tangent.gif",
        config,
    )


def generate(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> list[Path]:
    """Generate every derivative artifact."""
    print_group_header("Derivative as local sensitivity")
    configure_plot_style()
    validate_quadratic_derivative()
    artifacts = [
        create_local_sensitivity_figure(output_paths, config),
        create_secant_comparison(output_paths, config),
    ]
    if config.generate_gifs:
        artifacts.append(create_tangent_animation(output_paths, config))
    return artifacts


def main() -> None:
    """Generate derivative artifacts with default settings."""
    generate(topic_output_paths(), RenderConfig())


if __name__ == "__main__":
    main()
