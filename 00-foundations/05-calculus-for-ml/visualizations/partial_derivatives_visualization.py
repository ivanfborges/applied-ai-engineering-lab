"""Visualize partial derivatives as one-variable-at-a-time surface slices."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

try:
    from .utils import (
        OutputPaths,
        RenderConfig,
        configure_plot_style,
        print_group_header,
        save_figure,
        save_plotly_html,
        topic_output_paths,
    )
except ImportError:
    from utils import (  # type: ignore[no-redef]
        OutputPaths,
        RenderConfig,
        configure_plot_style,
        print_group_header,
        save_figure,
        save_plotly_html,
        topic_output_paths,
    )


def surface_function(
    x: np.ndarray | float,
    y: np.ndarray | float,
) -> np.ndarray | float:
    """Return f(x, y) = x² + 2y²."""
    return np.asarray(x) ** 2 + 2.0 * np.asarray(y) ** 2


def partial_x(x: float) -> float:
    """Return ∂f/∂x = 2x."""
    return 2.0 * x


def partial_y(y: float) -> float:
    """Return ∂f/∂y = 4y."""
    return 4.0 * y


def create_static_slices(
    output_paths: OutputPaths,
    config: RenderConfig,
    x0: float = 1.0,
    y0: float = -0.75,
) -> Path:
    """Create a surface and two slices that hold one variable fixed."""
    x = np.linspace(-2.2, 2.2, 120)
    y = np.linspace(-2.0, 2.0, 120)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = surface_function(x_grid, y_grid)

    figure = plt.figure(figsize=(15, 5))
    surface_axis = figure.add_subplot(1, 3, 1, projection="3d")
    x_slice_axis = figure.add_subplot(1, 3, 2)
    y_slice_axis = figure.add_subplot(1, 3, 3)

    surface_axis.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap="Blues",
        alpha=0.78,
        linewidth=0,
    )
    z0 = float(surface_function(x0, y0))
    surface_axis.scatter([x0], [y0], [z0], color="#d95f02", s=55)
    surface_axis.set(
        title=r"Surface $f(x,y)=x^2+2y^2$",
        xlabel=r"$x$",
        ylabel=r"$y$",
        zlabel=r"$f(x,y)$",
    )
    surface_axis.view_init(elev=26, azim=-58)

    x_slice = surface_function(x, y0)
    x_tangent = z0 + partial_x(x0) * (x - x0)
    x_slice_axis.plot(x, x_slice, color="#1f4e79", label=rf"$y={y0}$ fixed")
    x_slice_axis.plot(
        x,
        x_tangent,
        color="#d95f02",
        linestyle="--",
        label=rf"$\partial f/\partial x={partial_x(x0):.1f}$",
    )
    x_slice_axis.scatter([x0], [z0], color="#222222", zorder=5)
    x_slice_axis.set(
        title="Change x while holding y fixed",
        xlabel=r"$x$",
        ylabel=r"$f(x,y_0)$",
        ylim=(-1.0, 8.0),
    )
    x_slice_axis.legend()

    y_slice = surface_function(x0, y)
    y_tangent = z0 + partial_y(y0) * (y - y0)
    y_slice_axis.plot(y, y_slice, color="#1b9e77", label=rf"$x={x0}$ fixed")
    y_slice_axis.plot(
        y,
        y_tangent,
        color="#d95f02",
        linestyle="--",
        label=rf"$\partial f/\partial y={partial_y(y0):.1f}$",
    )
    y_slice_axis.scatter([y0], [z0], color="#222222", zorder=5)
    y_slice_axis.set(
        title="Change y while holding x fixed",
        xlabel=r"$y$",
        ylabel=r"$f(x_0,y)$",
        ylim=(-1.0, 9.0),
    )
    y_slice_axis.legend()

    figure.suptitle(
        "Partial Derivatives: Change One Coordinate at a Time",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "At the selected point, ∂f/∂x = 2.0 while ∂f/∂y = -3.0.",
        ha="center",
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    return save_figure(
        figure,
        output_paths.static / "partial_derivatives_slices.png",
        config,
    )


def create_interactive_surface(
    output_paths: OutputPaths,
    x0: float = 1.0,
    y0: float = -0.75,
) -> Path:
    """Create an offline Plotly surface with both partial-derivative directions."""
    x = np.linspace(-2.2, 2.2, 80)
    y = np.linspace(-2.0, 2.0, 80)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = surface_function(x_grid, y_grid)
    z0 = float(surface_function(x0, y0))
    direction_scale = 0.55

    figure = go.Figure()
    figure.add_surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        colorscale="Blues",
        opacity=0.78,
        showscale=False,
        name="Loss surface",
    )
    figure.add_trace(
        go.Scatter3d(
            x=[x0],
            y=[y0],
            z=[z0],
            mode="markers+text",
            marker={"size": 6, "color": "#222222"},
            text=["selected point"],
            textposition="top center",
            name="Selected point",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[x0, x0 + direction_scale],
            y=[y0, y0],
            z=[z0, z0 + partial_x(x0) * direction_scale],
            mode="lines+markers",
            line={"width": 8, "color": "#d95f02"},
            name=f"∂f/∂x = {partial_x(x0):.1f} (y fixed)",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[x0, x0],
            y=[y0, y0 + direction_scale],
            z=[z0, z0 + partial_y(y0) * direction_scale],
            mode="lines+markers",
            line={"width": 8, "color": "#1b9e77"},
            name=f"∂f/∂y = {partial_y(y0):.1f} (x fixed)",
        )
    )
    figure.update_layout(
        title="Partial Derivatives on f(x, y) = x² + 2y²",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "f(x, y)",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.1}},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 55},
        annotations=[
            {
                "text": (
                    "Each direction changes one variable and holds the other fixed."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.02,
                "showarrow": False,
            }
        ],
    )
    return save_plotly_html(
        figure,
        output_paths.interactive / "partial_derivatives_surface.html",
    )


def generate(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> list[Path]:
    """Generate static and interactive partial-derivative artifacts."""
    print_group_header("Partial derivatives")
    configure_plot_style()
    return [
        create_static_slices(output_paths, config),
        create_interactive_surface(output_paths),
    ]


def main() -> None:
    """Generate partial-derivative artifacts with default settings."""
    generate(topic_output_paths(), RenderConfig())


if __name__ == "__main__":
    main()
