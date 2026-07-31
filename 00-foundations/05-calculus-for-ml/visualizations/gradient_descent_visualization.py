"""Visualize gradients, descent trajectories, and learning-rate behavior."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation

try:
    from .utils import (
        OutputPaths,
        RenderConfig,
        configure_plot_style,
        print_group_header,
        save_animation,
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
        save_animation,
        save_figure,
        save_plotly_html,
        topic_output_paths,
    )


def loss_surface(
    weight_1: np.ndarray | float,
    weight_2: np.ndarray | float,
) -> np.ndarray | float:
    """Return L(w₁, w₂) = 0.5w₁² + 2w₂²."""
    return 0.5 * np.asarray(weight_1) ** 2 + 2.0 * np.asarray(weight_2) ** 2


def loss_gradient(weight_1: float, weight_2: float) -> np.ndarray:
    """Return ∇L = [w₁, 4w₂]."""
    return np.array([weight_1, 4.0 * weight_2], dtype=float)


def gradient_descent_path(
    initial_point: tuple[float, float],
    learning_rate: float,
    steps: int,
) -> np.ndarray:
    """Return parameter values produced by fixed-step gradient descent."""
    point = np.array(initial_point, dtype=float)
    path = [point.copy()]
    for _ in range(steps):
        point = point - learning_rate * loss_gradient(point[0], point[1])
        path.append(point.copy())
    return np.asarray(path)


def create_vector_field(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> Path:
    """Plot contours, uphill gradients, and one highlighted descent direction."""
    coordinates = np.linspace(-4.0, 4.0, 17)
    weight_1, weight_2 = np.meshgrid(coordinates, coordinates)
    loss = loss_surface(weight_1, weight_2)
    gradient_1 = weight_1
    gradient_2 = 4.0 * weight_2
    gradient_norm = np.hypot(gradient_1, gradient_2)
    safe_norm = np.where(gradient_norm == 0.0, 1.0, gradient_norm)

    figure, axis = plt.subplots(figsize=(9, 7))
    contour = axis.contour(
        weight_1,
        weight_2,
        loss,
        levels=14,
        cmap="Blues",
    )
    axis.clabel(contour, inline=True, fontsize=8)
    axis.quiver(
        weight_1,
        weight_2,
        gradient_1 / safe_norm,
        gradient_2 / safe_norm,
        color="#1f4e79",
        alpha=0.55,
        pivot="mid",
        label=r"Gradient $\nabla L$ (uphill)",
    )

    example_point = np.array([3.0, 2.0])
    example_gradient = loss_gradient(*example_point)
    direction = -example_gradient / np.linalg.norm(example_gradient)
    axis.quiver(
        [example_point[0]],
        [example_point[1]],
        [direction[0]],
        [direction[1]],
        color="#d95f02",
        scale=2.2,
        scale_units="xy",
        width=0.014,
        label=r"Negative gradient $-\nabla L$ (downhill)",
    )
    axis.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=180,
        color="#1b9e77",
        label="Global minimum",
        zorder=5,
    )
    axis.scatter(
        [example_point[0]],
        [example_point[1]],
        color="#222222",
        zorder=5,
    )
    axis.annotate(
        "Optimization follows the opposite\nof the uphill gradient.",
        xy=example_point + 0.85 * direction,
        xytext=(-3.8, 3.25),
        arrowprops={"arrowstyle": "->", "color": "#d95f02"},
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    axis.set(
        title=r"Gradient Field of $L(w_1,w_2)=0.5w_1^2+2w_2^2$",
        xlabel=r"Model parameter $w_1$",
        ylabel=r"Model parameter $w_2$",
        xlim=(-4.25, 4.25),
        ylim=(-4.25, 4.25),
        aspect="equal",
    )
    axis.legend(loc="lower right")
    return save_figure(
        figure,
        output_paths.static / "gradient_vector_field.png",
        config,
    )


def create_learning_rate_comparison(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> Path:
    """Compare slow, stable, and excessive learning rates."""
    settings = [
        (0.03, "Small: slow convergence", "#6baed6"),
        (0.20, "Appropriate: stable convergence", "#1b9e77"),
        (0.55, "Excessive: oscillation/divergence", "#d95f02"),
    ]
    initial_point = (4.0, 2.5)
    steps = 24
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    grid = np.linspace(-6.0, 6.0, 180)
    weight_1, weight_2 = np.meshgrid(grid, grid)
    contour_loss = loss_surface(weight_1, weight_2)
    axes[0].contour(
        weight_1,
        weight_2,
        contour_loss,
        levels=np.geomspace(0.2, 120.0, 14),
        cmap="Greys",
        alpha=0.65,
    )

    for learning_rate, label, color in settings:
        path = gradient_descent_path(initial_point, learning_rate, steps)
        losses = loss_surface(path[:, 0], path[:, 1])
        axes[0].plot(
            path[:, 0],
            path[:, 1],
            marker="o",
            markersize=3,
            color=color,
            label=label,
        )
        axes[1].plot(losses, marker="o", markersize=3, color=color, label=label)

    axes[0].scatter([0.0], [0.0], marker="*", s=150, color="#222222")
    axes[0].set(
        title="Parameter-space trajectories",
        xlabel=r"$w_1$",
        ylabel=r"$w_2$",
        xlim=(-6.0, 6.0),
        ylim=(-6.0, 6.0),
        aspect="equal",
    )
    axes[1].set(
        title="Loss by update",
        xlabel="Gradient-descent iteration",
        ylabel="Loss (log scale)",
        yscale="log",
    )
    axes[1].text(
        0.03,
        0.05,
        "The learning rate controls step size.\n"
        "A correct gradient cannot rescue an excessive step.",
        transform=axes[1].transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.suptitle(
        "Learning Rate Changes Optimization Behavior",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.92))
    return save_figure(
        figure,
        output_paths.static / "learning_rate_comparison.png",
        config,
    )


def create_trajectory_animation(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> Path:
    """Animate gradient descent and display its local update direction."""
    path = gradient_descent_path((4.0, 3.0), 0.08, config.frames - 1)
    grid = np.linspace(-4.5, 4.5, 180)
    weight_1, weight_2 = np.meshgrid(grid, grid)
    losses = loss_surface(weight_1, weight_2)

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.contour(weight_1, weight_2, losses, levels=16, cmap="Blues")
    trajectory_line, = axis.plot([], [], color="#d95f02", linewidth=2.5)
    current_point, = axis.plot([], [], "o", color="#222222", markersize=8)
    update_arrow = axis.quiver(
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        color="#d95f02",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.012,
    )
    status = axis.text(
        0.03,
        0.96,
        "",
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
    )
    axis.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=170,
        color="#1b9e77",
        label="Minimum",
        zorder=5,
    )
    axis.set(
        title="Gradient Descent Follows the Negative Gradient",
        xlabel=r"Model parameter $w_1$",
        ylabel=r"Model parameter $w_2$",
        xlim=(-4.5, 4.5),
        ylim=(-4.5, 4.5),
        aspect="equal",
    )
    axis.legend(loc="lower right")

    def update(frame_index: int) -> tuple[object, ...]:
        point = path[frame_index]
        gradient = loss_gradient(point[0], point[1])
        update_direction = -0.08 * gradient
        trajectory_line.set_data(
            path[: frame_index + 1, 0],
            path[: frame_index + 1, 1],
        )
        current_point.set_data([point[0]], [point[1]])
        update_arrow.set_offsets(point.reshape(1, 2))
        update_arrow.set_UVC(
            np.array([update_direction[0]]),
            np.array([update_direction[1]]),
        )
        status.set_text(
            f"iteration: {frame_index}\n"
            f"w₁={point[0]:.4f}, w₂={point[1]:.4f}\n"
            f"loss={float(loss_surface(*point)):.5f}\n"
            f"∇L=({gradient[0]:.3f}, {gradient[1]:.3f})\n"
            "orange arrow = -η∇L"
        )
        return trajectory_line, current_point, update_arrow, status

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
        output_paths.animations / "gradient_descent_trajectory.gif",
        config,
    )


def create_interactive_trajectory(output_paths: OutputPaths) -> Path:
    """Create an offline 3D loss surface and hoverable optimization path."""
    grid = np.linspace(-4.5, 4.5, 100)
    weight_1, weight_2 = np.meshgrid(grid, grid)
    loss = loss_surface(weight_1, weight_2)
    path = gradient_descent_path((4.0, 3.0), 0.08, 45)
    path_loss = loss_surface(path[:, 0], path[:, 1])
    iterations = np.arange(len(path))

    figure = go.Figure()
    figure.add_surface(
        x=weight_1,
        y=weight_2,
        z=loss,
        colorscale="Blues",
        opacity=0.72,
        showscale=False,
        name="Loss surface",
    )
    figure.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path_loss,
            mode="lines+markers",
            line={"color": "#d95f02", "width": 7},
            marker={"size": 3},
            customdata=iterations,
            hovertemplate=(
                "iteration=%{customdata}<br>"
                "w₁=%{x:.5f}<br>w₂=%{y:.5f}<br>"
                "loss=%{z:.6f}<extra></extra>"
            ),
            name="Gradient-descent trajectory",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[path[0, 0], path[-1, 0]],
            y=[path[0, 1], path[-1, 1]],
            z=[path_loss[0], path_loss[-1]],
            mode="markers+text",
            marker={"size": 7, "color": ["#222222", "#1b9e77"]},
            text=["initial", "final"],
            textposition="top center",
            name="Endpoints",
        )
    )
    figure.update_layout(
        title="Interactive Gradient-Descent Trajectory",
        scene={
            "xaxis_title": "w₁",
            "yaxis_title": "w₂",
            "zaxis_title": "Loss",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.1}},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    return save_plotly_html(
        figure,
        output_paths.interactive / "gradient_descent_3d.html",
    )


def create_saddle_point(output_paths: OutputPaths) -> Path:
    """Show that a zero gradient can occur at a saddle rather than a minimum."""
    grid = np.linspace(-2.5, 2.5, 100)
    x_grid, y_grid = np.meshgrid(grid, grid)
    z_grid = x_grid**2 - y_grid**2
    figure = go.Figure()
    figure.add_surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        colorscale="RdBu",
        opacity=0.8,
        showscale=False,
        contours={
            "z": {
                "show": True,
                "usecolormap": True,
                "project_z": True,
            }
        },
    )
    figure.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker={"size": 7, "color": "#222222"},
            text=["∇f = 0, but not a minimum"],
            textposition="top center",
            name="Saddle point",
        )
    )
    figure.update_layout(
        title="Saddle Point: f(x, y) = x² − y²",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "f(x, y)",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    return save_plotly_html(
        figure,
        output_paths.interactive / "saddle_point.html",
    )


def generate(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> list[Path]:
    """Generate all gradient and gradient-descent artifacts."""
    print_group_header("Gradient field and gradient descent")
    configure_plot_style()
    artifacts = [
        create_vector_field(output_paths, config),
        create_learning_rate_comparison(output_paths, config),
        create_interactive_trajectory(output_paths),
        create_saddle_point(output_paths),
    ]
    if config.generate_gifs:
        artifacts.append(create_trajectory_animation(output_paths, config))
    return artifacts


def main() -> None:
    """Generate gradient-descent artifacts with default settings."""
    generate(topic_output_paths(), RenderConfig())


if __name__ == "__main__":
    main()
