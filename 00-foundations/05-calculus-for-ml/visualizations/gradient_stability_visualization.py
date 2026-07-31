"""Visualize vanishing and exploding gradients across a simplified deep chain."""

from __future__ import annotations

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


LOCAL_DERIVATIVES = (0.5, 0.9, 1.0, 1.1, 1.5)
COLORS = ("#6baed6", "#2171b5", "#222222", "#fdae6b", "#d94801")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute sigmoid stably enough for the plotted interval."""
    return 1.0 / (1.0 + np.exp(-x))


def propagated_gradient(
    local_derivative: float,
    depth: np.ndarray,
) -> np.ndarray:
    """Return |g_final| times a repeated scalar derivative product."""
    return np.abs(local_derivative) ** depth


def create_stability_figure(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> Path:
    """Compare repeated derivative products and activation derivatives."""
    depth = np.arange(0, 61)
    activation_input = np.linspace(-7.0, 7.0, 500)
    sigmoid_output = sigmoid(activation_input)
    sigmoid_derivative = sigmoid_output * (1.0 - sigmoid_output)
    tanh_output = np.tanh(activation_input)
    tanh_derivative = 1.0 - tanh_output**2
    relu_derivative = np.where(activation_input > 0.0, 1.0, 0.0)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for factor, color in zip(LOCAL_DERIVATIVES, COLORS):
        axes[0].plot(
            depth,
            propagated_gradient(factor, depth),
            color=color,
            label=rf"local derivative magnitude = {factor}",
        )
    axes[0].axhline(1.0, color="#555555", linestyle=":", linewidth=1.5)
    axes[0].set(
        title="Repeated Local Derivatives Across Depth",
        xlabel="Number of composed layers",
        ylabel="Gradient magnitude (log scale)",
        yscale="log",
        ylim=(1e-20, 1e12),
    )
    axes[0].legend(fontsize=8)
    axes[0].text(
        0.03,
        0.05,
        "below 1 → vanishing\nnear 1 → stable scale\nabove 1 → exploding",
        transform=axes[0].transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    axes[1].plot(
        activation_input,
        sigmoid_derivative,
        color="#1f4e79",
        label="sigmoid derivative",
    )
    axes[1].plot(
        activation_input,
        tanh_derivative,
        color="#1b9e77",
        label="tanh derivative",
    )
    axes[1].plot(
        activation_input,
        relu_derivative,
        color="#d95f02",
        label="ReLU derivative",
    )
    axes[1].axvspan(-7.0, -3.0, color="#aaaaaa", alpha=0.12)
    axes[1].axvspan(3.0, 7.0, color="#aaaaaa", alpha=0.12)
    axes[1].annotate(
        "sigmoid/tanh saturation:\nderivatives approach zero",
        xy=(4.0, 0.04),
        xytext=(1.3, 0.45),
        arrowprops={"arrowstyle": "->"},
    )
    axes[1].set(
        title="Activation Derivatives Affect Gradient Flow",
        xlabel="Pre-activation z",
        ylabel="Local derivative",
        ylim=(-0.05, 1.1),
    )
    axes[1].legend()
    axes[1].text(
        0.02,
        0.04,
        "Activations are not the only cause.\n"
        "Initialization, depth, Jacobians, normalization,\n"
        "residual paths, and recurrence also matter.",
        transform=axes[1].transAxes,
        fontsize=8.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    figure.suptitle(
        "Why Gradients Can Vanish or Explode",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return save_figure(
        figure,
        output_paths.static / "vanishing_exploding_gradients.png",
        config,
    )


def create_propagation_animation(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> Path:
    """Animate the product of representative local derivative magnitudes."""
    max_depth = config.frames - 1
    depth = np.arange(0, max_depth + 1)
    figure, axis = plt.subplots(figsize=(9, 5.8))
    lines = []
    for factor, color in zip(LOCAL_DERIVATIVES, COLORS):
        line, = axis.plot(
            [],
            [],
            color=color,
            label=rf"$|d|={factor}$",
        )
        lines.append(line)
    status = axis.text(
        0.03,
        0.05,
        "",
        transform=axis.transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    axis.axhline(1.0, color="#555555", linestyle=":")
    axis.set(
        title="Gradient Propagation Through a Deep Scalar Chain",
        xlabel="Network depth",
        ylabel="Gradient magnitude (log scale)",
        xlim=(0, max(1, max_depth)),
        ylim=(1e-30, 1e18),
        yscale="log",
    )
    axis.legend(ncol=5, loc="upper center")

    def update(frame_index: int) -> tuple[object, ...]:
        visible_depth = depth[: frame_index + 1]
        for line, factor in zip(lines, LOCAL_DERIVATIVES):
            line.set_data(
                visible_depth,
                propagated_gradient(factor, visible_depth),
            )
        status.set_text(
            f"depth={frame_index}\n"
            "gradient = final gradient × product of local derivatives"
        )
        return (*lines, status)

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
        output_paths.animations / "gradient_propagation.gif",
        config,
    )


def generate(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> list[Path]:
    """Generate gradient-stability artifacts."""
    print_group_header("Vanishing and exploding gradients")
    configure_plot_style()
    artifacts = [create_stability_figure(output_paths, config)]
    if config.generate_gifs:
        artifacts.append(create_propagation_animation(output_paths, config))
    return artifacts


def main() -> None:
    """Generate stability artifacts with default settings."""
    generate(topic_output_paths(), RenderConfig())


if __name__ == "__main__":
    main()
