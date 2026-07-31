"""Visualize forward evaluation and backward chain-rule propagation."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch

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


def neuron_values(
    x: float = 1.5,
    weight: float = 0.4,
    bias: float = -0.2,
    target: float = 0.8,
) -> dict[str, float]:
    """Evaluate the neuron and every local or composed derivative."""
    pre_activation = weight * x + bias
    activation = math.tanh(pre_activation)
    loss = (activation - target) ** 2
    d_loss_d_activation = 2.0 * (activation - target)
    d_activation_d_pre_activation = 1.0 - activation**2
    d_loss_d_pre_activation = (
        d_loss_d_activation * d_activation_d_pre_activation
    )
    d_loss_d_weight = d_loss_d_pre_activation * x
    d_loss_d_bias = d_loss_d_pre_activation
    return {
        "x": x,
        "w": weight,
        "b": bias,
        "y": target,
        "z": pre_activation,
        "a": activation,
        "L": loss,
        "dL_da": d_loss_d_activation,
        "da_dz": d_activation_d_pre_activation,
        "dL_dz": d_loss_d_pre_activation,
        "dz_dw": x,
        "dz_db": 1.0,
        "dL_dw": d_loss_d_weight,
        "dL_db": d_loss_d_bias,
    }


def _loss(
    weight: float,
    bias: float,
    x: float = 1.5,
    target: float = 0.8,
) -> float:
    """Return only the scalar loss for finite-difference validation."""
    activation = math.tanh(weight * x + bias)
    return (activation - target) ** 2


def validate_gradients(epsilon: float = 1e-6) -> None:
    """Raise a clear error when the manual chain-rule gradients are wrong."""
    values = neuron_values()
    numerical_weight = (
        _loss(values["w"] + epsilon, values["b"])
        - _loss(values["w"] - epsilon, values["b"])
    ) / (2.0 * epsilon)
    numerical_bias = (
        _loss(values["w"], values["b"] + epsilon)
        - _loss(values["w"], values["b"] - epsilon)
    ) / (2.0 * epsilon)
    if not math.isclose(
        values["dL_dw"],
        numerical_weight,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise RuntimeError("Chain-rule check failed for dL/dw.")
    if not math.isclose(
        values["dL_db"],
        numerical_bias,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise RuntimeError("Chain-rule check failed for dL/db.")


def _draw_node(
    axis: Axes,
    center: tuple[float, float],
    title: str,
    value: str,
    color: str,
    active: bool,
) -> None:
    """Draw one rounded computational-graph node."""
    width = 1.45
    height = 1.0
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2.0, y - height / 2.0),
        width,
        height,
        boxstyle="round,pad=0.08",
        facecolor=color if active else "#eeeeee",
        edgecolor="#222222",
        linewidth=2.2 if active else 1.2,
        alpha=0.95,
    )
    axis.add_patch(patch)
    axis.text(x, y + 0.16, title, ha="center", va="center", fontweight="bold")
    axis.text(x, y - 0.18, value, ha="center", va="center", fontsize=9)


def _arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    text: str,
    color: str,
    active: bool,
    offset: float = 0.25,
) -> None:
    """Draw one graph edge and its derivative annotation."""
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "lw": 2.5 if active else 1.2,
            "color": color if active else "#aaaaaa",
        },
    )
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    axis.text(
        midpoint[0],
        midpoint[1] + offset,
        text,
        ha="center",
        va="center",
        color=color if active else "#888888",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )


def draw_computational_graph(axis: Axes, stage: int) -> None:
    """Draw the graph up to a selected forward/backward stage."""
    values = neuron_values()
    positions = {
        "x": (1.0, 4.2),
        "w": (1.0, 2.7),
        "b": (1.0, 1.2),
        "z": (4.0, 2.7),
        "a": (6.7, 2.7),
        "L": (9.3, 2.7),
    }

    nodes = [
        ("x", r"$x$", f"{values['x']:.2f}", "#9ecae1"),
        ("w", r"$w$", f"{values['w']:.2f}", "#9ecae1"),
        ("b", r"$b$", f"{values['b']:.2f}", "#9ecae1"),
        ("z", r"$z=wx+b$", f"{values['z']:.4f}", "#f6c85f"),
        ("a", r"$a=\tanh(z)$", f"{values['a']:.4f}", "#f6c85f"),
        ("L", r"$L=(a-y)^2$", f"{values['L']:.4f}", "#f6c85f"),
    ]
    for key, title, value, color in nodes:
        _draw_node(axis, positions[key], title, value, color, stage >= 0)

    _arrow(
        axis,
        (1.75, 4.0),
        (3.25, 3.0),
        r"$x$",
        "#1f4e79",
        stage >= 0,
    )
    _arrow(
        axis,
        (1.75, 2.7),
        (3.25, 2.7),
        r"$w$",
        "#1f4e79",
        stage >= 0,
    )
    _arrow(
        axis,
        (1.75, 1.4),
        (3.25, 2.4),
        r"$b$",
        "#1f4e79",
        stage >= 0,
        offset=-0.28,
    )
    _arrow(
        axis,
        (4.75, 2.7),
        (5.95, 2.7),
        rf"$da/dz={values['da_dz']:.4f}$",
        "#d95f02",
        stage >= 2,
    )
    _arrow(
        axis,
        (7.45, 2.7),
        (8.55, 2.7),
        rf"$dL/da={values['dL_da']:.4f}$",
        "#d95f02",
        stage >= 1,
    )

    if stage >= 2:
        axis.annotate(
            "",
            xy=(4.75, 2.25),
            xytext=(5.95, 2.25),
            arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#d95f02"},
        )
        axis.text(
            5.35,
            1.85,
            rf"$dL/dz={values['dL_dz']:.4f}$",
            ha="center",
            color="#d95f02",
        )
    if stage >= 3:
        axis.annotate(
            "",
            xy=(1.75, 2.25),
            xytext=(3.25, 2.25),
            arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#d95f02"},
        )
        axis.text(
            2.5,
            1.9,
            (
                rf"$dL/dw={values['dL_dw']:.4f}$"
                "\n"
                rf"local: $dz/dw=x={values['dz_dw']:.1f}$"
            ),
            ha="center",
            color="#d95f02",
        )
    if stage >= 4:
        axis.annotate(
            "",
            xy=(1.65, 1.0),
            xytext=(3.35, 2.1),
            arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#d95f02"},
        )
        axis.text(
            2.65,
            1.05,
            (
                rf"$dL/db={values['dL_db']:.4f}$"
                "\n"
                rf"local: $dz/db={values['dz_db']:.1f}$"
            ),
            ha="center",
            color="#d95f02",
        )

    stage_titles = [
        "1. Forward pass: compute z, a, and L",
        "2. Start backward: compute dL/da",
        "3. Propagate to z: dL/dz = dL/da × da/dz",
        "4. Propagate to w: dL/dw = dL/dz × dz/dw",
        "5. Propagate to b: dL/db = dL/dz × dz/db",
        "6. Complete chain rule",
    ]
    axis.set_title(stage_titles[stage], fontsize=14, fontweight="bold")
    if stage >= 5:
        axis.text(
            5.2,
            0.25,
            (
                r"$\frac{dL}{dw}=\frac{dL}{da}"
                r"\frac{da}{dz}\frac{dz}{dw}$"
                f" = ({values['dL_da']:.4f})"
                f"({values['da_dz']:.4f})({values['dz_dw']:.1f})"
                f" = {values['dL_dw']:.4f}\n"
                r"$\frac{dL}{db}=\frac{dL}{da}"
                r"\frac{da}{dz}\frac{dz}{db}$"
                f" = {values['dL_db']:.4f}"
            ),
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round", "facecolor": "#fff4cc", "alpha": 0.95},
        )
    axis.text(
        9.8,
        4.55,
        "Backpropagation computes gradients.\n"
        "An optimizer uses them to update parameters.",
        ha="right",
        va="top",
        color="#444444",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    axis.set(xlim=(0.0, 10.2), ylim=(-0.5, 5.0))
    axis.axis("off")


def create_static_graph(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> Path:
    """Render the completed forward and backward computational graph."""
    validate_gradients()
    figure, axis = plt.subplots(figsize=(13, 6.5))
    draw_computational_graph(axis, stage=5)
    return save_figure(
        figure,
        output_paths.static / "chain_rule_computational_graph.png",
        config,
    )


def create_backpropagation_animation(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> Path:
    """Animate the six forward/backward stages."""
    figure, axis = plt.subplots(figsize=(12, 6))

    def update(frame_index: int) -> tuple[object, ...]:
        axis.clear()
        stage = min(5, (frame_index * 6) // config.frames)
        draw_computational_graph(axis, stage)
        return tuple(axis.get_children())

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
        output_paths.animations / "chain_rule_backpropagation.gif",
        config,
    )


def generate(
    output_paths: OutputPaths,
    config: RenderConfig,
) -> list[Path]:
    """Generate static and animated chain-rule artifacts."""
    print_group_header("Chain rule and computational graph")
    configure_plot_style()
    artifacts = [create_static_graph(output_paths, config)]
    if config.generate_gifs:
        artifacts.append(create_backpropagation_animation(output_paths, config))
    return artifacts


def main() -> None:
    """Generate chain-rule artifacts with default settings."""
    generate(topic_output_paths(), RenderConfig())


if __name__ == "__main__":
    main()
