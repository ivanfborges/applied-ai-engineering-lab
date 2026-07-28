"""Generate small local GIF assets used by the visual explorer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import math_utils

ASSET_DIR = Path(__file__).resolve().parent / "assets"
FRAME_COUNT = 32
FPS = 12


def configure_axis(axis: plt.Axes, title: str) -> None:
    axis.set_xlim(-4.0, 4.0)
    axis.set_ylim(-4.0, 4.0)
    axis.set_aspect("equal")
    axis.axhline(0, color="#94A3B8", linewidth=0.8)
    axis.axvline(0, color="#94A3B8", linewidth=0.8)
    axis.grid(alpha=0.2)
    axis.set_title(title)


def arrow(axis: plt.Axes, start: np.ndarray, vector: np.ndarray, color: str, label: str) -> None:
    axis.arrow(
        start[0],
        start[1],
        vector[0],
        vector[1],
        width=0.035,
        head_width=0.22,
        length_includes_head=True,
        color=color,
    )
    endpoint = start + vector
    axis.text(endpoint[0] + 0.1, endpoint[1] + 0.1, label, color=color)


def save_vector_addition() -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    a = np.array([2.0, 1.0])

    def draw(frame: int) -> None:
        axis.clear()
        configure_axis(axis, "Vector addition: a + b")
        angle = 2 * np.pi * frame / FRAME_COUNT
        b = 1.7 * np.array([np.cos(angle), np.sin(angle)])
        result = a + b
        arrow(axis, np.zeros(2), a, "#2563EB", "a")
        arrow(axis, np.zeros(2), b, "#F97316", "b")
        arrow(axis, np.zeros(2), result, "#16A34A", "a + b")
        arrow(axis, a, b, "#94A3B8", "translated b")

    animation = FuncAnimation(figure, draw, frames=FRAME_COUNT)
    animation.save(ASSET_DIR / "vector_addition.gif", writer=PillowWriter(fps=FPS))
    plt.close(figure)


def save_vector_projection() -> None:
    figure, axis = plt.subplots(figsize=(6, 6))
    onto = np.array([3.0, 0.8])

    def draw(frame: int) -> None:
        axis.clear()
        configure_axis(axis, "Projection of a onto b")
        angle = 2 * np.pi * frame / FRAME_COUNT
        source = 2.8 * np.array([np.cos(angle), np.sin(angle)])
        projected = math_utils.projection(source, onto)
        arrow(axis, np.zeros(2), source, "#2563EB", "a")
        arrow(axis, np.zeros(2), onto, "#F97316", "b")
        arrow(axis, np.zeros(2), projected, "#16A34A", "proj_b(a)")
        axis.plot(
            [source[0], projected[0]],
            [source[1], projected[1]],
            linestyle="--",
            color="#16A34A",
        )

    animation = FuncAnimation(figure, draw, frames=FRAME_COUNT)
    animation.save(ASSET_DIR / "vector_projection.gif", writer=PillowWriter(fps=FPS))
    plt.close(figure)


def save_transformation_order() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    triangle = np.array([[0.0, 0.0], [1.8, 0.0], [0.4, 1.5], [0.0, 0.0]])
    scale = math_utils.scaling_matrix(1.8, 0.7)
    rotate = math_utils.rotation_matrix(60)
    scale_then_rotate = math_utils.compose_transformations([scale, rotate])
    rotate_then_scale = math_utils.compose_transformations([rotate, scale])

    def draw(frame: int) -> None:
        progress = 0.5 - 0.5 * np.cos(2 * np.pi * frame / FRAME_COUNT)
        for axis, target, title, color in (
            (axes[0], scale_then_rotate, "Scale then rotate · R @ S", "#2563EB"),
            (axes[1], rotate_then_scale, "Rotate then scale · S @ R", "#F97316"),
        ):
            axis.clear()
            configure_axis(axis, title)
            interpolated = (1 - progress) * np.eye(2) + progress * target
            transformed = math_utils.apply_transformation(triangle, interpolated)
            axis.plot(
                triangle[:, 0],
                triangle[:, 1],
                linestyle="--",
                color="#94A3B8",
            )
            axis.fill(transformed[:, 0], transformed[:, 1], color=color, alpha=0.45)
            axis.plot(transformed[:, 0], transformed[:, 1], color=color, linewidth=2)

    animation = FuncAnimation(figure, draw, frames=FRAME_COUNT)
    animation.save(
        ASSET_DIR / "transformation_order.gif",
        writer=PillowWriter(fps=FPS),
    )
    plt.close(figure)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    save_vector_addition()
    save_vector_projection()
    save_transformation_order()
    print(f"GIF assets written to {ASSET_DIR}")


if __name__ == "__main__":
    main()
