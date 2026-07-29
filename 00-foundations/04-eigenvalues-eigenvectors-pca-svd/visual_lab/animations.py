"""Matplotlib GIF animations for eigenvectors and SVD geometry."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "applied_ai_visual_lab_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .plotting import COLORS, apply_matplotlib_style, save_matplotlib_figure


def _grid_segments(extent: float = 2.1, lines: int = 11) -> list[np.ndarray]:
    coordinates = np.linspace(-extent, extent, lines)
    return [
        segment
        for value in coordinates
        for segment in (
            np.array([[-extent, value], [extent, value]]),
            np.array([[value, -extent], [value, extent]]),
        )
    ]


def _circle_points(samples: int = 240) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, samples)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix.T


def _draw_arrow(
    axis: plt.Axes,
    vector: np.ndarray,
    color: str,
    label: str | None = None,
    *,
    alpha: float = 1.0,
    width: float = 0.018,
) -> None:
    axis.quiver(
        0,
        0,
        vector[0],
        vector[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color,
        alpha=alpha,
        width=width,
        label=label,
        zorder=6,
    )


def _draw_transformed_geometry(
    axis: plt.Axes,
    transform: np.ndarray,
    *,
    grid_color: str = "#AEB8C2",
    circle_color: str = "#277DA1",
) -> None:
    for segment in _grid_segments():
        transformed = _transform_points(segment, transform)
        axis.plot(
            transformed[:, 0],
            transformed[:, 1],
            color=grid_color,
            alpha=0.45,
            linewidth=0.8,
        )
    transformed_circle = _transform_points(_circle_points(), transform)
    axis.plot(
        transformed_circle[:, 0],
        transformed_circle[:, 1],
        color=circle_color,
        linewidth=2.6,
        label="Transformed unit circle",
    )


def _configure_2d_axis(axis: plt.Axes, limit: float, title: str) -> None:
    axis.set(
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        xlabel="x",
        ylabel="y",
        aspect="equal",
        title=title,
    )
    axis.axhline(0, color="#A6AFB8", linewidth=1)
    axis.axvline(0, color="#A6AFB8", linewidth=1)


def generate_eigenvector_animation(
    gif_path: Path,
    static_path: Path,
    *,
    frames: int = 52,
) -> list[Path]:
    """Animate I -> A and show which vector directions remain invariant."""
    apply_matplotlib_style()
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    arbitrary_vectors = np.array(
        [[1.0, 0.25], [-0.55, 1.0], [0.25, -0.85]]
    )
    standard_basis = np.eye(2)

    figure, axis = plt.subplots(figsize=(8, 8))

    def draw(progress: float, *, final_static: bool = False) -> None:
        axis.clear()
        transform = (1.0 - progress) * np.eye(2) + progress * matrix
        _draw_transformed_geometry(axis, transform)

        for index, vector in enumerate(standard_basis):
            _draw_arrow(
                axis,
                _transform_points(vector[None, :], transform)[0],
                COLORS["gray"],
                "Transformed standard basis" if index == 0 else None,
                alpha=0.9,
                width=0.012,
            )
        for index, vector in enumerate(arbitrary_vectors):
            original = vector / np.linalg.norm(vector)
            transformed = transform @ original
            _draw_arrow(
                axis,
                original,
                COLORS["gray"],
                "Original arbitrary vectors" if index == 0 else None,
                alpha=0.36,
                width=0.007,
            )
            _draw_arrow(
                axis,
                transformed,
                COLORS["purple"],
                "Transformed arbitrary vectors" if index == 0 else None,
                alpha=0.82,
                width=0.011,
            )

        for index, (eigenvalue, eigenvector, color) in enumerate(
            zip(
                eigenvalues,
                eigenvectors.T,
                (COLORS["orange"], COLORS["red"]),
                strict=True,
            ),
            start=1,
        ):
            transformed = transform @ eigenvector
            _draw_arrow(
                axis,
                1.35 * eigenvector,
                color,
                f"Eigenvector v{index}",
                alpha=0.38,
                width=0.009,
            )
            _draw_arrow(
                axis,
                transformed,
                color,
                f"A(t)v{index}",
                width=0.016,
            )
            axis.text(
                transformed[0] * 1.07,
                transformed[1] * 1.07,
                f"v{index}: λ={eigenvalue:.2f}",
                color=color,
                fontweight="bold",
            )

        _configure_2d_axis(
            axis,
            4.5,
            "Eigenvectors Stay on Their Invariant Lines",
        )
        axis.text(
            0.98,
            0.97,
            f"A(t) = (1 − t)I + tA   ·   t={progress:.2f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=COLORS["navy"],
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        axis.text(
            0.02,
            0.02,
            (
                "Arbitrary vectors generally rotate.\n"
                "Each eigenvalue controls stretching along its invariant line."
            ),
            transform=axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
        )
        if final_static:
            axis.text(
                0.98,
                0.02,
                r"$Av_i=\lambda_i v_i$",
                transform=axis.transAxes,
                ha="right",
                fontsize=15,
                color=COLORS["navy"],
            )
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=False))
        axis.legend(unique.values(), unique.keys(), loc="upper left", fontsize=8)
        figure.tight_layout()

    progress_values = np.concatenate(
        [
            np.zeros(5),
            0.5 - 0.5 * np.cos(np.linspace(0, np.pi, frames - 12)),
            np.ones(7),
        ]
    )

    def update(frame_index: int) -> None:
        draw(float(progress_values[frame_index]))

    animation = FuncAnimation(
        figure,
        update,
        frames=len(progress_values),
        interval=90,
        repeat=True,
    )
    animation.save(gif_path, writer=PillowWriter(fps=11), dpi=95)
    draw(1.0, final_static=True)
    save_matplotlib_figure(figure, static_path, dpi=180)
    return [gif_path, static_path]


def _svd_stage_transform(
    progress: float,
    left: np.ndarray,
    singular_values: np.ndarray,
    right_t: np.ndarray,
) -> tuple[np.ndarray, str, str]:
    sigma = np.diag(singular_values)
    if progress <= 1.0:
        alpha = progress
        operation = (1.0 - alpha) * np.eye(2) + alpha * right_t
        return operation, "Stage 1 — apply Vᵀ", r"$x_1=V^\top x$"
    if progress <= 2.0:
        alpha = progress - 1.0
        scaling = (1.0 - alpha) * np.eye(2) + alpha * sigma
        return scaling @ right_t, "Stage 2 — apply Σ", r"$x_2=\Sigma V^\top x$"
    alpha = progress - 2.0
    output_rotation = (1.0 - alpha) * np.eye(2) + alpha * left
    return (
        output_rotation @ sigma @ right_t,
        "Stage 3 — apply U",
        r"$x_3=U\Sigma V^\top x=Xx$",
    )


def _draw_svd_frame(
    axis: plt.Axes,
    transform: np.ndarray,
    stage_label: str,
    equation: str,
    singular_values: np.ndarray,
    right_t: np.ndarray,
    left: np.ndarray,
) -> None:
    axis.clear()
    _draw_transformed_geometry(axis, transform)
    sample_vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.65], [-0.45, 0.9]]
    )
    for index, vector in enumerate(sample_vectors):
        transformed = transform @ vector
        _draw_arrow(
            axis,
            transformed,
            COLORS["purple"],
            "Sample vectors" if index == 0 else None,
            alpha=0.82,
            width=0.012,
        )

    # Right singular vectors describe input directions; left singular vectors
    # describe their final output directions after scaling.
    for index in range(2):
        _draw_arrow(
            axis,
            right_t[index],
            (COLORS["orange"], COLORS["red"])[index],
            f"right singular v{index + 1}",
            alpha=0.38,
            width=0.008,
        )
        _draw_arrow(
            axis,
            left[:, index] * singular_values[index],
            (COLORS["orange"], COLORS["red"])[index],
            f"σ{index + 1}u{index + 1}",
            alpha=0.72,
            width=0.011,
        )

    _configure_2d_axis(axis, 3.3, f"{stage_label}\n{equation}")
    axis.text(
        0.02,
        0.02,
        f"σ₁={singular_values[0]:.2f}   σ₂={singular_values[1]:.2f}",
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        color=COLORS["navy"],
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axis.legend(unique.values(), unique.keys(), loc="upper left", fontsize=8)


def generate_svd_animation(
    gif_path: Path,
    static_path: Path,
    *,
    frames_per_stage: int = 16,
) -> list[Path]:
    """Animate the geometric sequence V.T -> Sigma -> U."""
    apply_matplotlib_style()
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array([[1.8, 0.8], [-0.4, 1.2]])
    left, singular_values, right_t = np.linalg.svd(matrix)

    figure, axis = plt.subplots(figsize=(8, 8))
    progress_values = np.concatenate(
        [
            np.zeros(4),
            np.linspace(0.0, 1.0, frames_per_stage, endpoint=False),
            np.ones(4),
            np.linspace(1.0, 2.0, frames_per_stage, endpoint=False),
            np.full(4, 2.0),
            np.linspace(2.0, 3.0, frames_per_stage),
            np.full(7, 3.0),
        ]
    )

    def update(frame_index: int) -> None:
        transform, label, equation = _svd_stage_transform(
            float(progress_values[frame_index]),
            left,
            singular_values,
            right_t,
        )
        _draw_svd_frame(
            axis,
            transform,
            label,
            equation,
            singular_values,
            right_t,
            left,
        )
        figure.suptitle("SVD Geometry: Rotation/Reflection → Scaling → Rotation")
        figure.tight_layout(rect=(0, 0, 1, 0.96))

    animation = FuncAnimation(
        figure,
        update,
        frames=len(progress_values),
        interval=105,
        repeat=True,
    )
    animation.save(gif_path, writer=PillowWriter(fps=10), dpi=95)
    plt.close(figure)

    summary, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    stage_specs = [
        (np.eye(2), "Original", r"$x$"),
        (right_t, "Apply Vᵀ", r"$V^\top x$"),
        (np.diag(singular_values) @ right_t, "Apply Σ", r"$\Sigma V^\top x$"),
        (matrix, "Apply U", r"$U\Sigma V^\top x=Xx$"),
    ]
    for panel, (transform, label, equation) in zip(
        axes,
        stage_specs,
        strict=True,
    ):
        _draw_svd_frame(
            panel,
            transform,
            label,
            equation,
            singular_values,
            right_t,
            left,
        )
        panel.get_legend().remove()
    summary.suptitle(
        "SVD Decomposes One Linear Map into Orthogonal Directions and Scaling"
    )
    summary.text(
        0.5,
        0.015,
        "The last panel equals direct application of X; σ₁ and σ₂ set the ellipse semi-axis lengths.",
        ha="center",
        color=COLORS["gray"],
    )
    summary.tight_layout(rect=(0, 0.085, 1, 0.92))
    save_matplotlib_figure(summary, static_path, dpi=180)
    return [gif_path, static_path]
