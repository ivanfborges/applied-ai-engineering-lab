"""Generate GIF animations for vector spaces, projections, and PCA."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from visualization_utils import (
    DEFAULT_RANDOM_SEED,
    ensure_output_directories,
    gram_schmidt,
    normalize_vector,
    project_onto_subspace,
    project_onto_vector,
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#6B7280"
DARK = "#1F2937"
FLOAT_ARRAY = NDArray[np.float64]


def setup_axis(axis: Axes, limit: float = 4.0) -> None:
    """Reset and format a two-dimensional Cartesian animation axis."""
    axis.clear()
    axis.axhline(0, color=GRAY, linewidth=0.8)
    axis.axvline(0, color=GRAY, linewidth=0.8)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel("x₁")
    axis.set_ylabel("x₂")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle="--", alpha=0.25)


def arrow(
    axis: Axes,
    vector: FLOAT_ARRAY,
    *,
    origin: FLOAT_ARRAY | None = None,
    color: str,
    label: str,
    linestyle: str = "-",
    linewidth: float = 2.4,
) -> None:
    """Draw an animation-frame vector arrow."""
    start = np.zeros(2) if origin is None else np.asarray(origin, dtype=float)
    axis.annotate(
        "",
        xy=start + vector,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linestyle": linestyle,
            "linewidth": linewidth,
            "mutation_scale": 13,
        },
    )
    endpoint = start + vector
    axis.text(endpoint[0] + 0.08, endpoint[1] + 0.08, label, color=color, weight="bold")


def save_animation(
    animation: FuncAnimation,
    figure: plt.Figure,
    output_path: Path,
    fps: int,
) -> None:
    """Save an animation with Pillow and close its figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating: {output_path.name}")
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(figure)
    print(f"Saved: {output_path}")


def animate_linear_combinations(output_path: Path, fps: int) -> None:
    """Animate coefficients exploring the plane spanned by two vectors."""
    v1 = np.array([1.8, 0.45])
    v2 = np.array([-0.55, 1.65])
    frames = 72
    angles = np.linspace(0, 4 * np.pi, frames)
    coefficients = np.column_stack(
        [1.8 * np.sin(angles), 1.8 * np.sin(1.5 * angles + 0.5)]
    )
    positions = coefficients[:, :1] * v1 + coefficients[:, 1:] * v2
    figure, axis = plt.subplots(figsize=(7, 6.4))

    def update(frame: int) -> None:
        setup_axis(axis, 4.5)
        arrow(axis, v1, color=BLUE, label="v₁")
        arrow(axis, v2, color=ORANGE, label="v₂")
        start = max(0, frame - 35)
        trail = positions[start : frame + 1]
        axis.plot(
            trail[:, 0],
            trail[:, 1],
            color=SKY,
            linewidth=2,
            alpha=0.55,
            label="previous combinations",
        )
        arrow(axis, positions[frame], color=RED, label="av₁ + bv₂", linewidth=2.8)
        axis.scatter(*positions[frame], color=RED, s=45, zorder=5)
        a, b = coefficients[frame]
        axis.set_title("Linear Combinations Explore span(v₁, v₂)")
        axis.text(
            0.03,
            0.96,
            f"a = {a:+.2f}\nb = {b:+.2f}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.92},
        )
        axis.legend(loc="lower right")

    animation = FuncAnimation(figure, update, frames=frames, interval=1000 / fps)
    save_animation(animation, figure, output_path, fps)


def animate_projection_onto_vector(output_path: Path, fps: int) -> None:
    """Animate a moving vector and its projection onto a fixed direction."""
    direction = np.array([2.7, 0.9])
    direction_unit = normalize_vector(direction)
    frames = 72
    angles = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    figure, axis = plt.subplots(figsize=(7, 6.4))

    def update(frame: int) -> None:
        setup_axis(axis, 4.0)
        vector = 3.1 * np.array([np.cos(angles[frame]), np.sin(angles[frame])])
        projected = project_onto_vector(vector, direction)
        residual = vector - projected
        orthogonality = float(np.dot(residual, direction))
        assert np.allclose(orthogonality, 0.0, atol=1e-8)
        span_line = np.outer(np.linspace(-1.5, 1.5, 50), direction)
        axis.plot(
            span_line[:, 0],
            span_line[:, 1],
            color=GRAY,
            linestyle=":",
            linewidth=1.5,
        )
        arrow(axis, direction, color=BLUE, label="fixed u")
        arrow(axis, vector, color=RED, label="moving x")
        arrow(axis, projected, color=GREEN, label="projection")
        arrow(
            axis,
            residual,
            origin=projected,
            color=ORANGE,
            label="residual",
            linestyle="--",
        )
        # The small L-shaped polyline marks the 90-degree intersection.
        if np.linalg.norm(residual) > 1e-9:
            along = direction_unit * 0.22
            perpendicular = normalize_vector(residual) * 0.22
            corners = np.array(
                [
                    projected,
                    projected + along,
                    projected + along + perpendicular,
                    projected + perpendicular,
                ]
            )
            axis.plot(corners[:, 0], corners[:, 1], color=DARK, linewidth=1.1)
        axis.set_title("Projection onto a Fixed Vector")
        axis.text(
            0.03,
            0.96,
            f"u · residual = {orthogonality:.2e}",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.92},
        )

    animation = FuncAnimation(figure, update, frames=frames, interval=1000 / fps)
    save_animation(animation, figure, output_path, fps)


def animate_gram_schmidt(output_path: Path, fps: int) -> None:
    """Animate the conceptual stages of two-vector Gram-Schmidt."""
    v1 = np.array([2.4, 0.8])
    v2 = np.array([1.3, 2.5])
    q = gram_schmidt(np.column_stack([v1, v2]))
    q1, q2 = q[:, 0], q[:, 1]
    projection = project_onto_vector(v2, v1)
    orthogonal = v2 - projection
    stage_names = [
        "1. Start with two non-orthogonal vectors",
        "2. Project v₂ onto v₁",
        "3. Subtract the aligned component",
        "4. Obtain an orthogonal direction",
        "5. Normalize both directions",
        "6. Final orthonormal basis",
    ]
    frames_per_stage = max(4, fps // 2)
    frame_count = len(stage_names) * frames_per_stage
    figure, axis = plt.subplots(figsize=(7, 6.4))

    def update(frame: int) -> None:
        stage = min(frame // frames_per_stage, len(stage_names) - 1)
        setup_axis(axis, 3.5)
        axis.set_title(f"Gram–Schmidt: {stage_names[stage]}")
        arrow(axis, v1, color=BLUE, label="v₁")
        arrow(axis, v2, color=ORANGE, label="v₂", linestyle="--")

        if stage >= 1:
            arrow(axis, projection, color=PURPLE, label="projᵥ₁(v₂)")
            axis.plot(
                [projection[0], v2[0]],
                [projection[1], v2[1]],
                color=GRAY,
                linestyle=":",
            )
        if stage >= 2:
            arrow(
                axis,
                orthogonal,
                origin=projection,
                color=RED,
                label="v₂ − projection",
                linestyle="--",
            )
        if stage >= 3:
            arrow(axis, orthogonal, color=RED, label=r"$u_2 \perp v_1$")
            axis.text(
                0.03,
                0.08,
                f"v₁ · u₂ = {np.dot(v1, orthogonal):.2e}",
                transform=axis.transAxes,
                bbox={"facecolor": "white", "alpha": 0.92},
            )
        if stage >= 4:
            arrow(axis, q1, color=GREEN, label="q₁")
            arrow(axis, q2, color=RED, label="q₂")
        if stage >= 5:
            axis.text(
                0.03,
                0.96,
                "QᵀQ = I\nsame span, orthonormal coordinates",
                transform=axis.transAxes,
                va="top",
                weight="bold",
                bbox={"facecolor": "white", "alpha": 0.94},
            )

    animation = FuncAnimation(
        figure, update, frames=frame_count, interval=1000 / fps
    )
    save_animation(animation, figure, output_path, fps)


def animate_projection_onto_plane(output_path: Path, fps: int) -> None:
    """Animate a 3D point and its orthogonal projection onto a fixed plane."""
    basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.35, -0.2]])
    grid = np.linspace(-2.5, 2.5, 12)
    first, second = np.meshgrid(grid, grid)
    plane = first[..., None] * basis[:, 0] + second[..., None] * basis[:, 1]
    normal = normalize_vector(np.cross(basis[:, 0], basis[:, 1]))
    frames = 60
    angles = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    figure = plt.figure(figsize=(7.2, 6.5))
    axis = figure.add_subplot(111, projection="3d")

    def update(frame: int) -> None:
        axis.clear()
        base_point = (
            1.45 * np.cos(angles[frame]) * basis[:, 0]
            + 1.45 * np.sin(angles[frame]) * basis[:, 1]
        )
        height = 1.2 + 0.45 * np.sin(2 * angles[frame])
        vector = base_point + height * normal
        projected = project_onto_subspace(vector, basis)
        residual = vector - projected
        assert np.allclose(basis.T @ residual, 0.0, atol=1e-8)
        axis.plot_surface(
            plane[:, :, 0],
            plane[:, :, 1],
            plane[:, :, 2],
            color=SKY,
            alpha=0.24,
            edgecolor=GRAY,
            linewidth=0.15,
        )
        axis.quiver(0, 0, 0, *basis[:, 0], color=BLUE, linewidth=2.2)
        axis.quiver(0, 0, 0, *basis[:, 1], color=ORANGE, linewidth=2.2)
        axis.quiver(0, 0, 0, *vector, color=RED, linewidth=2.5)
        axis.quiver(0, 0, 0, *projected, color=GREEN, linewidth=2.5)
        axis.plot(
            [projected[0], vector[0]],
            [projected[1], vector[1]],
            [projected[2], vector[2]],
            color=PURPLE,
            linestyle="--",
            linewidth=2.5,
        )
        axis.scatter(*vector, color=RED, s=35, label="moving x")
        axis.scatter(*projected, color=GREEN, marker="s", s=35, label="projection")
        axis.set_xlim(-3, 3)
        axis.set_ylim(-3, 3)
        axis.set_zlim(-2.5, 3.5)
        axis.set_box_aspect((1, 1, 1))
        axis.set_xlabel("x₁")
        axis.set_ylabel("x₂")
        axis.set_zlabel("x₃")
        axis.set_title("Moving Point Projected onto a Fixed Plane")
        axis.legend(loc="upper left")
        axis.view_init(elev=24, azim=35 + frame * 0.35)

    animation = FuncAnimation(figure, update, frames=frames, interval=1000 / fps)
    save_animation(animation, figure, output_path, fps)


def pca_animation_data() -> tuple[FLOAT_ARRAY, FLOAT_ARRAY, PCA]:
    """Create synthetic data and its two-dimensional PCA reconstruction."""
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    latent = rng.normal(size=(90, 3))
    transform = np.array([[2.2, 0.2, 0.04], [1.2, 1.0, 0.07], [0.8, -0.5, 0.14]])
    points = latent @ transform.T
    pca = PCA(n_components=2)
    projected = pca.inverse_transform(pca.fit_transform(points))
    assert points.shape == projected.shape
    return points, projected, pca


def animate_pca_projection(output_path: Path, fps: int) -> None:
    """Animate observations moving onto their two-dimensional principal plane."""
    points, projected, pca = pca_animation_data()
    frames = 56
    transition = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, frames))
    extent = 3.0 * np.sqrt(pca.explained_variance_)
    first, second = np.meshgrid(
        np.linspace(-extent[0], extent[0], 10),
        np.linspace(-extent[1], extent[1], 10),
    )
    plane = (
        pca.mean_
        + first[..., None] * pca.components_[0]
        + second[..., None] * pca.components_[1]
    )
    combined = np.vstack([points, projected, plane.reshape(-1, 3)])
    center = combined.mean(axis=0)
    radius = float(np.max(np.ptp(combined, axis=0))) / 2
    figure = plt.figure(figsize=(7.2, 6.5))
    axis = figure.add_subplot(111, projection="3d")

    def update(frame: int) -> None:
        axis.clear()
        fraction = transition[frame]
        current = (1 - fraction) * points + fraction * projected
        axis.plot_surface(
            plane[:, :, 0],
            plane[:, :, 1],
            plane[:, :, 2],
            color=ORANGE,
            alpha=0.18,
            edgecolor=GRAY,
            linewidth=0.12,
        )
        axis.scatter(
            *current.T,
            color=BLUE,
            marker="o",
            s=20,
            alpha=0.66,
            label="observations",
        )
        for index in range(0, len(points), 15):
            axis.plot(
                [current[index, 0], projected[index, 0]],
                [current[index, 1], projected[index, 1]],
                [current[index, 2], projected[index, 2]],
                color=GRAY,
                linestyle=":",
                linewidth=0.7,
            )
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.set_xlabel("x₁")
        axis.set_ylabel("x₂")
        axis.set_zlabel("x₃")
        axis.set_title(
            "PCA Dimensionality Reduction\n"
            f"{fraction:.0%} moved to the principal plane | "
            f"variance retained {pca.explained_variance_ratio_.sum():.1%}"
        )
        axis.view_init(elev=24, azim=38)

    animation = FuncAnimation(figure, update, frames=frames, interval=1000 / fps)
    save_animation(animation, figure, output_path, fps)


AnimationGenerator = Callable[[Path, int], None]
ANIMATIONS: dict[str, tuple[str, AnimationGenerator]] = {
    "linear-combinations": (
        "linear_combinations_span.gif",
        animate_linear_combinations,
    ),
    "projection-vector": (
        "projection_onto_vector.gif",
        animate_projection_onto_vector,
    ),
    "gram-schmidt": ("gram_schmidt.gif", animate_gram_schmidt),
    "projection-plane": (
        "projection_onto_plane.gif",
        animate_projection_onto_plane,
    ),
    "pca-projection": ("pca_projection.gif", animate_pca_projection),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for GIF generation."""
    parser = argparse.ArgumentParser(
        description="Generate vector-space GIF animations using Pillow."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="GIF destination (default: visualizations/outputs/animations).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Animation frames per second (default: 15).",
    )
    parser.add_argument(
        "--only",
        choices=sorted(ANIMATIONS),
        help="Generate only one named animation.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available animation names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate all animations or one selected animation."""
    args = parse_args()
    if args.list:
        print("\n".join(sorted(ANIMATIONS)))
        return
    if args.fps <= 0:
        raise SystemExit("--fps must be a positive integer.")

    output_directory = (
        args.output_dir
        if args.output_dir is not None
        else ensure_output_directories()["animations"]
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    selected = {args.only: ANIMATIONS[args.only]} if args.only else ANIMATIONS
    print(f"Generating {len(selected)} animation(s) in {output_directory}")
    for filename, generator in selected.values():
        generator(output_directory / filename, args.fps)
    print("Animation generation complete.")


if __name__ == "__main__":
    main()
