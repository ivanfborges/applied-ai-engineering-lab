"""Generate the static figures for the vector-space visual laboratory."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Arc, Polygon
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from visualization_utils import (
    DEFAULT_RANDOM_SEED,
    calculate_cosine_similarity,
    create_synthetic_embedding_clusters,
    ensure_output_directories,
    nearest_neighbors_by_cosine,
    normalize_vector,
    project_onto_subspace,
    project_onto_vector,
    projection_matrix_from_basis,
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#6B7280"
DARK = "#1F2937"
LIGHT = "#E5E7EB"
FLOAT_ARRAY = NDArray[np.float64]


def configure_style() -> None:
    """Apply consistent, readable, colorblind-friendly plotting defaults."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.edgecolor": DARK,
            "axes.labelcolor": DARK,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.size": 10,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "savefig.facecolor": "white",
        }
    )


def setup_2d_axis(
    axis: Axes,
    x_limits: tuple[float, float] = (-4.0, 4.0),
    y_limits: tuple[float, float] = (-4.0, 4.0),
) -> None:
    """Format a Cartesian axis for geometric comparison."""
    axis.axhline(0, color=GRAY, linewidth=0.8)
    axis.axvline(0, color=GRAY, linewidth=0.8)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_xlabel("x₁")
    axis.set_ylabel("x₂")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True)


def draw_vector(
    axis: Axes,
    vector: FLOAT_ARRAY,
    *,
    origin: FLOAT_ARRAY | None = None,
    color: str = BLUE,
    label: str | None = None,
    linestyle: str = "-",
    linewidth: float = 2.2,
    zorder: int = 4,
) -> None:
    """Draw a labeled two-dimensional vector arrow."""
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
            "mutation_scale": 14,
        },
        zorder=zorder,
    )
    if label:
        # Annotation arrows are not automatic legend artists; add a matching handle.
        axis.plot(
            [],
            [],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
        endpoint = start + vector
        axis.text(
            endpoint[0] + 0.08,
            endpoint[1] + 0.08,
            label,
            color=color,
            weight="bold",
            zorder=zorder + 1,
        )


def draw_right_angle(
    axis: Axes,
    vertex: FLOAT_ARRAY,
    along: FLOAT_ARRAY,
    perpendicular: FLOAT_ARRAY,
    size: float = 0.22,
) -> None:
    """Draw a small right-angle marker from two perpendicular directions."""
    first = normalize_vector(along) * size
    second = normalize_vector(perpendicular) * size
    marker = Polygon(
        [vertex, vertex + first, vertex + first + second, vertex + second],
        closed=False,
        fill=False,
        edgecolor=DARK,
        linewidth=1.2,
        zorder=6,
    )
    axis.add_patch(marker)


def set_3d_equal(axis: Axes3D, points: FLOAT_ARRAY) -> None:
    """Use equal data ranges for all axes in a Matplotlib 3D view."""
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2
    radius = max(float(np.max(maximum - minimum)) / 2, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("x₁")
    axis.set_ylabel("x₂")
    axis.set_zlabel("x₃")


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    """Save and close one high-resolution Matplotlib figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def vectors_and_linear_combinations(output_directory: Path) -> None:
    """Visualize coefficient choices as points in a two-vector span."""
    v1 = np.array([2.0, 0.7])
    v2 = np.array([-0.6, 1.8])
    coefficients = np.arange(-2, 3)
    combinations = np.array([a * v1 + b * v2 for a in coefficients for b in coefficients])

    figure, axis = plt.subplots(figsize=(8, 7))
    setup_2d_axis(axis, (-6, 6), (-6, 6))
    axis.scatter(
        combinations[:, 0],
        combinations[:, 1],
        c=np.repeat(coefficients, len(coefficients)),
        cmap="viridis",
        marker="o",
        s=42,
        edgecolor="white",
        linewidth=0.6,
        label=r"$a v_1 + b v_2$",
        zorder=3,
    )
    draw_vector(axis, v1, color=BLUE, label=r"$v_1$")
    draw_vector(axis, v2, color=ORANGE, label=r"$v_2$")
    selected = 1.5 * v1 - v2
    draw_vector(axis, selected, color=RED, label=r"$1.5v_1-v_2$")
    axis.set_title("Vectors and Linear Combinations")
    axis.text(
        0.02,
        0.98,
        "Changing a and b moves the result throughout span(v₁, v₂).\n"
        "Because the vectors are independent, their span is ℝ².",
        transform=axis.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    save_figure(figure, output_directory / "01_vectors_linear_combinations.png")


def span_one_vs_two_vectors(output_directory: Path) -> None:
    """Contrast a one-dimensional span with a two-dimensional span."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    v = np.array([1.4, 0.8])
    independent = np.array([[1.5, -0.5], [0.4, 1.4]]).T

    setup_2d_axis(axes[0])
    scale = np.linspace(-4, 4, 100)
    line = np.outer(scale, v)
    axes[0].plot(line[:, 0], line[:, 1], color=SKY, linewidth=5, alpha=0.55)
    draw_vector(axes[0], v, color=BLUE, label=r"$v$")
    draw_vector(axes[0], -1.7 * v, color=BLUE, linestyle="--", label=r"$-1.7v$")
    axes[0].set_title("One vector: a 1D subspace")
    axes[0].text(
        0.03,
        0.96,
        "span(v) is a line through the origin\nDimension = 1",
        transform=axes[0].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.9},
    )

    setup_2d_axis(axes[1])
    grid = np.array(
        [
            a * independent[:, 0] + b * independent[:, 1]
            for a in np.linspace(-2, 2, 9)
            for b in np.linspace(-2, 2, 9)
        ]
    )
    axes[1].scatter(
        grid[:, 0],
        grid[:, 1],
        color=SKY,
        marker=".",
        s=60,
        alpha=0.65,
        label="Sampled combinations",
    )
    draw_vector(axes[1], independent[:, 0], color=BLUE, label=r"$v_1$")
    draw_vector(axes[1], independent[:, 1], color=ORANGE, label=r"$v_2$")
    axes[1].set_title("Two independent vectors: a 2D subspace")
    axes[1].text(
        0.03,
        0.96,
        "span(v₁, v₂) covers the plane\nDimension = 2",
        transform=axes[1].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.9},
    )
    axes[1].legend(loc="lower right")
    figure.suptitle("Span of One Vector versus Two Vectors", fontsize=15, weight="bold")
    figure.tight_layout()
    save_figure(figure, output_directory / "02_span_one_vs_two_vectors.png")


def independence_vs_dependence(output_directory: Path) -> None:
    """Show how a redundant direction changes matrix rank."""
    independent = np.array([[2.0, -0.5], [0.5, 1.8]]).T
    dependent = np.array([[1.2, 2.4], [0.8, 1.6]]).T
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))

    for axis, matrix, title in [
        (axes[0], independent, "Independent: two directions"),
        (axes[1], dependent, "Dependent: one direction"),
    ]:
        setup_2d_axis(axis, (-3.5, 3.5), (-3.5, 3.5))
        rank = int(np.linalg.matrix_rank(matrix))
        draw_vector(axis, matrix[:, 0], color=BLUE, label=r"$v_1$")
        draw_vector(axis, matrix[:, 1], color=ORANGE, linestyle="--", label=r"$v_2$")
        axis.set_title(title)
        axis.text(
            0.04,
            0.94,
            f"Matrix rank = {rank}",
            transform=axis.transAxes,
            va="top",
            weight="bold",
            bbox={"facecolor": "white", "alpha": 0.92},
        )
        axis.legend(loc="lower right")
    t = np.linspace(-3, 3, 100)
    dependent_line = np.outer(t, dependent[:, 0])
    axes[1].plot(
        dependent_line[:, 0],
        dependent_line[:, 1],
        color=GRAY,
        linestyle=":",
        linewidth=2,
        label="shared span",
    )
    axes[1].text(-3.2, -2.4, r"$v_2=2v_1$: no new direction", color=DARK)
    figure.suptitle("Linear Independence versus Dependence", fontsize=15, weight="bold")
    figure.tight_layout()
    save_figure(figure, output_directory / "03_independence_vs_dependence.png")


def different_bases_same_space(output_directory: Path) -> None:
    """Represent one geometric vector in standard and rotated coordinates."""
    vector = np.array([2.4, 1.4])
    standard = np.eye(2)
    angle = np.deg2rad(35)
    rotated = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    standard_coordinates = vector.copy()
    rotated_coordinates = rotated.T @ vector
    assert np.allclose(rotated @ rotated_coordinates, vector)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for axis in axes:
        setup_2d_axis(axis, (-1, 3.5), (-1, 3.5))
        draw_vector(axis, vector, color=RED, label=r"same $x$")

    draw_vector(axes[0], standard[:, 0], color=BLUE, label=r"$e_1$")
    draw_vector(axes[0], standard[:, 1], color=ORANGE, label=r"$e_2$")
    axes[0].set_title("Standard basis")
    axes[0].text(
        0.04,
        0.92,
        f"coordinates = ({standard_coordinates[0]:.2f}, "
        f"{standard_coordinates[1]:.2f})",
        transform=axes[0].transAxes,
        bbox={"facecolor": "white", "alpha": 0.92},
    )

    draw_vector(axes[1], rotated[:, 0], color=BLUE, label=r"$b_1$")
    draw_vector(axes[1], rotated[:, 1], color=ORANGE, label=r"$b_2$")
    axes[1].set_title("Rotated orthonormal basis")
    axes[1].text(
        0.04,
        0.92,
        f"coordinates = ({rotated_coordinates[0]:.2f}, "
        f"{rotated_coordinates[1]:.2f})",
        transform=axes[1].transAxes,
        bbox={"facecolor": "white", "alpha": 0.92},
    )
    figure.suptitle(
        "Different Bases, Same Vector Space and Geometric Vector",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "The coordinate values change with the basis; the red vector does not.",
        ha="center",
        color=DARK,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    save_figure(figure, output_directory / "04_different_bases_same_space.png")


def orthogonal_vs_nonorthogonal(output_directory: Path) -> None:
    """Compare orthonormal coordinates with a valid skew basis."""
    orthonormal = np.eye(2)
    nonorthogonal = np.array([[1.0, 0.65], [0.0, 1.0]])
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))

    for axis in axes:
        setup_2d_axis(axis, (-0.6, 2.2), (-0.6, 2.2))

    draw_vector(axes[0], orthonormal[:, 0], color=BLUE, label=r"$q_1$")
    draw_vector(axes[0], orthonormal[:, 1], color=ORANGE, label=r"$q_2$")
    axes[0].add_patch(
        Polygon(
            [[0, 0], [0.2, 0], [0.2, 0.2], [0, 0.2]],
            closed=False,
            fill=False,
            edgecolor=DARK,
        )
    )
    axes[0].set_title("Orthonormal basis")
    axes[0].text(
        0.04,
        0.94,
        "angle = 90°\nq₁ · q₂ = 0\ncoordinates = Qᵀx",
        transform=axes[0].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.92},
    )

    b1 = nonorthogonal[:, 0]
    b2 = nonorthogonal[:, 1]
    angle = np.degrees(
        np.arccos(np.clip(calculate_cosine_similarity(b1, b2), -1.0, 1.0))
    )
    draw_vector(axes[1], b1, color=BLUE, label=r"$b_1$")
    draw_vector(axes[1], b2, color=ORANGE, label=r"$b_2$")
    axes[1].add_patch(
        Arc((0, 0), 0.8, 0.8, angle=0, theta1=angle, theta2=90, color=GREEN, lw=2)
    )
    axes[1].set_title("Valid, non-orthogonal basis")
    axes[1].text(
        0.04,
        0.94,
        f"angle = {angle:.1f}°\nb₁ · b₂ = {np.dot(b1, b2):.2f}\n"
        "coordinates require solving Bc = x",
        transform=axes[1].transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.92},
    )
    figure.suptitle("Orthogonal versus Non-Orthogonal Bases", fontsize=15, weight="bold")
    figure.tight_layout()
    save_figure(figure, output_directory / "05_orthogonal_vs_nonorthogonal.png")


def projection_onto_vector(output_directory: Path) -> None:
    """Show the projection-residual decomposition in two dimensions."""
    vector = np.array([1.3, 3.2])
    direction = np.array([2.6, 1.0])
    projected = project_onto_vector(vector, direction)
    residual = vector - projected
    projection_matrix = projection_matrix_from_basis(direction[:, None])
    orthogonality = float(np.dot(residual, direction))
    assert np.allclose(orthogonality, 0.0, atol=1e-8)
    assert np.allclose(projection_matrix @ vector, projected)

    figure, axis = plt.subplots(figsize=(8, 7))
    setup_2d_axis(axis, (-0.8, 4.2), (-0.8, 4.2))
    line_scale = np.linspace(-1, 2, 100)
    line = np.outer(line_scale, direction)
    axis.plot(
        line[:, 0],
        line[:, 1],
        color=GRAY,
        linestyle=":",
        linewidth=2,
        label="span(u)",
    )
    draw_vector(axis, direction, color=BLUE, label=r"$u$")
    draw_vector(axis, vector, color=RED, label=r"$x$")
    draw_vector(axis, projected, color=GREEN, label=r"$\mathrm{proj}_u(x)$")
    draw_vector(
        axis,
        residual,
        origin=projected,
        color=ORANGE,
        linestyle="--",
        label="residual",
    )
    draw_right_angle(axis, projected, direction, residual)
    axis.set_title("Orthogonal Projection onto a Vector")
    axis.text(
        0.03,
        0.97,
        r"$\mathrm{projection}=\frac{u\cdot x}{u\cdot u}u$"
        f"\nresidual · u = {orthogonality:.2e}"
        f"\nP = {np.array2string(projection_matrix, precision=2)}"
        "\nPᵀ = P and P² = P",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.93},
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    save_figure(figure, output_directory / "06_projection_onto_vector.png")


def plane_geometry() -> tuple[FLOAT_ARRAY, FLOAT_ARRAY, FLOAT_ARRAY, FLOAT_ARRAY]:
    """Return a plane basis, source vector, projection, and residual."""
    basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.45, 0.25]])
    vector = np.array([1.4, 1.2, 2.3])
    projected = project_onto_subspace(vector, basis)
    residual = vector - projected
    assert np.allclose(basis.T @ residual, 0.0, atol=1e-8)
    return basis, vector, projected, residual


def projection_onto_plane(output_directory: Path) -> None:
    """Render an orthogonal projection onto a 3D plane through the origin."""
    basis, vector, projected, residual = plane_geometry()
    grid = np.linspace(-2.3, 2.3, 15)
    first, second = np.meshgrid(grid, grid)
    surface = (
        first[..., None] * basis[:, 0] + second[..., None] * basis[:, 1]
    )

    figure = plt.figure(figsize=(9, 7.5))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(
        surface[:, :, 0],
        surface[:, :, 1],
        surface[:, :, 2],
        color=SKY,
        alpha=0.28,
        edgecolor=GRAY,
        linewidth=0.2,
    )
    for basis_vector, color, label in [
        (basis[:, 0], BLUE, "basis b₁"),
        (basis[:, 1], ORANGE, "basis b₂"),
    ]:
        axis.quiver(0, 0, 0, *basis_vector, color=color, linewidth=2.5, label=label)
    axis.quiver(0, 0, 0, *vector, color=RED, linewidth=2.7, label="x")
    axis.quiver(0, 0, 0, *projected, color=GREEN, linewidth=2.7, label="projection")
    axis.quiver(
        *projected,
        *residual,
        color=PURPLE,
        linewidth=2.7,
        linestyle="--",
        label="orthogonal residual",
    )
    axis.scatter(*vector, color=RED, s=45)
    axis.scatter(*projected, color=GREEN, marker="s", s=45)
    set_3d_equal(axis, np.vstack([surface.reshape(-1, 3), vector, projected]))
    axis.set_title(r"Projection onto a Plane: Residual $\perp$ Subspace", pad=18)
    axis.legend(loc="upper left")
    axis.text2D(
        0.02,
        0.02,
        f"b₁ᵀr = {basis[:, 0] @ residual:.2e}\n"
        f"b₂ᵀr = {basis[:, 1] @ residual:.2e}",
        transform=axis.transAxes,
        bbox={"facecolor": "white", "alpha": 0.92},
    )
    figure.tight_layout()
    save_figure(figure, output_directory / "07_projection_onto_plane.png")


def orthogonal_decomposition(output_directory: Path) -> None:
    """Arrange projection and residual head-to-tail and verify Pythagoras."""
    vector = np.array([1.3, 3.2])
    direction = np.array([2.6, 1.0])
    projected = project_onto_vector(vector, direction)
    residual = vector - projected
    lhs = float(np.dot(vector, vector))
    rhs = float(np.dot(projected, projected) + np.dot(residual, residual))
    assert np.allclose(lhs, rhs, atol=1e-8)

    figure, axis = plt.subplots(figsize=(8, 7))
    setup_2d_axis(axis, (-0.7, 4.0), (-0.7, 4.0))
    draw_vector(axis, vector, color=RED, label=r"$x$")
    draw_vector(axis, projected, color=GREEN, label=r"$\hat{x}$")
    draw_vector(
        axis,
        residual,
        origin=projected,
        color=ORANGE,
        linestyle="--",
        label=r"$r=x-\hat{x}$",
    )
    draw_right_angle(axis, projected, projected, residual)
    axis.plot(
        [0, projected[0], vector[0]],
        [0, projected[1], vector[1]],
        color=GRAY,
        linewidth=0.8,
        linestyle=":",
    )
    axis.set_title(r"Orthogonal Decomposition: $x=\hat{x}+r$")
    axis.text(
        0.03,
        0.97,
        f"‖x‖² = {lhs:.3f}\n"
        f"‖projection‖² + ‖residual‖² = {rhs:.3f}\n"
        f"difference = {abs(lhs - rhs):.2e}",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.93},
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    save_figure(figure, output_directory / "08_orthogonal_decomposition.png")


def noisy_plane_data(
    sample_count: int = 160, random_seed: int = DEFAULT_RANDOM_SEED
) -> FLOAT_ARRAY:
    """Create synthetic 3D points concentrated around a two-dimensional plane."""
    rng = np.random.default_rng(random_seed)
    latent = rng.normal(size=(sample_count, 2))
    true_basis = np.array([[1.0, 0.0], [0.2, 1.0], [0.55, -0.3]])
    points = latent @ true_basis.T
    normal = normalize_vector(np.cross(true_basis[:, 0], true_basis[:, 1]))
    points += rng.normal(scale=0.08, size=(sample_count, 1)) * normal
    return points


def ambient_vs_intrinsic_dimension(output_directory: Path) -> None:
    """Estimate and display a two-dimensional subspace inside ambient R^3."""
    points = noisy_plane_data()
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(points)
    projected = pca.inverse_transform(coordinates)
    grid = np.linspace(-2.4, 2.4, 12)
    first, second = np.meshgrid(grid, grid)
    plane = (
        pca.mean_
        + first[..., None] * pca.components_[0]
        + second[..., None] * pca.components_[1]
    )

    figure = plt.figure(figsize=(9, 7.5))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color=BLUE,
        marker="o",
        s=20,
        alpha=0.58,
        label="synthetic observations",
    )
    axis.plot_surface(
        plane[:, :, 0],
        plane[:, :, 1],
        plane[:, :, 2],
        color=ORANGE,
        alpha=0.24,
        edgecolor=GRAY,
        linewidth=0.15,
    )
    axis.set_title("Ambient Dimension 3, Approximate Intrinsic Dimension 2", pad=18)
    set_3d_equal(axis, np.vstack([points, projected]))
    axis.legend(loc="upper left")
    axis.text2D(
        0.02,
        0.02,
        "The arrays have three coordinates, but most variation lies near\n"
        f"one plane. Two-PC explained variance = "
        f"{pca.explained_variance_ratio_.sum():.1%}.",
        transform=axis.transAxes,
        bbox={"facecolor": "white", "alpha": 0.93},
    )
    figure.tight_layout()
    save_figure(figure, output_directory / "09_ambient_vs_intrinsic_dimension.png")


def correlated_3d_data(
    sample_count: int = 150, random_seed: int = DEFAULT_RANDOM_SEED
) -> FLOAT_ARRAY:
    """Create a deterministic correlated three-dimensional synthetic dataset."""
    rng = np.random.default_rng(random_seed)
    latent = rng.normal(size=(sample_count, 3))
    transform = np.array([[2.4, 0.2, 0.05], [1.5, 1.1, 0.08], [0.9, -0.6, 0.16]])
    return latent @ transform.T + np.array([0.4, -0.2, 0.6])


def pca_as_projection(output_directory: Path) -> None:
    """Show PCA projection in ambient coordinates and reduced coordinates."""
    points = correlated_3d_data()
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(points)
    reconstructed = pca.inverse_transform(coordinates)
    reconstruction_error = float(np.mean(np.square(points - reconstructed)))
    explained = pca.explained_variance_ratio_

    plane_extent = 3.1 * np.sqrt(pca.explained_variance_)
    first_values = np.linspace(-plane_extent[0], plane_extent[0], 13)
    second_values = np.linspace(-plane_extent[1], plane_extent[1], 13)
    first, second = np.meshgrid(first_values, second_values)
    plane = (
        pca.mean_
        + first[..., None] * pca.components_[0]
        + second[..., None] * pca.components_[1]
    )

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(*points.T, color=BLUE, s=18, alpha=0.32, label="original 3D points")
    axis.scatter(
        *reconstructed.T,
        color=GREEN,
        marker="^",
        s=18,
        alpha=0.65,
        label="projected points",
    )
    axis.plot_surface(
        plane[:, :, 0],
        plane[:, :, 1],
        plane[:, :, 2],
        color=ORANGE,
        alpha=0.18,
        edgecolor=GRAY,
        linewidth=0.15,
    )
    for component_index, color in enumerate([RED, PURPLE]):
        component = pca.components_[component_index] * plane_extent[component_index]
        axis.quiver(
            *pca.mean_,
            *component,
            color=color,
            linewidth=3,
            label=f"PC{component_index + 1}",
        )
    for index in range(0, len(points), 15):
        axis.plot(
            [points[index, 0], reconstructed[index, 0]],
            [points[index, 1], reconstructed[index, 1]],
            [points[index, 2], reconstructed[index, 2]],
            color=GRAY,
            linestyle=":",
            linewidth=0.7,
        )
    set_3d_equal(axis, np.vstack([points, reconstructed]))
    axis.set_title("PCA as Projection onto a Two-Dimensional Subspace", pad=18)
    axis.legend(loc="upper left")
    axis.text2D(
        0.02,
        0.02,
        f"3D → 2D | explained variance = {explained.sum():.1%}\n"
        f"mean squared reconstruction error = {reconstruction_error:.4f}",
        transform=axis.transAxes,
        bbox={"facecolor": "white", "alpha": 0.93},
    )
    figure.tight_layout()
    save_figure(figure, output_directory / "10a_pca_projection_3d.png")

    figure_2d, axis_2d = plt.subplots(figsize=(8, 6.5))
    axis_2d.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        color=BLUE,
        marker="o",
        s=28,
        alpha=0.68,
        edgecolor="white",
        linewidth=0.4,
    )
    axis_2d.axhline(0, color=GRAY, linewidth=0.8)
    axis_2d.axvline(0, color=GRAY, linewidth=0.8)
    axis_2d.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    axis_2d.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
    axis_2d.set_title("The Same Observations in 2D PCA Coordinates")
    axis_2d.set_aspect("equal", adjustable="datalim")
    axis_2d.grid(True)
    axis_2d.text(
        0.02,
        0.98,
        f"Original dimension: 3\nReduced dimension: 2\nMSE: {reconstruction_error:.4f}",
        transform=axis_2d.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.93},
    )
    figure_2d.tight_layout()
    save_figure(figure_2d, output_directory / "10b_pca_coordinates_2d.png")


def embeddings_as_vector_space(output_directory: Path) -> None:
    """Visualize synthetic high-dimensional embedding neighborhoods with PCA."""
    embeddings, labels, query, _ = create_synthetic_embedding_clusters()
    neighbor_indices, cosine_scores = nearest_neighbors_by_cosine(
        embeddings, query, count=5
    )
    combined = np.vstack([embeddings, query])
    pca = PCA(n_components=2)
    visual_coordinates = pca.fit_transform(combined)
    points_2d = visual_coordinates[:-1]
    query_2d = visual_coordinates[-1]
    group_styles = {
        "Machine learning": (BLUE, "o"),
        "Cloud infrastructure": (ORANGE, "s"),
        "Legal documents": (GREEN, "^"),
    }

    figure, axis = plt.subplots(figsize=(9, 7))
    for group_name, (color, marker) in group_styles.items():
        mask = labels == group_name
        axis.scatter(
            points_2d[mask, 0],
            points_2d[mask, 1],
            color=color,
            marker=marker,
            s=42,
            alpha=0.72,
            label=group_name,
            edgecolor="white",
            linewidth=0.5,
        )
    axis.scatter(
        *query_2d,
        color=RED,
        marker="*",
        s=250,
        edgecolor=DARK,
        linewidth=0.8,
        label="synthetic query",
        zorder=6,
    )
    for index in neighbor_indices:
        neighbor = points_2d[index]
        axis.plot(
            [query_2d[0], neighbor[0]],
            [query_2d[1], neighbor[1]],
            color=RED,
            linestyle="--",
            linewidth=1,
            alpha=0.65,
        )
        axis.scatter(
            *neighbor,
            facecolor="none",
            edgecolor=RED,
            marker="o",
            s=110,
            linewidth=1.5,
            zorder=5,
        )
    neighbor_summary = "\n".join(
        f"#{rank}: cosine = {cosine_scores[index]:.3f}"
        for rank, index in enumerate(neighbor_indices, start=1)
    )
    axis.text(
        0.50,
        0.56,
        "Selected nearest neighbors\n" + neighbor_summary,
        transform=axis.transAxes,
        va="center",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.93},
    )
    axis.set_xlabel("PCA visualization component 1")
    axis.set_ylabel("PCA visualization component 2")
    axis.set_title("Synthetic Embeddings as Points in a Learned Vector Space")
    axis.grid(True)
    axis.legend(loc="best")
    axis.text(
        0.02,
        0.02,
        f"Actual vectors: {embeddings.shape[1]}D synthetic embeddings\n"
        f"Display only: 2D PCA ({pca.explained_variance_ratio_.sum():.1%} variance)\n"
        "Neighbors are selected by high-dimensional cosine similarity.",
        transform=axis.transAxes,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.93},
    )
    figure.tight_layout()
    save_figure(figure, output_directory / "11_synthetic_embedding_space.png")


def cosine_vs_euclidean(output_directory: Path) -> None:
    """Demonstrate candidates whose metric rankings disagree."""
    query = np.array([1.0, 0.0])
    candidates = {
        "A: same direction, large magnitude": np.array([3.0, 0.12]),
        "B: nearby, different direction": np.array([0.78, 0.48]),
        "C: nearby below query": np.array([0.88, -0.58]),
    }
    colors = [BLUE, ORANGE, GREEN]
    markers = ["o", "s", "^"]
    rows = []
    for name, candidate in candidates.items():
        rows.append(
            [
                name.split(":")[0],
                f"{np.linalg.norm(candidate):.2f}",
                f"{calculate_cosine_similarity(query, candidate):.3f}",
                f"{np.linalg.norm(query - candidate):.3f}",
            ]
        )

    figure, axis = plt.subplots(figsize=(11, 7))
    setup_2d_axis(axis, (-0.5, 3.5), (-1.4, 1.6))
    draw_vector(axis, query, color=RED, label="query q", linewidth=3)
    for (name, candidate), color, marker in zip(
        candidates.items(), colors, markers, strict=True
    ):
        draw_vector(axis, candidate, color=color, label=name.split(":")[0])
        axis.scatter(
            *candidate,
            color=color,
            marker=marker,
            s=70,
            edgecolor=DARK,
            linewidth=0.5,
        )
        axis.plot(
            [query[0], candidate[0]],
            [query[1], candidate[1]],
            color=color,
            linestyle=":",
            linewidth=1.2,
        )
    table = axis.table(
        cellText=rows,
        colLabels=["Vector", "Magnitude", "Cosine(q, ·)", "Euclidean(q, ·)"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.39, 0.61, 0.59, 0.31],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for column in range(4):
        table[(0, column)].set_facecolor(LIGHT)
        table[(0, column)].set_text_props(weight="bold")
    axis.set_title("Cosine Similarity versus Euclidean Distance")
    axis.text(
        0.39,
        0.25,
        "Cosine prefers A (direction). Euclidean prefers B (absolute position).\n"
        "The metric must match training, normalization, and retrieval intent.",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.93},
    )
    figure.tight_layout()
    save_figure(figure, output_directory / "12_cosine_vs_euclidean.png")


VISUALIZATIONS: dict[str, Callable[[Path], None]] = {
    "linear-combinations": vectors_and_linear_combinations,
    "span-dimension": span_one_vs_two_vectors,
    "independence": independence_vs_dependence,
    "change-of-basis": different_bases_same_space,
    "orthogonal-bases": orthogonal_vs_nonorthogonal,
    "projection-vector": projection_onto_vector,
    "projection-plane": projection_onto_plane,
    "orthogonal-decomposition": orthogonal_decomposition,
    "intrinsic-dimension": ambient_vs_intrinsic_dimension,
    "pca-projection": pca_as_projection,
    "embedding-space": embeddings_as_vector_space,
    "similarity-metrics": cosine_vs_euclidean,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for static figure generation."""
    parser = argparse.ArgumentParser(
        description="Generate static vector-space visualizations."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="PNG destination (default: visualizations/outputs/images).",
    )
    parser.add_argument(
        "--only",
        choices=sorted(VISUALIZATIONS),
        help="Generate only one named visualization.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available visualization names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate all requested static figures or one selected figure."""
    args = parse_args()
    if args.list:
        print("\n".join(sorted(VISUALIZATIONS)))
        return

    configure_style()
    output_directory = (
        args.output_dir
        if args.output_dir is not None
        else ensure_output_directories()["images"]
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    selected = (
        {args.only: VISUALIZATIONS[args.only]} if args.only else VISUALIZATIONS
    )
    print(f"Generating {len(selected)} static visualization(s) in {output_directory}")
    for generator in selected.values():
        generator(output_directory)
    print("Static visualization generation complete.")


if __name__ == "__main__":
    main()
