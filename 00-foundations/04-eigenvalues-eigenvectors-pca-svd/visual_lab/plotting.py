"""Static and standalone interactive plots for the visual laboratory."""

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
import plotly.graph_objects as go
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

from .datasets import (
    correlated_2d,
    correlated_3d,
    correlated_features,
    synthetic_grayscale_image,
)
from .math_utils import (
    align_component_signs,
    approximate_svd_compression_ratio,
    center_data,
    covariance_matrix,
    eigendecomposition_pca,
    explained_variance_from_singular_values,
    low_rank_approximation,
    reconstruct_data,
    svd_pca,
)

COLORS = {
    "navy": "#17324D",
    "blue": "#277DA1",
    "cyan": "#43AA8B",
    "orange": "#F8961E",
    "red": "#D1495B",
    "gold": "#F9C74F",
    "gray": "#6C757D",
    "light": "#E9ECEF",
    "purple": "#6F5AA8",
}


def apply_matplotlib_style() -> None:
    """Apply a restrained, readable style shared by static figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFC",
            "axes.edgecolor": "#BCC5CE",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.color": "#7A8793",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "figure.titlesize": 16,
            "legend.frameon": False,
        }
    )


def save_matplotlib_figure(
    figure: plt.Figure,
    output_path: Path,
    *,
    dpi: int = 180,
) -> Path:
    """Create the parent folder, save a high-resolution PNG, and close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def write_plotly_html(figure: go.Figure, output_path: Path) -> Path:
    """Write a standalone HTML document with Plotly embedded."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        auto_play=False,
    )
    return output_path


def covariance_pca_figure(data: np.ndarray) -> plt.Figure:
    """Build the covariance ellipse and PCA-direction explanation."""
    result = eigendecomposition_pca(data, n_components=2)
    ratios = result.explained_variance_ratio
    mean = result.mean

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.scatter(
        data[:, 0],
        data[:, 1],
        s=20,
        alpha=0.42,
        color=COLORS["blue"],
        edgecolors="none",
        label="Synthetic observations",
    )
    axis.scatter(
        *mean,
        marker="X",
        s=120,
        color=COLORS["navy"],
        label="Mean",
        zorder=6,
    )

    angle = np.degrees(
        np.arctan2(result.components[0, 1], result.components[0, 0])
    )
    ellipse = Ellipse(
        xy=mean,
        width=4.0 * np.sqrt(result.explained_variance[0]),
        height=4.0 * np.sqrt(result.explained_variance[1]),
        angle=angle,
        facecolor=COLORS["cyan"],
        edgecolor=COLORS["navy"],
        alpha=0.14,
        linewidth=2.0,
        label="2-standard-deviation covariance ellipse",
    )
    axis.add_patch(ellipse)

    component_colors = (COLORS["orange"], COLORS["red"])
    for index, (component, eigenvalue, ratio, color) in enumerate(
        zip(
            result.components,
            result.explained_variance,
            ratios,
            component_colors,
            strict=True,
        ),
        start=1,
    ):
        arrow = component * (2.0 * np.sqrt(eigenvalue))
        axis.quiver(
            mean[0],
            mean[1],
            arrow[0],
            arrow[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            width=0.010,
            zorder=7,
            label=f"PC{index}",
        )
        axis.text(
            mean[0] + arrow[0] * 1.05,
            mean[1] + arrow[1] * 1.05,
            (
                f"PC{index}\n"
                f"λ={eigenvalue:.2f}, EVR={ratio:.1%}\n"
                f"v=({component[0]:.2f}, {component[1]:.2f})"
            ),
            color=color,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
        )

    axis.axhline(mean[1], color=COLORS["gray"], linestyle="--", alpha=0.45)
    axis.axvline(mean[0], color=COLORS["gray"], linestyle="--", alpha=0.45)
    axis.set(
        title="Arrow length is proportional to √λ; the longer axis carries more variance.",
        xlabel="Original feature x₁",
        ylabel="Original feature x₂",
        aspect="equal",
    )
    axis.title.set_fontsize(11)
    axis.title.set_fontweight("normal")
    axis.title.set_color(COLORS["gray"])
    figure.suptitle(
        "Covariance Eigenvectors Define the PCA Coordinate System",
        y=0.98,
    )
    axis.legend(loc="upper left")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def generate_covariance_pca(output_path: Path) -> Path:
    """Generate the static 2D covariance and PCA visualization."""
    apply_matplotlib_style()
    return save_matplotlib_figure(
        covariance_pca_figure(correlated_2d()),
        output_path,
    )


def _axis_trace_3d(
    origin: np.ndarray,
    direction: np.ndarray,
    half_length: float,
    name: str,
    color: str,
    *,
    dash: str = "solid",
    width: int = 6,
) -> go.Scatter3d:
    endpoints = np.vstack(
        [origin - direction * half_length, origin + direction * half_length]
    )
    return go.Scatter3d(
        x=endpoints[:, 0],
        y=endpoints[:, 1],
        z=endpoints[:, 2],
        mode="lines",
        line={"color": color, "width": width, "dash": dash},
        name=name,
        hovertemplate=f"{name}<extra></extra>",
    )


def pca_3d_figure(data: np.ndarray) -> go.Figure:
    """Build an interactive 3D cloud with original and principal axes."""
    result = eigendecomposition_pca(data, n_components=3)
    centered = data - result.mean
    point_distance = np.linalg.norm(centered, axis=1)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode="markers",
            marker={
                "size": 4,
                "color": point_distance,
                "colorscale": "Viridis",
                "opacity": 0.68,
                "colorbar": {"title": "Distance<br>from mean"},
            },
            customdata=np.column_stack([np.arange(data.shape[0]), point_distance]),
            hovertemplate=(
                "point=%{customdata[0]:.0f}<br>x=%{x:.2f}<br>y=%{y:.2f}"
                "<br>z=%{z:.2f}<br>distance=%{customdata[1]:.2f}<extra></extra>"
            ),
            name="Synthetic observations",
        )
    )

    basis = np.eye(3)
    original_colors = ("#C9CED6", "#9EA7B2", "#737D89")
    for index in range(3):
        figure.add_trace(
            _axis_trace_3d(
                result.mean,
                basis[index],
                3.0,
                f"Original {('x', 'y', 'z')[index]} axis",
                original_colors[index],
                dash="dash",
                width=4,
            )
        )

    pc_colors = (COLORS["orange"], COLORS["red"], COLORS["cyan"])
    for index, (component, eigenvalue, ratio, color) in enumerate(
        zip(
            result.components,
            result.explained_variance,
            result.explained_variance_ratio,
            pc_colors,
            strict=True,
        ),
        start=1,
    ):
        figure.add_trace(
            _axis_trace_3d(
                result.mean,
                component,
                2.5 * np.sqrt(eigenvalue),
                f"PC{index} · {ratio:.1%} variance",
                color,
            )
        )

    figure.update_layout(
        title={
            "text": (
                "PCA Rotates the Coordinate System<br>"
                "<sup>Principal axes align with decreasing variance; "
                "drag to rotate and scroll to zoom.</sup>"
            )
        },
        template="plotly_white",
        scene={
            "xaxis_title": "Original x",
            "yaxis_title": "Original y",
            "zaxis_title": "Original z",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": 1.55, "z": 1.1}},
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 0, "r": 0, "b": 0, "t": 80},
    )
    return figure


def generate_pca_3d(output_path: Path) -> Path:
    """Generate a standalone interactive 3D PCA-axis visualization."""
    return write_plotly_html(pca_3d_figure(correlated_3d()), output_path)


def _principal_plane_surface(
    mean: np.ndarray,
    components: np.ndarray,
    scores: np.ndarray,
) -> go.Surface:
    extent_1 = np.max(np.abs(scores[:, 0])) * 1.08
    extent_2 = np.max(np.abs(scores[:, 1])) * 1.08
    axis_1 = np.linspace(-extent_1, extent_1, 12)
    axis_2 = np.linspace(-extent_2, extent_2, 12)
    grid_1, grid_2 = np.meshgrid(axis_1, axis_2)
    points = (
        mean[None, None, :]
        + grid_1[:, :, None] * components[0]
        + grid_2[:, :, None] * components[1]
    )
    return go.Surface(
        x=points[:, :, 0],
        y=points[:, :, 1],
        z=points[:, :, 2],
        opacity=0.24,
        colorscale=[[0, COLORS["cyan"]], [1, COLORS["cyan"]]],
        showscale=False,
        name="PC1–PC2 plane",
        hoverinfo="skip",
    )


def projection_3d_animation_figure(data: np.ndarray) -> go.Figure:
    """Animate 3D observations moving orthogonally onto the PC1-PC2 plane."""
    result = eigendecomposition_pca(data, n_components=2)
    projected = reconstruct_data(result.scores, result.components, result.mean)
    subset = np.linspace(0, data.shape[0] - 1, 22, dtype=int)

    line_x: list[float | None] = []
    line_y: list[float | None] = []
    line_z: list[float | None] = []
    for index in subset:
        line_x.extend([data[index, 0], projected[index, 0], None])
        line_y.extend([data[index, 1], projected[index, 1], None])
        line_z.extend([data[index, 2], projected[index, 2], None])

    moving_trace = go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode="markers",
        marker={"size": 4, "color": COLORS["blue"], "opacity": 0.68},
        name="Moving observations",
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
    )
    figure = go.Figure(
        data=[
            _principal_plane_surface(result.mean, result.components, result.scores),
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                line={"color": "rgba(108,117,125,0.42)", "width": 2},
                name="Projection paths (subset)",
                hoverinfo="skip",
            ),
            moving_trace,
            _axis_trace_3d(
                result.mean,
                result.components[0],
                3.5,
                "PC1",
                COLORS["orange"],
            ),
            _axis_trace_3d(
                result.mean,
                result.components[1],
                3.5,
                "PC2",
                COLORS["red"],
            ),
        ]
    )

    alphas = np.linspace(0.0, 1.0, 31)
    frames = []
    for frame_index, alpha in enumerate(alphas):
        positions = (1.0 - alpha) * data + alpha * projected
        frames.append(
            go.Frame(
                name=str(frame_index),
                data=[
                    go.Scatter3d(
                        x=positions[:, 0],
                        y=positions[:, 1],
                        z=positions[:, 2],
                    )
                ],
                traces=[2],
            )
        )
    figure.frames = frames

    slider_steps = [
        {
            "label": f"{alpha:.0%}",
            "method": "animate",
            "args": [
                [str(index)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for index, alpha in enumerate(alphas)
    ]
    figure.update_layout(
        title={
            "text": (
                "Orthogonal Projection onto the PC1–PC2 Plane<br>"
                "<sup>The discarded PC3 residual shrinks to zero.</sup>"
            )
        },
        template="plotly_white",
        scene={
            "xaxis_title": "Original x",
            "yaxis_title": "Original y",
            "zaxis_title": "Original z",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": 1.55, "z": 1.15}},
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 0.02,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 80, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Reset",
                        "method": "animate",
                        "args": [
                            ["0"],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Projection progress: "},
                "pad": {"b": 45},
                "steps": slider_steps,
            }
        ],
        margin={"l": 0, "r": 0, "b": 20, "t": 85},
    )
    return figure


def generate_projection_3d_html(output_path: Path) -> Path:
    """Generate the standalone animated 3D-to-2D projection."""
    return write_plotly_html(
        projection_3d_animation_figure(correlated_3d()),
        output_path,
    )


def generate_projection_comparison(output_path: Path) -> Path:
    """Generate a static original-3D versus PCA-score comparison."""
    apply_matplotlib_style()
    data = correlated_3d()
    result = eigendecomposition_pca(data, n_components=2)
    projected = reconstruct_data(result.scores, result.components, result.mean)

    figure = plt.figure(figsize=(14, 6))
    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis_2d = figure.add_subplot(1, 2, 2)
    color = result.scores[:, 0]

    axis_3d.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        c=color,
        cmap="viridis",
        s=17,
        alpha=0.54,
    )
    axis_3d.scatter(
        projected[:, 0],
        projected[:, 1],
        projected[:, 2],
        c=color,
        cmap="viridis",
        s=10,
        alpha=0.35,
    )
    axis_3d.set(
        title="Original 3D cloud and its PC1–PC2 projection",
        xlabel="x",
        ylabel="y",
        zlabel="z",
    )

    scatter = axis_2d.scatter(
        result.scores[:, 0],
        result.scores[:, 1],
        c=color,
        cmap="viridis",
        s=24,
        alpha=0.68,
        edgecolors="none",
    )
    axis_2d.axhline(0, color=COLORS["gray"], linewidth=1, alpha=0.5)
    axis_2d.axvline(0, color=COLORS["gray"], linewidth=1, alpha=0.5)
    axis_2d.set(
        title="Coordinates after projection",
        xlabel=f"PC1 ({result.explained_variance_ratio[0]:.1%} variance)",
        ylabel=f"PC2 ({result.explained_variance_ratio[1]:.1%} variance)",
    )
    figure.colorbar(scatter, ax=axis_2d, label="PC1 score")
    figure.suptitle(
        "PCA Replaces Three Original Coordinates with Two Principal Scores"
    )
    figure.tight_layout()
    return save_matplotlib_figure(figure, output_path)


def generate_explained_variance(output_path: Path) -> Path:
    """Plot explained variance and reconstruction error across retained ranks."""
    apply_matplotlib_style()
    data = correlated_features()
    pca = PCA().fit(data)
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    component_counts = np.arange(1, ratios.size + 1)
    threshold = 0.90
    selected = int(np.searchsorted(cumulative, threshold) + 1)

    errors = []
    for count in component_counts:
        scores = pca.transform(data)[:, :count]
        reconstruction = scores @ pca.components_[:count] + pca.mean_
        errors.append(float(np.mean((data - reconstruction) ** 2)))

    figure, (variance_axis, error_axis) = plt.subplots(
        1,
        2,
        figsize=(14, 5.8),
    )
    variance_axis.bar(
        component_counts,
        ratios,
        color=COLORS["blue"],
        alpha=0.82,
        label="Individual explained variance",
    )
    variance_axis.plot(
        component_counts,
        cumulative,
        color=COLORS["orange"],
        marker="o",
        linewidth=2.5,
        label="Cumulative explained variance",
    )
    variance_axis.axhline(
        threshold,
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.5,
        label=f"{threshold:.0%} target",
    )
    variance_axis.axvline(selected, color=COLORS["red"], linestyle=":", alpha=0.8)
    variance_axis.scatter(
        selected,
        cumulative[selected - 1],
        color=COLORS["red"],
        s=80,
        zorder=5,
    )
    variance_axis.annotate(
        f"Smallest k = {selected}\n({cumulative[selected - 1]:.1%} retained)",
        (selected, cumulative[selected - 1]),
        xytext=(12, -42),
        textcoords="offset points",
        color=COLORS["red"],
    )
    variance_axis.set(
        title="Eigenvalues determine explained variance",
        xlabel="Number of retained components",
        ylabel="Variance ratio",
        xticks=component_counts,
        ylim=(0, 1.05),
    )
    variance_axis.legend(loc="center right")

    error_axis.plot(
        component_counts,
        errors,
        color=COLORS["purple"],
        marker="o",
        linewidth=2.5,
    )
    error_axis.scatter(
        selected,
        errors[selected - 1],
        color=COLORS["red"],
        s=80,
        zorder=5,
    )
    error_axis.axvline(selected, color=COLORS["red"], linestyle=":", alpha=0.8)
    error_axis.set(
        title="Reconstruction error falls as rank increases",
        xlabel="Number of retained components",
        ylabel="Reconstruction MSE",
        xticks=component_counts,
    )
    figure.suptitle(
        "Component Selection Balances Information Retention and Compression"
    )
    figure.text(
        0.5,
        0.01,
        (
            "Synthetic 8-feature data generated from 3 latent factors. "
            "The threshold is a heuristic, not a downstream-quality guarantee."
        ),
        ha="center",
        color=COLORS["gray"],
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    return save_matplotlib_figure(figure, output_path)


def _rank_metrics(
    image: np.ndarray,
    singular_values: np.ndarray,
    rank: int,
    reconstruction: np.ndarray,
) -> tuple[float, float, float]:
    energy = float(
        np.sum(singular_values[:rank] ** 2) / np.sum(singular_values**2)
    )
    mse = float(np.mean((image - reconstruction) ** 2))
    compression = approximate_svd_compression_ratio(image.shape, rank)
    return energy, mse, compression


def generate_low_rank_static(output_path: Path) -> Path:
    """Generate low-rank reconstructions, singular decay, and error curve."""
    apply_matplotlib_style()
    image = synthetic_grayscale_image()
    _, singular_values, _ = np.linalg.svd(image, full_matrices=False)
    ranks = [1, 2, 5, 10, 20, min(image.shape)]

    figure, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    for axis, rank in zip(axes.flat[:6], ranks, strict=True):
        reconstruction, _ = low_rank_approximation(image, rank)
        energy, mse, compression = _rank_metrics(
            image,
            singular_values,
            rank,
            reconstruction,
        )
        axis.imshow(reconstruction, cmap="gray", vmin=0, vmax=1)
        rank_label = "full rank" if rank == min(image.shape) else f"rank {rank}"
        axis.set_title(
            f"{rank_label}\nenergy={energy:.1%} · MSE={mse:.4f}\n"
            f"approx. ratio={compression:.2f}:1",
            fontsize=9.5,
        )
        axis.axis("off")

    decay_axis = axes.flat[6]
    component_index = np.arange(1, singular_values.size + 1)
    decay_axis.semilogy(
        component_index,
        singular_values,
        color=COLORS["blue"],
        linewidth=2,
    )
    decay_axis.set(
        title="Singular-value decay",
        xlabel="Singular value index",
        ylabel="Singular value (log scale)",
    )

    error_axis = axes.flat[7]
    all_ranks = np.arange(1, min(image.shape) + 1)
    total_entries = image.size
    squared_tail = np.cumsum((singular_values[::-1] ** 2))[::-1]
    mse_by_rank = np.array(
        [
            squared_tail[rank] / total_entries
            if rank < singular_values.size
            else 0.0
            for rank in all_ranks
        ]
    )
    error_axis.plot(
        all_ranks,
        mse_by_rank,
        color=COLORS["purple"],
        linewidth=2.2,
    )
    error_axis.set(
        title="Reconstruction error",
        xlabel="Retained rank",
        ylabel="MSE",
    )

    figure.suptitle("Truncated SVD Reveals the Image's Low-Rank Structure")
    figure.text(
        0.5,
        0.015,
        (
            "Synthetic NumPy image. Compression ratio estimates dense values "
            "stored in Uₖ, Σₖ, and Vₖᵀ—not a universal file benchmark."
        ),
        ha="center",
        color=COLORS["gray"],
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.95))
    return save_matplotlib_figure(figure, output_path)


def low_rank_slider_figure(image: np.ndarray) -> go.Figure:
    """Build an interactive set of low-rank image reconstructions."""
    left, singular_values, right_t = np.linalg.svd(image, full_matrices=False)
    max_rank = min(image.shape)
    ranks = sorted({1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, max_rank})
    ranks = [rank for rank in ranks if rank <= max_rank]

    reconstructions = []
    metrics = []
    for rank in ranks:
        reconstruction = (
            left[:, :rank] * singular_values[:rank]
        ) @ right_t[:rank]
        reconstructions.append(reconstruction)
        metrics.append(_rank_metrics(image, singular_values, rank, reconstruction))

    traces = []
    for index, (rank, reconstruction, metric) in enumerate(
        zip(ranks, reconstructions, metrics, strict=True)
    ):
        energy, mse, compression = metric
        traces.append(
            go.Heatmap(
                z=reconstruction,
                colorscale="Gray",
                zmin=0,
                zmax=1,
                showscale=False,
                visible=index == 0,
                hovertemplate="row=%{y}<br>column=%{x}<br>intensity=%{z:.3f}<extra></extra>",
                name=f"rank {rank}",
            )
        )

    steps = []
    for index, (rank, metric) in enumerate(zip(ranks, metrics, strict=True)):
        energy, mse, compression = metric
        visible = [False] * len(traces)
        visible[index] = True
        steps.append(
            {
                "label": str(rank),
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": (
                            f"Rank {rank} SVD Reconstruction"
                            f"<br><sup>energy={energy:.1%} · MSE={mse:.5f} · "
                            f"approx. dense-value ratio={compression:.2f}:1</sup>"
                        )
                    },
                ],
            }
        )

    first_energy, first_mse, first_compression = metrics[0]
    figure = go.Figure(data=traces)
    figure.update_layout(
        title=(
            f"Rank {ranks[0]} SVD Reconstruction"
            f"<br><sup>energy={first_energy:.1%} · MSE={first_mse:.5f} · "
            f"approx. dense-value ratio={first_compression:.2f}:1</sup>"
        ),
        template="plotly_white",
        xaxis={"visible": False, "scaleanchor": "y"},
        yaxis={"visible": False, "autorange": "reversed"},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Selected rank: "},
                "pad": {"t": 45},
                "steps": steps,
            }
        ],
        margin={"l": 20, "r": 20, "t": 85, "b": 80},
        height=720,
    )
    return figure


def generate_low_rank_interactive(output_path: Path) -> Path:
    """Generate standalone low-rank reconstruction HTML."""
    return write_plotly_html(
        low_rank_slider_figure(synthetic_grayscale_image()),
        output_path,
    )


def generate_pca_svd_equivalence(output_path: Path) -> Path:
    """Compare three PCA implementations numerically and visually."""
    apply_matplotlib_style()
    data = correlated_features(n_features=8)
    n_components = data.shape[1]
    eigen_result = eigendecomposition_pca(data, n_components)
    svd_result = svd_pca(data, n_components)
    sklearn_model = PCA(n_components=n_components).fit(data)

    svd_components = align_component_signs(
        eigen_result.components,
        svd_result.components,
    )
    sklearn_components = align_component_signs(
        eigen_result.components,
        sklearn_model.components_,
    )
    centered, _ = center_data(data)
    eigen_scores = centered @ eigen_result.components.T
    svd_scores = centered @ svd_components.T
    sklearn_scores = centered @ sklearn_components.T

    eigen_reconstruction = eigen_scores @ eigen_result.components + eigen_result.mean
    svd_reconstruction = svd_scores @ svd_components + svd_result.mean
    sklearn_reconstruction = (
        sklearn_scores @ sklearn_components + sklearn_model.mean_
    )
    cosine_svd = np.abs(
        np.sum(eigen_result.components * svd_components, axis=1)
    )
    cosine_sklearn = np.abs(
        np.sum(eigen_result.components * sklearn_components, axis=1)
    )
    reconstruction_differences = [
        np.max(np.abs(eigen_reconstruction - svd_reconstruction)),
        np.max(np.abs(eigen_reconstruction - sklearn_reconstruction)),
        np.max(np.abs(svd_reconstruction - sklearn_reconstruction)),
    ]

    count = np.arange(1, n_components + 1)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    width = 0.25
    axes[0, 0].bar(
        count - width,
        eigen_result.explained_variance,
        width,
        label="Covariance eigh",
        color=COLORS["blue"],
    )
    axes[0, 0].bar(
        count,
        svd_result.explained_variance,
        width,
        label="σ² / (n − 1)",
        color=COLORS["orange"],
    )
    axes[0, 0].bar(
        count + width,
        sklearn_model.explained_variance_,
        width,
        label="scikit-learn PCA",
        color=COLORS["cyan"],
    )
    axes[0, 0].set(
        title="Equivalent variance spectra",
        xlabel="Component",
        ylabel="Variance",
        xticks=count,
    )
    axes[0, 0].legend()

    axes[0, 1].plot(
        count,
        cosine_svd,
        marker="o",
        linewidth=2,
        label="eigh vs direct SVD",
        color=COLORS["orange"],
    )
    axes[0, 1].plot(
        count,
        cosine_sklearn,
        marker="s",
        linewidth=2,
        label="eigh vs scikit-learn",
        color=COLORS["cyan"],
    )
    axes[0, 1].set(
        title="Direction agreement after sign alignment",
        xlabel="Component",
        ylabel="Absolute cosine similarity",
        xticks=count,
        ylim=(0.98, 1.002),
    )
    axes[0, 1].legend()

    axes[1, 0].bar(
        ["eigh–SVD", "eigh–sklearn", "SVD–sklearn"],
        reconstruction_differences,
        color=[COLORS["blue"], COLORS["orange"], COLORS["cyan"]],
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(
        title="Maximum reconstruction differences",
        ylabel="Maximum absolute difference (log scale)",
    )

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.5,
        0.62,
        r"$\lambda_i = \frac{\sigma_i^2}{n-1}$",
        ha="center",
        va="center",
        fontsize=28,
        color=COLORS["navy"],
    )
    axes[1, 1].text(
        0.5,
        0.36,
        (
            "Centered data:  X = UΣVᵀ\n"
            "PCA directions: rows of Vᵀ\n"
            "PCA scores: UΣ\n\n"
            "A sign flip changes orientation,\nnot the represented subspace."
        ),
        ha="center",
        va="center",
        fontsize=12,
        color=COLORS["gray"],
    )
    figure.suptitle("PCA and SVD Recover the Same Principal Subspace")
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    print("\nPCA-SVD equivalence validation")
    print("component | covariance lambda | sigma^2/(n-1) | |cos(eigh,SVD)|")
    for index in range(n_components):
        print(
            f"{index + 1:>9} | "
            f"{eigen_result.explained_variance[index]:>12.6f} | "
            f"{svd_result.explained_variance[index]:>10.6f} | "
            f"{cosine_svd[index]:>15.10f}"
        )
    print(
        "maximum reconstruction difference:",
        f"{max(reconstruction_differences):.3e}",
    )
    return save_matplotlib_figure(figure, output_path)


def _first_direction(data: np.ndarray, *, center: bool) -> tuple[np.ndarray, np.ndarray]:
    origin = data.mean(axis=0) if center else np.zeros(2)
    matrix = data - origin if center else data
    second_moment = matrix.T @ matrix / (matrix.shape[0] - 1)
    values, vectors = np.linalg.eigh(second_moment)
    return origin, vectors[:, np.argmax(values)]


def _draw_direction(
    axis: plt.Axes,
    data: np.ndarray,
    *,
    center: bool,
    title: str,
    observation: str,
) -> None:
    origin, direction = _first_direction(data, center=center)
    centered = data - origin
    half_length = max(
        1.0,
        float(np.quantile(np.linalg.norm(centered, axis=1), 0.92)),
    )
    endpoints = np.vstack(
        [origin - direction * half_length, origin + direction * half_length]
    )
    axis.scatter(
        data[:, 0],
        data[:, 1],
        s=13,
        alpha=0.38,
        color=COLORS["blue"],
        edgecolors="none",
    )
    axis.plot(
        endpoints[:, 0],
        endpoints[:, 1],
        color=COLORS["red"],
        linewidth=3,
        label="First direction",
    )
    axis.scatter(*origin, color=COLORS["navy"], marker="X", s=80, zorder=5)
    axis.set_title(title)
    axis.set_xlabel("feature 1")
    axis.set_ylabel("feature 2")
    axis.set_aspect("equal", adjustable="datalim")
    axis.text(
        0.03,
        0.03,
        observation,
        transform=axis.transAxes,
        fontsize=9,
        color=COLORS["navy"],
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none"},
    )


def generate_pca_pitfalls(output_path: Path) -> Path:
    """Compare centering, scale, and outlier effects on the first direction."""
    apply_matplotlib_style()
    base = correlated_2d(n_samples=240, noise=0.24)
    offset = base + np.array([8.0, -6.0])
    scaled = base * np.array([5.0, 1.0])
    outliers = np.vstack([base, [[6.5, -5.5], [7.0, -6.0], [7.5, -6.5]]])

    figure, axes = plt.subplots(2, 2, figsize=(14, 11))
    _draw_direction(
        axes[0, 0],
        base,
        center=True,
        title="1. Correctly centered covariance PCA",
        observation="PC1 follows variation around the sample mean.",
    )
    _draw_direction(
        axes[0, 1],
        offset,
        center=False,
        title="2. Large offset without centering",
        observation="The uncentered second moment points toward the offset.",
    )
    _draw_direction(
        axes[1, 0],
        scaled,
        center=True,
        title="3. Incompatible feature scales",
        observation="Multiplying feature 1 by 5 rotates PC1 toward that axis.",
    )
    _draw_direction(
        axes[1, 1],
        outliers,
        center=True,
        title="4. Influential outliers",
        observation="Squared deviations give extreme points high leverage.",
    )
    figure.suptitle(
        "PCA Directions Depend on Preprocessing and the Fitting Distribution"
    )
    figure.text(
        0.5,
        0.02,
        (
            "Centering defines variation around the mean. Standardization is a "
            "domain choice; outliers require investigation, not automatic removal."
        ),
        ha="center",
        color=COLORS["gray"],
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.95))
    return save_matplotlib_figure(figure, output_path)
