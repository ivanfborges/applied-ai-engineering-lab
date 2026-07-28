"""Generate self-contained Plotly HTML views for the visualization lab."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from visualization_utils import (
    DEFAULT_RANDOM_SEED,
    create_synthetic_embedding_clusters,
    ensure_output_directories,
    nearest_neighbors_by_cosine,
    project_onto_subspace,
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GRAY = "#6B7280"
FLOAT_ARRAY = NDArray[np.float64]


def vector_trace(
    start: FLOAT_ARRAY,
    vector: FLOAT_ARRAY,
    *,
    name: str,
    color: str,
    width: int = 7,
    dash: str = "solid",
    visible: bool | str = True,
) -> go.Scatter3d:
    """Create a 3D line with a terminal marker to represent a vector."""
    endpoint = start + vector
    return go.Scatter3d(
        x=[start[0], endpoint[0]],
        y=[start[1], endpoint[1]],
        z=[start[2], endpoint[2]],
        mode="lines+markers",
        line={"color": color, "width": width, "dash": dash},
        marker={"color": color, "size": [2, 6], "symbol": ["circle", "diamond"]},
        name=name,
        visible=visible,
        hovertemplate=(
            f"{name}<br>"
            f"endpoint=({endpoint[0]:.3f}, {endpoint[1]:.3f}, {endpoint[2]:.3f})"
            "<extra></extra>"
        ),
    )


def professional_layout(figure: go.Figure, title: str) -> None:
    """Apply consistent labels, aspect ratio, and interaction defaults."""
    figure.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 90, "b": 10},
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.85)"},
        scene={
            "xaxis_title": "x₁",
            "yaxis_title": "x₂",
            "zaxis_title": "x₃",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.15}},
        },
    )


def save_html(figure: go.Figure, output_path: Path) -> None:
    """Write a full HTML document with Plotly JavaScript embedded."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )
    print(f"Saved self-contained HTML: {output_path}")


def projection_plane_3d(output_path: Path) -> None:
    """Create an interactive orthogonal projection onto a plane."""
    basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.45, 0.25]])
    vector = np.array([1.4, 1.2, 2.3])
    projected = project_onto_subspace(vector, basis)
    residual = vector - projected
    assert np.allclose(basis.T @ residual, 0.0, atol=1e-8)

    grid = np.linspace(-2.4, 2.4, 18)
    first, second = np.meshgrid(grid, grid)
    surface = first[..., None] * basis[:, 0] + second[..., None] * basis[:, 1]
    figure = go.Figure(
        data=[
            go.Surface(
                x=surface[:, :, 0],
                y=surface[:, :, 1],
                z=surface[:, :, 2],
                name="plane span(b₁, b₂)",
                colorscale=[[0, SKY], [1, SKY]],
                opacity=0.35,
                showscale=False,
                hovertemplate="target plane<extra></extra>",
            ),
            vector_trace(np.zeros(3), basis[:, 0], name="basis b₁", color=BLUE),
            vector_trace(np.zeros(3), basis[:, 1], name="basis b₂", color=ORANGE),
            vector_trace(np.zeros(3), vector, name="original x", color=RED),
            vector_trace(
                np.zeros(3), projected, name="projection Px", color=GREEN
            ),
            vector_trace(
                projected,
                residual,
                name="residual x − Px",
                color=PURPLE,
                dash="dash",
            ),
        ]
    )
    professional_layout(figure, "Interactive Projection onto a Plane")
    figure.add_annotation(
        text=(
            f"b₁ᵀr = {basis[:, 0] @ residual:.2e}, "
            f"b₂ᵀr = {basis[:, 1] @ residual:.2e}<br>"
            "Rotate, zoom, and toggle traces in the legend."
        ),
        x=0.5,
        y=0.01,
        xref="paper",
        yref="paper",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
    )
    save_html(figure, output_path)


def correlated_points(
    sample_count: int = 130, random_seed: int = DEFAULT_RANDOM_SEED
) -> FLOAT_ARRAY:
    """Create deterministic correlated 3D data for interactive PCA."""
    rng = np.random.default_rng(random_seed)
    latent = rng.normal(size=(sample_count, 3))
    transform = np.array([[2.4, 0.2, 0.05], [1.5, 1.1, 0.08], [0.9, -0.6, 0.16]])
    return latent @ transform.T + np.array([0.4, -0.2, 0.6])


def pca_subspace_3d(output_path: Path) -> None:
    """Create an interactive PCA plane with toggleable original projections."""
    points = correlated_points()
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(points)
    projected = pca.inverse_transform(coordinates)
    assert projected.shape == points.shape
    reconstruction_error = float(np.mean(np.square(points - projected)))

    extent = 3.0 * np.sqrt(pca.explained_variance_)
    first, second = np.meshgrid(
        np.linspace(-extent[0], extent[0], 16),
        np.linspace(-extent[1], extent[1], 16),
    )
    plane = (
        pca.mean_
        + first[..., None] * pca.components_[0]
        + second[..., None] * pca.components_[1]
    )
    residual_segments: list[float | None] = []
    residual_y: list[float | None] = []
    residual_z: list[float | None] = []
    for index in range(0, len(points), 10):
        residual_segments.extend([points[index, 0], projected[index, 0], None])
        residual_y.extend([points[index, 1], projected[index, 1], None])
        residual_z.extend([points[index, 2], projected[index, 2], None])

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=plane[:, :, 0],
            y=plane[:, :, 1],
            z=plane[:, :, 2],
            name="principal plane",
            colorscale=[[0, ORANGE], [1, ORANGE]],
            opacity=0.25,
            showscale=False,
            hovertemplate="two-dimensional principal subspace<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker={"size": 4, "color": BLUE, "opacity": 0.55, "symbol": "circle"},
            name="original observations",
            hovertemplate="original (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=projected[:, 0],
            y=projected[:, 1],
            z=projected[:, 2],
            mode="markers",
            marker={"size": 4, "color": GREEN, "opacity": 0.72, "symbol": "diamond"},
            name="projected observations",
            hovertemplate="projection (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=residual_segments,
            y=residual_y,
            z=residual_z,
            mode="lines",
            line={"color": GRAY, "width": 2, "dash": "dot"},
            name="sample residuals",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        vector_trace(
            pca.mean_,
            pca.components_[0] * extent[0],
            name=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            color=RED,
        )
    )
    figure.add_trace(
        vector_trace(
            pca.mean_,
            pca.components_[1] * extent[1],
            name=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            color=PURPLE,
        )
    )
    professional_layout(figure, "Interactive PCA Subspace: 3D → 2D")
    figure.add_annotation(
        text=(
            f"Explained variance: {pca.explained_variance_ratio_.sum():.1%} | "
            f"mean squared reconstruction error: {reconstruction_error:.4f}<br>"
            "Use the legend to isolate the original points, projections, plane, and PCs."
        ),
        x=0.5,
        y=0.01,
        xref="paper",
        yref="paper",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.88)",
    )
    save_html(figure, output_path)


def synthetic_embedding_space_3d(output_path: Path) -> None:
    """Create an interactive PCA view of synthetic high-dimensional embeddings."""
    embeddings, labels, query, point_ids = create_synthetic_embedding_clusters()
    neighbor_indices, cosine_scores = nearest_neighbors_by_cosine(
        embeddings, query, count=6
    )
    selected_mask = np.zeros(len(embeddings), dtype=bool)
    selected_mask[neighbor_indices] = True

    combined = np.vstack([embeddings, query])
    pca = PCA(n_components=3)
    coordinates = pca.fit_transform(combined)
    points_3d = coordinates[:-1]
    query_3d = coordinates[-1]
    styles = {
        "Machine learning": (BLUE, "circle"),
        "Cloud infrastructure": (ORANGE, "square"),
        "Legal documents": (GREEN, "diamond"),
    }

    figure = go.Figure()
    for group_name, (color, symbol) in styles.items():
        indices = np.where(labels == group_name)[0]
        custom_data = np.column_stack(
            [
                np.asarray(point_ids, dtype=object)[indices],
                labels[indices],
                cosine_scores[indices],
                np.where(selected_mask[indices], "yes", "no"),
            ]
        )
        figure.add_trace(
            go.Scatter3d(
                x=points_3d[indices, 0],
                y=points_3d[indices, 1],
                z=points_3d[indices, 2],
                mode="markers",
                marker={
                    "size": 5,
                    "color": color,
                    "symbol": symbol,
                    "opacity": 0.78,
                    "line": {"color": color, "width": 0},
                },
                name=group_name,
                customdata=custom_data,
                hovertemplate=(
                    "ID: %{customdata[0]}<br>"
                    "group: %{customdata[1]}<br>"
                    "cosine to query: %{customdata[2]:.4f}<br>"
                    "selected neighbor: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    neighbor_custom_data = np.column_stack(
        [
            np.asarray(point_ids, dtype=object)[neighbor_indices],
            labels[neighbor_indices],
            cosine_scores[neighbor_indices],
        ]
    )
    figure.add_trace(
        go.Scatter3d(
            x=points_3d[neighbor_indices, 0],
            y=points_3d[neighbor_indices, 1],
            z=points_3d[neighbor_indices, 2],
            mode="markers",
            marker={
                "size": 10,
                "color": RED,
                "symbol": "circle-open",
                "line": {"color": RED, "width": 3},
            },
            name="selected nearest neighbors",
            customdata=neighbor_custom_data,
            hovertemplate=(
                "ID: %{customdata[0]}<br>"
                "group: %{customdata[1]}<br>"
                "cosine to query: %{customdata[2]:.4f}<br>"
                "selected neighbor: yes<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter3d(
            x=[query_3d[0]],
            y=[query_3d[1]],
            z=[query_3d[2]],
            mode="markers",
            marker={
                "size": 13,
                "color": RED,
                "symbol": "diamond",
                "line": {"color": "#1F2937", "width": 2},
            },
            name="synthetic query",
            hovertemplate=(
                f"synthetic query<br>source dimension: {embeddings.shape[1]}"
                "<extra></extra>"
            ),
        )
    )
    for rank, index in enumerate(neighbor_indices, start=1):
        neighbor = points_3d[index]
        figure.add_trace(
            go.Scatter3d(
                x=[query_3d[0], neighbor[0]],
                y=[query_3d[1], neighbor[1]],
                z=[query_3d[2], neighbor[2]],
                mode="lines",
                line={"color": RED, "width": 3, "dash": "dash"},
                name=f"neighbor #{rank}",
                legendgroup="neighbor links",
                showlegend=rank == 1,
                hovertemplate=(
                    f"neighbor rank {rank}<br>"
                    f"cosine similarity {cosine_scores[index]:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    professional_layout(figure, "Synthetic High-Dimensional Embedding Space")
    figure.add_annotation(
        text=(
            f"Source vectors: {embeddings.shape[1]}D | display: first 3 PCA components "
            f"({pca.explained_variance_ratio_.sum():.1%} variance)<br>"
            "Nearest neighbors are selected with cosine similarity before PCA."
        ),
        x=0.5,
        y=0.01,
        xref="paper",
        yref="paper",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.88)",
    )
    save_html(figure, output_path)


InteractiveGenerator = Callable[[Path], None]
INTERACTIVE_VIEWS: dict[str, tuple[str, InteractiveGenerator]] = {
    "projection-plane": ("projection_plane_3d.html", projection_plane_3d),
    "pca-subspace": ("pca_subspace_3d.html", pca_subspace_3d),
    "embedding-space": (
        "synthetic_embedding_space_3d.html",
        synthetic_embedding_space_3d,
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for interactive HTML generation."""
    parser = argparse.ArgumentParser(
        description="Generate self-contained Plotly 3D visualizations."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="HTML destination (default: visualizations/outputs/interactive).",
    )
    parser.add_argument(
        "--only",
        choices=sorted(INTERACTIVE_VIEWS),
        help="Generate only one named interactive view.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available interactive view names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate all interactive pages or one selected page."""
    args = parse_args()
    if args.list:
        print("\n".join(sorted(INTERACTIVE_VIEWS)))
        return
    output_directory = (
        args.output_dir
        if args.output_dir is not None
        else ensure_output_directories()["interactive"]
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    selected = (
        {args.only: INTERACTIVE_VIEWS[args.only]}
        if args.only
        else INTERACTIVE_VIEWS
    )
    print(f"Generating {len(selected)} interactive visualization(s)")
    for filename, generator in selected.values():
        generator(output_directory / filename)
    print("Interactive HTML generation complete.")


if __name__ == "__main__":
    main()
