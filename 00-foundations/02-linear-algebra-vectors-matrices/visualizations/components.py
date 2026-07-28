"""Plotly figure builders shared by the Streamlit application."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import math_utils

BLUE = "#2563EB"
ORANGE = "#F97316"
GREEN = "#16A34A"
PURPLE = "#7C3AED"
RED = "#DC2626"
GRAY = "#64748B"


def base_plane(title: str, limit: float = 6.0) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        template="plotly_white",
        height=590,
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    figure.update_xaxes(
        range=[-limit, limit],
        zeroline=True,
        zerolinewidth=2,
        gridcolor="#E2E8F0",
        constrain="domain",
        title="x",
    )
    figure.update_yaxes(
        range=[-limit, limit],
        zeroline=True,
        zerolinewidth=2,
        gridcolor="#E2E8F0",
        scaleanchor="x",
        scaleratio=1,
        title="y",
    )
    return figure


def add_vector(
    figure: go.Figure,
    vector: np.ndarray,
    name: str,
    color: str,
    origin: np.ndarray | None = None,
    dash: str = "solid",
) -> None:
    start = np.zeros(2) if origin is None else np.asarray(origin, dtype=float)
    endpoint = start + np.asarray(vector, dtype=float)
    figure.add_trace(
        go.Scatter(
            x=[start[0], endpoint[0]],
            y=[start[1], endpoint[1]],
            mode="lines+markers",
            line=dict(color=color, width=4, dash=dash),
            marker=dict(size=[5, 10], color=color),
            name=name,
            hovertemplate=(
                f"{name}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>"
            ),
        )
    )
    figure.add_annotation(
        x=endpoint[0],
        y=endpoint[1],
        ax=start[0],
        ay=start[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor=color,
        text="",
    )


def vector_arithmetic_figure(
    a: np.ndarray,
    b: np.ndarray,
    operation: str,
    scalar: float,
) -> tuple[go.Figure, np.ndarray]:
    if operation == "Addition: a + b":
        result = a + b
        expression = "a + b"
        translated = b
    elif operation == "Subtraction: a - b":
        result = a - b
        expression = "a - b"
        translated = -b
    else:
        result = scalar * a
        expression = f"{scalar:.1f}a"
        translated = None

    figure = base_plane(f"Vector arithmetic: {expression}")
    add_vector(figure, a, "a", BLUE)
    if operation != "Scalar multiplication: ca":
        add_vector(figure, b, "b", ORANGE)
    add_vector(figure, result, expression, GREEN)
    if translated is not None:
        add_vector(figure, translated, "translated component", GRAY, origin=a, dash="dash")
    return figure, result


def normalization_figure(vector: np.ndarray) -> go.Figure:
    figure = base_plane("Normalization preserves direction", limit=5.0)
    theta = np.linspace(0, 2 * np.pi, 240)
    figure.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            line=dict(color="#CBD5E1", dash="dot", width=2),
            name="unit circle",
        )
    )
    add_vector(figure, vector, "a", BLUE)
    if math_utils.vector_norm(vector) > math_utils.EPSILON:
        add_vector(figure, math_utils.normalize(vector), "normalized a", GREEN)
    return figure


def similarity_projection_figure(
    a: np.ndarray, b: np.ndarray
) -> tuple[go.Figure, dict[str, float | np.ndarray | bool]]:
    figure = base_plane("Dot product, angle, orthogonality, and projection")
    add_vector(figure, a, "a", BLUE)
    add_vector(figure, b, "b", ORANGE)

    dot = math_utils.dot_product(a, b)
    norm_a = math_utils.vector_norm(a)
    norm_b = math_utils.vector_norm(b)
    metrics: dict[str, float | np.ndarray | bool] = {
        "dot": dot,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "orthogonal": bool(abs(dot) < 1e-9 and norm_a > 0 and norm_b > 0),
    }
    if norm_a > math_utils.EPSILON and norm_b > math_utils.EPSILON:
        cosine = math_utils.cosine_similarity(a, b)
        angle = math_utils.angle_degrees(a, b)
        projected = math_utils.projection(a, b)
        metrics.update(cosine=cosine, angle=angle, projection=projected)
        add_vector(figure, projected, "projection of a onto b", GREEN)
        figure.add_trace(
            go.Scatter(
                x=[a[0], projected[0]],
                y=[a[1], projected[1]],
                mode="lines",
                line=dict(color=GREEN, dash="dot", width=2),
                name="orthogonal residual",
            )
        )
    return figure, metrics


def norms_and_distances_figure(
    p: np.ndarray, q: np.ndarray
) -> tuple[go.Figure, dict[str, float]]:
    figure = base_plane("Norm geometry and point-to-point distance")
    theta = np.linspace(0, 2 * np.pi, 240)
    figure.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            line=dict(color=GREEN, width=2),
            name="L2 unit ball",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[1, 0, -1, 0, 1],
            y=[0, 1, 0, -1, 0],
            mode="lines",
            line=dict(color=PURPLE, width=2),
            name="L1 unit ball",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[-1, 1, 1, -1, -1],
            y=[-1, -1, 1, 1, -1],
            mode="lines",
            line=dict(color=ORANGE, width=2),
            name="L∞ unit ball",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[p[0], q[0]],
            y=[p[1], q[1]],
            mode="markers+text",
            text=["p", "q"],
            textposition="top center",
            marker=dict(size=13, color=[BLUE, RED]),
            name="points",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[p[0], q[0]],
            y=[p[1], q[1]],
            mode="lines",
            line=dict(color=GREEN, width=4),
            name="Euclidean path",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[p[0], q[0], q[0]],
            y=[p[1], p[1], q[1]],
            mode="lines",
            line=dict(color=RED, width=3, dash="dash"),
            name="one Manhattan path",
        )
    )
    return figure, {
        "l1": math_utils.distance(p, q, 1),
        "l2": math_utils.distance(p, q, 2),
        "linf": math_utils.distance(p, q, np.inf),
    }


def transformation_figure(
    matrix: np.ndarray,
    title: str,
) -> go.Figure:
    figure = base_plane(title, limit=7.0)
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
    transformed_square = math_utils.apply_transformation(square, matrix)
    grid = np.arange(-3.0, 4.0)

    for value in grid:
        for line in (
            np.array([[-3.0, value], [3.0, value]]),
            np.array([[value, -3.0], [value, 3.0]]),
        ):
            transformed = math_utils.apply_transformation(line, matrix)
            figure.add_trace(
                go.Scatter(
                    x=transformed[:, 0],
                    y=transformed[:, 1],
                    mode="lines",
                    line=dict(color="#BFDBFE", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    figure.add_trace(
        go.Scatter(
            x=square[:, 0],
            y=square[:, 1],
            mode="lines",
            line=dict(color=GRAY, dash="dash", width=2),
            name="original square",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=transformed_square[:, 0],
            y=transformed_square[:, 1],
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.22)",
            line=dict(color=BLUE, width=3),
            name="transformed square",
        )
    )
    add_vector(figure, matrix @ np.array([1.0, 0.0]), "T(e1)", RED)
    add_vector(figure, matrix @ np.array([0.0, 1.0]), "T(e2)", GREEN)
    return figure


def transformation_order_figure(
    first_order: np.ndarray,
    second_order: np.ndarray,
    labels: tuple[str, str],
) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=labels,
        horizontal_spacing=0.08,
    )
    triangle = np.array([[0.0, 0.0], [2.0, 0.0], [0.5, 1.5], [0.0, 0.0]])
    colors = [BLUE, ORANGE]
    for column, (matrix, color) in enumerate(
        zip((first_order, second_order), colors), start=1
    ):
        transformed = math_utils.apply_transformation(triangle, matrix)
        figure.add_trace(
            go.Scatter(
                x=triangle[:, 0],
                y=triangle[:, 1],
                mode="lines",
                line=dict(color=GRAY, dash="dash"),
                name="original" if column == 1 else None,
                showlegend=column == 1,
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=transformed[:, 0],
                y=transformed[:, 1],
                fill="toself",
                line=dict(color=color, width=3),
                name=labels[column - 1],
            ),
            row=1,
            col=column,
        )
    figure.update_xaxes(range=[-5, 5], zeroline=True, gridcolor="#E2E8F0")
    figure.update_yaxes(
        range=[-5, 5],
        zeroline=True,
        gridcolor="#E2E8F0",
        scaleanchor="x",
        scaleratio=1,
    )
    figure.update_layout(
        template="plotly_white",
        height=520,
        title="Composition order: the rightmost matrix acts first",
        margin=dict(l=30, r=30, t=70, b=30),
    )
    return figure


def feature_scaling_figure(
    raw_points: np.ndarray,
    raw_query: np.ndarray,
    scaled_points: np.ndarray,
    scaled_query: np.ndarray,
) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Raw feature space", "Standardized feature space"),
    )
    for column, (points, query) in enumerate(
        ((raw_points, raw_query), (scaled_points, scaled_query)), start=1
    ):
        distances = np.linalg.norm(points - query, axis=1)
        nearest = int(np.argmin(distances))
        figure.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers+text",
                text=[f"P{i}" for i in range(len(points))],
                textposition="top center",
                marker=dict(size=12, color=BLUE),
                name="synthetic observations",
                showlegend=column == 1,
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[query[0]],
                y=[query[1]],
                mode="markers",
                marker=dict(size=16, color=RED, symbol="star"),
                name="query",
                showlegend=column == 1,
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[query[0], points[nearest, 0]],
                y=[query[1], points[nearest, 1]],
                mode="lines",
                line=dict(color=GREEN, width=4),
                name="nearest",
                showlegend=column == 1,
            ),
            row=1,
            col=column,
        )
    figure.update_layout(
        template="plotly_white",
        height=500,
        title="Feature scale can change nearest-neighbor geometry",
        margin=dict(l=30, r=30, t=70, b=30),
    )
    figure.update_xaxes(title="feature 1")
    figure.update_yaxes(title="feature 2")
    return figure


def embedding_ranking_figure(
    ranking: Sequence[tuple[str, float]],
    metric: str,
) -> go.Figure:
    labels = [item[0] for item in ranking][::-1]
    scores = [item[1] for item in ranking][::-1]
    figure = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker_color=[BLUE if index == len(scores) - 1 else "#93C5FD" for index in range(len(scores))],
            text=[f"{score:.3f}" for score in scores],
            textposition="auto",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=430,
        title=f"Synthetic retrieval ranking by {metric.lower()}",
        xaxis_title=metric,
        yaxis=dict(automargin=True),
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return figure


def concentration_figure(results: dict[str, np.ndarray]) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Mean Euclidean distance",
            "Relative spread of distances",
        ),
    )
    dimensions = results["dimensions"]
    figure.add_trace(
        go.Scatter(
            x=dimensions,
            y=results["mean_distance"],
            mode="lines+markers",
            line=dict(color=BLUE, width=3),
            name="mean distance",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=dimensions,
            y=results["coefficient_of_variation"],
            mode="lines+markers",
            line=dict(color=ORANGE, width=3),
            name="coefficient of variation",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=dimensions,
            y=results["relative_contrast"],
            mode="lines+markers",
            line=dict(color=PURPLE, width=3),
            name="(max - min) / min",
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(type="log", title="dimension")
    figure.update_yaxes(title="distance", row=1, col=1)
    figure.update_yaxes(title="relative spread", row=1, col=2)
    figure.update_layout(
        template="plotly_white",
        height=470,
        title="Distance concentration in synthetic Gaussian spaces",
        margin=dict(l=30, r=30, t=70, b=30),
        legend=dict(orientation="h", y=-0.18),
    )
    return figure
