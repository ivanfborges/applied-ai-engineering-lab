"""Standalone Plotly experiments for the Day 11 laboratory."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .visual_utils import AssetResult, ensure_asset_directory


def generate_standard_error_surface(*, quick: bool = False) -> AssetResult:
    """Render SE(n, sigma) as an interactive, rotatable surface."""
    n_points = 32 if quick else 64
    sigma_points = 24 if quick else 48
    sample_sizes = np.geomspace(1.0, 2_000.0, n_points)
    population_sd = np.linspace(0.5, 5.0, sigma_points)
    n_grid, sigma_grid = np.meshgrid(sample_sizes, population_sd)
    standard_error = sigma_grid / np.sqrt(n_grid)

    figure = go.Figure(
        data=[
            go.Surface(
                x=n_grid,
                y=sigma_grid,
                z=standard_error,
                colorscale="Viridis",
                colorbar={"title": "SE"},
                hovertemplate=(
                    "n=%{x:.0f}<br>sigma=%{y:.2f}<br>SE=%{z:.4f}<extra></extra>"
                ),
                contours={
                    "z": {
                        "show": True,
                        "usecolormap": True,
                        "highlightcolor": "white",
                        "project_z": True,
                    }
                },
            )
        ]
    )
    figure.update_layout(
        title="Standard error depends on both sample size and population variability",
        template="plotly_white",
        scene={
            "xaxis": {
                "title": "Sample size n",
                "type": "log",
                "tickvals": [1, 10, 100, 1_000, 2_000],
                "ticktext": ["1", "10", "100", "1,000", "2,000"],
            },
            "yaxis": {"title": "Population SD sigma"},
            "zaxis": {"title": "SE = sigma / sqrt(n)"},
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 0.95}},
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 70},
    )
    output = ensure_asset_directory() / "04_standard_error_surface.html"
    figure.write_html(
        output,
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    return AssetResult(
        "Interactive standard error surface",
        output,
        (
            "Surface domain: n=1..2,000; sigma=0.5..5.0",
            f"Grid={sigma_points}x{n_points}; SE range=[{standard_error.min():.4f}, {standard_error.max():.4f}]",
        ),
    )
