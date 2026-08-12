"""Standalone Plotly experiments for offline interaction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from visualizations.visual_utils import (
    COLORS,
    SEED,
    biased_selection_probabilities,
    create_population,
    repeated_sample_means,
    save_plotly,
)


def experiment_mse_surface() -> list[Path]:
    """Create a genuine three-dimensional MSE decomposition surface."""
    biases = np.linspace(-3.0, 3.0, 81)
    variances = np.linspace(0.0, 9.0, 81)
    bias_grid, variance_grid = np.meshgrid(biases, variances)
    mse = bias_grid**2 + variance_grid
    figure = go.Figure(
        data=[
            go.Surface(
                x=bias_grid,
                y=variance_grid,
                z=mse,
                colorscale="Viridis",
                colorbar={"title": "MSE"},
                hovertemplate=(
                    "Bias: %{x:.2f}<br>Variance: %{y:.2f}<br>"
                    "MSE: %{z:.2f}<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        title="How do bias and variance combine into mean squared error?",
        template="plotly_white",
        scene={
            "xaxis_title": "Estimator bias",
            "yaxis_title": "Estimator variance",
            "zaxis_title": "MSE = bias² + variance",
            "camera": {"eye": {"x": 1.55, "y": 1.45, "z": 1.05}},
        },
        annotations=[
            {
                "text": "Exact decomposition for squared error; no benchmark data.",
                "x": 0.5,
                "y": -0.08,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
            }
        ],
        margin={"l": 25, "r": 25, "t": 75, "b": 55},
    )
    return [save_plotly(figure, "mse_bias_variance_surface.html")]


def calculate_explorer_scenarios(
    *,
    trials: int = 700,
) -> tuple[float, list[dict[str, object]]]:
    """Precompute random-versus-biased scenarios for offline selection."""
    population = create_population()
    sample_sizes = (30, 100, 500, 2_000)
    selection_weights = (1.0, 3.0, 8.0)
    seed_sequence = np.random.SeedSequence(SEED + 30)
    child_seeds = iter(seed_sequence.spawn(len(sample_sizes) * len(selection_weights) * 2))
    scenarios: list[dict[str, object]] = []
    for sample_size in sample_sizes:
        for selection_weight in selection_weights:
            random_rng = np.random.default_rng(next(child_seeds))
            biased_rng = np.random.default_rng(next(child_seeds))
            random_estimates = repeated_sample_means(
                population.spend,
                sample_size=sample_size,
                trials=trials,
                rng=random_rng,
                replace=True,
            )
            biased_estimates = repeated_sample_means(
                population.spend,
                sample_size=sample_size,
                trials=trials,
                rng=biased_rng,
                probabilities=biased_selection_probabilities(
                    population,
                    premium_selection_weight=selection_weight,
                ),
                replace=True,
            )
            scenarios.append(
                {
                    "sample_size": sample_size,
                    "selection_weight": selection_weight,
                    "random": random_estimates,
                    "biased": biased_estimates,
                }
            )
    return population.true_mean, scenarios


def experiment_sampling_explorer() -> list[Path]:
    """Create a no-server explorer with precomputed reproducible scenarios."""
    true_mean, scenarios = calculate_explorer_scenarios()
    figure = go.Figure()
    for scenario_index, scenario in enumerate(scenarios):
        sample_size = int(scenario["sample_size"])
        selection_weight = float(scenario["selection_weight"])
        random_estimates = np.asarray(scenario["random"])
        biased_estimates = np.asarray(scenario["biased"])
        visible = scenario_index == 0
        figure.add_trace(
            go.Histogram(
                x=random_estimates,
                name=(
                    f"Random: mean={random_estimates.mean():.2f}, "
                    f"SE={random_estimates.std(ddof=0):.2f}"
                ),
                histnorm="probability density",
                opacity=0.58,
                marker_color=COLORS["blue"],
                visible=visible,
                hovertemplate="Estimate: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Histogram(
                x=biased_estimates,
                name=(
                    f"Biased: mean={biased_estimates.mean():.2f}, "
                    f"SE={biased_estimates.std(ddof=0):.2f}"
                ),
                histnorm="probability density",
                opacity=0.58,
                marker_color=COLORS["red"],
                visible=visible,
                hovertemplate="Estimate: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>",
            )
        )

    buttons = []
    for scenario_index, scenario in enumerate(scenarios):
        visibility = [False] * (2 * len(scenarios))
        visibility[2 * scenario_index] = True
        visibility[2 * scenario_index + 1] = True
        sample_size = int(scenario["sample_size"])
        selection_weight = float(scenario["selection_weight"])
        buttons.append(
            {
                "label": f"n={sample_size:,}, premium weight={selection_weight:g}",
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {
                        "title": (
                            "Random vs selection-biased sampling — "
                            f"n={sample_size:,}, premium weight={selection_weight:g}"
                        )
                    },
                ],
            }
        )

    first = scenarios[0]
    figure.update_layout(
        title=(
            "Random vs selection-biased sampling — "
            f"n={int(first['sample_size']):,}, premium weight={float(first['selection_weight']):g}"
        ),
        template="plotly_white",
        barmode="overlay",
        xaxis_title="Estimated population mean",
        yaxis_title="Probability density",
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.22,
                "yanchor": "top",
                "showactive": True,
            }
        ],
        shapes=[
            {
                "type": "line",
                "x0": true_mean,
                "x1": true_mean,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": COLORS["dark"], "dash": "dash", "width": 2},
            }
        ],
        annotations=[
            {
                "text": f"True population mean = {true_mean:.2f}",
                "x": true_mean,
                "y": 1.0,
                "xref": "x",
                "yref": "paper",
                "showarrow": True,
                "arrowhead": 2,
                "ax": -70,
                "ay": -35,
            },
            {
                "text": (
                    "Synthetic, 700 repeated samples per scenario, sampling with replacement. "
                    "Choose a precomputed configuration; no server is required."
                ),
                "x": 0.5,
                "y": -0.14,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
            },
        ],
        margin={"l": 65, "r": 30, "t": 130, "b": 90},
    )
    return [save_plotly(figure, "sampling_explorer.html")]


def generate_interactive_experiments() -> list[Path]:
    """Generate both self-contained interactive HTML experiments."""
    generated: list[Path] = []
    for generator in (experiment_mse_surface, experiment_sampling_explorer):
        generated.extend(generator())
    return generated
