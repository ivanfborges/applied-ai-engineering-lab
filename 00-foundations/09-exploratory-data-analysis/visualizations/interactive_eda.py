"""Generate the synthetic AI dashboard and standalone Plotly explorations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from visual_utils import (
    COLORS,
    IMAGE_DIRECTORY,
    INTERACTIVE_DIRECTORY,
    configure_matplotlib,
    ensure_output_directories,
    sample_excess_kurtosis,
    sample_skewness,
    save_figure,
    synthetic_ai_workload,
)

import matplotlib.pyplot as plt


def create_ai_workload_dashboard(
    output_path: Path = IMAGE_DIRECTORY / "ai_workload_dashboard.png",
) -> Path:
    """Create a static dashboard that connects EDA to AI telemetry."""
    configure_matplotlib()
    workload = synthetic_ai_workload()
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    model_colors = {"compact": COLORS["blue"], "capable": COLORS["orange"]}

    for model, frame in workload.groupby("model", sort=True):
        axes[0, 0].scatter(
            frame["input_tokens"],
            frame["latency_ms"],
            s=18,
            alpha=0.42,
            color=model_colors[model],
            label=model,
        )
    axes[0, 0].set(
        title="Input tokens generally increase latency",
        xlabel="Input tokens",
        ylabel="Latency (ms)",
        xscale="log",
        yscale="log",
    )
    axes[0, 0].legend(title="Model")

    request_types = sorted(workload["request_type"].unique())
    retrieval_groups = [
        workload.loc[
            workload["request_type"] == request_type,
            "retrieval_score",
        ].to_numpy()
        for request_type in request_types
    ]
    axes[0, 1].boxplot(
        retrieval_groups,
        tick_labels=request_types,
        patch_artist=True,
        boxprops={"facecolor": "#BFDBFE", "alpha": 0.75},
        medianprops={"color": COLORS["red"], "linewidth": 2},
    )
    axes[0, 1].set(
        title="Retrieval-score distributions differ by request type",
        xlabel="Request type",
        ylabel="Synthetic retrieval score",
        ylim=(0, 1),
    )
    axes[0, 1].tick_params(axis="x", rotation=12)

    axes[1, 0].scatter(
        workload["document_tokens"],
        workload["chunks_retrieved"],
        s=18,
        alpha=0.40,
        color=COLORS["purple"],
    )
    axes[1, 0].set(
        title="Larger documents generally create more chunks",
        xlabel="Synthetic document tokens",
        ylabel="Chunks retrieved",
        xscale="log",
    )

    percentiles = [50, 90, 95, 99]
    x_positions = np.arange(len(percentiles))
    width = 0.34
    for index, (model, frame) in enumerate(workload.groupby("model", sort=True)):
        latency_percentiles = np.percentile(frame["latency_ms"], percentiles)
        positions = x_positions + (index - 0.5) * width
        axes[1, 1].bar(
            positions,
            latency_percentiles,
            width=width,
            color=model_colors[model],
            alpha=0.78,
            label=model,
        )
    axes[1, 1].set(
        title="Tail percentiles expose model-specific latency risk",
        xlabel="Latency percentile",
        ylabel="Latency (ms)",
        xticks=x_positions,
        xticklabels=[f"P{percentile}" for percentile in percentiles],
        yscale="log",
    )
    axes[1, 1].legend(title="Model")

    figure.suptitle(
        "Synthetic educational AI workload — not production benchmark data\n"
        f"{len(workload):,} deterministic requests, seed 42",
        y=1.01,
    )
    figure.tight_layout()
    return save_figure(figure, output_path)


def create_ai_workload_3d(
    output_path: Path = INTERACTIVE_DIRECTORY / "ai_workload_3d.html",
) -> Path:
    """Create an offline 3D Plotly view for multivariate request exploration."""
    workload = synthetic_ai_workload()
    figure = px.scatter_3d(
        workload,
        x="input_tokens",
        y="output_tokens",
        z="latency_ms",
        color="request_type",
        symbol="model",
        size="cost_usd",
        size_max=17,
        opacity=0.68,
        hover_data={
            "input_tokens": ":,.0f",
            "output_tokens": ":,.0f",
            "latency_ms": ":,.1f",
            "cost_usd": ":.5f",
            "request_type": True,
            "model": True,
            "document_tokens": ":,.0f",
            "chunks_retrieved": True,
        },
        title=(
            "Synthetic educational AI workload: tokens, latency, cost, and request type"
        ),
    )
    figure.update_layout(
        template="plotly_white",
        scene={
            "xaxis_title": "Input tokens",
            "yaxis_title": "Output tokens",
            "zaxis_title": "Latency (ms)",
            "xaxis": {"type": "log"},
            "zaxis": {"type": "log"},
        },
        legend_title_text="Request type / model",
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        annotations=[
            {
                "text": "Rotate, zoom, pan, and hover. Values are synthetic.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 1.03,
                "showarrow": False,
            }
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )
    print(f"Saved: {output_path}")
    return output_path


def _distribution_samples() -> dict[str, np.ndarray]:
    """Return deterministic samples with comparable scale where appropriate."""
    rng = np.random.default_rng(42)
    return {
        "Normal": rng.normal(0.0, 1.0, 2_500),
        "Log-normal": rng.lognormal(0.0, 0.70, 2_500),
        "Student-t (df=5)": rng.standard_t(5, 2_500) * np.sqrt(3 / 5),
        "Uniform": rng.uniform(-np.sqrt(3), np.sqrt(3), 2_500),
    }


def _statistics_annotation(name: str, values: np.ndarray) -> list[dict[str, object]]:
    """Build a Plotly annotation for one distribution's sample statistics."""
    return [
        {
            "text": (
                f"<b>{name}</b><br>"
                f"mean={np.mean(values):.3f}<br>"
                f"median={np.median(values):.3f}<br>"
                f"population SD={np.std(values, ddof=0):.3f}<br>"
                f"sample skewness={sample_skewness(values):.3f}<br>"
                f"sample excess kurtosis={sample_excess_kurtosis(values):.3f}"
            ),
            "xref": "paper",
            "yref": "paper",
            "x": 0.98,
            "y": 0.95,
            "showarrow": False,
            "align": "left",
            "bgcolor": "rgba(255,255,255,0.92)",
            "bordercolor": "#CBD5E1",
            "borderwidth": 1,
        }
    ]


def create_distribution_explorer(
    output_path: Path = INTERACTIVE_DIRECTORY / "distribution_explorer.html",
) -> Path:
    """Create a dropdown-driven offline distribution comparison."""
    samples = _distribution_samples()
    names = list(samples)
    colors = [
        COLORS["blue"],
        COLORS["orange"],
        COLORS["purple"],
        COLORS["green"],
    ]
    figure = go.Figure()
    for index, (name, values) in enumerate(samples.items()):
        figure.add_trace(
            go.Histogram(
                x=values,
                nbinsx=65,
                histnorm="probability density",
                marker_color=colors[index],
                opacity=0.76,
                name=name,
                visible=index == 0,
                hovertemplate="value=%{x:.3f}<br>density=%{y:.4f}<extra></extra>",
            )
        )

    buttons = []
    for selected_index, name in enumerate(names):
        visibility = [
            index == selected_index for index in range(len(names))
        ]
        buttons.append(
            {
                "label": name,
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {
                        "title": f"{name}: shape, spread, and tail behavior",
                        "annotations": _statistics_annotation(
                            name, samples[name]
                        ),
                    },
                ],
            }
        )

    first_name = names[0]
    figure.update_layout(
        title=f"{first_name}: shape, spread, and tail behavior",
        template="plotly_white",
        xaxis_title="Observed value",
        yaxis_title="Probability density",
        bargap=0.03,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.02,
                "y": 1.13,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        annotations=_statistics_annotation(first_name, samples[first_name]),
        margin={"t": 100},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )
    print(f"Saved: {output_path}")
    return output_path


def generate_interactive_visuals() -> list[Path]:
    """Generate the AI dashboard and standalone HTML artifacts."""
    ensure_output_directories()
    return [
        create_ai_workload_dashboard(),
        create_ai_workload_3d(),
        create_distribution_explorer(),
    ]


def main() -> None:
    """Generate this module's outputs independently."""
    generated = generate_interactive_visuals()
    print(f"Generated {len(generated)} dashboard/interactive visualizations.")


if __name__ == "__main__":
    main()
