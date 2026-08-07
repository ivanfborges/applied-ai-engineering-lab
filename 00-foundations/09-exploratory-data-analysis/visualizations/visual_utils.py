"""Shared data, statistics, styling, and output helpers for the visual lab."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = TOPIC_DIRECTORY / "outputs"
IMAGE_DIRECTORY = OUTPUT_DIRECTORY / "images"
GIF_DIRECTORY = OUTPUT_DIRECTORY / "gifs"
INTERACTIVE_DIRECTORY = OUTPUT_DIRECTORY / "interactive"

COLORS = {
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "green": "#059669",
    "orange": "#EA580C",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "dark": "#0F172A",
    "light": "#E2E8F0",
    "yellow": "#CA8A04",
}


def ensure_output_directories() -> None:
    """Create all generated-output directories."""
    for directory in (IMAGE_DIRECTORY, GIF_DIRECTORY, INTERACTIVE_DIRECTORY):
        directory.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    """Apply a consistent, restrained technical-portfolio style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 115,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 16,
            "axes.titleweight": "bold",
        }
    )


def save_figure(figure: plt.Figure, output_path: Path) -> Path:
    """Save and close one static figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved: {output_path}")
    return output_path


def save_animation(
    figure: plt.Figure,
    update: Callable[[int], None],
    frame_count: int,
    output_path: Path,
    *,
    fps: int = 6,
    dpi: int = 90,
) -> Path:
    """Render a bounded Matplotlib animation as a GitHub-friendly GIF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 / fps,
        repeat=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
    print(f"Saved: {output_path}")
    return output_path


def unscaled_mad(values: np.ndarray) -> float:
    """Return the unscaled median absolute deviation."""
    array = np.asarray(values, dtype=float)
    center = float(np.median(array))
    return float(np.median(np.abs(array - center)))


def sample_skewness(values: np.ndarray) -> float:
    """Return pandas' bias-corrected sample skewness."""
    return float(pd.Series(np.asarray(values, dtype=float)).skew())


def sample_excess_kurtosis(values: np.ndarray) -> float:
    """Return pandas' unbiased sample excess kurtosis."""
    return float(pd.Series(np.asarray(values, dtype=float)).kurt())


def synthetic_ai_workload(
    sample_size: int = 1_800,
    seed: int = SEED,
) -> pd.DataFrame:
    """Create reproducible synthetic AI request telemetry.

    The magnitudes are educational and are not benchmark measurements.
    Relationships are introduced deliberately so the EDA panels have known
    structure to reveal.
    """
    if sample_size < 100:
        raise ValueError("sample_size must be at least 100.")
    rng = np.random.default_rng(seed)

    request_type = rng.choice(
        ["RAG Q&A", "Summarization", "Extraction"],
        size=sample_size,
        p=[0.48, 0.30, 0.22],
    )
    model = rng.choice(
        ["compact", "capable"],
        size=sample_size,
        p=[0.62, 0.38],
    )

    request_input_multiplier = np.select(
        [
            request_type == "RAG Q&A",
            request_type == "Summarization",
        ],
        [1.0, 2.4],
        default=1.4,
    )
    input_tokens = rng.lognormal(6.8, 0.55, sample_size)
    input_tokens = np.rint(input_tokens * request_input_multiplier).astype(int)

    output_shape = np.select(
        [
            request_type == "RAG Q&A",
            request_type == "Summarization",
        ],
        [2.6, 4.0],
        default=1.8,
    )
    output_tokens = np.rint(
        rng.gamma(shape=output_shape, scale=75.0)
    ).astype(int)
    output_tokens = np.maximum(output_tokens, 8)

    document_tokens = np.rint(
        input_tokens * rng.uniform(1.5, 5.0, sample_size)
    ).astype(int)
    chunks_retrieved = np.maximum(
        1,
        np.rint(document_tokens / 900 + rng.normal(0, 1.3, sample_size)),
    ).astype(int)
    chunks_retrieved = np.minimum(chunks_retrieved, 30)

    retrieval_location = np.select(
        [
            request_type == "RAG Q&A",
            request_type == "Summarization",
        ],
        [0.76, 0.57],
        default=0.66,
    )
    retrieval_score = np.clip(
        retrieval_location + rng.normal(0, 0.09, sample_size),
        0.05,
        0.99,
    )

    model_latency = np.where(model == "capable", 210.0, 85.0)
    latency_ms = (
        model_latency
        + 0.055 * input_tokens
        + 0.32 * output_tokens
        + 8.0 * chunks_retrieved
        + rng.lognormal(3.5, 0.50, sample_size)
    )
    spike_indices = rng.choice(sample_size, size=max(5, sample_size // 100), replace=False)
    latency_ms[spike_indices] += rng.lognormal(7.0, 0.45, len(spike_indices))

    input_rate = np.where(model == "capable", 3.0e-6, 0.8e-6)
    output_rate = np.where(model == "capable", 12.0e-6, 3.0e-6)
    cost_usd = input_tokens * input_rate + output_tokens * output_rate

    return pd.DataFrame(
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "document_tokens": document_tokens,
            "latency_ms": latency_ms,
            "retrieval_score": retrieval_score,
            "chunks_retrieved": chunks_retrieved,
            "cost_usd": cost_usd,
            "model": model,
            "request_type": request_type,
        }
    )


def simpsons_paradox_data(
    observations_per_group: int = 55,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate groups with negative within-group and positive overall trends."""
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for label, center in zip(("A", "B", "C"), (2.0, 5.0, 8.0), strict=True):
        x = rng.normal(center, 0.65, observations_per_group)
        y = 2.0 * center - 0.70 * x + rng.normal(
            0.0, 0.32, observations_per_group
        )
        frames.append(pd.DataFrame({"x": x, "y": y, "group": label}))
    return pd.concat(frames, ignore_index=True)


def anscombe_quartet() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return the four classic Anscombe datasets without network access."""
    x_common = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
    datasets = [
        (
            "I: approximately linear",
            x_common,
            np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
        ),
        (
            "II: curved",
            x_common,
            np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
        ),
        (
            "III: one influential y",
            x_common,
            np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
        ),
        (
            "IV: one influential x",
            np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=float),
            np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
        ),
    ]
    return datasets
