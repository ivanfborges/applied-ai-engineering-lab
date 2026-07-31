"""Shared output, styling, and saving helpers for the visual explorer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import Animation, PillowWriter
from matplotlib.figure import Figure
from plotly.graph_objects import Figure as PlotlyFigure


@dataclass(frozen=True)
class RenderConfig:
    """Rendering controls shared by all visualization groups."""

    dpi: int = 160
    frames: int = 90
    show: bool = False
    generate_gifs: bool = True
    fps: int = 15


@dataclass(frozen=True)
class OutputPaths:
    """Resolved locations for static, animated, and interactive artifacts."""

    root: Path
    static: Path
    animations: Path
    interactive: Path

    @classmethod
    def create(cls, root: Path) -> "OutputPaths":
        """Create the expected output tree and return its resolved paths."""
        root = root.expanduser().resolve()
        paths = cls(
            root=root,
            static=root / "static",
            animations=root / "animations",
            interactive=root / "interactive",
        )
        for directory in (
            paths.root,
            paths.static,
            paths.animations,
            paths.interactive,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return paths


def configure_plot_style() -> None:
    """Apply restrained, readable defaults across the Matplotlib figures."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 110,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "font.size": 10,
            "legend.frameon": True,
            "lines.linewidth": 2.0,
        }
    )


def safe_filename(name: str, suffix: str) -> str:
    """Convert a descriptive artifact name into a portable file name."""
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()
    if not normalized:
        raise ValueError("Artifact name must contain at least one safe character.")
    return f"{normalized}{suffix}"


def save_figure(
    figure: Figure,
    path: Path,
    config: RenderConfig,
) -> Path:
    """Save and close a Matplotlib figure, optionally displaying it first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=config.dpi, facecolor="white")
    print(f"  created {path}")
    if config.show:
        plt.show()
    plt.close(figure)
    return path


def save_animation(
    animation: Animation,
    figure: Figure,
    path: Path,
    config: RenderConfig,
) -> Path:
    """Save an animation with Pillow so no external encoder is required."""
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=config.fps), dpi=config.dpi)
    print(f"  created {path}")
    if config.show:
        plt.show()
    plt.close(figure)
    return path


def save_plotly_html(figure: PlotlyFigure, path: Path) -> Path:
    """Write a self-contained Plotly document that works offline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        path,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )
    print(f"  created {path}")
    return path


def topic_output_paths() -> OutputPaths:
    """Return the default output tree relative to the topic directory."""
    topic_directory = Path(__file__).resolve().parents[1]
    return OutputPaths.create(topic_directory / "outputs")


def print_group_header(title: str) -> None:
    """Print one concise progress heading."""
    print(f"\n[{title}]")


def artifact_counts(paths: list[Path]) -> tuple[int, int, int]:
    """Count PNG, GIF, and HTML artifacts."""
    static_count = sum(path.suffix.lower() == ".png" for path in paths)
    animation_count = sum(path.suffix.lower() == ".gif" for path in paths)
    interactive_count = sum(path.suffix.lower() == ".html" for path in paths)
    return static_count, animation_count, interactive_count


def main() -> None:
    """Create the default output directory tree."""
    paths = topic_output_paths()
    print(f"Output directories ready under: {paths.root}")


if __name__ == "__main__":
    main()
