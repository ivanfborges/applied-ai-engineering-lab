"""Command-line entry point for the calculus visual exploration."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REQUIRED_IMPORTS = {
    "numpy": "NumPy",
    "matplotlib": "Matplotlib",
    "plotly": "Plotly",
    "PIL": "Pillow",
}

VISUALIZATION_MODULES = {
    "derivative": "visualizations.derivative_visualization",
    "partials": "visualizations.partial_derivatives_visualization",
    "gradient-descent": "visualizations.gradient_descent_visualization",
    "chain-rule": "visualizations.chain_rule_visualization",
    "stability": "visualizations.gradient_stability_visualization",
    "numerical": "visualizations.numerical_gradient_visualization",
    "activations": "visualizations.activation_derivatives_visualization",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate static, animated, and interactive calculus visualizations."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--all",
        action="store_true",
        help="Generate every visualization at full quality (the default).",
    )
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Generate every group with fewer frames and a lower default DPI.",
    )
    mode.add_argument(
        "--only",
        choices=tuple(VISUALIZATION_MODULES),
        help="Generate one visualization group.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display Matplotlib figures after saving them.",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip GIF generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output root. Defaults to this topic's outputs directory.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        help="PNG and GIF resolution. Defaults to 120 quick or 160 full.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        help="Frames per animation. Defaults to 40 quick or 90 full.",
    )
    return parser


def missing_dependencies() -> list[str]:
    """Return human-readable names for unavailable runtime dependencies."""
    return [
        display_name
        for module_name, display_name in REQUIRED_IMPORTS.items()
        if importlib.util.find_spec(module_name) is None
    ]


def load_visualization_module(module_name: str) -> ModuleType:
    """Import one visualization module after dependency validation."""
    return importlib.import_module(module_name)


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    """Reject values that cannot produce meaningful artifacts."""
    if args.dpi is not None and args.dpi < 72:
        parser.error("--dpi must be at least 72.")
    if args.frames is not None and args.frames < 12:
        parser.error("--frames must be at least 12.")


def selected_groups(args: argparse.Namespace) -> list[str]:
    """Return either the requested group or every registered group."""
    if args.only:
        return [args.only]
    return list(VISUALIZATION_MODULES)


def main() -> int:
    """Generate the selected artifacts and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)

    missing = missing_dependencies()
    if missing:
        print(
            "Missing required dependencies: "
            + ", ".join(missing)
            + "\nInstall them with: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    if not args.show:
        # Select a file-only backend before any module imports pyplot. This
        # keeps normal generation independent of Tk and other GUI toolkits.
        import matplotlib

        matplotlib.use("Agg")

    # Imports are intentionally delayed so dependency failures remain concise.
    from visualizations.utils import (
        OutputPaths,
        RenderConfig,
        artifact_counts,
    )

    topic_directory = Path(__file__).resolve().parent
    output_root = (
        args.output_dir
        if args.output_dir is not None
        else topic_directory / "outputs"
    )
    output_paths = OutputPaths.create(output_root)
    frame_count = (
        args.frames if args.frames is not None else (40 if args.quick else 90)
    )
    dpi = args.dpi if args.dpi is not None else (120 if args.quick else 160)
    config = RenderConfig(
        dpi=dpi,
        frames=frame_count,
        show=args.show,
        generate_gifs=not args.no_gif,
        fps=15,
    )

    artifacts: list[Path] = []
    try:
        for group in selected_groups(args):
            module = load_visualization_module(VISUALIZATION_MODULES[group])
            generated = module.generate(output_paths, config)
            artifacts.extend(generated)
    except Exception as error:
        print(
            f"\nVisualization generation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    static_count, animation_count, interactive_count = artifact_counts(artifacts)
    print("\nVisual exploration completed.")
    print(f"\nStatic figures: {static_count}")
    print(f"Animations: {animation_count}")
    print(f"Interactive HTML files: {interactive_count}")
    print(f"\nOutput directory:\n{output_paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
