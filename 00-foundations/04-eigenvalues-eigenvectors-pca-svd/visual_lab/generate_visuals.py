"""Command-line entrypoint for generating visual laboratory assets."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from .animations import generate_eigenvector_animation, generate_svd_animation
from .plotting import (
    generate_covariance_pca,
    generate_explained_variance,
    generate_low_rank_interactive,
    generate_low_rank_static,
    generate_pca_3d,
    generate_pca_pitfalls,
    generate_pca_svd_equivalence,
    generate_projection_3d_html,
    generate_projection_comparison,
)

TOPIC_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = TOPIC_ROOT / "outputs"
STATIC_DIR = OUTPUT_ROOT / "static"
ANIMATION_DIR = OUTPUT_ROOT / "animations"
INTERACTIVE_DIR = OUTPUT_ROOT / "interactive"


def ensure_output_directories(output_root: Path = OUTPUT_ROOT) -> tuple[Path, ...]:
    """Create and return the static, animation, and interactive directories."""
    directories = (
        output_root / "static",
        output_root / "animations",
        output_root / "interactive",
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def expected_output_paths(output_root: Path = OUTPUT_ROOT) -> tuple[Path, ...]:
    """Return every artifact produced by the --all command."""
    return (
        output_root / "animations" / "eigenvectors_transformation.gif",
        output_root / "static" / "eigenvectors_transformation.png",
        output_root / "static" / "covariance_and_principal_components.png",
        output_root / "interactive" / "pca_3d_axes.html",
        output_root / "interactive" / "pca_projection_3d_to_2d.html",
        output_root / "static" / "pca_projection_comparison.png",
        output_root / "static" / "explained_variance_and_reconstruction.png",
        output_root / "animations" / "svd_geometric_decomposition.gif",
        output_root / "static" / "svd_geometric_decomposition.png",
        output_root / "static" / "svd_low_rank_reconstruction.png",
        output_root / "interactive" / "svd_rank_slider.html",
        output_root / "static" / "pca_svd_equivalence.png",
        output_root / "static" / "pca_common_pitfalls.png",
    )


def _task_definitions() -> dict[str, list[tuple[str, Callable[[], object]]]]:
    return {
        "eigenvectors": [
            (
                "Eigenvector transformation animation and final frame",
                lambda: generate_eigenvector_animation(
                    ANIMATION_DIR / "eigenvectors_transformation.gif",
                    STATIC_DIR / "eigenvectors_transformation.png",
                ),
            )
        ],
        "pca": [
            (
                "Covariance ellipse and principal directions",
                lambda: generate_covariance_pca(
                    STATIC_DIR / "covariance_and_principal_components.png"
                ),
            ),
            (
                "Interactive 3D PCA axes",
                lambda: generate_pca_3d(INTERACTIVE_DIR / "pca_3d_axes.html"),
            ),
            (
                "Interactive 3D-to-2D projection",
                lambda: generate_projection_3d_html(
                    INTERACTIVE_DIR / "pca_projection_3d_to_2d.html"
                ),
            ),
            (
                "Static 3D-to-2D projection comparison",
                lambda: generate_projection_comparison(
                    STATIC_DIR / "pca_projection_comparison.png"
                ),
            ),
            (
                "Explained variance and reconstruction error",
                lambda: generate_explained_variance(
                    STATIC_DIR / "explained_variance_and_reconstruction.png"
                ),
            ),
            (
                "PCA and SVD equivalence",
                lambda: generate_pca_svd_equivalence(
                    STATIC_DIR / "pca_svd_equivalence.png"
                ),
            ),
        ],
        "svd": [
            (
                "SVD geometry animation and four-stage summary",
                lambda: generate_svd_animation(
                    ANIMATION_DIR / "svd_geometric_decomposition.gif",
                    STATIC_DIR / "svd_geometric_decomposition.png",
                ),
            ),
            (
                "Static low-rank SVD reconstruction",
                lambda: generate_low_rank_static(
                    STATIC_DIR / "svd_low_rank_reconstruction.png"
                ),
            ),
            (
                "Interactive low-rank SVD slider",
                lambda: generate_low_rank_interactive(
                    INTERACTIVE_DIR / "svd_rank_slider.html"
                ),
            ),
        ],
        "pitfalls": [
            (
                "PCA centering, scaling, and outlier pitfalls",
                lambda: generate_pca_pitfalls(
                    STATIC_DIR / "pca_common_pitfalls.png"
                ),
            )
        ],
    }


def _flatten_paths(result: object) -> list[Path]:
    if isinstance(result, Path):
        return [result]
    if isinstance(result, (list, tuple)):
        return [path for path in result if isinstance(path, Path)]
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate visual assets for the Day 4 PCA/SVD laboratory."
    )
    parser.add_argument("--all", action="store_true", help="Generate every asset.")
    parser.add_argument(
        "--eigenvectors",
        action="store_true",
        help="Generate the eigenvector animation and final frame.",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Generate PCA plots, HTML interactions, and equivalence checks.",
    )
    parser.add_argument(
        "--svd",
        action="store_true",
        help="Generate SVD geometry and low-rank assets.",
    )
    parser.add_argument(
        "--pitfalls",
        action="store_true",
        help="Generate the PCA preprocessing-pitfalls figure.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate selected categories and return a process exit code."""
    args = build_parser().parse_args(argv)
    selected = [
        name
        for name in ("eigenvectors", "pca", "svd", "pitfalls")
        if args.all or getattr(args, name)
    ]
    if not selected:
        print(
            "No category selected. Use --all, --eigenvectors, --pca, "
            "--svd, or --pitfalls.",
            file=sys.stderr,
        )
        return 2

    ensure_output_directories()
    tasks = _task_definitions()
    generated: list[Path] = []
    total = sum(len(tasks[category]) for category in selected)
    task_number = 0

    try:
        for category in selected:
            for description, function in tasks[category]:
                task_number += 1
                print(f"[{task_number}/{total}] {description}...")
                result = function()
                paths = _flatten_paths(result)
                if not paths:
                    raise RuntimeError(
                        f"Generator returned no output path for: {description}"
                    )
                for path in paths:
                    if not path.is_file() or path.stat().st_size == 0:
                        raise RuntimeError(f"Expected output was not created: {path}")
                    generated.append(path)
                    print(f"  created {path.resolve()}")
    except Exception as error:
        print(
            f"Visual generation failed during task {task_number}: {error}",
            file=sys.stderr,
        )
        return 1

    print(f"\nGenerated {len(generated)} file(s) successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
