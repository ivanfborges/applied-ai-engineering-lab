"""Generate the Day 11 visual laboratory from deterministic synthetic data."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from visualizations.animated_experiments import (
    generate_ci_coverage_animation,
    generate_clt_animation,
)
from visualizations.applied_experiments import (
    generate_dependence_demo,
    generate_model_comparison,
    generate_practical_significance_demo,
)
from visualizations.interactive_experiments import generate_standard_error_surface
from visualizations.static_experiments import (
    generate_ci_construction,
    generate_clt_distribution_comparison,
    generate_confidence_width_comparison,
    generate_population_vs_sampling,
    generate_sd_vs_se,
    generate_standard_error_plot,
    generate_t_vs_normal,
)
from visualizations.visual_utils import (
    ASSET_DIRECTORY,
    TOPIC_DIRECTORY,
    AssetResult,
    configure_matplotlib,
)


EXPECTED_OUTPUTS = (
    "assets/01_population_vs_sampling.png",
    "assets/02_clt_convergence.gif",
    "assets/03_standard_error_vs_n.png",
    "assets/04_standard_error_surface.html",
    "assets/05_sd_vs_se.png",
    "assets/06_ci_construction.png",
    "assets/07_ci_coverage.gif",
    "assets/08_confidence_level_width.png",
    "assets/09_z_vs_t_distribution.png",
    "assets/10_skewness_and_sample_size.png",
    "assets/11_independence_violation.png",
    "assets/12_model_comparison_ci.png",
    "assets/13_practical_vs_statistical_significance.png",
)

RECOMMENDED_PREVIEWS = (
    "assets/02_clt_convergence.gif",
    "assets/07_ci_coverage.gif",
    "assets/11_independence_violation.png",
)

Generator = Callable[..., AssetResult]

GENERATORS: tuple[Generator, ...] = (
    generate_population_vs_sampling,
    generate_clt_animation,
    generate_standard_error_plot,
    generate_standard_error_surface,
    generate_sd_vs_se,
    generate_ci_construction,
    generate_ci_coverage_animation,
    generate_confidence_width_comparison,
    generate_t_vs_normal,
    generate_clt_distribution_comparison,
    generate_dependence_demo,
    generate_model_comparison,
    generate_practical_significance_demo,
)

SECTIONS: dict[str, tuple[Generator, ...]] = {
    "clt": (
        generate_population_vs_sampling,
        generate_clt_animation,
        generate_clt_distribution_comparison,
    ),
    "se": (
        generate_standard_error_plot,
        generate_standard_error_surface,
        generate_sd_vs_se,
    ),
    "ci": (
        generate_ci_construction,
        generate_ci_coverage_animation,
        generate_confidence_width_comparison,
        generate_t_vs_normal,
    ),
    "dependence": (generate_dependence_demo,),
    "comparison": (
        generate_model_comparison,
        generate_practical_significance_demo,
    ),
}


def generate_selected(
    generators: tuple[Generator, ...],
    *,
    quick: bool,
) -> list[AssetResult]:
    """Run the requested generators in conceptual order."""
    configure_matplotlib()
    return [generator(quick=quick) for generator in generators]


def report_results(results: list[AssetResult], *, full_run: bool) -> None:
    """Validate files and print only values returned by completed generators."""
    relative_paths = [
        result.path.relative_to(TOPIC_DIRECTORY).as_posix() for result in results
    ]
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("A visual generator returned a duplicate output path.")
    if full_run:
        missing = sorted(set(EXPECTED_OUTPUTS).difference(relative_paths))
        unexpected = sorted(set(relative_paths).difference(EXPECTED_OUTPUTS))
        if missing or unexpected:
            raise RuntimeError(
                f"Output manifest mismatch. Missing={missing}; unexpected={unexpected}"
            )

    for result in results:
        if not result.path.is_file() or result.path.stat().st_size == 0:
            raise RuntimeError(f"Generator did not create a non-empty file: {result.path}")
        relative = result.path.relative_to(TOPIC_DIRECTORY).as_posix()
        print(f"[OK] {result.label}: {relative}")
        for metric in result.metrics:
            print(f"     {metric}")

    print(f"\nAssets saved to: {ASSET_DIRECTORY.relative_to(TOPIC_DIRECTORY).as_posix()}/")
    if full_run:
        print("\nRecommended public-preview candidates after visual review:")
        for path in RECOMMENDED_PREVIEWS:
            print(f"- {path}")
        print("All generated assets remain ignored until explicit curation.")


def parse_arguments() -> argparse.Namespace:
    """Parse one section and an optional reduced-cost rendering mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=(*SECTIONS, "all"),
        default="all",
        help="Generate one conceptual section or the complete laboratory.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduce simulations, animation frames, and image resolution.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the requested assets and report their empirical checks."""
    args = parse_arguments()
    print("Generating Day 11 visual laboratory...")
    if args.quick:
        print("Quick mode: reduced simulation and rendering configuration.")
    generators = GENERATORS if args.section == "all" else SECTIONS[args.section]
    results = generate_selected(generators, quick=args.quick)
    report_results(results, full_run=args.section == "all")


if __name__ == "__main__":
    main()
