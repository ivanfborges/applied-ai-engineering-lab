"""Command-line entrypoint for the Day 10 visual laboratory."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from visualizations.animated_experiments import (
    experiment_more_biased_data,
    experiment_rare_event_sampling,
    experiment_sampling_distribution,
)
from visualizations.interactive_experiments import (
    experiment_mse_surface,
    experiment_sampling_explorer,
)
from visualizations.static_experiments import (
    experiment_bias_variance_target,
    experiment_correlated_observations,
    experiment_effective_sample_size,
    experiment_group_split,
    experiment_llm_evaluation,
    experiment_mse_heatmap,
    experiment_population_sampling,
    experiment_sample_size,
    experiment_sampling_strategies,
    experiment_weighting,
)
from visualizations.visual_utils import (
    TOPIC_DIRECTORY,
    configure_matplotlib,
    create_population,
)


EXPECTED_OUTPUTS = (
    "outputs/static/population_vs_samples.png",
    "outputs/static/sample_size_distributions.png",
    "outputs/static/sample_size_vs_standard_error.png",
    "outputs/static/bias_variance_target.png",
    "outputs/static/mse_bias_variance_heatmap.png",
    "outputs/static/sampling_strategy_comparison.png",
    "outputs/static/why_weighting_matters.png",
    "outputs/static/effective_sample_size.png",
    "outputs/static/correlated_observations.png",
    "outputs/static/group_split_vs_random_split.png",
    "outputs/static/llm_evaluation_sampling.png",
    "outputs/gifs/sampling_distribution.gif",
    "outputs/gifs/more_biased_data.gif",
    "outputs/gifs/rare_event_sampling.gif",
    "outputs/interactive/mse_bias_variance_surface.html",
    "outputs/interactive/sampling_explorer.html",
)

PUBLIC_PREVIEW_CANDIDATES = (
    "outputs/gifs/sampling_distribution.gif",
    "outputs/gifs/more_biased_data.gif",
    "outputs/static/sampling_strategy_comparison.png",
    "outputs/static/sample_size_vs_standard_error.png",
)


def _with_population(
    generator: Callable[[object], list[Path]],
) -> Callable[[], list[Path]]:
    return lambda: generator(create_population())


EXPERIMENTS: dict[str, tuple[Callable[[], list[Path]], ...]] = {
    "population": (_with_population(experiment_population_sampling),),
    "sampling": (experiment_sampling_distribution,),
    "sample-size": (_with_population(experiment_sample_size),),
    "bias": (experiment_more_biased_data,),
    "bias-variance": (experiment_bias_variance_target, experiment_mse_heatmap),
    "mse": (experiment_mse_heatmap, experiment_mse_surface),
    "stratified": (_with_population(experiment_sampling_strategies),),
    "weighting": (
        _with_population(experiment_weighting),
        experiment_effective_sample_size,
    ),
    "clusters": (experiment_correlated_observations, experiment_group_split),
    "rare-events": (experiment_rare_event_sampling,),
    "llm-evaluation": (experiment_llm_evaluation,),
    "interactive": (experiment_mse_surface, experiment_sampling_explorer),
}


def generate_all() -> list[Path]:
    """Generate each asset exactly once in conceptual order."""
    configure_matplotlib()
    population = create_population()
    generated: list[Path] = []
    for generator in (
        experiment_population_sampling,
        experiment_sample_size,
        experiment_sampling_strategies,
        experiment_weighting,
    ):
        generated.extend(generator(population))
    for generator in (
        experiment_bias_variance_target,
        experiment_mse_heatmap,
        experiment_effective_sample_size,
        experiment_correlated_observations,
        experiment_group_split,
        experiment_llm_evaluation,
        experiment_sampling_distribution,
        experiment_more_biased_data,
        experiment_rare_event_sampling,
        experiment_mse_surface,
        experiment_sampling_explorer,
    ):
        generated.extend(generator())
    return generated


def _relative_paths(paths: list[Path]) -> list[str]:
    return [path.relative_to(TOPIC_DIRECTORY).as_posix() for path in paths]


def _print_group(title: str, paths: list[Path]) -> None:
    print(f"\n{title}\n{'-' * len(title)}")
    for path in sorted(paths):
        print(path.relative_to(TOPIC_DIRECTORY).as_posix())


def report_generated_assets(paths: list[Path], *, full_run: bool) -> None:
    """Print only outputs returned successfully by generators."""
    relative = _relative_paths(paths)
    if len(relative) != len(set(relative)):
        raise RuntimeError("A generator returned the same output more than once.")
    if full_run:
        missing = sorted(set(EXPECTED_OUTPUTS).difference(relative))
        unexpected = sorted(set(relative).difference(EXPECTED_OUTPUTS))
        if missing or unexpected:
            raise RuntimeError(
                f"Output manifest mismatch. Missing={missing}; unexpected={unexpected}"
            )

    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Generator did not create a non-empty file: {path}")

    static = [path for path in paths if path.suffix.lower() == ".png"]
    animations = [path for path in paths if path.suffix.lower() == ".gif"]
    interactive = [path for path in paths if path.suffix.lower() == ".html"]
    print("\nDay 10 Visual Lab completed.")
    _print_group("Static", static)
    _print_group("Animations", animations)
    _print_group("Interactive", interactive)

    recommended = [
        TOPIC_DIRECTORY / candidate
        for candidate in PUBLIC_PREVIEW_CANDIDATES
        if candidate in relative
    ]
    if recommended:
        _print_group("Recommended README previews", recommended)
        print(
            "\nRecommendation basis: these assets most directly expose the "
            "sampling distribution, persistent selection bias, design comparison, "
            "and empirical-versus-theoretical standard error."
        )


def parse_arguments() -> argparse.Namespace:
    """Parse one bounded experiment selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=(*EXPERIMENTS, "all"),
        default="all",
        help="Generate one conceptual group or the complete laboratory.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the selected assets and print a verified index."""
    args = parse_arguments()
    if args.experiment == "all":
        generated = generate_all()
        report_generated_assets(generated, full_run=True)
        return

    configure_matplotlib()
    generated: list[Path] = []
    for generator in EXPERIMENTS[args.experiment]:
        generated.extend(generator())
    report_generated_assets(generated, full_run=False)


if __name__ == "__main__":
    main()
