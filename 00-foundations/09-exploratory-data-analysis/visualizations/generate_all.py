"""Generate the complete Day 9 Statistical EDA Visual Lab."""

from __future__ import annotations

from pathlib import Path

from central_tendency import generate_central_tendency_visuals
from correlation import generate_correlation_visuals
from dispersion import generate_dispersion_visuals
from distribution_shape import generate_distribution_shape_visuals
from interactive_eda import generate_interactive_visuals
from outliers import generate_outlier_visuals
from visual_utils import TOPIC_DIRECTORY


EXPECTED_OUTPUTS = (
    "outputs/images/why_squared_deviations.png",
    "outputs/images/bessel_correction.png",
    "outputs/images/skewness_comparison.png",
    "outputs/images/latency_percentiles.png",
    "outputs/images/iqr_boxplot_anatomy.png",
    "outputs/images/zscore_vs_iqr.png",
    "outputs/images/covariance_geometry.png",
    "outputs/images/pearson_noise_levels.png",
    "outputs/images/pearson_vs_spearman.png",
    "outputs/images/nonlinear_dependence.png",
    "outputs/images/correlation_causation.png",
    "outputs/images/simpsons_paradox.png",
    "outputs/images/anscombes_quartet.png",
    "outputs/images/ai_workload_dashboard.png",
    "outputs/gifs/mean_vs_median_outlier.gif",
    "outputs/gifs/robust_statistics.gif",
    "outputs/gifs/variance_and_std.gif",
    "outputs/gifs/skewness_animation.gif",
    "outputs/gifs/kurtosis_tail_behavior.gif",
    "outputs/gifs/outlier_effect_on_correlation.gif",
    "outputs/interactive/ai_workload_3d.html",
    "outputs/interactive/distribution_explorer.html",
)


def generate_all() -> list[Path]:
    """Run every bounded generator and return the generated paths."""
    generated: list[Path] = []
    generators = (
        generate_central_tendency_visuals,
        generate_dispersion_visuals,
        generate_distribution_shape_visuals,
        generate_outlier_visuals,
        generate_correlation_visuals,
        generate_interactive_visuals,
    )
    for generator in generators:
        generated.extend(generator())
    return generated


def _print_group(title: str, paths: list[Path]) -> None:
    """Print one categorized output group relative to the topic directory."""
    print(f"\n{title}:")
    for path in sorted(paths):
        print(f"  - {path.relative_to(TOPIC_DIRECTORY)}")


def main() -> None:
    """Generate and report every static, animated, and interactive output."""
    generated = generate_all()
    generated_relative = {
        path.relative_to(TOPIC_DIRECTORY).as_posix() for path in generated
    }
    missing = sorted(set(EXPECTED_OUTPUTS).difference(generated_relative))
    if missing:
        raise RuntimeError(f"Generators did not return expected outputs: {missing}")

    images = [path for path in generated if path.suffix.lower() == ".png"]
    animations = [path for path in generated if path.suffix.lower() == ".gif"]
    interactive = [path for path in generated if path.suffix.lower() == ".html"]

    print("\nDay 9 visualization lab generated successfully.")
    _print_group("Static images", images)
    _print_group("Animations", animations)
    _print_group("Interactive visualizations", interactive)


if __name__ == "__main__":
    main()
