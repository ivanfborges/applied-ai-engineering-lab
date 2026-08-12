"""Animated experiments that reveal sampling mechanisms over time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from visualizations.visual_utils import (
    COLORS,
    SEED,
    biased_selection_probabilities,
    calculate_estimator_statistics,
    configure_matplotlib,
    create_population,
    density_histogram,
    repeated_sample_means,
    save_animation,
    validate_mse_identity,
)


BIASED_SAMPLE_SIZES = np.array([20, 50, 100, 500, 2_000, 10_000])


def experiment_sampling_distribution() -> list[Path]:
    """Animate sample means accumulating into a sampling distribution."""
    configure_matplotlib()
    population = create_population()
    rng = np.random.default_rng(SEED + 20)
    sample_size = 80
    sample_count = 900
    samples = population.spend[
        rng.choice(
            population.size,
            size=(sample_count, sample_size),
            replace=True,
        )
    ]
    sample_means = samples.mean(axis=1)
    frame_ends = np.linspace(25, sample_count, 36, dtype=int)
    population_display = rng.choice(population.spend, size=4_000, replace=False)
    mean_bins = np.linspace(sample_means.min() - 1.0, sample_means.max() + 1.0, 35)

    figure, (population_axis, sampling_axis) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.5),
        gridspec_kw={"height_ratios": [1.1, 1.8]},
    )

    def update(frame: int) -> None:
        end = int(frame_ends[frame])
        current_sample = samples[end - 1]
        current_mean = float(sample_means[end - 1])
        observed_means = sample_means[:end]
        population_axis.clear()
        sampling_axis.clear()

        population_axis.hist(
            population_display,
            bins=55,
            density=True,
            color=COLORS["light"],
            edgecolor="white",
            label="Population density (display subset)",
        )
        population_axis.scatter(
            current_sample,
            np.full(sample_size, -0.00035),
            marker="|",
            s=55,
            color=COLORS["blue"],
            alpha=0.75,
            label=f"Current random sample, n={sample_size}",
        )
        population_axis.axvline(
            population.true_mean,
            color=COLORS["dark"],
            linestyle="--",
            linewidth=1.8,
            label=f"True mean {population.true_mean:.2f}",
        )
        population_axis.axvline(
            current_mean,
            color=COLORS["orange"],
            linewidth=2.0,
            label=f"Current estimate {current_mean:.2f}",
        )
        population_axis.set(
            title="One new sample produces one new estimate",
            xlabel="Synthetic spend",
            ylabel="Density",
        )
        population_axis.legend(loc="upper right", ncols=2)

        density_histogram(
            sampling_axis,
            observed_means,
            bins=mean_bins,
            color=COLORS["blue"],
            alpha=0.68,
        )
        sampling_axis.axvline(
            population.true_mean,
            color=COLORS["dark"],
            linestyle="--",
            linewidth=2.0,
            label="True population mean",
        )
        sampling_axis.axvline(
            current_mean,
            color=COLORS["orange"],
            linewidth=1.7,
            label="Current sample mean",
        )
        empirical_center = float(observed_means.mean())
        empirical_se = float(observed_means.std(ddof=0))
        sampling_axis.set(
            title="Why does an estimator have a distribution?",
            xlabel="Sample mean",
            ylabel="Density",
            xlim=(mean_bins[0], mean_bins[-1]),
        )
        sampling_axis.text(
            0.98,
            0.92,
            f"Samples drawn: {end}\n"
            f"Current estimate: {current_mean:.2f}\n"
            f"Mean of estimates: {empirical_center:.2f}\n"
            f"Empirical SE: {empirical_se:.2f}",
            transform=sampling_axis.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": COLORS["light"]},
        )
        sampling_axis.legend(loc="upper left")
        figure.tight_layout()

    return [
        save_animation(
            figure,
            update,
            len(frame_ends),
            "sampling_distribution.gif",
            fps=5,
            dpi=82,
        )
    ]


def calculate_biased_size_results(
    *,
    trials: int = 800,
) -> tuple[float, dict[int, np.ndarray], dict[int, dict[str, float]]]:
    """Calculate biased sampling distributions across increasing sample sizes."""
    population = create_population()
    rng = np.random.default_rng(SEED + 21)
    probabilities = biased_selection_probabilities(population)
    results: dict[int, np.ndarray] = {}
    summaries: dict[int, dict[str, float]] = {}
    for sample_size_value in BIASED_SAMPLE_SIZES:
        sample_size = int(sample_size_value)
        values = repeated_sample_means(
            population.spend,
            sample_size=sample_size,
            trials=trials,
            rng=rng,
            probabilities=probabilities,
            replace=True,
            batch_size=80,
        )
        results[sample_size] = values
        summary = calculate_estimator_statistics(values, population.true_mean)
        validate_mse_identity(summary)
        summaries[sample_size] = summary
    return population.true_mean, results, summaries


def experiment_more_biased_data() -> list[Path]:
    """Animate variance shrinking while empirical selection bias persists."""
    configure_matplotlib()
    true_mean, results, summaries = calculate_biased_size_results()
    all_estimates = np.concatenate(list(results.values()))
    x_min = min(true_mean - 8.0, float(np.percentile(all_estimates, 0.1)) - 4.0)
    x_max = float(np.percentile(all_estimates, 99.9)) + 4.0
    bins = np.linspace(x_min, x_max, 65)
    frame_indices = np.repeat(np.arange(len(BIASED_SAMPLE_SIZES)), 2)
    figure, (distribution_axis, trend_axis) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.5),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    def update(frame: int) -> None:
        nonlocal distribution_axis, trend_axis
        position = int(frame_indices[frame])
        sample_size = int(BIASED_SAMPLE_SIZES[position])
        values = results[sample_size]
        summary = summaries[sample_size]
        trend_axis.set_xscale("linear")
        trend_axis.set_yscale("linear")
        figure.clear()
        distribution_axis, trend_axis = figure.subplots(
            2,
            1,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )

        density_histogram(
            distribution_axis,
            values,
            bins=bins,
            color=COLORS["red"],
            alpha=0.72,
        )
        distribution_axis.axvline(
            true_mean,
            color=COLORS["dark"],
            linestyle="--",
            linewidth=2.3,
            label=f"True population mean {true_mean:.2f}",
        )
        distribution_axis.axvline(
            summary["expected_estimate"],
            color=COLORS["red"],
            linewidth=2.3,
            label=f"Mean biased estimate {summary['expected_estimate']:.2f}",
        )
        distribution_axis.annotate(
            "Bias persists",
            xy=(summary["expected_estimate"], 0.0),
            xytext=(true_mean + 18.0, distribution_axis.get_ylim()[1] * 0.72),
            arrowprops={"arrowstyle": "<->", "color": COLORS["orange"]},
            color=COLORS["orange"],
            ha="center",
        )
        distribution_axis.set(
            title=f"Why doesn't more biased data fix selection bias?  n = {sample_size:,}",
            xlabel="Estimated population mean",
            ylabel="Density",
            xlim=(x_min, x_max),
        )
        distribution_axis.text(
            0.98,
            0.94,
            f"Empirical bias: {summary['bias']:.2f}\n"
            f"Empirical variance: {summary['variance']:.3f}\n"
            f"Empirical SE: {np.sqrt(summary['variance']):.3f}",
            transform=distribution_axis.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": COLORS["light"]},
        )
        distribution_axis.legend(loc="upper left")

        sizes_so_far = BIASED_SAMPLE_SIZES[: position + 1]
        variances = [summaries[int(size)]["variance"] for size in sizes_so_far]
        biases = [abs(summaries[int(size)]["bias"]) for size in sizes_so_far]
        trend_axis.plot(
            sizes_so_far,
            variances,
            marker="o",
            color=COLORS["blue"],
            linewidth=2.0,
            label="Empirical variance ↓",
        )
        trend_axis.set_xscale("log")
        trend_axis.set_yscale("log")
        trend_axis.set(
            xlabel="Sample size n (log scale)",
            ylabel="Empirical variance (log scale)",
            xlim=(BIASED_SAMPLE_SIZES[0] * 0.8, BIASED_SAMPLE_SIZES[-1] * 1.25),
        )
        bias_axis = trend_axis.twinx()
        bias_axis.plot(
            sizes_so_far,
            biases,
            marker="s",
            linestyle="--",
            color=COLORS["red"],
            linewidth=1.8,
            label="Absolute empirical bias",
        )
        bias_axis.set_ylabel("Absolute empirical bias", color=COLORS["red"])
        bias_axis.set_ylim(0, max(biases) * 1.2)
        trend_axis.legend(loc="upper right")
        bias_axis.legend(loc="center right")
        figure.tight_layout()

    return [
        save_animation(
            figure,
            update,
            len(frame_indices),
            "more_biased_data.gif",
            fps=2,
            dpi=82,
        )
    ]


def calculate_rare_event_counts(
    *,
    prevalence: float = 0.01,
    sample_size: int = 50,
    samples: int = 1_000,
) -> np.ndarray:
    """Draw random rare-event counts under the stated production prevalence."""
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must be strictly between zero and one.")
    if sample_size <= 0 or samples <= 0:
        raise ValueError("sample_size and samples must be positive.")
    rng = np.random.default_rng(SEED + 22)
    return rng.binomial(sample_size, prevalence, size=samples)


def experiment_rare_event_sampling() -> list[Path]:
    """Animate how random samples can miss a rare event entirely."""
    configure_matplotlib()
    prevalence = 0.01
    sample_size = 50
    counts = calculate_rare_event_counts(
        prevalence=prevalence,
        sample_size=sample_size,
    )
    diagnostic_rare_count = 10
    frame_ends = np.linspace(40, counts.size, 25, dtype=int)
    maximum_count = max(5, int(counts.max()))
    figure, (count_axis, composition_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 7.2),
        gridspec_kw={"height_ratios": [1.8, 1.0]},
    )

    def update(frame: int) -> None:
        end = int(frame_ends[frame])
        observed = counts[:end]
        absent_rate = float(np.mean(observed == 0))
        count_axis.clear()
        composition_axis.clear()

        discrete_values = np.arange(maximum_count + 1)
        frequencies = np.array([np.mean(observed == value) for value in discrete_values])
        colors = [COLORS["red"]] + [COLORS["blue"]] * maximum_count
        count_axis.bar(discrete_values, frequencies, color=colors)
        count_axis.set(
            title="How often does a small random sample miss a 1% event?",
            xlabel=f"Rare events in one random sample (n={sample_size})",
            ylabel="Share of repeated samples",
            xticks=discrete_values,
            ylim=(0, 0.75),
        )
        count_axis.text(
            0.98,
            0.92,
            f"Samples observed: {end}\n"
            f"Zero rare events: {absent_rate:.1%}\n"
            f"Mean rare count: {observed.mean():.3f}",
            transform=count_axis.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": COLORS["light"]},
        )

        random_rare_count = int(counts[end - 1])
        rare_shares = [
            random_rare_count / sample_size,
            diagnostic_rare_count / sample_size,
        ]
        labels = ["Current random sample", "Stratified diagnostic sample"]
        composition_axis.bar(
            labels,
            np.asarray(rare_shares) * 100.0,
            color=[COLORS["blue"], COLORS["orange"]],
        )
        composition_axis.axhline(
            prevalence * 100.0,
            color=COLORS["dark"],
            linestyle="--",
            label="Production prevalence = 1%",
        )
        composition_axis.set(
            title="Targeted inclusion helps diagnosis, but its raw share is not prevalence",
            ylabel="Rare-event share in sample (%)",
            ylim=(0, 24),
        )
        composition_axis.legend(loc="upper left")
        figure.tight_layout()

    return [
        save_animation(
            figure,
            update,
            len(frame_ends),
            "rare_event_sampling.gif",
            fps=4,
            dpi=82,
        )
    ]


def generate_animated_experiments() -> list[Path]:
    """Generate the three bounded educational GIFs."""
    generated: list[Path] = []
    for generator in (
        experiment_sampling_distribution,
        experiment_more_biased_data,
        experiment_rare_event_sampling,
    ):
        generated.extend(generator())
    return generated
