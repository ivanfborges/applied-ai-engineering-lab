"""Static visual experiments for sampling, bias, and variance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt

from example import PREMIUM, REGULAR, SyntheticPopulation
from visualizations.visual_utils import (
    COLORS,
    SEED,
    biased_selection_probabilities,
    calculate_estimator_statistics,
    configure_matplotlib,
    create_population,
    density_histogram,
    population_weighted_estimate,
    repeated_sample_means,
    repeated_stratified_means,
    save_figure,
    validate_effective_sample_size,
    validate_mse_identity,
)


SAMPLE_SIZES = np.array([10, 30, 100, 500, 2_000])


def experiment_population_sampling(
    population: SyntheticPopulation,
) -> list[Path]:
    """Show that one sample is a partial, variable view of a population."""
    rng = np.random.default_rng(SEED + 1)
    sample_sizes = (50, 200, 1_000)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    bins = np.linspace(20, 470, 55)
    true_mean = population.true_mean

    density_histogram(
        axes[0, 0],
        population.spend,
        bins=bins,
        color=COLORS["gray"],
    )
    axes[0, 0].axvline(true_mean, color=COLORS["dark"], linewidth=2.2)
    axes[0, 0].set_title("Target population: all synthetic customers")
    axes[0, 0].text(
        0.97,
        0.92,
        f"Parameter: $\\mu$ = {true_mean:.2f}\nN = {population.size:,}",
        ha="right",
        va="top",
        transform=axes[0, 0].transAxes,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": COLORS["light"]},
    )

    for axis, sample_size in zip(axes.flat[1:], sample_sizes, strict=True):
        indices = rng.choice(population.size, size=sample_size, replace=False)
        sample = population.spend[indices]
        estimate = float(sample.mean())
        density_histogram(axis, sample, bins=bins, color=COLORS["blue"])
        axis.axvline(
            true_mean,
            color=COLORS["dark"],
            linewidth=1.8,
            linestyle="--",
            label=f"True mean {true_mean:.2f}",
        )
        axis.axvline(
            estimate,
            color=COLORS["orange"],
            linewidth=2.0,
            label=f"Sample mean {estimate:.2f}",
        )
        axis.set_title(f"One random sample: n = {sample_size:,}")
        axis.legend(loc="upper right")

    for axis in axes.flat:
        axis.set_xlabel("Synthetic customer spend")
        axis.set_ylabel("Density")
    figure.suptitle(
        "Population → random sample → estimate: each sample is only one view",
        y=1.01,
    )
    figure.text(
        0.5,
        -0.01,
        "Sampling without replacement; synthetic two-segment population.",
        ha="center",
        color=COLORS["gray"],
    )
    figure.tight_layout()
    return [save_figure(figure, "population_vs_samples.png")]


def calculate_sample_size_results(
    population: SyntheticPopulation,
    *,
    sample_sizes: np.ndarray = SAMPLE_SIZES,
    trials: int = 1_200,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Calculate empirical and theoretical SE under replacement sampling."""
    rng = np.random.default_rng(SEED + 2)
    estimates: dict[int, np.ndarray] = {}
    empirical_se: list[float] = []
    theoretical_se: list[float] = []
    sigma = float(np.std(population.spend, ddof=0))
    for sample_size_value in sample_sizes:
        sample_size = int(sample_size_value)
        values = repeated_sample_means(
            population.spend,
            sample_size=sample_size,
            trials=trials,
            rng=rng,
            replace=True,
        )
        estimates[sample_size] = values
        empirical_se.append(float(np.std(values, ddof=0)))
        theoretical_se.append(sigma / np.sqrt(sample_size))
    return estimates, np.asarray(empirical_se), np.asarray(theoretical_se)


def experiment_sample_size(
    population: SyntheticPopulation,
) -> list[Path]:
    """Compare the observed sampling spread with sigma / sqrt(n)."""
    estimates, empirical_se, theoretical_se = calculate_sample_size_results(
        population
    )
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(SAMPLE_SIZES)))
    all_values = np.concatenate(list(estimates.values()))
    center = population.true_mean
    half_width = max(center - np.percentile(all_values, 0.2), 25.0)
    bins = np.linspace(center - half_width, center + half_width, 55)

    figure, axis = plt.subplots(figsize=(11, 6.5))
    for color, sample_size in zip(colors, SAMPLE_SIZES, strict=True):
        values = estimates[int(sample_size)]
        density_histogram(
            axis,
            values,
            bins=bins,
            color=color,
            label=f"n={sample_size:,}; empirical SE={np.std(values):.2f}",
            alpha=0.33,
        )
    axis.axvline(
        center,
        color=COLORS["dark"],
        linestyle="--",
        linewidth=2.0,
        label=f"True mean {center:.2f}",
    )
    axis.set(
        title="Why do larger samples produce narrower sampling distributions?",
        xlabel="Sample mean of synthetic spend",
        ylabel="Density",
    )
    axis.legend(ncols=2)
    figure.text(
        0.5,
        0.01,
        "1,200 repeated samples per n, sampled with replacement from the same finite population.",
        ha="center",
        color=COLORS["gray"],
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    first = save_figure(figure, "sample_size_distributions.png")

    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    axis.plot(
        SAMPLE_SIZES,
        empirical_se,
        marker="o",
        linewidth=2.2,
        color=COLORS["blue"],
        label="Empirical SE (Monte Carlo)",
    )
    axis.plot(
        SAMPLE_SIZES,
        theoretical_se,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color=COLORS["orange"],
        label=r"Theoretical $\sigma/\sqrt{n}$",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        title="Does empirical standard error follow the $1/\\sqrt{n}$ relationship?",
        xlabel="Sample size n (log scale)",
        ylabel="Standard error of sample mean (log scale)",
    )
    axis.legend()
    for x_value, empirical, theoretical in zip(
        SAMPLE_SIZES, empirical_se, theoretical_se, strict=True
    ):
        relative_gap = 100.0 * (empirical - theoretical) / theoretical
        axis.annotate(
            f"{relative_gap:+.1f}%",
            (x_value, empirical),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8,
            color=COLORS["blue"],
        )
    figure.text(
        0.5,
        0.01,
        "Labels show empirical deviation from theory; agreement is computed, not assumed.",
        ha="center",
        color=COLORS["gray"],
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    second = save_figure(figure, "sample_size_vs_standard_error.png")
    return [first, second]


def experiment_bias_variance_target() -> list[Path]:
    """Map four explicit estimator distributions onto target-style panels."""
    rng = np.random.default_rng(SEED + 3)
    true_parameter = 0.0
    scenarios = (
        ("Low bias / low variance", 0.15, 0.55),
        ("Low bias / high variance", 0.15, 1.65),
        ("High bias / low variance", 2.35, 0.55),
        ("High bias / high variance", 2.35, 1.65),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 9))
    for axis, (title, configured_bias, configured_sd) in zip(
        axes.flat, scenarios, strict=True
    ):
        x_values = rng.normal(configured_bias, configured_sd, 120)
        y_values = rng.normal(0.0, configured_sd, 120)
        radial_estimates = x_values
        statistics = calculate_estimator_statistics(radial_estimates, true_parameter)
        for radius in (1.0, 2.0, 3.0, 4.0):
            axis.add_patch(
                patches.Circle(
                    (0.0, 0.0),
                    radius,
                    fill=False,
                    linewidth=1.0,
                    color=COLORS["light"],
                )
            )
        axis.axhline(0.0, color=COLORS["gray"], linewidth=0.7)
        axis.axvline(0.0, color=COLORS["gray"], linewidth=0.7)
        axis.scatter(x_values, y_values, s=18, alpha=0.62, color=COLORS["blue"])
        axis.scatter([0.0], [0.0], marker="x", s=100, linewidth=2.5, color=COLORS["red"])
        axis.scatter(
            [x_values.mean()],
            [y_values.mean()],
            marker="D",
            s=55,
            color=COLORS["orange"],
            edgecolor="white",
        )
        axis.set(
            title=(
                f"{title}\n"
                f"empirical bias={statistics['bias']:.2f}, "
                f"variance={statistics['variance']:.2f}"
            ),
            xlim=(-4.5, 5.0),
            ylim=(-4.5, 4.5),
            aspect="equal",
            xlabel="Estimator error along target axis",
            ylabel="Independent display dimension",
        )
    figure.suptitle(
        "Where do repeated estimates land? Center encodes bias; spread encodes variance",
        y=1.01,
    )
    figure.text(
        0.5,
        0.01,
        r"Simulated Normal estimators. $Bias(\hat\theta)=E[\hat\theta]-\theta$; "
        r"$Var(\hat\theta)=E[(\hat\theta-E[\hat\theta])^2]$.",
        ha="center",
        color=COLORS["gray"],
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    return [save_figure(figure, "bias_variance_target.png")]


def experiment_mse_heatmap() -> list[Path]:
    """Show the exact two-dimensional MSE decomposition for GitHub."""
    biases = np.linspace(-3.0, 3.0, 121)
    variances = np.linspace(0.0, 9.0, 121)
    bias_grid, variance_grid = np.meshgrid(biases, variances)
    mse = bias_grid**2 + variance_grid
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    image = axis.contourf(bias_grid, variance_grid, mse, levels=24, cmap="viridis")
    contours = axis.contour(
        bias_grid,
        variance_grid,
        mse,
        colors="white",
        linewidths=0.6,
        alpha=0.65,
    )
    axis.clabel(contours, inline=True, fontsize=7, fmt="%.0f")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("MSE")
    axis.set(
        title="How do estimator bias and variance combine into mean squared error?",
        xlabel=r"Bias $E[\hat\theta]-\theta$",
        ylabel=r"Variance $Var(\hat\theta)$",
    )
    axis.text(
        0.02,
        0.96,
        r"$MSE = Bias^2 + Variance$",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    figure.tight_layout()
    return [save_figure(figure, "mse_bias_variance_heatmap.png")]


def calculate_strategy_results(
    population: SyntheticPopulation,
    *,
    sample_size: int = 500,
    trials: int = 1_200,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Calculate repeated estimates for three explicitly different designs."""
    if sample_size % 2:
        raise ValueError("sample_size must be even for equal stratified allocation.")
    random_rng, biased_rng, stratified_rng = (
        np.random.default_rng(child)
        for child in np.random.SeedSequence(SEED + 4).spawn(3)
    )
    selection_probabilities = biased_selection_probabilities(population)
    estimates = {
        "Simple random": repeated_sample_means(
            population.spend,
            sample_size=sample_size,
            trials=trials,
            rng=random_rng,
            replace=True,
        ),
        "Selection biased": repeated_sample_means(
            population.spend,
            sample_size=sample_size,
            trials=trials,
            rng=biased_rng,
            probabilities=selection_probabilities,
            replace=True,
        ),
        "Stratified + weighted": repeated_stratified_means(
            population,
            regular_sample_size=sample_size // 2,
            premium_sample_size=sample_size // 2,
            trials=trials,
            rng=stratified_rng,
            replace=True,
        ),
    }
    summaries = {
        name: calculate_estimator_statistics(values, population.true_mean)
        for name, values in estimates.items()
    }
    for statistics in summaries.values():
        validate_mse_identity(statistics)
    return estimates, summaries


def experiment_sampling_strategies(
    population: SyntheticPopulation,
) -> list[Path]:
    """Compare center (bias) and width (variance) across three designs."""
    estimates, summaries = calculate_strategy_results(population)
    colors = {
        "Simple random": COLORS["blue"],
        "Selection biased": COLORS["red"],
        "Stratified + weighted": COLORS["green"],
    }
    figure, (axis, table_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 8.2),
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )
    common_bins = np.linspace(
        min(values.min() for values in estimates.values()) - 2.0,
        max(values.max() for values in estimates.values()) + 2.0,
        70,
    )
    for name, values in estimates.items():
        summary = summaries[name]
        density_histogram(
            axis,
            values,
            bins=common_bins,
            color=colors[name],
            label=(
                f"{name}: mean={summary['expected_estimate']:.2f}, "
                f"bias={summary['bias']:.2f}, SD={np.sqrt(summary['variance']):.2f}"
            ),
            alpha=0.42,
        )
        axis.axvline(
            summary["expected_estimate"],
            color=colors[name],
            linewidth=1.6,
        )
    axis.axvline(
        population.true_mean,
        color=COLORS["dark"],
        linestyle="--",
        linewidth=2.2,
        label=f"True population mean {population.true_mean:.2f}",
    )
    axis.set(
        title="How do sampling design choices change estimator center and width?",
        xlabel="Estimated population mean",
        ylabel="Density",
    )
    axis.legend(loc="upper right")

    table_axis.axis("off")
    cells = [
        [
            name,
            f"{summary['expected_estimate']:.3f}",
            f"{summary['bias']:.3f}",
            f"{np.sqrt(summary['variance']):.3f}",
            f"{summary['mse']:.3f}",
        ]
        for name, summary in summaries.items()
    ]
    table = table_axis.table(
        cellText=cells,
        colLabels=["Design", "Mean estimate", "Bias", "Empirical SE", "MSE"],
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    figure.text(
        0.5,
        0.01,
        "Synthetic population; 1,200 trials; n=500; sampling with replacement. "
        "The displayed ordering is a result of this configuration, not a universal guarantee.",
        ha="center",
        color=COLORS["gray"],
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return [save_figure(figure, "sampling_strategy_comparison.png")]


def experiment_weighting(population: SyntheticPopulation) -> list[Path]:
    """Show why a balanced diagnostic sample needs population weights."""
    rng = np.random.default_rng(SEED + 5)
    regular = population.spend[population.segments == REGULAR]
    premium = population.spend[population.segments == PREMIUM]
    regular_sample = rng.choice(regular, size=500, replace=False)
    premium_sample = rng.choice(premium, size=500, replace=False)
    sample = np.concatenate([regular_sample, premium_sample])
    naive_mean = float(sample.mean())
    regular_share = regular.size / population.size
    premium_share = premium.size / population.size
    weighted_estimate = population_weighted_estimate(
        [regular_sample.mean(), premium_sample.mean()],
        [regular_share, premium_share],
    )

    figure, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    labels = ["Regular", "Premium"]
    population_shares = np.array([regular_share, premium_share])
    sample_shares = np.array([0.5, 0.5])
    x_positions = np.arange(2)
    axes[0].bar(
        x_positions - 0.18,
        population_shares,
        width=0.36,
        color=COLORS["gray"],
        label="Population",
    )
    axes[0].bar(
        x_positions + 0.18,
        sample_shares,
        width=0.36,
        color=COLORS["blue"],
        label="Oversampled data",
    )
    axes[0].set(
        title="The sample composition was changed deliberately",
        ylabel="Share",
        xticks=x_positions,
        xticklabels=labels,
        ylim=(0, 1),
    )
    axes[0].legend()

    contribution_matrix = np.array(
        [
            [0.5 * regular_sample.mean(), 0.5 * premium_sample.mean()],
            [regular_share * regular_sample.mean(), premium_share * premium_sample.mean()],
        ]
    )
    contribution_labels = ["Naive 50/50", "Population weighted"]
    bottom = np.zeros(2)
    for index, (label, color) in enumerate(
        (("Regular contribution", COLORS["blue"]), ("Premium contribution", COLORS["orange"]))
    ):
        axes[1].bar(
            contribution_labels,
            contribution_matrix[:, index],
            bottom=bottom,
            label=label,
            color=color,
        )
        bottom += contribution_matrix[:, index]
    axes[1].set(
        title="Weights reconstruct population contributions",
        ylabel="Contribution to estimated mean spend",
    )
    axes[1].legend()

    metric_labels = ["True population", "Naive oversample", "Weighted oversample"]
    metric_values = [population.true_mean, naive_mean, weighted_estimate]
    bars = axes[2].bar(
        metric_labels,
        metric_values,
        color=[COLORS["dark"], COLORS["red"], COLORS["green"]],
    )
    axes[2].set(
        title="Which estimate answers the population question?",
        ylabel="Estimated mean spend",
    )
    axes[2].tick_params(axis="x", rotation=18)
    axes[2].bar_label(bars, fmt="%.2f", padding=3)
    axes[2].set_ylim(0, max(metric_values) * 1.2)

    figure.suptitle("Why weighting matters after deliberate oversampling", y=1.02)
    figure.text(
        0.5,
        -0.02,
        "Synthetic data; weights use known empirical population shares. Real corrections depend on a valid sampling frame.",
        ha="center",
        color=COLORS["gray"],
    )
    figure.tight_layout()
    return [save_figure(figure, "why_weighting_matters.png")]


def experiment_effective_sample_size() -> list[Path]:
    """Show how weight concentration reduces Kish effective sample size."""
    raw_size = 100
    scenarios = {
        "Equal": np.ones(raw_size),
        "Moderate": np.linspace(0.5, 2.0, raw_size),
        "Few large": np.concatenate([np.ones(95), np.full(5, 12.0)]),
        "One dominates": np.concatenate([np.ones(99), np.array([100.0])]),
    }
    scenario_ess = {
        name: validate_effective_sample_size(weights)
        for name, weights in scenarios.items()
    }
    concentration = np.geomspace(1.0, 150.0, 80)
    ess_curve = np.array(
        [
            validate_effective_sample_size(
                np.concatenate([np.ones(raw_size - 1), np.array([value])])
            )
            for value in concentration
        ]
    )

    figure, (top_axis, bottom_axis) = plt.subplots(2, 1, figsize=(11, 8))
    for position, (name, weights) in enumerate(scenarios.items()):
        sorted_weights = np.sort(weights)[::-1]
        top_axis.plot(
            np.arange(1, raw_size + 1),
            sorted_weights,
            linewidth=1.8,
            label=f"{name}: ESS={scenario_ess[name]:.1f}",
        )
    top_axis.set_yscale("log")
    top_axis.set(
        title="Which observations dominate as weights become unequal?",
        xlabel="Observation rank by weight",
        ylabel="Weight (log scale)",
    )
    top_axis.legend()

    bottom_axis.plot(
        concentration,
        ess_curve,
        color=COLORS["purple"],
        linewidth=2.4,
    )
    bottom_axis.axhline(
        raw_size,
        color=COLORS["dark"],
        linestyle="--",
        label=f"Raw sample size n={raw_size}",
    )
    bottom_axis.set_xscale("log")
    bottom_axis.set(
        title="Why can 100 weighted rows contain far less than 100 rows of information?",
        xlabel="Largest weight when the other 99 weights equal 1 (log scale)",
        ylabel="Kish effective sample size",
        ylim=(0, raw_size * 1.05),
    )
    bottom_axis.legend()
    bottom_axis.text(
        0.98,
        0.92,
        r"$ESS=(\sum_i w_i)^2/\sum_i w_i^2$",
        transform=bottom_axis.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    figure.tight_layout()
    return [save_figure(figure, "effective_sample_size.png")]


def calculate_dependence_results(
    *,
    rows: int = 10_000,
    clusters: int = 100,
    trials: int = 800,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compare equal row counts under independence and repeated clusters."""
    if rows % clusters:
        raise ValueError("rows must be divisible by clusters.")
    rng = np.random.default_rng(SEED + 6)
    cluster_sd = 25.0
    noise_sd = 10.0
    observations_per_cluster = rows // clusters
    marginal_sd = float(np.sqrt(cluster_sd**2 + noise_sd**2))
    independent_means = rng.normal(
        0.0,
        marginal_sd / np.sqrt(rows),
        size=trials,
    )
    cluster_effects = rng.normal(0.0, cluster_sd, size=(trials, clusters))
    cluster_noise_means = rng.normal(
        0.0,
        noise_sd / np.sqrt(observations_per_cluster),
        size=(trials, clusters),
    )
    clustered_means = (cluster_effects + cluster_noise_means).mean(axis=1)

    display_effects = rng.normal(0.0, cluster_sd, size=12)
    display_noise = rng.normal(0.0, noise_sd, size=(12, 35))
    display_values = display_effects[:, None] + display_noise
    intraclass_correlation = cluster_sd**2 / (cluster_sd**2 + noise_sd**2)
    return (
        independent_means,
        clustered_means,
        display_effects,
        display_values,
        intraclass_correlation,
    )


def experiment_correlated_observations() -> list[Path]:
    """Show that row count and independent information are not equivalent."""
    independent, clustered, effects, display, icc = calculate_dependence_results()
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    for customer, values in enumerate(display):
        axes[0].scatter(
            np.full(values.size, customer) + np.linspace(-0.22, 0.22, values.size),
            values,
            s=12,
            alpha=0.55,
        )
        axes[0].plot(customer, effects[customer], marker="D", color=COLORS["dark"], markersize=4)
    axes[0].set(
        title=f"Repeated rows share a customer effect\nconfigured ICC = {icc:.3f}",
        xlabel="Synthetic customer",
        ylabel="Observation value",
    )

    bins = np.linspace(
        min(independent.min(), clustered.min()),
        max(independent.max(), clustered.max()),
        55,
    )
    density_histogram(
        axes[1],
        independent,
        bins=bins,
        color=COLORS["blue"],
        label=f"10,000 users × 1; SE={np.std(independent):.3f}",
        alpha=0.5,
    )
    density_histogram(
        axes[1],
        clustered,
        bins=bins,
        color=COLORS["orange"],
        label=f"100 users × 100; SE={np.std(clustered):.3f}",
        alpha=0.5,
    )
    axes[1].axvline(0.0, color=COLORS["dark"], linestyle="--")
    axes[1].set(
        title="Same 10,000 rows, different sampling variability",
        xlabel="Dataset mean across 800 repetitions",
        ylabel="Density",
    )
    axes[1].legend()

    standard_errors = [np.std(independent), np.std(clustered)]
    bars = axes[2].bar(
        ["Independent rows", "Clustered rows"],
        standard_errors,
        color=[COLORS["blue"], COLORS["orange"]],
    )
    axes[2].bar_label(bars, fmt="%.3f", padding=3)
    axes[2].set(
        title="Dependence raises uncertainty despite equal row count",
        ylabel="Empirical standard error of dataset mean",
    )
    axes[2].set_ylim(0, max(standard_errors) * 1.2)
    figure.suptitle("Why do 10,000 rows not necessarily mean 10,000 independent observations?", y=1.02)
    figure.tight_layout()
    return [save_figure(figure, "correlated_observations.png")]


def calculate_group_split_results() -> dict[str, object]:
    """Measure identity leakage under row and group regression splits."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    rng = np.random.default_rng(SEED + 7)
    user_count = 180
    observations_per_user = 12
    users = np.repeat(np.arange(user_count), observations_per_user)
    user_effect = rng.normal(0.0, 30.0, user_count)
    target = 120.0 + user_effect[users] + rng.normal(0.0, 5.0, users.size)
    features = np.asarray([f"user_{user}" for user in users], dtype=object).reshape(-1, 1)
    indices = np.arange(users.size)
    random_train, random_test = train_test_split(
        indices,
        test_size=0.25,
        random_state=SEED,
    )
    group_train, group_test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(
            features,
            target,
            groups=users,
        )
    )

    def fit_and_score(train_indices: np.ndarray, test_indices: np.ndarray) -> float:
        pipeline = Pipeline(
            [
                (
                    "identity",
                    ColumnTransformer(
                        [("user", OneHotEncoder(handle_unknown="ignore"), [0])]
                    ),
                ),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        pipeline.fit(features[train_indices], target[train_indices])
        predictions = pipeline.predict(features[test_indices])
        return float(mean_absolute_error(target[test_indices], predictions))

    return {
        "users": users,
        "random_train": random_train,
        "random_test": random_test,
        "group_train": group_train,
        "group_test": group_test,
        "random_mae": fit_and_score(random_train, random_test),
        "group_mae": fit_and_score(group_train, group_test),
    }


def experiment_group_split() -> list[Path]:
    """Visualize user overlap and its measured effect on a simple model."""
    result = calculate_group_split_results()
    users = np.asarray(result["users"])
    selected_users = np.arange(16)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    def split_matrix(train_indices: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        matrix = np.zeros((len(selected_users), 12), dtype=int)
        train_set = set(train_indices.tolist())
        test_set = set(test_indices.tolist())
        for row, user in enumerate(selected_users):
            positions = np.flatnonzero(users == user)
            for column, index in enumerate(positions):
                if int(index) in train_set:
                    matrix[row, column] = 1
                elif int(index) in test_set:
                    matrix[row, column] = 2
        return matrix

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["white", COLORS["blue"], COLORS["orange"]])
    for axis, title, train_key, test_key in (
        (axes[0], "Random row split\nusers appear in both sets", "random_train", "random_test"),
        (axes[1], "Group split\nusers belong to one set", "group_train", "group_test"),
    ):
        matrix = split_matrix(
            np.asarray(result[train_key]),
            np.asarray(result[test_key]),
        )
        axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=2)
        axis.set(
            title=title,
            xlabel="Observation within user",
            ylabel="User ID (first 16)",
            yticks=np.arange(len(selected_users)),
            yticklabels=[f"U{user:02d}" for user in selected_users],
        )
    axes[0].legend(
        handles=[
            patches.Patch(color=COLORS["blue"], label="Train"),
            patches.Patch(color=COLORS["orange"], label="Test"),
        ],
        loc="lower right",
    )

    scores = [float(result["random_mae"]), float(result["group_mae"])]
    bars = axes[2].bar(
        ["Random row split", "Group split"],
        scores,
        color=[COLORS["blue"], COLORS["orange"]],
    )
    axes[2].bar_label(bars, fmt="%.2f", padding=3)
    axes[2].set(
        title="Actual error from the configured identity model",
        ylabel="Mean absolute error (lower is better)",
    )
    axes[2].set_ylim(0, max(scores) * 1.2)
    figure.suptitle("Can a row-level split leak repeated-user identity into evaluation?", y=1.02)
    figure.text(
        0.5,
        -0.02,
        "Synthetic regression uses only one-hot user identity and Ridge; this isolates split leakage, not model quality.",
        ha="center",
        color=COLORS["gray"],
    )
    figure.tight_layout()
    return [save_figure(figure, "group_split_vs_random_split.png")]


def calculate_llm_evaluation_results() -> dict[str, object]:
    """Return synthetic category mixes and deliberately assigned scores."""
    categories = np.array(
        ["Simple lookup", "Multi-step reasoning", "Summarization", "Edge cases"]
    )
    production_mix = np.array([0.60, 0.25, 0.10, 0.05])
    diagnostic_mix = np.full(4, 0.25)
    synthetic_scores = np.array([0.94, 0.79, 0.84, 0.52])
    return {
        "categories": categories,
        "production_mix": production_mix,
        "diagnostic_mix": diagnostic_mix,
        "synthetic_scores": synthetic_scores,
        "diagnostic_average": float(np.dot(diagnostic_mix, synthetic_scores)),
        "production_weighted": float(np.dot(production_mix, synthetic_scores)),
    }


def experiment_llm_evaluation() -> list[Path]:
    """Separate capability diagnosis from production-weighted estimation."""
    result = calculate_llm_evaluation_results()
    categories = np.asarray(result["categories"])
    production_mix = np.asarray(result["production_mix"])
    diagnostic_mix = np.asarray(result["diagnostic_mix"])
    scores = np.asarray(result["synthetic_scores"])
    x_positions = np.arange(len(categories))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    axes[0].bar(
        x_positions - 0.18,
        production_mix,
        width=0.36,
        color=COLORS["blue"],
        label="Production mix",
    )
    axes[0].bar(
        x_positions + 0.18,
        diagnostic_mix,
        width=0.36,
        color=COLORS["orange"],
        label="Balanced diagnostic mix",
    )
    axes[0].set(
        title="Two evaluation sets sample different questions",
        ylabel="Category share",
        xticks=x_positions,
        xticklabels=categories,
        ylim=(0, 0.7),
    )
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend()

    score_bars = axes[1].bar(categories, scores, color=COLORS["purple"])
    axes[1].bar_label(score_bars, fmt="%.2f", padding=3)
    axes[1].set(
        title="Synthetic category performance",
        ylabel="Synthetic score",
        ylim=(0, 1.05),
    )
    axes[1].tick_params(axis="x", rotation=25)

    aggregate_labels = ["Diagnostic\nunweighted", "Production\nweighted"]
    aggregate_values = [
        float(result["diagnostic_average"]),
        float(result["production_weighted"]),
    ]
    aggregate_bars = axes[2].bar(
        aggregate_labels,
        aggregate_values,
        color=[COLORS["orange"], COLORS["blue"]],
    )
    axes[2].bar_label(aggregate_bars, fmt="%.3f", padding=3)
    axes[2].set(
        title="Different weights answer different questions",
        ylabel="Aggregate synthetic score",
        ylim=(0, 1.05),
    )
    axes[2].text(
        0.5,
        0.08,
        "Diagnostic: where does it struggle?\nProduction: what should traffic experience?",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=9,
    )

    figure.suptitle("LLM/RAG evaluation sampling: coverage and prevalence are not the same", y=1.02)
    figure.text(
        0.5,
        -0.03,
        "Synthetic illustration only. Category scores and traffic shares are constructed, not benchmark measurements.",
        ha="center",
        color=COLORS["gray"],
    )
    figure.tight_layout()
    return [save_figure(figure, "llm_evaluation_sampling.png")]


def generate_static_experiments() -> list[Path]:
    """Generate every static experiment and return successful outputs."""
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
    ):
        generated.extend(generator())
    return generated
