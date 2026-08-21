"""Offline visual learning lab for maximum likelihood and MAP estimation.

Every dataset is synthetic or analytically specified. Static figures and GIFs
are generated only with ``--save``; normal single-demo execution displays the
figure locally. Use ``--quick`` for bounded smoke renders.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import beta as beta_distribution

from example import (
    bernoulli_log_likelihood,
    bernoulli_mle,
    beta_bernoulli_log_posterior,
    beta_bernoulli_map,
    grid_mode,
)
from from_scratch import (
    fit_logistic_regression,
    negative_log_posterior,
    negative_log_posterior_gradient,
    sigmoid,
)


DEFAULT_OUTPUT_DIRECTORY = Path(__file__).with_name("assets")
BLUE = "#2563EB"
ORANGE = "#EA580C"
GREEN = "#059669"
PURPLE = "#7C3AED"
RED = "#DC2626"
SLATE = "#334155"
LIGHT_BLUE = "#DBEAFE"
LIGHT_PURPLE = "#EDE9FE"
GRID = "#CBD5E1"


@dataclass(frozen=True)
class VisualResult:
    """Describe one rendered visual and its computed observations."""

    label: str
    path: Path | None
    metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class LogisticExperiment:
    """Store one deterministic single-feature logistic experiment."""

    x: np.ndarray
    labels: np.ndarray
    features: np.ndarray
    true_weights: np.ndarray
    mle_weights: np.ndarray
    map_weights: np.ndarray
    prior_std: float


def configure_plot_style() -> None:
    """Apply one restrained visual language across the learning narrative."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": GRID,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "axes.titleweight": "bold",
            "axes.labelcolor": SLATE,
            "text.color": SLATE,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def _finish_static(
    figure: plt.Figure,
    path: Path,
    *,
    save: bool,
    show: bool,
    tight_layout: bool = True,
) -> Path | None:
    if tight_layout:
        figure.tight_layout()
    written_path = None
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        written_path = path
    if show:
        plt.show()
    plt.close(figure)
    return written_path


def _finish_animation(
    animation: FuncAnimation,
    figure: plt.Figure,
    path: Path,
    *,
    save: bool,
    show: bool,
    fps: int,
) -> Path | None:
    written_path = None
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(path, writer=PillowWriter(fps=fps))
        written_path = path
    if show:
        plt.show()
    plt.close(figure)
    return written_path


def _relative_from_log(log_values: np.ndarray) -> np.ndarray:
    """Scale finite log values to a maximum of one for display only."""
    log_values = np.asarray(log_values, dtype=float)
    maximum = float(np.max(log_values))
    return np.exp(log_values - maximum)


def _beta_mode(alpha: float, beta: float) -> float:
    if alpha <= 1.0 or beta <= 1.0:
        raise ValueError("this visual requires an interior Beta mode.")
    return (alpha - 1.0) / (alpha + beta - 2.0)


def plot_probability_vs_likelihood(
    output_dir: Path, *, save: bool, show: bool
) -> VisualResult:
    """Contrast a Bernoulli probability model with a parameter likelihood."""
    probability = 0.7
    successes, trials = 7, 10
    grid = np.linspace(0.001, 0.999, 1_200)
    relative = _relative_from_log(
        bernoulli_log_likelihood(grid, successes, trials)
    )
    mle = bernoulli_mle(successes, trials)

    figure, (probability_axis, likelihood_axis) = plt.subplots(
        1, 2, figsize=(11.5, 4.8)
    )
    probability_axis.bar(
        [0, 1], [1.0 - probability, probability], color=[ORANGE, BLUE], width=0.62
    )
    probability_axis.set(
        xticks=[0, 1],
        ylim=(0.0, 1.0),
        xlabel="Possible observation x",
        ylabel="P(X=x | p=0.7)",
        title="Probability: parameter fixed",
    )
    probability_axis.text(
        0.5,
        0.93,
        "p fixed → observations vary",
        transform=probability_axis.transAxes,
        ha="center",
        fontweight="bold",
    )

    likelihood_axis.plot(grid, relative, color=PURPLE, linewidth=2.8)
    likelihood_axis.axvline(mle, color=RED, linestyle="--", linewidth=2)
    likelihood_axis.scatter([mle], [1.0], color=RED, zorder=4)
    likelihood_axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.08),
        xlabel="Candidate Bernoulli parameter p",
        ylabel="Relative likelihood (max scaled to 1)",
        title="Likelihood: observations fixed (7 successes, 3 failures)",
    )
    likelihood_axis.text(
        mle,
        0.83,
        "MLE = 0.70",
        ha="center",
        color=RED,
        fontweight="bold",
    )
    likelihood_axis.text(
        0.5,
        0.08,
        "data fixed → candidate parameters vary\nnot a probability density over p",
        transform=likelihood_axis.transAxes,
        ha="center",
        bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.92),
    )
    figure.suptitle("The same model expression answers two different questions", fontsize=15)
    path = output_dir / "01_probability_vs_likelihood.png"
    return VisualResult(
        "Probability versus likelihood",
        _finish_static(figure, path, save=save, show=show),
        (f"Analytical MLE: {mle:.6f}",),
    )


def animate_likelihood_accumulation(
    output_dir: Path,
    *,
    seed: int,
    save: bool,
    show: bool,
    quick: bool,
) -> VisualResult:
    """Animate relative Bernoulli likelihood as observations accumulate."""
    true_probability = 0.7
    observations = np.random.default_rng(seed).binomial(1, true_probability, 100)
    cumulative_successes = np.cumsum(observations)
    grid = np.linspace(0.001, 0.999, 900)
    frame_sizes = (
        np.unique(np.linspace(1, 100, 18, dtype=int))
        if quick
        else np.arange(1, 101)
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.5), dpi=85 if quick else 100)

    def update(frame: int) -> None:
        sample_size = int(frame_sizes[frame])
        successes = int(cumulative_successes[sample_size - 1])
        failures = sample_size - successes
        mle = successes / sample_size
        relative = _relative_from_log(
            bernoulli_log_likelihood(grid, successes, sample_size)
        )
        axis.clear()
        axis.plot(grid, relative, color=PURPLE, linewidth=2.8)
        axis.fill_between(grid, relative, color=PURPLE, alpha=0.14)
        axis.axvline(true_probability, color=GREEN, linewidth=2, label="true p = 0.70")
        axis.axvline(mle, color=RED, linestyle="--", linewidth=2, label=f"MLE = {mle:.3f}")
        axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.08),
            xlabel="Candidate p",
            ylabel="Relative likelihood (max scaled to 1)",
            title="Evidence reshapes the likelihood one observation at a time",
        )
        axis.legend(loc="upper left")
        axis.text(
            0.98,
            0.94,
            f"n = {sample_size}\nsuccesses = {successes}\nfailures = {failures}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.94),
        )
        axis.text(
            0.5,
            0.04,
            "More data usually concentrates likelihood under this identifiable Bernoulli model.",
            transform=axis.transAxes,
            ha="center",
            fontsize=9,
        )

    animation = FuncAnimation(
        figure, update, frames=len(frame_sizes), interval=130 if quick else 85, repeat=True
    )
    path = output_dir / "likelihood_accumulation.gif"
    written = _finish_animation(
        animation, figure, path, save=save, show=show, fps=7 if quick else 12
    )
    final_successes = int(cumulative_successes[-1])
    return VisualResult(
        "Likelihood accumulation",
        written,
        (
            f"Seed: {seed}",
            f"Final successes: {final_successes}/100",
            f"Final MLE: {final_successes / 100:.6f}",
        ),
    )


def plot_likelihood_log_likelihood(
    output_dir: Path, *, save: bool, show: bool
) -> VisualResult:
    """Show equivalent optima and floating-point underflow in direct products."""
    successes, trials = 70, 100
    grid = np.linspace(0.01, 0.99, 1_000)
    log_likelihood = bernoulli_log_likelihood(grid, successes, trials)
    relative = _relative_from_log(log_likelihood)
    shifted_log = log_likelihood - np.max(log_likelihood)
    shifted_nll = -shifted_log
    mle = successes / trials

    large_trials = 5_000
    large_successes = 3_500
    direct_product = 0.7**large_successes * 0.3 ** (large_trials - large_successes)
    stable_log = float(
        bernoulli_log_likelihood(0.7, large_successes, large_trials)
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.5))
    specifications = (
        (relative, "Relative likelihood", "maximize", PURPLE),
        (shifted_log, "Log-likelihood (shifted)", "maximize", BLUE),
        (shifted_nll, "Negative log-likelihood (shifted)", "minimize", ORANGE),
    )
    for axis, (values, title, objective, color) in zip(axes, specifications, strict=True):
        axis.plot(grid, values, color=color, linewidth=2.6)
        axis.axvline(mle, color=RED, linestyle="--", linewidth=1.8)
        axis.set(xlabel="Candidate p", title=title)
        axis.text(
            0.04,
            0.92,
            f"{objective} at p = {mle:.2f}",
            transform=axis.transAxes,
            va="top",
            color=RED,
            fontweight="bold",
        )
    figure.suptitle("Log transforms the scale—not the optimum", fontsize=15)
    figure.text(
        0.5,
        0.01,
        (
            f"5,000 observations at p=0.7: direct product = {direct_product:.1e}; "
            f"summed log-likelihood = {stable_log:.2f}"
        ),
        ha="center",
        bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.95),
    )
    path = output_dir / "03_log_likelihood.png"
    return VisualResult(
        "Likelihood, log-likelihood, and NLL",
        _finish_static(figure, path, save=save, show=show),
        (
            f"Shared optimum: {mle:.6f}",
            f"Direct likelihood product: {direct_product:.1e}",
            f"Stable log-likelihood: {stable_log:.6f}",
        ),
    )


def plot_prior_likelihood_posterior(
    output_dir: Path, *, save: bool, show: bool
) -> VisualResult:
    """Visualize the conjugate Beta-Bernoulli update and its summaries."""
    successes, trials = 7, 10
    alpha, beta = 2.0, 2.0
    posterior_alpha = alpha + successes
    posterior_beta = beta + trials - successes
    grid = np.linspace(0.001, 0.999, 1_400)
    prior = beta_distribution.pdf(grid, alpha, beta)
    likelihood = _relative_from_log(
        bernoulli_log_likelihood(grid, successes, trials)
    )
    posterior = beta_distribution.pdf(grid, posterior_alpha, posterior_beta)
    mle = bernoulli_mle(successes, trials)
    map_estimate = beta_bernoulli_map(
        successes, trials, alpha=alpha, beta=beta
    )
    prior_mean = alpha / (alpha + beta)
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), sharex=True)
    axes[0].plot(grid, prior, color=BLUE, linewidth=2.8)
    axes[0].fill_between(grid, prior, color=BLUE, alpha=0.13)
    axes[0].axvline(prior_mean, color=BLUE, linestyle="--")
    axes[0].set(title="1. Prior belief: Beta(2, 2)", ylabel="Density")
    axes[0].text(prior_mean, max(prior) * 0.75, "prior mean", ha="center", color=BLUE)

    axes[1].plot(grid, likelihood, color=ORANGE, linewidth=2.8)
    axes[1].fill_between(grid, likelihood, color=ORANGE, alpha=0.13)
    axes[1].axvline(mle, color=RED, linestyle="--")
    axes[1].set(title="2. Evidence: 7 successes, 3 failures", ylabel="Relative likelihood")
    axes[1].text(mle, 0.75, "MLE", ha="center", color=RED, fontweight="bold")

    axes[2].plot(grid, posterior, color=PURPLE, linewidth=2.8)
    axes[2].fill_between(grid, posterior, color=PURPLE, alpha=0.13)
    axes[2].axvline(map_estimate, color=RED, linestyle="--", label=f"MAP = {map_estimate:.3f}")
    axes[2].axvline(posterior_mean, color=GREEN, linestyle=":", linewidth=2, label=f"mean = {posterior_mean:.3f}")
    axes[2].set(title="3. Posterior: Beta(9, 5)", ylabel="Density")
    axes[2].legend(loc="upper left")
    for axis in axes:
        axis.set(xlim=(0.0, 1.0), xlabel="Bernoulli parameter p")
    figure.suptitle("Prior preference + observed evidence → posterior belief", fontsize=15)
    figure.text(
        0.5,
        0.015,
        "The likelihood is max-scaled only for display; it is not a density over p.",
        ha="center",
        fontsize=9,
    )
    path = output_dir / "prior_likelihood_posterior.png"
    return VisualResult(
        "Prior, likelihood, and posterior",
        _finish_static(figure, path, save=save, show=show),
        (
            f"MLE: {mle:.6f}",
            f"MAP: {map_estimate:.6f}",
            f"Posterior mean: {posterior_mean:.6f}",
        ),
    )


def plot_prior_strength(
    output_dir: Path, *, save: bool, show: bool
) -> VisualResult:
    """Compare how symmetric Beta prior concentration changes MAP."""
    successes, trials = 7, 10
    strengths = (1.0, 2.0, 10.0, 50.0)
    grid = np.linspace(0.001, 0.999, 1_200)
    mle = successes / trials
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True, sharey=True)
    map_values = []
    for axis, strength in zip(axes.flat, strengths, strict=True):
        prior = beta_distribution.pdf(grid, strength, strength)
        posterior = beta_distribution.pdf(
            grid, strength + successes, strength + trials - successes
        )
        prior_scaled = prior / np.max(prior)
        posterior_scaled = posterior / np.max(posterior)
        map_estimate = beta_bernoulli_map(
            successes, trials, alpha=strength, beta=strength
        )
        map_values.append(map_estimate)
        axis.plot(grid, prior_scaled, color=BLUE, linestyle="--", label="prior density (scaled)")
        axis.plot(grid, posterior_scaled, color=PURPLE, linewidth=2.4, label="posterior density (scaled)")
        axis.axvline(mle, color=ORANGE, linestyle=":", linewidth=2, label="MLE")
        axis.axvline(map_estimate, color=RED, linestyle="--", linewidth=2, label="MAP")
        axis.set(title=f"Beta({strength:g}, {strength:g}) → MAP {map_estimate:.3f}", xlim=(0, 1), ylim=(0, 1.08))
    axes[0, 0].legend(loc="upper left", fontsize=8)
    figure.supxlabel("Candidate p")
    figure.supylabel("Each curve independently scaled to a maximum of 1")
    figure.suptitle("Prior concentration controls injected evidence", fontsize=15)
    path = output_dir / "05_prior_strength.png"
    return VisualResult(
        "Prior strength",
        _finish_static(figure, path, save=save, show=show),
        tuple(
            f"Beta({strength:g}, {strength:g}) MAP: {estimate:.6f}"
            for strength, estimate in zip(strengths, map_values, strict=True)
        ),
    )


def _cumulative_estimates(
    observations: np.ndarray, *, alpha: float, beta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_sizes = np.arange(1, observations.size + 1)
    successes = np.cumsum(observations)
    mle = successes / sample_sizes
    map_estimates = (successes + alpha - 1.0) / (
        sample_sizes + alpha + beta - 2.0
    )
    return sample_sizes, mle, map_estimates


def plot_sample_size_effect(
    output_dir: Path, *, seed: int, save: bool, show: bool
) -> VisualResult:
    """Show fixed-prior MLE/MAP behavior along one accumulating sample path."""
    true_probability = 0.7
    alpha = beta = 5.0
    observations = np.random.default_rng(seed).binomial(1, true_probability, 1_000)
    sizes, mle, map_estimates = _cumulative_estimates(
        observations, alpha=alpha, beta=beta
    )
    checkpoints = np.array([5, 10, 30, 100, 1_000])

    figure, (estimate_axis, gap_axis) = plt.subplots(
        2, 1, figsize=(10.5, 7.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    estimate_axis.semilogx(sizes, mle, color=ORANGE, alpha=0.78, label="MLE")
    estimate_axis.semilogx(sizes, map_estimates, color=PURPLE, linewidth=2.2, label="MAP · Beta(5, 5)")
    estimate_axis.axhline(true_probability, color=GREEN, linestyle="--", linewidth=2, label="generator p = 0.70")
    estimate_axis.scatter(checkpoints, mle[checkpoints - 1], color=ORANGE, zorder=4)
    estimate_axis.scatter(checkpoints, map_estimates[checkpoints - 1], color=PURPLE, zorder=4)
    estimate_axis.set(ylabel="Point estimate", title="A fixed prior becomes relatively smaller along this sample path")
    estimate_axis.legend(loc="best")
    gap_axis.loglog(sizes, np.maximum(np.abs(mle - map_estimates), 1e-8), color=RED, linewidth=2)
    gap_axis.set(xlabel="Accumulated observations n (log scale)", ylabel="|MLE − MAP|", title="Realized point-estimate gap")
    figure.text(
        0.5,
        0.01,
        "Constructed IID Bernoulli sequence; the convergence message assumes a fixed prior and regular identifiable model.",
        ha="center",
        fontsize=9,
    )
    path = output_dir / "06_sample_size_effect.png"
    return VisualResult(
        "Sample size and fixed-prior influence",
        _finish_static(figure, path, save=save, show=show),
        tuple(
            f"n={n}: MLE={mle[n - 1]:.6f}, MAP={map_estimates[n - 1]:.6f}"
            for n in checkpoints
        ),
    )


def plot_wrong_prior(
    output_dir: Path, *, seed: int, save: bool, show: bool
) -> VisualResult:
    """Show how a strong misspecified prior can dominate limited data."""
    true_probability = 0.3
    alpha, beta = 20.0, 2.0
    observations = np.random.default_rng(seed).binomial(1, true_probability, 1_000)
    sizes, mle, map_estimates = _cumulative_estimates(
        observations, alpha=alpha, beta=beta
    )
    prior_mode = _beta_mode(alpha, beta)

    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    axis.semilogx(sizes, mle, color=ORANGE, alpha=0.75, label="MLE")
    axis.semilogx(sizes, map_estimates, color=PURPLE, linewidth=2.4, label="MAP · wrong Beta(20, 2) prior")
    axis.axhline(true_probability, color=GREEN, linestyle="--", linewidth=2, label="generator p = 0.30")
    axis.axhline(prior_mode, color=RED, linestyle=":", linewidth=2, label=f"prior mode = {prior_mode:.2f}")
    axis.set(
        ylim=(0.0, 1.0),
        xlabel="Accumulated observations n (log scale)",
        ylabel="Point estimate",
        title="A strong wrong prior can hurt before evidence overcomes it",
    )
    axis.legend(loc="best")
    axis.text(
        0.03,
        0.07,
        "This recovery is demonstrated only for a fixed prior and correctly specified IID Bernoulli model.",
        transform=axis.transAxes,
        bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.94),
    )
    path = output_dir / "07_wrong_prior.png"
    return VisualResult(
        "Misspecified prior",
        _finish_static(figure, path, save=save, show=show),
        (
            f"Prior mode: {prior_mode:.6f}",
            f"MAP at n=10: {map_estimates[9]:.6f}",
            f"MAP at n=1000: {map_estimates[-1]:.6f}",
            f"MLE at n=1000: {mle[-1]:.6f}",
        ),
    )


def gaussian_regression_objectives(
    *, seed: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return synthetic regression data and slope-indexed SSE/Gaussian NLL."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.5, 2.5, 55)
    intercept = 0.4
    true_slope = 1.8
    noise_std = 0.7
    y = intercept + true_slope * x + rng.normal(0.0, noise_std, x.size)
    candidates = np.linspace(0.4, 3.2, 500)
    residuals = y[None, :] - (intercept + candidates[:, None] * x[None, :])
    sse = np.sum(residuals**2, axis=1)
    gaussian_nll = (
        0.5 * x.size * math.log(2.0 * math.pi * noise_std**2)
        + sse / (2.0 * noise_std**2)
    )
    return x, y, candidates, sse, gaussian_nll, intercept, noise_std


def plot_gaussian_nll_vs_sse(
    output_dir: Path, *, seed: int, save: bool, show: bool
) -> VisualResult:
    """Make Gaussian NLL and squared-error optimization visibly equivalent."""
    x, y, slopes, sse, nll, intercept, noise_std = gaussian_regression_objectives(seed=seed)
    sse_index = int(np.argmin(sse))
    nll_index = int(np.argmin(nll))
    if sse_index != nll_index:
        raise AssertionError("Gaussian NLL and SSE grid minima must coincide.")
    best_slope = float(slopes[sse_index])
    rescaled_nll = 2.0 * noise_std**2 * (
        nll - 0.5 * x.size * math.log(2.0 * math.pi * noise_std**2)
    )

    figure, (curve_axis, data_axis) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    curve_axis.plot(slopes, sse, color=BLUE, linewidth=3, label="SSE")
    curve_axis.plot(slopes, rescaled_nll, color=ORANGE, linestyle="--", linewidth=2, label="Gaussian NLL after removing constant and scaling")
    curve_axis.axvline(best_slope, color=RED, linestyle=":", linewidth=2)
    curve_axis.set(xlabel="Candidate slope", ylabel="Objective on comparable scale", title="Fixed-variance Gaussian NLL and SSE share a minimum")
    curve_axis.legend(fontsize=8)
    data_axis.scatter(x, y, color=SLATE, alpha=0.72, s=28, label="synthetic observations")
    data_axis.plot(x, intercept + best_slope * x, color=RED, linewidth=2.5, label=f"best slope = {best_slope:.3f}")
    data_axis.set(xlabel="x", ylabel="y", title="Synthetic Gaussian regression")
    data_axis.legend()
    figure.suptitle("Gaussian residual model → NLL → squared residuals → SSE/MSE", fontsize=15)
    path = output_dir / "08_gaussian_nll_vs_sse.png"
    return VisualResult(
        "Gaussian likelihood and squared error",
        _finish_static(figure, path, save=save, show=show),
        (
            f"SSE grid argmin slope: {slopes[sse_index]:.6f}",
            f"Gaussian NLL grid argmin slope: {slopes[nll_index]:.6f}",
            f"Maximum curve discrepancy after exact rescaling: {np.max(np.abs(sse - rescaled_nll)):.3e}",
        ),
    )


def plot_bce(
    output_dir: Path, *, save: bool, show: bool
) -> VisualResult:
    """Show Bernoulli NLL as BCE and its asymmetric confidence penalty."""
    probabilities = np.linspace(0.001, 0.999, 1_000)
    positive_loss = -np.log(probabilities)
    negative_loss = -np.log1p(-probabilities)
    selected = np.array([0.99, 0.70, 0.50, 0.10, 0.01])
    selected_losses = -np.log(selected)
    labels = np.array([1.0, 0.0, 1.0, 1.0])
    predictions = np.array([0.8, 0.2, 0.55, 0.05])
    bernoulli_nll = -np.sum(
        labels * np.log(predictions) + (1.0 - labels) * np.log1p(-predictions)
    )
    bce = float(
        np.sum(
            np.where(labels == 1.0, -np.log(predictions), -np.log1p(-predictions))
        )
    )
    if not np.isclose(bernoulli_nll, bce):
        raise AssertionError("manual Bernoulli NLL and BCE must match.")

    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    axis.plot(probabilities, positive_loss, color=BLUE, linewidth=2.8, label="y=1: −log(p)")
    axis.plot(probabilities, negative_loss, color=ORANGE, linewidth=2.8, label="y=0: −log(1−p)")
    axis.scatter(selected, selected_losses, color=RED, zorder=4)
    for probability, loss in zip(selected, selected_losses, strict=True):
        axis.annotate(f"p={probability:.2f}\nloss={loss:.2f}", (probability, loss), xytext=(4, 7), textcoords="offset points", fontsize=8)
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 7.0),
        xlabel="Predicted probability of y=1",
        ylabel="Per-observation loss",
        title="Binary cross-entropy is Bernoulli negative log-likelihood",
    )
    axis.legend()
    axis.text(
        0.5,
        0.88,
        "Confident and wrong → very large loss",
        transform=axis.transAxes,
        ha="center",
        color=RED,
        fontweight="bold",
    )
    path = output_dir / "09_bce_curve.png"
    return VisualResult(
        "Bernoulli NLL and BCE",
        _finish_static(figure, path, save=save, show=show),
        (
            f"Manual Bernoulli NLL: {bernoulli_nll:.12f}",
            f"Manual BCE: {bce:.12f}",
            f"Absolute difference: {abs(bernoulli_nll - bce):.3e}",
        ),
    )


def build_logistic_experiment(
    *, seed: int = 14, sample_size: int = 80, prior_std: float = 0.75
) -> LogisticExperiment:
    """Generate a stable single-feature logistic example and fit MLE/MAP."""
    if sample_size < 30:
        raise ValueError("sample_size must be at least 30.")
    rng = np.random.default_rng(seed)
    x = np.sort(rng.normal(0.0, 1.0, sample_size))
    features = np.column_stack((np.ones(sample_size), x))
    true_weights = np.array([-0.35, 1.8])
    labels = rng.binomial(1, sigmoid(features @ true_weights)).astype(float)
    mle = fit_logistic_regression(features, labels, learning_rate=0.3)
    map_fit = fit_logistic_regression(
        features, labels, prior_std=prior_std, learning_rate=0.3
    )
    return LogisticExperiment(
        x=x,
        labels=labels,
        features=features,
        true_weights=true_weights,
        mle_weights=mle.weights,
        map_weights=map_fit.weights,
        prior_std=prior_std,
    )


def logistic_objective_grid(
    experiment: LogisticExperiment,
    intercept_values: np.ndarray,
    slope_values: np.ndarray,
    *,
    prior_std: float | None = None,
) -> np.ndarray:
    """Evaluate summed logistic NLL or negative log-posterior on a 2D grid."""
    intercept_grid, slope_grid = np.meshgrid(intercept_values, slope_values)
    logits = (
        intercept_grid[..., None]
        + slope_grid[..., None] * experiment.x[None, None, :]
    )
    objective = np.sum(
        np.logaddexp(0.0, logits)
        - experiment.labels[None, None, :] * logits,
        axis=-1,
    )
    if prior_std is not None:
        objective += slope_grid**2 / (2.0 * prior_std**2)
    return objective


def _surface_ranges(experiment: LogisticExperiment) -> tuple[np.ndarray, np.ndarray]:
    centers = np.vstack((experiment.mle_weights, experiment.map_weights, experiment.true_weights))
    intercept_values = np.linspace(min(-2.5, np.min(centers[:, 0]) - 1.0), max(1.5, np.max(centers[:, 0]) + 1.0), 95)
    slope_values = np.linspace(min(-1.5, np.min(centers[:, 1]) - 1.0), max(4.0, np.max(centers[:, 1]) + 1.0), 105)
    return intercept_values, slope_values


def plot_logistic_surface(
    output_dir: Path, *, seed: int, save: bool, show: bool, quick: bool
) -> VisualResult:
    """Render 2D contours and a 3D logistic NLL surface with the MLE."""
    experiment = build_logistic_experiment(seed=seed, sample_size=60 if quick else 80)
    intercept_values, slope_values = _surface_ranges(experiment)
    if quick:
        intercept_values = intercept_values[::2]
        slope_values = slope_values[::2]
    nll = logistic_objective_grid(experiment, intercept_values, slope_values)
    intercept_grid, slope_grid = np.meshgrid(intercept_values, slope_values)

    figure = plt.figure(figsize=(13.5, 5.5))
    contour_axis = figure.add_subplot(1, 2, 1)
    surface_axis = figure.add_subplot(1, 2, 2, projection="3d")
    contour = contour_axis.contourf(slope_grid, intercept_grid, nll, levels=28, cmap="viridis")
    contour_axis.scatter(experiment.mle_weights[1], experiment.mle_weights[0], color=RED, marker="*", s=180, edgecolor="white", label="MLE")
    contour_axis.set(xlabel="Slope w", ylabel="Intercept b", title="2D NLL contours")
    figure.colorbar(contour, ax=contour_axis, label="Summed NLL")
    surface_axis.plot_surface(slope_grid, intercept_grid, nll, cmap="viridis", alpha=0.88, linewidth=0, antialiased=True)
    mle_nll = negative_log_posterior(experiment.features, experiment.labels, experiment.mle_weights)
    surface_axis.scatter(experiment.mle_weights[1], experiment.mle_weights[0], mle_nll, color=RED, marker="*", s=120)
    surface_axis.set(xlabel="Slope w", ylabel="Intercept b", zlabel="NLL", title="3D optimization landscape")
    surface_axis.view_init(elev=28, azim=-58)
    figure.suptitle("Logistic MLE is the minimum of Bernoulli negative log-likelihood", fontsize=15)
    figure.subplots_adjust(left=0.06, right=0.94, bottom=0.10, top=0.84, wspace=0.28)
    path = output_dir / "logistic_surface.png"
    return VisualResult(
        "Logistic likelihood surface",
        _finish_static(figure, path, save=save, show=show, tight_layout=False),
        (
            f"MLE intercept: {experiment.mle_weights[0]:.6f}",
            f"MLE slope: {experiment.mle_weights[1]:.6f}",
            f"MLE summed NLL: {mle_nll:.6f}",
        ),
    )


def gradient_descent_path(
    experiment: LogisticExperiment,
    *,
    prior_std: float | None = None,
    iterations: int = 55,
    learning_rate: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic batch-gradient path and its mean objectives."""
    if iterations < 2:
        raise ValueError("iterations must be at least two.")
    weights = np.array([1.35, -1.25], dtype=float)
    path = [weights.copy()]
    objectives = [
        negative_log_posterior(
            experiment.features,
            experiment.labels,
            weights,
            prior_std=prior_std,
            reduction="mean",
        )
    ]
    for _ in range(iterations):
        gradient = negative_log_posterior_gradient(
            experiment.features,
            experiment.labels,
            weights,
            prior_std=prior_std,
            reduction="mean",
        )
        weights = weights - learning_rate * gradient
        path.append(weights.copy())
        objectives.append(
            negative_log_posterior(
                experiment.features,
                experiment.labels,
                weights,
                prior_std=prior_std,
                reduction="mean",
            )
        )
    return np.asarray(path), np.asarray(objectives)


def animate_gradient_descent(
    output_dir: Path,
    *,
    seed: int,
    save: bool,
    show: bool,
    quick: bool,
) -> VisualResult:
    """Animate model training as movement over the logistic NLL contours."""
    experiment = build_logistic_experiment(seed=seed, sample_size=60 if quick else 80)
    iterations = 22 if quick else 55
    path_values, objectives = gradient_descent_path(experiment, iterations=iterations)
    intercept_values, slope_values = _surface_ranges(experiment)
    nll = logistic_objective_grid(experiment, intercept_values, slope_values) / experiment.labels.size
    intercept_grid, slope_grid = np.meshgrid(intercept_values, slope_values)
    figure, axis = plt.subplots(figsize=(8.7, 6.0), dpi=85 if quick else 100)

    def update(frame: int) -> None:
        axis.clear()
        axis.contourf(slope_grid, intercept_grid, nll, levels=26, cmap="viridis", alpha=0.93)
        axis.plot(path_values[: frame + 1, 1], path_values[: frame + 1, 0], color="white", linewidth=2, marker="o", markersize=3)
        axis.scatter(path_values[frame, 1], path_values[frame, 0], color=RED, s=80, zorder=5)
        axis.scatter(experiment.mle_weights[1], experiment.mle_weights[0], marker="*", color=ORANGE, edgecolor="white", s=170, label="fitted MLE")
        axis.set(xlabel="Slope w", ylabel="Intercept b", title="Gradient descent minimizes negative log-likelihood")
        axis.legend(loc="upper right")
        axis.text(
            0.03,
            0.96,
            f"iteration = {frame}\nmean NLL = {objectives[frame]:.5f}\nb = {path_values[frame, 0]:.3f}\nw = {path_values[frame, 1]:.3f}",
            transform=axis.transAxes,
            va="top",
            bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.94),
        )

    animation = FuncAnimation(figure, update, frames=len(path_values), interval=180 if quick else 120, repeat=True)
    path = output_dir / "mle_optimization.gif"
    written = _finish_animation(animation, figure, path, save=save, show=show, fps=6 if quick else 9)
    return VisualResult(
        "Gradient descent on logistic NLL",
        written,
        (
            f"Initial mean NLL: {objectives[0]:.6f}",
            f"Final animated mean NLL: {objectives[-1]:.6f}",
            f"Animated final weights: {np.array2string(path_values[-1], precision=6)}",
        ),
    )


def plot_mle_vs_map_surface(
    output_dir: Path, *, seed: int, save: bool, show: bool, quick: bool
) -> VisualResult:
    """Show how a Gaussian slope prior changes logistic objective geometry."""
    experiment = build_logistic_experiment(seed=seed, sample_size=60 if quick else 80)
    intercept_values, slope_values = _surface_ranges(experiment)
    if quick:
        intercept_values = intercept_values[::2]
        slope_values = slope_values[::2]
    mle_objective = logistic_objective_grid(experiment, intercept_values, slope_values)
    map_objective = logistic_objective_grid(
        experiment, intercept_values, slope_values, prior_std=experiment.prior_std
    )
    intercept_grid, slope_grid = np.meshgrid(intercept_values, slope_values)
    mle_path, _ = gradient_descent_path(experiment, iterations=45)
    map_path, _ = gradient_descent_path(
        experiment, prior_std=experiment.prior_std, iterations=45
    )

    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), sharex=True, sharey=True)
    for axis, objective, title, solution, path_values in (
        (axes[0], mle_objective, "MLE: NLL", experiment.mle_weights, mle_path),
        (axes[1], map_objective, f"MAP: NLL + Gaussian prior (σ={experiment.prior_std})", experiment.map_weights, map_path),
    ):
        contour = axis.contourf(slope_grid, intercept_grid, objective, levels=28, cmap="viridis")
        axis.plot(path_values[:, 1], path_values[:, 0], color="white", linewidth=1.8, alpha=0.9)
        axis.scatter(solution[1], solution[0], color=RED, marker="*", edgecolor="white", s=180)
        axis.set(xlabel="Slope w", ylabel="Intercept b", title=title)
        figure.colorbar(contour, ax=axis, shrink=0.82)
    figure.suptitle("A zero-mean Gaussian prior bends the surface toward smaller slopes", fontsize=15)
    figure.text(
        0.5,
        0.01,
        "The intercept is not regularized. The penalty matches a summed NLL objective: w²/(2σ²).",
        ha="center",
        fontsize=9,
    )
    path = output_dir / "mle_vs_map_surface.png"
    return VisualResult(
        "MLE versus MAP objective surfaces",
        _finish_static(figure, path, save=save, show=show),
        (
            f"MLE slope: {experiment.mle_weights[1]:.6f}",
            f"MAP slope: {experiment.map_weights[1]:.6f}",
            f"Gaussian prior standard deviation: {experiment.prior_std:.6f}",
        ),
    )


def plot_prior_variance(
    output_dir: Path, *, seed: int, save: bool, show: bool
) -> VisualResult:
    """Compute coefficient shrinkage across Gaussian prior standard deviations."""
    experiment = build_logistic_experiment(seed=seed)
    prior_stds = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
    fits = [
        fit_logistic_regression(
            experiment.features,
            experiment.labels,
            prior_std=float(prior_std),
            learning_rate=0.3,
        )
        for prior_std in prior_stds
    ]
    slopes = np.array([abs(fit.weights[1]) for fit in fits])
    training_nll = np.array(
        [
            negative_log_posterior(
                experiment.features,
                experiment.labels,
                fit.weights,
                reduction="mean",
            )
            for fit in fits
        ]
    )
    mle_slope = abs(experiment.mle_weights[1])

    figure, coefficient_axis = plt.subplots(figsize=(9.5, 5.6))
    coefficient_axis.semilogx(prior_stds, slopes, color=PURPLE, marker="o", linewidth=2.5, label="|MAP slope|")
    coefficient_axis.axhline(mle_slope, color=ORANGE, linestyle="--", linewidth=2, label=f"|MLE slope| = {mle_slope:.3f}")
    coefficient_axis.set(xlabel="Gaussian prior standard deviation σ (log scale)", ylabel="Estimated coefficient magnitude", title="Smaller prior variance produces stronger zero-centered shrinkage")
    nll_axis = coefficient_axis.twinx()
    nll_axis.plot(prior_stds, training_nll, color=GREEN, marker="s", linestyle=":", linewidth=2, label="training mean NLL")
    nll_axis.set_ylabel("Training mean NLL", color=GREEN)
    handles, labels = coefficient_axis.get_legend_handles_labels()
    extra_handles, extra_labels = nll_axis.get_legend_handles_labels()
    coefficient_axis.legend(handles + extra_handles, labels + extra_labels, loc="best")
    figure.text(
        0.5,
        0.01,
        "Shrinkage is computed—not claimed to improve validation or deployment performance.",
        ha="center",
        fontsize=9,
    )
    path = output_dir / "13_prior_variance.png"
    return VisualResult(
        "Prior variance and coefficient shrinkage",
        _finish_static(figure, path, save=save, show=show),
        tuple(
            f"prior_std={std:g}: |slope|={slope:.6f}, training NLL={nll:.6f}"
            for std, slope, nll in zip(prior_stds, slopes, training_nll, strict=True)
        ),
    )


def plot_posterior_uncertainty(
    output_dir: Path, *, save: bool, show: bool
) -> VisualResult:
    """Contrast equal posterior modes with substantially different uncertainty."""
    grid = np.linspace(0.001, 0.999, 1_400)
    broad_shapes = (4.0, 4.0)
    narrow_shapes = (40.0, 40.0)
    broad = beta_distribution.pdf(grid, *broad_shapes)
    narrow = beta_distribution.pdf(grid, *narrow_shapes)
    broad_mode = _beta_mode(*broad_shapes)
    narrow_mode = _beta_mode(*narrow_shapes)

    figure, axis = plt.subplots(figsize=(9.4, 5.7))
    axis.plot(grid, broad, color=BLUE, linewidth=2.8, label="Posterior A: Beta(4, 4) · broad")
    axis.plot(grid, narrow, color=PURPLE, linewidth=2.8, label="Posterior B: Beta(40, 40) · narrow")
    axis.fill_between(grid, broad, color=BLUE, alpha=0.10)
    axis.fill_between(grid, narrow, color=PURPLE, alpha=0.10)
    axis.axvline(0.5, color=RED, linestyle="--", linewidth=2, label="MAP A = MAP B = 0.5")
    axis.set(xlim=(0.0, 1.0), xlabel="Parameter p", ylabel="Posterior density", title="Same MAP point estimate ≠ same uncertainty")
    axis.legend()
    axis.text(
        0.5,
        0.06,
        "MAP collapses each posterior to one mode; full Bayesian inference retains distribution shape.",
        transform=axis.transAxes,
        ha="center",
        bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.94),
    )
    path = output_dir / "14_posterior_uncertainty.png"
    return VisualResult(
        "MAP versus posterior uncertainty",
        _finish_static(figure, path, save=save, show=show),
        (f"Posterior A MAP: {broad_mode:.6f}", f"Posterior B MAP: {narrow_mode:.6f}"),
    )


def _box(
    axis: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    width: float = 0.20,
) -> None:
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.55", facecolor=color, edgecolor="white"),
    )


def _arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = SLATE) -> None:
    axis.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.0))


def plot_concept_map(
    output_dir: Path, *, save: bool, show: bool
) -> VisualResult:
    """Summarize the path from probabilistic assumptions to model training."""
    figure, axis = plt.subplots(figsize=(12.0, 7.2))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    stages = (
        (0.50, 0.90, "Observed data", SLATE),
        (0.50, 0.77, "Likelihood", BLUE),
        (0.50, 0.64, "Log-likelihood", BLUE),
        (0.50, 0.51, "Negative log-likelihood", ORANGE),
        (0.50, 0.38, "MLE and MAP objective", ORANGE),
        (0.50, 0.25, "Gradient optimization", PURPLE),
        (0.50, 0.12, "Model parameters", GREEN),
    )
    for x, y, label, color in stages:
        _box(axis, x, y, label, color=color)
    for first, second in zip(stages, stages[1:]):
        _arrow(axis, (first[0], first[1] - 0.045), (second[0], second[1] + 0.045))
    _box(axis, 0.16, 0.62, "Prior p(θ)", color=PURPLE)
    _box(axis, 0.16, 0.46, "Negative log-prior", color=PURPLE)
    _arrow(axis, (0.16, 0.58), (0.16, 0.50), color=PURPLE)
    _arrow(axis, (0.26, 0.46), (0.39, 0.39), color=PURPLE)
    axis.text(0.31, 0.45, "+", fontsize=20, fontweight="bold", color=PURPLE)
    axis.text(
        0.78,
        0.69,
        "Observation model → loss\n\nGaussian → MSE\nBernoulli → BCE\nCategorical → cross-entropy",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.7", facecolor=LIGHT_BLUE, edgecolor=BLUE),
    )
    axis.text(
        0.78,
        0.36,
        "Parameter prior → penalty\n\nGaussian → L2\nLaplace → L1\n\nScaling must match the objective",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.7", facecolor=LIGHT_PURPLE, edgecolor=PURPLE),
    )
    axis.set_title("MLE and MAP connect probabilistic modeling to modern training", fontsize=17, pad=14)
    axis.text(0.5, 0.02, "MAP adds a prior preference but remains a point estimate—not the full posterior.", ha="center", color=RED, fontweight="bold")
    path = output_dir / "15_mle_map_concept_map.png"
    return VisualResult(
        "MLE/MAP concept map",
        _finish_static(figure, path, save=save, show=show),
    )


EXPECTED_ASSET_FILENAMES = (
    "01_probability_vs_likelihood.png",
    "likelihood_accumulation.gif",
    "03_log_likelihood.png",
    "prior_likelihood_posterior.png",
    "05_prior_strength.png",
    "06_sample_size_effect.png",
    "07_wrong_prior.png",
    "08_gaussian_nll_vs_sse.png",
    "09_bce_curve.png",
    "logistic_surface.png",
    "mle_optimization.gif",
    "mle_vs_map_surface.png",
    "13_prior_variance.png",
    "14_posterior_uncertainty.png",
    "15_mle_map_concept_map.png",
)


DEMO_NAMES = (
    "bernoulli",
    "accumulation",
    "log-space",
    "prior",
    "sample-size",
    "wrong-prior",
    "gaussian",
    "bce",
    "surface",
    "optimization",
    "map",
    "prior-variance",
    "uncertainty",
    "concept-map",
)


def generate_demo(
    demo: str,
    output_dir: Path,
    *,
    seed: int = 14,
    save: bool = False,
    show: bool = False,
    quick: bool = False,
) -> list[VisualResult]:
    """Generate one named demonstration or the complete visual narrative."""
    if demo != "all" and demo not in DEMO_NAMES:
        raise ValueError(f"unknown demo: {demo}")
    configure_plot_style()
    if demo == "all":
        results: list[VisualResult] = []
        for name in DEMO_NAMES:
            results.extend(
                generate_demo(
                    name,
                    output_dir,
                    seed=seed,
                    save=save,
                    show=show,
                    quick=quick,
                )
            )
        return results

    generators: dict[str, Callable[[], list[VisualResult]]] = {
        "bernoulli": lambda: [plot_probability_vs_likelihood(output_dir, save=save, show=show)],
        "accumulation": lambda: [animate_likelihood_accumulation(output_dir, seed=seed, save=save, show=show, quick=quick)],
        "log-space": lambda: [plot_likelihood_log_likelihood(output_dir, save=save, show=show)],
        "prior": lambda: [
            plot_prior_likelihood_posterior(output_dir, save=save, show=show),
            plot_prior_strength(output_dir, save=save, show=show),
        ],
        "sample-size": lambda: [plot_sample_size_effect(output_dir, seed=seed, save=save, show=show)],
        "wrong-prior": lambda: [plot_wrong_prior(output_dir, seed=seed, save=save, show=show)],
        "gaussian": lambda: [plot_gaussian_nll_vs_sse(output_dir, seed=seed, save=save, show=show)],
        "bce": lambda: [plot_bce(output_dir, save=save, show=show)],
        "surface": lambda: [plot_logistic_surface(output_dir, seed=seed, save=save, show=show, quick=quick)],
        "optimization": lambda: [animate_gradient_descent(output_dir, seed=seed, save=save, show=show, quick=quick)],
        "map": lambda: [plot_mle_vs_map_surface(output_dir, seed=seed, save=save, show=show, quick=quick)],
        "prior-variance": lambda: [plot_prior_variance(output_dir, seed=seed, save=save, show=show)],
        "uncertainty": lambda: [plot_posterior_uncertainty(output_dir, save=save, show=show)],
        "concept-map": lambda: [plot_concept_map(output_dir, save=save, show=show)],
    }
    return generators[demo]()


def parse_arguments() -> argparse.Namespace:
    """Parse demonstration, output, display, and smoke-render options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=(*DEMO_NAMES, "all"), default="bernoulli")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--save", action="store_true", help="Write PNG/GIF assets.")
    parser.add_argument("--show", action="store_true", help="Display figures locally.")
    parser.add_argument("--quick", action="store_true", help="Use reduced animation frames and surface grids.")
    return parser.parse_args()


def main() -> None:
    """Run the selected visual explanation and print computed observations."""
    args = parse_arguments()
    show = args.show or not args.save
    print("Day 14 — MLE and MAP Visual Learning Lab")
    print("Synthetic or analytically specified data only.")
    print("Interpretations remain candidates for author review.\n")
    results = generate_demo(
        args.demo,
        args.output_dir,
        seed=args.seed,
        save=args.save,
        show=show,
        quick=args.quick,
    )
    for result in results:
        location = result.path.name if result.path else "displayed without saving"
        print(f"[OK] {result.label}: {location}")
        for metric in result.metrics:
            print(f"     {metric}")
    if args.save:
        print(f"\nOutput directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
