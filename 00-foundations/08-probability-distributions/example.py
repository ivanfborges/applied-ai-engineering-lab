"""Visualize six probability distributions using deterministic synthetic data."""

from __future__ import annotations

import argparse
from math import comb, factorial
from pathlib import Path

import matplotlib
import numpy as np


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for command-line arguments."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Return P(K=k) for a Binomial(n, p) variable."""
    return comb(n, k) * p**k * (1.0 - p) ** (n - k)


def poisson_pmf(k: int, rate: float) -> float:
    """Return P(X=k) for a Poisson(rate) variable."""
    return float(np.exp(-rate) * rate**k / factorial(k))


def exponential_pdf(x: np.ndarray, rate: float) -> np.ndarray:
    """Return the Exponential(rate) density for non-negative x."""
    return rate * np.exp(-rate * x)


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Return the Normal(mean, std**2) density."""
    coefficient = 1.0 / (std * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * ((x - mean) / std) ** 2
    return coefficient * np.exp(exponent)


def lognormal_pdf(
    x: np.ndarray,
    log_mean: float,
    log_std: float,
) -> np.ndarray:
    """Return a Log-normal density parameterized on the log scale."""
    density = np.zeros_like(x, dtype=float)
    positive = x > 0.0
    values = x[positive]
    coefficient = 1.0 / (values * log_std * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * ((np.log(values) - log_mean) / log_std) ** 2
    density[positive] = coefficient * np.exp(exponent)
    return density


def print_summary(
    name: str,
    samples: np.ndarray,
    theoretical_mean: float,
    theoretical_variance: float,
) -> None:
    """Print empirical population moments beside theoretical moments."""
    print(f"\n{name}")
    print(f"  empirical mean:        {np.mean(samples):.4f}")
    print(f"  theoretical mean:      {theoretical_mean:.4f}")
    print(f"  empirical variance:    {np.var(samples, ddof=0):.4f}")
    print(f"  theoretical variance:  {theoretical_variance:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    default_output = (
        Path(__file__).resolve().parent
        / "outputs"
        / "static"
        / "probability_distributions.png"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Sample six distributions from synthetic data, compare moments, "
            "and save empirical/theoretical plots."
        )
    )
    parser.add_argument(
        "--sample-size",
        type=positive_int,
        default=50_000,
        help="Synthetic observations per distribution (default: 50000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path for the generated PNG.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate samples, print moment checks, and create the visualization."""
    args = parse_args()

    # Headless mode keeps the default command reliable in terminals and CI.
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)

    bernoulli_p = 0.70
    binomial_n, binomial_p = 20, 0.35
    poisson_rate = 4.0
    exponential_rate = 0.50
    normal_mean, normal_std = 10.0, 2.0
    lognormal_log_mean, lognormal_log_std = 2.0, 0.60

    # Every dataset below is synthetic and generated from the stated family.
    samples = {
        "Bernoulli": rng.binomial(1, bernoulli_p, args.sample_size),
        "Binomial": rng.binomial(
            binomial_n, binomial_p, args.sample_size
        ),
        "Poisson": rng.poisson(poisson_rate, args.sample_size),
        "Exponential": rng.exponential(
            1.0 / exponential_rate, args.sample_size
        ),
        "Normal": rng.normal(normal_mean, normal_std, args.sample_size),
        "Log-normal": rng.lognormal(
            lognormal_log_mean, lognormal_log_std, args.sample_size
        ),
    }

    lognormal_mean = float(
        np.exp(lognormal_log_mean + lognormal_log_std**2 / 2.0)
    )
    lognormal_variance = float(
        (np.exp(lognormal_log_std**2) - 1.0)
        * np.exp(2.0 * lognormal_log_mean + lognormal_log_std**2)
    )
    moments = {
        "Bernoulli": (
            bernoulli_p,
            bernoulli_p * (1.0 - bernoulli_p),
        ),
        "Binomial": (
            binomial_n * binomial_p,
            binomial_n * binomial_p * (1.0 - binomial_p),
        ),
        "Poisson": (poisson_rate, poisson_rate),
        "Exponential": (
            1.0 / exponential_rate,
            1.0 / exponential_rate**2,
        ),
        "Normal": (normal_mean, normal_std**2),
        "Log-normal": (lognormal_mean, lognormal_variance),
    }

    print(
        "Dataset: deterministic synthetic samples "
        f"(seed={args.seed}, n={args.sample_size:,} per distribution)"
    )
    for name, distribution_samples in samples.items():
        print_summary(name, distribution_samples, *moments[name])

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].hist(
        samples["Bernoulli"],
        bins=[-0.5, 0.5, 1.5],
        density=True,
        rwidth=0.75,
        alpha=0.70,
        label="Empirical",
    )
    axes[0, 0].scatter(
        [0, 1],
        [1.0 - bernoulli_p, bernoulli_p],
        color="tab:orange",
        s=70,
        zorder=3,
        label="Theoretical PMF",
    )
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set(title="Bernoulli", xlabel="Outcome", ylabel="Probability")

    binomial_support = np.arange(binomial_n + 1)
    binomial_probabilities = np.array(
        [
            binomial_pmf(int(k), binomial_n, binomial_p)
            for k in binomial_support
        ]
    )
    axes[0, 1].hist(
        samples["Binomial"],
        bins=np.arange(-0.5, binomial_n + 1.5),
        density=True,
        rwidth=0.75,
        alpha=0.70,
        label="Empirical",
    )
    axes[0, 1].plot(
        binomial_support,
        binomial_probabilities,
        "o",
        color="tab:orange",
        label="Theoretical PMF",
    )
    axes[0, 1].set(
        title="Binomial",
        xlabel="Successes in 20 trials",
        ylabel="Probability",
    )

    poisson_support = np.arange(16)
    poisson_probabilities = np.array(
        [poisson_pmf(int(k), poisson_rate) for k in poisson_support]
    )
    axes[0, 2].hist(
        samples["Poisson"],
        bins=np.arange(-0.5, 16.5),
        density=True,
        rwidth=0.75,
        alpha=0.70,
        label="Empirical",
    )
    axes[0, 2].plot(
        poisson_support,
        poisson_probabilities,
        "o",
        color="tab:orange",
        label="Theoretical PMF",
    )
    axes[0, 2].set(
        title="Poisson",
        xlabel="Events per interval",
        ylabel="Probability",
    )

    exponential_limit = float(np.quantile(samples["Exponential"], 0.995))
    exponential_x = np.linspace(0.0, exponential_limit, 400)
    axes[1, 0].hist(
        samples["Exponential"],
        bins=80,
        density=True,
        alpha=0.70,
        label="Empirical",
    )
    axes[1, 0].plot(
        exponential_x,
        exponential_pdf(exponential_x, exponential_rate),
        color="tab:orange",
        linewidth=2,
        label="Theoretical PDF",
    )
    axes[1, 0].set(
        title="Exponential",
        xlabel="Waiting time",
        ylabel="Density",
        xlim=(0.0, exponential_limit),
    )

    normal_x = np.linspace(
        normal_mean - 4.0 * normal_std,
        normal_mean + 4.0 * normal_std,
        400,
    )
    axes[1, 1].hist(
        samples["Normal"],
        bins=80,
        density=True,
        alpha=0.70,
        label="Empirical",
    )
    axes[1, 1].plot(
        normal_x,
        normal_pdf(normal_x, normal_mean, normal_std),
        color="tab:orange",
        linewidth=2,
        label="Theoretical PDF",
    )
    axes[1, 1].set(title="Normal", xlabel="Value", ylabel="Density")

    lognormal_limit = float(np.quantile(samples["Log-normal"], 0.995))
    lognormal_x = np.linspace(0.001, lognormal_limit, 400)
    axes[1, 2].hist(
        samples["Log-normal"],
        bins=100,
        density=True,
        alpha=0.70,
        label="Empirical",
    )
    axes[1, 2].plot(
        lognormal_x,
        lognormal_pdf(
            lognormal_x,
            lognormal_log_mean,
            lognormal_log_std,
        ),
        color="tab:orange",
        linewidth=2,
        label="Theoretical PDF",
    )
    axes[1, 2].set(
        title="Log-normal",
        xlabel="Positive value",
        ylabel="Density",
        xlim=(0.0, lognormal_limit),
    )

    for axis in axes.flat:
        axis.grid(alpha=0.20)
        axis.legend()

    figure.suptitle(
        "Synthetic Samples vs. Theoretical Probability Distributions",
        fontsize=15,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"\nSaved: {args.output.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
