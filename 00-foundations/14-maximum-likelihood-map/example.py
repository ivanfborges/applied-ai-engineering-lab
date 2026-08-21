"""Bernoulli examples for maximum likelihood and MAP estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BernoulliEstimate:
    """Store one deterministic MLE/MAP comparison."""

    successes: int
    trials: int
    mle: float
    map_estimate: float


def _validate_counts(successes: int, trials: int) -> None:
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer.")
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer.")
    if trials <= 0:
        raise ValueError("trials must be positive.")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials.")


def _validate_beta_prior(alpha: float, beta: float) -> tuple[float, float]:
    alpha = float(alpha)
    beta = float(beta)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive.")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive.")
    return alpha, beta


def bernoulli_mle(successes: int, trials: int) -> float:
    """Return the Bernoulli maximum-likelihood estimate k / n."""
    _validate_counts(successes, trials)
    return successes / trials


def beta_bernoulli_map(
    successes: int,
    trials: int,
    *,
    alpha: float,
    beta: float,
) -> float:
    """Return a unique mode of the Beta-Bernoulli posterior.

    A U-shaped posterior has two boundary modes, while a uniform posterior has
    no unique mode. Those cases raise an explicit error instead of silently
    applying the interior-mode formula outside its valid domain.
    """
    _validate_counts(successes, trials)
    alpha, beta = _validate_beta_prior(alpha, beta)
    posterior_alpha = alpha + successes
    posterior_beta = beta + trials - successes

    if posterior_alpha > 1.0 and posterior_beta > 1.0:
        return (posterior_alpha - 1.0) / (
            posterior_alpha + posterior_beta - 2.0
        )
    if posterior_alpha <= 1.0 < posterior_beta:
        return 0.0
    if posterior_beta <= 1.0 < posterior_alpha:
        return 1.0
    if posterior_alpha < 1.0 and posterior_beta == 1.0:
        return 0.0
    if posterior_beta < 1.0 and posterior_alpha == 1.0:
        return 1.0
    raise ValueError("the posterior does not have a unique mode.")


def bernoulli_log_likelihood(
    probability: float | np.ndarray,
    successes: int,
    trials: int,
) -> float | np.ndarray:
    """Evaluate the Bernoulli log-likelihood without multiplying probabilities."""
    _validate_counts(successes, trials)
    probabilities = np.asarray(probability, dtype=float)
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("probability must contain finite values in [0, 1].")

    values = np.zeros_like(probabilities)
    with np.errstate(divide="ignore"):
        if successes:
            values += successes * np.log(probabilities)
        if successes < trials:
            values += (trials - successes) * np.log1p(-probabilities)
    return float(values) if values.ndim == 0 else values


def beta_bernoulli_log_posterior(
    probability: float | np.ndarray,
    successes: int,
    trials: int,
    *,
    alpha: float,
    beta: float,
) -> float | np.ndarray:
    """Evaluate the unnormalized Beta-Bernoulli log-posterior."""
    alpha, beta = _validate_beta_prior(alpha, beta)
    _validate_counts(successes, trials)
    probabilities = np.asarray(probability, dtype=float)
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("probability must contain finite values in [0, 1].")
    success_power = successes + alpha - 1.0
    failure_power = trials - successes + beta - 1.0
    values = np.zeros_like(probabilities)
    with np.errstate(divide="ignore", invalid="ignore"):
        if success_power != 0.0:
            values += success_power * np.log(probabilities)
        if failure_power != 0.0:
            values += failure_power * np.log1p(-probabilities)
    return float(values) if values.ndim == 0 else values


def grid_mode(grid: np.ndarray, log_values: np.ndarray) -> float:
    """Return the grid coordinate with the largest finite log value."""
    grid = np.asarray(grid, dtype=float)
    log_values = np.asarray(log_values, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("grid must be a one-dimensional array with at least two values.")
    if grid.shape != log_values.shape:
        raise ValueError("grid and log_values must have the same shape.")
    if np.any(~np.isfinite(grid)) or np.any(np.isnan(log_values)):
        raise ValueError("grid must be finite and log_values cannot contain NaN.")
    return float(grid[int(np.argmax(log_values))])


def compare_sample_sizes(
    *, alpha: float = 2.0, beta: float = 2.0
) -> tuple[BernoulliEstimate, ...]:
    """Compare a fixed 70% success rate at four deterministic sample sizes."""
    rows = []
    for trials in (10, 100, 1_000, 10_000):
        successes = 7 * trials // 10
        rows.append(
            BernoulliEstimate(
                successes=successes,
                trials=trials,
                mle=bernoulli_mle(successes, trials),
                map_estimate=beta_bernoulli_map(
                    successes, trials, alpha=alpha, beta=beta
                ),
            )
        )
    return tuple(rows)


def main() -> None:
    """Run deterministic analytical and grid-based comparisons."""
    successes, trials = 7, 10
    alpha, beta = 2.0, 2.0
    grid = np.linspace(0.0001, 0.9999, 100_000)

    mle = bernoulli_mle(successes, trials)
    map_estimate = beta_bernoulli_map(
        successes, trials, alpha=alpha, beta=beta
    )
    grid_mle = grid_mode(
        grid, bernoulli_log_likelihood(grid, successes, trials)
    )
    grid_map = grid_mode(
        grid,
        beta_bernoulli_log_posterior(
            grid, successes, trials, alpha=alpha, beta=beta
        ),
    )

    print("Beta-Bernoulli estimation: 7 successes in 10 trials, Beta(2, 2) prior")
    print(f"Analytical MLE: {mle:.6f} | grid mode: {grid_mle:.6f}")
    print(f"Analytical MAP: {map_estimate:.6f} | grid mode: {grid_map:.6f}")
    print()
    print("Fixed empirical success rate with the same prior")
    print("successes/trials       MLE       MAP    absolute gap")
    for row in compare_sample_sizes(alpha=alpha, beta=beta):
        print(
            f"{row.successes:>5}/{row.trials:<5}       "
            f"{row.mle:.6f}  {row.map_estimate:.6f}  "
            f"{abs(row.mle - row.map_estimate):.6f}"
        )


if __name__ == "__main__":
    main()
