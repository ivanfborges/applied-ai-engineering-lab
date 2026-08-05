"""Demonstrate Bayes' theorem and the base-rate effect with synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from visualizations import bayes_posterior, simulate_fraud_population


DEFAULT_FRAUD_RATE = 0.01
DEFAULT_TRUE_POSITIVE_RATE = 0.90
DEFAULT_FALSE_POSITIVE_RATE = 0.05


def confusion_table(data: pd.DataFrame) -> pd.DataFrame:
    """Return a complete 2x2 count table with readable labels."""
    table = pd.crosstab(data["is_fraud"], data["has_alert"])
    table = table.reindex(index=[False, True], columns=[False, True], fill_value=0)
    table.index = ["Legitimate", "Fraud"]
    table.columns = ["No alert", "Alert"]
    return table


def save_base_rate_plot(
    true_positive_rate: float,
    false_positive_rate: float,
    observed_prior: float,
    output_path: Path,
) -> Figure:
    """Plot how the prior changes P(Fraud | Alert) for a fixed detector."""
    import matplotlib.pyplot as plt

    base_rates = np.linspace(0.001, 0.20, 300)
    posteriors = np.array(
        [
            bayes_posterior(
                prior=rate,
                true_positive_rate=true_positive_rate,
                false_positive_rate=false_positive_rate,
            )
            for rate in base_rates
        ]
    )
    observed_posterior = bayes_posterior(
        prior=observed_prior,
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
    )

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(base_rates, posteriors, linewidth=2.5, color="tab:blue")
    axis.scatter(
        [observed_prior],
        [observed_posterior],
        color="tab:red",
        zorder=3,
        label=(
            f"Example: prior={observed_prior:.1%}, "
            f"posterior={observed_posterior:.1%}"
        ),
    )
    axis.set(
        title="Base-rate effect for a fixed synthetic fraud detector",
        xlabel="Prior fraud probability P(Fraud)",
        ylabel="Posterior probability P(Fraud | Alert)",
    )
    axis.set_xlim(float(base_rates.min()), float(base_rates.max()))
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    return figure


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_output = Path(__file__).resolve().parent / "outputs"
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a synthetic fraud detector and visualize the "
            "Bayesian base-rate effect."
        )
    )
    parser.add_argument(
        "--n-transactions",
        type=int,
        default=200_000,
        help="Number of synthetic transactions (default: 200000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory for the generated chart.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the chart after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The default mode works in CI and terminals without a graphical backend.
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = simulate_fraud_population(
        population_size=args.n_transactions,
        prior=DEFAULT_FRAUD_RATE,
        true_positive_rate=DEFAULT_TRUE_POSITIVE_RATE,
        false_positive_rate=DEFAULT_FALSE_POSITIVE_RATE,
        seed=args.seed,
    )

    theoretical_posterior = bayes_posterior(
        prior=DEFAULT_FRAUD_RATE,
        true_positive_rate=DEFAULT_TRUE_POSITIVE_RATE,
        false_positive_rate=DEFAULT_FALSE_POSITIVE_RATE,
    )
    alerted = data.loc[data["has_alert"], "is_fraud"]
    if alerted.empty:
        raise RuntimeError(
            "The simulation generated no alerts; increase the sample size."
        )

    empirical_posterior = float(alerted.mean())
    empirical_fraud_rate = float(data["is_fraud"].mean())
    empirical_fraud_variance = float(data["is_fraud"].var(ddof=0))
    theoretical_fraud_variance = (
        DEFAULT_FRAUD_RATE * (1.0 - DEFAULT_FRAUD_RATE)
    )

    output_path = args.output_dir / "base_rate_effect.png"
    figure = save_base_rate_plot(
        true_positive_rate=DEFAULT_TRUE_POSITIVE_RATE,
        false_positive_rate=DEFAULT_FALSE_POSITIVE_RATE,
        observed_prior=DEFAULT_FRAUD_RATE,
        output_path=output_path,
    )

    print("Dataset: synthetic binary transactions (fixed random seed)")
    print(f"Transactions: {len(data):,}")
    print(f"Configured fraud rate:       {DEFAULT_FRAUD_RATE:.2%}")
    print(f"True-positive rate:          {DEFAULT_TRUE_POSITIVE_RATE:.2%}")
    print(f"False-positive rate:         {DEFAULT_FALSE_POSITIVE_RATE:.2%}")
    print("\nConfusion table")
    print(confusion_table(data))
    print(
        f"\nTheoretical P(Fraud | Alert): {theoretical_posterior:.4f}"
    )
    print(f"Empirical P(Fraud | Alert):   {empirical_posterior:.4f}")
    print(f"\nTheoretical E[Fraud]:         {DEFAULT_FRAUD_RATE:.4f}")
    print(f"Empirical E[Fraud]:           {empirical_fraud_rate:.4f}")
    print(
        f"Theoretical Var(Fraud):       {theoretical_fraud_variance:.6f}"
    )
    print(f"Empirical Var(Fraud):         {empirical_fraud_variance:.6f}")
    print(f"\nSaved: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
