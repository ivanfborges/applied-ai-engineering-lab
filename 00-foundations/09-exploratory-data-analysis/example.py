"""Statistically disciplined EDA for a synthetic AI inference workload."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from from_scratch import iqr_bounds


SEED = 42
SAMPLE_SIZE = 1_000
INJECTED_LATENCY_SPIKES_MS = np.array(
    [1_500.0, 1_800.0, 2_200.0, 2_600.0, 3_000.0]
)


@dataclass(frozen=True)
class AnalysisResult:
    """Hold the reusable tables and diagnostics produced by the analysis."""

    summary: pd.DataFrame
    correlations: pd.DataFrame
    latency_outlier_count: int
    latency_iqr_bounds: tuple[float, float]
    quadratic_pearson: float
    dataset: pd.DataFrame


def generate_synthetic_workload(
    sample_size: int = SAMPLE_SIZE,
    seed: int = SEED,
    *,
    inject_latency_spikes: bool = True,
) -> pd.DataFrame:
    """Generate synthetic request metrics with five explicit latency spikes."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise ValueError("sample_size must be an integer.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if not isinstance(inject_latency_spikes, bool):
        raise ValueError("inject_latency_spikes must be a boolean.")
    if inject_latency_spikes and sample_size < len(INJECTED_LATENCY_SPIKES_MS):
        raise ValueError(
            "sample_size must be at least the number of injected latency spikes."
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    rng = np.random.default_rng(seed)
    input_tokens = rng.lognormal(mean=7.0, sigma=0.6, size=sample_size)
    retrieved_chunks = rng.poisson(lam=4.0, size=sample_size) + 1
    latency_ms = (
        120.0
        + 0.08 * input_tokens
        + 11.0 * retrieved_chunks
        + rng.normal(loc=0.0, scale=35.0, size=sample_size)
    )
    latency_ms = np.maximum(latency_ms, 1.0)
    if inject_latency_spikes:
        latency_ms[: len(INJECTED_LATENCY_SPIKES_MS)] += (
            INJECTED_LATENCY_SPIKES_MS
        )
    estimated_cost_usd = 0.000002 * input_tokens**1.15

    return pd.DataFrame(
        {
            "input_tokens": input_tokens,
            "retrieved_chunks": retrieved_chunks,
            "latency_ms": latency_ms,
            "estimated_cost_usd": estimated_cost_usd,
        }
    )


def describe_numeric_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """Summarize center, spread, shape, and tail quantiles."""
    if dataset.empty:
        raise ValueError("dataset cannot be empty.")
    if dataset.select_dtypes(include="number").shape[1] != dataset.shape[1]:
        raise ValueError("dataset must contain only numeric columns.")
    numeric_values = dataset.to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("dataset must contain only finite values.")

    rows: dict[str, dict[str, float]] = {}
    for column in dataset.columns:
        series = dataset[column]
        q1, q3 = series.quantile([0.25, 0.75])
        rows[column] = {
            "mean": series.mean(),
            "median": series.median(),
            "sample_variance": series.var(ddof=1),
            "sample_std": series.std(ddof=1),
            "iqr": q3 - q1,
            "mad_unscaled": (series - series.median()).abs().median(),
            "skewness": series.skew(),
            "excess_kurtosis": series.kurt(),
            "p95": series.quantile(0.95),
            "p99": series.quantile(0.99),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def analyze_workload(dataset: pd.DataFrame) -> AnalysisResult:
    """Calculate complementary EDA diagnostics without modifying the input."""
    required_columns = {
        "input_tokens",
        "retrieved_chunks",
        "latency_ms",
        "estimated_cost_usd",
    }
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"dataset is missing required columns: {missing}")

    numeric_dataset = dataset.loc[:, sorted(required_columns)]
    try:
        numeric_values = numeric_dataset.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("required columns must contain numeric values.") from error
    if not np.isfinite(numeric_values).all():
        raise ValueError("dataset must contain only finite values.")

    bounds = iqr_bounds(dataset["latency_ms"])
    latency_outliers = ~dataset["latency_ms"].between(*bounds, inclusive="both")
    correlations = pd.concat(
        {
            "pearson": numeric_dataset.corr(method="pearson"),
            "spearman": numeric_dataset.corr(method="spearman"),
        },
        names=["method"],
    )

    quadratic_x = np.linspace(-5.0, 5.0, 501)
    quadratic_y = quadratic_x**2
    quadratic_pearson = float(np.corrcoef(quadratic_x, quadratic_y)[0, 1])

    return AnalysisResult(
        summary=describe_numeric_columns(numeric_dataset),
        correlations=correlations,
        latency_outlier_count=int(latency_outliers.sum()),
        latency_iqr_bounds=bounds,
        quadratic_pearson=quadratic_pearson,
        dataset=dataset.copy(),
    )


def main() -> None:
    """Run and report one deterministic synthetic EDA experiment."""
    baseline_result = analyze_workload(
        generate_synthetic_workload(inject_latency_spikes=False)
    )
    result = analyze_workload(generate_synthetic_workload())
    baseline_latency = baseline_result.summary.loc["latency_ms"]
    latency = result.summary.loc["latency_ms"]
    token_cost_pearson = result.correlations.loc[
        ("pearson", "input_tokens"), "estimated_cost_usd"
    ]
    token_cost_spearman = result.correlations.loc[
        ("spearman", "input_tokens"), "estimated_cost_usd"
    ]

    print("SYNTHETIC EDA EXPERIMENT")
    print(
        "Hypothesis: rare latency spikes will affect non-robust summaries and "
        "tail metrics more than the median."
    )
    print(
        "Configuration: "
        f"seed={SEED}, requests={SAMPLE_SIZE}, "
        f"injected_spikes={len(INJECTED_LATENCY_SPIKES_MS)}"
    )
    print("\nSummary statistics")
    print(result.summary.round(6).to_string())
    print("\nSelected results")
    print(f"Latency mean (ms): {latency['mean']:.3f}")
    print(f"Latency median (ms): {latency['median']:.3f}")
    print(f"Latency P99 (ms): {latency['p99']:.3f}")
    print(f"Latency skewness: {latency['skewness']:.3f}")
    print(f"Latency excess kurtosis: {latency['excess_kurtosis']:.3f}")
    print(f"Latency IQR-flagged observations: {result.latency_outlier_count}")
    print("\nEffect of injecting the five latency spikes")
    for statistic in ("mean", "median", "sample_std", "p99", "excess_kurtosis"):
        change = latency[statistic] - baseline_latency[statistic]
        print(
            f"{statistic}: {baseline_latency[statistic]:.3f} -> "
            f"{latency[statistic]:.3f} (change={change:+.3f})"
        )
    print(f"Token-cost Pearson correlation: {token_cost_pearson:.6f}")
    print(f"Token-cost Spearman correlation: {token_cost_spearman:.6f}")
    print(
        "Pearson correlation for x and x^2 on a symmetric grid: "
        f"{result.quadratic_pearson:.6f}"
    )
    print(
        "\nInterpretation: the gap among median, mean, "
        "and P99 is more operationally informative than the mean alone; "
        "correlation must be matched to the relationship being investigated."
    )
    print(
        "Limitation: this deterministic synthetic workload and its injected "
        "spikes do not estimate any real production distribution or benchmark."
    )


if __name__ == "__main__":
    main()
