"""Tests for the Day 9 numerical core and synthetic analysis."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))

from example import (  # noqa: E402
    INJECTED_LATENCY_SPIKES_MS,
    analyze_workload,
    generate_synthetic_workload,
)
from from_scratch import (  # noqa: E402
    covariance,
    interquartile_range,
    iqr_bounds,
    mean,
    median,
    median_absolute_deviation,
    pearson_correlation,
    population_excess_kurtosis,
    population_skewness,
    quantile,
    spearman_correlation,
    standard_deviation,
    variance,
)


class DescriptiveStatisticsTests(unittest.TestCase):
    """Compare the explicit formulas with known results and library behavior."""

    def test_center_and_spread(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 20.0]

        self.assertAlmostEqual(mean(values), 6.0)
        self.assertAlmostEqual(median(values), 3.0)
        self.assertAlmostEqual(variance(values), 50.0)
        self.assertAlmostEqual(variance(values, ddof=1), 62.5)
        self.assertAlmostEqual(standard_deviation(values), math.sqrt(50.0))
        self.assertAlmostEqual(interquartile_range(values), 2.0)
        self.assertAlmostEqual(median_absolute_deviation(values), 1.0)
        self.assertEqual(iqr_bounds(values), (-1.0, 7.0))

    def test_quantiles_match_numpy_default_linear_method(self) -> None:
        values = [10.0, 2.0, 8.0, 4.0, 6.0, 12.0]

        for probability in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
            with self.subTest(probability=probability):
                self.assertAlmostEqual(
                    quantile(values, probability),
                    float(np.quantile(values, probability)),
                )

    def test_covariance_and_correlations_match_pandas(self) -> None:
        first = [1.0, 2.0, 2.0, 4.0, 5.0]
        second = [5.0, 6.0, 6.0, 10.0, 20.0]
        frame = pd.DataFrame({"first": first, "second": second})

        self.assertAlmostEqual(covariance(first, second), frame.cov().iloc[0, 1])
        self.assertAlmostEqual(
            pearson_correlation(first, second),
            frame.corr(method="pearson").iloc[0, 1],
        )
        self.assertAlmostEqual(
            spearman_correlation(first, second),
            frame.corr(method="spearman").iloc[0, 1],
        )

    def test_population_moments_match_numpy_formulas(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 20.0])
        centered = values - values.mean()
        expected_skewness = np.mean(centered**3) / np.mean(centered**2) ** 1.5
        expected_kurtosis = np.mean(centered**4) / np.mean(centered**2) ** 2 - 3

        self.assertAlmostEqual(population_skewness(values), expected_skewness)
        self.assertAlmostEqual(
            population_excess_kurtosis(values),
            expected_kurtosis,
        )

    def test_invalid_inputs_raise_explicit_errors(self) -> None:
        invalid_calls = (
            lambda: mean([]),
            lambda: mean([1.0, math.nan]),
            lambda: quantile([1.0], 1.1),
            lambda: variance([1.0], ddof=1),
            lambda: covariance([1.0, 2.0], [1.0]),
            lambda: pearson_correlation([1.0, 1.0], [2.0, 3.0]),
            lambda: population_skewness([2.0, 2.0]),
            lambda: population_excess_kurtosis([2.0, 2.0]),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()


class SyntheticWorkloadTests(unittest.TestCase):
    """Verify reproducibility and invariants without asserting benchmark claims."""

    def test_generator_is_reproducible_and_validates_configuration(self) -> None:
        first = generate_synthetic_workload(sample_size=20, seed=7)
        second = generate_synthetic_workload(sample_size=20, seed=7)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.shape, (20, 4))
        self.assertTrue((first["latency_ms"] > 0.0).all())
        with self.assertRaises(ValueError):
            generate_synthetic_workload(
                sample_size=len(INJECTED_LATENCY_SPIKES_MS) - 1
            )
        without_spikes = generate_synthetic_workload(
            sample_size=2,
            inject_latency_spikes=False,
        )
        self.assertEqual(len(without_spikes), 2)

    def test_analysis_reports_expected_structures_and_relationships(self) -> None:
        dataset = generate_synthetic_workload()
        result = analyze_workload(dataset)

        self.assertEqual(result.summary.shape, (4, 10))
        self.assertGreaterEqual(
            result.latency_outlier_count,
            len(INJECTED_LATENCY_SPIKES_MS),
        )
        self.assertAlmostEqual(result.quadratic_pearson, 0.0, places=12)
        spearman = result.correlations.loc[
            ("spearman", "input_tokens"), "estimated_cost_usd"
        ]
        self.assertAlmostEqual(spearman, 1.0)
        pd.testing.assert_frame_equal(result.dataset, dataset)

    def test_analysis_rejects_missing_or_non_finite_data(self) -> None:
        incomplete = pd.DataFrame({"latency_ms": [100.0]})
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            analyze_workload(incomplete)

        non_finite = generate_synthetic_workload(sample_size=10)
        non_finite.loc[0, "latency_ms"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            analyze_workload(non_finite)

        non_numeric = generate_synthetic_workload(sample_size=10)
        non_numeric["latency_ms"] = non_numeric["latency_ms"].astype(object)
        non_numeric.loc[0, "latency_ms"] = "slow"
        with self.assertRaisesRegex(ValueError, "numeric"):
            analyze_workload(non_numeric)


if __name__ == "__main__":
    unittest.main()
