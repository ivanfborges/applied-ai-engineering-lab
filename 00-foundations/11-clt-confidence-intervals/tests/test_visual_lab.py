"""Numerical and manifest tests for the Day 11 visual laboratory."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))

from visual_lab import (  # noqa: E402
    EXPECTED_OUTPUTS,
    GENERATORS,
    RECOMMENDED_PREVIEWS,
    SECTIONS,
)
from visualizations.applied_experiments import (  # noqa: E402
    calculate_dependence_results,
    calculate_model_comparison,
    calculate_practical_significance,
)
from visualizations.visual_utils import (  # noqa: E402
    exponential_sample_means,
    simulate_normal_t_intervals,
    standardized_sample_means,
)


class VisualNumericalTests(unittest.TestCase):
    """Check statistical mechanisms with simulation-appropriate tolerances."""

    def test_exponential_mean_spread_tracks_theoretical_standard_error(self) -> None:
        sample_size = 30
        scale = 2.0
        means = exponential_sample_means(
            sample_size,
            8_000,
            seed=77,
            scale=scale,
        )

        empirical = float(np.std(means, ddof=0))
        theoretical = scale / np.sqrt(sample_size)
        self.assertAlmostEqual(float(np.mean(means)), scale, delta=0.04)
        self.assertAlmostEqual(empirical, theoretical, delta=0.02)

    def test_skewed_source_converges_more_slowly(self) -> None:
        lognormal_small = standardized_sample_means(
            "Lognormal", 5, 5_000, seed=81
        )
        lognormal_large = standardized_sample_means(
            "Lognormal", 100, 5_000, seed=82
        )

        def skewness(values: np.ndarray) -> float:
            centered = values - np.mean(values)
            return float(np.mean((centered / np.std(values, ddof=0)) ** 3))

        self.assertGreater(skewness(lognormal_small), 1.0)
        self.assertLess(skewness(lognormal_large), skewness(lognormal_small))

    def test_normal_data_t_interval_coverage_is_near_nominal(self) -> None:
        _, _, _, covered = simulate_normal_t_intervals(
            sample_size=25,
            intervals=8_000,
            confidence=0.95,
            seed=91,
        )

        self.assertAlmostEqual(float(np.mean(covered)), 0.95, delta=0.015)

    def test_dependence_increases_variability_at_equal_row_count(self) -> None:
        result = calculate_dependence_results(trials=3_000)

        self.assertAlmostEqual(
            result.independent_empirical_se,
            result.naive_se,
            delta=0.003,
        )
        self.assertGreater(
            result.clustered_empirical_se,
            5.0 * result.naive_se,
        )
        self.assertAlmostEqual(
            result.clustered_empirical_se,
            result.cluster_aware_se,
            delta=0.012,
        )

    def test_model_comparison_interval_targets_direct_difference(self) -> None:
        result = calculate_model_comparison()

        self.assertAlmostEqual(result.difference, result.mean_b - result.mean_a)
        self.assertLess(result.difference_ci[0], result.difference)
        self.assertGreater(result.difference_ci[1], result.difference)

    def test_detectable_effect_remains_below_illustrative_threshold(self) -> None:
        result = calculate_practical_significance()

        lower, upper = result.confidence_interval
        self.assertGreater(lower, 0.0)
        self.assertLess(upper, result.practical_threshold)

    def test_invalid_simulation_inputs_raise_explicit_errors(self) -> None:
        invalid_calls = (
            lambda: exponential_sample_means(0, 10, seed=1),
            lambda: exponential_sample_means(10, 0, seed=1),
            lambda: exponential_sample_means(10, 10, seed=1, scale=0.0),
            lambda: standardized_sample_means("Unknown", 10, 10, seed=1),
            lambda: simulate_normal_t_intervals(
                sample_size=1,
                intervals=10,
                seed=1,
            ),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


class VisualManifestTests(unittest.TestCase):
    """Keep the CLI inventory and conceptual grouping internally consistent."""

    def test_manifest_has_expected_types_and_unique_names(self) -> None:
        self.assertEqual(len(EXPECTED_OUTPUTS), len(set(EXPECTED_OUTPUTS)))
        self.assertEqual(sum(path.endswith(".png") for path in EXPECTED_OUTPUTS), 10)
        self.assertEqual(sum(path.endswith(".gif") for path in EXPECTED_OUTPUTS), 2)
        self.assertEqual(sum(path.endswith(".html") for path in EXPECTED_OUTPUTS), 1)
        self.assertTrue(set(RECOMMENDED_PREVIEWS).issubset(EXPECTED_OUTPUTS))

    def test_sections_cover_every_generator_once_conceptually(self) -> None:
        section_generators = {
            generator for generators in SECTIONS.values() for generator in generators
        }

        self.assertEqual(section_generators, set(GENERATORS))


if __name__ == "__main__":
    unittest.main()
