"""Numerical tests for the Day 12 visual laboratory."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))

from example import generate_synthetic_scores  # noqa: E402
from gif_exports import GENERATORS  # noqa: E402
from statistical_utils import (  # noqa: E402
    benjamini_hochberg,
    confidence_test_connection,
    normal_error_rates,
    normal_mean_test,
    one_sample_t_power,
    paired_visual_summary,
    power_surface,
    practical_significance_scenarios,
    sign_flip_distribution,
    simulate_multiple_testing,
    simulate_t_experiments,
)


class TestAndPowerTests(unittest.TestCase):
    """Verify analytical calculations and their qualitative relationships."""

    def test_normal_mean_test_matches_normal_tail_probability(self) -> None:
        result = normal_mean_test(0.12, 64, 0.40, alpha=0.05)

        self.assertAlmostEqual(result.standard_error, 0.05)
        self.assertAlmostEqual(result.statistic, 2.4)
        self.assertAlmostEqual(result.p_value, 2.0 * stats.norm.sf(2.4))
        self.assertEqual(result.reject, result.p_value < 0.05)

    def test_error_rates_partition_the_alternative_distribution(self) -> None:
        result = normal_error_rates(0.20, 50, 1.0, alpha=0.05)

        self.assertAlmostEqual(result.beta + result.power, 1.0)
        self.assertGreater(result.power, result.alpha)
        self.assertTrue(0.0 <= result.beta <= 1.0)

    def test_noncentral_t_power_increases_with_effect_and_sample_size(self) -> None:
        small_n = one_sample_t_power(20, 0.3)
        large_n = one_sample_t_power(100, 0.3)
        large_effect = one_sample_t_power(20, 0.8)

        self.assertGreater(large_n, small_n)
        self.assertGreater(large_effect, small_n)
        self.assertAlmostEqual(one_sample_t_power(80, 0.0), 0.05)

    def test_power_surface_is_bounded_and_oriented(self) -> None:
        surface = power_surface([10, 50, 200], [0.0, 0.3, 0.8])

        self.assertEqual(surface.shape, (3, 3))
        self.assertTrue(np.all((surface >= 0.0) & (surface <= 1.0)))
        self.assertGreater(surface[-1, -1], surface[-1, 0])
        self.assertGreater(surface[-1, -1], surface[1, -1])


class SimulationTests(unittest.TestCase):
    """Check reproducibility and simulation behavior with explicit tolerances."""

    def test_true_null_p_values_are_approximately_uniform(self) -> None:
        data = simulate_t_experiments(
            true_mean=0.0,
            sample_size=30,
            standard_deviation=1.0,
            simulations=8_000,
            seed=7,
        )

        self.assertAlmostEqual(float(data["p_value"].mean()), 0.5, delta=0.02)
        self.assertAlmostEqual(float(data["reject"].mean()), 0.05, delta=0.01)

    def test_simulated_power_is_close_to_noncentral_t_power(self) -> None:
        sample_size = 40
        effect = 0.35
        data = simulate_t_experiments(
            true_mean=effect,
            sample_size=sample_size,
            standard_deviation=1.0,
            simulations=8_000,
            seed=8,
        )
        theoretical = one_sample_t_power(sample_size, effect)

        self.assertAlmostEqual(
            float(data["reject"].mean()), theoretical, delta=0.025
        )

    def test_seeded_simulation_is_reproducible(self) -> None:
        kwargs = dict(
            true_mean=0.1,
            sample_size=12,
            standard_deviation=0.5,
            simulations=50,
            seed=9,
        )
        first = simulate_t_experiments(**kwargs)
        second = simulate_t_experiments(**kwargs)

        self.assertTrue(first.equals(second))


class AppliedAnalysisTests(unittest.TestCase):
    """Verify paired, randomization, multiplicity, and interval calculations."""

    def test_paired_summary_matches_scipy_and_preserves_difference(self) -> None:
        baseline, candidate = generate_synthetic_scores(seed=10)
        summary = paired_visual_summary(baseline, candidate)
        scipy_result = stats.ttest_rel(candidate, baseline)

        self.assertAlmostEqual(
            summary.mean_difference, float(np.mean(candidate - baseline))
        )
        self.assertAlmostEqual(summary.paired_t_statistic, scipy_result.statistic)
        self.assertAlmostEqual(summary.paired_p_value, scipy_result.pvalue)
        self.assertLess(summary.confidence_interval[0], summary.mean_difference)
        self.assertGreater(summary.confidence_interval[1], summary.mean_difference)

    def test_sign_flip_distribution_is_reproducible_and_corrected(self) -> None:
        differences = np.linspace(-0.02, 0.06, 30)
        first = sign_flip_distribution(differences, permutations=4_000, seed=11)
        second = sign_flip_distribution(differences, permutations=4_000, seed=11)

        np.testing.assert_array_equal(first.null_statistics, second.null_statistics)
        self.assertEqual(first.p_value, second.p_value)
        self.assertGreater(first.p_value, 0.0)
        self.assertAlmostEqual(first.p_value, (first.extreme_count + 1) / 4_001)

    def test_benjamini_hochberg_uses_largest_passing_rank(self) -> None:
        rejected = benjamini_hochberg([0.001, 0.010, 0.030, 0.200], alpha=0.05)

        np.testing.assert_array_equal(rejected, [True, True, True, False])

    def test_all_null_multiplicity_matches_expected_patterns(self) -> None:
        result = simulate_multiple_testing(
            hypotheses=20, repetitions=6_000, alpha=0.05, seed=12
        )

        expected_uncorrected = 1.0 - 0.95**20
        self.assertAlmostEqual(
            result.familywise_rates["No correction"],
            expected_uncorrected,
            delta=0.025,
        )
        self.assertAlmostEqual(
            result.familywise_rates["Bonferroni"], 0.05, delta=0.015
        )
        self.assertAlmostEqual(
            result.familywise_rates["Benjamini-Hochberg"], 0.05, delta=0.015
        )
        self.assertAlmostEqual(
            result.mean_discoveries["No correction"], 1.0, delta=0.05
        )

    def test_matching_interval_and_test_have_equivalent_decisions(self) -> None:
        for estimate in (0.0, 0.10, 0.25):
            result = confidence_test_connection(estimate, 0.05, 0.95)
            lower, upper = result.confidence_interval
            self.assertEqual(result.reject, not (lower <= 0.0 <= upper))

    def test_practical_scenarios_cover_four_distinct_cases(self) -> None:
        scenarios = practical_significance_scenarios(0.02)

        self.assertEqual(len(scenarios), 4)
        self.assertEqual(
            scenarios["statistically_significant"].tolist(),
            [True, True, False, False],
        )
        self.assertEqual(
            scenarios["point_estimate_exceeds_threshold"].tolist(),
            [True, False, True, False],
        )

    def test_invalid_inputs_raise_explicit_errors(self) -> None:
        invalid_calls = (
            lambda: normal_mean_test(0.1, 1, 1.0),
            lambda: normal_mean_test(0.1, 10, 0.0),
            lambda: normal_error_rates(-0.1, 10, 1.0),
            lambda: one_sample_t_power(10, math.inf),
            lambda: simulate_t_experiments(
                true_mean=0.0,
                sample_size=501,
                standard_deviation=1.0,
                simulations=10,
            ),
            lambda: benjamini_hochberg([0.1, 1.1]),
            lambda: confidence_test_connection(0.1, 0.0),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


class VisualManifestTests(unittest.TestCase):
    """Keep the bounded GIF inventory intentional."""

    def test_three_unique_gif_generators_are_registered(self) -> None:
        names = [generator.__name__ for generator in GENERATORS]

        self.assertEqual(len(names), 3)
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
