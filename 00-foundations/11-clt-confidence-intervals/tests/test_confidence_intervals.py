"""Tests for the Day 11 numerical helpers and synthetic experiments."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))

from example import simulate_clt, simulate_t_interval_coverage  # noqa: E402
from from_scratch import (  # noqa: E402
    normal_mean_confidence_interval,
    sample_mean,
    sample_variance,
    standard_error_mean,
    wilson_score_interval,
)


class ConfidenceIntervalFormulaTests(unittest.TestCase):
    """Verify formulas and explicit validation boundaries."""

    def test_sample_statistics(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]

        self.assertAlmostEqual(sample_mean(values), 2.5)
        self.assertAlmostEqual(sample_variance(values), 5.0 / 3.0)
        self.assertAlmostEqual(standard_error_mean(values), math.sqrt(5.0 / 12.0))

    def test_normal_interval_is_symmetric_about_mean(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        lower, upper = normal_mean_confidence_interval(values)

        self.assertAlmostEqual((lower + upper) / 2.0, sample_mean(values))
        self.assertGreater(upper, lower)

    def test_wilson_interval_handles_boundary_counts(self) -> None:
        zero_lower, zero_upper = wilson_score_interval(0, 20)
        full_lower, full_upper = wilson_score_interval(20, 20)

        self.assertEqual(zero_lower, 0.0)
        self.assertGreater(zero_upper, 0.0)
        self.assertLess(full_lower, 1.0)
        self.assertEqual(full_upper, 1.0)

    def test_invalid_formula_inputs_raise_explicit_errors(self) -> None:
        invalid_calls = (
            lambda: sample_mean([]),
            lambda: sample_variance([1.0]),
            lambda: standard_error_mean([1.0, math.inf]),
            lambda: normal_mean_confidence_interval([1.0, 2.0], confidence=1.0),
            lambda: wilson_score_interval(-1, 10),
            lambda: wilson_score_interval(11, 10),
            lambda: wilson_score_interval(1, 0),
            lambda: wilson_score_interval(1.0, 10),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


class SyntheticExperimentTests(unittest.TestCase):
    """Check deterministic design behavior, not real-world performance."""

    def test_clt_experiment_is_reproducible(self) -> None:
        first = simulate_clt((5, 30), simulations=500, seed=7)
        second = simulate_clt((5, 30), simulations=500, seed=7)

        self.assertEqual(first, second)

    def test_configured_clt_experiment_matches_theory(self) -> None:
        summaries = simulate_clt()

        for summary in summaries:
            relative_error = abs(
                summary.empirical_standard_error
                - summary.theoretical_standard_error
            ) / summary.theoretical_standard_error
            self.assertLess(relative_error, 0.04)
            self.assertAlmostEqual(summary.mean_of_means, 2.0, delta=0.03)
        self.assertGreater(summaries[0].skewness, summaries[-1].skewness)
        self.assertGreater(
            summaries[0].empirical_standard_error,
            summaries[-1].empirical_standard_error,
        )

    def test_configured_t_interval_coverage_is_near_nominal(self) -> None:
        summary = simulate_t_interval_coverage()

        self.assertGreater(summary.empirical_coverage, 0.93)
        self.assertLess(summary.empirical_coverage, 0.97)
        self.assertGreater(summary.mean_interval_width, 0.0)

    def test_invalid_experiment_inputs_raise_explicit_errors(self) -> None:
        invalid_calls = (
            lambda: simulate_clt([]),
            lambda: simulate_clt([0]),
            lambda: simulate_clt([5], simulations=0),
            lambda: simulate_clt([5], population_mean=0.0),
            lambda: simulate_t_interval_coverage(sample_size=1),
            lambda: simulate_t_interval_coverage(confidence=0.0),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


if __name__ == "__main__":
    unittest.main()
