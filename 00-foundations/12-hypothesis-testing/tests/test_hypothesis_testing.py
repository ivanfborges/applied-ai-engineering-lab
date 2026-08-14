"""Tests for the Day 12 paired-test helpers and synthetic example."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))

from example import analyze_paired_scores, generate_synthetic_scores  # noqa: E402
from from_scratch import paired_sign_flip_test, paired_statistics  # noqa: E402


class PairedStatisticsTests(unittest.TestCase):
    """Verify first-principles formulas and validation boundaries."""

    def test_paired_statistics_match_scipy_t_statistic(self) -> None:
        differences = np.array([0.04, 0.03, -0.01, 0.05, 0.02])
        result = paired_statistics(differences)
        scipy_result = stats.ttest_1samp(differences, popmean=0.0)

        self.assertAlmostEqual(result.mean_difference, float(np.mean(differences)))
        self.assertAlmostEqual(result.t_statistic, float(scipy_result.statistic))
        self.assertAlmostEqual(
            result.cohens_dz,
            float(np.mean(differences) / np.std(differences, ddof=1)),
        )

    def test_exact_sign_flip_enumerates_every_assignment(self) -> None:
        result = paired_sign_flip_test([1.0, 2.0, 3.0])

        self.assertTrue(result.exact)
        self.assertEqual(result.samples_evaluated, 8)
        self.assertAlmostEqual(result.p_value, 0.25)

    def test_monte_carlo_sign_flip_is_reproducible(self) -> None:
        differences = np.linspace(-0.02, 0.04, 30)
        first = paired_sign_flip_test(
            differences, permutations=2_000, seed=7, exact_max_pairs=10
        )
        second = paired_sign_flip_test(
            differences, permutations=2_000, seed=7, exact_max_pairs=10
        )

        self.assertEqual(first, second)
        self.assertFalse(first.exact)
        self.assertGreater(first.p_value, 0.0)

    def test_invalid_inputs_raise_explicit_errors(self) -> None:
        invalid_calls = (
            lambda: paired_statistics([]),
            lambda: paired_statistics([1.0, math.inf]),
            lambda: paired_statistics([1.0, 1.0]),
            lambda: paired_sign_flip_test([1.0, 2.0], alternative="invalid"),
            lambda: paired_sign_flip_test([1.0, 2.0], permutations=0),
            lambda: paired_sign_flip_test([1.0, 2.0], exact_max_pairs=-1),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


class SyntheticEvaluationTests(unittest.TestCase):
    """Check deterministic experiment behavior, not real-system performance."""

    def test_synthetic_generation_is_reproducible_and_bounded(self) -> None:
        first = generate_synthetic_scores(seed=5)
        second = generate_synthetic_scores(seed=5)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue(np.all((first[0] >= 0.0) & (first[0] <= 1.0)))
        self.assertTrue(np.all((first[1] >= 0.0) & (first[1] <= 1.0)))

    def test_analysis_reports_consistent_two_sided_inference(self) -> None:
        baseline, candidate = generate_synthetic_scores()
        result = analyze_paired_scores(
            baseline, candidate, sign_flip_permutations=10_000
        )
        lower, upper = result.confidence_interval

        self.assertEqual(result.sample_size, 80)
        self.assertGreater(result.mean_difference, 0.0)
        self.assertLess(lower, result.mean_difference)
        self.assertGreater(upper, result.mean_difference)
        self.assertEqual(result.t_p_value < result.alpha, not (lower <= 0.0 <= upper))
        self.assertAlmostEqual(result.t_statistic, result.cohens_dz * math.sqrt(80))
        self.assertLess(abs(result.t_p_value - result.sign_flip_p_value), 0.02)

    def test_invalid_evaluation_inputs_raise_explicit_errors(self) -> None:
        invalid_calls = (
            lambda: generate_synthetic_scores(sample_size=1),
            lambda: generate_synthetic_scores(difference_sd=0.0),
            lambda: analyze_paired_scores([1.0], [1.0]),
            lambda: analyze_paired_scores([1.0, 2.0], [1.0]),
            lambda: analyze_paired_scores([1.0, math.nan], [1.0, 2.0]),
            lambda: analyze_paired_scores([1.0, 2.0], [1.1, 2.1], alpha=1.0),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


if __name__ == "__main__":
    unittest.main()
