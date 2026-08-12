"""Tests for the Day 10 numerical core and sampling experiment."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))

from example import (  # noqa: E402
    generate_synthetic_population,
    run_sampling_experiment,
    selection_biased_mean,
    simple_random_mean,
    stratified_mean,
)
from from_scratch import (  # noqa: E402
    effective_sample_size,
    estimator_statistics,
    sample_mean_variance,
    weighted_mean,
)


class EstimatorFormulaTests(unittest.TestCase):
    """Verify explicit formulas and their validation boundaries."""

    def test_empirical_mse_decomposition_is_exact(self) -> None:
        statistics = estimator_statistics([8.0, 10.0, 12.0], 11.0)

        self.assertAlmostEqual(statistics["expected_estimate"], 10.0)
        self.assertAlmostEqual(statistics["bias"], -1.0)
        self.assertAlmostEqual(statistics["variance"], 8.0 / 3.0)
        self.assertAlmostEqual(statistics["mse"], 11.0 / 3.0)
        self.assertAlmostEqual(
            statistics["mse"],
            statistics["variance_plus_bias_squared"],
        )

    def test_weighted_mean_and_effective_sample_size(self) -> None:
        self.assertAlmostEqual(weighted_mean([10.0, 20.0], [0.75, 0.25]), 12.5)
        self.assertAlmostEqual(effective_sample_size([1.0, 1.0, 1.0]), 3.0)
        self.assertLess(effective_sample_size([1.0, 1.0, 10.0]), 3.0)

    def test_sample_mean_variance_and_finite_population_correction(self) -> None:
        self.assertAlmostEqual(sample_mean_variance(100.0, 25), 4.0)
        expected = 4.0 * (100 - 25) / (100 - 1)
        self.assertAlmostEqual(
            sample_mean_variance(100.0, 25, population_size=100),
            expected,
        )
        self.assertEqual(
            sample_mean_variance(100.0, 100, population_size=100),
            0.0,
        )

    def test_invalid_formula_inputs_raise_explicit_errors(self) -> None:
        invalid_calls = (
            lambda: estimator_statistics([], 1.0),
            lambda: estimator_statistics([1.0], math.inf),
            lambda: weighted_mean([1.0], [1.0, 2.0]),
            lambda: weighted_mean([1.0], [-1.0]),
            lambda: effective_sample_size([0.0, 0.0]),
            lambda: sample_mean_variance(-1.0, 2),
            lambda: sample_mean_variance(1.0, 0),
            lambda: sample_mean_variance(1.0, 11, population_size=10),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


class SamplingExperimentTests(unittest.TestCase):
    """Check reproducibility and design invariants, not benchmark claims."""

    def test_population_and_experiment_are_reproducible(self) -> None:
        first = generate_synthetic_population(population_size=2_000, seed=7)
        second = generate_synthetic_population(population_size=2_000, seed=7)
        np.testing.assert_array_equal(first.segments, second.segments)
        np.testing.assert_array_equal(first.spend, second.spend)

        first_result = run_sampling_experiment(
            first,
            sample_size=100,
            trials=30,
            seed=11,
        )
        second_result = run_sampling_experiment(
            second,
            sample_size=100,
            trials=30,
            seed=11,
        )
        self.assertEqual(first_result, second_result)

    def test_controlled_design_exposes_bias_and_stratification(self) -> None:
        population = generate_synthetic_population()
        summaries = run_sampling_experiment(population)

        random = summaries["simple_random"]
        biased = summaries["selection_biased"]
        stratified = summaries["stratified_weighted"]
        self.assertLess(abs(random["bias"]), 2.0)
        self.assertGreater(biased["bias"], 40.0)
        self.assertLess(abs(stratified["bias"]), 2.0)
        self.assertLess(stratified["variance"], random["variance"])
        for statistics in summaries.values():
            self.assertAlmostEqual(
                statistics["mse"],
                statistics["variance_plus_bias_squared"],
            )

    def test_sampling_functions_reject_invalid_designs(self) -> None:
        population = generate_synthetic_population(population_size=100)
        rng = np.random.default_rng(1)
        invalid_calls = (
            lambda: simple_random_mean(population, 101, rng),
            lambda: selection_biased_mean(
                population, 10, rng, premium_selection_odds=0.0
            ),
            lambda: stratified_mean(population, 100, 1, rng),
            lambda: run_sampling_experiment(population, sample_size=11),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


if __name__ == "__main__":
    unittest.main()
