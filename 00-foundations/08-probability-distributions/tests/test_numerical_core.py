"""Regression tests for the Day 8 numerical examples."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))

from distribution_utils import (  # noqa: E402
    calculate_empirical_statistics,
    simulate_poisson_process,
    total_absolute_probability_difference,
    validate_positive,
    validate_probability,
)
from from_scratch import (  # noqa: E402
    fit_bernoulli,
    fit_binomial,
    fit_exponential,
    fit_lognormal,
    fit_normal,
    fit_poisson,
)


class DistributionUtilityTests(unittest.TestCase):
    """Validate reusable calculations and their failure modes."""

    def test_empirical_statistics_use_population_moments(self) -> None:
        summary = calculate_empirical_statistics(np.array([1.0, 2.0, 3.0, 4.0]))

        self.assertAlmostEqual(summary.mean, 2.5)
        self.assertAlmostEqual(summary.variance, 1.25)
        self.assertAlmostEqual(summary.std, math.sqrt(1.25))
        self.assertAlmostEqual(summary.median, 2.5)
        self.assertAlmostEqual(summary.p90, 3.7)

    def test_parameter_validators_reject_invalid_values(self) -> None:
        validate_positive(0.1, "rate")
        validate_probability(0.0)
        validate_probability(1.0)

        for value in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(positive=value):
                with self.assertRaises(ValueError):
                    validate_positive(value, "rate")

        for value in (-0.01, 1.01, math.inf, math.nan):
            with self.subTest(probability=value):
                with self.assertRaises(ValueError):
                    validate_probability(value)

    def test_probability_distance_and_shape_validation(self) -> None:
        distance = total_absolute_probability_difference(
            np.array([0.2, 0.8]),
            np.array([0.3, 0.7]),
        )

        self.assertAlmostEqual(distance, 0.2)
        with self.assertRaises(ValueError):
            total_absolute_probability_difference(
                np.array([1.0]),
                np.array([0.5, 0.5]),
            )

    def test_poisson_process_is_bounded_and_reproducible(self) -> None:
        first_arrivals, first_waits = simulate_poisson_process(
            rate=3.0,
            duration=5.0,
            rng=np.random.default_rng(42),
        )
        second_arrivals, second_waits = simulate_poisson_process(
            rate=3.0,
            duration=5.0,
            rng=np.random.default_rng(42),
        )

        np.testing.assert_allclose(first_arrivals, second_arrivals)
        np.testing.assert_allclose(first_waits, second_waits)
        self.assertTrue(np.all(np.diff(first_arrivals) > 0.0))
        self.assertTrue(np.all(first_arrivals <= 5.0))
        np.testing.assert_allclose(first_arrivals, np.cumsum(first_waits))


class FromScratchEstimatorTests(unittest.TestCase):
    """Check the explicit maximum-likelihood formulas."""

    def test_closed_form_estimates(self) -> None:
        self.assertAlmostEqual(fit_bernoulli([0, 1, 1, 0]).parameters["p"], 0.5)
        self.assertAlmostEqual(
            fit_binomial([2, 4, 3], number_of_trials=5).parameters["p"],
            0.6,
        )
        self.assertAlmostEqual(fit_poisson([1, 2, 3]).parameters["lambda"], 2.0)
        self.assertAlmostEqual(
            fit_exponential([1.0, 2.0, 3.0]).parameters["lambda"],
            0.5,
        )

        normal = fit_normal([1.0, 2.0, 3.0]).parameters
        self.assertAlmostEqual(normal["mu"], 2.0)
        self.assertAlmostEqual(normal["variance"], 2.0 / 3.0)

        lognormal = fit_lognormal([1.0, math.e, math.e**2]).parameters
        self.assertAlmostEqual(lognormal["log_mu"], 1.0)
        self.assertAlmostEqual(lognormal["log_variance"], 2.0 / 3.0)

    def test_estimators_reject_values_outside_their_support(self) -> None:
        invalid_calls = (
            lambda: fit_bernoulli([0, 0.5, 1]),
            lambda: fit_binomial([6], number_of_trials=5),
            lambda: fit_poisson([-1]),
            lambda: fit_exponential([0, 0]),
            lambda: fit_normal([]),
            lambda: fit_lognormal([0, 1]),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()


if __name__ == "__main__":
    unittest.main()
