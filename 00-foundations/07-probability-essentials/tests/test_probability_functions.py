"""Tests for reusable Day 7 probability functions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TOPIC_DIR = Path(__file__).resolve().parents[1]
if str(TOPIC_DIR) not in sys.path:
    sys.path.insert(0, str(TOPIC_DIR))

from visualizations import (  # noqa: E402
    bayes_posterior,
    bernoulli_variance,
    discrete_variance,
    expected_decision_threshold,
    expected_value,
    frechet_bounds,
    joint_probability_table,
    simulate_binary_joint_distribution,
    validate_discrete_distribution,
    validate_probability,
)


class ProbabilityFunctionTests(unittest.TestCase):
    """Validate probability calculations and their input constraints."""

    def test_bayes_posterior_known_values(self) -> None:
        posterior = bayes_posterior(0.01, 0.90, 0.05)
        self.assertAlmostEqual(posterior, 0.153846153846, places=10)

    def test_probability_validation_accepts_boundaries(self) -> None:
        validate_probability(0.0, "value")
        validate_probability(1.0, "value")

    def test_probability_validation_rejects_invalid_values(self) -> None:
        for value in (-0.01, 1.01, np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_probability(value, "value")

    def test_frechet_bounds(self) -> None:
        lower, upper = frechet_bounds(0.70, 0.60)
        self.assertAlmostEqual(lower, 0.30)
        self.assertAlmostEqual(upper, 0.60)

    def test_expected_value_of_fair_die(self) -> None:
        values = np.arange(1, 7)
        probabilities = np.full(6, 1 / 6)
        self.assertAlmostEqual(expected_value(values, probabilities), 3.5)

    def test_variance_of_fair_die(self) -> None:
        values = np.arange(1, 7)
        probabilities = np.full(6, 1 / 6)
        self.assertAlmostEqual(
            discrete_variance(values, probabilities),
            35 / 12,
        )

    def test_bernoulli_variance_is_maximal_at_half(self) -> None:
        self.assertAlmostEqual(bernoulli_variance(0.5), 0.25)

    def test_decision_threshold(self) -> None:
        self.assertAlmostEqual(expected_decision_threshold(5.0, 100.0), 0.05)

    def test_invalid_probability_distribution(self) -> None:
        invalid_cases = (
            ([1, 2], [0.4, 0.4]),
            ([1, 2], [1.1, -0.1]),
            ([1, 2, 3], [0.5, 0.5]),
            ([], []),
        )
        for values, probabilities in invalid_cases:
            with self.subTest(values=values, probabilities=probabilities):
                with self.assertRaises(ValueError):
                    validate_discrete_distribution(values, probabilities)

    def test_invalid_joint_probability(self) -> None:
        with self.assertRaises(ValueError):
            joint_probability_table(0.20, 0.30, 0.25)

    def test_simulation_matches_theoretical_joint_probabilities(self) -> None:
        probability_a = 0.40
        probability_b = 0.30
        intersection = 0.18
        sample = simulate_binary_joint_distribution(
            probability_a,
            probability_b,
            intersection,
            sample_size=200_000,
            seed=42,
        )
        empirical_a = sample["event_a"].mean()
        empirical_b = sample["event_b"].mean()
        empirical_intersection = (
            sample["event_a"] & sample["event_b"]
        ).mean()
        self.assertAlmostEqual(empirical_a, probability_a, delta=0.005)
        self.assertAlmostEqual(empirical_b, probability_b, delta=0.005)
        self.assertAlmostEqual(
            empirical_intersection,
            intersection,
            delta=0.005,
        )

    def test_zero_probability_alert_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bayes_posterior(0.0, 0.0, 0.0)

    def test_invalid_decision_costs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            expected_decision_threshold(-1.0, 100.0)
        with self.assertRaises(ValueError):
            expected_decision_threshold(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()

