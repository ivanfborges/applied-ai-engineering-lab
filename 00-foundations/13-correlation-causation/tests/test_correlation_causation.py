"""Tests for the Day 13 numerical helpers and synthetic experiments."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))

from example import (  # noqa: E402
    analyze_collider,
    analyze_confounding,
    generate_confounding_data,
    nonlinear_dependence,
)
from from_scratch import (  # noqa: E402
    ols_coefficients,
    partial_correlation,
    pearson_correlation,
)


class NumericalHelperTests(unittest.TestCase):
    """Verify formulas and explicit validation boundaries."""

    def test_pearson_matches_numpy(self) -> None:
        x = np.array([-2.0, -1.0, 1.0, 4.0])
        y = np.array([5.0, 2.0, 3.0, 9.0])

        self.assertAlmostEqual(
            pearson_correlation(x, y), float(np.corrcoef(x, y)[0, 1])
        )

    def test_ols_recovers_exact_linear_coefficients(self) -> None:
        x1 = np.arange(6, dtype=float)
        x2 = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        outcome = 3.0 + 2.0 * x1 - 4.0 * x2

        result = ols_coefficients(np.column_stack((x1, x2)), outcome)

        np.testing.assert_allclose(result, np.array([3.0, 2.0, -4.0]))

    def test_partial_correlation_removes_shared_linear_cause(self) -> None:
        control = np.arange(1.0, 9.0)
        control_design = np.column_stack((np.ones(control.size), control))
        raw_x = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        x_residual = raw_x - control_design @ np.linalg.lstsq(
            control_design, raw_x, rcond=None
        )[0]
        raw_y = np.array([1, 1, -1, -1, -1, -1, 1, 1], dtype=float)
        orthogonal_design = np.column_stack((control_design, x_residual))
        y_residual = raw_y - orthogonal_design @ np.linalg.lstsq(
            orthogonal_design, raw_y, rcond=None
        )[0]
        x = 2.0 * control + x_residual
        y = -3.0 * control + y_residual

        self.assertAlmostEqual(partial_correlation(x, y, control), 0.0, places=12)

    def test_invalid_inputs_raise_explicit_errors(self) -> None:
        invalid_calls = (
            lambda: pearson_correlation([1.0], [2.0]),
            lambda: pearson_correlation([1.0, 1.0], [2.0, 3.0]),
            lambda: pearson_correlation([1.0, math.nan], [2.0, 3.0]),
            lambda: ols_coefficients([[1.0], [1.0]], [2.0]),
            lambda: ols_coefficients([[1.0], [1.0]], [2.0, 3.0]),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


class SyntheticExperimentTests(unittest.TestCase):
    """Check constructed behavior, not performance on real-world data."""

    def test_generation_is_reproducible(self) -> None:
        first = generate_confounding_data(seed=7)
        second = generate_confounding_data(seed=7)

        for first_array, second_array in zip(first, second, strict=True):
            np.testing.assert_array_equal(first_array, second_array)

    def test_adjustment_recovers_known_effect_in_constructed_model(self) -> None:
        result = analyze_confounding(
            label="test", true_effect=0.0, randomized_exposure=False
        )

        self.assertGreater(result.correlation, 0.7)
        self.assertGreater(result.naive_coefficient, 2.0)
        self.assertLess(abs(result.adjusted_coefficient), 0.1)

    def test_randomization_removes_large_confounding_bias(self) -> None:
        result = analyze_confounding(
            label="test", true_effect=2.0, randomized_exposure=True
        )

        self.assertLess(abs(result.naive_coefficient - 2.0), 0.15)
        self.assertLess(abs(result.adjusted_coefficient - 2.0), 0.1)

    def test_nonlinearity_and_collider_traps(self) -> None:
        collider = analyze_collider()

        self.assertLess(abs(nonlinear_dependence()), 1e-12)
        self.assertLess(abs(collider.population_correlation), 0.05)
        self.assertLess(collider.selected_correlation, -0.3)

    def test_invalid_simulation_inputs_raise_explicit_errors(self) -> None:
        invalid_calls = (
            lambda: generate_confounding_data(sample_size=2),
            lambda: generate_confounding_data(true_effect=math.inf),
            lambda: analyze_collider(sample_size=99),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


if __name__ == "__main__":
    unittest.main()
