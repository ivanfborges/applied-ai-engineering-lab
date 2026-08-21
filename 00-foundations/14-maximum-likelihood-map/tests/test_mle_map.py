"""Tests for the Day 14 estimators and logistic-regression objective."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))

from example import (  # noqa: E402
    bernoulli_log_likelihood,
    bernoulli_mle,
    beta_bernoulli_log_posterior,
    beta_bernoulli_map,
    compare_sample_sizes,
    grid_mode,
)
from from_scratch import (  # noqa: E402
    fit_logistic_regression,
    generate_synthetic_classification,
    negative_log_posterior,
    negative_log_posterior_gradient,
    sigmoid,
)


class BernoulliEstimationTests(unittest.TestCase):
    """Verify analytical estimates, grid modes, and boundary behavior."""

    def test_analytical_estimates_match_dense_grid_modes(self) -> None:
        grid = np.linspace(0.00001, 0.99999, 200_000)
        mle = bernoulli_mle(7, 10)
        map_estimate = beta_bernoulli_map(7, 10, alpha=2.0, beta=2.0)

        grid_mle = grid_mode(grid, bernoulli_log_likelihood(grid, 7, 10))
        grid_map = grid_mode(
            grid,
            beta_bernoulli_log_posterior(
                grid, 7, 10, alpha=2.0, beta=2.0
            ),
        )

        self.assertAlmostEqual(mle, 0.7)
        self.assertAlmostEqual(map_estimate, 2.0 / 3.0)
        self.assertLess(abs(grid_mle - mle), 1e-5)
        self.assertLess(abs(grid_map - map_estimate), 1e-5)

    def test_fixed_prior_influence_decreases_with_sample_size(self) -> None:
        gaps = [abs(row.mle - row.map_estimate) for row in compare_sample_sizes()]

        self.assertTrue(all(left > right for left, right in zip(gaps, gaps[1:])))

    def test_boundary_posterior_modes_are_explicit(self) -> None:
        self.assertEqual(beta_bernoulli_map(0, 1, alpha=1.0, beta=1.0), 0.0)
        self.assertEqual(beta_bernoulli_map(1, 1, alpha=1.0, beta=1.0), 1.0)
        self.assertEqual(beta_bernoulli_map(0, 1, alpha=0.5, beta=0.5), 0.0)

    def test_log_likelihood_handles_probability_boundaries(self) -> None:
        self.assertEqual(bernoulli_log_likelihood(0.0, 0, 4), 0.0)
        self.assertEqual(bernoulli_log_likelihood(1.0, 4, 4), 0.0)
        self.assertEqual(bernoulli_log_likelihood(0.0, 1, 4), -math.inf)
        self.assertEqual(bernoulli_log_likelihood(1.0, 3, 4), -math.inf)

    def test_invalid_estimation_inputs_raise_explicit_errors(self) -> None:
        invalid_calls = (
            lambda: bernoulli_mle(-1, 10),
            lambda: bernoulli_mle(1, 0),
            lambda: bernoulli_mle(True, 10),
            lambda: beta_bernoulli_map(1, 2, alpha=0.0, beta=1.0),
            lambda: bernoulli_log_likelihood(math.nan, 1, 2),
            lambda: grid_mode(np.array([0.1]), np.array([0.0])),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


class LogisticMapTests(unittest.TestCase):
    """Check numerical stability, gradient correctness, and shrinkage."""

    def test_sigmoid_is_stable_for_large_finite_logits(self) -> None:
        probabilities = sigmoid(np.array([-1_000.0, 0.0, 1_000.0]))

        np.testing.assert_allclose(probabilities, np.array([0.0, 0.5, 1.0]))

    def test_map_gradient_matches_finite_differences(self) -> None:
        features = np.array(
            [[1.0, -1.0, 0.5], [1.0, 0.2, -0.3], [1.0, 1.5, 0.7]]
        )
        labels = np.array([0.0, 1.0, 1.0])
        weights = np.array([-0.2, 0.4, -0.1])
        epsilon = 1e-6
        numerical = np.empty_like(weights)
        for index in range(weights.size):
            step = np.zeros_like(weights)
            step[index] = epsilon
            numerical[index] = (
                negative_log_posterior(
                    features, labels, weights + step, prior_std=1.3
                )
                - negative_log_posterior(
                    features, labels, weights - step, prior_std=1.3
                )
            ) / (2.0 * epsilon)

        analytical = negative_log_posterior_gradient(
            features, labels, weights, prior_std=1.3
        )
        np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-7)

    def test_sum_and_mean_scale_the_complete_map_objective(self) -> None:
        features, labels, _ = generate_synthetic_classification(
            seed=4, sample_size=40
        )
        weights = np.array([0.1, -0.2, 0.3, -0.4])

        summed = negative_log_posterior(
            features, labels, weights, prior_std=0.8, reduction="sum"
        )
        averaged = negative_log_posterior(
            features, labels, weights, prior_std=0.8, reduction="mean"
        )
        summed_gradient = negative_log_posterior_gradient(
            features, labels, weights, prior_std=0.8, reduction="sum"
        )
        averaged_gradient = negative_log_posterior_gradient(
            features, labels, weights, prior_std=0.8, reduction="mean"
        )

        self.assertAlmostEqual(summed / features.shape[0], averaged)
        np.testing.assert_allclose(
            summed_gradient / features.shape[0], averaged_gradient
        )

    def test_stronger_prior_shrinks_non_intercept_coefficients(self) -> None:
        features, labels, _ = generate_synthetic_classification(
            seed=9, sample_size=160
        )
        mle = fit_logistic_regression(features, labels)
        weak_map = fit_logistic_regression(features, labels, prior_std=2.0)
        strong_map = fit_logistic_regression(features, labels, prior_std=0.4)

        norms = [
            np.linalg.norm(fit.weights[1:])
            for fit in (mle, weak_map, strong_map)
        ]
        self.assertGreater(norms[0], norms[1])
        self.assertGreater(norms[1], norms[2])

    def test_synthetic_generation_is_reproducible(self) -> None:
        first = generate_synthetic_classification(seed=3, sample_size=30)
        second = generate_synthetic_classification(seed=3, sample_size=30)
        for first_array, second_array in zip(first, second, strict=True):
            np.testing.assert_array_equal(first_array, second_array)

    def test_invalid_logistic_inputs_raise_explicit_errors(self) -> None:
        features = np.ones((4, 2))
        labels = np.array([0.0, 1.0, 0.0, 1.0])
        invalid_calls = (
            lambda: sigmoid(math.inf),
            lambda: negative_log_posterior(features, labels, np.zeros(3)),
            lambda: negative_log_posterior(
                features, np.array([0.0, 2.0, 0.0, 1.0]), np.zeros(2)
            ),
            lambda: negative_log_posterior(
                features, labels, np.zeros(2), prior_std=-1.0
            ),
            lambda: fit_logistic_regression(features, labels, max_iterations=0),
            lambda: generate_synthetic_classification(sample_size=10),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_call()


if __name__ == "__main__":
    unittest.main()
