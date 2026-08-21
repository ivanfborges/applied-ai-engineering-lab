"""Numerical and rendering checks for the Day 14 visual learning lab."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")

TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))

from mle_map_visual_lab import (  # noqa: E402
    EXPECTED_ASSET_FILENAMES,
    _cumulative_estimates,
    build_logistic_experiment,
    gaussian_regression_objectives,
    generate_demo,
    gradient_descent_path,
    logistic_objective_grid,
)


class VisualNumericalTests(unittest.TestCase):
    """Verify the relationships that the rendered assets claim to show."""

    def test_gaussian_nll_and_sse_have_the_same_grid_minimum(self) -> None:
        _, _, _, sse, nll, _, _ = gaussian_regression_objectives(seed=5)

        self.assertEqual(int(np.argmin(sse)), int(np.argmin(nll)))

    def test_logistic_grid_minimum_matches_fitted_mle(self) -> None:
        experiment = build_logistic_experiment(seed=8, sample_size=70)
        intercepts = np.linspace(-1.5, 1.0, 151)
        slopes = np.linspace(0.2, 3.5, 181)
        objective = logistic_objective_grid(experiment, intercepts, slopes)
        slope_index, intercept_index = np.unravel_index(
            int(np.argmin(objective)), objective.shape
        )

        self.assertLess(
            abs(intercepts[intercept_index] - experiment.mle_weights[0]), 0.03
        )
        self.assertLess(abs(slopes[slope_index] - experiment.mle_weights[1]), 0.03)

    def test_gradient_descent_lowers_the_nll(self) -> None:
        experiment = build_logistic_experiment(seed=9, sample_size=70)
        _, objectives = gradient_descent_path(experiment, iterations=70)

        self.assertLess(objectives[-1], objectives[0])
        self.assertTrue(np.all(np.diff(objectives) <= 1e-12))

    def test_gaussian_prior_shrinks_the_slope(self) -> None:
        experiment = build_logistic_experiment(seed=4, sample_size=70, prior_std=0.6)

        self.assertLess(
            abs(experiment.map_weights[1]), abs(experiment.mle_weights[1])
        )

    def test_wrong_prior_influence_decreases_on_constructed_sequence(self) -> None:
        observations = np.random.default_rng(14).binomial(1, 0.3, 1_000)
        _, mle, map_estimates = _cumulative_estimates(
            observations, alpha=20.0, beta=2.0
        )

        self.assertGreater(abs(map_estimates[9] - mle[9]), 0.20)
        self.assertLess(abs(map_estimates[-1] - mle[-1]), 0.02)

    def test_invalid_visual_inputs_raise_explicit_errors(self) -> None:
        with self.assertRaises(ValueError):
            build_logistic_experiment(sample_size=10)
        with self.assertRaises(ValueError):
            generate_demo("unknown", Path("unused"))


class VisualRenderingTests(unittest.TestCase):
    """Smoke-render every static and animated output at reduced cost."""

    def test_quick_full_lab_writes_the_expected_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            results = generate_demo(
                "all",
                output_directory,
                seed=14,
                save=True,
                show=False,
                quick=True,
            )

            generated = {result.path.name for result in results if result.path}
            self.assertEqual(generated, set(EXPECTED_ASSET_FILENAMES))
            for filename in EXPECTED_ASSET_FILENAMES:
                path = output_directory / filename
                with self.subTest(filename=filename):
                    self.assertTrue(path.is_file())
                    self.assertGreater(path.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
