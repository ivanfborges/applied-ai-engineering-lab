"""Tests for the Day 13 visual-causality data and rendering pipeline."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from scipy import stats


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))

from day13_visual_causality import (  # noqa: E402
    EXPECTED_FILENAMES,
    build_collider_experiment,
    build_confounding_experiment,
    build_rag_experiment,
    build_simpson_experiment,
    correlation_relationships,
    generate_all,
)
from from_scratch import pearson_correlation  # noqa: E402


class VisualExperimentTests(unittest.TestCase):
    """Verify the properties each visual is designed to teach."""

    def test_correlation_shapes_distinguish_linear_and_monotonic_measures(self) -> None:
        relationships = correlation_relationships(seed=42)
        monotonic_x, monotonic_y = relationships["C. Nonlinear monotonic"]
        curved_x, curved_y = relationships["D. Non-monotonic"]
        outlier_x, outlier_y = relationships["F. Influential outlier"]

        monotonic_pearson = pearson_correlation(monotonic_x, monotonic_y)
        monotonic_spearman = float(stats.spearmanr(monotonic_x, monotonic_y).statistic)
        curved_pearson = pearson_correlation(curved_x, curved_y)
        outlier_pearson = pearson_correlation(outlier_x, outlier_y)
        outlier_spearman = float(stats.spearmanr(outlier_x, outlier_y).statistic)

        self.assertGreater(monotonic_spearman, monotonic_pearson + 0.04)
        self.assertLess(abs(curved_pearson), 0.25)
        self.assertGreater(outlier_spearman, outlier_pearson + 0.25)

    def test_confounding_adjustment_is_closer_under_constructed_assumptions(self) -> None:
        result = build_confounding_experiment(seed=42)

        self.assertGreater(abs(result.naive_coefficient - result.true_effect), 1.0)
        self.assertLess(
            abs(result.adjusted_coefficient - result.true_effect),
            abs(result.naive_coefficient - result.true_effect),
        )

    def test_simpson_reversal_and_collider_association_are_stable(self) -> None:
        simpson = build_simpson_experiment(seed=42)
        collider = build_collider_experiment(seed=42)

        self.assertGreater(simpson.aggregate_slope, 0.0)
        self.assertLess(simpson.low_intent_slope, 0.0)
        self.assertLess(simpson.high_intent_slope, 0.0)
        self.assertGreater(
            abs(collider.selected_correlation),
            abs(collider.population_correlation) + 0.25,
        )

    def test_randomized_rag_assignment_breaks_complexity_association(self) -> None:
        result = build_rag_experiment(seed=42)

        self.assertLess(abs(result.randomized_assignment_correlation), 0.08)
        self.assertLess(
            abs(result.randomized_slope - result.true_top_k_effect),
            abs(result.observational_slope - result.true_top_k_effect),
        )

    def test_quick_generation_creates_every_expected_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            results = generate_all(output_dir, seed=42, quick=True)

            self.assertEqual(tuple(result.path.name for result in results), EXPECTED_FILENAMES)
            for filename in EXPECTED_FILENAMES:
                path = output_dir / filename
                with self.subTest(filename=filename):
                    self.assertTrue(path.is_file())
                    self.assertGreater(path.stat().st_size, 0)
            html = (output_dir / "04_confounder_3d.html").read_text(encoding="utf-8")
            self.assertIn("Looking only at exposure and purchase", html)
            self.assertIn("scatter3d", html)
            self.assertIn("surface", html)


if __name__ == "__main__":
    unittest.main()
