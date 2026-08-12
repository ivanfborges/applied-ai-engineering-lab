"""Numerical and manifest tests for the Day 10 visual laboratory."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))

from visual_lab import EXPECTED_OUTPUTS, PUBLIC_PREVIEW_CANDIDATES  # noqa: E402
from visualizations.animated_experiments import (  # noqa: E402
    BIASED_SAMPLE_SIZES,
    calculate_biased_size_results,
    calculate_rare_event_counts,
)
from visualizations.interactive_experiments import (  # noqa: E402
    calculate_explorer_scenarios,
)
from visualizations.static_experiments import (  # noqa: E402
    SAMPLE_SIZES,
    calculate_dependence_results,
    calculate_group_split_results,
    calculate_llm_evaluation_results,
    calculate_sample_size_results,
    calculate_strategy_results,
)
from visualizations.visual_utils import (  # noqa: E402
    create_population,
    validate_effective_sample_size,
)


class VisualNumericalTests(unittest.TestCase):
    """Check statistical mechanisms with Monte Carlo-appropriate tolerances."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.population = create_population()

    def test_sample_size_standard_errors_track_theory(self) -> None:
        _, empirical, theoretical = calculate_sample_size_results(
            self.population,
            trials=800,
        )

        self.assertTrue(np.all(np.diff(empirical) < 0.0))
        np.testing.assert_allclose(empirical, theoretical, rtol=0.12)
        self.assertEqual(empirical.shape, SAMPLE_SIZES.shape)

    def test_strategy_centers_and_variances_follow_configured_design(self) -> None:
        _, summaries = calculate_strategy_results(self.population, trials=800)

        self.assertLess(abs(summaries["Simple random"]["bias"]), 1.5)
        self.assertGreater(summaries["Selection biased"]["bias"], 40.0)
        self.assertLess(abs(summaries["Stratified + weighted"]["bias"]), 1.5)
        self.assertLess(
            summaries["Stratified + weighted"]["variance"],
            summaries["Simple random"]["variance"],
        )
        for summary in summaries.values():
            self.assertAlmostEqual(
                summary["mse"],
                summary["variance_plus_bias_squared"],
            )

    def test_biased_data_gets_narrower_while_bias_persists(self) -> None:
        _, _, summaries = calculate_biased_size_results(trials=500)
        first = summaries[int(BIASED_SAMPLE_SIZES[0])]
        last = summaries[int(BIASED_SAMPLE_SIZES[-1])]

        self.assertLess(last["variance"], first["variance"] / 100.0)
        self.assertGreater(abs(last["bias"]), 40.0)
        self.assertLess(abs(last["bias"] - first["bias"]), 8.0)

    def test_effective_sample_size_bounds(self) -> None:
        equal = validate_effective_sample_size(np.ones(100))
        concentrated = validate_effective_sample_size(
            np.concatenate([np.ones(99), np.array([100.0])])
        )

        self.assertAlmostEqual(equal, 100.0)
        self.assertLess(concentrated, equal)
        self.assertGreaterEqual(concentrated, 1.0)

    def test_dependence_increases_mean_variability_at_equal_row_count(self) -> None:
        independent, clustered, _, _, icc = calculate_dependence_results(trials=600)

        self.assertGreater(icc, 0.5)
        self.assertGreater(np.std(clustered), 5.0 * np.std(independent))

    def test_group_split_exposes_identity_generalization_gap(self) -> None:
        results = calculate_group_split_results()

        self.assertLess(float(results["random_mae"]), float(results["group_mae"]))
        random_overlap = set(results["users"][results["random_train"]]).intersection(
            results["users"][results["random_test"]]
        )
        group_overlap = set(results["users"][results["group_train"]]).intersection(
            results["users"][results["group_test"]]
        )
        self.assertTrue(random_overlap)
        self.assertFalse(group_overlap)

    def test_rare_event_absence_matches_binomial_scale(self) -> None:
        counts = calculate_rare_event_counts(samples=2_000)
        observed_absence = float(np.mean(counts == 0))
        theoretical_absence = 0.99**50

        self.assertAlmostEqual(observed_absence, theoretical_absence, delta=0.04)

    def test_llm_aggregates_use_their_declared_mixes(self) -> None:
        result = calculate_llm_evaluation_results()

        self.assertAlmostEqual(sum(result["production_mix"]), 1.0)
        self.assertAlmostEqual(sum(result["diagnostic_mix"]), 1.0)
        self.assertAlmostEqual(
            result["production_weighted"],
            float(np.dot(result["production_mix"], result["synthetic_scores"])),
        )
        self.assertNotEqual(
            result["diagnostic_average"],
            result["production_weighted"],
        )

    def test_interactive_scenarios_are_reproducible_and_complete(self) -> None:
        first_mean, first = calculate_explorer_scenarios(trials=30)
        second_mean, second = calculate_explorer_scenarios(trials=30)

        self.assertEqual(first_mean, second_mean)
        self.assertEqual(len(first), 12)
        for first_scenario, second_scenario in zip(first, second, strict=True):
            np.testing.assert_array_equal(first_scenario["random"], second_scenario["random"])
            np.testing.assert_array_equal(first_scenario["biased"], second_scenario["biased"])


class VisualManifestTests(unittest.TestCase):
    """Keep the public CLI inventory stable and internally consistent."""

    def test_manifest_has_expected_categories_and_unique_names(self) -> None:
        self.assertEqual(len(EXPECTED_OUTPUTS), len(set(EXPECTED_OUTPUTS)))
        self.assertEqual(sum(path.endswith(".png") for path in EXPECTED_OUTPUTS), 11)
        self.assertEqual(sum(path.endswith(".gif") for path in EXPECTED_OUTPUTS), 3)
        self.assertEqual(sum(path.endswith(".html") for path in EXPECTED_OUTPUTS), 2)
        self.assertTrue(set(PUBLIC_PREVIEW_CANDIDATES).issubset(EXPECTED_OUTPUTS))


if __name__ == "__main__":
    unittest.main()
