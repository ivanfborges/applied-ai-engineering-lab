"""Unit tests for the visual explorer's numerical foundation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

VISUALIZATIONS = Path(__file__).resolve().parents[1] / "visualizations"
sys.path.insert(0, str(VISUALIZATIONS))

import math_utils  # noqa: E402


class VectorOperationTests(unittest.TestCase):
    def test_dot_norm_distance_and_angle(self) -> None:
        a = np.array([3.0, 4.0])
        b = np.array([-4.0, 3.0])
        self.assertAlmostEqual(math_utils.dot_product(a, b), 0.0)
        self.assertAlmostEqual(math_utils.vector_norm(a), 5.0)
        self.assertAlmostEqual(math_utils.distance(a, b), np.sqrt(50.0))
        self.assertAlmostEqual(math_utils.angle_degrees(a, b), 90.0)

    def test_normalization_has_unit_length(self) -> None:
        normalized = math_utils.normalize([3.0, 4.0])
        np.testing.assert_allclose(normalized, [0.6, 0.8])
        self.assertAlmostEqual(math_utils.vector_norm(normalized), 1.0)

    def test_projection(self) -> None:
        projected = math_utils.projection([3.0, 2.0], [1.0, 0.0])
        np.testing.assert_allclose(projected, [3.0, 0.0])

    def test_zero_vector_errors_are_explicit(self) -> None:
        with self.assertRaises(ValueError):
            math_utils.normalize([0.0, 0.0])
        with self.assertRaises(ValueError):
            math_utils.cosine_similarity([1.0, 0.0], [0.0, 0.0])
        with self.assertRaises(ValueError):
            math_utils.projection([1.0, 0.0], [0.0, 0.0])


class MatrixOperationTests(unittest.TestCase):
    def test_rotation(self) -> None:
        matrix = math_utils.rotation_matrix(90)
        transformed = math_utils.apply_transformation([[1.0, 0.0]], matrix)
        np.testing.assert_allclose(transformed, [[0.0, 1.0]], atol=1e-12)

    def test_composition_follows_application_order(self) -> None:
        scale = math_utils.scaling_matrix(2.0, 1.0)
        rotate = math_utils.rotation_matrix(90)
        composed = math_utils.compose_transformations([scale, rotate])
        result = math_utils.apply_transformation([[1.0, 0.0]], composed)
        np.testing.assert_allclose(result, [[0.0, 2.0]], atol=1e-12)

    def test_transformation_order_changes_result(self) -> None:
        scale = math_utils.scaling_matrix(2.0, 1.0)
        rotate = math_utils.rotation_matrix(45)
        first = math_utils.compose_transformations([scale, rotate])
        second = math_utils.compose_transformations([rotate, scale])
        self.assertFalse(np.allclose(first, second))


class AppliedExampleTests(unittest.TestCase):
    def test_standardization_centers_and_scales_features(self) -> None:
        values = np.array([[1.0, 100.0], [2.0, 300.0], [3.0, 500.0]])
        standardized, _, _ = math_utils.standardize_features(values)
        np.testing.assert_allclose(standardized.mean(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(standardized.std(axis=0), [1.0, 1.0])

    def test_embedding_ranking_uses_metric_direction(self) -> None:
        labels = ["same direction, large", "same direction", "orthogonal"]
        embeddings = np.array([[10.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        query = np.array([1.0, 0.0])
        dot_ranking = math_utils.rank_embedding_labels(
            labels, embeddings, query, "Dot product"
        )
        cosine_ranking = math_utils.rank_embedding_labels(
            labels, embeddings, query, "Cosine similarity"
        )
        self.assertEqual(dot_ranking[0][0], "same direction, large")
        self.assertAlmostEqual(cosine_ranking[0][1], 1.0)
        self.assertAlmostEqual(cosine_ranking[1][1], 1.0)

    def test_distance_concentration_is_deterministic(self) -> None:
        first = math_utils.distance_concentration([2, 10, 100], seed=7)
        second = math_utils.distance_concentration([2, 10, 100], seed=7)
        for key in first:
            np.testing.assert_allclose(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
