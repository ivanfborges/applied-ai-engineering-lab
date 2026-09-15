'''Tests for the Day 16 end-to-end classification workflow.'''

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIRECTORY))

from example import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_inference_bundle,
    generate_synthetic_dataset,
    load_inference_bundle,
    predict_with_bundle,
    save_inference_bundle,
    select_threshold_for_recall,
    split_dataset,
    train_model,
    validate_features,
)


class DataContractTests(unittest.TestCase):
    '''Verify deterministic generation, partitions, and explicit input errors.'''

    def test_synthetic_dataset_is_reproducible(self) -> None:
        first = generate_synthetic_dataset(sample_size=300, seed=4)
        second = generate_synthetic_dataset(sample_size=300, seed=4)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first[TARGET_COLUMN].unique()), {0, 1})
        self.assertTrue(first['monthly_usage_hours'].isna().any())

    def test_split_indices_are_disjoint_and_cover_every_row(self) -> None:
        data = generate_synthetic_dataset(sample_size=500, seed=7)
        splits = split_dataset(data, seed=7)
        index_sets = (
            set(splits.X_train.index),
            set(splits.X_validation.index),
            set(splits.X_test.index),
        )

        self.assertTrue(index_sets[0].isdisjoint(index_sets[1]))
        self.assertTrue(index_sets[0].isdisjoint(index_sets[2]))
        self.assertTrue(index_sets[1].isdisjoint(index_sets[2]))
        self.assertEqual(set.union(*index_sets), set(data.index))
        self.assertEqual(
            tuple(map(len, index_sets)),
            (300, 100, 100),
        )

    def test_invalid_feature_contract_raises_explicit_errors(self) -> None:
        valid = generate_synthetic_dataset(sample_size=200).drop(
            columns=TARGET_COLUMN
        )
        missing = valid.drop(columns='plan')
        invalid_numeric = valid.copy()
        invalid_numeric['tenure_months'] = invalid_numeric[
            'tenure_months'
        ].astype(object)
        invalid_numeric.loc[0, 'tenure_months'] = 'not-a-number'
        infinite = valid.copy()
        infinite.loc[0, 'monthly_usage_hours'] = np.inf

        with self.assertRaisesRegex(ValueError, 'Missing required'):
            validate_features(missing)
        with self.assertRaisesRegex(ValueError, 'non-numeric'):
            validate_features(invalid_numeric)
        with self.assertRaisesRegex(ValueError, 'infinite'):
            validate_features(infinite)
        with self.assertRaises(TypeError):
            validate_features(np.ones((2, len(FEATURE_COLUMNS))))

    def test_threshold_is_selected_on_validation_probabilities(self) -> None:
        labels = np.array([0, 1, 1, 0])
        probabilities = np.array([0.1, 0.4, 0.8, 0.3])

        threshold = select_threshold_for_recall(
            labels,
            probabilities,
            target_recall=0.75,
        )

        self.assertAlmostEqual(threshold, 0.4)
        with self.assertRaises(ValueError):
            select_threshold_for_recall(
                labels,
                probabilities,
                target_recall=0.0,
            )


class InferenceBundleTests(unittest.TestCase):
    '''Exercise fitted preprocessing and serialization as one inference unit.'''

    @classmethod
    def setUpClass(cls) -> None:
        data = generate_synthetic_dataset(sample_size=600, seed=11)
        cls.splits = split_dataset(data, seed=11)
        cls.training = train_model(
            cls.splits.X_train,
            cls.splits.y_train,
            c_values=(0.1, 1.0),
            cv_splits=3,
            seed=11,
        )
        probabilities = cls.training.pipeline.predict_proba(
            cls.splits.X_validation
        )[:, 1]
        cls.threshold = select_threshold_for_recall(
            cls.splits.y_validation,
            probabilities,
            target_recall=0.70,
        )

    def test_pipeline_handles_missing_values_and_unknown_category(self) -> None:
        raw_row = pd.DataFrame(
            [
                {
                    'tenure_months': 18,
                    'monthly_usage_hours': np.nan,
                    'support_tickets_90d': 1,
                    'days_since_last_active': 9,
                    'plan': 'enterprise',
                }
            ]
        )
        bundle = create_inference_bundle(
            self.training.pipeline,
            self.threshold,
            seed=11,
        )

        probabilities, predictions = predict_with_bundle(bundle, raw_row)

        self.assertEqual(probabilities.shape, (1,))
        self.assertEqual(predictions.shape, (1,))
        self.assertTrue(0 <= probabilities[0] <= 1)
        self.assertIn(predictions[0], (0, 1))

    def test_serialized_bundle_preserves_predictions(self) -> None:
        rows = self.splits.X_test.iloc[:12]
        bundle = create_inference_bundle(
            self.training.pipeline,
            self.threshold,
            seed=11,
        )
        expected_probabilities, expected_predictions = predict_with_bundle(
            bundle,
            rows,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'bundle.pkl'
            save_inference_bundle(bundle, path)
            loaded = load_inference_bundle(path)
            actual_probabilities, actual_predictions = predict_with_bundle(
                loaded,
                rows,
            )

        np.testing.assert_array_equal(actual_predictions, expected_predictions)
        np.testing.assert_allclose(
            actual_probabilities,
            expected_probabilities,
            rtol=0.0,
            atol=0.0,
        )


if __name__ == '__main__':
    unittest.main()
