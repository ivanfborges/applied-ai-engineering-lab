"""Numerical and manifest tests for the Statistical EDA Visual Lab."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


TOPIC_DIRECTORY = Path(__file__).resolve().parents[1]
VISUALIZATION_DIRECTORY = TOPIC_DIRECTORY / "visualizations"
sys.path.insert(0, str(VISUALIZATION_DIRECTORY))

from generate_all import EXPECTED_OUTPUTS  # noqa: E402
from visual_utils import (  # noqa: E402
    anscombe_quartet,
    simpsons_paradox_data,
    synthetic_ai_workload,
)


class VisualDataTests(unittest.TestCase):
    """Validate the known structures used by the educational charts."""

    def test_synthetic_workload_is_reproducible_and_has_expected_schema(self) -> None:
        first = synthetic_ai_workload(sample_size=250, seed=11)
        second = synthetic_ai_workload(sample_size=250, seed=11)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(
            set(first.columns),
            {
                "input_tokens",
                "output_tokens",
                "document_tokens",
                "latency_ms",
                "retrieval_score",
                "chunks_retrieved",
                "cost_usd",
                "model",
                "request_type",
            },
        )
        self.assertTrue(first["retrieval_score"].between(0, 1).all())
        self.assertTrue((first["latency_ms"] > 0).all())
        self.assertGreater(
            first[["input_tokens", "latency_ms"]].corr().iloc[0, 1],
            0.25,
        )
        self.assertGreater(
            first[["document_tokens", "chunks_retrieved"]].corr().iloc[0, 1],
            0.50,
        )

    def test_simpsons_paradox_reverses_aggregate_direction(self) -> None:
        data = simpsons_paradox_data()
        overall = data[["x", "y"]].corr().iloc[0, 1]
        within_group = [
            frame[["x", "y"]].corr().iloc[0, 1]
            for _, frame in data.groupby("group")
        ]

        self.assertGreater(overall, 0.50)
        self.assertTrue(all(correlation < -0.50 for correlation in within_group))

    def test_anscombe_summaries_are_nearly_equal(self) -> None:
        summaries = []
        for _, x, y in anscombe_quartet():
            summaries.append(
                (
                    np.mean(x),
                    np.mean(y),
                    np.var(x, ddof=1),
                    np.var(y, ddof=1),
                    np.corrcoef(x, y)[0, 1],
                )
            )

        reference = np.array(summaries[0])
        for summary in summaries[1:]:
            np.testing.assert_allclose(summary, reference, atol=0.01)

    def test_output_manifest_is_unique_and_bounded(self) -> None:
        self.assertEqual(len(EXPECTED_OUTPUTS), len(set(EXPECTED_OUTPUTS)))
        self.assertEqual(len(EXPECTED_OUTPUTS), 22)
        self.assertTrue(all(path.startswith("outputs/") for path in EXPECTED_OUTPUTS))
        self.assertTrue(
            all(
                Path(path).suffix.lower() in {".png", ".gif", ".html"}
                for path in EXPECTED_OUTPUTS
            )
        )


if __name__ == "__main__":
    unittest.main()
