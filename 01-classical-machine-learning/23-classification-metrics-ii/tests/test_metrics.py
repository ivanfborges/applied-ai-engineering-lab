"""Checks for the educational ranking and threshold helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import pairwise_roc_auc, select_threshold, threshold_metrics


def test_pairwise_auc_matches_sklearn_with_ties():
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.9, 0.6, 0.4, 0.2, 0.2]
    assert pairwise_roc_auc(labels, scores) == pytest.approx(
        roc_auc_score(labels, scores)
    )
    assert pairwise_roc_auc(labels, np.array(scores) ** 4) == pytest.approx(
        pairwise_roc_auc(labels, scores)
    )


def test_threshold_selection_and_counts():
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    threshold = select_threshold(labels, scores, min_precision=0.6)
    assert threshold == pytest.approx(0.5)
    assert threshold_metrics(labels, scores, threshold) == {
        "tp": 3, "fp": 2, "fn": 0, "tn": 1,
        "precision": 0.6, "recall": 1.0, "f1": 0.75, "alerts": 5,
    }


def test_infeasible_threshold_and_invalid_inputs():
    with pytest.raises(ValueError, match="no nonempty threshold"):
        select_threshold([0, 1], [0.9, 0.1], min_precision=0.75)
    with pytest.raises(ValueError, match="both classes"):
        pairwise_roc_auc([1, 1], [0.2, 0.3])
    with pytest.raises(ValueError, match="finite"):
        threshold_metrics([0, 1], [0.2, float("nan")], 0.5)
    with pytest.raises(ValueError, match="binary"):
        threshold_metrics([0, 2], [0.2, 0.3], 0.5)
    with pytest.raises(ValueError, match="nonzero"):
        pairwise_roc_auc([], [])
    with pytest.raises(ValueError, match="min_precision"):
        select_threshold([0, 1], [0.2, 0.8], min_precision=0)
