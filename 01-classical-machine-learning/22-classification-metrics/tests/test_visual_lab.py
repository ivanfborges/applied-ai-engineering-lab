"""Numerical checks for the Day 22 visual lab's reusable calculations."""

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest


TOPIC = Path(__file__).resolve().parents[1]


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load_module("from_scratch.py", "day22_visual_core")
with patch.dict(sys.modules, {"from_scratch": core}):
    lab = load_module("visualize_classification_metrics.py", "day22_visual_lab")


def test_threshold_counts_and_monotonic_changes():
    actual = [0, 0, 1, 1]
    probabilities = [0.1, 0.8, 0.4, 0.9]
    rows = lab.threshold_table(actual, probabilities, [0.2, 0.5, 0.95])
    assert [(r["tn"], r["fp"], r["fn"], r["tp"]) for r in rows] == [
        (1, 1, 0, 2), (1, 1, 1, 1), (2, 0, 2, 0)
    ]
    assert [r["predicted_positives"] for r in rows] == [3, 2, 0]
    assert [r["recall"] for r in rows] == [1, 0.5, 0]
    assert all(sum(r[k] for k in ("tn", "fp", "fn", "tp")) == 4 for r in rows)


def test_undefined_rates_and_invalid_scores():
    result = lab.metrics_at_threshold([0, 1], [0.1, 0.2], 0.5)
    assert result["precision"] == 0
    assert result["recall"] == 0
    assert result["f1"] == 0
    for scores, threshold in [([0.2, np.nan], 0.5), ([0.2, 1.2], 0.5),
                              ([0.2, 0.8], -0.1)]:
        with pytest.raises(ValueError):
            lab.metrics_at_threshold([0, 1], scores, threshold)
    with pytest.raises(ValueError):
        lab.threshold_table([0], [0.2], [])


def test_harmonic_mean_and_prevalence_identity():
    assert lab.harmonic_mean(0, 0) == 0
    assert lab.harmonic_mean(1, 0.1) == pytest.approx(2 / 11)
    p = np.array([0.01, 0.5])
    rates = lab.rates_at_prevalence(p, tpr=0.8, fpr=0.05)
    expected_precision = 0.8 * p / (0.8 * p + 0.05 * (1 - p))
    np.testing.assert_allclose(rates["precision"], expected_precision)
    np.testing.assert_allclose(rates["recall"], [0.8, 0.8])
    assert rates["precision"][0] < rates["precision"][1]
    np.testing.assert_allclose(
        rates["f1"], lab.harmonic_mean(rates["precision"], rates["recall"])
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: lab.harmonic_mean(-0.1, 0.5),
        lambda: lab.rates_at_prevalence([0, 0.5], 0.8, 0.05),
        lambda: lab.rates_at_prevalence([0.1], 1.1, 0.05),
    ],
)
def test_invalid_rate_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_constructed_imbalance_case_is_consistent():
    baseline, detector = lab.imbalance_case()
    assert (baseline["tn"], baseline["fp"], baseline["fn"], baseline["tp"]) == (
        990, 0, 10, 0
    )
    assert (detector["tn"], detector["fp"], detector["fn"], detector["tp"]) == (
        978, 12, 3, 7
    )
    assert baseline["accuracy"] == 0.99
    assert baseline["recall"] == 0
    assert detector["recall"] == 0.7

def test_validation_threshold_selector_and_absent_constraints():
    rows = [
        {"threshold": 0.2, "f1": 0.5, "recall": 0.9, "precision": 0.4},
        {"threshold": 0.4, "f1": 0.8, "recall": 0.7, "precision": 0.8},
        {"threshold": 0.8, "f1": 0.5, "recall": 0.1, "precision": 0.95},
    ]
    selected = lab.select_thresholds(rows)
    assert selected["best_f1"]["threshold"] == 0.4
    assert selected["recall_90"]["threshold"] == 0.2
    assert selected["precision_90"]["threshold"] == 0.8
    assert lab.select_thresholds(rows[1:])["recall_90"] is None
    with pytest.raises(ValueError):
        lab.select_thresholds([])
