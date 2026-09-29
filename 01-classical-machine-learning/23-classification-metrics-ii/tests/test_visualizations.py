"""Numerical checks for the Day 23 visual laboratory."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visualizations import (
    choose_policies,
    fit_calibrators,
    make_splits,
    prevalence_experiment,
    threshold_table,
)


def test_threshold_table_and_policy_choices():
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    rows = threshold_table(y, p, np.array([0.0, 0.5, 0.7, 1.0]))
    for row in rows:
        tn, fp, fn, tp = confusion_matrix(
            y, p >= row["threshold"], labels=[0, 1]
        ).ravel()
        assert (row["tn"], row["fp"], row["fn"], row["tp"]) == (
            tn, fp, fn, tp
        )
        assert row["alerts"] == tp + fp
        assert row["fpr"] == pytest.approx(fp / (fp + tn))
    choices = choose_policies(
        rows, min_precision=0.6, cost_fp=1, cost_fn=9, capacity=3
    )
    assert choices["f1"]["f1"] == max(row["f1"] for row in rows)
    assert choices["precision"]["precision"] >= 0.6
    assert choices["capacity"]["alerts"] <= 3
    assert choices["cost"] in rows


def test_weighted_prevalence_keeps_conditional_ranking_fixed():
    results = prevalence_experiment()
    assert [row["prevalence"] for row in results] == [0.5, 0.1, 0.01, 0.001]
    assert np.allclose([row["roc_auc"] for row in results],
                       results[0]["roc_auc"], atol=1e-12)
    assert all(0 <= row["ap"] <= 1 for row in results)
    assert all(0 <= row["precision_at_threshold"] <= 1 for row in results)


def test_calibration_mappings_use_separate_data_and_valid_probabilities():
    labels, scores = make_splits()
    assert all(len(labels[name]) == 1000 for name in labels)
    p = scores["test"]
    assert 0 <= roc_auc_score(labels["test"], p) <= 1
    assert np.array_equal(np.argsort(p), np.argsort(p**4))
    mapped = fit_calibrators(
        labels["calibration"], scores["calibration"], p
    )
    assert np.array_equal(mapped["Original"], p)
    assert all(np.all((values >= 0) & (values <= 1))
               for values in mapped.values())


def test_invalid_policy_and_generated_output_directory(tmp_path):
    rows = threshold_table(np.array([0, 1]), np.array([0.1, 0.9]))
    with pytest.raises(ValueError, match="Invalid policy"):
        choose_policies(rows, cost_fn=-1)
    from visualizations import plot_bayes_precision, style

    style()
    target = tmp_path / "nested"
    plot_bayes_precision(target)
    assert (target / "precision_vs_prevalence.png").is_file()
