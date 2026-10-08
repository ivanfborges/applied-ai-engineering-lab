"""Smoke-test actual library workflows, not historical numerical results."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import example


@pytest.mark.parametrize("dataset", ["numeric", "mixed"])
def test_three_library_fit_predict_and_validation_selection(dataset, monkeypatch, tmp_path):
    # Local core-only runs may skip this integration check. CI installs and
    # imports every library explicitly before running the isolated test files.
    for package in ("xgboost", "lightgbm", "catboost"):
        pytest.importorskip(package, reason="requires the shared boosting extra")

    X, y = make_classification(
        n_samples=160, n_features=4, n_informative=3, n_redundant=0,
        weights=[0.6, 0.4], random_state=example.SEED,
    )
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
    categories = []
    if dataset == "mixed":
        # Small string categories exercise the actual train-only representations.
        frame["merchant"] = [f"merchant_{i % 8}" for i in range(len(frame))]
        frame["channel"] = [f"channel_{i % 3}" for i in range(len(frame))]
        categories = ["merchant", "channel"]

    def small_data(requested):
        assert requested == dataset
        return frame.copy(), y.copy(), list(categories)

    monkeypatch.setattr(example, "make_data", small_data)
    monkeypatch.setattr(example, "ROUNDS", 12)
    monkeypatch.setattr(example, "PATIENCE", 3)
    monkeypatch.chdir(tmp_path)
    record = example.run_experiment(dataset)

    configuration = record["configuration"]
    assert configuration["synthetic"] is True
    assert configuration["split_sizes"] == {"train": 96, "validation": 32, "test": 32}
    indices = np.concatenate(list(configuration["split_indices"].values()))
    np.testing.assert_array_equal(np.sort(indices), np.arange(160))
    candidates = record["results"]["candidates"]
    assert [row["model"] for row in candidates] == ["XGBoost", "LightGBM", "CatBoost"]
    for row in candidates:
        assert 1 <= row["selected_rounds"] <= 12
        for partition in ("train", "validation"):
            assert set(row[partition]) == {"log_loss", "roc_auc", "brier"}
            assert np.isfinite(list(row[partition].values())).all()
            assert 0 <= row[partition]["roc_auc"] <= 1
            assert 0 <= row[partition]["brier"] <= 1

    winner = min(candidates, key=lambda row: row["validation"]["log_loss"])["model"]
    assert record["results"]["selected_model"] == winner
    for key in ("selected_test", "constant_baseline_test"):
        assert np.isfinite(list(record["results"][key].values())).all()
    widths = configuration["feature_counts"]
    assert widths["LightGBM"] == widths["CatBoost"] == len(frame.columns)
    if categories:
        assert widths["XGBoost"] > widths["LightGBM"]
    else:
        assert widths["XGBoost"] == 4
    assert record["review_status"] == "pending author review"
    assert not list(tmp_path.iterdir()), "the workflow must not write training artifacts"
