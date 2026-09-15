"""Boundary tests and offline artifact smoke tests for Day 17."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))
import visual_experiments as exp
from validation_visual_lab import CATALOG, run_view


@pytest.mark.parametrize("n", [10, 11, 120])
def test_three_way_partition(n):
    parts = exp.split_three(n)
    exp.assert_partition(n, *parts)
    for a, b in zip(parts, exp.split_three(n)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        exp.checked_seed(seed)


@pytest.mark.parametrize("parts", [
    ([0, 1], [1, 2]), ([0], [2]), ([0, 1], [2, 4]),
    ([0.0, 1.0], [2.0]), ([], [0, 1, 2]),
])
def test_invalid_partitions(parts):
    with pytest.raises(ValueError):
        exp.assert_partition(3, *parts)


def test_fold_coverage_and_class_proportions():
    result = exp.kfold_experiment()
    held_out = np.concatenate([v for _, v in result["folds"]])
    np.testing.assert_array_equal(np.sort(held_out), np.arange(600))
    assert np.isfinite(result["scores"]).all()
    rates = exp.stratification_experiment()
    np.testing.assert_allclose(rates["rates"][1], rates["global_rate"])


def test_group_membership_is_indivisible():
    result = exp.group_experiment()
    np.testing.assert_array_equal(result["fractions"]["GroupKFold"].sum(axis=1), np.ones(100))
    assert set(np.unique(result["fractions"]["GroupKFold"])) <= {0, 1}
    assert result["overlap"]["GroupKFold"] == [0] * 5
    for train, valid in result["folds"]["GroupKFold"]:
        assert set(result["groups"][train]).isdisjoint(result["groups"][valid])


def test_temporal_windows_and_reserved_test():
    result = exp.drift_experiment()
    previous = set()
    for train, valid in result["folds"]["Expanding"]:
        assert previous <= set(train)
        assert train.max() < valid.min()
        assert valid.max() < 480
        previous = set(train)
    for train, valid in result["folds"]["Rolling"]:
        assert len(train) == 120
        assert train.max() < valid.min()
    assert result["slope"][0] == 1
    assert result["slope"][-1] == -1


def test_preprocessing_stats_are_fold_local():
    result = exp.preprocessing_experiment()
    train = result["folds"][0][0]
    np.testing.assert_allclose(result["local"].mean_, result["X"][train].mean(axis=0))
    altered = result["X"].copy()
    altered[300:] += 500
    local = StandardScaler().fit(altered[train])
    np.testing.assert_allclose(local.mean_, result["local"].mean_)
    assert not np.allclose(StandardScaler().fit(altered).mean_, result["global"].mean_)


def test_winners_use_only_validation_and_keep_first_tie():
    np.testing.assert_array_equal(exp.running_winners([0.2, 0.4, 0.4, 0.1, 0.5]), [0, 1, 1, 1, 4])
    result = exp.overfitting_experiment()
    assert np.all(np.diff(result["best_validation"]) >= 0)
    np.testing.assert_array_equal(result["winner_test"], result["test"][result["winners"]])


@pytest.mark.parametrize("bad", [[], [np.nan], [[1, 2]]])
def test_invalid_selection_scores(bad):
    with pytest.raises(ValueError):
        exp.running_winners(bad)


def test_nested_mapping_and_outer_score():
    result = exp.nested_experiment()
    X, y = exp.classification_data(n=360)
    for (train, valid), inner, row in zip(result["outer"], result["inner"], result["rows"]):
        for t, v in inner:
            assert set(t) | set(v) == set(train)
            assert set(t).isdisjoint(v)
            assert set(t).isdisjoint(valid)
            assert set(v).isdisjoint(valid)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=row["C"], max_iter=1000))
        model.fit(X[train], y[train])
        expected = roc_auc_score(y[valid], model.predict_proba(X[valid])[:, 1])
        assert row["outer_auc"] == pytest.approx(expected)


def test_purchase_average_uses_only_past():
    result = exp.purchase_experiment()
    assert result["asof_march"] == pytest.approx(result["monthly"][:3].mean())
    assert result["full_year"] == pytest.approx(result["monthly"].mean())


def test_cli_list_and_invalid_view():
    command = [sys.executable, str(TOPIC / "validation_visual_lab.py")]
    listing = subprocess.run(command + ["--list"], capture_output=True, text=True, check=True)
    assert "17 - Run all" in listing.stdout
    invalid = subprocess.run(command + ["--view", "99"], capture_output=True, text=True)
    assert invalid.returncode == 2


def test_offline_png_gif_html_and_records(tmp_path):
    record = run_view(1, 42, tmp_path)
    with Image.open(tmp_path / "train_val_test_split.gif") as gif:
        assert gif.n_frames > 5
        gif.seek(gif.n_frames-1)
        assert gif.width >= 800
    with Image.open(tmp_path / "train_val_test_split.png") as png:
        png.verify()
    run_view(14, 42, tmp_path)
    page = (tmp_path / "temporal_split_3d.html").read_text(encoding="utf-8")
    assert "Plotly.newPlot" in page
    assert '<script src=' not in page
    saved = json.loads((tmp_path / "experiment_01.json").read_text(encoding="utf-8"))
    assert saved["result"] == record["result"]
    assert len(CATALOG) == 16
    assert (tmp_path / "index.html").is_file()
