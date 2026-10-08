"""Numerical semantics, immutable splits, sampling boundaries and render smoke checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

TOPIC = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("day33_visual", TOPIC / "visual_imbalanced_data.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_threshold_counts_and_metrics_match_library_at_ties():
    y, p = [0, 1, 0, 1, 0, 1], np.array([0, 0.2, 0.2, 0.7, 0.9, 1])
    thresholds = [0, 0.2, 0.5, 1, np.nextafter(1.0, np.inf)]
    curves = lab.threshold_metrics(y, p, thresholds)
    for i, threshold in enumerate(thresholds):
        predicted = p >= threshold
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        assert [curves[m][i] for m in ("tn", "fp", "fn", "tp")] == [tn, fp, fn, tp]
        assert curves["precision"][i] == precision_score(y, predicted, zero_division=0)
        assert curves["recall"][i] == recall_score(y, predicted, zero_division=0)
        assert curves["f1"][i] == f1_score(y, predicted, zero_division=0)
    for metric in ("tp", "fp", "recall", "alert_rate"):
        assert (np.diff(curves[metric]) <= 0).all()
    assert (np.diff(curves["fn"]) >= 0).all()


def test_validation_policy_optimizes_distinct_decisions_and_handles_cost_ties():
    y, p = [0, 1, 0, 1], [0.1, 0.3, 0.4, 0.9]
    policies = lab.validation_policies(y, p)
    candidates = [0, 0.2, 0.35, 0.65, 1.0]
    reference = lab.threshold_metrics(y, p, candidates)
    chosen = lab.threshold_metrics(y, p, [policies["f1"]])
    assert chosen["f1"][0] == max(reference["f1"])
    for scenario in policies["cost"]:
        selected = lab.threshold_metrics(y, p, [scenario["threshold"]])
        assert lab.empirical_cost(selected, scenario["c_fp"], scenario["c_fn"])[0] == min(
            lab.empirical_cost(reference, scenario["c_fp"], scenario["c_fn"])
        )
    # Equal F1 decisions preserve score groups and choose the highest threshold.
    tied = lab.validation_policies([0, 0], [1, 1])
    assert tied["f1"] > 1
    assert all(item["threshold"] > 1 for item in tied["cost"])


@pytest.mark.parametrize("y,p,t", [
    ([], [], [0.5]), ([0, 2], [0.1, 0.2], [0.5]),
    ([0, 1], [0.1], [0.5]), ([0, 1], [[0.1], [0.2]], [0.5]),
    ([0, 1], [np.nan, 0.2], [0.5]), ([0, 1], [-0.1, 0.2], [0.5]),
    ([0, 1], [0.1, 1.2], [0.5]), ([0, 1], [0.1, 0.2], []),
    ([0, 1], [0.1, 0.2], [-1]), ([0, 1], [0.1, 0.2], [np.inf]),
])
def test_score_and_threshold_inputs_are_checked(y, p, t):
    with pytest.raises(ValueError):
        lab.threshold_metrics(y, p, t)


@pytest.mark.parametrize("costs", [(0, 1), (-1, 20), (1, np.inf), (np.nan, 2)])
def test_cost_inputs_are_checked(costs):
    curves = lab.threshold_metrics([0, 1], [0.2, 0.8], [0.5])
    with pytest.raises(ValueError):
        lab.empirical_cost(curves, *costs)


def test_prevalence_precision_matches_bayes_and_increases():
    prevalence = np.array([0.001, 0.01, 0.05, 0.5])
    result = lab.precision_from_prevalence(prevalence)
    np.testing.assert_allclose(result, [0.90 * p / (0.90 * p + 0.05 * (1 - p)) for p in prevalence])
    assert (np.diff(result) > 0).all()
    assert lab.precision_from_prevalence(0.2, tpr=1, fpr=0) == 1


@pytest.mark.parametrize("prevalence,tpr,fpr", [(0, .9, .05), (1, .9, .05), (.1, -1, .05), (.1, .9, 2), (.1, 0, 0)])
def test_invalid_bayes_inputs_fail(prevalence, tpr, fpr):
    with pytest.raises(ValueError):
        lab.precision_from_prevalence(prevalence, tpr, fpr)


def test_interpolation_uses_one_scalar_and_endpoints():
    anchor, neighbor = np.array([2., 4.]), np.array([6., 8.])
    np.testing.assert_array_equal(lab.interpolate(anchor, neighbor, 0), anchor)
    np.testing.assert_array_equal(lab.interpolate(anchor, neighbor, 1), neighbor)
    np.testing.assert_array_equal(lab.interpolate(anchor, neighbor, .25), [3, 5])


@pytest.mark.parametrize("anchor,neighbor,fraction", [
    ([1], [2, 3], .5), ([1, 2], [3, 4], -1), ([1, 2], [3, 4], 1.1),
    ([np.nan, 2], [3, 4], .5), ([1, 2], [3, 4], [0.2, 0.3]),
])
def test_invalid_interpolation_inputs_fail(anchor, neighbor, fraction):
    with pytest.raises(ValueError):
        lab.interpolate(anchor, neighbor, fraction)


def test_smote_pair_excludes_self_even_with_duplicates():
    minority = np.array([[0., 0.], [0., 0.], [10., 10.]])
    i, j = lab.smote_pair(minority)
    assert i != j
    assert np.linalg.norm(minority[i] - minority[j]) == 0
    with pytest.raises(ValueError):
        lab.smote_pair([[1, 2]])


def test_deterministic_disjoint_stratified_splits_and_train_only_scaling():
    data, repeat = lab.make_dataset(), lab.make_dataset()
    np.testing.assert_array_equal(data["X"], repeat["X"])
    all_rows = np.concatenate([data[label] for label in ("train", "validation", "test")])
    np.testing.assert_array_equal(np.sort(all_rows), np.arange(3000))
    assert len(np.unique(all_rows)) == 3000
    assert [len(data[label]) for label in ("train", "validation", "test")] == [1800, 600, 600]
    models, weights = lab.fit_models(data)
    expected_train = data["X"][data["train"]]
    for model in models.values():
        scaler = model.named_steps["standardscaler"]
        np.testing.assert_allclose(scaler.mean_, expected_train.mean(axis=0))
        assert scaler.n_samples_seen_ == 1800
    assert weights["1"] > weights["0"]


def test_sampling_balances_training_and_preserves_original_splits():
    deps = lab.optional_dependencies()
    data = lab.make_dataset()
    old_X, old_y = data["X"].copy(), data["y"].copy()
    X, y = data["X"][data["train"]], data["y"][data["train"]]
    panels = lab.resample_training(X, y, deps)
    original_counts = np.bincount(y)
    np.testing.assert_array_equal(np.bincount(panels["Original training"]["y"]), original_counts)
    for label in ("Random oversampling", "SMOTE"):
        np.testing.assert_array_equal(np.bincount(panels[label]["y"]), [original_counts.max()] * 2)
    np.testing.assert_array_equal(np.bincount(panels["Random undersampling"]["y"]), [original_counts.min()] * 2)
    assert sum(panels["Random oversampling"]["multiplicity"]) == len(panels["Random oversampling"]["y"])
    np.testing.assert_allclose(panels["SMOTE"]["X"][:len(X)], X)
    np.testing.assert_array_equal(data["X"], old_X)
    np.testing.assert_array_equal(data["y"], old_y)
    for row in panels["Random undersampling"]["X"]:
        assert any(np.allclose(row, source) for source in X)


def test_calibration_bin_values_and_constant_scores():
    mean, observed, brier = lab.calibration_summary([0, 1, 0, 1], [0.5] * 4)
    np.testing.assert_array_equal(mean, [0.5])
    np.testing.assert_array_equal(observed, [0.5])
    assert brier == 0.25


def test_missing_optional_dependency_has_install_instruction(monkeypatch):
    original = lab.importlib.import_module

    def missing(name):
        if name.startswith("imblearn"):
            raise ImportError("test fixture missing dependency")
        return original(name)

    monkeypatch.setattr(lab.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="pip install"):
        lab.optional_dependencies()


def test_static_gif_and_offline_html_rendering(tmp_path):
    y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.3, 0.9])
    png, gif, smote, html = [tmp_path / name for name in ("counts.png", "threshold.gif", "smote.gif", "cost.html")]
    lab.plot_confusion_thresholds(y, p, png)
    lab.animate_threshold(y, p, gif, frames=3)
    lab.animate_smote(np.array([[1., 2.], [3., 4.]]), smote, frames=3)
    go = lab.optional_dependencies()["plotly.graph_objects"]
    lab.plot_3d_cost_surface(y, p, html, go)
    with Image.open(png) as image:
        assert image.width >= 1000
    for path in (gif, smote):
        with Image.open(path) as image:
            assert image.n_frames == 3
            assert image.info.get("duration", 0) > 0
    content = html.read_text(encoding="utf-8")
    assert "<html>" in content
    assert '<script src="https://cdn.plot.ly' not in content
    assert "plotly.js" in content.lower()
    assert "Exact validation minima" in content


def test_generate_lab_freezes_policies_before_scoring_test(monkeypatch, tmp_path):
    # No-render orchestration check: real split boundaries with stubbed model scores.
    dependencies = lab.optional_dependencies()
    monkeypatch.setattr(lab, "optional_dependencies", lambda: dependencies)
    selected = []
    original_select = lab.validation_policies

    def select(y, p):
        assert len(y) == 600
        selected.append(1)
        return original_select(y, p)

    class FixedModel:
        def predict_proba(self, X):
            if len(X) in (600, 2000) and selected:
                assert len(selected) == 1
            elif len(X) == 600:
                assert not selected
            p = 1 / (1 + np.exp(-X[:, 0]))
            return np.column_stack([1-p, p])

        @property
        def named_steps(self):
            class Identity:
                def transform(self, X):
                    return X
            return {"standardscaler": Identity()}

    def fit(data):
        assert len(data["train"]) == int(.6 * len(data["y"]))
        return {"Ordinary": FixedModel(), "Balanced weights": FixedModel()}, {"0": 1., "1": 10.}

    monkeypatch.setattr(lab, "validation_policies", select)
    monkeypatch.setattr(lab, "fit_models", fit)
    for name in (
        "plot_class_distribution", "plot_accuracy_trap", "plot_confusion_thresholds",
        "animate_threshold", "plot_threshold_tradeoff", "plot_class_weight_effect",
        "plot_resampling_comparison", "animate_smote", "plot_smote_failure",
        "plot_roc_vs_pr", "plot_precision_prevalence", "plot_business_cost",
        "plot_3d_cost_surface", "plot_ranking_vs_decision",
    ):
        monkeypatch.setattr(lab, name, lambda *args, **kwargs: {})
    monkeypatch.setattr(lab, "plot_calibration", lambda *args, **kwargs: {
        "Ordinary": {"brier": .1}, "Balanced weights": {"brier": .2}
    })
    report = lab.generate_lab(tmp_path, frames=2)
    assert len(selected) == 1
    assert report["interpretation_status"] == "Pending author review"
    assert (tmp_path / "visual_run.json").is_file()
