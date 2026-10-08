"""Mechanism identities, fit boundaries, frozen stress models, and rendering."""
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

TOPIC = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load("day34_visual_core_tests", "missing_data_visual_core.py")
previous = sys.modules.get("missing_data_visual_core")
sys.modules["missing_data_visual_core"] = core
try:
    visual = load("day34_visual_tests", "visualize_missing_data.py")
finally:
    if previous is None:
        del sys.modules["missing_data_visual_core"]
    else:
        sys.modules["missing_data_visual_core"] = previous


def test_complete_data_is_deterministic_and_nontrivial():
    X, y = core.generate_dataset()
    other, labels = core.generate_dataset()
    np.testing.assert_array_equal(X, other)
    np.testing.assert_array_equal(y, labels)
    assert X.shape == (2400, 4)
    assert np.isfinite(X).all() and (X[:, 1] > 0).all()
    assert set(y) == {0, 1}
    assert .2 < y.mean() < .8


@pytest.mark.parametrize("mechanism", ["MCAR", "MAR", "MNAR"])
def test_masks_preserve_truth_and_only_hide_income(mechanism):
    X, _ = core.generate_dataset()
    original = X.copy()
    masked, mask, p = core.inject_mechanism(X, mechanism)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(masked[:, [0, 2, 3]], X[:, [0, 2, 3]])
    np.testing.assert_array_equal(masked[~mask, 1], X[~mask, 1])
    assert np.isnan(masked[mask, 1]).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_mcar_independence_mar_observed_driver_mnar_hidden_driver():
    X, _ = core.generate_dataset()
    changed_income = X.copy()
    changed_income[:, 1] *= 10
    changed_age = X.copy()
    changed_age[:, 0] += 40
    for alternate in (changed_income, changed_age):
        np.testing.assert_array_equal(core.inject_mechanism(X, "MCAR")[1],
                                      core.inject_mechanism(alternate, "MCAR")[1])
    np.testing.assert_array_equal(core.inject_mar(X)[1], core.inject_mar(changed_income)[1])
    assert not np.array_equal(core.inject_mar(X)[1], core.inject_mar(changed_age)[1])
    np.testing.assert_array_equal(core.inject_mnar(X)[1], core.inject_mnar(changed_age)[1])
    assert not np.array_equal(core.inject_mnar(X)[1], core.inject_mnar(changed_income)[1])


def test_progressive_masks_are_nested_and_zero_stays_complete():
    X, _ = core.generate_dataset()
    for mechanism in ("MCAR", "MAR", "MNAR"):
        assert not core.inject_mechanism(X, mechanism, fraction=0)[1].any()
        early = core.inject_mechanism(X, mechanism, fraction=.3)[1]
        late = core.inject_mechanism(X, mechanism, fraction=.8)[1]
        assert not (early & ~late).any()
    low = core.inject_mcar(X, .1, 71)
    high = core.inject_mcar(X, .6, 71)
    assert not (np.isnan(low) & ~np.isnan(high)).any()


@pytest.mark.parametrize("p", [-1, 1.1, float("nan"), float("inf"), True, "bad", [0.5]])
def test_invalid_probabilities_are_explicit(p):
    with pytest.raises(ValueError):
        core.inject_mcar([[1, 2]], p, 42)


def test_invalid_data_and_unknown_mechanism():
    with pytest.raises(ValueError):
        core.generate_dataset(99)
    with pytest.raises(ValueError):
        core.inject_mechanism([[1, np.nan]], "MNAR")
    with pytest.raises(ValueError):
        core.inject_mechanism([[1, 2]], "UNKNOWN")
    with pytest.raises(ValueError):
        core.imputed_versions([[1, np.nan], [2, np.nan]])


def test_mean_filling_variance_identity_and_no_mutation():
    X = np.array([[1, 10], [2, np.nan], [3, 30], [4, np.nan], [5, 50]], dtype=float)
    original = X.copy()
    with threadpool_limits(limits=1):
        filled = core.imputed_versions(X)
    observed = X[np.isfinite(X[:, 1]), 1]
    assert filled["Mean"][:, 1].mean() == pytest.approx(observed.mean())
    assert filled["Mean"][:, 1].var() == pytest.approx(len(observed) / len(X) * observed.var())
    for result in filled.values():
        assert np.isfinite(result).all()
        np.testing.assert_array_equal(result[np.isfinite(X)], X[np.isfinite(X)])
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("strategy", core.STRATEGIES)
def test_predictive_pipelines_keep_training_statistics(strategy):
    X, y = core.generate_dataset(200, 14)
    train = core.inject_mcar(X[:140], .2, 71)
    test = core.inject_mcar(X[140:] * 10, .2, 72)
    with threadpool_limits(limits=1):
        model = core.model_for(strategy, 14).fit(train, y[:140])
        prediction = model.predict_proba(test)
    assert np.isfinite(prediction).all()
    if strategy in ("Median", "Median + indicator"):
        np.testing.assert_allclose(model.steps[0][1].statistics_, np.nanmedian(train, axis=0))
    elif strategy == "Scaled KNN":
        np.testing.assert_allclose(model.steps[0][1].mean_, np.nanmean(train, axis=0))


def test_shift_fits_once_and_does_not_fit_test_statistics(monkeypatch):
    seed = 12
    train, test, labels, heldout_labels = core.split_dataset(seed, 200)
    expected = core.inject_mcar(train, .10, seed + 1000)
    instances = []

    class Spy:
        def __init__(self):
            self.fits, self.predictions = 0, 0
            instances.append(self)

        def fit(self, X, y):
            np.testing.assert_array_equal(X, expected)
            np.testing.assert_array_equal(y, labels)
            self.fits += 1
            return self

        def predict_proba(self, X):
            assert len(X) == len(test)
            self.predictions += 1
            p = core.sigmoid(np.nan_to_num(X[:, 2]))
            return np.column_stack((1-p, p))

    monkeypatch.setattr(core, "model_for", lambda strategy, seed: Spy())
    records = core.run_missingness_shift_experiment(seeds=[seed], rates=[.1, .5, .6], n=200)
    assert len(records) == 9
    assert len(instances) == 3
    assert all(instance.fits == 1 and instance.predictions == 3 for instance in instances)
    assert all(np.isfinite(record["auc"]) for record in records)


def test_summary_is_sample_sd_not_confidence_interval():
    rows = [dict(strategy="Median", rate=.2, auc=value) for value in (.6, .8, .7)]
    result = core.summarize(rows)[0]
    assert result["mean"] == pytest.approx(.7)
    assert result["std"] == pytest.approx(.1)
    assert result["repeats"] == 3


def test_indicator_metrics_reproduce():
    with threadpool_limits(limits=1):
        first = core.run_indicator_experiment(n=400)
        second = core.run_indicator_experiment(n=400)
    assert first["auc"] == second["auc"]
    assert all(np.isfinite(value) for value in first["auc"].values())
    np.testing.assert_array_equal(first["masked"], second["masked"])


def test_figure_saving_creates_image_and_closes_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(visual, "OUTPUT_DIR", tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    number = fig.number
    visual.save(fig, "smoke.png")
    assert (tmp_path / "smoke.png").stat().st_size > 1000
    assert not plt.fignum_exists(number)


def test_plotly_absence_is_graceful(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    X, _ = core.generate_dataset(100)
    assert visual.create_interactive_3d(X) == []
    assert "pip install plotly" in capsys.readouterr().out
