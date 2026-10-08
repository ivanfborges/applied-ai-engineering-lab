"""Visual-lab geometry, fitted boundaries, availability and rendering checks."""
import builtins
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

TOPIC = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("day32_visual", TOPIC / "visual_feature_engineering.py")
visual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visual)


def test_neighbor_metric_changes_with_training_fitted_scaling():
    raw, scaled, scaler, train = visual.scaling_data()
    np.testing.assert_allclose(scaler.mean_, raw[train].mean(axis=0))
    np.testing.assert_allclose(scaler.scale_, raw[train].std(axis=0))
    assert set(visual.nearest_indices(raw, 0)) != set(visual.nearest_indices(scaled, 0))
    assert 0 not in visual.nearest_indices(raw, 0)
    old_mean = scaler.mean_.copy()
    shifted = raw[70:].copy()
    shifted[:, 1] *= 100
    scaler.transform(shifted)
    np.testing.assert_array_equal(scaler.mean_, old_mean)


@pytest.mark.parametrize("values,reference,k", [
    ([[1, np.nan], [2, 3]], 0, 1),
    ([[1, 2], [3, 4]], -1, 1),
    ([[1, 2], [3, 4]], 0, 2),
    ([[1, 2], [3, 4]], True, 1),
    ([[], []], 0, 1),
])
def test_neighbors_reject_invalid_inputs(values, reference, k):
    with pytest.raises(ValueError):
        visual.nearest_indices(values, reference, k)


def test_bins_use_training_boundaries_and_quantile_counts():
    train, holdout = visual.binning_data()
    model = visual.fit_bins(train, 8, "quantile")
    original = model.bin_edges_[0].copy()
    np.testing.assert_allclose(original, np.quantile(train, np.linspace(0, 1, 9)))
    counts = np.bincount(model.transform(train[:, None]).ravel().astype(int))
    np.testing.assert_array_equal(counts, np.repeat(40, 8))
    model.transform((holdout * 100)[:, None])
    np.testing.assert_array_equal(model.bin_edges_[0], original)
    uniform = visual.fit_bins(train, 8, "uniform")
    np.testing.assert_allclose(np.diff(uniform.bin_edges_[0]),
                               np.repeat(np.ptp(train) / 8, 8))


@pytest.mark.parametrize("values,bins,strategy", [
    ([1, 1], 4, "quantile"), ([1, 2], 1, "uniform"),
    ([1, np.inf], 4, "quantile"), ([1, 2], 4, "invalid"),
])
def test_bins_reject_invalid_inputs(values, bins, strategy):
    with pytest.raises(ValueError):
        visual.fit_bins(values, bins, strategy)


def test_lifted_plane_reconstructs_model_decision_scores():
    values, labels, train, holdout = visual.interaction_data()
    models = visual.train_boundary_models(values, labels, train)
    weight, intercept = visual.interaction_plane(models[1])
    lifted = np.column_stack([values, values[:, 0] * values[:, 1]])
    np.testing.assert_allclose(lifted @ weight + intercept,
                               models[1].decision_function(values), atol=1e-12)
    np.testing.assert_allclose(models[0].steps[0][1].mean_, values[train].mean(axis=0))
    np.testing.assert_allclose(models[1].steps[1][1].mean_, lifted[train].mean(axis=0))
    assert set(train).isdisjoint(holdout)
    assert np.any(labels != (values[:, 0] * values[:, 1] > 0))


def test_history_window_start_cutoff_availability_and_fixed_width():
    events, cutoff = visual.transaction_data()
    history = visual.eligible_history(events, 101, cutoff)
    assert set(history.index) == {1, 2, 3, 4, 5, 7}
    features = visual.history_features(events, 101, cutoff)
    assert features == {
        "transactions_last_7d": 2, "transactions_last_30d": 6,
        "average_amount_30d": pytest.approx(575 / 6),
        "max_amount_30d": 180., "distinct_merchants_30d": 3,
        "days_since_last_transaction": 1.,
    }
    changed = events.copy()
    changed.loc[[0, 6, 8, 9, 10], "amount"] = 1e9
    changed["target"] = 1
    assert visual.history_features(changed, 101, cutoff) == features
    assert len(visual.history_features(events, 999, cutoff)) == len(features)
    assert visual.history_features(events, 999, cutoff)["days_since_last_transaction"] is None


def test_history_excludes_events_available_exactly_at_prediction():
    events, cutoff = visual.transaction_data()
    events.loc[7, "available_at"] = cutoff
    assert visual.history_features(events, 101, cutoff)["transactions_last_30d"] == 5


@pytest.mark.parametrize("case", ["naive", "missing", "bad_amount", "bad_availability", "bad_window"])
def test_history_rejects_ambiguous_or_invalid_inputs(case):
    events, cutoff = visual.transaction_data()
    days = 30
    if case == "naive":
        cutoff = cutoff.tz_localize(None)
    elif case == "missing":
        events = events.drop(columns="merchant")
    elif case == "bad_amount":
        events.loc[0, "amount"] = np.inf
    elif case == "bad_availability":
        events.loc[0, "available_at"] = cutoff - pd.Timedelta(days=43)
    else:
        days = 0
    with pytest.raises(ValueError):
        visual.eligible_history(events, 101, cutoff, days=days)


def test_optional_plotly_missing_keeps_static_interaction_output(tmp_path, monkeypatch, capsys):
    original_import = builtins.__import__

    def without_plotly(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("Plotly unavailable for this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_plotly)
    visual.configure_style()
    paths = visual.create_interaction_visualization(tmp_path)
    assert [path.name for path in paths] == ["06_interaction_2d.png"]
    with Image.open(paths[0]) as image:
        assert image.format == "PNG"
        assert image.width > 800
    assert "Skipped" in capsys.readouterr().err


def test_pillow_gif_pauses_and_loops_without_ffmpeg(tmp_path):
    fig, ax = visual.plt.subplots(figsize=(3, 2))
    text = ax.text(0.5, 0.5, "", ha="center")

    def update(state):
        text.set_text(str(state))

    path = visual.save_gif(fig, update, [0] * 6 + [1] * 6, tmp_path, "smoke.gif")
    with Image.open(path) as image:
        assert image.info["loop"] == 0
        duration = 0
        for frame in range(image.n_frames):
            image.seek(frame)
            duration += image.info["duration"]
        assert duration == 6000
