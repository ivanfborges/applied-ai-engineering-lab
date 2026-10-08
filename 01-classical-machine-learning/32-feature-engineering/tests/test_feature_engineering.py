"""Numerical parity, point-in-time boundaries and train-only preprocessing."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load_module("day32_scratch", "from_scratch.py")
# The example's adjacent educational helper is isolated from other topics.
previous = sys.modules.get("from_scratch")
sys.modules["from_scratch"] = scratch
try:
    example = load_module("day32_example", "example.py")
finally:
    if previous is None:
        del sys.modules["from_scratch"]
    else:
        sys.modules["from_scratch"] = previous


def test_standardization_matches_library_including_constant_column():
    train = np.array([[1, 7], [3, 7], [8, 7]], dtype=float)
    holdout = np.array([[-100, 9], [20, 7]], dtype=float)
    mean, scale = scratch.fit_standardization(train)
    library = StandardScaler().fit(train)
    np.testing.assert_allclose(mean, library.mean_)
    np.testing.assert_allclose(scale, library.scale_)
    np.testing.assert_allclose(
        scratch.transform_standardization(holdout, mean, scale),
        library.transform(holdout),
    )
    assert scale[1] == 1


def test_quantile_thresholds_match_library_on_distinct_data():
    train = np.arange(9.0)
    thresholds = scratch.fit_quantile_thresholds(train, 4)
    np.testing.assert_array_equal(thresholds, [2, 4, 6])
    kwargs = dict(n_bins=4, encode="ordinal", strategy="quantile", subsample=None)
    if "quantile_method" in example.inspect.signature(KBinsDiscretizer).parameters:
        kwargs["quantile_method"] = "linear"
    library = KBinsDiscretizer(**kwargs).fit(train[:, None])
    values = np.array([-100, 2, 3, 4, 6, 100])
    np.testing.assert_array_equal(
        scratch.transform_bins(values, thresholds),
        library.transform(values[:, None]).ravel(),
    )


def test_ties_and_constant_bins_have_explicit_behavior():
    assert scratch.fit_quantile_thresholds([5, 5, 5]).size == 0
    np.testing.assert_array_equal(scratch.transform_bins([-9, 5, 99], []), [0, 0, 0])
    thresholds = scratch.fit_quantile_thresholds([0, 0, 0, 1, 1, 2, 2, 2], 8)
    assert np.all(np.diff(thresholds) > 0)
    assert np.all((thresholds > 0) & (thresholds < 2))


def test_product_only_returns_interaction():
    np.testing.assert_array_equal(scratch.pair_product([[2, 3], [-4, 5]]), [[6], [-20]])


@pytest.mark.parametrize("values", [[], [1, 2], [[np.nan]], [[np.inf]]])
def test_standardization_rejects_invalid_arrays(values):
    with pytest.raises(ValueError):
        scratch.fit_standardization(values)


@pytest.mark.parametrize("bins", [True, 1, 2.5])
def test_quantile_fit_rejects_invalid_bin_count(bins):
    with pytest.raises(ValueError):
        scratch.fit_quantile_thresholds([1, 2, 3], bins)


def test_transforms_reject_bad_parameters_and_overflow():
    with pytest.raises(ValueError):
        scratch.transform_standardization([[1, 2]], [1], [1])
    with pytest.raises(ValueError):
        scratch.transform_standardization([[1]], [1], [0])
    for thresholds in [[2, 1], [1, 1], [np.nan], [[1]]]:
        with pytest.raises(ValueError):
            scratch.transform_bins([0], thresholds)
    with pytest.raises(ValueError):
        scratch.pair_product([[1, 2, 3]])
    with pytest.raises(ValueError):
        scratch.pair_product([[1e308, 1e308]])


def boundary_data():
    snapshots = pd.DataFrame({
        "customer_id": [2, 1, 3],
        "prediction_time": ["2026-09-02", "2026-09-01", "2026-09-01"],
    }, index=[40, 10, 70])
    # ID 1: include the window start and a normal event; exclude old, cutoff,
    # future, late-arriving and exactly-at-cutoff availability events.
    events = pd.DataFrame({
        "customer_id": [1, 1, 1, 1, 1, 1, 1, 2],
        "event_time": ["2026-08-02", "2026-08-20", "2026-08-01", "2026-09-01",
                       "2026-09-02", "2026-08-21", "2026-08-22", "2026-09-01"],
        "available_at": ["2026-08-02", "2026-08-21", "2026-08-01", "2026-09-01",
                         "2026-09-02", "2026-09-03", "2026-09-01", "2026-09-01"],
        "amount": [10., 20., 1000., 1000., 1000., 1000., 1000., 40.],
    })
    return snapshots, events


def test_history_boundaries_row_order_and_absent_customer():
    snapshots, events = boundary_data()
    actual = example.aggregate_history(snapshots, events)
    expected = pd.DataFrame({
        "txn_count_30d": [1, 2, 0], "txn_mean_30d": [40., 15., 0.],
        "txn_sum_30d": [40., 30., 0.], "txn_max_30d": [40., 20., 0.],
    }, index=snapshots.index)
    pd.testing.assert_frame_equal(actual, expected)


def test_future_and_late_event_changes_cannot_change_history():
    snapshots, events = boundary_data()
    original = example.aggregate_history(snapshots, events)
    changed = events.copy()
    changed.loc[[2, 3, 4, 5, 6], "amount"] = 1e9
    pd.testing.assert_frame_equal(original, example.aggregate_history(snapshots, changed))
    changed["target"] = 1
    changed_snapshots = snapshots.assign(target=[0, 1, 0])
    pd.testing.assert_frame_equal(original, example.aggregate_history(changed_snapshots, changed))


def test_empty_event_table_returns_zero_history():
    snapshots, events = boundary_data()
    result = example.aggregate_history(snapshots, events.iloc[:0])
    assert (result.to_numpy() == 0).all()


@pytest.mark.parametrize("case", ["duplicate", "null_id", "null_time", "bad_amount",
                                 "early_availability", "missing_column", "bad_window"])
def test_history_rejects_invalid_inputs(case):
    snapshots, events = boundary_data()
    kwargs = {}
    if case == "duplicate":
        snapshots.loc[40, "customer_id"] = 1
    elif case == "null_id":
        events.loc[0, "customer_id"] = np.nan
    elif case == "null_time":
        events.loc[0, "event_time"] = None
    elif case == "bad_amount":
        events.loc[0, "amount"] = np.inf
    elif case == "early_availability":
        events.loc[0, "available_at"] = "2026-08-01"
    elif case == "missing_column":
        events = events.drop(columns="event_time")
    else:
        kwargs["window_days"] = 0
    with pytest.raises(ValueError):
        example.aggregate_history(snapshots, events, **kwargs)


def test_pipeline_holdout_changes_leave_fitted_parameters_unchanged():
    features, labels, _ = example.make_synthetic_data(160)
    train, holdout = features.iloc[:120], features.iloc[120:].copy()
    model = example.build_pipeline("engineered").fit(train, labels.iloc[:120])
    preprocessor = model.named_steps["features"]
    transforms = preprocessor.named_transformers_
    scaler = transforms["numeric"]
    bins = transforms["age_bins"]
    encoder = transforms["categorical"]
    interaction = transforms["interaction"]
    mean, scale = scaler.mean_.copy(), scaler.scale_.copy()
    edges = bins.bin_edges_[0].copy()
    categories = [values.copy() for values in encoder.categories_]
    product_mean = interaction.named_steps["scale"].mean_.copy()
    train_representation = preprocessor.transform(train).copy()
    np.testing.assert_allclose(mean, train[example.RAW_NUMERIC + example.AGGREGATES].mean())
    np.testing.assert_allclose(edges, np.quantile(train["age"], np.linspace(0, 1, 5)))
    holdout["income"] *= 100
    holdout["age"] = 100
    holdout["region"] = "unseen-region"
    holdout["channel"] = "unseen-channel"
    probabilities = model.predict_proba(holdout)
    assert np.isfinite(probabilities).all()
    np.testing.assert_array_equal(encoder.transform(holdout[example.CATEGORICAL]), 0)
    np.testing.assert_array_equal(scaler.mean_, mean)
    np.testing.assert_array_equal(scaler.scale_, scale)
    np.testing.assert_array_equal(bins.bin_edges_[0], edges)
    for old, new in zip(categories, encoder.categories_):
        np.testing.assert_array_equal(old, new)
    np.testing.assert_array_equal(interaction.named_steps["scale"].mean_, product_mean)
    np.testing.assert_array_equal(preprocessor.transform(train), train_representation)
    assert train_representation.shape[1] == 19
    product_input = interaction.named_steps["center"].transform(train[["income", "balance"]])
    np.testing.assert_allclose(
        interaction.named_steps["product"].transform(product_input),
        product_input[:, :1] * product_input[:, 1:],
    )
    # ID, prediction time and any accidentally supplied label are dropped.
    changed_train = train.assign(customer_id=-999, prediction_time="2099-01-01", target=1)
    np.testing.assert_array_equal(preprocessor.transform(changed_train), train_representation)


def test_cross_validation_fits_preprocessing_within_each_training_fold():
    features, labels, _ = example.make_synthetic_data(150)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train, validation in folds.split(features, labels):
        model = example.build_pipeline("engineered").fit(features.iloc[train], labels.iloc[train])
        transforms = model.named_steps["features"].named_transformers_
        np.testing.assert_allclose(
            transforms["numeric"].mean_,
            features.iloc[train][example.RAW_NUMERIC + example.AGGREGATES].mean(),
        )
        np.testing.assert_allclose(
            transforms["age_bins"].bin_edges_[0],
            np.quantile(features.iloc[train]["age"], np.linspace(0, 1, 5)),
        )
        assert np.isfinite(model.predict_proba(features.iloc[validation])).all()


def test_generator_is_deterministic_and_pipeline_variants_are_explicit():
    first = example.make_synthetic_data(100)
    second = example.make_synthetic_data(100)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_series_equal(first[1], second[1])
    pd.testing.assert_frame_equal(first[2], second[2])
    for variant, count in zip(example.VARIANTS, [10, 14, 19]):
        model = example.build_pipeline(variant).fit(first[0], first[1])
        assert model.named_steps["features"].transform(first[0]).shape[1] == count
    with pytest.raises(ValueError):
        example.build_pipeline("invalid")
    with pytest.raises(ValueError):
        example.make_synthetic_data(1)
