"""Numerical and Streamlit smoke checks for the Day 35 visual learning lab."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance as library_permutation
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from streamlit.testing.v1 import AppTest

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))
import visual_core as core
import visualizations as viz

SMALL_CONFIG = (400, 0.95, 1.6, 20, 4, 5)


@pytest.fixture(scope="module")
def lab():
    return core.build_lab(SMALL_CONFIG)


def test_gini_and_weighted_split_match_known_counts():
    assert core.gini([]) == 0
    assert core.gini([0, 0]) == 0
    assert core.gini([0, 1]) == 0.5
    np.testing.assert_allclose(core.gini([0, 1, 1, 1]), 0.375)
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    result = core.split_statistics(X, y, 0, 1.5)
    assert result["gain"] == 0.5
    np.testing.assert_array_equal(result["counts"], [[2, 2], [2, 0], [0, 2]])
    tree = DecisionTreeClassifier(max_depth=1, random_state=35).fit(X, y)
    assert tree.tree_.impurity[0] == result["parent"]
    assert core.split_statistics(X, y, 0, -10)["gain"] == 0


@pytest.mark.parametrize("labels", [[0, 2], [[0, 1]], [np.nan]])
def test_invalid_gini_rejected(labels):
    with pytest.raises(ValueError):
        core.gini(labels)


@pytest.mark.parametrize("config", [
    (200, .9, 1, 20, 4, 5), (400, 1, 1, 20, 4, 5),
    (400, .9, -1, 20, 4, 5), (400, .9, 1, 0, 4, 5),
    (400, .9, 1, 20, 99, 5), (400, .9, 1, 20, 4, 0),
    (400, np.nan, 1, 20, 4, 5),
])
def test_invalid_training_configuration(config):
    with pytest.raises(ValueError):
        core.build_lab(config)


def test_seeded_generation_training_split_and_probabilities(lab):
    repeated = core.build_lab(SMALL_CONFIG)
    np.testing.assert_array_equal(lab["X"], repeated["X"])
    np.testing.assert_array_equal(lab["y"], repeated["y"])
    assert not set(lab["train_ids"]) & set(lab["test_ids"])
    a = core.positive_probability(lab["model"], lab["X_test"])
    b = core.positive_probability(repeated["model"], repeated["X_test"])
    np.testing.assert_array_equal(a, b)
    assert np.all((0 <= a) & (a <= 1))
    assert np.corrcoef(lab["X"][:, :2].T)[0, 1] > 0.9


def test_metric_direction_and_invalid_predictions():
    y = np.array([0, 0, 1, 1])
    good, bad = np.array([.05, .1, .9, .95]), np.array([.95, .9, .1, .05])
    for metric in core.METRICS:
        assert core.metric_score(y, good, metric) > core.metric_score(y, bad, metric)
    for values in ([1.2, .1, .9, .9], [np.nan, .1, .9, .9]):
        with pytest.raises(ValueError):
            core.metric_score(y, values, "ROC-AUC")
    with pytest.raises(ValueError):
        core.metric_score(y, good, "accuracy")


def test_permutation_draw_and_shared_pair_preserve_relation(lab):
    X = lab["X_test"].copy()
    path = core.permutation_path(lab, (0, 1), repeat=2, frames=5)
    np.testing.assert_array_equal(path["states"][0], X)
    final = path["states"][-1]
    np.testing.assert_array_equal(final[:, :2], X[path["order"], :2])
    np.testing.assert_array_equal(final[:, 2:], X[:, 2:])
    np.testing.assert_array_equal(lab["X_test"], X)
    assert not np.allclose(path["states"][2][:, 0], final[:, 0])
    for p in path["probabilities"]:
        assert np.all((p >= 0) & (p <= 1))


def test_custom_permutation_mc_agreement_and_negative_values():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 2))
    y = X[:, 0] + .3*X[:, 1]
    model = LinearRegression().fit(X, y)
    own = core.permutation_importance(model, X, y, scoring="neg_mean_squared_error", repeats=300)
    library = library_permutation(model, X, y, scoring="neg_mean_squared_error", n_repeats=300, random_state=35)
    se = np.sqrt((own["importances_std"]**2 + library.importances_std**2)/300)
    assert np.all(abs(own["importances_mean"]-library.importances_mean) < 4*se)
    negative = core.permutation_importance(
        model, X, y, repeats=5,
        scoring=lambda model, rows, target: -float(np.mean(rows[:, 0] == X[:, 0]))
    )
    assert np.all(negative["importances"][0] < 0)


def test_pdp_average_ice_and_library_agreement(lab):
    response = core.response_curves(lab, rows=50, grid_size=11)
    np.testing.assert_array_equal(response["pdp"], response["ice"].mean(axis=0))
    reference = response["reference"]
    library = partial_dependence(lab["model"], reference, [0], method="brute", kind="both",
                                 response_method="predict_proba", grid_resolution=11)
    own = core.pdp_ice(lambda rows: core.positive_probability(lab["model"], rows),
                      reference, 0, library["grid_values"][0])
    np.testing.assert_allclose(own["pdp"], library["average"][0])
    np.testing.assert_allclose(own["ice"], library["individual"][0])
    centered = response["ice"] - response["ice"][:, :1]
    np.testing.assert_array_equal(centered[:, 0], 0)


def test_surface_matches_direct_average_and_fixed_prediction(lab):
    result = core.surfaces(lab, row=3, grid_size=5, reference_count=12)
    reference = lab["X_test"][:12].copy()
    reference[:, 0], reference[:, 2] = result["signal"][2], result["context"][1]
    expected = core.positive_probability(lab["model"], reference).mean()
    np.testing.assert_allclose(result["pdp"][1, 2], expected)
    row = lab["X_test"][3].copy()
    row[0], row[2] = result["signal"][2], result["context"][1]
    np.testing.assert_allclose(result["fixed"][1, 2], core.positive_probability(lab["model"], [row])[0])
    assert np.all((result["pdp"] >= 0) & (result["pdp"] <= 1))
    assert np.all((result["fixed"] >= 0) & (result["fixed"] <= 1))


def test_product_coalitions_orders_and_shapley_values():
    game = core.shapley_product(2, 3)
    assert list(game["coalitions"].values()) == [0, 0, 0, 6]
    np.testing.assert_array_equal(game["paths"][0]["contributions"], [0, 6])
    np.testing.assert_array_equal(game["paths"][1]["contributions"], [6, 0])
    np.testing.assert_array_equal(game["values"], [3, 3])
    assert game["baseline"] + game["values"].sum() == game["prediction"]
    negative = core.shapley_product(-2, 3)
    np.testing.assert_array_equal(negative["values"], [-3, -3])


def test_actual_background_sizes_above_default_cap_and_reconstruction(lab):
    first = core.forest_shap(lab, background_size=25, explained_count=10)
    large = core.forest_shap(lab, background_size=200, explained_count=10)
    assert len(large["background_ids"]) == 200
    np.testing.assert_array_equal(first["background_ids"], large["background_ids"][:25])
    assert not set(large["background_ids"]) & set(lab["test_ids"])
    for result in (first, large):
        np.testing.assert_allclose(result["baseline"] + result["values"].sum(axis=1),
                                   result["predictions"], rtol=0, atol=1e-6)
        assert result["error"] < 1e-6
    np.testing.assert_array_equal(first["predictions"], large["predictions"])


def test_support_band_uses_synthetic_conditional_distribution():
    rows = np.array([[0, 0, 0, 0], [0, 10, 0, 0]], dtype=float)
    np.testing.assert_array_equal(core.proxy_outside_band(rows, .95), [False, True])
    original = rows.copy()
    replaced = core.replacement_rows(rows, 0, 1)
    np.testing.assert_array_equal(rows, original)
    np.testing.assert_array_equal(replaced[:, 0], 1)


def test_cardinality_control_and_invalid_leakage_are_explicit(lab):
    control = core.cardinality_experiment()
    assert control["candidates"][0] > control["candidates"][1]
    assert len(control["pi"]) == 2
    invalid = core.leakage_experiment(lab)
    assert 0 <= invalid["invalid_auc"] <= 1
    assert len(invalid["invalid_mdi"]) == 5
    assert invalid["invalid_auc"] > invalid["clean_auc"]


def test_plotly_animation_fixed_axes_and_waterfall_units(lab):
    path = core.permutation_path(lab, (0,), frames=4)
    fig = viz.permutation_animation(lab, path, "Negative log loss", 0)
    assert len(fig.frames) == 4
    assert list(fig.layout.yaxis.range) == [-.04, 1.04]
    assert list(fig.layout.yaxis2.range) == [-.04, 1.04]
    assert list(fig.layout.xaxis.range) == list(fig.layout.xaxis2.range)
    assert all("xaxis" not in frame.layout.to_plotly_json() for frame in fig.frames)
    result = core.forest_shap(lab, 25, explained_count=5)
    waterfall = viz.waterfall_plot(result)
    assert list(waterfall.data[0].measure) == ["absolute", "relative", "relative", "relative", "relative", "total"]
    assert waterfall.layout.yaxis.title.text == "Positive-class probability units"


def test_streamlit_all_tabs_branches_and_cache_reuse(monkeypatch):
    fit_calls = []
    original_fit = core.RandomForestClassifier.fit

    def counted_fit(self, X, y, *args, **kwargs):
        fit_calls.append(len(X))
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(core.RandomForestClassifier, "fit", counted_fit)
    app = AppTest.from_file(str(TOPIC/"app.py"), default_timeout=90).run()
    assert not app.exception
    app.session_state["lab_config"] = (400, .95, 1.6, 20, 4, 5)
    app.run()
    assert not app.exception
    tab_names = [
        "Tree Importance / MDI", "Permutation Importance", "PDP & 3D Surfaces", "ICE",
        "SHAP", "Explainability Pitfalls", "Summary & Interview Review"
    ]
    for name in tab_names:
        app.session_state["lab_tab"] = name
        app.run()
        assert not app.exception, (name, [x.value for x in app.exception])
    # Exercise hidden branches in the actual lazy-tab application.
    cases = [
        ("Tree Importance / MDI", {"cardinality_toggle": True}),
        ("Permutation Importance", {"permutation_metric": "Negative log loss", "joint_shuffle": True}),
        ("PDP & 3D Surfaces", {"pdp_mode": "3D comparison", "surface_kind": "fixed", "interaction_compare": True}),
        ("ICE", {"centered": True}),
        ("SHAP", {"shap_view": "Waterfall"}),
        ("SHAP", {"shap_view": "Beeswarm"}),
        ("SHAP", {"shap_view": "Dependence"}),
        ("SHAP", {"shap_view": "Background sensitivity"}),
        ("Explainability Pitfalls", {"pitfall": "Unrealistic feature combinations"}),
        ("Explainability Pitfalls", {"pitfall": "Data leakage (deliberately invalid)", "run_leakage": True}),
    ]
    for name, state in cases:
        app.session_state["lab_tab"] = name
        for key, value in state.items():
            app.session_state[key] = value
        app.run()
        assert not app.exception, (name, state, [x.value for x in app.exception])
    # Display changes leave the active training configuration alone.
    active = app.session_state["lab_config"]
    before_fits = len(fit_calls)
    app.slider(key="observation").set_value(4).run()
    assert len(fit_calls) == before_fits
    assert app.session_state["lab_config"] == active
    assert not app.exception
    # Smallest data setting and the largest permitted background remain bounded.
    app.slider(key="observation").set_value(79).run()
    app.session_state["background_size"] = 300
    app.session_state["other_background_size"] = 300
    app.session_state["shap_config"] = (300, 35)
    app.session_state["lab_config"] = (300, .95, 1.6, 20, 4, 5)
    app.session_state["lab_tab"] = "SHAP"
    app.session_state["shap_view"] = "Waterfall"
    app.run()
    assert not app.exception
    assert app.slider(key="observation").value <= 74
    assert app.session_state["shap_config"][0] <= 225
