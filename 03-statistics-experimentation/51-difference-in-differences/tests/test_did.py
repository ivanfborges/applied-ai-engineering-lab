"""Hand contrasts, balanced-panel equivalence and identification failure cases."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example import difference_in_differences, fit_twfe, manual_did, simulate_panel, validate_panel


def test_hand_contrast_allows_different_baselines():
    assert difference_in_differences(80, 105, 100, 110) == 15
    assert difference_in_differences(80, 90, 100, 110) == 0
    assert difference_in_differences(20, 15, 40, 38) == -3


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, [1, 2]])
def test_invalid_cell_means(value):
    with pytest.raises(ValueError):
        difference_in_differences(value, 1, 2, 3)


@pytest.mark.parametrize("seed", [0, 51, 123])
def test_regression_and_first_differences_agree(seed):
    panel = simulate_panel(seed=seed, n_units=20)
    model = fit_twfe(panel)
    estimate = manual_did(panel)
    assert model.params["treated:post"] == pytest.approx(estimate, abs=1e-10)
    means = panel.assign(post=panel.period >= 0).groupby(["unit", "post"]).outcome.mean().unstack()
    changes = means[True] - means[False]
    assert changes.iloc[:10].mean() - changes.iloc[10:].mean() == pytest.approx(estimate)
    critical = t.ppf(0.975, 19)
    np.testing.assert_allclose(model.conf_int().loc["treated:post"],
                               estimate + np.array([-1, 1]) * critical * model.bse["treated:post"])
    assert model.cov_type == "cluster"
    assert model.df_resid_inference == 19


def test_invariance_to_unit_levels_and_common_period_shocks():
    panel = simulate_panel(n_units=20)
    changed = panel.copy()
    changed["outcome"] += 17 * changed.unit + 5 * changed.period**2
    assert manual_did(changed) == pytest.approx(manual_did(panel))
    assert manual_did(panel.sample(frac=1, random_state=51)) == pytest.approx(manual_did(panel))


def test_known_effect_and_nonparallel_trend_shift():
    panel = simulate_panel(n_units=20, effect=0)
    treated = simulate_panel(n_units=20, effect=4)
    violation = simulate_panel(n_units=20, effect=4, differential_trend=0.8)
    assert manual_did(treated) - manual_did(panel) == pytest.approx(4)
    assert manual_did(violation) - manual_did(treated) == pytest.approx(2.4)
    pre = treated.loc[treated.period < 0]
    pre_violation = violation.loc[violation.period < 0]
    assert manual_did(pre_violation, -1) - manual_did(pre, -1) == pytest.approx(1.2)
    pd.testing.assert_frame_equal(pre, panel.loc[panel.period < 0])


@pytest.mark.parametrize("fault", ["empty", "missing_column", "missing_value", "infinite", "text",
                                    "duplicate", "unbalanced", "one_group", "nonbinary",
                                    "switching", "fractional_period", "no_pre", "no_post"])
def test_panel_rejects_ambiguous_input(fault):
    panel = simulate_panel(n_units=8)
    cutoff = 0
    if fault == "empty": panel = panel.iloc[:0]
    elif fault == "missing_column": panel = panel.drop(columns="outcome")
    elif fault == "missing_value": panel.loc[0, "unit"] = np.nan
    elif fault == "infinite": panel.loc[0, "outcome"] = np.inf
    elif fault == "text": panel["outcome"] = "invalid"
    elif fault == "duplicate": panel = pd.concat([panel, panel.iloc[:1]])
    elif fault == "unbalanced": panel = panel.iloc[1:]
    elif fault == "one_group": panel["treated"] = 1
    elif fault == "nonbinary": panel.loc[0, "treated"] = 2
    elif fault == "switching": panel.loc[0, "treated"] = 0
    elif fault == "fractional_period": panel["period"] = panel.period + 0.5
    elif fault == "no_pre": cutoff = -3
    elif fault == "no_post": cutoff = 3
    with pytest.raises(ValueError):
        validate_panel(panel, cutoff)


@pytest.mark.parametrize("cutoff", [np.nan, np.inf, 0.5])
def test_invalid_cutoff(cutoff):
    with pytest.raises(ValueError): validate_panel(simulate_panel(n_units=8), cutoff)


@pytest.mark.parametrize("kwargs", [{"n_units": 3}, {"n_units": 5}, {"n_units": 8.0},
                                     {"n_units": True}, {"seed": -1}, {"seed": 2.5},
                                     {"effect": np.inf}, {"differential_trend": np.nan}])
def test_invalid_simulation_config(kwargs):
    with pytest.raises(ValueError): simulate_panel(**kwargs)


def test_panel_validation_does_not_mutate_input_and_ignores_stale_post():
    panel = simulate_panel(n_units=8)
    panel["post"] = 99
    original = panel.copy(deep=True)
    validated = validate_panel(panel)
    pd.testing.assert_frame_equal(panel, original)
    assert set(validated.post) == {0, 1}
    pd.testing.assert_frame_equal(simulate_panel(n_units=8), simulate_panel(n_units=8))


def test_inference_rejects_single_treated_cluster():
    panel = simulate_panel(n_units=4)
    panel["treated"] = (panel.unit == 0).astype(int)
    with pytest.raises(ValueError, match="at least two"):
        fit_twfe(panel)
