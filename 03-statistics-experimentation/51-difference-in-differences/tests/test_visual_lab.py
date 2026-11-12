"""Independent numerical checks for the 120-team visual lab and its artifacts."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from charts import event_plot, geometry, plot_3d, trend_chart
from estimation import (change_interval, event_study, oracle_decomposition, placebo,
                        regression_summary, sampling_distribution, trajectories)
from example import manual_did
from export_gifs import NAMES, export_all
from simulation import adoption_schedule, expected_bias, simulate


def test_default_panel_and_potential_outcome_consistency():
    panel = simulate()
    assert panel.shape[0] == 1440
    assert panel.unit.nunique() == 120
    assert panel.period.nunique() == 12
    assert panel.drop_duplicates("unit").treated.value_counts().to_dict() == {1: 60, 0: 60}
    assert not panel.duplicated(["unit", "period"]).any()
    np.testing.assert_array_equal(panel.exposed, panel.treated * (panel.period >= 0))
    np.testing.assert_allclose(panel.outcome, np.where(panel.treated, panel.y1, panel.y0))
    np.testing.assert_allclose(panel.y1-panel.y0, 4*(panel.period >= 0))
    pd.testing.assert_frame_equal(panel, simulate())


@pytest.mark.parametrize("seed", [0, 51, 99])
def test_noise_free_parallel_changes_and_twfe(seed):
    panel = simulate(seed=seed, noise=0, baseline_gap=13, n_teams=12)
    means = panel.groupby(["period", "treated"]).mean_y0.mean().unstack()
    np.testing.assert_allclose(means[1].diff().dropna(), means[0].diff().dropna(), atol=1e-12)
    summary = regression_summary(panel)
    assert summary["manual"] == pytest.approx(4)
    assert summary["twfe"] == pytest.approx(4)


@pytest.mark.parametrize("config,bias", [({"slope": 0.8}, 4.8), ({"slope": -1.2}, -7.2),
                                        ({"spillover": 0.5}, -2), ({"anticipation": 3}, -1),
                                        ({"slope": 0.2, "spillover": 0.5, "anticipation": 3}, -1.8)])
def test_assumption_violation_components(config, bias):
    panel = simulate(noise=0, **config)
    values = oracle_decomposition(panel)
    assert values["truth"] == pytest.approx(4)
    assert values["identification_bias"] == pytest.approx(bias)
    assert values["estimate"] == pytest.approx(4+bias)
    assert expected_bias(**config) == pytest.approx(bias)
    assert values["noise_error"] == pytest.approx(0, abs=1e-12)


def test_spillover_and_anticipation_only_affect_defined_observations():
    base = simulate()
    spill = simulate(spillover=0.5)
    early = simulate(anticipation=3)
    np.testing.assert_allclose(spill.outcome-base.outcome, 2*(base.treated == 0)*(base.period >= 0))
    np.testing.assert_allclose(early.outcome-base.outcome, 3*(base.treated == 1)*base.period.isin([-2, -1]))
    pd.testing.assert_series_equal(base.y0, spill.y0)
    pd.testing.assert_series_equal(base.y0, early.y0)


@pytest.mark.parametrize("config", [{}, {"slope": 0.4}, {"spillover": 0.7}, {"anticipation": 2}])
def test_fast_cluster_covariance_matches_actual_statsmodels(config):
    panel = simulate(n_teams=16, **config)
    full = regression_summary(panel)
    fast = change_interval(panel)
    for key, other in [("estimate", "twfe"), ("se", "se"), ("low", "low"), ("high", "high")]:
        assert fast[key] == pytest.approx(full[other], abs=1e-10)
    assert full["df"] == 15


def test_synthetic_error_decomposition_and_counterfactual_anchor():
    panel = simulate(slope=0.3, spillover=0.2, anticipation=1.5)
    values = oracle_decomposition(panel)
    assert values["error"] == pytest.approx(values["identification_bias"]+values["noise_error"])
    data = trajectories(panel)
    assert data.loc[0:, "estimated_gap"].mean() == pytest.approx(manual_did(panel))
    shifted = trajectories(simulate(baseline_gap=-15))
    assert shifted.loc[0:, "estimated_gap"].mean() == pytest.approx(manual_did(simulate()))


def test_event_study_matches_saturated_statsmodels_points_and_covariance():
    panel = simulate(n_teams=12, slope=0.3)
    events = event_study(panel)
    terms = []
    periods = [period for period in sorted(panel.period.unique()) if period != -1]
    for index, period in enumerate(periods):
        name = f"event_{index}"
        panel[name] = panel.treated * (panel.period == period)
        terms.append(name)
    model = smf.ols("outcome ~ C(unit) + C(period) + " + " + ".join(terms), data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel.unit, "use_correction": True, "df_correction": True}, use_t=True)
    for name, period in zip(terms, periods):
        row = events.set_index("period").loc[period]
        assert row.estimate == pytest.approx(model.params[name], abs=1e-10)
        assert row.se == pytest.approx(model.bse[name], abs=1e-10)
        np.testing.assert_allclose([row.low, row.high], model.conf_int().loc[name], atol=1e-10)
    reference = events.loc[events.reference].iloc[0]
    assert reference.period == -1
    assert reference.estimate == reference.se == 0


def test_placebo_excludes_every_post_observation():
    panel = simulate()
    altered = panel.copy()
    altered.loc[altered.period >= 0, "outcome"] += 10000
    assert placebo(panel) == placebo(altered)
    assert placebo(panel)["max_period_used"] == -1
    assert placebo(simulate(noise=0))["estimate"] == pytest.approx(0, abs=1e-12)
    assert placebo(simulate(noise=0, slope=0.8))["estimate"] == pytest.approx(2.4)
    with pytest.raises(ValueError):
        placebo(panel, 0)
    with pytest.raises(ValueError):
        event_study(panel, reference=0)


def test_independent_sampling_distribution_has_expected_scale_and_target():
    draws = sampling_distribution(draws=40, noise=2, n_teams=20)
    paired = sampling_distribution(draws=40, noise=4, n_teams=20)
    shifted = sampling_distribution(draws=40, noise=2, n_teams=20, slope=0.5)
    assert draws.estimate.nunique() == 40
    assert draws.estimate.std(ddof=1) > 0
    np.testing.assert_allclose(paired.estimate-4, 2*(draws.estimate-4), atol=1e-12)
    np.testing.assert_allclose(paired.se, 2*draws.se, atol=1e-12)
    np.testing.assert_allclose(shifted.estimate-draws.estimate, 3, atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"noise": -1}, {"spillover": -0.1}, {"spillover": 1.1},
                                     {"anticipation": -1}, {"slope": np.inf}, {"seed": -1},
                                     {"n_teams": 3}, {"n_teams": 12.0}])
def test_invalid_simulation_inputs(kwargs):
    with pytest.raises(ValueError):
        simulate(**kwargs)


def test_figures_preserve_statistical_encodings():
    panel = simulate()
    data = trajectories(panel)
    fig = trend_chart(panel, True)
    assert len(fig.data) == 5
    np.testing.assert_allclose(fig.data[-1].y, [data.loc[5, "estimated_cf"], data.loc[5, "treated"]])
    animation = geometry()
    assert len(animation.frames) == 6
    np.testing.assert_allclose(animation.frames[-1].data[-1].y, [110, 125])
    three_d = plot_3d(panel, 3, True)
    np.testing.assert_array_equal(three_d.data[4].x, [3, 3])
    np.testing.assert_array_equal(three_d.data[4].y, [1, 1])
    assert three_d.layout.scene.zaxis.range[0] > 30
    assert event_plot(event_study(panel)).data[-1].name.startswith("Reference -1")
    assert adoption_schedule().loc["Early adopters", 0] == 1
    assert adoption_schedule().loc["Late adopters", 0] == 0
    assert adoption_schedule().loc["Late adopters", 3] == 1
    assert adoption_schedule().loc["Never treated"].sum() == 0


def test_two_gifs_export_in_temporary_directory(tmp_path):
    result = export_all(tmp_path, frames=12, fps=3, width=640)
    for name in NAMES:
        with Image.open(tmp_path / name) as gif:
            assert gif.is_animated
            assert gif.n_frames >= 6
            assert gif.size == (640, 400)
            for frame in range(gif.n_frames):
                gif.seek(frame)
                gif.load()
    assert result["violation"]["end"]["estimate"] == pytest.approx(13)
    assert result["interpretation_status"] == "author review required"
    with pytest.raises(FileExistsError):
        export_all(tmp_path, frames=12, fps=3, width=640)
