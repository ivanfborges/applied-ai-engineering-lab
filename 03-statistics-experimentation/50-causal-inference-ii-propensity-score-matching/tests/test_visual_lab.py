"""Numerical contracts, real widget events and bounded Pillow export checks."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual_core import (
    Design, analyze_teams, assignment, population_truth, record_analysis,
    repeated_study, repeated_summary, separation_sweep, simulate_teams,
)
from visualizations import (
    draw_balance, draw_confounding, draw_estimators, draw_extremes, draw_hidden,
    draw_ipw, draw_matching, draw_overlap, draw_surface, make_dashboard, save_chapter, style,
)


def test_potential_outcomes_sign_units_and_reproducibility():
    design = Design(heterogeneity=3)
    a, b = simulate_teams(600, 50, design), simulate_teams(600, 50, design)
    for key in ("x", "z", "t", "y0", "y1", "y", "true_scores", "hidden_u"):
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_allclose(a["y1"]-a["y0"], -8-3*a["z"][:, 0])
    np.testing.assert_allclose(a["y"], np.where(a["t"] == 1, a["y1"], a["y0"]))
    assert np.all(a["tau"] < 0) and np.all(a["y"] > 0)
    assert np.all((a["true_scores"] > 0) & (a["true_scores"] < 1))
    assert min(np.bincount(a["t"])) >= 2


def test_paired_separation_sweep_changes_only_assignment():
    baseline = simulate_teams(300, 50, Design(separation=0))
    poor = simulate_teams(300, 50, Design(separation=4))
    for key in ("x", "y0", "y1", "tau", "hidden_u"):
        np.testing.assert_array_equal(baseline[key], poor[key])
    assert np.any(baseline["t"] != poor["t"])
    assert np.ptp(poor["true_scores"]) > np.ptp(baseline["true_scores"])


def test_population_targets_are_distinct_under_heterogeneity_and_integral_converges():
    for h in (0, 3):
        design = Design(heterogeneity=h)
        low, high = population_truth(design, 24), population_truth(design, 48)
        assert high["ate"] == pytest.approx(-8)
        assert high["att"] == pytest.approx(low["att"], abs=1e-10)
        if h == 0:
            assert high["att"] == pytest.approx(-8)
        else:
            assert high["att"] < high["ate"]-1


def test_matching_indices_replacement_and_weighted_contrast():
    result = analyze_teams(simulate_teams(600, 50))
    data, pairs = result["data"], result["full_matches"]
    assert np.all(data["t"][pairs.control] == 0)
    assert np.all(data["t"][pairs.treated] == 1)
    assert len(np.unique(pairs.control)) < len(pairs.control)
    by_pairs = np.mean(data["y"][pairs.treated]-data["y"][pairs.control])
    assert result["summaries"]["Matching ATT"]["estimate"] == pytest.approx(by_pairs)
    assert np.all(result["matches"].distance <= .2)
    assert result["matches"].treated.size+result["matches"].unmatched.size == data["t"].sum()


def test_weights_ess_balance_and_att_target():
    result = analyze_teams(simulate_teams(600, 50, Design(heterogeneity=3)))
    t, p = result["data"]["t"], result["scores"]
    expected = np.array([1/prob if arm else 1/(1-prob) for arm, prob in zip(t, p)])
    np.testing.assert_allclose(result["weights"]["IPW ATE"], expected)
    expected_att = np.array([1 if arm else prob/(1-prob) for arm, prob in zip(t, p)])
    np.testing.assert_allclose(result["weights"]["IPW ATT"], expected_att)
    for summary in result["summaries"].values():
        assert np.all(summary["ess"] > 0)
        assert np.all(summary["ess"] <= np.bincount(t)+1e-10)
    assert result["retained_truth"] == pytest.approx(result["data"]["tau"][result["matches"].treated].mean())


def test_zero_caliper_explicitly_exposes_empty_restricted_population():
    result = analyze_teams(simulate_teams(200, 50), caliper=0)
    assert not result["matches"].treated.size
    assert "Caliper ATT" not in result["summaries"]
    assert result["retained_truth"] is None


def test_structural_nonpositivity_is_not_clipped_or_fitted_away():
    data = simulate_teams(600, 50, Design(structural=True))
    selected = data["x"][:, 0] > 7.5
    assert selected.any()
    assert np.all(data["true_scores"][selected] == 1)
    assert np.all(data["t"][selected] == 1)
    with pytest.raises(ValueError, match="Structural nonpositivity"):
        analyze_teams(data)


def test_hidden_variable_is_withheld_from_assignment_fit():
    data = simulate_teams(600, 50, Design(hidden=1.5))
    result = analyze_teams(data)
    assert result["model"].n_features_in_ == 2
    actual = assignment(data["z"], data["design"], data["hidden_u"])
    np.testing.assert_allclose(actual, data["true_scores"])
    assert np.max(np.abs(actual-assignment(data["z"], data["design"]))) > .1


def test_independent_repetitions_targets_and_summary_arithmetic():
    rows = repeated_study(150, 50, 4)
    assert rows == repeated_study(150, 50, 4)
    assert len({row["seed"] for row in rows}) == 8
    assert len(rows) == 24
    for row in rows:
        assert row["error"] == pytest.approx(row["estimate"]-row["target"])
        if row["scenario"] == "Heterogeneous effect" and row["method"] == "Matching ATT":
            assert row["target"] < -9
    summary = repeated_summary(rows)[0]
    errors = [row["error"] for row in rows if row["scenario"] == summary["scenario"] and row["method"] == summary["method"]]
    assert summary["mean_error"] == pytest.approx(np.mean(errors))
    assert summary["mcse_mean_error"] == pytest.approx(np.std(errors, ddof=1)/2)
    assert summary["rmse"] == pytest.approx(np.sqrt(np.mean(np.square(errors))))


@pytest.mark.parametrize("kwargs", [{"separation": -1}, {"separation": 5}, {"effect": 1},
                                     {"effect": -15}, {"hidden": -1}, {"heterogeneity": 4},
                                     {"separation": np.nan}])
def test_invalid_design(kwargs):
    with pytest.raises(ValueError):
        Design(**kwargs)


def test_other_invalid_configuration():
    for n in (0, 1, 19, 25.5, True):
        with pytest.raises(ValueError):
            simulate_teams(n)
    with pytest.raises(ValueError):
        repeated_study(repetitions=1)
    with pytest.raises(ValueError):
        separation_sweep(frames=2)
    with pytest.raises(ValueError):
        population_truth(Design(), order=2)


def test_record_is_json_finite():
    import json
    result = analyze_teams(simulate_teams(100, 50))
    payload = json.dumps(record_analysis(result), allow_nan=False)
    assert "sample_att" in payload


def test_all_chapter_draw_functions_and_no_open_figure_leaks():
    style()
    result = analyze_teams(simulate_teams(120, 50))
    structural = simulate_teams(120, 50, Design(structural=True))
    hidden = analyze_teams(simulate_teams(120, 50, Design(hidden=1.5)))
    rows = repeated_study(100, 50, 3)
    sweep = separation_sweep(120, 50, 3)
    draws = [lambda f: draw_confounding(f, result), lambda f: draw_surface(f, result),
             lambda f: draw_overlap(f, result, structural), lambda f: draw_matching(f, result, 1),
             lambda f: draw_ipw(f, result, .5), lambda f: draw_ipw(f, result, 1, "ATT"),
             lambda f: draw_balance(f, result), lambda f: draw_extremes(f, sweep, 2),
             lambda f: draw_estimators(f, rows), lambda f: draw_hidden(f, hidden)]
    for draw in draws:
        fig = plt.figure(figsize=(12.8, 8))
        try:
            draw(fig)
            fig.canvas.draw()
            assert fig.axes
            if "09  Observed balance" in fig._suptitle.get_text():
                renderer = fig.canvas.get_renderer()
                assert all(label.get_window_extent(renderer).x0 >= 0
                           for label in fig.axes[0].get_yticklabels())
            if "03  " in fig._suptitle.get_text():
                renderer = fig.canvas.get_renderer()
                legend = fig.axes[0].get_legend().get_window_extent(renderer)
                assert legend.y0 >= fig.axes[0].get_window_extent(renderer).y0
        finally:
            plt.close(fig)
    assert not plt.get_fignums()


def test_pillow_png_and_animated_gif_export(tmp_path):
    result = analyze_teams(simulate_teams(80, 50))
    files = save_chapter(tmp_path, "smoke", lambda fig, fraction: draw_ipw(fig, result, fraction), [0, 1], dpi=45)
    assert len(files) == 2
    with Image.open(files[0]) as png:
        assert png.format == "PNG" and png.size[0] > 1000
    with Image.open(files[1]) as gif:
        assert gif.format == "GIF" and gif.n_frames == 2
        first = np.asarray(gif.convert("RGB")).copy()
        gif.seek(1)
        assert not np.array_equal(first, np.asarray(gif.convert("RGB")))
    assert not plt.get_fignums()


def mouse_click(fig, axes, x_fraction=.5):
    x, y = axes.transAxes.transform((x_fraction, .5))
    for kind in ("button_press_event", "button_release_event"):
        fig.canvas.callbacks.process(kind, MouseEvent(kind, fig.canvas, x, y, button=1))


def test_dashboard_real_mouse_slider_checkbutton_reset_and_3d():
    fig, controls, toggle, reset, state = make_dashboard(150, 50)
    try:
        fig.canvas.draw()
        old = state["result"]["scores"].copy()
        mouse_click(fig, controls["Separation"].ax, .8)
        assert controls["Separation"].val == pytest.approx(3.2, abs=.05)
        assert not np.array_equal(old, state["result"]["scores"])
        toggle.set_active(0)
        assert state["result"]["data"]["design"].heterogeneity == 3
        controls["Caliper"].set_val(0)
        assert not state["result"]["matches"].treated.size
        fig.axes[0].view_init(elev=35, azim=20)
        fig.canvas.draw()
        mouse_click(fig, reset.ax)
        assert controls["Separation"].val == 1
        assert not toggle.get_status()[0]
        assert state["updates"] >= 6
    finally:
        plt.close(fig)
