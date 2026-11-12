"""Numerical contracts and small temporary-output rendering checks."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import gain_curve, ipw_effect
from uplift_visual_core import (
    assignment_probability, build_payload, newcombe_difference, policy_selection,
    policy_summary, randomized_data, response_surfaces, sample_size_study, wilson,
)
from visualize_uplift import write_dashboard


def test_surface_identity_and_probability_bounds():
    axis = np.linspace(-2.5, 2.5, 31)
    a, b = np.meshgrid(axis, axis)
    x = np.column_stack((a.ravel(), b.ravel()))
    for strength in (0, .5, 1, 2):
        p0, p1, tau = response_surfaces(x, strength)
        assert np.all((0 < p0) & (p0 < 1) & (0 < p1) & (p1 < 1))
        np.testing.assert_allclose(tau, p1 - p0, atol=0, rtol=0)
        if strength == 0:
            np.testing.assert_array_equal(tau, 0)
        else:
            assert tau.min() < 0 < tau.max()
    point = np.array([[1, -2]])
    l0 = -.3 + .85 + 1.3 + .25 * np.sin(-2)
    shift = .9 * np.tanh(-2) - .7 * np.tanh(1) + .5 * np.sin(-2)
    p0, p1, _ = response_surfaces(point)
    np.testing.assert_allclose([p0[0], p1[0]], [expit(l0), expit(l0 + shift)])


@pytest.mark.parametrize("x,strength", [([[1]], 1), ([[np.nan, 0]], 1), ([[0, 1]], -1),
                                           ([[0, 1]], np.inf), ([[0, 1]], 3)])
def test_invalid_surface_inputs(x, strength):
    with pytest.raises(ValueError):
        response_surfaces(x, strength)


@pytest.mark.parametrize("regime", ["randomized", "observational", "poor_overlap"])
def test_assignment_and_data_reproducibility(regime):
    first, second = randomized_data(500, 52, regime), randomized_data(500, 52, regime)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert np.all((first["e"] > 0) & (first["e"] < 1))
    assert set(np.unique(first["y"])) == {0, 1}
    assert set(np.unique(first["t"])) == {0, 1}
    if regime == "randomized":
        np.testing.assert_array_equal(first["e"], .5)
    else:
        assert first["e"].std() > .1


def test_ipw_and_gain_exact_arithmetic():
    z = ipw_effect([1, 1, 0, 0], [1, 0, 1, 0], [.25, .75, .5, .5])
    np.testing.assert_allclose(z, [4, -4, 0, 0])
    for score in ([4, 3, 2, 1], [1, 1, 1, 1], [2, 4, 3, 1]):
        q, gains = gain_curve(score, z)
        assert q[0] == 0 and q[-1] == 1
        assert gains[0] == 0 and gains[-1] == pytest.approx(np.mean(z))
    # Averaging over the actual assignment mechanism cancels its probabilities.
    e, p0, p1 = .2, .3, .7
    z = ipw_effect([1, 0, 1, 0], [1, 1, 0, 0], e)
    probabilities = [e * p1, e * (1-p1), (1-e) * p0, (1-e) * (1-p0)]
    assert np.dot(z, probabilities) == pytest.approx(p1-p0)


def test_wilson_boundaries_and_newcombe_symmetry():
    assert wilson(0, 10)[0] == pytest.approx(0, abs=1e-15)
    assert wilson(10, 10)[1] == pytest.approx(1, abs=1e-15)
    np.testing.assert_allclose(wilson(5, 10), [.236593090512564, .763406909487436])
    d, low, high = newcombe_difference(7, 10, 3, 10)
    d2, low2, high2 = newcombe_difference(3, 10, 7, 10)
    assert d == pytest.approx(.4)
    np.testing.assert_allclose([d2, low2, high2], [-d, -high, -low])
    assert -1 <= low < d < high <= 1


@pytest.mark.parametrize("args", [(0,0), (11,10), (-1,10)])
def test_invalid_interval_counts(args):
    with pytest.raises(ValueError):
        wilson(*args)


def test_sample_intervals_are_finite_and_use_actual_subgroup_counts():
    study = sample_size_study(repetitions=2)
    for size, records in study.items():
        assert len(records) == 2
        for row in records:
            assert sum(r["n0"] + r["n1"] for r in row) == int(size)
            for r in row:
                assert r["low"] <= r["estimate"] <= r["high"]
                assert r["n0"] > 0 and r["n1"] > 0


def test_policy_cost_capacity_and_equal_score_ties():
    scores = np.array([.2, .1, .05, -.1])
    np.testing.assert_array_equal(policy_selection(scores, 100, 5, 100), [1, 1, 0, 0])
    assert not policy_selection(scores, 100, 100, 100).any()
    assert not policy_selection(scores, 0, 0, 100).any()
    assert not policy_selection(scores, 100, 0, 0).any()
    np.testing.assert_array_equal(policy_selection([1, 1, 1, 0], 1, 0, 50), [1, 1, 0, 0])
    assert policy_selection(np.ones(100), 1, 0, 58).sum() == 58
    data = dict(score=scores, tau=np.array([.3, .2, .1, -.2]), z=np.array([2, -2, 0, 0]))
    result = policy_summary(data, 100, 5, 100)
    assert result["selected"] == 2
    assert result["contact_cost"] == 10
    assert result["true_conversions"] == pytest.approx(.5)
    assert result["true_net"] == pytest.approx(40)
    assert result["ipw_net"] == pytest.approx(-10)


@pytest.mark.parametrize("capacity", [-1, 101, np.nan, [30]])
def test_invalid_policy_capacity(capacity):
    with pytest.raises(ValueError):
        policy_selection([.1, .2], capacity_percent=capacity)


@pytest.fixture(scope="module")
def payload():
    return build_payload(n_train=300, n_test=200, repetitions=2, grid_size=9)


def test_shared_models_and_independent_holdout(payload):
    assert payload["config"]["n_train"] == 300 and payload["config"]["n_test"] == 200
    assert len(payload["test"]["z"]) == 200
    grid = payload["grid"]
    np.testing.assert_allclose(grid["t"], grid["t1"] - grid["t0"])
    np.testing.assert_allclose(grid["s"], grid["s1"] - grid["s0"])
    np.testing.assert_allclose(grid["tau"], grid["p1"] - grid["p0"])
    for curve in payload["curves"].values():
        assert curve["gain"][-1] == pytest.approx(payload["metrics"]["ipw_ate"])
    assert payload["metrics"]["s_rmse"] >= 0 and payload["metrics"]["t_rmse"] >= 0


def test_offline_html_contains_all_modules_and_no_external_resources(payload, tmp_path):
    from html.parser import HTMLParser

    class Resources(HTMLParser):
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            for key in ("src", "href"):
                if key in attrs:
                    assert not attrs[key].startswith(("http:", "https:", "//"))

    index = write_dashboard(payload, tmp_path)
    text = index.read_text(encoding="utf-8")
    Resources().feed(text)
    unfilled = [marker for marker in ("__DATA__", "__PLOTLY__", "__LAB_JS__") if marker in text]
    assert not unfilled
    assert text.count('class="module') == 9
    assert "Plotly.react" in text
    assert "connect-src 'none'" in text


def test_animation_failure_has_static_fallback(tmp_path, monkeypatch):
    from uplift_animations import export_animation, plt, FuncAnimation

    def fail(*args, **kwargs):
        raise OSError("simulated writer failure")

    monkeypatch.setattr(FuncAnimation, "save", fail)
    fig, ax = plt.subplots(figsize=(2, 2))
    with pytest.warns(RuntimeWarning, match="wrote"):
        result = export_animation(fig, lambda frame: ax.set_title(str(frame)), tmp_path / "fallback.gif", frames=4)
    assert result["kind"] == "png_fallback"
    assert (tmp_path / "fallback.png").is_file()
