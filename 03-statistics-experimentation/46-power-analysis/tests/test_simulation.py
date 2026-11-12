"""Deterministic inference agreement and uncertainty-aware Monte Carlo checks."""
from math import sqrt
from pathlib import Path
import sys
import numpy as np
import pytest
from statsmodels.stats.proportion import proportions_ztest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation import pooled_z_test, simulate_experiments, synthetic_intervals


@pytest.mark.parametrize("xc,xt", [(100, 110), (30, 10), (0, 20), (500, 600)])
def test_pooled_z_pvalues_match_statsmodels(xc, xt):
    result = pooled_z_test(xc, xt, 1000, 1000)
    _, p = proportions_ztest([xt, xc], [1000, 1000], alternative="two-sided", prop_var=False)
    assert result["p_value"] == pytest.approx(p)
    assert result["effect"] == pytest.approx((xt - xc) / 1000)


def test_vectorization_and_degenerate_convention():
    result = pooled_z_test(np.array([0, 100, 1000]), np.array([0, 110, 1000]), 1000, 1000)
    assert result["degenerate"].tolist() == [True, False, True]
    assert result["p_value"][[0, 2]].tolist() == [1, 1]
    assert not result["reject"][[0, 2]].any()


@pytest.mark.parametrize("mode", ["null", "alternative"])
def test_rejection_agrees_with_planning_within_uncertainty(mode):
    result = simulate_experiments(n=14751, repetitions=20000, seed=46, mode=mode)
    expected = result["analytical"]
    mcse = sqrt(expected * (1 - expected) / 20000)
    # Four Monte Carlo SEs plus a finite-sample approximation allowance.
    assert abs(result["rejection_rate"] - expected) < 4 * mcse + 0.003
    assert result["ci_low"] <= result["rejection_rate"] <= result["ci_high"]
    assert result["degenerate_draws"] == 0
    if mode == "null":
        assert expected == pytest.approx(0.05)
        assert result["config"]["effect"] == 0


def test_seed_and_monte_carlo_uncertainty():
    a = simulate_experiments(repetitions=1000)
    b = simulate_experiments(repetitions=1000)
    assert np.array_equal(a["outcomes"]["effect"], b["outcomes"]["effect"])
    large = simulate_experiments(repetitions=20000)
    assert large["ci_high"] - large["ci_low"] < a["ci_high"] - a["ci_low"]
    assert synthetic_intervals(0.1, 0.01, (300, 3000)) == synthetic_intervals(0.1, 0.01, (300, 3000))


@pytest.mark.parametrize("kwargs", [{"n": 10.5}, {"n": True}, {"n": 0},
    {"repetitions": 100001}, {"repetitions": 0}, {"seed": -1}, {"seed": True}, {"mode": "peeking"}])
def test_invalid_simulation_inputs(kwargs):
    with pytest.raises(ValueError):
        simulate_experiments(**kwargs)


@pytest.mark.parametrize("counts", [(1.0, 2), (-1, 2), (1001, 2), (True, 2)])
def test_invalid_success_counts(counts):
    with pytest.raises(ValueError):
        pooled_z_test(*counts, 1000, 1000)
