"""Inference agreement, design boundaries and deterministic simulation checks."""

from math import sqrt
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import binomtest, chi2_contingency, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import example
from from_scratch import sample_size_per_arm, two_proportion_test


def test_assumed_counts_and_unpooled_interval():
    result = two_proportion_test(1500, 15000, 1650, 15000)
    assert result.difference == pytest.approx(0.01)
    assert result.relative_lift == pytest.approx(0.10)
    expected_se = sqrt((0.10 * 0.90 + 0.11 * 0.89) / 15000)
    assert result.se_difference == pytest.approx(expected_se)
    assert result.ci_low == pytest.approx(0.01 - norm.ppf(0.975) * expected_se)
    assert result.ci_high == pytest.approx(0.01 + norm.ppf(0.975) * expected_se)
    assert result.se_null != result.se_difference


@pytest.mark.parametrize("counts", [(100, 1000, 120, 1000), (56, 800, 440, 2100),
                                  (1500, 15000, 1650, 15000), (800, 1000, 700, 900)])
def test_pearson_chi_square_and_normal_tail_agreement(counts):
    x_a, n_a, x_b, n_b = counts
    result = two_proportion_test(*counts)
    reference = chi2_contingency([[x_a, n_a - x_a], [x_b, n_b - x_b]], correction=False)
    assert result.z_statistic ** 2 == pytest.approx(reference.statistic)
    assert result.p_value == pytest.approx(reference.pvalue, rel=1e-10, abs=1e-14)
    assert result.p_value == pytest.approx(2 * norm.sf(abs(result.z_statistic)))


def test_swap_symmetry_and_null_result():
    ab = two_proportion_test(100, 1000, 120, 1100)
    ba = two_proportion_test(120, 1100, 100, 1000)
    assert ab.z_statistic == pytest.approx(-ba.z_statistic)
    assert ab.p_value == pytest.approx(ba.p_value)
    assert ab.ci_low == pytest.approx(-ba.ci_high)
    null = two_proportion_test(100, 1000, 200, 2000)
    assert null.p_value == 1
    assert null.difference == 0


@pytest.mark.parametrize("counts", [(0, 100, 0, 100), (100, 100, 100, 100),
                                  (0, 100, 100, 100), (4, 100, 10, 100)])
def test_sparse_and_degenerate_tables_refused(counts):
    with pytest.raises(ValueError, match="Sparse"):
        two_proportion_test(*counts)


@pytest.mark.parametrize("counts", [(-1, 100, 10, 100), (101, 100, 10, 100),
                                  (10, 0, 10, 100), (10.0, 100, 10, 100),
                                  (True, 100, 10, 100), (10, 100, 10, np.nan)])
def test_invalid_counts_refused(counts):
    with pytest.raises(ValueError):
        two_proportion_test(*counts)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, np.nan, np.inf, True, "0.05"])
def test_invalid_alpha_refused(alpha):
    with pytest.raises(ValueError):
        two_proportion_test(10, 100, 12, 100, alpha)


def test_more_confidence_widens_interval_and_more_data_narrows_it():
    base = two_proportion_test(100, 1000, 120, 1000)
    more_data = two_proportion_test(200, 2000, 240, 2000)
    more_confidence = two_proportion_test(100, 1000, 120, 1000, alpha=0.01)
    assert more_data.se_difference == pytest.approx(base.se_difference / sqrt(2))
    assert more_confidence.ci_low < base.ci_low
    assert more_confidence.ci_high > base.ci_high


def test_sample_size_against_independent_two_tail_normal_power_inversion():
    # Independently integrate both rejection tails with SciPy, then solve n.
    p_a, p_b = 0.10, 0.11
    pooled = (p_a + p_b) / 2

    def power(n):
        threshold = norm.isf(0.025) * sqrt(2 * pooled * (1 - pooled) / n)
        se_alt = sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) / n)
        return norm.cdf((-threshold - (p_b - p_a)) / se_alt) + norm.sf(
            (threshold - (p_b - p_a)) / se_alt)

    reference_n = brentq(lambda n: power(n) - 0.8, 100, 100000)
    planned_n = sample_size_per_arm(p_a, p_b)
    assert abs(planned_n - reference_n) < 1
    assert power(planned_n) == pytest.approx(0.8, abs=0.0001)
    assert sample_size_per_arm(0.10, 0.105) > planned_n
    assert sample_size_per_arm(p_a, p_b, power=0.9) > planned_n
    assert sample_size_per_arm(p_a, p_b, alpha=0.01) > planned_n


@pytest.mark.parametrize("args", [(0, .1), (.1, 1), (.1, .1), (.1, .11, .05, .4),
                                (.1, .11, .05, 1), (.1, np.nan)])
def test_invalid_planning_inputs(args):
    with pytest.raises(ValueError):
        sample_size_per_arm(*args)


def test_assignment_and_outcomes_reproducible_at_user_unit():
    first = example.simulate_users()
    second = example.simulate_users()
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert np.unique(first[0]).size == 30000
    assert not np.array_equal(first[1], example.simulate_users(seed=46)[1])
    result = example.analyze_users(*first)
    assert result["n_a"] + result["n_b"] == 30000
    assert result["srm_p_value"] == pytest.approx(
        binomtest(result["n_b"], 30000, 0.5).pvalue)


def test_srm_flags_a_broken_allocation_even_when_outcome_rates_match():
    assignment = np.r_[np.zeros(1000, dtype=int), np.ones(2000, dtype=int)]
    outcomes = np.r_[np.arange(1000) % 10 == 0, np.arange(2000) % 10 == 0]
    result = example.analyze_users(np.arange(3000), assignment, outcomes)
    assert result["srm_p_value"] < 0.001
    assert result["p_value"] == 1
    assert example.analyze_users(np.arange(3000), assignment, outcomes,
                                 expected_share_b=2 / 3)["srm_p_value"] > 0.9


@pytest.mark.parametrize("problem", ["duplicate", "missing", "nonbinary", "length", "matrix", "empty_arm"])
def test_user_data_contract(problem):
    ids, assignment, outcomes = example.simulate_users(n_users=1000)
    if problem == "duplicate":
        ids[0] = ids[1]
    elif problem == "missing":
        outcomes = outcomes.astype(float)
        outcomes[0] = np.nan
    elif problem == "nonbinary":
        assignment[0] = 2
    elif problem == "length":
        outcomes = outcomes[:-1]
    elif problem == "matrix":
        outcomes = outcomes[:, None]
    elif problem == "empty_arm":
        assignment[:] = 0
    with pytest.raises(ValueError):
        example.analyze_users(ids, assignment, outcomes)


@pytest.mark.parametrize("kwargs", [{"n_users": 0}, {"n_users": 1.5}, {"p_a": 0},
                                  {"p_b": np.nan}, {"seed": -1}])
def test_invalid_simulation_inputs(kwargs):
    with pytest.raises(ValueError):
        example.simulate_users(**kwargs)


def test_repetitions_account_for_all_runs_and_report_mc_uncertainty():
    report = example.repeated_experiments(1000, .1, .1, repetitions=40, seed=7)
    assert report == example.repeated_experiments(1000, .1, .1, repetitions=40, seed=7)
    low, high = report["rejection_rate_mc_ci95"]
    assert low <= report["rejection_rate"] <= high
    assert report["rejection_rate"] == report["rejections"] / 40
    assert 0 <= report["wald_coverage"] <= 1
    with pytest.raises(ValueError):
        example.repeated_experiments(1000, .1, .1, repetitions=0)
