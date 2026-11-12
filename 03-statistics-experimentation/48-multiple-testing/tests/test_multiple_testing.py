"""Known answers, library agreement and decision/denominator contracts."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.stats import false_discovery_control, norm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import METHODS, adjust_pvalues, reject_hypotheses
from example import (estimate, family_metrics, main, normal_look_pvalues,
                     peeking_decisions, simulate_study)


@pytest.mark.parametrize("method,expected", [
    ("bonferroni", [0.005, 0.06, 0.105, 0.245, 1.0]),
    ("holm", [0.005, 0.048, 0.063, 0.098, 0.2]),
    ("bh", [0.005, 0.03, 0.035, 0.06125, 0.2]),
])
def test_hand_calculated_adjustments(method, expected):
    np.testing.assert_allclose(adjust_pvalues([.001, .012, .021, .049, .2], method), expected)


def test_bh_uses_last_passing_rank_even_when_earlier_rank_fails():
    # Rank 1 fails .03 <= .025, but rank 2 passes .04 <= .05: reject both.
    assert reject_hypotheses([.03, .04], method="bh").tolist() == [True, True]
    assert reject_hypotheses([.03, .04], method="holm").tolist() == [False, False]


@pytest.mark.parametrize("method,library", [("bonferroni", "bonferroni"),
                                         ("holm", "holm"), ("bh", "fdr_bh")])
@pytest.mark.parametrize("p", [[0, 1, .05, .05], [.7], [.9, .002, .03, .002, .05],
                               np.random.default_rng(480).uniform(size=101)])
def test_statsmodels_agreement_and_input_preservation(method, library, p):
    original = np.array(p, dtype=float)
    saved = original.copy()
    expected_mask, expected_p, _, _ = multipletests(original, alpha=.05, method=library)
    np.testing.assert_allclose(adjust_pvalues(original, method), expected_p, atol=1e-15)
    np.testing.assert_array_equal(reject_hypotheses(original, method=method), expected_mask)
    np.testing.assert_array_equal(original, saved)


def test_scipy_bh_agreement_and_permutation_equivariance():
    p = np.array([.03, .002, .6, .004, .03, .95])
    permutation = np.array([4, 0, 3, 1, 5, 2])
    np.testing.assert_allclose(adjust_pvalues(p), false_discovery_control(p))
    for method in METHODS:
        np.testing.assert_allclose(adjust_pvalues(p[permutation], method),
                                   adjust_pvalues(p, method)[permutation])


@pytest.mark.parametrize("method", METHODS)
def test_singleton_endpoints_and_inclusive_boundary(method):
    assert reject_hypotheses([.05], .05, method).tolist() == [True]
    assert reject_hypotheses([0, 1], .05, method).tolist() == [True, False]
    assert reject_hypotheses([1, 1], .05, method).tolist() == [False, False]
    assert reject_hypotheses([0, 0], .05, method).tolist() == [True, True]


@pytest.mark.parametrize("p", [[], [[.1]], [.1, np.nan], [np.inf], [-.01], [1.01],
                              [1j], ["invalid"], .1])
def test_invalid_pvalues_fail(p):
    with pytest.raises(ValueError):
        adjust_pvalues(p)


@pytest.mark.parametrize("level", [0, 1, -.1, np.nan, np.inf, [], True, 1j,
                                  np.complex64(.05 + 1j)])
def test_invalid_levels_fail(level):
    with pytest.raises(ValueError):
        reject_hypotheses([.1], level)
    with pytest.raises(ValueError):
        peeking_decisions([[.1]], level)


def test_unknown_method_fails():
    with pytest.raises(ValueError):
        adjust_pvalues([.1], "fdr")


def test_fdr_is_mean_fdp_including_empty_discoveries_not_pooled_ratio():
    rejected = np.array([[True, False], [True, True], [False, False]])
    result = family_metrics(rejected, np.array([True, False]))
    assert result["fwer"]["estimate"] == pytest.approx(2 / 3)
    assert result["fdr"]["estimate"] == .5  # Mean(1, 1/2, 0), not 2/3 pooled.
    assert result["power"]["estimate"] == pytest.approx(1 / 3)
    assert result["fdr"]["mc_se"] == pytest.approx(np.std([1, .5, 0], ddof=1) / np.sqrt(3))
    global_null = family_metrics(rejected, np.array([True, True]))
    assert global_null["fwer"] == global_null["fdr"]
    assert global_null["power"] is None


@pytest.mark.parametrize("rejected,null", [([[True]], [True]), ([[1], [0]], [True]),
                                          ([[True], [False]], [True, False]),
                                          ([[True], [False]], [1])])
def test_invalid_discovery_masks_fail(rejected, null):
    with pytest.raises(ValueError):
        family_metrics(rejected, null)


def test_cumulative_looks_use_block_sums_and_current_sample_sizes():
    sums = np.array([[2., 3., -1.], [0., 0., 0.]])
    looks = [4, 9, 16]
    expected_z = [[1., 5 / 3, 1.], [0., 0., 0.]]
    np.testing.assert_allclose(normal_look_pvalues(sums, looks), 2 * norm.sf(np.abs(expected_z)))


@pytest.mark.parametrize("looks", [[], [0], [10, 10], [20, 10], [10.5, 20], [True]])
def test_invalid_look_schedule_fails(looks):
    with pytest.raises(ValueError):
        normal_look_pvalues(np.zeros((2, len(looks))), looks)


@pytest.mark.parametrize("sums", [[[1]], [[np.nan, 0]], [[1j, 0]], [1, 2]])
def test_invalid_block_sums_fail(sums):
    with pytest.raises(ValueError):
        normal_look_pvalues(sums, [10, 20])


def test_peeking_crossings_and_one_look_equivalence():
    decisions = peeking_decisions([[.04, .9], [.9, .04], [.01, .9], [.9, .9]])
    assert decisions["fixed_horizon"].tolist() == [False, True, False, False]
    assert decisions["naive_peeking"].tolist() == [True, True, True, False]
    assert decisions["bonferroni_looks"].tolist() == [False, False, True, False]
    single = peeking_decisions([[.04], [.9]])
    np.testing.assert_array_equal(single["fixed_horizon"], single["naive_peeking"])
    np.testing.assert_array_equal(single["fixed_horizon"], single["bonferroni_looks"])


@pytest.mark.parametrize("values", [[], [1], [[1, 2]], [1, np.nan]])
def test_invalid_monte_carlo_samples_fail(values):
    with pytest.raises(ValueError):
        estimate(values)


@pytest.mark.parametrize("trials,seed", [(1, 48), (2.5, 48), (True, 48), (10, -1), (10, 1.5)])
def test_invalid_simulation_configuration_fails(trials, seed):
    with pytest.raises(ValueError):
        simulate_study(trials, seed)


def test_cli_writes_report_and_simulation_is_repeatable(tmp_path, capsys):
    import json
    output = tmp_path / "nested" / "study.json"
    report = main(["--trials", "30", "--seed", "48", "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == report
    captured = json.loads(capsys.readouterr().out)
    assert captured == report
    report.pop("executed_at_utc")
    assert report == simulate_study(30, 48)
    assert report["synthetic"] is True
    assert report["interpretation_status"] == "author review required"
    assert report["analytic_uncorrected_global_null_fwer"] == pytest.approx(1 - .95 ** 20)
