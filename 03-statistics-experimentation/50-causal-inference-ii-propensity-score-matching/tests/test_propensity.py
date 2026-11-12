"""Hand calculations, estimator targets, diagnostics and explicit failure cases."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.special import expit
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import (
    effective_sample_size, ipw_weights, match_propensity, matching_weights,
    standardized_mean_difference, weighted_effect,
)
from example import analyze, main, run_study, simulate


def test_logit_matching_replacement_ties_and_frequency_weights():
    scores = expit([0.0, 0.0, 1.0, 1.0, 1.0])
    t = np.array([0, 0, 1, 1, 1])
    result = match_propensity(scores, t)
    np.testing.assert_array_equal(result.control, [0, 0, 0])
    np.testing.assert_allclose(result.distance, 1)
    weights = matching_weights(result, t)
    np.testing.assert_array_equal(weights, [3, 0, 1, 1, 1])
    y = np.array([2, 200, 3, 5, 7])
    assert weighted_effect(y, t, weights) == pytest.approx(3)
    assert effective_sample_size(weights[t == 0]) == 1


def test_matching_agrees_with_external_nearest_neighbor_search():
    rng = np.random.default_rng(50)
    scores = expit(rng.normal(size=60))
    t = np.tile([0, 1], 30)
    matched = match_propensity(scores, t)
    logits = np.log(scores) - np.log1p(-scores)
    controls = np.flatnonzero(t == 0)
    search = NearestNeighbors(n_neighbors=1).fit(logits[controls, None])
    distances, indices = search.kneighbors(logits[matched.treated, None])
    np.testing.assert_array_equal(matched.control, controls[indices[:, 0]])
    np.testing.assert_allclose(matched.distance, distances[:, 0], atol=1e-14)


def test_inclusive_caliper_and_unmatched_population():
    # Exact same-score pairs survive the zero caliper; the other treated is dropped.
    t = np.array([0, 0, 1, 1])
    p = np.array([.5, .25, .5, .9])
    result = match_propensity(p, t, caliper=0)
    np.testing.assert_array_equal(result.treated, [2])
    np.testing.assert_array_equal(result.unmatched, [3])
    distance = match_propensity(p, t).distance.max()
    assert match_propensity(p, t, caliper=distance).treated.size == 2
    assert match_propensity(p, t, caliper=np.nextafter(distance, 0)).treated.size == 1


def test_no_matches_fail_explicitly_when_forming_an_effect():
    t = np.array([0, 0, 1, 1])
    result = match_propensity([.1, .2, .8, .9], t, caliper=0)
    assert result.treated.size == 0
    assert result.unmatched.size == 2
    with pytest.raises(ValueError, match="No treated"):
        matching_weights(result, t)


def test_ate_and_att_weights_and_hajek_normalization_by_hand():
    t = np.array([0, 0, 1, 1])
    p = [.25, .5, .5, .75]
    y = [1, 3, 6, 10]
    ate = ipw_weights(p, t, "ate")
    att = ipw_weights(p, t, "att")
    np.testing.assert_allclose(ate, [4/3, 2, 2, 4/3])
    np.testing.assert_allclose(att, [1/3, 1, 1, 1])
    assert weighted_effect(y, t, ate) == pytest.approx(5.4)
    assert weighted_effect(y, t, att) == pytest.approx(5.5)
    # Constant arm-specific scaling cancels in normalized contrasts.
    assert weighted_effect(y, t, ate * np.where(t == 1, 7, 3)) == pytest.approx(5.4)


def test_heterogeneous_effects_exact_strata_target_ate_and_att_differ():
    # Two equally common strata, probabilities 1/4 and 3/4; exact arm counts.
    x = np.repeat([0, 1], 4)
    t = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    p = np.where(x == 0, .25, .75)
    tau = 2 + 4*x
    y = 3*x + t*tau
    assert weighted_effect(y, t, ipw_weights(p, t, "ate")) == pytest.approx(4)
    assert weighted_effect(y, t, ipw_weights(p, t, "att")) == pytest.approx(5)
    matches = match_propensity(p, t)
    assert weighted_effect(y, t, matching_weights(matches, t)) == pytest.approx(5)


def test_smd_uses_original_denominator_for_weighted_means():
    x = np.array([[0], [2], [2], [4]], dtype=float)
    t = np.array([0, 0, 1, 1])
    assert standardized_mean_difference(x, t)[0] == pytest.approx(np.sqrt(2))
    # Fixed original pooled SD is sqrt(2), even though reweighted variances differ.
    assert standardized_mean_difference(x, t, [0, 2, 2, 0])[0] == 0
    assert standardized_mean_difference(x, t, [3, 1, 1, 3])[0] == pytest.approx(3 / np.sqrt(2))
    np.testing.assert_allclose(standardized_mean_difference(x * 10 + 7, t), standardized_mean_difference(x, t))


def test_smd_zero_reference_variance_is_not_silently_hidden():
    t = [0, 0, 1, 1]
    assert standardized_mean_difference(np.ones((4, 1)), t)[0] == 0
    assert np.isposinf(standardized_mean_difference([[0], [0], [1], [1]], t)[0])


def test_ess_equal_concentrated_and_rescaled_weights():
    assert effective_sample_size([1, 1, 1]) == pytest.approx(3)
    assert effective_sample_size([0, 5, 0]) == 1
    assert effective_sample_size([1, 2, 3]) == pytest.approx(36/14)
    assert effective_sample_size([1e200, 2e200, 3e200]) == pytest.approx(36/14)


@pytest.mark.parametrize("p", [[0, .4, .6, .7], [1, .4, .6, .7], [-.1, .4, .6, .7],
                                  [np.nan, .4, .6, .7], [.2, .3], [[.2, .3, .4, .5]]])
def test_invalid_scores_rejected_by_both_adjustments(p):
    for function in (ipw_weights, match_propensity):
        with pytest.raises(ValueError):
            function(p, [0, 0, 1, 1])


@pytest.mark.parametrize("t", [[0, 0], [1, 1], [0, 2], [0, np.nan], [], [[0, 1]]])
def test_invalid_treatment_rejected(t):
    with pytest.raises(ValueError):
        ipw_weights(np.full(np.asarray(t).size, .5), t)


@pytest.mark.parametrize("caliper", [-1, np.nan, np.inf])
def test_invalid_caliper(caliper):
    with pytest.raises(ValueError):
        match_propensity([.2, .3, .4, .5], [0, 0, 1, 1], caliper)


@pytest.mark.parametrize("weights", [[0, 0, 1, 1], [1, 1, 0, 0], [-1, 1, 1, 1],
                                       [1, np.inf, 1, 1], [1, 1], [[1, 1, 1, 1]]])
def test_invalid_analysis_weights(weights):
    with pytest.raises(ValueError):
        weighted_effect([1, 2, 3, 4], [0, 0, 1, 1], weights)


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [np.nan, 1], [], [[1, 2]]])
def test_invalid_ess_weights(weights):
    with pytest.raises(ValueError):
        effective_sample_size(weights)


def test_other_input_contracts():
    with pytest.raises(ValueError):
        ipw_weights([.3, .7], [0, 1], "unknown")
    with pytest.raises(ValueError):
        weighted_effect([1, 2, 3], [0, 1], [1, 1])
    for x, t in [([[1], [2]], [0, 1]), ([[np.nan], [0], [1], [2]], [0, 0, 1, 1]),
                 ([0, 1, 2, 3], [0, 0, 1, 1])]:
        with pytest.raises(ValueError):
            standardized_mean_difference(x, t)


@pytest.mark.parametrize("kwargs", [{"n_samples": 3}, {"n_samples": 5.5}, {"n_samples": True},
                                        {"assignment_strength": 0}, {"assignment_strength": np.inf},
                                        {"heterogeneity": np.nan}])
def test_invalid_simulation_configuration(kwargs):
    with pytest.raises(ValueError):
        simulate(**kwargs)


def test_reproducibility_scm_truth_and_restricted_estimand():
    data = simulate(300, 123, heterogeneity=1)
    for actual, expected in zip(data, simulate(300, 123, heterogeneity=1)):
        np.testing.assert_array_equal(actual, expected)
    x, t, y, tau, _ = data
    np.testing.assert_array_equal(tau, 2 + x[:, 0])
    result = analyze(x, t, y, tau)
    assert result["methods"]["ipw_ate"]["truth"] == pytest.approx(tau.mean())
    assert result["methods"]["match_att"]["truth"] == pytest.approx(tau[t == 1].mean())
    assert result["retained"] + result["unmatched"] == t.sum()
    assert not np.isclose(tau.mean(), tau[t == 1].mean())


def test_console_integration_and_configuration_errors(capsys):
    scenarios = main(["--n-samples", "200", "--repetitions", "2"])
    assert set(scenarios) == {1.0, 3.0}
    text = capsys.readouterr().out
    assert "SYNTHETIC ONLY" in text and "MCSE_mean_error" in text
    for repetitions in (0, 1, 2.5, True):
        with pytest.raises(ValueError):
            run_study(repetitions=repetitions)
    with pytest.raises(SystemExit):
        main(["--repetitions", "1"])
