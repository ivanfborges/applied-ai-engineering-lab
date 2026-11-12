"""Hand identities, tie policies, causal scoring and small integration checks."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import (
    cost_aware_policy, fit_binned_uplift, gain_curve, ipw_effect,
    policy_value, ranking_metrics, top_fraction,
)
from example import fit_t_learner, positive_probability, run_once, simulate, summarize


def test_ipw_unequal_assignment_by_hand():
    np.testing.assert_allclose(ipw_effect([1, 1, 0, 0], [1, 0, 1, 0], 0.25), [4, -4 / 3, 0, 0])
    np.testing.assert_allclose(ipw_effect([1, 1], [1, 0], [0.2, 0.6]), [5, -2.5])


@pytest.mark.parametrize("propensity", [0.1, 0.5, 0.9])
def test_exact_assignment_expectation_recovers_cate(propensity):
    p0, p1 = 0.3, 0.7
    weights = np.array([propensity * p1, propensity * (1 - p1),
                        (1 - propensity) * p0, (1 - propensity) * (1 - p0)])
    z = ipw_effect([1, 0, 1, 0], [1, 1, 0, 0], propensity)
    assert weights @ z == pytest.approx(p1 - p0)


@pytest.mark.parametrize("propensity", [0, 1, -0.1, np.nan, np.inf, [0.5], [[0.5, 0.5]]])
def test_invalid_propensity(propensity):
    with pytest.raises(ValueError):
        ipw_effect([1, 0], [1, 0], propensity)


@pytest.mark.parametrize("outcome,treatment", [([], []), ([1], [0, 1]), ([2], [0]),
                                                        ([np.nan], [1]), ([0], [0.2]), ([[1]], [0])])
def test_invalid_observations(outcome, treatment):
    with pytest.raises(ValueError):
        ipw_effect(outcome, treatment)


def test_gain_denominator_endpoints_direction_and_area():
    scores, z = [4, 3, 2, 1], [2, 0, -2, 0]
    q, gain = gain_curve(scores, z)
    np.testing.assert_allclose(q, [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_allclose(gain, [0, 0.5, 0.5, 0, 0])
    assert ranking_metrics(scores, z) == pytest.approx({"auuc": 0.25, "qini_area": 0.25})
    assert ranking_metrics([-v for v in scores], z)["qini_area"] == pytest.approx(-0.25)


def test_ties_do_not_exploit_outcome_order():
    q1, gain1 = gain_curve([2, 2, 1, 1], [2, -1, 4, -3])
    q2, gain2 = gain_curve([2, 2, 1, 1], [-1, 2, -3, 4])
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_allclose(gain1, gain2)
    assert ranking_metrics([1, 1, 1], [4, -2, 3])["qini_area"] == pytest.approx(0)


def test_constant_and_zero_effect_curve():
    q, gain = gain_curve([4, 1, 3, 2], np.full(4, 0.05))
    np.testing.assert_allclose(gain, q * 0.05)
    assert ranking_metrics([4, 1, 3, 2], np.full(4, 0.05))["qini_area"] == pytest.approx(0)
    np.testing.assert_array_equal(gain_curve([3, 2, 1], [0, 0, 0])[1], 0)


def test_ranking_endpoints_agree_even_if_ate_is_negative():
    z = [-2, 0, -2, 1]
    for scores in ([1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1]):
        assert gain_curve(scores, z)[1][-1] == pytest.approx(np.mean(z))


def test_policy_capacity_ties_and_strict_break_even():
    np.testing.assert_array_equal(top_fraction([1, 1, 1, 0], 0.5), [1, 1, 0, 0])
    np.testing.assert_array_equal(cost_aware_policy([0.2, 0.1, 0.05, -0.1], 100, 5, 1), [1, 1, 0, 0])
    assert cost_aware_policy([0.5, 0.4], capacity=0).sum() == 0
    assert cost_aware_policy([0.5, 0.4], value=0, cost=0, capacity=1).sum() == 0
    assert top_fraction(np.arange(10), 0.3).sum() == 3


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.nan, [0.5]])
def test_invalid_capacity(fraction):
    with pytest.raises(ValueError):
        cost_aware_policy([0.1, 0.2], capacity=fraction)


def test_policy_net_value_uses_all_eligible_and_counts_all_contacts():
    assert policy_value([1, 0, 1, 0], [2, -2, 0, 0], 100, 5) == pytest.approx(47.5)
    assert policy_value([0, 0], [2, -2]) == 0
    assert policy_value([1, 1], [0, 0], 100, 5) == -5


@pytest.mark.parametrize("kwargs", [{"cost": -1}, {"value": np.inf}, {"value": [100]}])
def test_invalid_economics(kwargs):
    with pytest.raises(ValueError):
        policy_value([1, 0], [2, -2], **kwargs)


def test_training_quantiles_hand_effects_and_out_of_range_prediction():
    model = fit_binned_uplift([-2, -1, 1, 2], [0, 1, 0, 1], [0, 1, 1, 0], 2)
    np.testing.assert_allclose(model.effects, [1, -1])
    np.testing.assert_array_equal(model.counts, [[1, 1], [1, 1]])
    np.testing.assert_allclose(model.predict([-100, 0, 100]), [1, -1, -1])
    np.testing.assert_array_equal(model.cuts, [0])


def test_constant_feature_collapses_bins_and_empty_arm_errors():
    model = fit_binned_uplift([1, 1, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1])
    np.testing.assert_allclose(model.predict([0, 1, 2]), 1)
    with pytest.raises(ValueError, match="both treatment arms"):
        fit_binned_uplift([-2, -1, 1, 2], [0, 0, 1, 1], [0, 1, 0, 1], 2)


@pytest.mark.parametrize("bins", [0, -1, 1.5, True])
def test_invalid_bins(bins):
    with pytest.raises(ValueError):
        fit_binned_uplift([0, 1], [0, 1], [0, 1], bins)


def test_generator_reproducibility_probabilities_and_homogeneous_control():
    first, second = simulate(n=500), simulate(n=500)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert first[3].min() < 0 < first[3].max()
    np.testing.assert_allclose(simulate(n=500, scenario="homogeneous")[3], 0.05, atol=1e-16)


@pytest.mark.parametrize("kwargs", [{"n": 10}, {"n": 100.5}, {"propensity": 1}, {"scenario": "unknown"}])
def test_invalid_generator_configuration(kwargs):
    with pytest.raises(ValueError):
        simulate(**kwargs)


def test_single_class_models_and_missing_arm():
    x = np.arange(8).reshape(-1, 1)
    control, treated = fit_t_learner(x, [0] * 4 + [1] * 4, [0] * 4 + [1] * 4)
    np.testing.assert_allclose(positive_probability(control, x), 0)
    np.testing.assert_allclose(positive_probability(treated, x), 1)
    with pytest.raises(ValueError, match="both treatment arms"):
        fit_t_learner(x, [0] * 8, [0] * 8)


def test_small_end_to_end_run_and_evaluation_does_not_leak_into_fitting(monkeypatch):
    import example
    original = example.simulate
    original_fit = example.fit_t_learner
    original_top = example.top_fraction
    original_cost = example.cost_aware_policy
    fits, selections = [], []

    def capture_fit(x, t, y, seed):
        fits.append((x.copy(), t.copy(), y.copy()))
        return original_fit(x, t, y, seed)

    def capture_top(*args):
        selected = original_top(*args)
        selections.append(selected.copy())
        return selected

    def capture_cost(*args):
        selected = original_cost(*args)
        selections.append(selected.copy())
        return selected

    monkeypatch.setattr(example, "fit_t_learner", capture_fit)
    monkeypatch.setattr(example, "top_fraction", capture_top)
    monkeypatch.setattr(example, "cost_aware_policy", capture_cost)
    result = run_once(n=600, seed=91)

    def changed_holdout(*args, **kwargs):
        x, t, y, tau = original(*args, **kwargs)
        _, test = example.train_test_split(np.arange(len(y)), test_size=0.4, random_state=91)
        y[test] = 1 - y[test]
        tau[test] += 100  # Oracle truth cannot affect the learned policies either.
        return x, t, y, tau

    monkeypatch.setattr(example, "simulate", changed_holdout)
    changed = run_once(n=600, seed=91)
    for name in result["policies"]:
        assert result["policies"][name]["treated_fraction"] == changed["policies"][name]["treated_fraction"]
    for before, after in zip(fits[0], fits[1]):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(selections[:3], selections[3:]):
        np.testing.assert_array_equal(before, after)


def test_summary_mcse_is_about_replication_mean():
    assert summarize([1, 3]) == pytest.approx({"mean": 2, "sd_across_runs": np.sqrt(2), "mcse_of_mean": 1})
    assert summarize([1])["mcse_of_mean"] is None
