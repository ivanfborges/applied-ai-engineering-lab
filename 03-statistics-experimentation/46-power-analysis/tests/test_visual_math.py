"""Numerical identities, library agreement and design constraints."""
from pathlib import Path
import sys
import numpy as np
import pytest
from statsmodels.stats.proportion import samplesize_proportions_2indep_onetail

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example import normal_power
from from_scratch import sample_size_per_arm
from power_math import (approximation_warning, cluster_information, detectable_effect,
                        required_n, sampling_model, theoretical_power, traffic_plan)


@pytest.mark.parametrize("p,effect", [(0.1, 0.01), (0.4, 0.03), (0.8, -0.02)])
def test_reference_agreement_and_integer_sizing(p, effect):
    n = required_n(p, effect)
    reference = samplesize_proportions_2indep_onetail(effect, p, power=0.8, alpha=0.05)
    assert abs(sample_size_per_arm(p, p + effect) - reference) <= 1
    assert abs(n - reference) <= 1
    assert theoretical_power(p, effect, n) >= 0.8
    assert theoretical_power(p, effect, n - 1) < 0.8
    assert theoretical_power(p, effect, n) == pytest.approx(normal_power(p, p + effect, n, n))


def test_sizing_sensitivity_and_monotone_power():
    base = required_n(0.1, 0.01)
    assert required_n(0.1, 0.01, target=0.9) > base
    assert required_n(0.1, 0.01, alpha=0.01) > base
    assert required_n(0.1, 0.005) > base > required_n(0.1, 0.02)
    powers = [theoretical_power(0.1, 0.01, n) for n in (100, 1000, 10000, 100000)]
    assert powers == sorted(powers)


@pytest.mark.parametrize("p,effect,n", [(0.1, 0, 1000), (0.1, 0.01, 100), (0.9, -0.1, 50000), (0.001, 0.001, 100)])
def test_distribution_probabilities_and_bounds(p, effect, n):
    model = sampling_model(p, effect, n)
    assert model["alpha"] == pytest.approx(0.05)
    assert model["beta"] + model["power"] == pytest.approx(1)
    assert 0 <= model["power"] <= 1
    assert model["power"] == pytest.approx(theoretical_power(p, effect, n))
    if effect == 0:
        assert model["power"] == pytest.approx(0.05)


def test_mde_inverts_power():
    small = detectable_effect(0.1, 6000)
    large = detectable_effect(0.1, 15000)
    assert small > large > 0
    assert theoretical_power(0.1, small, 6000) == pytest.approx(0.8, abs=1e-8)
    assert detectable_effect(0.999, 1, target=0.99) is None


def test_cluster_heuristic_boundaries():
    assert cluster_information(1000, 10, 0) == (10000, 1, 10000)
    assert cluster_information(1000, 10, 1) == (10000, 10, 1000)
    assert cluster_information(1000, 1, 0.9) == (1000, 1, 1000)
    assert cluster_information(1000, 10, 0.2)[2] < cluster_information(1000, 10, 0.1)[2]


def test_expected_traffic_and_maturity_are_distinct():
    plan = traffic_plan(14751, 2500, 0.4, 1, delay=2, window=14)
    assert plan["assigned"][-1] == 7000
    assert plan["mature"][-1] == plan["available_n"] == 6000
    assert plan["reach_day"] == 32
    assert not plan["sufficient"]
    assert np.all(plan["mature"] <= plan["assigned"])
    assert traffic_plan(100, 100, 1, 1, delay=5, window=2)["available_n"] == 0
    assert traffic_plan(100, 100, 1, 1, delay=0, window=2)["sufficient"]


@pytest.mark.parametrize("bad", [0, 1, -0.1, np.nan, np.inf, True, "0.1"])
def test_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        theoretical_power(bad, 0.01, 100)
    with pytest.raises(ValueError):
        theoretical_power(0.1, 0.01, 100, alpha=bad)


@pytest.mark.parametrize("bad", [0, -1, True, np.inf, np.nan, "100"])
def test_invalid_sample_sizes(bad):
    with pytest.raises(ValueError):
        theoretical_power(0.1, 0.01, bad)
    with pytest.raises(ValueError):
        detectable_effect(0.1, bad)


def test_invalid_design_inputs_and_sparse_warning():
    for delta in (-0.2, 0.9, np.nan):
        with pytest.raises(ValueError):
            theoretical_power(0.1, delta, 100)
    for rho in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError):
            cluster_information(100, 10, rho)
    with pytest.raises(ValueError):
        required_n(0.1, 0)
    with pytest.raises(ValueError):
        traffic_plan(100, 100, 1.5, 1, 0, 14)
    with pytest.raises(ValueError):
        traffic_plan(100, 100, 1, 1, 1.5, 14)
    assert approximation_warning(0.001, 0.001, 100)
    assert not approximation_warning(0.1, 0.01, 15000)
