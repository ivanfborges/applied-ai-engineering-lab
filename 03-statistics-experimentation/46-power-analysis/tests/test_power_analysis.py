"""Numerical identities, sizing boundaries and traffic-planning behavior."""

from math import sqrt
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example import calendar_days, normal_power, required_sample_sizes
from from_scratch import sample_size_per_arm


def test_null_power_is_alpha_and_arm_swap_is_symmetric():
    for alpha in (0.01, 0.05, 0.10):
        assert normal_power(0.2, 0.2, 500, 2000, alpha) == pytest.approx(alpha)
    assert normal_power(0.1, 0.14, 800, 1600) == pytest.approx(
        normal_power(0.14, 0.1, 1600, 800)
    )


@pytest.mark.parametrize("n", [50, 500, 5000])
def test_power_matches_integration_over_two_rejection_regions(n):
    # For this fixed balanced case: pooled=0.105, variances=0.18795/n and 0.1879/n.
    cutoff = norm.isf(0.025) * sqrt(0.18795 / n)
    distribution = norm(loc=0.01, scale=sqrt(0.1879 / n))
    left = quad(distribution.pdf, -np.inf, -cutoff, epsabs=1e-10)[0]
    right = quad(distribution.pdf, cutoff, np.inf, epsabs=1e-10)[0]
    assert normal_power(0.10, 0.11, n, n) == pytest.approx(left + right, abs=1e-9)
    if n == 50:
        assert left > 0.005  # A one-tail-only implementation would miss meaningful mass.


@pytest.mark.parametrize("rates", [(0.10, 0.11), (0.10, 0.105), (0.50, 0.52), (0.80, 0.75)])
def test_balanced_integer_boundary_and_closed_form_agreement(rates):
    nc, nt = required_sample_sizes(*rates)
    assert nc == nt
    assert normal_power(*rates, nc, nt) >= 0.80
    assert normal_power(*rates, nc - 1, nt - 1) < 0.80
    assert abs(sample_size_per_arm(*rates) - nc) <= 1
    assert sample_size_per_arm(*rates) == sample_size_per_arm(*rates[::-1])


def test_baseline_and_sensitivity_to_assumptions():
    base = required_sample_sizes(0.10, 0.11)[0]
    assert base == sample_size_per_arm(0.10, 0.11) == 14751
    assert required_sample_sizes(0.10, 0.105)[0] > base
    assert required_sample_sizes(0.10, 0.12)[0] < base
    assert required_sample_sizes(0.10, 0.11, alpha=0.01)[0] > base
    assert required_sample_sizes(0.10, 0.11, target_power=0.90)[0] > base
    powers = [normal_power(0.10, 0.11, n, n) for n in (100, 1000, 10000, 100000)]
    assert powers == sorted(powers)
    assert powers[-1] > 0.999


@pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0])
def test_unequal_allocation_rounding_and_target(ratio):
    nc, nt = required_sample_sizes(0.10, 0.11, ratio=ratio)
    assert abs(nt - ratio * nc) <= max(1, ratio)
    assert normal_power(0.10, 0.11, nc, nt) >= 0.80
    if ratio != 1:
        assert nc + nt > sum(required_sample_sizes(0.10, 0.11))


def test_search_cap_and_zero_effect_are_explicit():
    with pytest.raises(ValueError, match="search cap"):
        required_sample_sizes(0.10, 0.10000001)
    with pytest.raises(ValueError, match="nonzero"):
        required_sample_sizes(0.10, 0.10)
    with pytest.raises(ValueError, match="nonzero"):
        sample_size_per_arm(0.10, 0.10)


@pytest.mark.parametrize("bad", [0, 1, -0.1, np.nan, np.inf, True, "0.10"])
def test_invalid_probabilities(bad):
    for calculator in (sample_size_per_arm, required_sample_sizes):
        with pytest.raises(ValueError):
            calculator(bad, 0.11)
        with pytest.raises(ValueError):
            calculator(0.10, bad)
        with pytest.raises(ValueError):
            calculator(0.10, 0.11, alpha=bad)
        with pytest.raises(ValueError):
            calculator(0.10, 0.11, target_power=bad)
    with pytest.raises(ValueError):
        normal_power(0.1, 0.11, 100, 100, alpha=bad)


@pytest.mark.parametrize("power,alpha", [(0.5, 0.05), (0.4, 0.05), (0.8, 0.9)])
def test_closed_form_and_solver_restrict_target_domain(power, alpha):
    for calculator in (sample_size_per_arm, required_sample_sizes):
        with pytest.raises(ValueError, match="exceed"):
            calculator(0.1, 0.11, alpha=alpha, target_power=power)


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf, True, "100"])
def test_invalid_counts_and_ratios(bad):
    with pytest.raises(ValueError):
        normal_power(0.1, 0.11, bad, 100)
    with pytest.raises(ValueError):
        normal_power(0.1, 0.11, 100, bad)
    with pytest.raises(ValueError):
        required_sample_sizes(0.1, 0.11, ratio=bad)


def test_duration_accounts_for_slow_arm_floor_and_outcome_delay():
    assert calendar_days(14751, 14751, 2500, 0.4, outcome_delay_days=2) == 32
    assert calendar_days(14751, 14751, 10000, 0.4,
                         minimum_enrollment_days=7, outcome_delay_days=2) == 10
    assert calendar_days(100, 200, 100, 1.0, treatment_share=2 / 3) == 3
    assert calendar_days(100, 200.1, 100, 1.0, treatment_share=2 / 3) == 4
    assert calendar_days(100, 200, 100, 1.0, treatment_share=0.5) == 4
    assert calendar_days(1, 1, 1000, 1.0,
                         minimum_enrollment_days=7, outcome_delay_days=2) == 9


@pytest.mark.parametrize("kwargs", [
    {"daily_unique_units": 0}, {"eligibility": 0}, {"eligibility": 1.1},
    {"eligibility": np.nan}, {"treatment_share": 1}, {"n_control": -1},
    {"outcome_delay_days": -1}, {"outcome_delay_days": 1.5},
    {"minimum_enrollment_days": True}, {"minimum_enrollment_days": np.inf},
])
def test_invalid_business_inputs(kwargs):
    inputs = dict(n_control=100, n_treatment=100, daily_unique_units=100, eligibility=1.0)
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        calendar_days(**inputs)
