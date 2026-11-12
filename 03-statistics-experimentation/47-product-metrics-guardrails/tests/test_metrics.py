"""Known-answer contracts, cohort boundaries, missingness and library agreement."""

import json
from math import sqrt
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example import generate_users, main
from metrics import conversion_lift, nearest_rank, rate, scorecard, zero_incident_upper_bound


@pytest.fixture
def users():
    return pd.DataFrame({
        "user_id": list("abcdef"),
        "arm": ["control"] * 4 + ["treatment"] * 2,
        "followup_days": [8, 7, 6, 9, 8, 8],
        "served_ok": [True, False, True, True, True, True],
        "converted": [True, False, False, False, True, True],
        "returned_d7": pd.array([True, None, None, False, True, False], dtype="boolean"),
        "subscriber_at_start": [True, True, False, False, False, True],
        "churned_7d": pd.array([False, True, None, None, None, False], dtype="boolean"),
        "latency_s": [1., 2., 3., 10., 2., 4.],
        "revenue": [20., 0., 0., 0., 20., 20.],
        "cost": [1., 2., 3., 4., 1., 1.],
        "unsafe": [False, False, True, False, False, False],
    })


def test_known_scorecard_keeps_failures_and_cohort_denominators(users):
    summary = scorecard(users)
    control = summary.loc["control"]
    assert control["assigned_users"] == 4
    assert control["conversion"] == 0.25  # Not 1/3 among successful requests.
    assert control["error_rate"] == 0.25
    assert control["d7_eligible"] == 2  # Day 7 has not finished the [7, 8) window.
    assert control["d7_retention"] == 0.5
    assert control["churn_eligible"] == 2  # Day 7 has finished [0, 7).
    assert control["subscription_churn_7d"] == 0.5
    assert control["p95_latency_s"] == 10
    assert control["revenue_per_user"] == 5
    assert control["cost_per_request"] == 2.5
    assert control["contribution_per_user"] == 2.5
    assert control["cost_per_conversion"] == 10
    assert control["unsafe_per_request"] == 0.25
    assert summary.loc["treatment", "assigned_users"] == 2
    assert summary.loc["treatment", "subscription_churn_7d"] == 0
    assert summary.loc["treatment", "d7_retention"] == 0.5


def test_empty_metric_populations_are_unavailable_not_zero(users):
    users["followup_days"] = 1
    users["returned_d7"] = pd.array([None] * len(users), dtype="boolean")
    users["churned_7d"] = pd.array([None] * len(users), dtype="boolean")
    users["converted"] = False
    summary = scorecard(users)
    assert (summary["d7_eligible"] == 0).all()
    assert (summary["churn_eligible"] == 0).all()
    assert summary[["d7_retention", "subscription_churn_7d", "cost_per_conversion"]].isna().all().all()
    assert (summary["conversion"] == 0).all()


@pytest.mark.parametrize("probability", [0, 0.01, 0.5, 0.95, 1])
def test_nearest_rank_matches_numpy_inverse_cdf_with_ties(probability):
    values = [10., 1., 1., 5., 3., 20.]
    assert nearest_rank(values, probability) == np.quantile(values, probability, method="inverted_cdf")


@pytest.mark.parametrize("values, probability", [([], 0.95), ([1, np.nan], 0.95),
    ([1, np.inf], 0.95), ([[1, 2]], 0.95), (["1", "2"], 0.95),
    ([1, 2], -0.1), ([1, 2], 1.1), ([1, 2], np.nan)])
def test_invalid_quantile_inputs(values, probability):
    with pytest.raises(ValueError):
        nearest_rank(values, probability)


@pytest.mark.parametrize("events, eligible", [(0, 0), (-1, 10), (11, 10), (1.5, 10), (1, True)])
def test_invalid_rate_inputs(events, eligible):
    with pytest.raises(ValueError):
        rate(events, eligible)


def test_unequal_sample_conversion_interval_matches_scipy_normal_calculation():
    result = conversion_lift(12, 100, 45, 300)
    se = sqrt(0.12 * 0.88 / 100 + 0.15 * 0.85 / 300)
    assert result["absolute_lift"] == pytest.approx(0.03)
    assert result["relative_lift"] == pytest.approx(0.25)
    assert result["standard_error"] == pytest.approx(se)
    assert [result["ci_low"], result["ci_high"]] == pytest.approx(norm.interval(0.95, loc=0.03, scale=se))
    swapped = conversion_lift(45, 300, 12, 100)
    assert swapped["ci_low"] == pytest.approx(-result["ci_high"])


@pytest.mark.parametrize("args", [(0, 100, 5, 100), (101, 100, 10, 100),
                                 (10, 100, 10, 100, 0), (10, 100, 10, 100, np.nan)])
def test_invalid_or_sparse_conversion_interval(args):
    with pytest.raises(ValueError):
        conversion_lift(*args)


@pytest.mark.parametrize("n, alpha", [(1, .05), (1000, .05), (10000, .01)])
def test_zero_incident_exact_bound_matches_scipy(n, alpha):
    expected = binomtest(0, n, alternative="less").proportion_ci(1 - alpha, method="exact").high
    assert zero_incident_upper_bound(n, alpha) == pytest.approx(expected, abs=1e-12)
    assert zero_incident_upper_bound(n, alpha) > 0


@pytest.mark.parametrize("n, alpha", [(0, .05), (-1, .05), (1.5, .05), (10, 0), (10, 1), (10, np.nan)])
def test_invalid_zero_incident_bound(n, alpha):
    with pytest.raises(ValueError):
        zero_incident_upper_bound(n, alpha)


@pytest.mark.parametrize("column, index, value", [
    ("user_id", 0, "b"), ("user_id", 0, ""), ("user_id", 0, None),
    ("arm", 0, "unknown"), ("latency_s", 0, np.nan), ("cost", 0, -1),
    ("revenue", 0, np.inf), ("served_ok", 0, None), ("converted", 0, None),
    ("unsafe", 0, None), ("returned_d7", 0, None), ("returned_d7", 1, False),
    ("churned_7d", 1, None), ("churned_7d", 2, False),
    ("converted", 1, True), ("unsafe", 1, True), ("followup_days", 0, 0),
])
def test_malformed_or_missing_telemetry_is_rejected(users, column, index, value):
    # Object dtype lets tests supply malformed data without pandas coercion hiding it.
    if column in ("served_ok", "converted", "unsafe", "user_id"):
        users[column] = users[column].astype(object)
    users.loc[index, column] = value
    with pytest.raises(ValueError):
        scorecard(users)


def test_required_columns_and_both_arms(users):
    for invalid in (users.drop(columns="unsafe"), users.iloc[:0], users.iloc[:4]):
        with pytest.raises(ValueError):
            scorecard(invalid)


def test_generated_data_is_reproducible_and_obeys_maturity_contract():
    first = generate_users(100, 47)
    pd.testing.assert_frame_equal(first, generate_users(100, 47))
    assert not first.equals(generate_users(100, 48))
    summary = scorecard(first)
    assert summary["assigned_users"].tolist() == [100, 100]
    assert (summary["d7_eligible"] < 100).all()


@pytest.mark.parametrize("n, seed", [(0, 47), (1.5, 47), (True, 47), (10, -1), (10, 1.5)])
def test_invalid_generation_configuration(n, seed):
    with pytest.raises(ValueError):
        generate_users(n, seed)


def test_cli_writes_unique_strict_json_records(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["example.py", "--users-per-arm", "1000", "--seed", "99",
                                    "--output-dir", str(tmp_path)])
    main()
    main()
    records = sorted(tmp_path.glob("scorecard-*.json"))
    assert len(records) == 2
    first, second = [json.loads(path.read_text(encoding="utf-8")) for path in records]
    assert first["scorecard"] == second["scorecard"]
    assert first["interpretation_status"] == "author review required"
    assert first["configuration"]["seed"] == 99
    assert first["zero_incident_exercise"]["incidents"] == 0
    assert "no guardrail inference" in capsys.readouterr().out
