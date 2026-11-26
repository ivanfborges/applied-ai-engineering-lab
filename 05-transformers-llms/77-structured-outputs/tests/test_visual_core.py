"""Checks for the executed synthetic experiments and theoretical retry models."""

from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visual_core as core


@pytest.mark.parametrize("temperature", [.1, .5, 1., 3.])
def test_masking_and_probability_identity(temperature):
    row = core.masking([1.2, .8, 2, -.2, .1], [True, True, False, False, False], temperature)
    q, p, allowed = np.array(row["constrained"]), np.array(row["original"]), np.array(row["mask"])
    assert_allclose(q.sum(), 1)
    assert_allclose(p.sum(), 1)
    assert np.all(q[~allowed] == 0)
    assert_allclose(q[allowed], p[allowed] / p[allowed].sum())
    assert row["masked_logits"][2:] == [None, None, None]


@pytest.mark.parametrize("temperature", [0, -1, np.inf, np.nan])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        core.masking([1, 2], [True, False], temperature)


def test_empty_mask_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        core.masking([1, 2], [False, False])


def test_grammar_trace_samples_only_permitted_tokens():
    rows = core.decoding_trace()
    assert rows == core.decoding_trace()
    state = core.GrammarState()
    for row in rows:
        assert row["selected"] in core.outgoing(state)
        assert_allclose(sum(row["constrained"]), 1)
        assert all(p == 0 for p, keep in zip(row["constrained"], row["mask"]) if not keep)
        state = core.advance(state, row["selected"])
    assert state.phase == "terminated"
    assert rows[-1]["selected"] == '<EOS>'


@pytest.mark.parametrize("token", ['true', 'null', '"1500"', '}', '<EOS>'])
def test_numeric_state_rejects_incompatible_continuations(token):
    with pytest.raises(ValueError, match="Forbidden"):
        core.advance(core.GrammarState("value", 1), token)


@pytest.mark.parametrize("token", ['}', '<EOS>', '"currency"'])
def test_missing_required_fields_cannot_terminate(token):
    with pytest.raises(ValueError):
        core.advance(core.GrammarState("property", 0), token)


def test_grammar_counterexamples_leave_state_unchanged():
    rows = core.grammar_rejections()
    assert len(rows) == 3
    assert all(row["state_unchanged"] and row["selected"] not in row["allowed"] for row in rows)


def test_funnel_counts_use_actual_gates_and_denominators():
    data = core.validation_experiment(per_category=3)
    assert data["population"] == 27
    assert [r["survivors"] for r in data["stages"]] == [24, 12, 9, 6, 3, 3]
    assert [r["rejected"] for r in data["stages"]] == [3, 12, 3, 3, 3, 0]
    assert_allclose(data["stages"][-1]["end_to_end"], 1/9)
    assert data == core.validation_experiment(per_category=3)
    fabricated = next(r for r in data["rows"] if r["category"] == "fabricated")
    assert fabricated["schema_score"] == 1
    assert fabricated["semantic_score"] < .5
    assert fabricated["passed"] == [True, True, True, True, False, None]
    malformed = next(r for r in data["rows"] if r["category"] == "malformed")
    assert malformed["passed"] == [False, None, None, None, None, None]


def test_strict_types_and_extra_policy_are_independent():
    rows = core.strictness_experiment()
    get = lambda name, strict, extra: next(r for r in rows if r["case"] == name and r["strict"] == strict and r["extra"] == extra)
    assert get("numeric string", False, "forbid")["coercion"] == ["amount_cents"]
    assert get("boolean string", False, "forbid")["coercion"] == ["approved"]
    assert not get("numeric string", True, "ignore")["accepted"]
    assert get("unexpected field", True, "ignore")["dropped"] == ["extra"]
    assert not get("unexpected field", False, "forbid")["accepted"]
    assert get("nullable note", True, "forbid")["accepted"]
    assert not get("null amount", False, "ignore")["accepted"]


@pytest.mark.parametrize("p", [0., .01, .2, .525, .99, 1.])
@pytest.mark.parametrize("n", [1, 2, 5, 10, 30])
def test_retry_probability_and_expected_attempt_bounds(p, n):
    row = core.retry_metrics(p, n)
    assert 0 <= row["success"] <= 1
    assert 1 - 1e-12 <= row["attempts"] <= n + 1e-12
    assert_allclose(row["attempts"], sum((1-p)**k for k in range(n)), rtol=1e-12)
    assert_allclose(row["success"] + row["exhaustion"], 1)
    assert row["success"] <= core.retry_metrics(p, n + 1)["success"]
    if p > 0:
        assert_allclose(row["cost_per_accepted"], .02 / p)
    else:
        assert row["cost_per_accepted"] is None


@pytest.mark.parametrize("p,target", [(0,.95), (0,0), (1,1), (.2,1), (.525,.95), (.01,.99), (.9,.9)])
def test_minimum_integer_attempt_budget(p, target):
    n = core.minimum_attempts(p, target)
    if n is None:
        assert p == 0 or target == 1
    else:
        assert n >= 1 and core.retry_metrics(p, n)["success"] >= target - 1e-14
        if n > 1:
            assert core.retry_metrics(p, n - 1)["success"] < target


@pytest.mark.parametrize("p,n", [(-.1,1), (1.1,1), (np.nan,1), (.5,0), (.5,True), (.5,1.5)])
def test_retry_invalid_inputs(p, n):
    with pytest.raises(ValueError):
        core.retry_metrics(p, n)


def test_monte_carlo_matches_each_correct_analytical_model():
    sim = core.retry_simulation(samples=100_000)
    for row in sim["results"]:
        # Six standard errors plus one discrete outcome: conservative for 30
        # correlated budget comparisons, not a fitted benchmark tolerance.
        assert abs(row["simulated_success"] - row["analytical_success"]) <= 6 * row["success_se"] + 1/sim["samples"]
        assert abs(row["simulated_attempts"] - row["analytical_attempts"]) <= 6 * row["attempts_se"] + 1/sim["samples"]
    mixture = [r for r in sim["results"] if r["scenario"] == "persistent mixture"]
    assert mixture[-1]["analytical_success"] < .75
    first = [r["analytical_success"] for r in sim["results"] if r["n"] == 1]
    assert_allclose(first, [.525] * 3)
    assert core.retry_simulation(samples=100) == core.retry_simulation(samples=100)


def test_retry_controller_and_permanent_preflight_outcomes():
    rows = core.retry_workflows()
    assert all(row["number"] <= row["budget"] for row in rows)
    permanent = [row for row in rows if row["code"] == "permanent"]
    assert len(permanent) == 2
    assert all(row["number"] == 0 and row["cost"] == 0 for row in permanent)
    exhaustion = [row for row in rows if row["scenario"] == "exhaustion"]
    assert len(exhaustion) == 3 and exhaustion[-1]["final"] == "exhausted"


@pytest.mark.parametrize("configured,authorized", [(False, True), (True, False)])
def test_preflight_failures_prevent_any_generation_call(configured, authorized):
    def generate(*args):
        pytest.fail("Permanent preflight error must prevent generation")

    with pytest.raises(core.PermanentError):
        core.guarded_extraction(generate, core.synthetic_reference(), configured=configured, authorized=authorized)


def test_valid_tool_proposal_requires_independent_permission():
    rows = core.guardrail_paths()
    denied = next(r for r in rows if not r["authorized"])
    assert all(denied["validation"])
    assert denied["stop"] == 9 and not denied["executed"]
    assert [r["attempt"] for r in rows if r["scenario"] == "retry exhaustion"] == [1, 2, 3]
