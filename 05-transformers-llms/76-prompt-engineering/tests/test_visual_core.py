"""Independent numerical and control-flow checks, without real inference."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visual_core as core
import visual_real as real


def test_entropy_known_values_and_confident_wrong_counterexample():
    assert core.entropy([1, 0, 0, 0]) == 0
    assert core.entropy([.25] * 4) == pytest.approx(2)
    assert core.entropy([.5, .5]) == pytest.approx(1)
    rows = core.conditioning()
    assert all(sum(row["probabilities"]) == pytest.approx(1) for row in rows)
    assert rows[-1]["winner"] != "Billing"
    assert rows[-1]["entropy_bits"] < rows[-2]["entropy_bits"]
    assert rows[-1]["p_correct"] < rows[-2]["p_correct"]


@pytest.mark.parametrize("p", [[0, 0], [.2, .2], [-.1, 1.1], [float("nan"), 1], [[1]]])
def test_entropy_rejects_invalid_distributions(p):
    with pytest.raises(ValueError):
        core.entropy(p)


def test_nearest_demo_order_invariance_including_ties():
    points = np.array([[0., 0.], [3, 0]])
    demos = np.array([[-1., 0.], [1., 0.]])
    expected = [0, 1]
    np.testing.assert_array_equal(core.nearest_demo(points, demos, [0, 1]), expected)
    np.testing.assert_array_equal(core.nearest_demo(points, demos[::-1], [1, 0]), expected)
    data = core.in_context(42, 100)
    np.testing.assert_array_equal(data["stages"][2]["prediction"], data["stages"][3]["prediction"])
    assert (data["stages"][2]["prediction"] == data["truth"]).mean() > (data["stages"][1]["prediction"] == data["truth"]).mean()
    assert (data["stages"][4]["prediction"] == 0).all()


def test_authored_ablation_metrics_are_computed_from_actual_responses():
    data = core.synthetic_ablation()
    assert len(data["variants"]) == 6 and len(data["cases"]) == 12
    for variant in data["variants"]:
        metrics = core.extra_metrics(data["cases"], variant["responses"])
        assert metrics == variant["metrics"]
        assert metrics["attack_n"] == 1
        assert metrics["n"] == 12
        assert metrics["exact_match_rate"] <= metrics["schema_valid_rate"] <= metrics["json_valid_rate"]
    structured = data["variants"][1]
    assert structured["metrics"]["exact_match_rate"] == pytest.approx(10 / 12)
    assert structured["metrics"]["schema_valid_rate"] == 1


def test_optimization_equation_and_context_penalty():
    row = core.optimization(np.array([0., 1.]), np.array([0., 16.]))
    assert row["accuracy"][0] == pytest.approx(.4)
    assert row["tokens"][0] == 120
    assert row["latency"][0] == pytest.approx(.15 + .0008 * 120)
    assert core.optimization(.8, 16)["accuracy"] < core.optimization(.8, 6)["accuracy"]
    np.testing.assert_allclose(row["utility"], row["accuracy"] - .00002 * row["tokens"] - .03 * row["latency"])
    with pytest.raises(ValueError):
        core.optimization(2, 3)


def test_generalization_uses_development_only_for_selection():
    data = core.generalization(42, 80, 12)
    for row in data.values():
        assert np.all(np.diff(row["development"]) >= 0)
        assert np.all(row["selected"] <= np.arange(80))
        np.testing.assert_allclose(row["gap"], row["development"] - row["holdout"])
    np.testing.assert_allclose(data["Noise selection"]["true_probability"], .72)
    assert data["Genuine improvement"]["true_probability"][-1] > .55


def test_validation_oracle_and_business_gate_are_separate():
    rows = core.validation_pipeline()
    assert rows[0]["stages"][:3] == [False, None, None]
    assert rows[1]["stages"][:3] == [True, False, None]
    assert rows[2]["stages"][:3] == [True, True, False]
    assert rows[4]["stages"][:4] == [True, True, True, False]
    assert rows[4]["decision"] == "human review"
    assert rows[5]["decision"] == "accept"
    assert rows[6]["stages"][:4] == [True, True, True, False]


def test_selective_metrics_hand_calculation_and_zero_coverage():
    confidence, correct = [.2, .6, .8, .9], [1, 0, 1, 0]
    m = core.selective_metrics(confidence, correct, .6)
    assert m["accepted"] == 3 and m["review"] == 1
    assert m["coverage"] == .75 and m["abstention_rate"] == .25
    assert m["errors"] == 2 and m["correct_accepted"] == 1
    assert m["risk"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(1 / 3)
    empty = core.selective_metrics(confidence, correct, 1)
    assert empty["risk"] is None and empty["accuracy"] is None
    assert empty["coverage"] == 0 and empty["review"] == 4
    assert core.operational_loss(confidence, correct, .6, 2, 10) == pytest.approx(22 / 4)


@pytest.mark.parametrize("scores,correct,threshold", [([], [], .5), ([.2], [1, 0], .5), ([1.1], [1], .5), ([.2], [2], .5), ([.2], [0], -1), ([float("nan")], [1], .5)])
def test_invalid_selective_inputs(scores, correct, threshold):
    with pytest.raises(ValueError):
        core.selective_metrics(scores, correct, threshold)


def test_calibration_bins_include_probability_one_and_report_empty_bins():
    bins = core.calibration_bins(np.array([0, .5, 1]), np.array([0, 1, 1]), 4)
    assert sum(row["n"] for row in bins) == 3
    assert bins[1] == {"n": 0, "confidence": None, "accuracy": None}
    assert bins[-1]["confidence"] == 1 and bins[-1]["accuracy"] == 1


def test_reliability_reproducible_with_declared_miscalibration():
    a = core.reliability(42, 200)
    b = core.reliability(42, 200)
    for name in a:
        np.testing.assert_array_equal(a[name]["confidence"], b[name]["confidence"])
        np.testing.assert_array_equal(a[name]["correct"], b[name]["correct"])
    calibrated, reversed_score = a.values()
    np.testing.assert_allclose(reversed_score["confidence"], .99 - .8 * calibrated["confidence"])


def test_pareto_dominance_directions_and_duplicates():
    # B dominates A; C trades accuracy for cost/latency; B duplicates both survive.
    np.testing.assert_array_equal(core.pareto_mask([2, 1, .5, 1], [2, 1, 3, 1], [.8, .9, .6, .9]), [False, True, True, True])
    with pytest.raises(ValueError):
        core.pareto_mask([1], [float("nan")], [.9])


def test_configurable_hypothetical_prices_and_pareto():
    free = core.configurations(42, 10, 0, 0)
    priced = core.configurations(42, 10, 1, 2)
    assert (free["cost"] == 0).all()
    np.testing.assert_allclose(priced["cost"], (priced["input_tokens"] + 2 * priced["output_tokens"]) / 1e6)
    np.testing.assert_array_equal(priced["accuracy"], free["accuracy"])


def test_state_machine_retry_deadline_and_permanent_failure():
    scenarios = core.failure_scenarios()
    rows = {name: core.run_pipeline(attempts) for name, attempts in scenarios.items()}
    assert rows["JSON recovery"]["outcome"] == "accepted"
    assert rows["JSON recovery"]["calls"] == 2
    assert rows["Transient API failure"]["outcome"] == "accepted"
    assert rows["Persistent schema failure"]["calls"] == 3
    assert rows["Semantic mismatch"]["calls"] == 1
    assert rows["Permanent configuration failure"]["calls"] == 1
    assert rows["Missing evidence"]["outcome"] == "human review"
    assert rows["Latency budget exhausted"]["calls"] == 1
    assert all(row["elapsed"] <= 1.2 + 1e-12 and row["calls"] <= 3 for row in rows.values())


def test_iid_retry_formula_and_correlated_counterexample():
    row = core.retry_tradeoff(3, .5)
    assert row["iid_success"] == .875
    assert row["iid_expected_calls"] == 1.75
    assert row["correlated_success"] == .5
    assert row["correlated_expected_calls"] == 2


def test_real_budget_is_checked_before_network_or_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="worst-case budget"):
        real.collect(tmp_path, repetitions=1, max_attempts=1, max_requests=1,
                     case_limit=12, temperature=0, max_tokens=100, timeout=3)
    with pytest.raises(ValueError, match="requires GEMINI_API_KEY"):
        real.collect(tmp_path, repetitions=1, max_attempts=1, max_requests=72,
                     case_limit=12, temperature=0, max_tokens=100, timeout=3)


def recording():
    row = {"case_id": "a", "category": "complete", "expected": {"invoice_id": "A", "total_usd": 1},
           "prompt_version": "V2", "repetition": 0, "attempts": [
               {"raw": "bad JSON", "error": None, "latency_seconds": .2, "input_tokens": None, "output_tokens": None},
               {"raw": '{"invoice_id":"A","total_usd":99}', "error": None, "latency_seconds": .3, "input_tokens": 10, "output_tokens": 5}],
           "score": {"exact_match": True}}
    return {"schema_version": 1, "evidence_type": "test_recording", "model": "fake-not-a-model", "records": [row]}


def test_replay_recomputes_forged_score_and_preserves_missing_usage(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(recording()), encoding="utf-8")
    # Output directory creation belongs to CLI; exercise the actual replay boundary.
    output = tmp_path / "replay"
    output.mkdir()
    result = real.replay(source, output)
    group = result["summary"]["groups"][0]
    assert group["metrics"]["exact_match_rate"] == 0
    assert group["first_attempt_metrics"]["schema_valid_rate"] == 0
    assert group["metrics"]["schema_valid_rate"] == 1
    assert group["usage"]["input_tokens"] is None
    assert group["cost_usd"] is None
    assert group["mean_total_latency_seconds"] == .5 and group["retries"] == 1
    assert "TEST RECORDING" in result["html"].read_text(encoding="utf-8")


def test_fake_sdk_collect_has_bounded_calls_and_does_not_record_key(monkeypatch, tmp_path):
    calls = []
    cases = core.evaluation_cases()[:2]
    lookup = {case["document"]: case["expected"] for case in cases}

    def generate_content(**kwargs):
        calls.append(kwargs)
        doc = json.loads(kwargs["contents"])["document"]
        return SimpleNamespace(text=json.dumps(lookup[doc]), usage_metadata=None)

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["http_options"].retry_options.attempts == 1
            self.models = SimpleNamespace(generate_content=generate_content)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    types = SimpleNamespace(HttpOptions=lambda **kw: SimpleNamespace(**kw),
                            HttpRetryOptions=lambda **kw: SimpleNamespace(**kw),
                            GenerateContentConfig=lambda **kw: SimpleNamespace(**kw))
    fake_genai = SimpleNamespace(Client=Client, types=types)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types)
    monkeypatch.setenv("GEMINI_API_KEY", "TEST-SECRET-NEVER-LOG")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setattr(real, "version", lambda _: "test-sdk")
    path = real.collect(tmp_path, repetitions=1, max_attempts=1, max_requests=12,
                        case_limit=2, temperature=0, max_tokens=100, timeout=3)
    assert len(calls) == 12
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "completed" and len(record["records"]) == 12
    assert "TEST-SECRET-NEVER-LOG" not in path.read_text(encoding="utf-8")
