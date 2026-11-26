"""Application boundaries, retry budgets and numerical masking invariants."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest
from pydantic import ValidationError


TOPIC = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"day77_{name}", TOPIC / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


app = load_module("example")
core = load_module("from_scratch")


@pytest.mark.parametrize("raw", [
    '{"total_cents":', '```json\n{}\n```', '{} trailing',
    '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e999}',
    '{"x":1,"x":2}', '{"nested":{"x":1,"x":2}}',
])
def test_invalid_serialization(raw):
    with pytest.raises(app.OutputError) as error:
        app.parse_invoice(raw)
    assert error.value.stage == "parse"


@pytest.mark.parametrize("value", ["100000", 100000.0, True, -1, None, 10**12 + 1])
def test_strict_cents_reject_invalid_values(value):
    data = app.synthetic_reference().model_dump()
    data["subtotal_cents"] = value
    with pytest.raises(app.OutputError) as error:
        app.parse_invoice(json.dumps(data))
    assert error.value.stage == "schema"


@pytest.mark.parametrize("changes", [
    {"unexpected": "field"}, {"currency": "JPY"}, {"schema_version": "2"},
    {"document_id": " "}, {"purchase_order": ""}, {"document_type": "contract"},
])
def test_contract_rejects_drift(changes):
    with pytest.raises(app.OutputError) as error:
        app.parse_invoice(json.dumps(app.synthetic_reference().model_dump() | changes))
    assert error.value.stage == "schema"


@pytest.mark.parametrize("raw", ["[]", "null", "true", "42"])
def test_root_must_be_an_object(raw):
    with pytest.raises(app.OutputError) as error:
        app.parse_invoice(raw)
    assert error.value.stage == "schema"


def test_required_nullable_is_not_optional_or_zero():
    data = app.synthetic_reference().model_dump()
    assert app.parse_invoice(json.dumps(data)).purchase_order is None
    del data["purchase_order"]
    with pytest.raises(app.OutputError):
        app.parse_invoice(json.dumps(data))
    zero = app.synthetic_reference().model_dump() | {
        "subtotal_cents": 0, "tax_cents": 0, "total_cents": 0,
    }
    app.check_business_rules(app.parse_invoice(json.dumps(zero)))


def test_schema_export_cannot_replace_cross_field_or_reference_checks():
    schema = app.Invoice.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(app.Invoice.model_fields)
    assert schema["properties"]["subtotal_cents"]["type"] == "integer"
    assert schema["properties"]["subtotal_cents"]["minimum"] == 0
    assert {item["type"] for item in schema["properties"]["purchase_order"]["anyOf"]} == {"string", "null"}
    inconsistent = app.parse_invoice(json.dumps(app.synthetic_reference().model_dump() | {"total_cents": 1}))
    with pytest.raises(app.OutputError, match="inconsistent_total"):
        app.check_business_rules(inconsistent)
    fabricated = app.parse_invoice(json.dumps(app.synthetic_reference().model_dump() | {
        "subtotal_cents": 200_000, "total_cents": 205_000,
    }))
    app.check_business_rules(fabricated)
    with pytest.raises(app.OutputError, match="reference_mismatch"):
        app.check_reference(fabricated, app.synthetic_reference())


def test_budget_feedback_and_early_success():
    calls = []
    responses = [app.Response("{}"), app.Response(app.synthetic_reference().model_dump_json())]

    def generate(number, feedback):
        calls.append((number, feedback))
        return responses[number - 1]

    result = app.extract_with_retries(generate, app.synthetic_reference(), max_attempts=5)
    assert result.status == "accepted"
    assert result.invoice == app.synthetic_reference()
    assert calls == [(1, None), (2, "schema:contract_violation")]


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_exhaustion_has_exact_budget_and_no_invoice(budget):
    calls = []

    def generate(number, feedback):
        calls.append(number)
        return app.Response("{}")

    result = app.extract_with_retries(generate, app.synthetic_reference(), budget)
    assert result.status == "exhausted" and result.invoice is None
    assert calls == list(range(1, budget + 1))
    assert len(result.attempts) == budget


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "2"])
def test_invalid_budget_does_not_call_provider(budget):
    def generate(*args):
        pytest.fail("provider must not run")

    with pytest.raises(ValueError, match="positive integer"):
        app.extract_with_retries(generate, app.synthetic_reference(), budget)


@pytest.mark.parametrize("response,status,stage", [
    (app.Response(status="refusal"), "refusal", "provider"),
    (app.Response(app.synthetic_reference().model_dump_json(), "incomplete"), "incomplete", "provider"),
    (app.Response(" " * 16_385), "rejected", "guardrail"),
    (app.Response(json.dumps(app.synthetic_reference().model_dump() | {
        "subtotal_cents": 200_000, "total_cents": 205_000,
    })), "rejected", "reference"),
])
def test_terminal_conditions_never_consume_a_second_response(response, status, stage):
    def generate(number, feedback):
        assert number == 1 and feedback is None
        return response

    result = app.extract_with_retries(generate, app.synthetic_reference())
    assert result.status == status and result.invoice is None
    assert result.attempts[0].stage == stage


def test_programming_and_transport_errors_are_not_swallowed():
    def broken(*args):
        raise RuntimeError("transport failure")

    with pytest.raises(RuntimeError, match="transport failure"):
        app.extract_with_retries(broken, app.synthetic_reference())
    with pytest.raises(ValueError, match="unknown response status"):
        app.extract_with_retries(lambda *args: app.Response(status="unknown"), app.synthetic_reference())
    with pytest.raises(TypeError, match="string"):
        app.parse_invoice(None)
    with pytest.raises(ValidationError):
        app.synthetic_reference().total_cents = 1


def test_demo_stages():
    results = app.run_demo()["scenarios"]
    assert [a["stage"] for a in results["repair"]["attempts"]] == ["parse", "schema", "business", "accepted"]
    assert results["fabricated"]["calls"] == 1
    assert results["exhaustion"]["calls"] == 4


def test_mask_matches_renormalized_probabilities():
    scores = np.array([1.2, 0.8, 2.0, -0.2, 0.1])
    allowed = np.array([True, True, False, False, False])
    original = core.constrained_softmax(scores, np.ones(5, dtype=bool))
    constrained = core.constrained_softmax(scores, allowed)
    assert_allclose(constrained[allowed], original[allowed] / original[allowed].sum())
    assert np.all(constrained[~allowed] == 0)
    assert_allclose(constrained.sum(), 1)
    assert_allclose(core.constrained_softmax(scores + 1000, allowed), constrained)


def test_extreme_logits_and_forbidden_maximum():
    assert_allclose(core.constrained_softmax([1e308, -1e308], [True, True]), [1, 0])
    assert_allclose(core.constrained_softmax([1e308, -1e308], [False, True]), [0, 1])
    assert_allclose(core.constrained_softmax([0, 0, 0], [True, False, True]), [.5, 0, .5])


@pytest.mark.parametrize("scores,mask", [
    ([], []), ([[1, 2]], [[True, True]]), ([1, 2], [True]),
    ([1, 2], [1, 0]), ([1, 2], [False, False]),
    ([np.inf], [True]), ([np.nan], [True]), ([-np.inf], [True]),
])
def test_invalid_masking_inputs(scores, mask):
    with pytest.raises(ValueError):
        core.constrained_softmax(scores, mask)


@pytest.mark.parametrize("script", ["example.py", "from_scratch.py"])
def test_scripts_work_from_another_directory(script, tmp_path):
    run = subprocess.run([sys.executable, str(TOPIC / script)], cwd=tmp_path,
                         text=True, capture_output=True, check=True)
    assert json.loads(run.stdout)["synthetic"] is True
