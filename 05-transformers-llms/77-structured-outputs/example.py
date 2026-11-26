"""Offline synthetic invoice validation and bounded recovery; no LLM calls."""

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Invoice(BaseModel):
    """Versioned, strict application contract; arithmetic is checked separately."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    schema_version: Literal["1"]
    document_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Z0-9-]+$")
    document_type: Literal["invoice"]
    currency: Literal["BRL", "USD", "EUR"]
    subtotal_cents: int = Field(ge=0, le=10**12)
    tax_cents: int = Field(ge=0, le=10**12)
    total_cents: int = Field(ge=0, le=10**12)
    purchase_order: str | None = Field(min_length=1, max_length=64)


class OutputError(ValueError):
    def __init__(self, stage: str, code: str):
        super().__init__(code)
        self.stage = stage
        self.code = code


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non-JSON constant")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite number")
    return number


def parse_invoice(raw: str) -> Invoice:
    """Reject ambiguous serialization before strict Python-object validation."""
    if not isinstance(raw, str):
        raise TypeError("raw response must be a string")
    if len(raw) > 16_384:
        raise OutputError("guardrail", "response_too_long")
    try:
        data = json.loads(
            raw, object_pairs_hook=_unique_object,
            parse_constant=_reject_constant, parse_float=_finite_float,
        )
    except (ValueError, RecursionError) as exc:
        raise OutputError("parse", "invalid_json") from exc
    try:
        return Invoice.model_validate(data)
    except ValidationError as exc:
        # Return bounded diagnostics without echoing untrusted values or references.
        raise OutputError("schema", "contract_violation") from exc


def check_business_rules(invoice: Invoice) -> None:
    if invoice.subtotal_cents + invoice.tax_cents != invoice.total_cents:
        raise OutputError("business", "inconsistent_total")


def check_reference(invoice: Invoice, expected: Invoice) -> None:
    """Exact-match oracle for synthetic fixtures, not an inference-time verifier."""
    if invoice != expected:
        raise OutputError("reference", "reference_mismatch")


@dataclass(frozen=True)
class Response:
    text: str = ""
    status: Literal["complete", "refusal", "incomplete"] = "complete"


@dataclass(frozen=True)
class Attempt:
    number: int
    stage: str
    code: str


@dataclass(frozen=True)
class Outcome:
    status: str
    invoice: Invoice | None
    attempts: tuple[Attempt, ...]


def extract_with_retries(
    generate: Callable[[int, str | None], Response],
    expected: Invoice,
    max_attempts: int = 4,
) -> Outcome:
    """Budget counts all calls. Retry parse/schema/business errors only."""
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    check_business_rules(expected)
    history = []
    feedback = None
    for number in range(1, max_attempts + 1):
        response = generate(number, feedback)
        if response.status not in {"complete", "refusal", "incomplete"}:
            raise ValueError("unknown response status")
        if response.status != "complete":
            history.append(Attempt(number, "provider", response.status))
            return Outcome(response.status, None, tuple(history))
        try:
            invoice = parse_invoice(response.text)
            check_business_rules(invoice)
            check_reference(invoice, expected)
        except OutputError as exc:
            history.append(Attempt(number, exc.stage, exc.code))
            if exc.stage in {"reference", "guardrail"}:
                return Outcome("rejected", None, tuple(history))
            feedback = f"{exc.stage}:{exc.code}"
        else:
            history.append(Attempt(number, "accepted", "all_checks_passed"))
            return Outcome("accepted", invoice, tuple(history))
    return Outcome("exhausted", None, tuple(history))


def synthetic_reference() -> Invoice:
    return Invoice(
        schema_version="1", document_id="SYN-77", document_type="invoice",
        currency="BRL", subtotal_cents=100_000, tax_cents=5_000,
        total_cents=105_000, purchase_order=None,
    )


def run_demo() -> dict:
    expected = synthetic_reference()
    valid = expected.model_dump()
    string_amount = dict(valid, subtotal_cents="100000")
    inconsistent = dict(valid, total_cents=105_001)
    fabricated = dict(valid, subtotal_cents=200_000, total_cents=205_000)
    scenarios = {
        "repair": [Response('{"document_id":'), Response(json.dumps(string_amount)),
                   Response(json.dumps(inconsistent)), Response(json.dumps(valid))],
        "fabricated": [Response(json.dumps(fabricated)), Response(json.dumps(valid))],
        "exhaustion": [Response("{}") for _ in range(4)],
        "refusal": [Response(status="refusal"), Response(json.dumps(valid))],
        "incomplete": [Response(expected.model_dump_json(), "incomplete")],
    }
    results = {}
    for name, responses in scenarios.items():
        def generate(number, feedback):
            return responses[number - 1]

        outcome = extract_with_retries(generate, expected, max_attempts=4)
        results[name] = {
            "status": outcome.status, "calls": len(outcome.attempts),
            "attempts": [asdict(attempt) for attempt in outcome.attempts],
            "invoice": outcome.invoice.model_dump() if outcome.invoice else None,
        }
    return {
        "synthetic": True, "author_review_required": True,
        "configuration": {"max_attempts": 4, "reference": expected.model_dump()},
        "schema": Invoice.model_json_schema(), "scenarios": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="write a new ignored JSON record")
    args = parser.parse_args()
    report = run_demo()
    now = datetime.now(timezone.utc)
    report["executed_at_utc"] = now.isoformat()
    report["versions"] = {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in ("pydantic", "numpy")},
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.save:
        output = Path(__file__).resolve().parent / "outputs" / f"run-{now:%Y%m%dT%H%M%S%fZ}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(f"Saved {output.name} under ignored outputs/.")


if __name__ == "__main__":
    main()
