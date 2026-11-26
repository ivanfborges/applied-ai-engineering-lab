"""Strict educational extraction contract and reference-based offline scoring."""

import json
from collections.abc import Callable


FIELDS = ("invoice_id", "total_usd")


def schema_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        return False
    identifier, total = value["invoice_id"], value["total_usd"]
    return (identifier is None or isinstance(identifier, str) and bool(identifier.strip())) and (
        total is None or type(total) is int and total >= 0
    )


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"Nonstandard JSON constant: {value}")


def validate_response(raw: str) -> dict:
    """Reject duplicate keys, NaN/Infinity, coercion and automatic JSON repair."""
    if not isinstance(raw, str):
        raise ValueError("raw response must be a string")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except ValueError:
        return {"json_valid": False, "schema_valid": False, "data": None}
    valid = schema_valid(value)
    return {"json_valid": True, "schema_valid": valid,
            "data": value if valid else None}


def score_response(raw: str | None, expected: dict, error: str | None = None) -> dict:
    """References are evaluation labels, never an online acceptance oracle."""
    if not schema_valid(expected):
        raise ValueError("expected must satisfy the extraction contract")
    if error is not None:
        if not isinstance(error, str) or not error.strip() or raw is not None:
            raise ValueError("an error requires a nonempty label and raw=null")
        validation = {"json_valid": False, "schema_valid": False, "data": None}
    else:
        validation = validate_response(raw)
    data = validation["data"]
    complete = data is not None and all(data[key] is not None for key in FIELDS)
    exact = data == expected
    return {**validation, "operational_error": error is not None,
            "exact_match": exact,
            "correct_fields": sum(data[key] == expected[key] for key in FIELDS)
            if data is not None else 0,
            "complete_output": complete,
            "incorrect_complete_output": complete and not exact}


def evaluate(cases: list[dict], responses: list[dict]) -> dict:
    """Require one response per case so missing/duplicate records cannot bias rates."""
    if not cases:
        raise ValueError("cases cannot be empty")
    case_ids = [case["id"] for case in cases]
    response_ids = [response["case_id"] for response in responses]
    if len(set(case_ids)) != len(case_ids) or len(set(response_ids)) != len(response_ids):
        raise ValueError("case and response IDs must be unique")
    if set(case_ids) != set(response_ids):
        raise ValueError("responses must cover exactly the evaluation case IDs")
    by_id = {response["case_id"]: response for response in responses}
    rows = []
    for case in cases:
        response = by_id[case["id"]]
        rows.append({"case_id": case["id"], "category": case["category"],
                     **score_response(response["raw"], case["expected"], response.get("error"))})
    n = len(rows)
    complete = sum(row["complete_output"] for row in rows)
    return {"n": n,
            "json_valid_rate": sum(row["json_valid"] for row in rows) / n,
            "schema_valid_rate": sum(row["schema_valid"] for row in rows) / n,
            "exact_match_rate": sum(row["exact_match"] for row in rows) / n,
            "field_accuracy": sum(row["correct_fields"] for row in rows) / (n * len(FIELDS)),
            "operational_errors": sum(row["operational_error"] for row in rows),
            "complete_output_coverage": complete / n,
            "selective_risk_among_complete":
            sum(row["incorrect_complete_output"] for row in rows) / complete if complete else None,
            "rows": rows}


def request_with_validation(generate: Callable[[], str], max_attempts: int = 2) -> dict:
    """Retry malformed outputs or timeouts, stopping at the first schema-valid result.

    No semantic oracle is used. Other exceptions propagate. This controller has
    no backoff, deadline, provider adapter or production acceptance policy.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    trace = []
    for _ in range(max_attempts):
        try:
            validation = validate_response(generate())
        except TimeoutError:
            trace.append({"status": "timeout"})
            continue
        trace.append({"status": "schema_valid" if validation["schema_valid"] else "invalid_output"})
        if validation["schema_valid"]:
            return {"schema_valid": True, "data": validation["data"], "trace": trace}
    return {"schema_valid": False, "data": None, "trace": trace}
