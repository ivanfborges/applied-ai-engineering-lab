"""Deterministic fixtures, restricted JSON grammar and synthetic retry experiments."""

from dataclasses import asdict, dataclass
import json
import math
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from example import Invoice, OutputError, _finite_float, _reject_constant, _unique_object
from example import Response, extract_with_retries, synthetic_reference
from from_scratch import constrained_softmax


SEED = 777
DEMOS = ("space", "decoding", "masking", "grammar", "validation", "strictness",
         "semantics", "retries", "reliability", "surface", "cost", "correlation", "guardrails")
COLORS = {"rejected": "#C44E52", "pending": "#D89021", "allowed": "#0072B2",
          "accepted": "#00866A", "evidence": "#7859A6", "ink": "#243746"}
TOKENS = ('{', '}', ':', ',', '"document_type"', '"amount"', '"currency"',
          '"invoice"', '"receipt"', '1000', '1500', '2000', '"BRL"', '"USD"',
          'true', 'null', '"1500"', '<EOS>')
FIELDS = ('"document_type"', '"amount"', '"currency"')
VALUES = (('"invoice"', '"receipt"'), ('1000', '1500', '2000'), ('"BRL"', '"USD"'))
STAGES = ("JSON parsing", "Schema / types", "Domain constraints",
          "Cross-field consistency", "Source evidence", "Application acceptance")


@dataclass(frozen=True)
class GrammarState:
    phase: str = "start"
    field: int = 0


def outgoing(state: GrammarState) -> tuple[str, ...]:
    """A fixed-order, three-property schema over whole toy tokens."""
    return {
        "start": ('{',), "property": (FIELDS[state.field],), "colon": (':',),
        "value": VALUES[state.field],
        "separator": ('}',) if state.field == 2 else (',',),
        "completed": ('<EOS>',), "terminated": (),
    }[state.phase]


def advance(state: GrammarState, token: str) -> GrammarState:
    if token not in outgoing(state):
        raise ValueError(f"Forbidden {token!r} in {state.phase}, field={state.field}")
    phases = {"start": "property", "property": "colon", "colon": "value",
              "value": "separator", "completed": "terminated"}
    if state.phase == "separator":
        return GrammarState("completed", 2) if token == '}' else GrammarState("property", state.field + 1)
    return GrammarState(phases[state.phase], state.field)


def masking(logits, allowed, temperature: float = 1.0) -> dict:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    scores = np.asarray(logits, dtype=float)
    with np.errstate(over="ignore"):
        scaled = scores / temperature
    if not np.isfinite(scaled).all():
        raise ValueError("scaled logits must be finite")
    mask = np.asarray(allowed)
    original = constrained_softmax(scaled, np.ones(scores.shape, dtype=bool))
    constrained = constrained_softmax(scaled, mask)
    return {"logits": scores.tolist(), "scaled_logits": scaled.tolist(),
            "mask": mask.tolist(), "original": original.tolist(),
            "masked_logits": [float(v) if keep else None for v, keep in zip(scaled, mask)],
            "constrained": constrained.tolist(), "mass": float(original[mask].sum())}


def decoding_trace(seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    state = GrammarState()
    prefix = ""
    rows = []
    while state.phase != "terminated":
        allowed = outgoing(state)
        logits = np.sin(np.arange(len(TOKENS)) * .8 + len(rows) * .4)
        logits[10] += 1.5
        mask = np.array([token in allowed for token in TOKENS])
        probabilities = masking(logits, mask)
        selected = str(rng.choice(TOKENS, p=probabilities["constrained"]))
        rows.append({"step": len(rows) + 1, "prefix": prefix, "phase": state.phase,
                     "field": state.field, "allowed": list(allowed), "selected": selected,
                     **probabilities})
        state = advance(state, selected)
        if selected != '<EOS>':
            prefix += selected
    json.loads(prefix)
    return rows


def grammar_rejections() -> list[dict]:
    trials = [(GrammarState("property", 0), '{', '<EOS>', "Premature termination"),
              (GrammarState("property", 1), '{"document_type":"invoice",', '}', "Missing required amount and currency"),
              (GrammarState("value", 1), '{"document_type":"invoice","amount":', 'true', "Incompatible boolean value")]
    rows = []
    for state, prefix, token, label in trials:
        try:
            advance(state, token)
        except ValueError as exc:
            rows.append({"phase": state.phase, "field": state.field, "prefix": prefix,
                         "selected": token, "allowed": list(outgoing(state)), "trial": label,
                         "error": str(exc), "state_unchanged": True})
        else:
            raise AssertionError("Intended grammar counterexample unexpectedly passed")
    return rows


class ShapeInvoice(BaseModel):
    """Bounds deliberately excluded so domain rejection is independently visible."""
    model_config = ConfigDict(strict=True, extra="forbid")
    schema_version: Literal["1"]
    document_id: str
    document_type: Literal["invoice"]
    currency: Literal["BRL", "USD", "EUR"]
    subtotal_cents: int
    tax_cents: int
    total_cents: int
    purchase_order: str | None


def fixture_population(seed: int = SEED, per_category: int = 12) -> list[dict]:
    if type(per_category) is not int or per_category < 1:
        raise ValueError("per_category must be a positive integer")
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(per_category):
        expected = synthetic_reference().model_dump()
        expected.update(document_id=f"SYN-{index}", subtotal_cents=100_000 + 100 * index,
                        total_cents=105_000 + 100 * index)
        variants = {
            "malformed": '{"document_id":',
            "missing": {k: v for k, v in expected.items() if k != "currency"},
            "wrong type": expected | {"subtotal_cents": str(expected["subtotal_cents"])},
            "extra field": expected | {"unrequested": "ignore this"},
            "invalid enum": expected | {"currency": "JPY"},
            "negative range": expected | {"subtotal_cents": -1},
            "arithmetic": expected | {"total_cents": expected["total_cents"] + 1},
            "fabricated": expected | {"document_id": f"FAB-{index}", "currency": "USD",
                                        "subtotal_cents": expected["subtotal_cents"] + 10_000,
                                        "tax_cents": expected["tax_cents"] + 1_000,
                                        "total_cents": expected["total_cents"] + 11_000},
            "correct": expected.copy(),
        }
        for category, value in variants.items():
            raw = value if isinstance(value, str) else json.dumps(value)
            rows.append({"id": f"{category}-{index}", "category": category, "raw": raw,
                         "expected": expected, "position_noise": rng.normal(0, .16, 3).tolist()})
    return rows


def evaluate_fixture(row: dict) -> dict:
    """Sequential independent gates; unreached gates are None, not failures."""
    passed = [None] * len(STAGES)
    data = None
    reason = "accepted"
    try:
        data = json.loads(row["raw"], object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant, parse_float=_finite_float)
        passed[0] = True
    except ValueError:
        passed[0], reason = False, "malformed JSON"
    if passed[0]:
        try:
            ShapeInvoice.model_validate(data)
            passed[1] = True
        except ValidationError as exc:
            passed[1], reason = False, exc.errors(include_url=False)[0]["type"]
    if passed[1]:
        try:
            Invoice.model_validate(data)
            passed[2] = True
        except ValidationError as exc:
            passed[2], reason = False, exc.errors(include_url=False)[0]["type"]
    if passed[2]:
        passed[3] = data["subtotal_cents"] + data["tax_cents"] == data["total_cents"]
        if not passed[3]:
            reason = "subtotal + tax != total"
    if passed[3]:
        passed[4] = data == row["expected"]
        if not passed[4]:
            reason = "synthetic reference mismatch"
    if passed[4]:
        passed[5] = True
    # Scores are explicit fractions of checks; neither is a probability.
    shape_checks = [isinstance(data, dict)]
    reference = row["expected"]
    if isinstance(data, dict):
        shape_checks += [set(data) == set(reference)]
        for key, value in reference.items():
            actual = data.get(key)
            shape_checks.append(key in data and
                                (actual is None or type(actual) is str) if key == "purchase_order"
                                else key in data and type(actual) is type(value))
        shape_checks += [data.get("currency") in ("BRL", "USD", "EUR"),
                         data.get("document_type") == "invoice", data.get("schema_version") == "1"]
        shape_checks += [type(data.get(k)) is int and 0 <= data[k] <= 10**12
                         for k in ("subtotal_cents", "tax_cents", "total_cents")]
        identifier = data.get("document_id")
        shape_checks += [isinstance(identifier, str) and 1 <= len(identifier) <= 32 and
                         all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-" for c in identifier)]
    else:
        shape_checks += [False] * (1 + len(reference) + 7)
    schema_score = sum(shape_checks) / len(shape_checks)
    semantics = [isinstance(data, dict) and type(data.get(k)) is type(v) and data.get(k) == v
                 for k, v in reference.items()]
    arithmetic = (isinstance(data, dict) and all(type(data.get(k)) is int for k in
                  ("subtotal_cents", "tax_cents", "total_cents")) and
                  data["subtotal_cents"] + data["tax_cents"] == data["total_cents"])
    return row | {"passed": passed, "reason": reason,
                  "schema_score": schema_score, "semantic_score": (sum(semantics) + arithmetic) / (len(semantics) + 1)}


def validation_experiment(seed: int = SEED, per_category: int = 12) -> dict:
    rows = [evaluate_fixture(row) for row in fixture_population(seed, per_category)]
    statistics = []
    for index, name in enumerate(STAGES):
        entering = sum(row["passed"][index] is not None for row in rows)
        survivors = sum(row["passed"][index] is True for row in rows)
        reasons = pd.Series([row["reason"] for row in rows if row["passed"][index] is False], dtype=str).value_counts().to_dict()
        statistics.append({"stage": name, "entering": entering, "rejected": entering - survivors,
                           "survivors": survivors, "conditional_pass": survivors / entering if entering else None,
                           "end_to_end": survivors / len(rows), "reasons": reasons})
    return {"rows": rows, "stages": statistics, "population": len(rows)}


def strictness_experiment() -> list[dict]:
    base = {"amount_cents": 1500, "approved": True, "currency": "BRL", "note": None}
    cases = {"correct": base, "numeric string": base | {"amount_cents": "1500"},
             "boolean string": base | {"approved": "true"},
             "missing amount": {k: v for k, v in base.items() if k != "amount_cents"},
             "null amount": base | {"amount_cents": None}, "nullable note": base | {"note": None},
             "unexpected field": base | {"extra": "value"}, "invalid enum": base | {"currency": "JPY"},
             "negative amount": base | {"amount_cents": -1}}
    results = []
    for strict in (False, True):
        for extra in ("ignore", "forbid"):
            model = create_model("Comparison", __config__=ConfigDict(strict=strict, extra=extra),
                                 amount_cents=(int, Field(ge=0)), approved=(bool, ...),
                                 currency=(Literal["BRL", "USD"], ...), note=(str | None, ...))
            for name, data in cases.items():
                try:
                    parsed = model.model_validate(data).model_dump()
                    coerced = [k for k in parsed if k in data and type(parsed[k]) is not type(data[k])]
                    result = {"accepted": True, "output": parsed, "coercion": coerced,
                              "dropped": sorted(set(data) - set(parsed)), "error": ""}
                except ValidationError as exc:
                    result = {"accepted": False, "output": None, "coercion": [], "dropped": [],
                              "error": "; ".join(f"{'.'.join(map(str,e['loc']))}: {e['type']}" for e in exc.errors(include_url=False))}
                results.append({"case": name, "input": data, "strict": strict, "extra": extra, **result})
    return results


def retry_metrics(p: float, n: int, cost: float = .02, duration: float = .8) -> dict:
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if not np.isfinite([cost, duration]).all() or cost < 0 or duration < 0:
        raise ValueError("cost and duration must be nonnegative and finite")
    success = 1.0 if p == 1 else -math.expm1(n * math.log1p(-p))
    attempts = n if p == 0 else success / p
    expected_cost = cost * attempts
    return {"p": p, "n": n, "success": success, "attempts": attempts,
            "cost": expected_cost, "latency": duration * attempts,
            "cost_per_accepted": expected_cost / success if success > 0 else None,
            "exhaustion": (1 - p) ** n, "marginal": p * (1 - p) ** (n - 1)}


def minimum_attempts(p: float, target: float) -> int | None:
    retry_metrics(p, 1)
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("target must be in [0, 1]")
    if target == 0:
        return 1
    if p == 0 or (target == 1 and p < 1):
        return None
    if p == 1:
        return 1
    answer = max(1, math.ceil(math.log1p(-target) / math.log1p(-p)))
    while answer > 1 and retry_metrics(p, answer - 1)["success"] >= target:
        answer -= 1
    return answer


def retry_simulation(seed: int = SEED, samples: int = 100_000, max_attempts: int = 10) -> dict:
    if type(samples) is not int or samples < 2 or samples > 1_000_000:
        raise ValueError("samples must be an integer in [2, 1000000]")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 30:
        raise ValueError("max_attempts must be an integer in [1, 30]")
    rng = np.random.default_rng(seed)
    conditional = np.minimum(.525 + .12 * np.arange(max_attempts), .95)
    persistent = rng.random(samples) < .25
    probabilities = {"independent": np.full(max_attempts, .525),
                     "persistent mixture": np.full(max_attempts, .7),
                     "feedback aware": conditional}
    results = []
    for name, chance in probabilities.items():
        draws = rng.random((samples, max_attempts)) < chance
        if name == "persistent mixture":
            draws[persistent] = False
        any_success = draws.any(axis=1)
        first = np.where(any_success, draws.argmax(axis=1) + 1, max_attempts + 1)
        for n in range(1, max_attempts + 1):
            attempts = np.minimum(first, n)
            if name == "persistent mixture":
                survival = .25 + .75 * .3 ** np.arange(n + 1)
                success = 1 - survival[n]
                expected = sum(survival[:n])
            else:
                survival = np.r_[1., np.cumprod(1 - chance[:n])]
                success, expected = 1 - survival[-1], sum(survival[:-1])
            empirical = float(np.mean(first <= n))
            results.append({"scenario": name, "n": n, "analytical_success": float(success),
                            "simulated_success": empirical, "exhaustion": 1 - empirical,
                            "analytical_attempts": float(expected), "simulated_attempts": float(attempts.mean()),
                            "success_se": math.sqrt(success * (1 - success) / samples),
                            "attempts_se": float(attempts.std(ddof=1) / math.sqrt(samples))})
    return {"seed": seed, "samples": samples, "max_attempts": max_attempts,
            "persistent_fraction": .25, "recoverable_p": .7,
            "feedback_conditional_p": conditional.tolist(), "results": results}


class PermanentError(ValueError):
    """Synthetic configuration/permission errors are not regeneration failures."""


def guarded_extraction(generate, expected, max_attempts=3, *, configured=True, authorized=True):
    if not configured:
        raise PermanentError("configuration error")
    if not authorized:
        raise PermanentError("authorization error")
    return extract_with_retries(generate, expected, max_attempts)


def retry_workflows() -> list[dict]:
    reference = synthetic_reference()
    good = Response(reference.model_dump_json())
    wrong = Response(json.dumps(reference.model_dump() | {"total_cents": 2}))
    sequences = {"first pass": [good], "format recovery": [Response("{"), good],
                 "schema then semantic": [Response("{}"), wrong, good],
                 "exhaustion": [Response("{}") for _ in range(3)]}
    rows = []
    for name, responses in sequences.items():
        outcome = guarded_extraction(lambda number, feedback: responses[number - 1], reference, 3)
        for attempt in outcome.attempts:
            rows.append({"scenario": name, **asdict(attempt), "budget": 3,
                         "feedback": f"{attempt.stage}:{attempt.code}",
                         "cost": .02 * attempt.number, "latency": .8 * attempt.number,
                         "final": outcome.status if attempt == outcome.attempts[-1] else "regenerate"})
    for name in ("configuration error", "authorization error"):
        try:
            guarded_extraction(lambda *_: good, reference,
                               configured=name != "configuration error", authorized=name != "authorization error")
        except PermanentError as exc:
            rows.append({"scenario": str(exc), "number": 0, "budget": 3, "stage": "preflight",
                         "code": "permanent", "feedback": "stop before generation", "cost": 0.,
                         "latency": 0., "final": "blocked"})
        else:
            raise AssertionError("Permanent preflight condition unexpectedly passed")
    return rows


PIPELINE = ("Source", "Input guardrails", "Prompt + schema", "LLM generation",
            "Constrained decoding", "Output parsing", "Schema validation", "Semantic validation",
            "Evidence verification", "Authorization", "Application / tool")


def guardrail_paths() -> list[dict]:
    source = fixture_population(per_category=1)
    evaluated = {row["category"]: evaluate_fixture(row) for row in source}
    paths = []
    for name, category, authorized in (("accepted invoice", "correct", True),
                                        ("wrong arithmetic", "arithmetic", True),
                                        ("missing evidence", "fabricated", True),
                                        ("valid tool call / no permission", "correct", False),
                                        ("format recovery", "malformed", True),
                                        ("retry exhaustion", "missing", True)):
        row = evaluated[category]
        if row["passed"][0] is False:
            stop, branch = 5, "Recoverable -> feedback -> regeneration"
        elif row["passed"][1] is False:
            stop, branch = 6, "Retry budget exhausted -> human review"
        elif row["passed"][3] is False:
            stop, branch = 7, "Semantic failure -> human review"
        elif row["passed"][4] is False:
            stop, branch = 8, "Missing evidence -> human review"
        elif not authorized:
            stop, branch = 9, "Authorization denied -> no tool execution"
        else:
            stop, branch = 10, "Accepted for application consumption"
        paths.append({"scenario": name, "stop": stop, "branch": branch, "authorized": authorized,
                      "validation": row["passed"], "tool": {"action": "post_invoice", "document_id": "SYN-0"},
                      "executed": stop == 10, "attempt": 1})
    recovery = next(p for p in paths if p["scenario"] == "format recovery")
    paths.insert(paths.index(recovery) + 1, paths[0] | {
        "scenario": "format recovery / regenerated", "attempt": 2,
        "branch": "Validated regenerated response -> application consumption"})
    exhaustion = paths.pop()
    for attempt in range(1, 4):
        paths.append(exhaustion | {"attempt": attempt,
                     "branch": "Schema failure -> feedback -> regeneration" if attempt < 3 else exhaustion["branch"]})
    return paths


def compute_lab(seed: int = SEED, samples: int = 100_000) -> dict:
    return {"decoding": decoding_trace(seed), "grammar_rejections": grammar_rejections(), "validation": validation_experiment(seed),
            "strictness": strictness_experiment(), "simulation": retry_simulation(seed, samples),
            "workflows": retry_workflows(), "guardrails": guardrail_paths(),
            "retry_grid": [retry_metrics(float(p), n) for p in np.linspace(0, 1, 41) for n in range(1, 11)]}
