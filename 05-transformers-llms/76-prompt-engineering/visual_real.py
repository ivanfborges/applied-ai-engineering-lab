"""Explicit opt-in Gemini runner and offline replay of its recorded responses.

Optional SDK: google-genai (https://googleapis.github.io/python-genai/).
No import of the SDK or environment access occurs during simulation/replay.
Install the optional SDK separately only when real mode is needed; the repository
currently has no canonical Gemini extra. No dependency files outside Day 76 are
modified by this lab. Real mode records only synthetic documents from this topic.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
from time import perf_counter

import plotly.graph_objects as go

from evaluation import score_response, validate_response
from visual_core import evaluation_cases, extra_metrics, prompt_variants
from visual_html import card, state, write_page


def validate_budget(case_count: int, variants: int, repetitions: int, max_attempts: int, max_requests: int):
    if any(type(value) is not int or value < 1 for value in [case_count, variants, repetitions, max_attempts, max_requests]):
        raise ValueError("request counts, repetitions and attempts must be positive integers")
    if repetitions > 10 or max_attempts > 5 or max_requests > 500:
        raise ValueError("limits: repetitions <=10, attempts <=5, requests <=500")
    required = case_count * variants * repetitions * max_attempts
    if required > max_requests:
        raise ValueError(f"Balanced evaluation needs a worst-case budget of {required} requests; increase --max-requests or reduce --case-limit/--repetitions/--max-attempts")


def collect(output: Path, *, repetitions: int, max_attempts: int, max_requests: int,
            case_limit: int, temperature: float, max_tokens: int, timeout: float) -> Path:
    """Run only when CLI --mode real is explicit; write progress after every case."""
    cases = evaluation_cases()[:case_limit]
    variants = prompt_variants()
    validate_budget(len(cases), len(variants), repetitions, max_attempts, max_requests)
    key, model = os.environ.get("GEMINI_API_KEY"), os.environ.get("GEMINI_MODEL")
    if not key or not model:
        raise ValueError("Real mode requires GEMINI_API_KEY and GEMINI_MODEL; no calls were made")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Optional real mode requires google-genai; install with python -m pip install google-genai. Simulation/replay use existing repository dependencies.") from exc
    generation = {"temperature": temperature, "max_output_tokens": max_tokens}
    record = {"schema_version": 1, "evidence_type": "real_llm", "model": model,
              "sdk_version": version("google-genai"), "generation": generation,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "hypothesis": "Compare fixed prompt configurations on identical synthetic development cases; no population conclusion.",
              "interpretation_status": "author review required", "status": "running",
              "limitation": "Small synthetic development set; no holdout, cost estimates or probability measurements.",
              "request_limit": max_requests, "sdk_attempts_per_call": 1, "records": []}
    path = output / "results" / "real-evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    checkpoint()
    requests = 0
    stop = False
    # Millisecond timeout; SDK internal retries disabled so explicit call budget is meaningful.
    with genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=int(timeout * 1000), retry_options=types.HttpRetryOptions(attempts=1))) as client:
        for repetition in range(repetitions):
            for prompt_name, instructions in variants.items():
                for case in cases:
                    attempts = []
                    for attempt in range(max_attempts):
                        if requests >= max_requests:
                            raise RuntimeError("Explicit request budget exhausted; partial results retained")
                        requests += 1
                        start = perf_counter()
                        try:
                            response = client.models.generate_content(
                                model=model, contents=json.dumps({"document": case["document"]}),
                                config=types.GenerateContentConfig(system_instruction=instructions, **generation))
                        except Exception as exc:
                            # Do not serialize exception text/URLs, which could contain credentials.
                            code = getattr(exc, "code", None)
                            transient = isinstance(exc, TimeoutError) or code in [408, 429, 500, 502, 503, 504]
                            attempts.append({"raw": None, "error": type(exc).__name__, "http_status": code if isinstance(code, int) else None,
                                             "retryable": transient, "latency_seconds": perf_counter() - start,
                                             "input_tokens": None, "output_tokens": None})
                            if not transient:
                                stop = True
                                break
                            continue
                        raw = response.text or ""
                        usage = response.usage_metadata
                        attempts.append({"raw": raw, "error": None, "latency_seconds": perf_counter() - start,
                                         "input_tokens": getattr(usage, "prompt_token_count", None),
                                         "output_tokens": getattr(usage, "candidates_token_count", None)})
                        if validate_response(raw)["schema_valid"]:
                            break
                    final = attempts[-1]
                    record["records"].append({"case_id": case["id"], "category": case["category"], "input_text": case["document"],
                        "expected": case["expected"], "prompt_version": prompt_name, "prompt_text": instructions,
                        "generation": generation, "repetition": repetition, "attempts": attempts,
                        "retry_count": len(attempts) - 1, "score": score_response(final["raw"], case["expected"], final.get("error")),
                        "cost_usd": None, "output_probabilities": None})
                    checkpoint()
                    if stop:
                        break
                if stop:
                    break
            if stop:
                break
    record["status"] = "stopped_on_permanent_error" if stop else "completed"
    record["requests_made"] = requests
    record["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    return path


def replay(path: Path, output: Path) -> dict:
    """Recompute metrics from raw recorded responses, ignoring stored scores."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("evidence_type") not in ["real_llm", "test_recording"] or not record.get("records"):
        raise ValueError("Expected nonempty schema_version=1 real_llm/test_recording results; no simulation fallback")
    grouped = defaultdict(list)
    for row in record["records"]:
        grouped[(row["prompt_version"], row["repetition"])].append(row)
    summaries, states = [], []
    for (prompt_name, repetition), rows in grouped.items():
        cases = [{"id": row["case_id"], "category": row["category"], "expected": row["expected"]} for row in rows]
        responses = [{"case_id": row["case_id"], "raw": row["attempts"][-1]["raw"], "error": row["attempts"][-1].get("error")} for row in rows]
        metrics = extra_metrics(cases, responses)
        first_responses = [{"case_id": row["case_id"], "raw": row["attempts"][0]["raw"], "error": row["attempts"][0].get("error")} for row in rows]
        first = extra_metrics(cases, first_responses)
        durations = [sum(attempt["latency_seconds"] for attempt in row["attempts"]) for row in rows]
        usage = {}
        for key in ("input_tokens", "output_tokens"):
            values = [attempt.get(key) for row in rows for attempt in row["attempts"]]
            usage[key] = sum(values) if all(value is not None for value in values) else None
        summary = {"prompt_version": prompt_name, "repetition": repetition, "metrics": metrics,
                   "first_attempt_metrics": first, "mean_total_latency_seconds": sum(durations) / len(durations),
                   "retries": sum(len(row["attempts"]) - 1 for row in rows), "usage": usage, "cost_usd": None}
        summaries.append(summary)
        keys = ["json_valid_rate", "schema_valid_rate", "exact_match_rate", "field_accuracy", "unsupported_value_rate"]
        chart = go.Figure()
        chart.add_trace(go.Bar(x=keys, y=[first[k] for k in keys], name="First attempt"))
        chart.add_trace(go.Bar(x=keys, y=[metrics[k] for k in keys], name="Final attempt"))
        chart.update_layout(title=prompt_name + f" / repetition {repetition}", yaxis=dict(range=[0, 1.05], title="Recorded response rate"))
        states.append(state(prompt_name + f" / rep {repetition}", chart,
            "Metrics recomputed from recorded raw outputs. Missing usage remains unavailable.\nModel: " + str(record.get("model")) + "\nStatus: " + str(record.get("status")) + "\n" + json.dumps(summary["usage"]),
            {"n": metrics["n"], "API errors": metrics["operational_errors"], "Retries": summary["retries"], "Mean total latency (s)": round(summary["mean_total_latency_seconds"], 3)}))
    sets = [{row["case_id"] for row in rows} for rows in grouped.values()]
    balanced = all(ids == sets[0] for ids in sets)
    evidence = "RECORDED REAL LLM RESPONSES" if record["evidence_type"] == "real_llm" else "TEST RECORDING — NOT REAL INFERENCE"
    item = card("recorded", "Recorded response evaluation", "What did the recorded model actually return?",
        "Read error categories and denominators before comparing prompts; partial runs can be unbalanced.",
        f"Balanced observed case sets: {balanced}. Repetitions remain separate; no probabilities or costs inferred. Author review required.", states, "Recorded evaluation", evidence=evidence)
    destination = output / "recorded_evaluation.html"
    write_page(destination, [item], "Recorded prompt evaluation")
    summary = {"source": str(path), "evidence_type": record["evidence_type"], "balanced_observed_case_sets": balanced,
               "groups": summaries, "interpretation_status": "author review required"}
    (output / "results").mkdir(parents=True, exist_ok=True)
    (output / "results" / "replayed-metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"html": destination, "summary": summary}
