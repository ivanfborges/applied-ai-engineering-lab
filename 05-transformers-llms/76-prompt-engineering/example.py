"""Inspect prompts and run an offline evaluator demonstration. No API requests."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

from evaluation import evaluate, request_with_validation
from prompts import PROMPTS, build_request


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-prompt", choices=sorted(PROMPTS))
    parser.add_argument("--case", default="subtotal", help="Case for --show-prompt")
    args = parser.parse_args()
    dataset_path = HERE / "evaluation_dataset.json"
    response_path = HERE / "response_fixtures.json"
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))["cases"]
    if args.show_prompt:
        selected = next((case for case in cases if case["id"] == args.case), None)
        if selected is None:
            parser.error(f"Unknown case: {args.case}")
        print(json.dumps(build_request(selected["document"], args.show_prompt), indent=2))
        return
    responses = json.loads(response_path.read_text(encoding="utf-8"))["responses"]
    report = evaluate(cases, responses)
    scripted = iter(["Not JSON", '{"invoice_id":"B22","total_usd":999999}'])
    report["retry_demo"] = request_with_validation(lambda: next(scripted))
    report["evidence_type"] = "hand_authored_fixture_evaluation_not_llm_results"
    report["interpretation_status"] = "author review required"
    report["configuration"] = {
        "python": platform.python_version(), "platform": platform.system(),
        "model": None, "api_requests": 0, "randomness": "none",
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "responses_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
    }
    now = datetime.now(timezone.utc)
    report["executed_at_utc"] = now.isoformat()
    output_dir = HERE / "outputs"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / f"fixture-evaluation-{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"Local record: {output}")


if __name__ == "__main__":
    main()
