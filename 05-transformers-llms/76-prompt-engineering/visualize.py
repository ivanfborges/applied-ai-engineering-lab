"""Generate the Day 76 visual lab: offline simulation by default.

Install simulation/test dependencies from the repository root: pip install -e ".[dev]"
Optional real evaluation only: pip install google-genai (no canonical extra exists).
Examples (from this topic directory):
  python visualize.py --quick
  python visualize.py --experiment reliability --seed 42
  python visualize.py --experiment all --duration 6 --fps 4 --width 1100 --height 700
  python visualize.py --results-file outputs/real-.../results/real-evaluation.json
  python visualize.py --mode real --case-limit 2 --max-requests 12
Real mode requires GEMINI_API_KEY/GEMINI_MODEL and can incur charges. It is never
selected implicitly. SDK internal retries are disabled; --max-attempts is explicit.
Figures and manifests are dated and ignored. Public documentation/previews await
author review. No default command overwrites a previous run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np

from visual_render import Gallery, serializable


EXPERIMENTS = ["conditioning", "anatomy", "in_context", "ablation", "optimization",
               "generalization", "validation", "reliability", "pareto", "failure_handling", "injection"]
HERE = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["simulation", "real"], default="simulation")
    p.add_argument("--experiment", choices=["all"] + EXPERIMENTS, default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--duration", type=float, default=6)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--height", type=int, default=700)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    p.add_argument("--results-file", type=Path)
    p.add_argument("--input-price-per-million", type=float, default=.50, help="Hypothetical simulation USD price, not provider pricing")
    p.add_argument("--output-price-per-million", type=float, default=1.50)
    p.add_argument("--max-requests", type=int, default=72)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--max-attempts", type=int, default=1)
    p.add_argument("--case-limit", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--max-output-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=30, help="Real request timeout, seconds")
    return p


def main(argv: list[str] | None = None) -> Path:
    p = parser()
    args = p.parse_args(argv)
    if not (0 <= args.seed <= 2**32 - 20 and 0 < args.duration <= 30 and 1 <= args.fps <= 12
            and 640 <= args.width <= 2400 and 480 <= args.height <= 1600 and 72 <= args.dpi <= 200):
        p.error("invalid seed/render limits: duration (0,30], fps [1,12], width [640,2400], height [480,1600], dpi [72,200]")
    if not np.isfinite([args.duration, args.temperature, args.timeout, args.input_price_per_million, args.output_price_per_million]).all() or min(args.input_price_per_million, args.output_price_per_million) < 0:
        p.error("finite settings and nonnegative hypothetical prices required")
    if not 1 <= args.case_limit <= 12 or not 0 <= args.temperature <= 2 or not 1 <= args.max_output_tokens <= 8192 or not 0 < args.timeout <= 300:
        p.error("invalid real-mode case/temperature/token/timeout limit")
    if args.results_file and args.mode == "real":
        p.error("--results-file is replay only; omit --mode real to avoid ambiguous intent")
    now = datetime.now(timezone.utc)
    kind = "replay" if args.results_file else args.mode
    output = args.output_dir / f"{kind}-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    output.mkdir(parents=True, exist_ok=False)
    if args.results_file or args.mode == "real":
        from visual_real import collect, replay
        try:
            source = args.results_file or collect(output, repetitions=args.repetitions,
                max_attempts=args.max_attempts, max_requests=args.max_requests, case_limit=args.case_limit,
                temperature=args.temperature, max_tokens=args.max_output_tokens, timeout=args.timeout)
            result = replay(source, output)
        except (ValueError, RuntimeError) as exc:
            p.exit(2, f"Evaluation stopped: {exc}\nNo simulation substituted. Check partial records at {output}\n")
        print(f"Recorded evaluation: {result['html']}")
        return output
    gallery = Gallery(output, seed=args.seed, quick=args.quick, duration=args.duration,
        fps=args.fps, width=args.width, height=args.height, dpi=args.dpi,
        input_price=args.input_price_per_million, output_price=args.output_price_per_million)
    selected = EXPERIMENTS if args.experiment == "all" else [args.experiment]
    for name in selected:
        print(f"Rendering {name} (synthetic)", flush=True)
        getattr(gallery, name)()
    gallery.dashboard()
    manifest = {"schema_version": 1, "evidence_type": "synthetic_visual_lab_not_llm_results",
                "executed_at_utc": now.isoformat(), "mode": "simulation", "seed": args.seed,
                "configuration": {**vars(args), "output_dir": str(args.output_dir), "results_file": None},
                "environment": {"python": platform.python_version(), "os": platform.system(),
                    **{name: version(name) for name in ["numpy", "matplotlib", "Pillow", "plotly"]}},
                "api_requests": 0, "experiments": selected,
                "hypothesis": "Expose consequences of explicit synthetic probability, response, selection and decision assumptions; not rank real prompts.",
                "interpretation_status": "author review required",
                "interpretation_candidate": "Schema validity, evidence support and operational acceptance are separate; apparent benefits depend on the simulated mechanism.",
                "limitation": "All models, prices, probabilities, confidence and response rules are synthetic. No empirical LLM, pricing, latency, security or production evidence.",
                "results": gallery.records,
                "artifacts": [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in gallery.artifacts],
                "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [HERE / name for name in ["visualize.py", "visual_core.py", "visual_render.py", "visual_html.py", "visual_real.py", "evaluation.py", "prompts.py", "evaluation_dataset.json"]]}}
    (output / "results" / "simulation-manifest.json").write_text(json.dumps(serializable(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    review = ["# Day 76 visual lab — local review package", "",
        "All results here are synthetic. No LLM was called. Author review required.", "",
        "This dated run is separate from the earlier analytical evaluator and all previous visual runs.",
        "Public READMEs, guides and preview allowlists are unchanged pending artifact review.", "",
        "## Install and run", "",
        "From the repository root: `python -m pip install -e \".[dev]\"`.",
        "The simulation uses NumPy, Matplotlib, Pillow and Plotly already declared in the root pyproject.",
        "Optional real mode only: `python -m pip install google-genai`. No canonical Gemini extra exists yet; it is not installed by this lab.", "",
        "```powershell", "python 05-transformers-llms/76-prompt-engineering/visualize.py --quick",
        "python 05-transformers-llms/76-prompt-engineering/visualize.py --experiment reliability --seed 42",
        "python 05-transformers-llms/76-prompt-engineering/visualize.py --experiment optimization",
        "python 05-transformers-llms/76-prompt-engineering/visualize.py --experiment all --duration 12 --fps 3",
        "python 05-transformers-llms/76-prompt-engineering/visualize.py --results-file path/to/real-evaluation.json",
        "# Explicit opt-in; can incur charges; requires GEMINI_API_KEY and GEMINI_MODEL:",
        "python 05-transformers-llms/76-prompt-engineering/visualize.py --mode real --case-limit 2 --max-requests 12", "```", "",
        "`--output-dir` selects a parent for a fresh timestamped run. `--width`, `--height`, `--dpi`, `--duration`, `--fps` control exports.",
        "`--input-price-per-million` and `--output-price-per-million` affect hypothetical simulation prices only.",
        "Real mode bounds requests/repetitions/attempts and disables SDK internal retries; it does not silently fall back to simulation.",
        "Recorded raw responses, usage (or null), latency, exceptions by type, references, prompts and attempts remain local.",
        "A real response never supplies a full output probability distribution. Costs remain unavailable in real replay.", "",
        "## Learning questions and assumptions", ""]
    for item in gallery.cards:
        review.extend(["### " + item["title"], "", item["question"], "", "Takeaway candidate: " + item["takeaway"],
                       "", "Assumptions: " + item["assumption"], ""])
    review.extend(["## Executed configuration", "", "```json",
                   json.dumps({"seed": args.seed, "quick": args.quick, "experiments": selected,
                               "environment": manifest["environment"]}, indent=2), "```", "",
                   "The manifest records hypotheses, formulas/data, results, limitations, versions and hashes.",
                   "Neither generation nor this package establishes browser correctness; inspect the separate verification record.", "",
                   "## Recommended preview candidates (not copied/unignored)", "",
                   "1. `animations/01_prompt_conditioning.gif`: includes the confident-wrong counterexample.",
                   "2. `figures/07_validation_matrix.png`: separates parsing, schema, evidence and business gates.",
                   "3. `figures/08_calibration.png`: connects confidence quality to abstention.", "",
                   "Keep the dashboard and remaining exports regenerable/ignored. Select previews only after inspection and author review."])
    (output / "REVIEW.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    print(f"Artifacts: {len(gallery.artifacts)} | Dashboard: {output / 'prompt_engineering_dashboard.html'}", flush=True)
    return output


if __name__ == "__main__":
    main()
