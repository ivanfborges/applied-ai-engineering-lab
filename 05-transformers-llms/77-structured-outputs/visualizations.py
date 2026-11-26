"""Generate selected Day 77 visual demonstrations and immutable local run records."""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import webbrowser

from visual_core import DEMOS, SEED, compute_lab
from visual_html import render_html
from visual_render import render_static


QUESTIONS = {
    "space": "Which synthetic responses survive progressively stricter acceptance gates?",
    "decoding": "Which tokens may extend this prefix, and which one is sampled?",
    "masking": "How does removing candidates change normalized probabilities?",
    "grammar": "How does each emitted toy token change the allowed continuations?",
    "validation": "Which actual validator rejects each response, and with what denominator?",
    "strictness": "How do type coercion and extra-field policy differ independently?",
    "semantics": "Can high structural compliance coexist with incorrect extracted values?",
    "retries": "When does a bounded controller regenerate, accept, block or exhaust?",
    "reliability": "What integer attempt budget reaches a theoretical reliability target?",
    "surface": "How do per-attempt success and total budget jointly affect reliability?",
    "cost": "Which reliability targets fit an expected cost budget?",
    "correlation": "How do permanent failures and conditional recovery change retry gains?",
    "guardrails": "Why can a valid tool proposal still be blocked before execution?",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def gallery(root, artifacts):
    cards = []
    for name in artifacts:
        safe = html.escape(name, quote=True)
        content = (f'<img loading="lazy" src="{safe}" alt="{safe.replace("_", " ")}">'
                   if name.endswith((".png", ".gif")) else f'<a href="{safe}">Open standalone interactive explorer</a>')
        cards.append(f'<article><h2>{safe}</h2>{content}</article>')
    (root / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Day 77 Structured Outputs visual lab</title><style>body{font:16px system-ui;max-width:1250px;'
        'margin:2rem auto;padding:1rem;background:#f5f8fc;color:#243746}article{background:white;padding:1rem;'
        'border:1px solid #d9e2eb;margin:1rem 0;border-radius:10px}img{max-width:100%;height:auto}h2{font-size:18px}</style>'
        '<h1>Structured Outputs: local visual laboratory</h1><p>Synthetic CPU calculations, fixed toy grammar '
        'and explicit theoretical retry assumptions. No provider, credentials or real tool execution. '
        'Author interpretation review required.</p>' + "".join(cards) + '</html>', encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--all", action="store_true")
    select.add_argument("--demo", choices=DEMOS, action="append")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--preview", action="store_true", help="80 dpi instead of 110; 10000 default MC requests")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--samples", type=int, help="MC requests per scenario; default 100000 (preview 10000)")
    parser.add_argument("--output-dir", type=Path, help="New directory; relative paths are topic-relative")
    parser.add_argument("--show", action="store_true", help="Open the local gallery after export")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(f"{name}: {QUESTIONS[name]}" for name in DEMOS))
        return
    samples = args.samples if args.samples is not None else (10_000 if args.preview else 100_000)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if not 2 <= samples <= 1_000_000:
        parser.error("samples must be in [2, 1000000]")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    topic = Path(__file__).resolve().parent
    root = args.output_dir or topic / "outputs" / f"visual-{stamp}"
    if not root.is_absolute():
        root = topic / root
    if root.exists():
        parser.error("Output directory already exists; use a new path to preserve historical records.")
    root.mkdir(parents=True)
    names = list(dict.fromkeys(args.demo or DEMOS))
    print(f"Compute synthetic records | seed={args.seed} | samples={samples}", flush=True)
    lab = compute_lab(args.seed, samples)
    versions = {"python": platform.python_version(), "platform": platform.platform(),
                **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pydantic", "matplotlib", "plotly", "Pillow")}}
    write_json(root / "experiment.json", {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(), "synthetic": True,
        "configuration": {"seed": args.seed, "samples": samples, "preview": args.preview, "demos": names},
        "versions": versions, "hypothesis": "Separate gates reject distinct errors; independent and latent-failure retry models have different recovery curves.",
        "result": lab,
        "interpretation_candidate": "Representation constraints, source fidelity and permission checks establish different properties; author review required.",
        "limitation": "Hand-authored fixtures, fixed finite grammar and synthetic Monte Carlo; no LLM, causal feedback, real prices, measured latency or live authorization system.",
    })
    artifacts = []
    for name in names:
        print(f"Render {name}: {QUESTIONS[name]}", flush=True)
        artifacts.extend(render_static(name, root, lab, args.preview))
        artifacts.extend(render_html(name, root, lab))
    gallery(root, artifacts)
    artifacts.append("index.html")
    sources = ("visualizations.py", "visual_core.py", "visual_render.py", "visual_html.py", "visual_controls.js", "example.py", "from_scratch.py")
    write_json(root / "manifest.json", {"schema": 1, "versions": versions,
               "source_sha256": {name: sha256(topic / name) for name in sources},
               "artifacts": {name: {"sha256": sha256(root / name), "bytes": (root / name).stat().st_size} for name in artifacts},
               "experiment_sha256": sha256(root / "experiment.json"),
               "questions": {name: QUESTIONS[name] for name in names},
               "interpretation_status": "author review required", "public_previews": []})
    print(f"Generated {len(artifacts) - 1} demonstrations. Gallery: {root / 'index.html'}", flush=True)
    if args.show:
        webbrowser.open((root / "index.html").as_uri())


if __name__ == "__main__":
    main()
