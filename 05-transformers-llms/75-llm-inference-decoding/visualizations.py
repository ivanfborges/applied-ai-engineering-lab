"""Generate bounded synthetic decoding visuals and a dated offline gallery."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform

import matplotlib
import numpy as np
import PIL
import plotly

from visual_core import GROUPS, LabConfig, PRESETS, compute_group, serializable
from visual_render import render_group

HYPOTHESES = {
    "temperature": "Unfiltered entropy grows with temperature for fixed finite nonuniform logits.",
    "top-k": "Removing ranked candidates requires renormalization and changes the sampling policy.",
    "top-p": "The same cumulative threshold retains different support for different distributions.",
    "interactions": "Temperature and top-k change the input distribution used by top-p.",
    "generation": "Greedy local maximization can miss the exact best fixed-depth model sequence.",
    "stopping": "EOS, token caps and text stops produce distinct finish reasons and visible text.",
    "reproducibility": "Resetting seed replays a stream; shared RNG calls advance it across runs.",
    "sampling": "Nested empirical frequencies fluctuate around a fixed categorical policy.",
    "cache": "Cached K/V avoid repeated projections; a fixed-layer attention result is preserved numerically.",
}
LIMITATIONS = {
    "temperature": "Token entropy is not factual confidence; no real language model is evaluated.",
    "top-k": "Synthetic vocabulary; finite-precision positive support can be smaller than selected k.",
    "top-p": "Inclusive prefix and stable lower-ID ties are this decoder's conventions.",
    "interactions": "Computed grid points only; flat tiles extend nearest samples, without smoothed boundaries.",
    "generation": "Binary depth-3 fixed-length tree, no EOS or length normalization; likelihood is not usefulness.",
    "stopping": "Handwritten fragments, non-streaming string matching, no tokenizer or context-window enforcement.",
    "reproducibility": "16 runs per protocol; NumPy fixture only, no cross-version/distributed guarantee.",
    "sampling": "One seed, finite samples, shared initialization across configurations; TV need not decline monotonically.",
    "cache": "Explicit theoretical counts, plus single-layer numerical equality; no latency or full-model benchmark.",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(serializable(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def gallery(root, artifacts):
    items = []
    for name in artifacts:
        escaped = html.escape(name, quote=True)
        if name.endswith((".png", ".gif")):
            content = f'<img src="{escaped}" alt="{escaped.replace("_", " ")}" loading="lazy">'
        else:
            content = f'<a href="{escaped}">Open interactive computed landscape</a>'
        items.append(f'<article><h2>{html.escape(name.replace("_", " "))}</h2>{content}</article>')
    (root / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Day 75 decoding visual laboratory</title><style>body{font:16px system-ui;margin:2rem auto;max-width:1200px;'
        'padding:1rem;color:#203246;background:#f6f8fa}article{padding:1rem;background:white;margin:1rem 0;'
        'border:1px solid #ddd;border-radius:8px}img{max-width:100%;height:auto}h2{font-size:1.1rem}</style>'
        '<h1>LLM inference and decoding: synthetic visual laboratory</h1>'
        '<p>Calculated distributions, seeded toy experiments and explicit theoretical cache counts. '
        'No real-model quality, production or latency benchmark. Interpretations require author review.</p>'
        '<p>Live Python playground: launch interactive_demo.py with Streamlit. HTML landscapes run offline.</p>'
        + "".join(items) + '</html>', encoding="utf-8")
    return artifacts + ["index.html"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--group", choices=GROUPS, action="append", dest="groups")
    for group in GROUPS:
        parser.add_argument(f"--{group}", action="append_const", const=group, dest="groups")
    parser.add_argument("--quick", action="store_true", help="10 sweep frames and 13x13 grids; sampling stays at 10000")
    parser.add_argument("--seed", type=int, default=275)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, help="Must not already exist")
    parser.add_argument("--reuse-run", type=Path, help="Render saved measurements into a new directory without rerunning experiments")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(f"{group}: {HYPOTHESES[group]}" for group in GROUPS)); return
    if args.all and args.groups:
        parser.error("Use --all or selected groups, not both.")
    names = list(dict.fromkeys(args.groups or GROUPS))
    try:
        config = LabConfig(args.seed, args.quick, args.samples, args.beam_width)
    except ValueError as error:
        parser.error(str(error))
    source = None
    if args.reuse_run:
        source = json.loads((args.reuse_run / "manifest.json").read_text(encoding="utf-8"))
        config = LabConfig(**source["configuration"])
        if any(group not in source["groups"] for group in names):
            parser.error("Selected groups are not all present in the saved run.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = args.output_dir or Path(__file__).resolve().parent / "outputs" / f"visual-{stamp}"
    if root.exists():
        parser.error("Output directory exists; select a new directory to preserve history.")
    root.mkdir(parents=True)
    (root / "experiments").mkdir()
    versions = dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
                    plotly=plotly.__version__, pillow=PIL.__version__, platform=platform.platform())
    artifacts = []
    print(f"Day 75 | seed={config.seed} | quick={config.quick} | {root}", flush=True)
    for name in names:
        print(f"{'Reuse' if source else 'Calculate'} {name}", flush=True)
        if source:
            path = args.reuse_run / "experiments" / f"{name}.json"
            if digest(path) != source["measurement_sha256"][name]:
                raise ValueError(f"Saved measurement digest mismatch: {name}")
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            record = dict(group=name, executed_at_utc=datetime.now(timezone.utc).isoformat(),
                          configuration=asdict(config), presets=PRESETS, hypothesis=HYPOTHESES[name],
                          versions=versions, result=compute_group(name, config),
                          interpretation_candidate="Compare calculated states against the hypothesis; author review required.",
                          limitation=LIMITATIONS[name], data="synthetic")
        write_json(root / "experiments" / f"{name}.json", record)
        print(f"Render {name}", flush=True)
        artifacts.extend(render_group(name, root, record["result"]))
    artifacts = gallery(root, artifacts)
    sources = ("from_scratch.py", "visual_core.py", "visual_render.py", "visualizations.py", "interactive_demo.py")
    manifest = dict(schema=1, rendered_at_utc=datetime.now(timezone.utc).isoformat(), groups=names,
                    configuration=asdict(config), versions=versions,
                    source_sha256={name: digest(Path(__file__).parent / name) for name in sources},
                    measurement_sha256={name: digest(root / "experiments" / f"{name}.json") for name in names},
                    artifact_sha256={name: digest(root / name) for name in artifacts}, artifacts=artifacts,
                    reused_measurements=str(args.reuse_run.resolve()) if source else None,
                    interpretation_status="author review required", public_previews=[],
                    scope="Synthetic CPU calculations and explicit theoretical counts; no real-model experiment")
    write_json(root / "manifest.json", manifest)
    print(f"Generated {len(artifacts)} artifacts. Gallery: {root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
