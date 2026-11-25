"""Generate the Day 67 scientific visual laboratory, without network calls or servers."""

import os
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import platform
import webbrowser

import matplotlib
import numpy as np
import PIL
from PIL import Image
import plotly
import sklearn
import torch

from example import Config, make_split, reversal_targets
import transformer_figures as figures
import transformer_html as html
from transformer_lab import (array, capture_attention, capture_model, default_trace, equivariance_demo,
    model_at, playground_cases, shared_pca, train_comparison)
from from_scratch import sinusoidal_positions

GROUPS = {
    "architecture": "01 · How do encoder and decoder blocks compose?",
    "positional_encoding": "02 · How is order represented geometrically?",
    "self_attention": "03 · How do Q/K/V yield a contextual vector?",
    "causal_attention": "04 · Which target positions are visible?",
    "cross_attention": "05 · How does the decoder consult source memory?",
    "multi_head": "06 · What do independent projection subspaces compute?",
    "feed_forward": "07 · How does an FFN mix features within each position?",
    "residuals": "08 · What geometry and gradient path does addition preserve?",
    "layer_norm": "09 · Which dimensions does LayerNorm normalize?",
    "representations": "10 · How do tokens move through encoder layers?",
    "training_inference": "11 · What information differs at training and generation?",
    "target_leakage": "12 · Does unmasked teacher forcing create a shortcut?",
    "recurrent_comparison": "13 · How do recurrent and attention paths differ?",
    "complexity": "14 · How do pair counts and score storage scale?",
    "training_experiment": "15 · What changes over real training checkpoints?",
    "playground": "16 · What changes under controlled forward interventions?",
}
QUICK = ("architecture", "positional_encoding", "causal_attention", "cross_attention", "representations",
         "training_inference", "target_leakage", "complexity", "training_experiment", "playground")
TRAINING_RELEVANT = {"cross_attention", "multi_head", "residuals", "layer_norm", "representations",
                     "training_inference", "target_leakage", "training_experiment", "playground"}
REQUIRES_TRAINING = {"training_inference", "target_leakage", "training_experiment"}
INTERACTIVE = {"positional_encoding", "multi_head", "residuals", "representations", "complexity", "training_experiment", "playground"}


class ScriptLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.external.extend(value for key, value in attrs if key == "src")


def validate_assets(folder):
    """Decode all GIF frames and PNGs; check bundled HTML structure (browser checks are separate)."""
    assets = []
    for path in sorted(folder.iterdir()):
        if path.suffix not in (".gif", ".png", ".html") or path.name == "index.html":
            continue
        item = {"name": path.name, "bytes": path.stat().st_size}
        if path.suffix in (".gif", ".png"):
            with Image.open(path) as image:
                item.update(width=image.width, height=image.height, frames=getattr(image, "n_frames", 1))
                if path.suffix == ".gif":
                    if image.n_frames < 2 or image.info.get("loop") != 0:
                        raise ValueError(f"invalid animation: {path}")
                    for frame in range(image.n_frames):
                        image.seek(frame)
                        image.load()
                        if image.info.get("duration", 0) <= 0:
                            raise ValueError(f"missing frame duration: {path}")
                else:
                    image.verify()
        else:
            source = path.read_text(encoding="utf-8")
            parser = ScriptLinks()
            parser.feed(source)
            if parser.external or "window.labReady" not in source or "plotly.js" not in source.lower():
                raise ValueError(f"HTML must bundle its runtime and expose readiness: {path}")
        assets.append(item)
    return assets


def write_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False, default=lambda value: value.tolist()) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="all 16 groups (default)")
    mode.add_argument("--quick", action="store_true", help="10 core groups; smaller split and default 80 updates per arm")
    mode.add_argument("--only", nargs="+", choices=tuple(GROUPS), help="generate only these groups")
    parser.add_argument("--interactive", action="store_true", help="HTML only for selected groups, or all seven HTMLs by default")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--device", choices=("cpu",), default="cpu", help="CPU preserves the existing study model contract")
    parser.add_argument("--training-steps", type=int, help="default 400, or 80 in quick mode; range 1–2000")
    parser.add_argument("--sequence-length", type=int, default=5, help="content length, 3–10")
    parser.add_argument("--num-heads", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--fps", type=int, default=2, help="1–12; each frame is an explanatory state")
    parser.add_argument("--dpi", type=int, default=90, help="40–200")
    parser.add_argument("--show", action="store_true", help="open the local gallery after generation")
    parser.add_argument("--skip-training", action="store_true", help="reuse matching cache; otherwise label/skip dependent views")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(f"{name:22} {question}" for name, question in GROUPS.items()))
        return None
    steps = args.training_steps if args.training_steps is not None else (80 if args.quick else 400)
    if not 1 <= steps <= 2000 or not 3 <= args.sequence_length <= 10 or not 1 <= args.fps <= 12 or not 40 <= args.dpi <= 200:
        parser.error("require 1–2000 training steps, length 3–10, fps 1–12 and dpi 40–200")
    config = Config(seed=args.seed, sequence_length=args.sequence_length, heads=args.num_heads, steps=steps,
                    train_size=128 if args.quick else 512, test_size=32 if args.quick else 128)
    selected = list(dict.fromkeys(args.only or (QUICK if args.quick else GROUPS)))
    if args.interactive:
        selected = [group for group in selected if group in INTERACTIVE]
        if not selected:
            parser.error("selected groups have no HTML artifact")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = args.output_dir / "visualizations" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    metrics_path = args.output_dir / "metrics" / f"visual_run_{stamp}.json"
    report = {"executed_at_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
              "groups_requested": selected, "groups_generated": [], "skipped": [], "failures": [],
              "interpretation": "author review required", "data": "synthetic only; no external model/dataset",
              "historical_boundary": "Separate visual experiment; does not replace the 2026-11-25 seed-42 analytical run.",
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                  "matplotlib": matplotlib.__version__, "plotly": plotly.__version__, "pillow": PIL.__version__,
                  "sklearn": sklearn.__version__, "os": platform.platform(), "device": "cpu", "torch_threads": 1,
                  "precision": "FP32 model; float64 mathematical controls", "fps": args.fps, "dpi": args.dpi}}
    comparison = None
    if set(selected) & TRAINING_RELEVANT:
        print("Preparing paired experiment/cache (one CPU thread)…", flush=True)
        comparison, cache, reused = train_comparison(config, args.output_dir / "checkpoints", args.skip_training)
        report.update(checkpoint=str(cache), cache_reused=reused, training_available=comparison is not None)
        if comparison is not None:
            report["experiment"] = {key: value for key, value in comparison.items() if key != "states"}
    model, trace = default_trace(config, comparison)
    _, initial_trace = default_trace(config)
    traces = [(0, initial_trace)]
    if comparison is not None:
        traces = []
        _, test = make_split(config)
        decoder, _ = reversal_targets(test[:1])
        for index, step in enumerate(comparison["checkpoints"]):
            state = capture_model(model_at(comparison, checkpoint=index), test[:1], decoder)
            state["classification"] = "Untrained numerical demonstration" if step == 0 else "Trained model observation"
            traces.append((step, state))
    style = {"fps": args.fps, "dpi": args.dpi}
    for group in selected:
        if group in REQUIRES_TRAINING and comparison is None:
            report["skipped"].append({"group": group, "reason": "training skipped and no compatible checkpoint exists"})
            print(f"Skipped {group}: no trained checkpoint", flush=True)
            continue
        print(f"Generating {GROUPS[group]}", flush=True)
        try:
            if group == "architecture":
                figures.architecture(folder, config, **style)
            elif group == "positional_encoding":
                demo = equivariance_demo(args.seed)
                report["permutation_control"] = {key: value for key, value in demo.items() if "error" in key or "change" in key}
                if not args.interactive:
                    figures.positional(folder, demo, **style)
                html.positions_page(folder / "02_positional_encoding_3d.html")
            elif group == "self_attention":
                words = "The animal did not cross the street because it was tired".split()
                untrained, _ = default_trace(config)
                tokens = torch.arange(2, 2 + len(words)).reshape(1, -1)
                x = untrained.embedding(tokens) * config.d_model ** .5 + torch.tensor(sinusoidal_positions(len(words), config.d_model), dtype=torch.float32)
                _, snapshot = capture_attention(untrained.transformer.encoder.layers[0].self_attn, x, x)
                figures.self_attention(folder, snapshot, words, **style)
            elif group == "causal_attention":
                attention = next(a for a in initial_trace["attention"] if a["mechanism"] == "decoder")
                figures.causal(folder, attention, initial_trace["decoder_input"], **style)
            elif group == "cross_attention":
                figures.cross_attention(folder, trace, initial_trace, **style)
            elif group == "multi_head":
                if not args.interactive:
                    figures.multihead(folder, trace, **style)
                html.attention_page(folder / "06_attention_heads_interactive.html", [(traces[-1][0], trace)])
            elif group == "feed_forward":
                figures.feed_forward(folder, args.seed, **style)
            elif group == "residuals":
                if not args.interactive:
                    figures.residual(folder, trace, **style)
                html.residual_page(folder / "08_residual_vectors_3d.html", trace)
            elif group == "layer_norm":
                figures.normalization(folder, trace, **style)
            elif group == "representations":
                if not args.interactive:
                    figures.representations(folder, trace, **style)
                html.representations_page(folder / "10_representation_evolution_3d.html", traces)
            elif group == "training_inference":
                figures.training_inference(folder, trace, comparison["history"]["masked"][-1], **style)
            elif group == "target_leakage":
                figures.experiments(folder, comparison, **style)
            elif group == "recurrent_comparison":
                figures.recurrent(folder, **style)
            elif group == "complexity":
                if not args.interactive:
                    figures.complexity(folder, config, **style)
                html.memory_page(folder / "14_attention_memory_3d.html")
            elif group == "training_experiment":
                if not args.interactive:
                    figures.training_progress(folder, comparison, **style)
                html.attention_page(folder / "15_training_attention_evolution.html", traces, "15 · Attention, memory and output distributions during actual training", evolution=True)
            elif group == "playground":
                cases = playground_cases(config, comparison)
                report["playground_case_count"] = len(cases)
                report["instrumentation_max_error"] = max(case["instrumentation_max_error"] for case in cases)
                html.playground_page(folder / "16_transformer_playground.html", cases)
            report["groups_generated"].append(group)
        except Exception as error:
            report["failures"].append({"group": group, "error": f"{type(error).__name__}: {error}"})
            write_report(report, metrics_path)
            raise
        write_report(report, metrics_path)
    report["assets"] = validate_assets(folder)
    write_report(report, metrics_path)
    entries = "".join(f'<li><a href="{escape(asset["name"])}">{escape(asset["name"])}</a> · {asset["bytes"] / 1024:.0f} KiB</li>' for asset in report["assets"])
    gallery = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Day 67 Transformer laboratory</title><style>body{{font:17px system-ui;max-width:1100px;margin:30px auto;padding:20px;line-height:1.7}}a{{color:#2563eb}}</style>
<h1>Day 67 · Transformer visual laboratory</h1><p>Synthetic educational views. Conceptual, mathematical and trained observations are labeled in each artifact. Author interpretation review required.</p>
<p>Run {stamp}; seed {config.seed}; {config.steps} updates per arm; CPU. Root documentation and preview publication await author approval.</p>
<p><a href="../../metrics/visual_run_{stamp}.json">Configuration, measured metrics, asset sizes and skipped groups</a></p><ol>{entries}</ol></html>'''
    (folder / "index.html").write_text(gallery, encoding="utf-8")
    print(f"Validated {len(report['assets'])} assets; {len(report['skipped'])} groups skipped. Gallery: {folder / 'index.html'}", flush=True)
    if args.show:
        webbrowser.open((folder / "index.html").resolve().as_uri())
    return folder


if __name__ == "__main__":
    main()
