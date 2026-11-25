"""Generate the Day 71 CPU visual laboratory, with standalone offline HTML."""

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import platform
import sys
import webbrowser

import matplotlib
import numpy as np
from PIL import Image
import PIL
import plotly
import torch

from visual_experiments import (
    SEED, model_experiment, attention_experiment, objective_experiment,
    generation_experiment, cache_experiment, rag_experiment, validate_experiments,
)
from visual_html import (
    builtin, attention_page, geometry_page, generation_page, cache_page,
    complexity_page, rag_page,
)
import visual_figures as figures

SECTIONS = ("architecture", "attention", "perturbation", "geometry", "objectives",
            "generation", "cache", "complexity", "rag")


def inspect_artifact(path):
    """Decode raster files and check self-contained Plotly payloads after export."""
    if path.suffix in (".png", ".gif"):
        with Image.open(path) as image:
            frames = getattr(image, "n_frames", 1)
            if path.suffix == ".gif" and frames < 2:
                raise ValueError(f"GIF has fewer than two distinct frames: {path.name}")
            for frame in range(frames):
                image.seek(frame)
                image.load()
            return {"width": image.width, "height": image.height, "frames": frames}
    text = path.read_text(encoding="utf-8")
    if 'id="lab-data"' not in text or "Plotly.react" not in text or "plotly.js" not in text.lower():
        raise ValueError(f"missing intended Plotly payload/library: {path.name}")
    if '<script src="http' in text or '<script src="//' in text:
        raise ValueError(f"external script dependency: {path.name}")
    return {"embedded_plotly": True, "computed_payload": True}


def generate(output_dir, *, section=None, quick=False, no_gif=False):
    output_dir = Path(output_dir)
    selected = list(SECTIONS) if section is None else [section]
    if any(name not in SECTIONS for name in selected):
        raise ValueError(f"section must be one of {SECTIONS}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse reused run directories before doing numerical or rendering work.
    if any(output_dir.iterdir()):
        raise FileExistsError("output directory is not empty; choose a new run directory")
    figures.configure()
    print(f"Computing seed-{SEED} synthetic CPU experiments...", flush=True)
    model = model_experiment(generation_steps=3 if quick else 5)
    attention_data = attention_experiment(model)
    objectives = objective_experiment()
    generation = generation_experiment(steps=5 if quick else 9)
    cache = cache_experiment(total=10 if quick else 18)
    rag = rag_experiment()
    validation = validate_experiments(model, attention_data, cache)
    artifacts, failures, skipped = [], [], []

    def emit(name, render):
        if no_gif and name.endswith(".gif"):
            skipped.append(name)
            return
        print(f"  Rendering {name}", flush=True)
        path = output_dir / name
        try:
            render(path)
            info = inspect_artifact(path)
            artifacts.append({"name": name, "bytes": path.stat().st_size, **info})
        except Exception as exc:
            failures.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAILED {name}: {exc}", flush=True)

    def png(name, draw):
        emit(name, lambda path: figures.save_png(path, draw, quick))

    def gif(name, draw, frames):
        emit(name, lambda path: figures.save_gif(path, draw, frames, quick))

    for name in selected:
        print(f"Section: {name}", flush=True)
        if name == "architecture":
            png("architecture_comparison.png", lambda fig: figures.architecture(fig, model))
            gif("architecture_information_flow.gif", lambda fig, i: figures.architecture(fig, model, i), 16)
        elif name == "attention":
            for kind in ("masks", "logits", "probabilities"):
                png(f"attention_{kind}.png", lambda fig, kind=kind: figures.attention_panels(fig, attention_data, kind))
            gif("attention_masking_process.gif", lambda fig, i: figures.masking_frame(fig, attention_data, i), 4)
            emit("interactive_attention_inspector.html", lambda path: attention_page(path, attention_data))
            cross = attention_data["cross"]
            png("cross_attention_heatmap.png", lambda fig: figures.cross_frame(fig, cross))
            gif("cross_attention_information_flow.gif", lambda fig, i: figures.cross_frame(fig, cross, i), len(cross["query_labels"]))
            emit("interactive_cross_attention.html", lambda path: attention_page(path, attention_data, True))
        elif name == "perturbation":
            png("token_perturbation.png", lambda fig: figures.perturbation(fig, model))
            png("causal_invariance.png", lambda fig: figures.perturbation(fig, model, invariance=True))
            gif("token_perturbation.gif", lambda fig, i: figures.perturbation(fig, model, i), 6)
        elif name == "geometry":
            emit("hidden_representation_geometry.html", lambda path: geometry_page(path, model))
            emit("hidden_representation_perturbation.html", lambda path: geometry_page(path, model, True))
            gif("hidden_representation_evolution.gif", lambda fig, i: figures.geometry_frame(fig, model, i), 11)
        elif name == "objectives":
            png("pretraining_objectives_comparison.png", lambda fig: figures.objectives_comparison(fig, objectives))
            for key, filename in (("mlm", "masked_language_modeling.gif"), ("causal", "causal_language_modeling.gif"), ("span", "span_corruption.gif")):
                frames = len(objectives[key].get("selected", objectives[key]["labels"]))
                gif(filename, lambda fig, i, key=key: figures.objective(fig, objectives, key, i), frames)
        elif name == "generation":
            run = next(r for r in generation["runs"] if r["mode"] == "sampling" and r["temperature"] == 1)
            gif("autoregressive_generation.gif", lambda fig, i: figures.generation_frame(fig, generation, i), len(run["steps"]))
            emit("generation_probability_distributions.html", lambda path: generation_page(path, generation))
        elif name == "cache":
            png("kv_cache_computation.png", lambda fig: figures.cache_figure(fig, cache))
            gif("kv_cache_generation.gif", lambda fig, i: figures.cache_figure(fig, cache, i), len(cache["lengths"]))
            emit("kv_cache_memory.html", lambda path: cache_page(path, cache))
        elif name == "complexity":
            png("attention_complexity.png", lambda fig: figures.complexity(fig))
            png("attention_complexity_log_scale.png", lambda fig: figures.complexity(fig, True))
            emit("attention_complexity_3d.html", lambda path: complexity_page(path, quick))
        elif name == "rag":
            png("rag_architecture_pipeline.png", lambda fig: figures.rag_figure(fig, rag))
            gif("rag_information_flow.gif", lambda fig, i: figures.rag_figure(fig, rag, i), 7)
            emit("retrieval_and_reranking.html", lambda path: rag_page(path, rag))

    now = datetime.now(timezone.utc).isoformat()
    manifest = {"executed_at_utc": now, "seed": SEED, "quick": quick, "sections": selected,
                "no_gif": no_gif, "versions": {"python": platform.python_version(), "numpy": np.__version__,
                "torch": torch.__version__, "matplotlib": matplotlib.__version__, "plotly": plotly.__version__, "pillow": PIL.__version__},
                "hypothesis": "Causal prefixes are invariant; source conditioning is permitted; exact cached single-head attention matches prefix recomputation.",
                "configuration": {"device":"cpu", "dtype":"float64", "threads":1,"width":16,"heads":4,
                                  "ffn_width":48,"layers_per_stack":1,"source_tokens":6,"target_tokens":7,
                                  "dropout":0,"post_ln":True,"perturbation":"+10 last input, feature 0", "training_updates":0},
                "validation": validation, "artifacts": artifacts, "skipped": skipped, "failures": failures,
                "interpretation_candidate": "Observed synthetic controls are consistent with the declared information paths; author review required.",
                "limitations": "Random models and toy objectives/retrieval; no learned translation, pretrained benchmark, measured latency, production evidence or author approval."}
    payload = {"model": model, "attention": attention_data, "objectives": objectives,
               "generation": generation, "cache": cache, "rag": rag}
    (output_dir / "numerical_data.json").write_text(json.dumps(builtin(payload), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(builtin(manifest), indent=2, allow_nan=False), encoding="utf-8")
    links = ''.join(f'<li><a href="{html.escape(item["name"])}">{html.escape(item["name"])}</a></li>' for item in artifacts)
    (output_dir / "index.html").write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Day 71 local visual laboratory</title><body style="font:16px Arial;max-width:1000px;margin:40px auto"><h1>Day 71 visual laboratory</h1><p>Seed {SEED}; generated {html.escape(now)}. Computed synthetic experiments, conceptual diagrams and theoretical counts. Interpretations require author review.</p><ul>{links}</ul><p>Numerical checks: {sum(validation["checks"].values())}/{len(validation["checks"])}. Exports: {len(artifacts)} successful; {len(failures)} failed; {len(skipped)} intentionally skipped.</p></body></html>', encoding="utf-8")
    print(f"Validation: {sum(validation['checks'].values())}/{len(validation['checks'])} numerical assertions passed; cache max error {cache['max_abs_error']:.3e}.", flush=True)
    print(f"Exports: {len(artifacts)} successful, {len(skipped)} skipped, {len(failures)} failed. Gallery: {output_dir / 'index.html'}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=SECTIONS, help="Default: generate all nine sections.")
    parser.add_argument("--quick", action="store_true", help="Lower raster resolution, shorter generation and smaller cache/grid.")
    parser.add_argument("--no-gif", action="store_true", help="Skip animations; retain static and interactive exports.")
    parser.add_argument("--show", action="store_true", help="Open the generated gallery and selected images in local viewers.")
    parser.add_argument("--output-dir", type=Path, help="A new empty output directory. Default: topic outputs/visual_<UTC timestamp>.")
    args = parser.parse_args()
    output = args.output_dir or Path(__file__).resolve().parent / "outputs" / ("visual_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    try:
        report = generate(output, section=args.section, quick=args.quick, no_gif=args.no_gif)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    if args.show:
        webbrowser.open((output.resolve()/"index.html").as_uri())
        for item in report["artifacts"]:
            if item["name"].endswith(".png"):
                with Image.open(output/item["name"]) as image:
                    image.show()
                break
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
