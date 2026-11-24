"""Generate the deterministic Day 66 visual lab locally on CPU."""

import os

# Set before NumPy import. Existing user thread limits take precedence.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import platform
import webbrowser

import matplotlib
import numpy as np
import PIL
import plotly

from attention_figures import alignment, pipeline, projections, scaling, temperature, value_animations
from attention_html import context_page, geometry_page, memory_page, softmax_page, values_page
from attention_lab import complexity_experiment, geometry_state, softmax_state
from attention_sequences import complexity, contextual, masking, multihead, rag_cache, self_cross


EXPERIMENTS = {
    "projections": "01 · How does each input row become Q, K and V?",
    "geometry": "02 · How do vector directions and magnitudes change weights?",
    "pipeline": "03 · Which operations turn Q/K/V into contextual output?",
    "scaling": "04 · What changes when Gaussian key dimension increases?",
    "softmax": "05 · How do logits and temperature affect all weights?",
    "values": "06 · How do mixtures and changing V affect output geometry?",
    "context": "07 · How can engineered contexts change a shared bank vector?",
    "masking": "08 · Which representations can each query access?",
    "cross": "09 · How do query/source origins change attention shape?",
    "heads": "10 · What do parallel projection spaces and concatenation do?",
    "complexity": "11 · How do pair counts, bytes and CPU time grow?",
    "inference": "12 · How do retrieval and cached decoding relate to attention?",
}


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--all", action="store_true", help="generate all 12 groups (default)")
    choice.add_argument("--experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--quick", action="store_true", help="fewer frames/queries/timing lengths; same 12 groups")
    parser.add_argument("--list", action="store_true", help="list questions and exit without generating")
    parser.add_argument("--output-dir", type=Path, help="parent for a new dated run; default: topic/outputs/runs")
    parser.add_argument("--open", action="store_true", help="open the resulting local gallery in the default browser")
    args = parser.parse_args(argv)
    if args.list:
        for name, question in EXPERIMENTS.items():
            print(f"{name:12} {question}")
        return None
    parent = args.output_dir or Path(__file__).resolve().parent/"outputs"/"runs"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = parent/stamp
    folder.mkdir(parents=True, exist_ok=False)
    selected = [args.experiment] if args.experiment else list(EXPERIMENTS)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), quick=args.quick, seed=66,
                  experiments=selected, interpretations="author review required", data="synthetic only",
                  environment=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
                                   pillow=PIL.__version__, plotly=plotly.__version__, os=platform.platform(),
                                   processor=platform.processor(), logical_cpus=os.cpu_count(), dtype="float64",
                                   thread_environment={name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}))
    # Numerical snapshots let browser checks compare JS controls against Python.
    report["reference_cases"] = dict(geometry_default=geometry_state(), geometry_rotated=geometry_state(90, 0, 2, 3, True),
                                     softmax_default=softmax_state(), softmax_cold=softmax_state([20, -20, -20, -20], temperature=.1))
    for experiment in selected:
        print(f"Generating {experiment}: {EXPERIMENTS[experiment]}", flush=True)
        if experiment == "projections":
            projections(folder/"01_qkv_projections.gif", args.quick)
        elif experiment == "geometry":
            geometry_page(folder/"02_query_key_geometry.html")
            alignment(folder/"02_query_key_alignment.gif", args.quick)
        elif experiment == "pipeline":
            pipeline(folder/"03_attention_pipeline.gif", folder/"03_attention_pipeline.png")
        elif experiment == "scaling":
            report["scaling"] = scaling(folder, args.quick)
        elif experiment == "softmax":
            softmax_page(folder/"05_softmax_playground.html")
            temperature(folder/"05_softmax_temperature.gif", args.quick)
        elif experiment == "values":
            values_page(folder/"06_weighted_value_geometry.html")
            value_animations(folder/"06_weighted_value_aggregation.gif", folder/"06_changing_values.gif", args.quick)
        elif experiment == "context":
            context_page(folder/"07_contextual_embeddings.html")
            contextual(folder/"07_contextual_attention.gif", args.quick)
        elif experiment == "masking":
            report["future_intervention"] = masking(folder/"08_attention_masks.gif", folder/"08_full_vs_causal_attention.png", folder/"08_future_token_leakage.gif")
        elif experiment == "cross":
            self_cross(folder/"09_self_vs_cross_attention.gif", folder/"09_cross_attention_heatmap.png")
        elif experiment == "heads":
            multihead(folder/"10_multi_head_attention_preview.gif")
        elif experiment == "complexity":
            report["timings"] = complexity_experiment(args.quick)
            complexity(folder/"11_attention_complexity.png", folder/"11_attention_matrix_growth.gif", report["timings"])
            memory_page(folder/"11_attention_memory.html")
        elif experiment == "inference":
            report["inference"] = rag_cache(folder/"12_attention_in_rag.gif", folder/"12_kv_cache_inference.gif")
    assets = sorted(p for p in folder.iterdir() if p.suffix in (".gif", ".png", ".html"))
    report["assets"] = [dict(name=p.name, bytes=p.stat().st_size) for p in assets]
    links = "".join(f'<li><a href="{escape(p.name)}">{escape(p.name)}</a> · {p.stat().st_size/1024:.0f} KiB</li>' for p in assets)
    gallery = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
               '<title>Day 66 attention laboratory</title><style>body{font:18px system-ui;max-width:1000px;margin:40px auto;padding:20px;line-height:1.7}a{color:#2563eb}</style>'
               '<h1>Day 66 · Attention laboratory</h1><p>Open a standalone explorer or animation. Synthetic fixtures; author interpretation review required.</p>'
               '<p>Blue: queries · Orange: keys · Green: values · Purple: weights/output · Gray: masks/inactive</p>'
               f'<p>Mode: {"quick" if args.quick else "full"} · UTC run {stamp}. Numerical metadata: <a href="run.json">run.json</a>.</p><ol>{links}</ol></html>')
    (folder/"index.html").write_text(gallery, encoding="utf-8")
    (folder/"run.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=serializable)+"\n", encoding="utf-8")
    print(f"Generated {len(assets)} assets; gallery: {folder/'index.html'}", flush=True)
    if args.open:
        webbrowser.open((folder/"index.html").resolve().as_uri())
    return folder


if __name__ == "__main__":
    main()
