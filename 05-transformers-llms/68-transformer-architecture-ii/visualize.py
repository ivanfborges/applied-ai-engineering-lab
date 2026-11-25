"""Generate the complete local Day 68 MHA learning lab; no network or GPU required."""

import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import platform
import time
import webbrowser

import matplotlib
import numpy as np
import PIL
from PIL import Image
import plotly
import torch

import visual_figures as figures
import visual_html as html
from visual_core import numerical_report

GROUPS = {
    "architecture": "01 · How does the complete forward pass transform tensors?",
    "subspaces": "02 · How do full-input projections change token geometry?",
    "geometry": "03 · How do query/key direction and magnitude determine attention?",
    "matrices": "04 · Which row/column products produce each attention tensor?",
    "heads": "05 · What differences does an averaged attention map hide?",
    "graph": "06 · Which token keys does each query route through?",
    "single-vs-multi": "07 · What can independent routing retrieve in a constructed case?",
    "scaling": "08 · How does scaling affect dot-product variance and softmax?",
    "head-count": "09 · What changes with head count at fixed model width?",
    "causal": "10 · Does a future-token perturbation change earlier outputs?",
    "concatenation": "11 · How does W_O mix individual head contributions?",
    "diversity": "12 · Can identical maps produce distinct value outputs?",
    "sharing": "13 · Which Q heads share keys/values in MHA, GQA and MQA?",
    "kv-cache": "14 · Which factors determine theoretical cache payload?",
    "complexity": "15 · Which arithmetic and storage terms grow quadratically?",
    "flash": "15b · How does tiled online softmax preserve dense attention?",
    "validation": "16 · Which numerical evidence supports the lab?",
}


class AssetLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("src", "href") and value:
                self.links.append((tag, value))


def validate_exports(folder):
    """Decode real raster frames and validate local references; browser is separate."""
    result = []
    for path in sorted(folder.rglob("*")):
        if path.suffix not in (".gif", ".png", ".html"):
            continue
        item = {"path": path.relative_to(folder).as_posix(), "bytes": path.stat().st_size}
        if path.suffix in (".png", ".gif"):
            with Image.open(path) as image:
                item.update(width=image.width, height=image.height, frames=getattr(image, "n_frames", 1))
                if path.suffix == ".gif" and (image.n_frames < 2 or image.info.get("loop") != 0):
                    raise ValueError(f"GIF must loop over at least two frames: {path}")
                for frame in range(getattr(image, "n_frames", 1)):
                    image.seek(frame); image.load()
                    if path.suffix == ".gif" and image.info.get("duration", 0) < 100:
                        raise ValueError(f"GIF stage too short: {path}")
        else:
            source = path.read_text(encoding="utf-8")
            parser = AssetLinks(); parser.feed(source)
            for tag, link in parser.links:
                if link.startswith(("http:", "https:", "//")):
                    raise ValueError(f"external resource in {path}: {link}")
                local = link.split("#", 1)[0]
                if local and not local.startswith(("data:", "javascript:")) and not (path.parent / local).exists():
                    raise ValueError(f"missing exported link in {path}: {local}")
            if path.name not in ("index.html", "gallery.html"):
                if "window.labReady" not in source or "plotly.js" not in source.lower():
                    raise ValueError(f"missing bundled runtime/readiness in {path}")
        result.append(item)
    return result


def generate(output_root, selected, *, seed=168, quick=False, dpi=90):
    """Use a fresh dated directory; never overwrite prior runs or public previews."""
    if seed < 0 or not 50 <= dpi <= 160 or not selected or any(group not in GROUPS for group in selected):
        raise ValueError("require nonnegative seed, dpi 50–160 and valid selected experiments")
    selected = list(dict.fromkeys(selected))
    timestamp = datetime.now(timezone.utc)
    folder = Path(output_root) / f"run-{timestamp:%Y%m%dT%H%M%S%fZ}"
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    samples = 1000 if quick else 4000
    torch.set_num_threads(1)
    report = {
        "recorded_at_utc": timestamp.isoformat(), "phase": "visual; distinct from seed-68 analytical run",
        "configuration": {"seed": seed, "quick": quick, "dpi": dpi, "samples": samples,
                          "device": "cpu", "dtype": "float64", "training_updates": 0,
                          "fixture_shape": [1, 6, 16], "heads": 4, "head_width": 4,
                          "matrices_shape": [1, 4, 8], "matrices_heads": 2,
                          "selected": selected, "data": "synthetic normal vectors or explicitly constructed controls",
                          "random_streams": {"fixture": seed, "subspaces": seed + 1, "scaling": seed + 2,
                                             "causal_perturbation": seed + 3, "matrices_input": seed + 4}},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "torch": torch.__version__, "matplotlib": matplotlib.__version__,
                        "plotly": plotly.__version__, "pillow": PIL.__version__},
        "hypotheses": ["Identical projections match PyTorch at tolerance 1e-12.",
                       "Causal prefix outputs resist future perturbation.",
                       "Equal Q/K heads can have distinct value outputs.",
                       "Online block softmax matches the dense head calculation.",
                       "IID raw dot-product variance grows with dimension, scaled variance is approximately one."],
        "results": numerical_report(seed, samples),
        "interpretation_candidates": ["The tested operations are consistent with the stated forward mechanism.",
                                      "Map similarity alone does not determine head-output equivalence.",
                                      "Theoretical payload/count curves describe formulas, not measured hardware."],
        "review_status": "author review required",
        "limitations": ["Random, untrained synthetic inputs; no linguistic semantics or model-quality ranking.",
                        "PCA views are separately fitted and lose dimensions; use exact distances for comparisons.",
                        "Constructed single/shared routing comparison has fixed value channels, not arbitrary output projections.",
                        "Cache/storage excludes metadata, scales, allocator overhead, weights and temporary activations.",
                        "CPU online softmax is an educational calculation, not a FlashAttention GPU implementation.",
                        "Raster decoding and HTML reference checks do not replace browser or author interpretation review."],
    }
    source = Path(__file__).resolve().parent
    report["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted([*source.glob("*.py"), *source.glob("*.js")])}
    result = report["results"]
    experiments = {
        "parity": ("Identical float64 weights agree within 1e-12.", result["parity"],
                   "Tested forward tensors agree.", "No gradient, mixed-precision or optimized-kernel claim."),
        "causal": ("Perturbing token 4 does not affect causal positions 0–3.",
                   {key: result[key] for key in ("causal_prefix_max_abs_change", "bidirectional_prefix_max_abs_change")},
                   "The tested layer blocks future influence.", "Layer-level synthetic intervention, not pipeline leakage evaluation."),
        "equal_maps_distinct_values": ("Equal Q/K heads need not produce equal value outputs.",
                   {key: result[key] for key in ("identical_map_error", "distinct_value_output_l2")},
                   "Map equivalence does not determine output equivalence.", "No downstream task or pruning validation."),
        "scaling": ("IID normal dot-product variance grows with d; scaled variance is near one.", result["scaling"],
                   "Observed sample variances follow the dimensional scaling tendency.", "Finite paired Monte Carlo sample; no task-quality claim."),
        "online_softmax": ("Blockwise online softmax matches the same dense head.", result["online_softmax_max_abs_error"],
                   "The tested online accumulator preserves the dense result.", "CPU identity, not a FlashAttention GPU implementation or benchmark."),
        "constructed_routing": ("Separate fixed value channels can retrieve two differently selected payloads.", result["constructed_routing"],
                   "Independent routing escapes the restricted shared-mixture line segment.", "Manually specified logits and fixed encoding; no universal model superiority."),
    }
    report["experiments"] = {name: {"hypothesis": hypothesis, "configuration": report["configuration"],
                                     "result": measured, "interpretation_candidate": interpretation,
                                     "limitation": limit, "review_status": "author review required"}
                              for name, (hypothesis, measured, interpretation, limit) in experiments.items()}
    artifacts, animations = {}, {}
    # Write the record before gallery validation so its local JSON link resolves.
    (folder / "report.json").write_text(json.dumps(html.serializable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for group in selected:
        print(f"Generating {GROUPS[group]}", flush=True)
        names = []
        if group in html.PAGES:
            names.append(html.write_page(group, folder, seed, samples).name)
        if group in figures.ANIMATIONS:
            animation = figures.ANIMATIONS[group](folder, seed, quick, dpi)
            animations[group] = animation
            names.extend([animation["gif"], *animation["frames"]])
        if group == "heads":
            names.append(figures.attention_png(folder, seed, dpi))
        artifacts[group] = names
    html.write_gallery(folder, selected, artifacts, animations)
    report["artifacts"] = artifacts
    report["validated_exports"] = validate_exports(folder)
    report["generation_seconds"] = time.perf_counter() - start
    report["browser_validation"] = "not performed by generator; record separately after actual interaction checks"
    (folder / "report.json").write_text(json.dumps(html.serializable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Gallery: {folder / 'index.html'}", flush=True)
    print(f"Numerical results: {json.dumps(html.serializable(report['results']), allow_nan=False)}", flush=True)
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="all 15 requested visual questions, plus validation (default)")
    parser.add_argument("--experiment", action="append", choices=tuple(GROUPS), help="repeat to select multiple experiments")
    parser.add_argument("--quick", action="store_true", help="all selected groups, fewer subspace/concatenation stages and 1000 instead of 4000 samples")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--seed", type=int, default=168)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--show", action="store_true", help="open the completed local gallery")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(f"{key:18} {question}" for key, question in GROUPS.items()))
        return None
    if args.all and args.experiment:
        parser.error("choose --all or --experiment")
    if args.seed < 0 or not 50 <= args.dpi <= 160:
        parser.error("require seed >=0 and dpi 50–160")
    folder = generate(args.output_dir, args.experiment or list(GROUPS), seed=args.seed, quick=args.quick, dpi=args.dpi)
    if args.show:
        webbrowser.open((folder / "index.html").resolve().as_uri())
    return folder


if __name__ == "__main__":
    main()
