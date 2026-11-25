"""Generate the offline Day 69 positional learning lab on a bounded local CPU."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import time
import webbrowser

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import matplotlib
import numpy as np
import PIL
from PIL import Image
import plotly
import sklearn

from visual_core import Config, json_safe, numerical_report
import visual_figures as figures
import visual_html as html


GROUPS = tuple(html.PAGES)


class AssetLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ("src", "href") and value)


def validate_exports(folder):
    """Decode every raster frame and resolve page resources; no browser claim."""
    result = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix not in (".png", ".gif", ".html"):
            continue
        item = {"path": path.relative_to(folder).as_posix(), "bytes": path.stat().st_size}
        if path.suffix in (".png", ".gif"):
            with Image.open(path) as image:
                item.update(width=image.width, height=image.height, frames=getattr(image, "n_frames", 1))
                for frame in range(item["frames"]):
                    image.seek(frame)
                    image.load()
                if path.suffix == ".gif" and (item["frames"] < 2 or image.info.get("loop") != 0):
                    raise ValueError(f"Animation must contain at least two frames and loop: {path}")
        else:
            source = path.read_text(encoding="utf-8")
            parser = AssetLinks()
            parser.feed(source)
            for link in parser.links:
                if link.startswith(("http:", "https:", "//")):
                    raise ValueError(f"External page dependency: {path}: {link}")
                local = link.split("#", 1)[0]
                if local and not (path.parent / local).is_file():
                    raise ValueError(f"Missing local page dependency: {path}: {local}")
            if path.name != "index.html" and 'id="lab-data"' not in source:
                raise ValueError(f"Missing executable numerical page payload: {path}")
        result.append(item)
    return result


def generate(output_root, selected, config, formats=("html", "png", "gif")):
    config.validate()
    selected = list(dict.fromkeys(selected))
    if not selected or any(name not in GROUPS for name in selected):
        raise ValueError("Select at least one known demonstration.")
    if not formats or any(name not in ("html", "png", "gif") for name in formats):
        raise ValueError("formats must contain html, png and/or gif.")
    recorded = datetime.now(timezone.utc)
    folder = Path(output_root) / f"run-{recorded:%Y%m%dT%H%M%S%fZ}"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    results = numerical_report(config)
    report = {
        "recorded_at_utc": recorded.isoformat(), "phase": "Day 69 visual lab; separate from analytical seed-69 record",
        "configuration": asdict(config), "selected": selected, "formats": list(formats),
        "scene_configuration": {
            "random_streams": {"content_and_projections": config.seed, "rotary_planes": config.seed+1, "precision_qk": config.seed+2},
            "permutation_tokens": 6, "permutation_indices": [0,4,2,3,1,5],
            "comparison_static_tokens": min(config.seq_len,64), "comparison_report_tokens": min(config.seq_len,128),
            "comparison_html_initial_tokens": min(config.seq_len,32), "html_content_fixture_rows": 128,
            "comparison_mask": "causal for all six methods; permutation intervention is unmasked",
            "pca_reference_positions": max(128,config.seq_len),
            "rope_2d": {"query_position":2,"key_position":5,"frequency":.5,"query_angle":.3,"key_angle":1.,"query_norm":1.,"key_norm":.8},
            "alibi": {"keys":32,"query":24,"flat_logits":0.,"distant_key":2,"strong_content_logit":12.,"slopes":[.02,.1,.5,1.]},
            "extrapolation": {"hypothetical_training_length":128,"context_lengths":[128,256,512,1024,4096],"sampled_positions":96,"actual_model_trained":False},
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "matplotlib": matplotlib.__version__, "plotly": plotly.__version__,
                        "scikit_learn": sklearn.__version__, "pillow": PIL.__version__},
        "results": results,
        "experiments": {
            "permutation": {"hypothesis": "Shared unmasked content-only attention is permutation equivariant; fixed positions generally break that equality.",
                            "result": {key: value for key, value in results.items() if key.startswith("permutation")},
                            "interpretation_candidate": "Endpoint tensors agree with the stated symmetry in this synthetic fixture.",
                            "limitation": "No training, causal mask or language-quality evaluation in this intervention."},
            "sinusoidal": {"hypothesis": "Paired sinusoidal norms and equal-displacement dot products follow their identities.",
                           "result": {key: value for key, value in results.items() if key.startswith("sinusoidal")},
                           "interpretation_candidate": "Measured residuals are consistent with the identities at float64 precision.",
                           "limitation": "Pure position vectors, not semantic similarity or arbitrary projected attention."},
            "geometry": {"hypothesis": "A shared uncentered linear projection preserves vector addition.",
                         "result": {key: value for key, value in results.items() if "projection" in key or "pca" in key},
                         "interpretation_candidate": "Addition is preserved; PCA variance reports retained variance, not all original distances.",
                         "limitation": "Per-width PCA bases are incompatible; prefix views reuse one fixed basis."},
            "rope": {"hypothesis": "Fixed-content rotary dots are preserved under a common shift; construction precision matters at large origins.",
                     "result": {"common_shift_error": results["rope_2d_common_shift_error"], "precision": results["precision"]},
                     "interpretation_candidate": "Small-shift invariance and large-origin residuals separate algebra from floating-point construction.",
                     "limitation": "The named dtype covers phase, trig and dot; float32 also rounds input content. No trained-context evaluation."},
            "alibi_comparison": {"hypothesis": "Causal probabilities normalize; supplied positive slopes impose recency bias with flat content.",
                                 "result": {"normalization": results["method_probability_row_errors"], "flat_content": results["alibi_flat_content"]},
                                 "interpretation_candidate": "Stronger supplied slopes reduce expected distance in the flat-content fixture.",
                                 "limitation": "Same untrained weights do not establish a ranking of trained methods; slope schedule is illustrative."},
            "rag": {"hypothesis": "Moving one evidence block can keep the context budget fixed.",
                    "result": "Interactive prompt-construction design only; no LLM evaluation or accuracy measured.",
                    "interpretation_candidate": "The page specifies an evaluation protocol, not evidence of a position effect on quality.",
                    "limitation": "Whitespace display units are not model tokens; no model is invoked."},
        },
        "review_status": "author review required", "training_updates": 0,
        "limitations": ["Synthetic deterministic content and random absolute tables; no trained language model.",
                        "128 is a hypothetical training boundary, not an actual trained model's window.",
                        "Local resource/raster checks do not establish browser behavior, production readiness or scientific approval."],
        "browser_validation": "not performed by generator; record separately after actual browser interactions",
    }
    for experiment in report["experiments"].values():
        experiment["configuration"] = asdict(config)
        experiment["review_status"] = "author review required"
    source = Path(__file__).resolve().parent
    report["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted([*source.glob("*.py"), *source.glob("*.js")])}
    pages = {name: html.PAGES[name] for name in selected} if "html" in formats else {}
    if pages:
        html.write_runtime(folder)
    artifacts = {}
    for group in selected:
        print(f"Generating {group}: {html.DESCRIPTIONS[group][0]}", flush=True)
        names = []
        if "html" in formats:
            for name in pages[group]:
                html.write_page(folder, group, config, name, pages)
                names.append(name)
        data = figures.prepare(group, config)
        if "png" in formats:
            for name in figures.STATICS.get(group, []):
                names.append(figures.save_static(folder, group, name, config, data))
        if "gif" in formats:
            for name in figures.ANIMATIONS.get(group, []):
                names.append(figures.save_animation(folder, group, name, config, data))
        artifacts[group] = names
    report["artifacts"] = artifacts
    record = folder / "report.json"
    record.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    html.write_gallery(folder, selected, artifacts)
    report["validated_exports"] = validate_exports(folder)
    report["generation_seconds"] = time.perf_counter() - started
    record.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Gallery: {folder / 'index.html'}", flush=True)
    print(f"Record: {record}", flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--demo", choices=GROUPS, action="append")
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--base", type=float, default=10000.)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=269)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent/"outputs"/"visual")
    parser.add_argument("--show", action="store_true", help="Open only the generated gallery in a browser.")
    parser.add_argument("--save", action="store_true", help="Explicit export intent; exports are also saved without this flag.")
    parser.add_argument("--format", choices=("all", "html", "png", "gif"), action="append")
    parser.add_argument("--quick", action="store_true", help="Default to HTML/PNG only; cap frames at 8 and dpi at 75.")
    args = parser.parse_args()
    if args.list or not (args.all or args.demo or args.quick):
        for group in GROUPS:
            print(f"{group:22s} {html.DESCRIPTIONS[group][0]}")
        return
    config = Config(args.seq_len, args.dim, args.base, args.frames, args.fps, args.seed, args.dpi)
    if args.quick:
        config = replace(config, frames=min(config.frames, 8), dpi=min(config.dpi, 75))
    selected = list(GROUPS) if args.all or not args.demo else args.demo
    formats = args.format or (["html", "png"] if args.quick else ["all"])
    if "all" in formats:
        formats = ["html", "png", "gif"]
    try:
        folder = generate(args.output_dir, selected, config, formats)
    except ValueError as exc:
        parser.error(str(exc))
    if args.show:
        webbrowser.open((folder / "index.html").resolve().as_uri())


if __name__ == "__main__":
    main()
