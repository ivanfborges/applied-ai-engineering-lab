"""Generate the Day 70 offline visual lab; CPU, synthetic data, no training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import time
import webbrowser

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

try:
    import matplotlib
    import numpy as np
    import PIL
    from PIL import Image
    import plotly
    import torch
except ImportError as error:
    raise SystemExit('Install canonical extras in an isolated environment: python -m pip install -e ".[dev,deep-learning]"') from error

from visual_core import (Config, backward_decomposition, dense_diagnostics, ffn_fixture,
                         geometry_fixture, json_safe, normalization_interventions,
                         numerical_checks, residual_fixture, swiglu_fixture, transformer_diagnostics)
import visual_figures
import visual_html

TOPICS = tuple(visual_figures.RENDERERS)


class _Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ("src", "href") and value)


def validate_exports(folder: Path):
    """Decode every image frame and check resources; not browser validation."""
    records = []
    for path in sorted(folder.iterdir()):
        if path.suffix not in (".png", ".gif", ".html", ".csv"):
            continue
        item = {"name": path.name, "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix in (".png", ".gif"):
            with Image.open(path) as picture:
                item.update(width=picture.width, height=picture.height, frames=getattr(picture, "n_frames", 1))
                for frame in range(item["frames"]):
                    picture.seek(frame)
                    picture.load()
                if path.suffix == ".gif" and (item["frames"] < 2 or picture.info.get("loop") != 0):
                    raise ValueError(f"Invalid animation: {path}")
        elif path.suffix == ".html":
            text = path.read_text(encoding="utf-8")
            parser = _Resources()
            parser.feed(text)
            for link in parser.links:
                if link.startswith(("http:", "https:", "//")):
                    raise ValueError(f"External HTML resource: {path}: {link}")
                local = link.split("#", 1)[0]
                if local and not (path.parent / local).is_file():
                    raise ValueError(f"Missing gallery resource: {path}: {local}")
            if path.name != "index.html" and ('id="lab-data"' not in text or "plotly.js" not in text):
                raise ValueError(f"Expected standalone executable Plotly page: {path}")
        records.append(item)
    return records


def _write_csv(path: Path, rows):
    rows = [{k: v for k, v in row.items() if k != "gradient_samples"} for row in rows]
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _gallery(folder, selected, generated, config):
    sections = []
    for group in selected:
        question, explanation = visual_html.EXPLANATIONS[group]
        links = "".join(f'<li><a href="{escape(path.name)}">{escape(path.name)}</a> · {path.stat().st_size / 1024:.1f} KiB</li>' for path in generated[group])
        sections.append(f'<section><h2>{escape(group)}: {escape(question)}</h2><p>{escape(explanation)}</p><ul>{links}</ul></section>')
    source = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
              f'<title>Day 70 visual laboratory</title><style>{visual_html.STYLE}</style></head><body><main>'
              '<div class="badge">LOCAL EXPERIMENT GALLERY · AUTHOR REVIEW REQUIRED</div><h1>Inside a Transformer block</h1>'
              f'<p>Synthetic seed {config.seed}. Each HTML is self-contained and works offline. PNG/GIF/CSV files stay alongside this index. '
              'No training, pretrained model, external data or latency benchmark.</p>' + "".join(sections) +
              '<p><a href="run.json">Dated configuration, versions, checks, experiment hypotheses and artifact manifest</a></p></main></body></html>')
    (folder / "index.html").write_text(source, encoding="utf-8")


def generate(output_root: Path, selected, config: Config, formats=("png", "gif", "html")):
    config.validate()
    selected = list(dict.fromkeys(selected))
    if not selected or any(group not in TOPICS for group in selected):
        raise ValueError("Select at least one known topic")
    if not formats or any(fmt not in ("png", "gif", "html") for fmt in formats):
        raise ValueError("formats must contain png, gif and/or html")
    torch.set_num_threads(1)
    recorded = datetime.now(timezone.utc)
    folder = Path(output_root) / f"visual-run-{recorded:%Y%m%dT%H%M%S%fZ}"
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checks = numerical_checks(config.seed)
    for value in checks["numpy_pytorch"].values():
        if value > 1e-10:
            raise AssertionError("NumPy/PyTorch numerical agreement failed")
    if checks["ffn_other_tokens_change"] != 0 or checks["backward_decomposition_error"] > 1e-10:
        raise AssertionError("Token independence or backward decomposition failed")
    data = {"ffn": ffn_fixture(config.seed), "geometry": geometry_fixture(config.seed, 48 if config.quick else 96),
            "residual": residual_fixture(config.seed), "swiglu": swiglu_fixture(config.seed),
            "normalization": normalization_interventions(config.seed), "block": backward_decomposition(config.seed),
            "norm_vectors": np.random.default_rng(config.seed + 6).normal(size=(48 if config.quick else 128, 3)),
            "activation_inputs": np.random.default_rng(config.seed + 7).normal(size=512 if config.quick else 2048)}
    experiments = {}
    if "gradients" in selected:
        print("Computing paired dense-network gradients...", flush=True)
        data["dense"] = dense_diagnostics(config)
        _write_csv(folder / "dense_gradient_measurements.csv", data["dense"])
        experiments["dense"] = {"hypothesis": "Residual identity paths alter measured initialization-time gradient profiles.",
                                "rows": len(data["dense"]), "depths": config.depths, "seeds": config.seeds,
                                "configuration": "independently initialized Linear(d,d)+tanh; same prefix, input, random target; mean MSE; no normalization",
                                "results": [r for r in data["dense"] if r["layer"] in (0, r["depth"])],
                                "interpretation_candidate": "Compare actual profiles, signs and seed variation before inferring any benefit; author review required.",
                                "limitation": "Random-target initialization, zero optimizer updates; no convergence evidence."}
    if "pre-post-norm" in selected:
        print("Computing paired Transformer depth/readout diagnostics...", flush=True)
        data["transformer"] = transformer_diagnostics(config)
        _write_csv(folder / "transformer_measurements.csv", data["transformer"])
        experiments["transformer"] = {"hypothesis": "Norm placement and final readout change gradients and relative update magnitudes.",
                                      "rows": len(data["transformer"]), "depths": config.depths, "seeds": config.seeds,
                                      "configuration": {"batch": 2, "tokens": config.tokens, "d_model": config.d_model,
                                                        "heads": 4, "d_ff": config.d_ff, "dtype": "float64", "eps": 1e-5,
                                                        "dropout": 0., "attention": "bidirectional, no mask",
                                                        "initialization": "PyTorch module defaults, same prefix and copied Pre/Post parameters",
                                                        "inputs": "standard normal; local CPU generator seed+200, then target from same stream",
                                                        "conditions": ["pre", "post", "pre+final-LN non-affine readout"], "loss": "mean MSE", "optimizer_steps": 0},
                                      "results": [r for r in data["transformer"] if r["layer"] in (0, r["depth"])],
                                      "interpretation_candidate": "Separate normalized output scale, readout gradients and internal stream growth; author review required.",
                                      "limitation": "Synthetic initialization only; seed ranges are not confidence intervals or trained-model rankings."}
    generated = {}
    # Renderers produce a coherent group together; formats filter before expensive rendering.
    for group in selected:
        print(f"Exporting {group}...", flush=True)
        paths = []
        if "png" in formats or "gif" in formats:
            paths.extend(visual_figures.RENDERERS[group](folder, config, data))
        if "html" in formats:
            for filename, mode in visual_html.PAGES.get(group, []):
                paths.append(visual_html.write_page(folder, filename, mode, config, data))
        generated[group] = paths
    # CSVs belong to the corresponding learning section.
    if "gradients" in selected:
        generated["gradients"].append(folder / "dense_gradient_measurements.csv")
    if "pre-post-norm" in selected:
        generated["pre-post-norm"].append(folder / "transformer_measurements.csv")
    report = {
        "recorded_at_utc": recorded.isoformat(), "phase": "Day 70 visual lab; separate from analytical seed-70 runs",
        "configuration": asdict(config), "topics": selected, "requested_formats": formats,
        "format_policy": "raster groups export paired PNG/GIF together; html-only skips raster rendering",
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "torch": torch.__version__, "matplotlib": matplotlib.__version__,
                        "plotly": plotly.__version__, "pillow": PIL.__version__, "torch_threads": torch.get_num_threads()},
        "numerical_checks": checks, "experiments": experiments,
        "fixture_policy": {"ffn": "4x8 -> 4x32 -> 4x8; seed", "geometry": "3->12->3; seed+1",
                           "residual": "8 layers in original 3D; seed+2", "swiglu": "8->nearest-budget-width->8; seed+3; biases",
                           "normalization": "B=2,T=4,D=3; seed+5; PyTorch BN running moments",
                           "block": "B=1,T=4,d=8,heads=2,h=32; complete actual Pre-LN graph"},
        "interpretation_status": "author review required", "public_asset_selection": "pending human review; all exports ignored",
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("visualizations.py", "visual_core.py", "visual_figures.py", "visual_html.py", "visual_controls.js", "example.py", "from_scratch.py")},
        "optional_unexecuted": ["training", "mixed precision", "latency benchmarking", "pretrained models"],
    }
    # Save before the gallery so its provenance link resolves during resource validation.
    (folder / "run.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _gallery(folder, selected, generated, config)
    report["artifacts"] = validate_exports(folder)
    report["generation_seconds"] = time.perf_counter() - started
    (folder / "run.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return folder, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--topic", nargs="+", choices=TOPICS)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--seed", type=int, default=270)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--expansion", type=float, default=4.)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--quick", action="store_true", help="two seeds, 8 tokens, 6/12/requested-depth sweeps, fewer display samples/frames")
    parser.add_argument("--show", action="store_true", help="open only the gallery, never all HTML tabs")
    parser.add_argument("--html-only", action="store_true", help="compute selected topics and export HTML/CSV without raster rendering")
    parser.add_argument("--validate-only", action="store_true", help="small numerical checks only, no output directory")
    args = parser.parse_args()
    if args.list:
        print("\n".join(TOPICS))
        return
    config = Config(args.seed, args.device, args.depth, args.d_model, args.expansion, args.quick)
    try:
        config.validate()
        torch.set_num_threads(1)
        if args.validate_only:
            print(json.dumps(json_safe(numerical_checks(config.seed)), indent=2, allow_nan=False))
            return
        selected = args.topic or (TOPICS if args.all or args.quick else ("ffn",))
        folder, report = generate(args.output_dir, selected, config, ("html",) if args.html_only else ("png", "gif", "html"))
    except (ValueError, FloatingPointError) as error:
        parser.error(str(error))
    print(f"Generated {len(report['artifacts'])} validated artifacts in {folder}")
    print("Interpretation and public preview selection: author review required")
    if args.show:
        webbrowser.open((folder / "index.html").resolve().as_uri())


if __name__ == "__main__":
    main()
