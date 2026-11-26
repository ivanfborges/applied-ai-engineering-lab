"""Execute 14 bounded synthetic fine-tuning demos and export an offline gallery."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

try:
    import matplotlib
    import numpy as np
    import PIL
    import plotly
    import torch
    from visual_core import DEMOS, LabConfig, compute_demo, serializable
    from visual_render import render_demo
    from visual_html import write_gallery
except ImportError as exc:
    raise SystemExit('Missing dependency. From the repository root run: python -m pip install -e ".[deep-learning,dev]"') from exc


HYPOTHESES = {
    "landscape": "Adaptation starts at the optimized base location on a different objective.",
    "sft": "Cross-entropy updates increase the demonstrated response probability.",
    "instruction": "Single-task and multitask adaptation can affect numerical rules differently.",
    "reward": "BT fitting approximates synthetic utility comparisons; extra flips can change held-out loss.",
    "rlhf": "Positive-beta exact optimization approaches the analytic regularized policy.",
    "dpo": "A preference ratio can improve without identifying absolute response quality.",
    "comparison": "Population BT-DPO and exact reward optimization can approach the same optimum under matched assumptions.",
    "lora": "Rank limits the reconstruction family; finite training need not reach the SVD bound.",
    "geometry": "A bounded-rank family can exclude the unconstrained matrix optimum.",
    "efficiency": "Small ranks reduce trainable counts under the stated architecture.",
    "qlora": "Uniform low-bit reconstruction changes weights; rank-2 updates may reduce reconstruction error.",
    "forgetting": "Adapting to a conflicting rule can reduce performance on the earlier rule, including with LoRA.",
    "hacking": "Maximizing a misspecified length proxy can reduce the separately defined utility.",
    "pipeline": "SFT and preference updates can change style probabilities while retaining synthetic task identity.",
}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(serializable(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Run all 14 demos (also the default)")
    selection.add_argument("--demo", choices=DEMOS, action="append", help="Repeat for multiple demos")
    parser.add_argument("--quick", action="store_true", help="80 updates, 10 GIF frames, matrices 16x16, one repetition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list", action="store_true", help="List demo names without training or rendering")
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing paths are refused")
    parser.add_argument("--reuse-run", type=Path, help="Render saved JSON from an earlier run; never retrain")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(f"{name:12s} {description}" for name, description in DEMOS.items()))
        return
    try:
        config = LabConfig(args.seed, args.quick)
    except ValueError as exc:
        parser.error(str(exc))
    names = list(dict.fromkeys(args.demo or DEMOS))
    source_manifest = None
    if args.reuse_run:
        manifest_path = args.reuse_run / "manifest.json"
        if not manifest_path.is_file():
            parser.error("--reuse-run must contain manifest.json")
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = LabConfig(**source_manifest["configuration"])
        for name in names:
            if name not in source_manifest["demos"]:
                parser.error(f"Saved run does not contain demo {name}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = args.output_dir or Path(__file__).resolve().parent / "outputs" / f"visual-{stamp}"
    if root.exists():
        parser.error("Output directory exists; select a new path to preserve historical runs")
    root.mkdir(parents=True)
    measurements = root / "experiments"; measurements.mkdir()
    versions = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                "numpy": np.__version__, "matplotlib": matplotlib.__version__, "plotly": plotly.__version__, "pillow": PIL.__version__}
    reports, sources = {}, {}
    print(f"Day 74 | seed={config.seed} | quick={config.quick} | {root}", flush=True)
    for name in names:
        print(f"{'Reuse' if args.reuse_run else 'Calculate/train'} {name}", flush=True)
        if args.reuse_run:
            source = args.reuse_run / "experiments" / f"{name}.json"
            if digest(source) != source_manifest["measurement_sha256"][name]:
                parser.error(f"Saved measurement digest mismatch: {name}")
            report = json.loads(source.read_text(encoding="utf-8"))
            sources[name] = {"run": str(args.reuse_run.resolve()), "sha256": digest(source)}
        else:
            report = compute_demo(name, config)
            report.update(executed_at_utc=datetime.now(timezone.utc).isoformat(), versions=versions,
                          hypothesis=HYPOTHESES[name], interpretation_candidate="Inspect the measured trajectory against the stated hypothesis and limitations; author review required")
        write_json(measurements / f"{name}.json", report)
        reports[name] = report
        print(f"Render {name}", flush=True)
        render_demo(name, root, report, config)
    artifacts = write_gallery(root, reports)
    source_names = ("visual_core.py", "visual_render.py", "visual_html.py", "visualize_fine_tuning.py", "example.py", "from_scratch.py")
    manifest = {"schema": 1, "rendered_at_utc": datetime.now(timezone.utc).isoformat(),
                "configuration": asdict(config), "actual_settings": {"updates": config.steps, "frames": config.frames,
                "matrix_dimension": config.dimension, "samples": config.samples, "instruction_forgetting_repetitions": config.repetitions},
                "versions": versions, "demos": names, "reused_measurements": sources,
                "measurement_sha256": {name: digest(measurements / f"{name}.json") for name in names},
                "source_sha256": {name: digest(Path(__file__).parent / name) for name in source_names},
                "artifact_sha256": {name: digest(root / name) for name in artifacts}, "artifacts": artifacts,
                "interpretation_status": "author review required", "public_previews": [],
                "limitation": "Synthetic local CPU calculations/training; no pretrained LLM, human feedback, GPU or production benchmark"}
    write_json(root / "manifest.json", manifest)
    print(f"Generated {len(artifacts)} artifacts. Open {root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
