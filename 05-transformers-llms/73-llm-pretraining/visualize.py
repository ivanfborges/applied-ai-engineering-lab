"""Run measured pretraining experiments or render offline educational visualizations."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

try:
    from visual_core import LabConfig, Scaling, cached_experiments
    from visual_render import (render_attention, render_compute, render_data, render_embeddings,
                               render_generalization, render_leakage, render_objectives,
                               render_probabilities, render_scaling, render_softmax,
                               render_training, write_gallery)
except ImportError as exc:
    raise SystemExit('Install the canonical dependencies: python -m pip install -e ".[deep-learning,dev]"') from exc


SECTIONS = ("objectives", "training", "probabilities", "softmax", "embeddings", "attention",
            "generalization", "leakage", "scaling", "compute", "data")
NEEDS = {
    "objectives": ["clm", "mlm"], "training": ["clm", "mlm"], "probabilities": ["clm"],
    "embeddings": ["clm"], "attention": ["clm", "mlm"],
    "generalization": ["clm", "large_restricted", "longer", "more_unique"],
    "leakage": ["clm", "unmasked"],
    "data": ["clm", "duplicated", "single_grammar", "conflicting", "domain_imbalanced"],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--section", choices=SECTIONS, action="append", help="Repeat to select several sections; defaults to objectives")
    parser.add_argument("--quick", action="store_true", help="60 updates, width 24, one layer, batch 16, seven snapshots")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-run", type=Path, help="Read experiment JSON from this prior run's experiments/ directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--render-only", action="store_true", help="Require existing measurements; never train")
    for flag in ("steps", "width", "heads", "layers", "ffn", "batch-size", "snapshots"):
        parser.add_argument("--" + flag, type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--masking-probability", type=float)
    for name in ("E", "A", "B", "alpha", "beta"):
        parser.add_argument("--" + name, type=float)
    args = parser.parse_args(argv)
    sections = list(SECTIONS) if args.all else list(dict.fromkeys(args.section or ["objectives"]))
    values = {"seed": args.seed, "device": args.device}
    if args.quick:
        values.update(steps=60, width=24, layers=1, ffn=48, batch_size=16, snapshots=7)
    for key in ("steps", "width", "heads", "layers", "ffn", "batch_size", "snapshots", "learning_rate", "masking_probability"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    try:
        config = LabConfig(**values)
        scaling = Scaling(**{key: getattr(args, key) for key in ("E", "A", "B", "alpha", "beta") if getattr(args, key) is not None})
    except ValueError as exc:
        parser.error(str(exc))
    root = args.output_dir or Path(__file__).resolve().parent / "outputs" / datetime.now(timezone.utc).strftime("visual-%Y%m%dT%H%M%S%fZ")
    root.mkdir(parents=True, exist_ok=True)
    experiment_folder = (args.reuse_run or root) / "experiments"
    names = list(dict.fromkeys(name for section in sections for name in NEEDS.get(section, [])))
    print(f"Day 73 · sections={','.join(sections)} · output={root}", flush=True)
    if args.render_only or args.reuse_run:
        reports = {}
        for name in names:
            path = experiment_folder / f"{name}.json"
            if not path.exists():
                parser.error(f"Missing saved run {path}; train first or select an analytical section")
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
            if reports[name].get("schema") != 1:
                parser.error(f"Unsupported measurement schema in {path}")
        print(f"Reused {len(reports)} saved experiments; no training", flush=True)
    else:
        reports = cached_experiments(config, experiment_folder, names)
    if args.train_only:
        print("Training complete; rendering not requested", flush=True)
        return
    for section in sections:
        print(f"Render {section}", flush=True)
        if section == "objectives":
            render_objectives(root, reports)
        elif section == "training":
            render_training(root, reports)
        elif section == "probabilities":
            render_probabilities(root, reports["clm"])
        elif section == "softmax":
            render_softmax(root)
        elif section == "embeddings":
            render_embeddings(root, reports["clm"])
        elif section == "attention":
            render_attention(root, reports)
        elif section == "generalization":
            render_generalization(root, reports)
        elif section == "leakage":
            render_leakage(root, reports)
        elif section == "scaling":
            render_scaling(root, scaling)
        elif section == "compute":
            render_compute(root, scaling)
        elif section == "data":
            render_data(root, reports)
    assets = write_gallery(root)
    provenance = {"rendered_at_utc": datetime.now(timezone.utc).isoformat(), "requested_configuration": asdict(config),
                  "scaling_assumptions": asdict(scaling), "sections": sections,
                  "measurement_sources": {name: {"path": str((experiment_folder / f"{name}.json").resolve()),
                                                 "sha256": hashlib.sha256((experiment_folder / f"{name}.json").read_bytes()).hexdigest(),
                                                 "executed_at_utc": report["executed_at_utc"],
                                                 "actual_configuration": report["configuration"]} for name, report in reports.items()},
                  "interpretation_status": "author review required", "assets": assets}
    (root / "render_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Generated {len(assets)} visual artifacts; open {root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
