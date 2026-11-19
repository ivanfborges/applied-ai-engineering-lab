"""Generate the Day 62 CPU visual lab in a fresh local run directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from visual_core import LabConfig, np, run_lab, torch
import visual_figures as visuals


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def create_run_directory(parent: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = parent / f"run-{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def generate(config: LabConfig, parent: Path, *, mode: str = "quick", interactive: bool = False) -> Path:
    config.validate()
    if mode not in ("quick", "all", "interactive"):
        raise ValueError("mode must be quick, all or interactive")
    if mode in ("all", "interactive") or interactive:
        visuals.plotly_modules()  # Fail before training if the requested HTML dependency is missing.
    destination = create_run_directory(parent)
    lab = run_lab(config)
    visuals.theme()
    if mode in ("quick", "all"):
        visuals.dataset_visual(lab, destination, config.frames)
        visuals.loss_visual(destination)
        visuals.training_visual(lab, destination)
        visuals.dashboard_static(lab, destination)
    if mode == "all":
        visuals.forward_visual(lab, destination)
        visuals.backward_visual(lab, destination)
        visuals.landscape_contour(lab, destination)
        visuals.optimizer_visual(lab, destination)
        visuals.boundary_visual(lab, destination, config.frames)
        visuals.mode_visual(lab, destination)
    if mode in ("all", "interactive") or interactive:
        visuals.landscape_html(lab, destination, config.frames)
        visuals.optimizer_html(lab, destination)
        visuals.dashboard_html(lab, destination)
    (destination / "numerical_records.json").write_text(
        json.dumps(lab, default=json_value, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    import matplotlib
    import PIL
    import importlib.metadata
    try:
        plotly_version = importlib.metadata.version("plotly")
    except importlib.metadata.PackageNotFoundError:
        plotly_version = None
    summary = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
        "configuration": asdict(config), "versions": {"python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "Pillow": PIL.__version__, "plotly": plotly_version},
        "runtime": {"device": "cpu", "torch_threads": torch.get_num_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "hypothesis": "Measured updates change the decision function; optimizer, rate and batch size change the path on a fixed objective.",
        "models": {name: {"parameters": lab[name]["parameters"], "best_epoch": lab[name]["best_epoch"],
                          "best_metrics": lab[name]["history"][lab[name]["best_epoch"]],
                          "total_steps": lab[name]["total_steps"]} for name in ("mlp", "linear")},
        "selected": lab["selected"], "selected_test": lab["selected_test"],
        "logistic": {name: {"initial_loss": float(run["loss"][0]), "final_loss": float(run["loss"][-1]),
                             "uphill_steps": run["uphill_steps"]} for name, run in lab["logistic"]["runs"].items()},
        "interpretation_candidate": "Inspect the measured traces and held-out metrics within this configuration; author review required.",
        "limitations": ["Single synthetic seed; no universal ranking or production claim.",
                        "Visual protocol differs from the historical analytical 2-16-1 experiment.",
                        "Dashboard mixes pre- and post-update quantities with explicit labels, never different steps.",
                        "Same number of updates with different batch sizes is not equal compute or equal examples.",
                        "HTMLs bundle Plotly; replay is prerecorded and does not retrain in the browser."],
        "artifacts": [{"file": p.name, "bytes": p.stat().st_size,
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in sorted(destination.iterdir()) if p.is_file()],
    }
    (destination / "manifest.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true", help="Essential PNGs and one small GIF; reduced default training")
    modes.add_argument("--all", action="store_true", help="All ten visualizations, including GIFs and offline HTML")
    parser.add_argument("--interactive", action="store_true", help="Generate offline HTML; can accompany --quick")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "assets")
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    args = parser.parse_args()
    mode = "all" if args.all else "interactive" if args.interactive and not args.quick else "quick"
    quick = mode == "quick"
    config = LabConfig(seed=args.seed, samples=args.samples if args.samples is not None else 240 if quick else 600,
                       epochs=args.epochs if args.epochs is not None else 16 if quick else 40,
                       batch_size=args.batch_size, frames=args.frames if args.frames is not None else 12 if quick else 20,
                       grid_size=args.grid_size if args.grid_size is not None else 25 if quick else 35,
                       learning_rate=args.learning_rate)
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    destination = generate(config, args.output_dir, mode=mode, interactive=args.interactive)
    print(f"Generated {mode} lab: {destination}")
    print("See manifest.json for actual configuration, metrics, versions and output hashes.")


if __name__ == "__main__":
    main()
