"""Day 55 CPU visual lab: quick, full, interactive and standalone 3D modes."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"


def verify_outputs(paths):
    from PIL import Image
    records = []
    for candidate in paths:
        path = Path(candidate)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty artifact: {path}")
        record = {"file": path.name, "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix in (".png", ".gif"):
            with Image.open(path) as picture:
                count = getattr(picture, "n_frames", 1)
                durations = []
                for frame in range(count):
                    picture.seek(frame)
                    picture.load()
                    durations.append(picture.info.get("duration", 0))
                record.update(width=picture.width, height=picture.height, decoded_frames=count)
                if path.suffix == ".gif":
                    if count < 2:
                        raise ValueError(f"Not an animation: {path}")
                    record.update(duration_ms=sum(durations), loop=picture.info.get("loop"))
        elif path.suffix == ".html":
            html = path.read_text(encoding="utf-8")
            if "Plotly.newPlot" not in html or re.search(r'<script[^>]+src\s*=', html, re.I):
                raise ValueError(f"HTML must embed its Plotly runtime: {path}")
            record["embedded_runtime"] = True
        records.append(record)
    return records


def build_lab(output=DEFAULT_OUTPUT, mode="quick", frames=32, epochs=400, grid=61):
    from visual_math import autograd_checks, bounded_integer, dying_neuron, gradient_chains, train_moons
    if mode not in ("quick", "full", "3d"):
        raise ValueError("batch mode must be quick, full or 3d")
    bounded_integer(frames, "frames", 12, 80)
    bounded_integer(epochs, "epochs", 1, 2000)
    bounded_integer(grid, "grid", 15, 121)
    # Select the noninteractive backend before importing pyplot in batch mode.
    import matplotlib
    matplotlib.use("Agg")
    from visual_render import Renderer
    renderer = Renderer(output, frames=frames)
    paths = []
    print("Checking analytical derivatives against CPU PyTorch autograd ...", flush=True)
    checks = autograd_checks()
    if mode in ("quick", "full"):
        for name in ("activations", "gradients", "dying", "gelu", "temperature", "jacobian", "attention"):
            print(f"Generating {name} ...", flush=True)
            paths.append(getattr(renderer, name)())
    if mode == "full":
        print("Generating four bounded Pillow animations ...", flush=True)
        paths.append(renderer.transformation())
        paths.extend([renderer.gradients(animate=True), renderer.dying(animate=True), renderer.temperature(animate=True)])
    study = None
    if mode in ("full", "3d"):
        print(f"Training matched CPU models ({epochs} updates each) ...", flush=True)
        study = train_moons(epochs=epochs, grid=grid)
        paths.append(renderer.boundaries(study))
        from interactive_3d import export_3d
        paths.extend(export_3d(renderer.output, study))
    neuron = dying_neuron()
    chains = gradient_chains()
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "matplotlib", "Pillow", "plotly", "scikit-learn", "torch")}
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "executor": "CPU local Python visual laboratory", "interpretation_status": "author review required",
        "hypotheses": ["Analytical local derivatives agree with float64 autograd away from kinks.",
                       "Controlled backward sensitivities multiply both local activation slopes and weights.",
                       "A nonzero Leaky ReLU slope on negative inputs allows updates in this constructed inactive-neuron example.",
                       "Matched nonlinear MLPs can represent curved boundaries; training outcomes are configuration-specific."],
        "versions": {"python": platform.python_version(), **versions},
        "configuration": {"mode": mode, "frames_requested": frames, "seed": 55, "epochs": epochs if study else None,
                          "grid": grid if study else None, "outputs": str(renderer.output)},
        "autograd": checks,
        "controlled_chains": [{"label": c["label"], "local_derivative": c["local_derivative"],
                               "depth_20_gradient_magnitude": float(c["magnitude"][-1])} for c in chains],
        "dying_neuron": {name: {"initial_loss": h[0]["loss"], "final_loss": h[-1]["loss"],
                                 "initial_gradient_norm": h[0]["gradient_norm"], "final_bias": h[-1]["b"]}
                         for name, h in neuron["traces"].items()},
        "files": verify_outputs(paths),
        "limitations": ["Synthetic controlled demonstrations; no production or universal activation-quality claim.",
                        "Single model seed/split, no test set or multi-seed uncertainty; timing is not an inference benchmark.",
                        "Export decoding is not browser or native Matplotlib window validation."]}
    if study:
        report["mlp"] = {"configuration": study["configuration"],
                         "models": {name: {"metrics": result["metrics"],
                                           "initial_parameter_sha256": result["initial_parameter_sha256"],
                                           "checkpoint_epochs": [s["epoch"] for s in result["snapshots"]]}
                                    for name, result in study["models"].items()}}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = renderer.output / f"run_{mode}_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for path in paths:
        print(f"Generated: {path}", flush=True)
    print(f"Dated execution record: {report_path}", flush=True)
    return report


def main(default_mode="quick"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full", "interactive", "3d"), default=default_mode)
    parser.add_argument("--explorer", choices=("activations", "softmax"), default="activations",
                        help="Choose one interactive window; close it to return to the terminal")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=32, help="12-80 frames per GIF; full mode only")
    parser.add_argument("--epochs", type=int, default=400, help="1-2000 matched full-batch Adam updates per model")
    parser.add_argument("--grid", type=int, default=61, help="15-121 points per grid axis")
    args = parser.parse_args()
    try:
        if args.mode == "interactive":
            import matplotlib
            if matplotlib.get_backend().lower() == "agg":
                # Explicitly request the standard local GUI; no browser/server is used.
                matplotlib.use("TkAgg")
            from visual_render import activation_explorer, softmax_explorer
            import matplotlib.pyplot as plt
            view = activation_explorer() if args.explorer == "activations" else softmax_explorer()
            # Keep the slider objects alive while the GUI event loop runs.
            plt.show()
            return view
        return build_lab(args.output, args.mode, args.frames, args.epochs, args.grid)
    except (ImportError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Visual lab error: {error}\n")


if __name__ == "__main__":
    main()
