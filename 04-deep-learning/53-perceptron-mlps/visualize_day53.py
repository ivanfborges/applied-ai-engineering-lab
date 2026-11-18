"""Reproducible offline Day 53 lab. Run --help for bounded export modes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys

import numpy as np

from from_scratch import constructive_xor, initialize_parameters
from visual_math import XOR_X, XOR_Y, check_gradients, training_trace

MODES = ("neuron", "perceptron", "xor", "activations", "features", "forward", "training", "landscape", "approximation")


def verify_outputs(paths):
    """Decode every raster frame and reject HTML with external script sources."""
    from PIL import Image
    records = []
    for path in paths:
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty output: {path}")
        record = {"file": path.name, "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix in (".png", ".gif"):
            with Image.open(path) as image:
                record.update({"width": image.width, "height": image.height, "frames": image.n_frames if path.suffix == ".gif" else 1})
                for frame in range(record["frames"]):
                    image.seek(frame)
                    image.load()
        elif path.suffix == ".html":
            html = path.read_text(encoding="utf-8")
            if "Plotly.newPlot" not in html or re.search(r'<script[^>]+src\s*=', html, re.IGNORECASE):
                raise ValueError(f"HTML must embed Plotly rather than load external scripts: {path}")
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if not all(np.isfinite(data[key]).all() for key in data.files):
                    raise ValueError(f"checkpoint arrays must be finite: {path}")
        records.append(record)
    return records


def build_lab(output, *, mode="all", quick=False, seed=42):
    if mode not in (*MODES, "all"):
        raise ValueError(f"unknown mode: {mode}")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    from visual_render import Renderer
    import matplotlib
    import PIL
    import scipy
    import sklearn
    renderer = Renderer(output, quick=quick, seed=seed)
    if not np.array_equal(constructive_xor(XOR_X), XOR_Y):
        raise ValueError("constructive XOR verification failed")
    gradient_error = check_gradients(initialize_parameters(2, 8, seed=seed))
    if gradient_error > 1e-7:
        raise ValueError(f"gradient check failed: max absolute error {gradient_error}")
    modes = MODES if mode == "all" else (mode,)
    trace = None
    if any(item in modes for item in ("training", "landscape")):
        epochs = 2000 if quick else 10000
        print(f"Computing {epochs:,} full-batch XOR updates; retaining genuine checkpoints...", flush=True)
        trace = training_trace(seed, epochs=epochs, frames=renderer.frames)
        checkpoint = renderer.path("07_training_checkpoints.npz")
        np.savez_compressed(checkpoint, epochs=[s["epoch"] for s in trace["states"]], losses=trace["losses"],
                            **{key: np.stack([s["parameters"][key] for s in trace["states"]]) for key in ("W1", "b1", "W2", "b2")})
        final = trace["states"][-1]
        renderer.results["training"] = {"epochs": epochs, "learning_rate": trace["learning_rate"],
                                        "architecture": [2, 8, 1], "seed": seed, "training_points": 4,
                                        "initial_loss": float(trace["losses"][0]), "final_loss": final["loss"],
                                        "final_training_accuracy": float(np.mean((final["probabilities"] >= 0.5) == XOR_Y)),
                                        "checkpoints": len(trace["states"]), "probabilities": final["probabilities"].tolist()}
    for item in modes:
        print(f"Rendering {item}...", flush=True)
        method = getattr(renderer, item)
        method(trace) if item in ("training", "landscape") else method()
    print("Decoding exported images/GIF frames and checking embedded HTML...", flush=True)
    files = verify_outputs(renderer.files)
    try:
        import plotly
        plotly_version = plotly.__version__
    except ImportError:
        plotly_version = "not installed; no HTML mode requested"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "mode": mode, "quick": quick, "seed": seed,
        "data": "synthetic; truth tables and noisy latent XOR corners",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                        "plotly": plotly_version, "pillow": PIL.__version__},
        "render_config": {"grid_size": renderer.size, "dpi": renderer.dpi, "requested_frames": renderer.frames},
        "checks": {"constructive_xor_correct": True, "gradient_max_abs_error": gradient_error,
                   "files_decoded": len(files), "browser_interaction": "requires separate validation"},
        "results": renderer.results, "files": files, "interpretation_status": "author review required",
    }
    name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ.json")
    (renderer.output / name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified {len(files)} artifacts. Run record: {renderer.output / name}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--quick", action="store_true", help="41-point grids, 20 requested frames, 2,000 training updates")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "visual_lab")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        build_lab(args.output_dir, mode=args.mode, quick=args.quick, seed=args.seed)
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        print(f"Day 53 lab error: {exc}\nRun from the repository root after installing: python -m pip install -e \".[dev]\"", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
