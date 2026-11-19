"""Day 54 visual learning lab. Exports locally; --list needs no dependencies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
from uuid import uuid4


MODES = {
    "computational_graph": "Forward values, backward derivatives and a separate optimizer step",
    "chain_rule": "Local derivative products and local versus finite loss changes",
    "loss_landscape": "Actual 3D neural loss, contours and evaluated gradient descent",
    "gradient_flow": "Exactly controlled contracting, preserved and amplifying scalar chains",
    "activation_gradients": "Sigmoid, tanh, ReLU and their local derivatives",
    "learning_rates": "Fixed-data and fixed-initialization update-size comparison",
    "gradient_check": "Analytical versus central finite-difference gradients across epsilon",
    "training_engineering": "Global norm clipping and weighted unequal-microbatch accumulation",
}


def verify_outputs(paths):
    from PIL import Image
    records = []
    for candidate in paths:
        path = Path(candidate)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty output: {path}")
        record = {"file": path.name, "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix in (".png", ".gif"):
            with Image.open(path) as picture:
                count = getattr(picture, "n_frames", 1)
                record.update(width=picture.width, height=picture.height, decoded_frames=count)
                durations = []
                for k in range(count):
                    picture.seek(k); picture.load()
                    durations.append(picture.info.get("duration", 0))
                if path.suffix == ".gif":
                    if count < 2:
                        raise ValueError(f"animation has fewer than two decoded frames: {path}")
                    record["duration_ms"] = sum(durations)
        elif path.suffix == ".html":
            html = path.read_text(encoding="utf-8")
            if "Plotly.newPlot" not in html or re.search(r'<script[^>]+src\s*=', html, re.I):
                raise ValueError(f"HTML must embed its runtime: {path}")
            record["embedded_plotly"] = True
        records.append(record)
    return records


def build_lab(output, *, mode="all", quick=False, frames=None, grid=None, steps=180):
    if mode not in (*MODES, "all"):
        raise ValueError(f"unknown mode: {mode}")
    frames = frames if frames is not None else (24 if quick else 60)
    grid = grid if grid is not None else (41 if quick else 91)
    for name, value, low, high in (("frames", frames, 18, 120), ("grid", grid, 21, 151), ("steps", steps, 1, 2000)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low},{high}]")
    try:
        import numpy as np
        import matplotlib
        import plotly
        import PIL
        from visual_math import (
            accumulation, central_difference, clip_global_norm, descent,
            loss_gradient, optional_autograd_check, scalar_graph, synthetic_data,
        )
        from visual_render import Renderer
    except ModuleNotFoundError as error:
        raise ValueError(
            f"Missing dependency {error.name!r}. From the repository root run "
            'python -m pip install -e ".[dev]"; for a standalone topic folder use '
            'python -m pip install -r requirements.txt. PyTorch is optional.'
        ) from error
    x, y = synthetic_data()
    theta = np.array([.7, 1.2])
    analytical = loss_gradient(theta, x, y)[1]
    numerical = central_difference(theta, x, y)
    if not np.allclose(analytical, numerical, atol=1e-9, rtol=1e-7):
        raise ValueError("analytical gradients failed the pre-render central-difference check")
    traces = [descent(x, y, learning_rate=rate, steps=steps) for rate in (.001, .01, .1, 1.)]
    renderer = Renderer(output, frames=frames, grid=grid)
    paths = []
    selected = list(MODES) if mode == "all" else [mode]
    for name in selected:
        print(f"Generating {name} ...", flush=True)
        render = getattr(renderer, name)
        result = render(x, y, traces) if name in ("loss_landscape", "learning_rates") else (
            render(x, y) if name in ("gradient_check", "training_engineering") else render())
        paths.extend(result)
    records = verify_outputs(paths)
    batch = accumulation(x, y)
    gradient = loss_gradient((.5, .8), x, y)[1]
    clipping_threshold = .6 * np.linalg.norm(gradient)
    scalar = scalar_graph()
    scalar_summary = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in scalar.items()}
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__, "plotly": plotly.__version__, "Pillow": PIL.__version__},
        "configuration": {"modes": selected, "frames_requested": frames, "grid": grid,
                          "observations": 41, "initial_weights": [.5, .8], "steps_per_rate": steps,
                          "rates": [.001, .01, .1, 1.], "dtype": "float64", "randomness": "none"},
        "hypotheses": [
            "The biased scalar graph's chain-rule product equals its w1 derivative.",
            "Recorded gradient descent losses equal the actual objective at each recorded parameter position.",
            "Changing only learning rate changes finite parameter steps and observed loss trajectories.",
            "A controlled scalar chain propagates an upstream derivative by an exact product of local factors.",
            "Central differences agree with manual gradients at an intermediate epsilon; rounding can worsen tiny-epsilon error.",
            "Weighted microbatch means reproduce the frozen-parameter full-batch gradient; clipping bounds its norm.",
        ],
        "scalar_graph": scalar_summary,
        "gradient_check": {"at": theta.tolist(), "epsilon": 1e-5, "analytical": analytical.tolist(),
                           "numerical": numerical.tolist(), "max_absolute_error": float(abs(analytical-numerical).max())},
        "trajectories": [{"rate": t["learning_rate"], "status": t["status"],
                          "initial_loss": float(t["loss"][0]), "final_loss": float(t["loss"][-1]),
                          "final_weights": t["theta"][-1].tolist(), "completed_updates": len(t["loss"])-1,
                          "loss_increases": int(np.count_nonzero(np.diff(t["loss"]) > 1e-14))}
                         for t in traces],
        "clipping": {"gradient": gradient.tolist(), "threshold": float(clipping_threshold),
                     "before_norm": float(np.linalg.norm(gradient)),
                     "after_norm": float(np.linalg.norm(clip_global_norm(gradient, clipping_threshold)))},
        "accumulation": {"sizes": batch["sizes"], "full": batch["full"].tolist(),
                         "weighted": batch["weighted"].tolist(),
                         "max_absolute_error": float(abs(batch["full"]-batch["weighted"]).max())},
        "autograd": optional_autograd_check(x, y),
        "interpretation_candidate": "Inspect actual trajectories and numerical identities; author review required.",
        "limitations": "Synthetic shallow models and controlled scalar chains; no holdout, trained deep MLP, GPU, throughput or production claims.",
        "outputs": records,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    record_path = Path(output) / f"run-{stamp}.json"
    with record_path.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(paths)} outputs. Dated record: {record_path}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--all", action="store_true", help="generate all eight modules (also the default)")
    choice.add_argument("--only", choices=list(MODES), help="generate a single module")
    choice.add_argument("--list", action="store_true", help="list modules without loading plotting dependencies")
    parser.add_argument("--output", type=Path,
                        help="New or empty directory; default: a fresh ignored outputs/visual-lab/run-* directory")
    parser.add_argument("--quick", action="store_true", help="24 frames and a 41x41 surface grid; same numerical experiment")
    parser.add_argument("--frames", type=int, help="18..120 frames; default 60")
    parser.add_argument("--grid", type=int, help="21..151 points per grid axis; default 91")
    parser.add_argument("--steps", type=int, default=180, help="1..2000 optimization updates per rate; default 180")
    args = parser.parse_args(argv)
    if args.list:
        for name, question in MODES.items():
            print(f"{name:24} {question}")
        return
    try:
        output = args.output
        if output is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = (Path(__file__).resolve().parent / "outputs" / "visual-lab"
                      / f"run-{stamp}-{uuid4().hex[:8]}")
        build_lab(output, mode=args.only or "all", quick=args.quick,
                  frames=args.frames, grid=args.grid, steps=args.steps)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Visual lab error: {error}\n")


if __name__ == "__main__":
    main()
