"""Generate the Day 56 optimizer laboratory locally; no PyTorch or network required."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import webbrowser

import matplotlib
matplotlib.use("Agg")
import numpy as np
import PIL
import plotly

from from_scratch import METHODS
from visual_math import (LABELS, Objective, compare, constant_gradient, learning_rates,
                         reference_rates, simulate)
from visual_render import (QUALITY, bias_figure, decay_figure, diagnostic_figure, export,
                           export_surface, momentum_figure, noise_figure, rate_figure,
                           surface_figure, trajectory_figure)


SCENES = ("trajectories", "surface3d", "momentum", "rmsprop", "adam", "adamw", "learning-rate", "noise")
QUESTIONS = {
    "trajectories": "How do five rules move from the same start on the same objective?",
    "surface3d": "Do the recorded paths remain on the exact objective surface at each inspected step?",
    "momentum": "How does the previous direction contribute to the next update?",
    "rmsprop": "How do squared-gradient histories rescale coordinate updates?",
    "adam": "How do first/second moments and zero-initialization correction change an update?",
    "adamw": "Where does decay enter the update, and what happens when decay is zero?",
    "learning-rate": "Which behavior actually occurs at three objective-specific learning rates?",
    "noise": "How does controlled synthetic gradient noise interact with retained history?",
}


def trace_record(trace):
    return {"optimizer": trace.method, "learning_rate": trace.lr, "weight_decay": trace.weight_decay,
            "noise_std": trace.noise, "recorded_steps": trace.steps, "status": trace.status,
            "initial_loss": float(trace.losses[0]), "final_loss": float(trace.losses[-1]),
            "final_parameter_norm": float(np.linalg.norm(trace.points[-1])),
            "points": trace.points.tolist(), "losses": trace.losses.tolist()}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    selection = result.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Generate every scene (also the default).")
    selection.add_argument("--scene", choices=SCENES, help="Generate one named scene.")
    result.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"outputs")
    result.add_argument("--quality", choices=QUALITY, default="standard")
    result.add_argument("--steps", type=int, default=80, help="Actual updates, 1..2000; animation frames are sampled.")
    result.add_argument("--seed", type=int, default=56)
    result.add_argument("--objective", choices=("quadratic", "rosenbrock"), default="quadratic")
    result.add_argument("--a", type=float, default=40.)
    result.add_argument("--b", type=float, default=1.)
    result.add_argument("--angle", type=float, default=30., help="Quadratic rotation in degrees.")
    result.add_argument("--start", type=float, nargs=2, default=[-2., 1.5], metavar=("X", "Y"))
    result.add_argument("--optimizer", choices=("all",)+METHODS, default="all",
                        help="Filters trajectory/3D scenes; chooses rate-sweep method (all uses SGD).")
    result.add_argument("--learning-rate", type=float, help="Override rates except the independent three-rate sweep.")
    result.add_argument("--momentum", type=float, default=.9)
    result.add_argument("--rho", type=float, default=.99)
    result.add_argument("--beta1", type=float, default=.9)
    result.add_argument("--beta2", type=float, default=.999)
    result.add_argument("--epsilon", type=float, default=1e-8)
    result.add_argument("--weight-decay", type=float, default=.1, help="Targeted Adam/AdamW pair only; other scenes use zero.")
    result.add_argument("--noise-std", type=float, default=0., help="Synthetic Gaussian noise; rate sweep stays exact.")
    result.add_argument("--noise-low", type=float, default=.5, help="Low-noise level in the dedicated noise scene.")
    result.add_argument("--noise-high", type=float, default=3., help="High-noise level in the dedicated noise scene.")
    result.add_argument("--log-surface", action="store_true", help="Display log(1+loss), labeled explicitly in the 3D scene.")
    result.add_argument("--static-only", action="store_true", help="Export PNG/HTML, omitting GIF encoding.")
    result.add_argument("--show", action="store_true", help="Open the generated HTML, or first PNG, in the default browser.")
    return result


def generate(args):
    started = perf_counter()
    objective = Objective(args.objective, args.a, args.b, args.angle)
    common = dict(start=args.start, steps=args.steps, seed=args.seed, noise=args.noise_std,
                  momentum=args.momentum, rho=args.rho, beta1=args.beta1,
                  beta2=args.beta2, epsilon=args.epsilon)
    methods = METHODS if args.optimizer == "all" else (args.optimizer,)
    selected = SCENES if args.scene is None else (args.scene,)
    args.output.mkdir(parents=True, exist_ok=True)
    frames, _, _, resolution = QUALITY[args.quality]
    rates = reference_rates(objective)
    records, artifacts = {}, []

    def run(method, obj=objective, **changes):
        return simulate(obj, method, changes.pop("lr", rates[method] if args.learning_rate is None else args.learning_rate), **(common | changes))

    def save(fig_update, *, png, gif=None, steps=args.steps):
        fig, callback = fig_update
        return export(fig, callback, steps, args.output, png=png,
                      gif=None if args.static_only else gif, quality=args.quality)

    shared = None
    if "trajectories" in selected or "surface3d" in selected:
        shared = compare(objective, methods=methods, lr=args.learning_rate, **common)
    for scene in selected:
        print(f"Generating {scene}...", flush=True)
        traces = []
        extra = {}
        if scene == "trajectories":
            traces = shared
            files = save(trajectory_figure(objective, traces, resolution),
                         png="01_optimizer_trajectories.png", gif="01_optimizer_trajectories.gif")
        elif scene == "surface3d":
            traces = shared
            figure = surface_figure(objective, traces, resolution=resolution, frames=frames,
                                     log_surface=args.log_surface)
            files = [export_surface(figure, args.output)]
        elif scene == "momentum":
            traces = [run(method) for method in ("sgd", "momentum")]
            files = save(momentum_figure(objective, traces, args.momentum, resolution),
                         png="03_momentum_internal_state.png", gif="03_sgd_vs_momentum.gif")
        elif scene == "rmsprop":
            traces = [run("rmsprop")]
            files = save(diagnostic_figure(traces[0], "rmsprop", args.epsilon, rho=args.rho),
                         png="04_rmsprop_internal_state.png", gif="04_rmsprop_adaptive_scaling.gif")
        elif scene == "adam":
            traces = [run("adam")]
            files = save(diagnostic_figure(traces[0], "adam", args.epsilon, beta1=args.beta1, beta2=args.beta2),
                         png="05_adam_moments.png", gif="05_adam_internal_mechanics.gif")
            constant = constant_gradient(beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
            fig = bias_figure(constant, args.beta1, args.beta2)
            files += export(fig, lambda step: None, 20, args.output, png="05_adam_bias_correction.png", quality=args.quality)
            extra = {"constant_gradient": constant["gradient"].tolist(), "constant_steps": 20,
                     "first_hat_max_error": float(np.max(np.abs(constant["first_hat"]-constant["gradient"]))),
                     "second_hat_max_error": float(np.max(np.abs(constant["second_hat"]-constant["gradient"]**2)))}
        elif scene == "adamw":
            decay_objective = (Objective(args.objective, args.a, args.b, args.angle, center=(1., -.7))
                               if args.objective == "quadratic" else objective)
            pair = [run(method, decay_objective, weight_decay=args.weight_decay,
                        lr=rates["adam"] if args.learning_rate is None else args.learning_rate) for method in ("adam", "adamw")]
            control = [run(method, decay_objective, weight_decay=0.,
                           lr=rates["adam"] if args.learning_rate is None else args.learning_rate) for method in ("adam", "adamw")]
            traces = pair+control
            extra = {"objective_center": list(decay_objective.center),
                     "zero_decay_max_path_difference": float(np.max(np.abs(control[0].points-control[1].points)))}
            files = save(decay_figure(decay_objective, pair, control, resolution),
                         png="06_weight_decay_comparison.png", gif="06_adam_vs_adamw.gif")
        elif scene == "learning-rate":
            method = "sgd" if args.optimizer == "all" else args.optimizer
            presets = learning_rates(objective, method)
            traces = [run(method, lr=rate, noise=0.) for rate in presets.values()]
            labels = [f"{label}: lr={rate:g}" for label, rate in presets.items()]
            files = save(rate_figure(objective, traces, labels, resolution),
                         png="07_learning_rate_curves.png", gif="07_learning_rate_sensitivity.gif")
        else:
            if not 0 <= args.noise_low < args.noise_high:
                raise ValueError("noise scene requires 0 <= noise-low < noise-high.")
            groups = [[run(method, noise=noise) for noise in (0., args.noise_low, args.noise_high)]
                      for method in ("sgd", "momentum", "adam")]
            traces = [trace for group in groups for trace in group]
            files = save(noise_figure(objective, groups, resolution),
                         png="08_gradient_noise_curves.png", gif="08_gradient_noise.gif")
        artifacts.extend(files)
        records[scene] = {"question": QUESTIONS[scene],
                          "hypothesis": "The stated gradient histories and update rules expose this mechanism under the recorded configuration.",
                          "runs": [trace_record(t) for t in traces], "checks": extra,
                          "interpretation_candidate": "Inspect the measured paths and states for this configuration; no optimizer superiority follows.",
                          "interpretation_status": "author review required",
                          "limitation": "Toy objective; illustrative rates; Gaussian perturbation is not a literal mini-batch gradient; no held-out metric."}
    now = datetime.now(timezone.utc)
    manifest_path = args.output/f"visual-run-{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    source_names = ("from_scratch.py", "visual_math.py", "visual_render.py", "visualizations.py")
    manifest = {"executed_at_utc": now.isoformat(), "configuration": settings,
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                                "Pillow": PIL.__version__, "plotly": plotly.__version__},
                "source_sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in source_names},
                "scenes": records, "render_elapsed_seconds": perf_counter()-started,
                "artifacts": [{"file": p.name, "bytes": p.stat().st_size,
                               "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts],
                "interpretation_status": "author review required",
                "render_limitations": "Discrete sampled frames, no interpolated optimizer states; log-loss display floor 1e-30 only; no benchmark."}
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if args.show and artifacts:
        target = next((p for p in artifacts if p.suffix == ".html"), artifacts[0])
        webbrowser.open(target.resolve().as_uri())
    print(json.dumps({"manifest": str(manifest_path), "artifacts": len(artifacts),
                      "seconds": manifest["render_elapsed_seconds"], "interpretation_status": "author review required"}, indent=2))
    return manifest


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        generate(args)
    except (ValueError, FloatingPointError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
