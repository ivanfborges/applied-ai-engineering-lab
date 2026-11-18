"""Export bounded Day 57 figures, GIFs and standalone offline HTML without the dashboard."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from experiments import train_visuals
from visual_math import batchnorm_trace, stopping_replay
from visual_render import (bn_figure, bn_html_figure, boundary_figure, checkpoint_figure,
                           comparison_figure, dropout_figure, early_figure, lambda_html_figure,
                           run_by_name, save_animation, stopping_html_figure, trajectory_figure,
                           STOPPING_CONTROLS_JS)


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def verify_outputs(paths):
    records = []
    for path in map(Path, paths):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty output: {path}")
        record = {"file": path.name, "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix in (".gif", ".png"):
            with Image.open(path) as picture:
                count = getattr(picture, "n_frames", 1)
                duration = 0
                for index in range(count):
                    picture.seek(index)
                    picture.load()
                    duration += picture.info.get("duration", 0)
                record.update(width=picture.width, height=picture.height, frames=count, duration_ms=duration)
                if path.suffix == ".gif" and (count < 2 or duration == 0 or picture.info.get("loop") != 0):
                    raise ValueError(f"Invalid animation: {path}")
        elif path.suffix == ".html":
            html = path.read_text(encoding="utf-8")
            if "Plotly.newPlot" not in html or re.search(r'<script[^>]+src\s*=', html, re.I):
                raise ValueError(f"Plotly runtime must be embedded: {path}")
            record["offline_runtime"] = True
        records.append(record)
    return records


def export_lab(output, *, epochs=120, n=400, grid=41, frames=24, seeds=(157,), evaluate_test=False):
    if isinstance(frames, bool) or not isinstance(frames, int) or not 8 <= frames <= 40:
        raise ValueError("frames must be an integer in [8, 40].")
    if not seeds or len(seeds) > 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide one to five distinct seeds.")
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError("An execution record exists here. Choose a new output directory to preserve it.")
    studies = []
    for seed in seeds:
        print(f"Training six configurations, seed {seed}…", flush=True)
        studies.append(train_visuals(epochs, n, seed, grid, evaluate_test=evaluate_test))
    study = studies[0]
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    def png(name, figure):
        path = output / name
        figure.savefig(path, dpi=120)
        plt.close(figure)
        paths.append(path)
    print("Exporting bounded static figures and animations…", flush=True)
    baseline = run_by_name(study, "baseline")
    replay = stopping_replay(baseline["history"])
    trace = batchnorm_trace(16, gamma=1.5, beta=.5)
    png("decision_boundary_comparison.png", boundary_figure(study, selected=True))
    png("dropout_activation_distribution.png", dropout_figure())
    png("weight_norm_comparison.png", trajectory_figure())
    png("batchnorm_distributions.png", bn_figure(trace))
    png("checkpoint_comparison.png", checkpoint_figure(study, replay))
    png("integrated_comparison.png", comparison_figure(study))
    paths.append(save_animation(output, "overfitting_evolution.gif", np.unique(np.linspace(0, epochs, frames).astype(int)),
                                lambda epoch: boundary_figure(study, int(epoch))))
    paths.append(save_animation(output, "dropout_neurons.gif", list(range(frames)), lambda index: dropout_figure(pass_index=index)))
    paths.append(save_animation(output, "optimizer_trajectories.gif", np.unique(np.linspace(0, 80, frames).astype(int)),
                                lambda step: trajectory_figure(step=int(step))))
    paths.append(save_animation(output, "batchnorm_transformation.gif", np.unique(np.linspace(0, 39, frames).astype(int)),
                                lambda batch: bn_figure(trace, int(batch))))
    paths.append(save_animation(output, "early_stopping_animation.gif", np.unique(np.linspace(1, replay["stop_epoch"], frames).astype(int)),
                                lambda epoch: early_figure(baseline, replay, int(epoch))))
    for name, figure in (("weight_decay_3d.html", lambda_html_figure()),
                          ("batchnorm_running_statistics.html", bn_html_figure()),
                          ("early_stopping_interactive.html", stopping_html_figure(baseline, replay))):
        path = output / name
        figure.write_html(path, include_plotlyjs=True, full_html=True, auto_play=False,
                          config={"displaylogo": False, "responsive": True},
                          post_script=STOPPING_CONTROLS_JS if name == "early_stopping_interactive.html" else None)
        paths.append(path)
    summary = []
    for item in studies:
        for run in item["runs"]:
            summary.append({"seed": item["configuration"]["seed"], "configuration": run["name"],
                            "validation_loss": run["selected"]["validation"]["loss"],
                            "validation_accuracy": run["selected"]["validation"]["accuracy"],
                            "selected_epoch": run["selected"]["epoch"], "epochs_run": run["epochs_run"],
                            "steps": run["steps"], "linear_norm": run["selected"]["weight_norm"],
                            "duration_seconds": run["duration_seconds"]})
    import pandas as pd
    table = pd.DataFrame(summary)
    table.to_csv(output / "configuration_metrics.csv", index=False)
    aggregation = table.groupby("configuration")["validation_loss"].agg(["mean", "std", "count"])
    aggregation.to_csv(output / "seed_variability.csv")
    for item in studies:
        filename = output / f"training_seed_{item['configuration']['seed']}.json"
        filename.write_text(json.dumps(item, default=serializable, allow_nan=False), encoding="utf-8")
    manifest = {"executed_at_utc": datetime.now(timezone.utc).isoformat(),
                "configuration": {"frames_max": frames, "epochs_max": epochs, "n": n, "grid": grid,
                                  "seeds": list(seeds), "test_evaluation_per_seed": int(evaluate_test)},
                "render_versions": {name: importlib.metadata.version(name) for name in ("matplotlib", "plotly", "Pillow", "streamlit")},
                "artifacts": verify_outputs(paths), "results": summary,
                "selected_configurations": {str(item["configuration"]["seed"]): item["selected_configuration"] for item in studies},
                "final_tests": {str(item["configuration"]["seed"]): item["final_test"] for item in studies},
                "hypothesis": study["hypothesis"], "interpretation_candidate": study["interpretation_candidate"],
                "interpretation_status": "author review required", "limitation": study["limitation"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / datetime.now(timezone.utc).strftime("visual-%Y%m%dT%H%M%SZ"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--grid", type=int, default=41)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[157])
    parser.add_argument("--evaluate-test", action="store_true", help="One frozen-selection test evaluation per seed; never use it for tuning.")
    args = vars(parser.parse_args())
    args.pop("export")
    try:
        report = export_lab(**args)
    except ValueError as error:
        parser.error(str(error))
    print(f"Generated and decoded {len(report['artifacts'])} visual artifacts. Interpretation review required.")


if __name__ == "__main__":
    main()
