"""Day 61 CPU learning lab: ten measured Matplotlib demonstrations, Pillow exports and GUI controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Run/export all ten demonstrations")
    selection.add_argument("--demo", choices=["network", "latent", "reconstruction", "training", "threshold",
                                           "comparison", "bottleneck", "failures", "denoising", "scaling"])
    selection.add_argument("--list", action="store_true", help="List demonstrations and expected filenames without training")
    parser.add_argument("--quick", action="store_true", help="45 epochs, 12 frames, 2 bottleneck seeds; alone selects all demos")
    parser.add_argument("--interactive", action="store_true", help="Open actual Matplotlib widgets; only --demo latent/threshold")
    parser.add_argument("--backend", default="TkAgg", help="GUI backend for --interactive (e.g. TkAgg or QtAgg); exports always use Agg")
    parser.add_argument("--output-dir", type=Path, help="New or empty directory; relative paths are relative to the current working directory")
    parser.add_argument("--seed", type=int, default=42, help="Dataset seed; bottleneck initialization seeds start here")
    args = parser.parse_args(argv)
    if not (args.all or args.demo or args.quick or args.list):
        parser.error("Choose --all, --demo NAME, --quick or --list")
    if args.interactive and args.demo not in ("latent", "threshold"):
        parser.error("--interactive requires --demo latent or --demo threshold")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.interactive and args.output_dir is not None:
        parser.error("Interactive mode opens a GUI; use a separate export command for --output-dir")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Apply resource settings before numerical imports. This is not a machine-wide CPU cap.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        import matplotlib
        matplotlib.use(args.backend if args.interactive else "Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
        import PIL
        import sklearn
        import torch
        from visual_core import LabExperiments, save_checkpoints, serializable
        from visual_figures import DEMOS, latent_view, style, threshold_view
    except ImportError as error:
        raise SystemExit(f'Missing dependency: {error}. From the repository root run: python -m pip install -e ".[deep-learning]"') from error
    if args.list:
        for name, (_, outputs) in DEMOS.items():
            print(f"{name:15s} {'  '.join(outputs)}" + (" [GUI controls available]" if name in ("latent", "threshold") else ""))
        return 0
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    style()
    if args.interactive:
        # Probe the requested backend before training, so unsupported GUIs fail early.
        try:
            probe = plt.figure()
            plt.close(probe)
        except (ImportError, RuntimeError) as error:
            raise SystemExit(f"GUI backend {args.backend} could not open a figure: {error}. Run export mode, or use a local Python with Tk/Qt support.") from error
        if str(matplotlib.get_backend()).lower() in ("agg", "pdf", "svg", "ps", "cairo", "template"):
            raise SystemExit("Interactive mode needs a GUI backend such as TkAgg or QtAgg")
        lab = LabExperiments(args.quick, args.seed)
        fig, api = latent_view(lab, True) if args.demo == "latent" else threshold_view(lab, True)
        # Retain widget references for the entire GUI lifetime.
        fig._day61_controls = api
        print(f"Interactive {args.demo} ready on {matplotlib.get_backend()}; close its window to exit.", flush=True)
        plt.show()
        return 0
    output = args.output_dir or Path(__file__).resolve().parent / "outputs" / "day61" / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output must be a new or empty directory; existing files are preserved: {output}")
    output.mkdir(parents=True, exist_ok=True)
    names = [args.demo] if args.demo else list(DEMOS)
    lab = LabExperiments(args.quick, args.seed)
    print(f"{'Quick' if args.quick else 'Full'} CPU protocol | output: {output.resolve()}", flush=True)
    for index, name in enumerate(names, 1):
        print(f"[{index}/{len(names)}] {name}", flush=True)
        DEMOS[name][0](lab, output)
        for filename in DEMOS[name][1]:
            path = output / filename
            with PIL.Image.open(path) as asset:
                if path.suffix == ".gif" and asset.n_frames < 2:
                    raise RuntimeError(f"Animation has fewer than two frames: {path}")
                print(f"  Saved {filename}: {asset.size}, {getattr(asset, 'n_frames', 1)} frame(s)", flush=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "phase": "visual lab, distinct from analytical example.py",
        "hypotheses": ["Reconstruction is not a general anomaly probability", "Bottleneck/capacity alter reconstruction and detection differently",
                       "Curved structure can favor nonlinear reconstruction; advantage must be measured",
                       "Corrupted-input/clean-target training may improve denoising under matching noise"],
        "config": {"quick": args.quick, "dataset_seed": args.seed, "epochs": lab.epochs,
                   "geometry_epochs": lab.epochs * 2, "requested_frames": lab.frames, "demos": names,
                   "device": "cpu", "torch_threads": 1, "deterministic_algorithms": True,
                   "threshold_quantiles": [.95, .975, .99], "quantile_method": "higher", "alert_rule": "score > threshold",
                   "prevalence": float(lab.data.labels.mean()),
                   "split_sizes": {k: len(getattr(lab.data, k)) for k in ("train", "validation", "calibration", "test")}},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                     "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__, "pillow": PIL.__version__},
        "data": serializable(lab.data), "experiments": serializable(lab.cache),
        "interpretation_candidate": "Inspect subgroup failures and measured comparisons; a proposed effect may be absent.",
        "interpretation_status": "author review required",
        "limitations": "Synthetic data, short fixed budgets, initialization seeds only, no real-world/production/cross-platform claim. GUI interactions with test scores are exploratory, not model selection.",
    }
    with (output / "experiment_record.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    # Save the actual checkpoints used for the base animation, separate from predictions in JSON.
    if "base" in lab.cache:
        save_checkpoints(lab.cache["base"]["checkpoints"], output / "training_checkpoints.pt")
    print("Saved experiment_record.json with actual predictions/configuration. Interpretations require author review.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
