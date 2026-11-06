"""Generate the ten-scene, offline Day 42 recommendation learning lab."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parent


def json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--all", action="store_true", help="Generate all ten scenes")
    modes.add_argument("--scene", type=int, choices=range(1, 11), help="Generate one scene independently")
    modes.add_argument("--list", action="store_true", help="List scene titles without generating files")
    parser.add_argument("--fast", action="store_true", help="Fewer saved animation states and thinning repeats; same model fit")
    parser.add_argument("--animate-matrix", action="store_true", help="Also animate progressive observations in scene 1")
    parser.add_argument("--output-dir", type=Path, help="Default: topic/outputs/day42 (fast uses a separate fast/ subfolder)")
    args = parser.parse_args(argv)
    try:
        import matplotlib
        import numpy as np
        import pandas as pd
        import PIL
        import plotly
        import sklearn
        import recommender_visual_core as core
        import recommender_visuals as scenes
    except ImportError as exc:
        parser.error(f'Missing/incompatible dependency ({exc}). From the repo root: python -m pip install -e ".[dev]"')
    if args.list:
        for number, (name, question, _, _) in scenes.SCENES.items():
            print(f"{number:2d}. {name}: {question}")
        return 0
    selected = [args.scene] if args.scene else list(scenes.SCENES)
    profile = "fast" if args.fast else "full"
    default = ROOT / "outputs" / "day42"
    if args.fast:
        default /= "fast"
    out = (args.output_dir or default).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_files = [ROOT / name for name in ("visualize_day42.py", "recommender_visual_core.py",
                                             "recommender_visuals.py", "factorization.py", "from_scratch.py")]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in source_files)).hexdigest()
    catalog_path = out / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
    if catalog and catalog.get("profile") != profile:
        parser.error("output directory contains another profile; choose a different --output-dir")
    # After source changes, rebuild gallery membership only from this execution.
    if catalog.get("source_sha256") != digest:
        catalog = dict(profile=profile, source_sha256=digest, scenes={})
    ctx = core.build_context(args.fast)
    ctx["animate_matrix"] = args.animate_matrix
    records = {}
    for number in selected:
        print(f"Generating scene {number}: {scenes.SCENES[number][0]} [{profile}]...", flush=True)
        record = scenes.RENDERERS[number](ctx, out)
        for filename in record["files"]:
            path = out / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"missing generated asset: {filename}")
        name, question, hypothesis, limitation = scenes.SCENES[number]
        record.update(name=name, question=question, hypothesis=hypothesis,
                      interpretation_candidate="Inspect the measured observations; author review required.", limitation=limitation)
        records[str(number)] = record
        catalog["scenes"][str(number)] = record
        print("  " + ", ".join(record["files"]), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = dict(executed_at_utc=stamp, profile=profile, seed=core.SEED, source_sha256=digest,
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                sklearn=sklearn.__version__, matplotlib=matplotlib.__version__,
                                plotly=plotly.__version__, pillow=PIL.__version__),
                  configuration=dict(shape=[12, 10], factorization=ctx["factor"]["configuration"],
                                     train_pairs=int(ctx["data"]["train_mask"].sum()),
                                     heldout_pairs=int(ctx["data"]["test_mask"].sum()),
                                     animation_states=ctx["frames"], animate_matrix=args.animate_matrix),
                  interpretation_status="author review required", scenes=records,
                  assets={filename: dict(bytes=(out / filename).stat().st_size,
                          sha256=hashlib.sha256((out / filename).read_bytes()).hexdigest())
                          for record in records.values() for filename in record["files"]})
    (out / f"run-{stamp}.json").write_text(json.dumps(report, indent=2, default=json_value, allow_nan=False) + "\n", encoding="utf-8")
    catalog_path.write_text(json.dumps(catalog, indent=2, default=json_value, allow_nan=False) + "\n", encoding="utf-8")
    scenes.write_gallery(out, catalog["scenes"])
    print(f"Open locally: {out / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
