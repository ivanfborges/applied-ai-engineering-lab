"""Generate the Day 40 offline manifold-learning laboratory on CPU."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
QUICK = dict(digits_n=180, swiss_n=220, iterations=500, epochs=150, toy_steps=180)
STANDARD = dict(digits_n=400, swiss_n=500, iterations=1000, epochs=400, toy_steps=400)
GROUPS = {
    "quick": ["swiss", "digits-3d", "optimization", "parameters"],
    "interactive": ["swiss", "digits-3d", "graph", "parameters", "neighbors"],
    "animations": ["probabilities", "optimization", "parameters"],
    "experiments": ["parameters", "geometry-traps", "stability", "neighbors"],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("quick", "interactive", "animations", "experiments", "all", "list"):
        mode.add_argument(f"--{name}", action="store_true")
    mode.add_argument("--demo", choices=["swiss", "digits-3d", "probabilities", "optimization", "kernels",
                                        "graph", "parameters", "geometry-traps", "stability", "neighbors"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--fast", action="store_true", help="Use the small quick profile for any selected group")
    parser.add_argument("--force", action="store_true", help="Explicitly replace existing visual assets; fit cache remains reusable")
    args = parser.parse_args(argv)
    if args.list:
        for name, group in GROUPS.items():
            print(f"--{name:14s} {', '.join(group)}")
        print("--all             all ten modules in learning order")
        print("--demo NAME       one independent module")
        return 0

    import matplotlib
    matplotlib.use("Agg")
    import manifold_visual_core as core
    import manifold_visualizations as scenes

    settings = QUICK.copy() if args.quick or args.fast else STANDARD.copy()
    selected = list(scenes.MODULES) if args.all else ([args.demo] if args.demo else
               next(group for name, group in GROUPS.items() if getattr(args, name)))
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = root / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
    digest = hashlib.sha256(b"".join((ROOT / f).read_bytes() for f in (
        "visualize.py", "manifold_visual_core.py", "manifold_visualizations.py", "from_scratch.py", "example.py"))).hexdigest()
    scenes.prepare(root)
    for name in selected:
        previous = catalog.get(name)
        if previous and all((root / f).exists() for f in previous["files"]) and not args.force:
            if previous["settings"] != settings or previous["generator_sha256"] != digest:
                raise ValueError(f"{name} has a different profile/source; use a separate --output-dir or explicit --force")
            print(f"Skipped complete module: {name}; --force explicitly regenerates assets", flush=True)
            continue
        print(f"Generating {name}...", flush=True)
        record = scenes.MODULES[name][0](root, settings)
        if any(not (root / f).is_file() for f in record["files"]):
            raise RuntimeError(f"{name} did not produce its declared files")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record.update(module=name, settings=settings, generator_sha256=digest,
                      completed_at_utc=stamp, versions=core.versions(),
                      interpretation="author review required; exploratory evidence, not task validation")
        (root / f"run_{name}_{stamp}.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        catalog[name] = record
        catalog_path.write_text(json.dumps(catalog, indent=2, allow_nan=False), encoding="utf-8")
        scenes.write_index(root, catalog)
        print(f"Generated {name}: {', '.join(record['files'])}", flush=True)
    scenes.write_index(root, catalog)
    print(f"Open locally: {root / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
