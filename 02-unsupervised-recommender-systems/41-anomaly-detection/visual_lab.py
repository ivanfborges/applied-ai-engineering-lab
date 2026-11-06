"""Generate the Day 41 offline anomaly-detection learning lab on Windows or Linux."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parent
SECTION_NAMES = ("robust_statistics", "relationships", "isolation_forest", "one_class_svm",
                 "landscape", "thresholds", "monitoring")
HYPOTHESES = {
    "robust_statistics": "Increasing high-latency contamination moves mean/std more than median/MAD for this sample.",
    "relationships": "A pair within marginal robust bounds can exceed a covariance-distance contour.",
    "isolation_forest": "Actual random partitions give variable paths; averaging corrected paths gives stable-seed isolation scores.",
    "one_class_svm": "Gamma/nu and feature units change fitted RBF support and the anomaly boundary.",
    "landscape": "Different detectors define different native-score geometries over identical features.",
    "thresholds": "Frozen-score ranking remains fixed as policy changes; contamination-only IF variants can have identical AP and different alerts.",
    "monitoring": "A fixed reference flags benign distribution shifts; scripted past-only recalibration changes later alerts.",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--all", action="store_true", help="Generate every section with the full profile")
    modes.add_argument("--quick", action="store_true", help="Smaller data/grid/frame counts; every section unless --section is given")
    parser.add_argument("--section", choices=SECTION_NAMES, help="Generate a single named section")
    parser.add_argument("--output-dir", type=Path, help="Default: topic/outputs/visual_lab/{quick,full}")
    args = parser.parse_args(argv)
    if not (args.all or args.quick or args.section):
        parser.error("select --all, --quick or --section NAME")
    if args.all and args.section:
        parser.error("--all cannot be combined with --section; use --quick --section for a smaller single section")
    try:
        import matplotlib
        import numpy as np
        import PIL
        import plotly
        import scipy
        import sklearn
        import anomaly_visual_core as core
        import anomaly_visuals as scenes
    except ImportError as exc:
        parser.error(f"Missing/incompatible lab dependency ({exc}). From the repository root run: python -m pip install -e \".[dev]\"")
    profile = "quick" if args.quick else "full"
    selected = [args.section] if args.section else list(SECTION_NAMES)
    root = (args.output_dir or ROOT / "outputs" / "visual_lab" / profile).resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = root / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else dict(profile=profile, sections={})
    if catalog["profile"] != profile:
        parser.error("output directory already contains a different profile; select a separate --output-dir")
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in (
        "visual_lab.py", "anomaly_visual_core.py", "anomaly_visuals.py", "example.py", "from_scratch.py"))).hexdigest()
    records = {}
    for name in selected:
        print(f"Generating {name} [{profile}]...", flush=True)
        result = scenes.SECTIONS[name](root, profile)
        for filename in result["files"]:
            if not (root / filename).is_file() or (root / filename).stat().st_size == 0:
                raise RuntimeError(f"section {name} failed to export {filename}")
        records[name] = dict(**result, hypothesis=HYPOTHESES[name],
            interpretation_candidate="Inspect the recorded observations with the corresponding visual question; interpretation requires author review.",
            limitation=scenes.EXPLANATIONS[name][-1])
        catalog["sections"][name] = records[name]
        print("  " + ", ".join(result["files"]), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = dict(executed_at_utc=stamp, profile=profile, configuration=core.PROFILES[profile],
        seed=core.SEED, source_sha256=digest, interpretation_status="author review required",
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                      sklearn=sklearn.__version__, matplotlib=matplotlib.__version__, plotly=plotly.__version__, pillow=PIL.__version__),
        sections=records, files={f: dict(bytes=(root / f).stat().st_size,
            sha256=hashlib.sha256((root / f).read_bytes()).hexdigest()) for r in records.values() for f in r["files"]},
        limitations="Deterministic synthetic teaching scenarios; no real-system benchmark, autonomous incident gating or production validation. Quick/full populations differ.")
    (root / f"run-{stamp}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    catalog_path.write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    scenes.write_index(root, catalog["sections"])
    print(f"Open locally: {root / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
