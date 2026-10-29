"""Generate the Day 39 PCA lab with bounded, independent offline demos."""

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parent
DEMOS = {
    "geometry": ("geometry", ("01_pca_geometry.gif", "01_pca_geometry.png")),
    "3d": ("interactive_3d", ("03_pca_3d_interactive.html",)),
    "explained-variance": ("explained_variance", ("02_explained_variance.gif", "02_explained_variance.png")),
    "reconstruction": ("reconstruction", ("04_pca_reconstruction.gif", "04_pca_reconstruction.png")),
    "scaling": ("scaling", ("05_scaling_comparison.png",)),
    "predictive-information": ("predictive_information", ("06_variance_vs_predictive_information.png",)),
    "outliers": ("outliers", ("07_outlier_sensitivity.gif", "07_outlier_sensitivity.png")),
    "loadings": ("loadings", ("08_component_loadings.png",)),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Generate all eight demos in learning order")
    selection.add_argument("--demo", choices=DEMOS)
    selection.add_argument("--list", action="store_true", help="List demos and output filenames")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--fast", action="store_true", help="Fewer transition frames and lower image DPI")
    parser.add_argument("--force", action="store_true", help="Explicitly regenerate complete existing demos")
    parser.add_argument("--show", action="store_true", help="Also display local Matplotlib figures after export")
    args = parser.parse_args(argv)
    if args.list:
        for name, (_, files) in DEMOS.items():
            print(f"{name:24s} {'; '.join(files)}")
        return 0

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import numpy as np
    import PIL
    import plotly
    import scipy
    import sklearn
    from threadpoolctl import threadpool_limits
    import pca_visualizations as scenes

    scenes.style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "matplotlib": matplotlib.__version__, "plotly": plotly.__version__, "pillow": PIL.__version__}
    selected = list(DEMOS) if args.all else [args.demo]
    for name in selected:
        function, files = DEMOS[name]
        if not args.force and all((args.output_dir / f).is_file() for f in files):
            print(f"Skipped complete demo: {name}; use --force to regenerate", flush=True)
            continue
        print(f"Generating {name}...", flush=True)
        with threadpool_limits(limits=1):
            metrics = getattr(scenes, function)(args.output_dir, fast=args.fast, show=args.show)
        missing = [f for f in files if not (args.output_dir / f).is_file()]
        if missing:
            raise RuntimeError(f"Demo {name} did not generate {missing}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record = {"run_utc": stamp, "demo": name, "fast": args.fast, "versions": versions,
                  "outputs": list(files), "measurements": metrics, "interpretation": "author review required"}
        (args.output_dir / f"run_{name}_{stamp}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Generated {name}: {', '.join(files)}", flush=True)
    entries = []
    for name, (_, files) in DEMOS.items():
        available = [f for f in files if (args.output_dir / f).is_file()]
        if available:
            links = " · ".join(f'<a href="{html.escape(f)}">{html.escape(f)}</a>' for f in available)
            entries.append(f"<li><strong>{html.escape(name)}</strong><br>{links}</li>")
    (args.output_dir / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>PCA visual lab</title>'
        '<style>body{font:18px/1.6 system-ui;max-width:1050px;margin:40px auto;padding:20px;color:#172b45}'
        'li{margin-bottom:22px}a{color:#0072b2}</style><h1>See PCA working</h1>'
        '<p>Open an artifact below. The 3D explorer is independent and works offline. '
        'Read VISUAL_LAB.md for assumptions and author-review status.</p><ul>' + "".join(entries) + '</ul></html>',
        encoding="utf-8")
    print(f"Open {args.output_dir / 'index.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
