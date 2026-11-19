"""Generate the offline Day 58 laboratory: quick, full or interactive-only."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import webbrowser

from experiments import Config, cached_bundle, torch
from visualizations import export_all


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full", "interactive"), default="quick")
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--depth", type=int, help="1–50; defaults to 30 (quick/interactive), 50 (full)")
    parser.add_argument("--width", type=int, default=64, help="2–128; training width capped at 32; surface at 48")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/day58/visual-lab"))
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--no-display", action="store_true", help="Explicit default: no browser or GUI windows")
    display.add_argument("--open", action="store_true", help="Open the completed offline dashboard in the default browser")
    args = parser.parse_args(argv)
    try:
        config = Config(depth=args.depth if args.depth is not None else (50 if args.mode == "full" else 30),
                        width=args.width, seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(1)
    started = time.perf_counter()
    full = args.mode == "full"
    bundle, cache, reused = cached_bundle(config, args.output_dir, full=full)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir / f"{args.mode}-{timestamp}"
    print(f"Rendering {args.mode} artifacts to {output}", flush=True)
    artifacts = export_all(bundle, output, interactive_only=args.mode == "interactive")
    record = {"created_at_utc": timestamp, "mode": args.mode, "cache_reused": reused,
              "measurements": str(cache.resolve()), "measurement_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
              "interpretation_status": "author review required", "elapsed_seconds": time.perf_counter()-started,
              "rendering_versions": {name: version(name) for name in ("matplotlib", "plotly", "Pillow")},
              "artifacts": [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size,
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts]}
    (output/"run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    dashboard = output/"interactive/dashboard.html"
    print(f"Created {len(artifacts)} artifacts in {record['elapsed_seconds']:.1f}s; measurements: {cache}", flush=True)
    print(f"Dashboard: {dashboard.resolve()}", flush=True)
    if args.open:
        webbrowser.open(dashboard.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
