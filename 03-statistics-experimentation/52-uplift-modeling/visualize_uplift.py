"""Generate and open the complete offline Day 52 uplift learning lab."""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import time
import webbrowser

import matplotlib
import numpy as np
import PIL
import plotly
from plotly.offline import get_plotlyjs
import scipy
import sklearn

from uplift_visual_core import build_payload


ROOT = Path(__file__).resolve().parent


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_dashboard(payload, directory, assets=None):
    """Inline data, local Plotly, UI and images so index.html works by itself."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    html = (ROOT / "visual_lab.html").read_text(encoding="utf-8")
    data = json.dumps(payload, default=json_default, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("__PLOTLY__", get_plotlyjs())
    html = html.replace("__DATA__", data)
    html = html.replace("__LAB_JS__", (ROOT / "visual_lab.js").read_text(encoding="utf-8"))

    def embed(match):
        name = match.group(1)
        asset = (assets or {}).get(name)
        if asset is None:
            return '<p class="caption">Animation export disabled; the interactive explanation remains available.</p>'
        path = directory / "assets" / asset["file"]
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = "image/gif" if asset["kind"] == "gif" else "image/png"
        return f'<img class="animation" alt="{name.replace("_", " ")}; synthetic teaching animation" src="data:{mime};base64,{encoded}">'

    html = re.sub(r"<!-- ASSET:(\w+) -->", embed, html)
    path = directory / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def generate(directory=ROOT / "visualizations", seed=520, n_train=6000, n_test=3000,
             repetitions=30, frames=24, gifs=True):
    started = time.perf_counter()
    if n_train < 200 or n_test < 100 or not 2 <= repetitions <= 200 or not 4 <= frames <= 60:
        raise ValueError("require train>=200, test>=100, repetitions in [2,200], frames in [4,60]")
    directory = Path(directory).resolve()
    print("Generating synthetic data, three forests and repeated sampling studies...", flush=True)
    payload = build_payload(seed, n_train, n_test, repetitions)
    assets = {}
    if gifs:
        from uplift_animations import generate_animations
        print("Exporting four bounded Pillow animations...", flush=True)
        assets = generate_animations(payload, directory / "assets", frames)
    print("Embedding Plotly, data, interface and images in offline HTML...", flush=True)
    index = write_dashboard(payload, directory, assets)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "Day 52 visual lab; separate generator from historical analytical study",
        "hypotheses": ["Response probability and incremental effect need not rank people identically",
                       "Small randomized subgroups yield variable effect estimates",
                       "Poor overlap can amplify IPW variability"],
        "config": payload["config"], "metrics": payload["metrics"], "assets": assets,
        "sample_sizes": payload["sample_sizes"], "overlap_records": {
            name: data["records"] for name, data in payload["overlap"].items()
        },
        "interpretation_status": "author review required",
        "limitations": ["Synthetic data, no production evidence", "Single model-comparison train/test seed",
                        "Pointwise approximate subgroup intervals; no curve/policy confidence intervals",
                        "True known propensity in overlap example; no hidden-confounder adjustment"],
        "versions": dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         sklearn=sklearn.__version__, plotly=plotly.__version__,
                         matplotlib=matplotlib.__version__, pillow=PIL.__version__),
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    records = directory / "runs"
    records.mkdir(parents=True, exist_ok=True)
    record_path = records / ("visual-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    with record_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, default=json_default, allow_nan=False, indent=2)
    print(json.dumps({"index": str(index), "record": str(record_path), "metrics": record["metrics"],
                      "assets": assets, "elapsed_seconds": record["elapsed_seconds"]}, indent=2), flush=True)
    return index, record_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "visualizations")
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--test", type=int, default=3000)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--no-gifs", action="store_true", help="Keep interactive modules, skip animation exports")
    parser.add_argument("--no-open", action="store_true", help="Do not launch the default browser")
    args = parser.parse_args()
    try:
        index, _ = generate(args.output_dir, args.seed, args.train, args.test,
                            args.repetitions, args.frames, not args.no_gifs)
    except ValueError as error:
        parser.error(str(error))
    if not args.no_open:
        if not webbrowser.open(index.as_uri()):
            print(f"Browser did not open automatically. Open {index} manually.")


if __name__ == "__main__":
    main()
