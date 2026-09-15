"""Generate and explore the Day 17 Visual Validation Lab offline."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import platform
import sys

import matplotlib
import numpy as np
import PIL
import plotly
import sklearn

import visual_experiments as exp
import visualizations as visuals

CATALOG = [
    ("Train / validation / test", "Where does each observation go?", "A disjoint holdout assigns each row exactly one development or test role.", visuals.visualize_train_val_test),
    ("K-Fold animation", "How does validation rotate across fits?", "Each row is held out once, while the estimator is refitted for each fold.", visuals.animate_kfold),
    ("Stratified K-Fold", "Do folds preserve the rare class?", "Stratification can stabilize fold class proportions without fixing dependence.", visuals.compare_kfold_stratified),
    ("Group leakage", "Can customer recognition masquerade as generalization?", "Random row folds can reward stable entity signatures under an unseen-customer target.", visuals.visualize_group_leakage),
    ("Temporal split", "Does training contain the validation future?", "Random and chronological splits create different information boundaries.", visuals.visualize_temporal_split),
    ("Expanding / rolling", "Which past observations survive retraining?", "A rolling window trades historical sample size for recency under drift.", visuals.animate_windows),
    ("Concept drift", "How does changing P(Y|X) affect validation?", "Mixed-period validation and future prediction need not have similar errors.", visuals.simulate_concept_drift),
    ("Preprocessing / feature selection", "Who influences fitted transformations?", "Held-out covariates or labels can influence globally fitted preprocessing.", visuals.demonstrate_preprocessing_leakage),
    ("Target leakage", "Is the predictor using a post-outcome measurement?", "A target-derived feature changes the information problem being scored.", visuals.demonstrate_target_leakage),
    ("Temporal feature leakage", "Can March know full-year purchase statistics?", "As-of and full-period aggregation use different information windows.", visuals.visualize_temporal_feature_leakage),
    ("Validation overfitting", "Can selection learn validation noise?", "Searching more null candidates offers more chances to select favorable noise.", visuals.simulate_validation_overfitting),
    ("Nested CV", "What separates tuning from outer evaluation?", "Inner selection can remain isolated from every outer holdout.", visuals.visualize_nested_cv),
    ("Holdout variability", "How much does the partition affect the estimate?", "A fixed dataset can yield different estimates across split seeds.", visuals.plot_holdout_variability),
    ("3D temporal boundary", "Where does time separate train and holdouts?", "Temporal membership forms disjoint time slices in the same feature cloud.", visuals.plot_temporal_split_3d),
    ("Production information boundary", "What exists at inference time?", "Feature availability and generalization population define the valid experiment.", visuals.visualize_production_boundary),
    ("RAG / evaluation questions", "Are chunks the right independent unit?", "Document families and prompt-development questions need explicit evaluation boundaries.", visuals.visualize_rag_leakage),
]


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported record type: {type(value).__name__}")


def write_gallery(output):
    """Link only completed records and their existing generated artifacts."""
    sections = []
    for path in sorted(output.glob("experiment_*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        assets = []
        for name in entry["artifacts"]:
            if not (output / name).is_file():
                continue
            safe_name = html.escape(name, quote=True)
            if Path(name).suffix == ".html":
                assets.append(f'<p><a href="{safe_name}">Open interactive 3D view</a></p>')
            else:
                assets.append(f'<a href="{safe_name}"><img loading="lazy" src="{safe_name}" alt="{safe_name}"></a>')
        sections.append(
            f'<section id="view-{entry["view"]}"><h2>{entry["view"]:02d} / {html.escape(entry["title"])}</h2>'
            f'<p>{html.escape(entry["question"])}</p>' + "".join(assets) +
            '<p><strong>Interpretation candidate - author review:</strong> ' +
            html.escape(entry["interpretation_candidate_for_author_review"]) +
            '</p><p class="muted">Limitation: ' + html.escape(entry["limitation"]) +
            f'</p><p class="muted">Seed {entry["seed"]} | <a href="{path.name}">Experiment record</a></p></section>'
        )
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Day 17 / Visual Validation Lab</title>
<style>body{font:17px/1.6 system-ui,sans-serif;color:#183348;background:#f7f8fa;margin:0}
main{max-width:1150px;margin:auto;padding:48px 24px}h1{font-size:42px;line-height:1.15}
h2{font-size:27px}section{padding:25px 0 45px;border-top:1px solid #c8d1d9}
img{display:block;width:100%;height:auto;margin:18px 0;border-radius:8px}
a{color:#246caa}.muted{color:#596e7e;font-size:15px}.eyebrow{letter-spacing:.12em;font-size:13px}
</style><main><p class="eyebrow">APPLIED AI ENGINEERING LAB / DAY 17</p>
<h1>Visual Validation Lab</h1><p>What information is the model allowed to see?</p>
<p>All data is synthetic. Metrics come from executed code. Interpretation candidates require author review.
Click a figure to open it; the 3D view runs offline in your browser.</p>"""
    (output / "index.html").write_text(document + "".join(sections) + "</main></html>", encoding="utf-8")


def run_view(number, seed, output):
    if not 1 <= number <= len(CATALOG):
        raise ValueError("view must be between 1 and 16")
    seed = exp.checked_seed(seed)
    output = Path(output)
    title, question, hypothesis, function = CATALOG[number-1]
    print(f"\n{number:02d} / {title}\nQuestion: {question}\nHypothesis: {hypothesis}", flush=True)
    renderer = visuals.Renderer(output)
    result = function(renderer, seed)
    result.update({
        "view": number, "title": title, "question": question, "hypothesis": hypothesis,
        "seed": seed, "data_source": "synthetic, generated in memory",
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                     "plotly": plotly.__version__, "pillow": PIL.__version__},
        "artifacts": renderer.artifacts,
    })
    (output / f"experiment_{number:02d}.json").write_text(
        json.dumps(result, default=json_value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_gallery(output)
    print("Interpretation candidate (author review): " + result["interpretation_candidate_for_author_review"], flush=True)
    print("Limitation: " + result["limitation"], flush=True)
    print("Saved: " + ", ".join(renderer.artifacts), flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Generate all 16 views.")
    selection.add_argument("--view", type=int, choices=range(1, 17), help="Generate a single numbered view.")
    selection.add_argument("--list", action="store_true", help="List questions without fitting or rendering.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        exp.checked_seed(args.seed)
        if args.list or (args.view is None and not args.all):
            for i, (title, question, _, _) in enumerate(CATALOG, 1):
                print(f"{i:2d} - {title}: {question}")
            print("17 - Run all")
            if args.list:
                return 0
            if not sys.stdin.isatty():
                print("Noninteractive execution: use --all or --view N.", file=sys.stderr)
                return 2
            chosen = input("Choose 1..17 (q to quit): ").strip()
            if chosen.lower() == "q":
                return 0
            if not chosen.isdigit() or not 1 <= int(chosen) <= 17:
                raise ValueError("choose a number from 1 to 17")
            args.all = chosen == "17"
            args.view = int(chosen) if not args.all else None
        for number in (range(1, 17) if args.all else [args.view]):
            run_view(number, args.seed, args.output)
        print(f"\nOpen the local gallery: {(args.output / 'index.html').resolve()}")
        return 0
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
