"""Generate the offline Classical NLP learning lab from one command-line entry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
from pathlib import Path
import platform
import traceback


OUTPUTS = {
    "tokenization": ("Token boundaries, sequence length and actual local BPE", ["01_tokenization.gif", "01_tokenization_explorer.html"]),
    "normalization": ("Normalization information loss and Unicode code points", ["02_normalization_pipeline.gif", "02_unicode_normalization.png"]),
    "morphology": ("Real NLTK stems and supplied-POS WordNet mappings", ["03_stemming_vs_lemmatization.html", "03_morphological_transformations.gif"]),
    "ngrams": ("Contiguous windows and occurrence counts", ["04_ngram_sliding_window.gif", "04_ngram_explorer.html"]),
    "vocabulary": ("Observed dimensions, combinatorial bounds and actual CSR bytes", ["05_vocabulary_growth.png", "05_feature_sparsity.png", "05_feature_space_explorer.html"]),
    "dtm": ("Vocabulary columns, counts and loss of order", ["06_document_term_matrix.gif", "06_bag_of_words_comparison.png"]),
    "tfidf": ("Counts, document frequency, smoothed IDF and vector normalization", ["07_tfidf_heatmaps.png", "07_idf_decay.png", "07_tfidf_surface_3d.html"]),
    "vectors": ("Original-space cosine versus a three-component SVD projection", ["08_document_vector_space_3d.html", "08_cosine_similarity_heatmap.png"]),
    "cosine": ("Direction, magnitude and zero-vector geometry", ["09_cosine_similarity.gif", "09_vector_geometry_3d.html"]),
    "language_model": ("Boundary-aware next-token counts, smoothing and sampled generation", ["10_ngram_transition_graph.html", "10_language_generation.gif", "10_smoothing_comparison.png"]),
    "retrieval": ("Hand-judged synthetic queries and preprocessing failures", ["11_preprocessing_ranking_comparison.gif", "11_retrieval_metrics.png"]),
    "ocr": ("Paired repeated synthetic corruptions and retrieval robustness", ["12_ocr_robustness.png", "12_ocr_corruption_explorer.html"]),
    "rag": ("Illustrative hybrid retrieval architecture, without fabricated dense results", ["13_hybrid_rag_architecture.png"]),
}


def validate_asset(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size < 100:
        raise ValueError(f"Missing or empty generated artifact: {path}")
    result = {"path": path.name, "bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "status": "generated and structurally readable"}
    if path.suffix in (".gif", ".png"):
        from PIL import Image
        with Image.open(path) as image:
            result.update({"width": image.width, "height": image.height, "frames": getattr(image, "n_frames", 1)})
            for i in range(result["frames"]):
                image.seek(i); image.load()
    else:
        from html.parser import HTMLParser
        class ExternalTags(HTMLParser):
            def handle_starttag(self, tag, attrs):
                for name, value in attrs:
                    if name in ("src", "href") and value and value.startswith(("http:", "https:", "//")):
                        raise ValueError(f"External asset dependency: {value}")
        contents = path.read_text(encoding="utf-8")
        ExternalTags().feed(contents)
        if "<html" not in contents or "<script" not in contents:
            raise ValueError("Interactive HTML must be a complete executable document")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--topic", action="append", choices=tuple(OUTPUTS))
    selection.add_argument("--list", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "assets" / "generated")
    p.add_argument("--seed", type=int, default=63)
    p.add_argument("--quick", action="store_true", help="Lower raster resolution and 4 OCR trials instead of 20")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.list:
        for name, (purpose, _) in OUTPUTS.items(): print(f"{name:16} {purpose}")
        return 0
    if args.seed < 0: parser().error("seed must be nonnegative")
    from nlp_lab import core
    from nlp_lab.figures import GENERATORS
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib", "plotly", "Pillow", "networkx", "streamlit")}
    try: versions["nltk"] = importlib.metadata.version("nltk")
    except importlib.metadata.PackageNotFoundError: versions["nltk"] = "unavailable"
    report = {"run_id": stamp, "seed": args.seed, "quick": args.quick, "python": platform.python_version(),
              "versions": versions, "synthetic_data": True, "author_review": "required", "topics": {}}
    source_root = Path(__file__).resolve().parent
    report["source_sha256"] = {str(path.relative_to(source_root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [source_root / "visualizations.py", *sorted((source_root / "nlp_lab").glob("*.py"))]}
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"topics": {}}
    failed = False
    for name in OUTPUTS if args.all else dict.fromkeys(args.topic):
        purpose, filenames = OUTPUTS[name]
        print(f"Generating {name}...", flush=True)
        try:
            result = GENERATORS[name](output, args.quick, args.seed)
            artifacts = [validate_asset(output / filename) for filename in filenames]
            limited = bool(result.get("unavailable") or result.get("wordnet_unavailable"))
            status = "executed with resource limitations" if limited else "executed"
            entry = {"status": status, "run_id": stamp, "purpose": purpose, "artifacts": artifacts,
                     "classification": "regenerable artifact; no public preview selected"}
            report["topics"][name] = {"status": status, "result": result}
        except core.ResourceUnavailable as exc:
            entry = {"status": "resource unavailable", "run_id": stamp, "purpose": purpose, "limitation": str(exc), "artifacts": []}
            report["topics"][name] = entry
            print(str(exc), flush=True)
        except Exception as exc:
            failed = True
            entry = {"status": "failed", "run_id": stamp, "purpose": purpose, "error": str(exc), "artifacts": []}
            report["topics"][name] = entry
            traceback.print_exc()
        manifest["topics"][name] = entry
        print(f"{name}: {entry['status']}", flush=True)
    record = f"experiments-{stamp}.json"
    (output / record).write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    manifest["latest_run"] = record
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    lines = ["# Generated Classical NLP lab", "", f"Latest dated record: `{record}`. Seed {args.seed}; quick={args.quick}.",
             "All artifacts are regenerable and ignored. Interpretation review remains required.", "", "| Topic | Question | Status | Relative files |", "|---|---|---|---|"]
    for name, entry in manifest["topics"].items():
        links = ", ".join(f"[{a['path']}]({a['path']})" for a in entry["artifacts"])
        lines.append(f"| {name} | {entry['purpose']} | {entry['status']} | {links} |")
    (output / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}\nDated evidence: {output / record}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
