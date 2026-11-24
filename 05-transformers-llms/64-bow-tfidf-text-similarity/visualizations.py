"""Generate the Day 64 offline visual laboratory and dated execution evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import importlib.metadata
import json
from pathlib import Path
import platform
import traceback
import webbrowser


SECTIONS = {
    "tokens": ("How do tokens map to vocabulary columns?", ["01_text_to_vector.gif"]),
    "tfidf": ("What changes across weighting conventions?", ["02_representation_comparison.png", "03_tfidf_transformation.gif", "12_tfidf_weight_surface_3d.html"]),
    "sparse": ("What does CSR actually store?", ["04_sparse_matrix_structure.png", "05_csr_representation.gif", "06_sparse_vs_dense_memory.png"]),
    "tf": ("How do TF saturation and length assumptions differ?", ["07_term_frequency_comparison.png", "08_term_frequency_interactive.html"]),
    "idf": ("How does document prevalence affect IDF?", ["09_idf_comparison.png", "10_idf_rarity_animation.gif", "11_idf_interactive.html"]),
    "vectors": ("How do IDF and L2 change true vocabulary vectors?", ["13_bow_vector_space_3d.html", "14_tfidf_vector_transformation.gif", "15_normalized_vector_space_3d.html"]),
    "cosine": ("Which effects depend on angle or magnitude?", ["16_cosine_similarity_2d.html", "17_cosine_similarity_3d.html", "18_cosine_similarity_animation.gif"]),
    "l2": ("When do dot, cosine and distance rankings agree?", ["19_l2_normalization.gif", "20_similarity_metrics_comparison.png"]),
    "similarity": ("How does representation change document similarities?", ["21_document_similarity_heatmaps.png", "22_document_similarity_clusters.png"]),
    "projections": ("What does a low-dimensional SVD view leave out?", ["23_document_projection_2d.html", "24_document_projection_3d.html", "25_bow_vs_tfidf_projection.png"]),
    "ngrams": ("What happens to word order, negation and OCR overlap?", ["26_ngram_feature_comparison.png", "27_ngram_tokenization.gif", "28_ocr_similarity_comparison.png"]),
    "retrieval": ("Which features contribute to a query score?", ["29_query_ranking_comparison.png", "30_term_score_contributions.png"]),
    "inverted": ("How do postings select candidate document IDs?", ["31_inverted_index_search.gif"]),
    "bm25": ("How do saturation and length affect lexical ranking?", ["32_bm25_vs_tfidf_ranking.png", "33_bm25_saturation_3d.html", "34_bm25_length_normalization.gif"]),
    "mismatch": ("Where does partial lexical overlap fail?", ["35_vocabulary_mismatch.png"]),
    "drift": ("What changes when corpus statistics are refreshed?", ["37_corpus_drift_animation.gif"]),
    "hybrid": ("How are two actual lexical rankings fused?", ["38_hybrid_retrieval_pipeline.png", "39_reciprocal_rank_fusion.gif", "40_hybrid_retrieval_comparison.png"]),
    "evaluation": ("Which fixed judged queries fail for each method?", ["41_retrieval_evaluation.png", "42_per_query_failure_analysis.png"]),
}
QUICK = ("tokens", "tfidf", "idf", "vectors", "cosine", "l2", "similarity", "retrieval")
EXPERIMENTS = {
    "tokens": ("Every analyzed token maps to the same sorted-vocabulary column in counts and CSR.", "Six DOCUMENTS; lowercase Unicode word tokens, length>=2.", "No linguistic segmentation or identifier-aware parsing."),
    "tfidf": ("Equal local counts can have different IDF weights; nonzero L2 rows have norm one.", "Six DOCUMENTS; raw TF, smoothed IDF; five aligned representations.", "Weight distinctiveness does not establish relevance."),
    "sparse": ("At these occupancies, CSR component storage is smaller than float64 dense storage.", "Seed 64; (N,V,unique)=(10,50,5),(80,1000,20),(300,5000,30); Poisson(2)+1 counts.", "Array bytes exclude object overhead, allocator, peak RSS and latency."),
    "tf": ("Log TF grows more slowly than raw TF; BM25 approaches a finite limit at fixed length ratio.", "TF=0..100; length=avgdl=100; k1=1.2,b=.75. Interactive inputs enforce TF<=length.", "Different factor scales; repetition replaces other tokens at fixed length."),
    "idf": ("Increasing DF at fixed N decreases all three IDF variants.", "N=10,100,1000; 1<=DF<=N; animation fixed N=10.", "Rarity is not semantic importance; DF=0 is outside the plotted domain."),
    "vectors": ("IDF can alter direction; subsequent L2 scaling preserves that weighted direction.", "Four documents; explicit vocabulary [machine,learning,data]; smoothed IDF.", "True three-feature fixture is not a projection or a general text corpus."),
    "cosine": ("Positive vector scaling preserves cosine; rotation changes it; zero-vector cosine is undefined.", "GIF angles 0..180 degrees; HTML magnitudes 0..4, 3D azimuth/elevation.", "Signed general-vector examples differ from nonnegative TF-IDF."),
    "l2": ("A large off-angle document can win raw dot while an aligned document wins cosine.", "Q=[1,1]; D0=[1,1],D1=[100,0],D2=[0,0]; separate radial animation.", "Unit-vector distance identity excludes zero rows."),
    "similarity": ("Representation changes off-diagonal overlap while nonzero cosine diagonals remain one.", "Six DOCUMENTS; binary/count/word TF-IDF/char_wb 3..5; average linkage on 1-cos.", "Similarity and clusters are lexical, not relevance labels."),
    "projections": ("SVD gives approximate views that can change original cosine rankings.", "36 documents; six categories; three components, ten iterations, random_state=64; separate count/TF-IDF fits.", "Explained variance does not measure retrieval-rank preservation."),
    "ngrams": ("Bigrams distinguish reordered tokens; character features retain partial OCR overlap.", "Seven NGRAM_DOCUMENTS; word unigram/bigram/combined and char_wb 3..5.", "N-grams do not establish negation understanding or general OCR accuracy."),
    "retrieval": ("Normalized feature products and BM25 term contributions sum to their method's score.", "14 SEARCH_DOCUMENTS; query=document retrieval; positive top-5; k1=1.2,b=.75.", "Exhaustive teaching computation; score scales are method-specific."),
    "inverted": ("The union of query postings contains all positive word-overlap candidates.", "14 SEARCH_DOCUMENTS; query=document retrieval; OR candidates; BM25 top-3.", "Illustrative access counts are not measured latency."),
    "bm25": ("Saturation and corpus-relative length can change the TF-IDF/BM25 ordering.", "Five explicit repetition/filler documents; query=rare common; positive IDF; k1=1.2,b=.75.", "No parity with Lucene norms or production quality claim."),
    "mismatch": ("Word TF-IDF can have zero overlap on synonyms; character features retain some spelling overlap.", "Five fixed pairs; fit on all ten strings; word TF-IDF vs char_wb 3..5.", "No dense embeddings, learned semantics or benchmark."),
    "drift": ("Appending alpha-only text changes relative IDF and can reverse the initial top pair.", "Three initial documents; append alpha novel 0..6 times; query=alpha beta; frozen vs refitted TF-IDF.", "Designed drift fixture; both pipelines use internally consistent IDF versions."),
    "hybrid": ("RRF sums contributions from actual word/character candidate ranks, including missing items.", "14 SEARCH_DOCUMENTS; fictional SYN-204 query; top-5 per list; RRF constant=60; context top-3.", "Lexical+lexical fusion; no neural retrieval or generated answer."),
    "evaluation": ("Fixed task judgments expose per-query tradeoffs and failures hidden by averages.", "14 documents; eight fixed graded queries; k=3; exponential DCG gains; full-positive MRR; macro means.", "No held-out validation, author-approved labels or population inference."),
}


def validate_asset(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 100:
        raise ValueError(f"Missing or empty asset: {path}")
    record = {"file": f"{path.parent.name}/{path.name}", "bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if path.suffix in (".png", ".gif"):
        from PIL import Image
        with Image.open(path) as image:
            record.update(width=image.width, height=image.height, frames=getattr(image, "n_frames", 1))
            durations = []
            for frame in range(record["frames"]):
                image.seek(frame); image.load(); durations.append(image.info.get("duration", 0))
            record["duration_ms"] = sum(durations)
    else:
        class CheckTags(HTMLParser):
            def handle_starttag(self, tag, attrs):
                for name, value in attrs:
                    if name in ("src", "href") and value and value.startswith(("http:", "https:", "//")):
                        raise ValueError(f"Remote HTML dependency: {value}")
        source = path.read_text(encoding="utf-8")
        CheckTags().feed(source)
        if "<html" not in source or "<script" not in source:
            raise ValueError("Expected a complete interactive HTML document")
    return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    select = p.add_mutually_exclusive_group(required=True)
    select.add_argument("--all", action="store_true")
    select.add_argument("--quick", action="store_true", help="Essential sections only, lower raster resolution")
    select.add_argument("--section", choices=tuple(SECTIONS), action="append")
    select.add_argument("--asset", choices=tuple(asset for _, assets in SECTIONS.values() for asset in assets), action="append")
    select.add_argument("--list", action="store_true")
    p.add_argument("--output-dir", type=Path, help="New empty destination; defaults to a unique dated outputs/ directory")
    p.add_argument("--show", action="store_true", help="Open generated HTML pages after successful generation")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.list:
        for section, (question, assets) in SECTIONS.items():
            print(f"{section}: {question}\n  " + ", ".join(assets))
        print("Optional dense-model comparison (36): not implemented; no model downloads or fabricated scores.")
        return 0
    from text_lab import core, figures, interactive
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or Path(__file__).resolve().parent / "outputs" / stamp
    if output.exists() and any(output.iterdir()):
        raise ValueError("output-dir must be empty; preserve historical runs by choosing a new directory")
    output.mkdir(parents=True, exist_ok=True)
    export = figures.Exporter(output, quick=args.quick, targets=args.asset)
    selected = list(SECTIONS) if args.all else list(QUICK) if args.quick else list(dict.fromkeys(args.section or [
        name for name, (_, assets) in SECTIONS.items() if any(asset in assets for asset in args.asset)]))
    functions = {"tokens": figures.tokens, "tfidf": figures.tfidf, "sparse": figures.sparse_storage,
                 "tf": figures.frequency, "idf": figures.rarity, "vectors": figures.vectors,
                 "cosine": figures.cosine, "l2": figures.normalization, "similarity": figures.similarities,
                 "projections": figures.projections, "ngrams": figures.ngrams, "retrieval": figures.retrieval,
                 "inverted": figures.inverted, "bm25": figures.bm25, "mismatch": figures.mismatch,
                 "drift": figures.drift, "hybrid": figures.hybrid, "evaluation": figures.evaluate}
    report = {"run_id": stamp, "seed": 64, "quick": args.quick, "platform": platform.platform(),
              "versions": {name: importlib.metadata.version(name) for name in
                           ("numpy", "scipy", "scikit-learn", "matplotlib", "plotly", "Pillow")},
              "python": platform.python_version(), "data": "deterministic synthetic fixtures; no external data",
              "interpretation": "author review required", "numerical_validation": core.validate_numerics(),
              "optional_dense": {"status": "not executed", "reason": "No embedding model or download configured; RRF uses word and character TF-IDF"},
              "sections": {}}
    failed = False
    for section in selected:
        print(f"Generating {section}...", flush=True)
        try:
            result = functions[section](export)
            interactive.generate(export, section)
            expected = [name for name in SECTIONS[section][1] if export.wants(name)]
            assets = [validate_asset(export.path(name)) for name in expected]
            report["sections"][section] = {"status": "executed", "question": SECTIONS[section][0],
                "hypothesis": EXPERIMENTS[section][0],
                "configuration": {"seed": 64, "fixture": EXPERIMENTS[section][1], "quick": args.quick},
                "result": result, "interpretation_candidate": "Inspect computed changes with the stated representation and normalization; author review required.",
                "limitation": EXPERIMENTS[section][2],
                "assets": assets}
            print(f"{section}: {len(assets)} assets generated and structurally readable", flush=True)
        except Exception as error:
            failed = True
            report["sections"][section] = {"status": "failed", "error": str(error)}
            traceback.print_exc()
    result_folder = output / "results"; result_folder.mkdir(exist_ok=True)
    record = result_folder / f"experiments-{stamp}.json"
    record.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    lines = ["# Day 64 generated visualization catalog", "", f"Run: `{stamp}`. Seed 64. Quick={args.quick}.",
             "All assets remain ignored regenerable artifacts. Author interpretation review required.", "",
             "| Filename | Concept | Format | Interpretation / boundary |", "|---|---|---|---|"]
    for section, entry in report["sections"].items():
        for asset in entry.get("assets", []):
            file = asset["file"]
            lines.append(f"| [{file}]({file}) | {SECTIONS[section][0]} | {Path(file).suffix[1:]} | Computed synthetic mechanism; inspect the section's limitations |")
    lines += ["", "Dense comparison 36: not executed. RRF is lexical + lexical, without generated semantic scores.",
              f"Execution evidence: [dated JSON]({record.relative_to(output).as_posix()})."]
    (output / "CATALOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in export.paths:
        print(path, flush=True)
    print(f"Numerical validation: {report['numerical_validation']['status']}; dense comparison: not executed\nRecord: {record}", flush=True)
    if args.show and not failed:
        for path in export.paths:
            if path.suffix == ".html": webbrowser.open(path.resolve().as_uri())
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
