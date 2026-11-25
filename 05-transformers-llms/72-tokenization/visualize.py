"""Export all 16 Day 72 visual groups, or one section, on local CPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import pandas as pd
from PIL import Image

from visual_core import ROOT, MODELS_DIR, SEED, prepare_tokenizers, build_experiments
import visual_figures as figures
from visual_html import write_page, write_gallery

SECTIONS = ("pipeline", "bpe", "inference", "wordpiece", "unigram", "sentencepiece",
            "unicode", "comparison", "vocabulary", "embedding", "attention",
            "multilingual", "ocr", "rag", "context", "cost")


def inspect_artifact(path):
    if path.suffix in (".png", ".gif"):
        with Image.open(path) as image:
            frames = getattr(image, "n_frames", 1)
            for frame in range(frames):
                image.seek(frame); image.load()
            if path.suffix == ".gif" and frames < 2:
                raise ValueError(f"animation must contain distinct frames: {path.name}")
            return {"width": image.width, "height": image.height, "frames": frames}
    if path.suffix == ".html":
        source = path.read_text(encoding="utf-8")
        if '<script src="http' in source or '<script src="//' in source:
            raise ValueError("external script dependency")
        if path.name != "index.html" and ('id="lab-data"' not in source or "Plotly.react" not in source):
            raise ValueError("interactive page missing data or Plotly rendering")
    return {}


def generate(output_dir, *, section=None, quick=False):
    if section is not None and section not in SECTIONS:
        raise ValueError(f"unknown section; expected one of {SECTIONS}")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("choose a new, empty output directory; prior runs are preserved")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT.parents[1] / ".local/tiktoken"))
    local = prepare_tokenizers(MODELS_DIR)
    data = build_experiments(local, quick=quick)
    figures.configure()
    artifacts = []

    def emit(name, renderer):
        print(f"  Rendering {name}", flush=True)
        path = output_dir / name
        renderer(path)
        artifacts.append({"name": name, "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          **inspect_artifact(path)})

    def png(name, draw):
        emit(name, lambda p: figures.save_png(p, draw, quick))

    def gif(name, draw, frames):
        emit(name, lambda p: figures.save_gif(p, draw, frames, quick))

    def page(name, kind):
        emit(name, lambda p: write_page(p, kind, data, local))

    def csv(name, rows):
        emit(name, lambda p: pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8"))

    for name in SECTIONS if section is None else (section,):
        print(f"Section: {name}", flush=True)
        if name == "pipeline":
            png("01_tokenization_pipeline.png", lambda f: figures.pipeline(f, data))
            gif("01_tokenization_pipeline.gif", lambda f, i: figures.pipeline(f, data, i), 28)
        elif name == "bpe":
            png("02_bpe_training_final.png", lambda f: figures.bpe(f, data))
            gif("02_bpe_training.gif", lambda f, i: figures.bpe(f, data, i), 21)
            csv("02_bpe_merge_history.csv", [{k: s[k] for k in ("step", "selected", "vocabulary_size", "weighted_tokens", "compression")} for s in data["bpe"]])
        elif name == "inference":
            gif("03_bpe_inference.gif", lambda f, i: figures.inference(f, data, i), sum(map(len, data["inference"].values())))
            page("03_bpe_inference.html", "inference")
        elif name == "wordpiece":
            png("04_bpe_vs_wordpiece.png", lambda f: figures.association(f, data))
            page("04_bpe_vs_wordpiece.html", "association")
            gif("04_wordpiece_inference.gif", lambda f, i: figures.wordpiece(f, data, i), len(data["wordpiece"]["trace"]))
        elif name == "unigram":
            png("05_unigram_lattice.png", lambda f: figures.unigram(f, data))
            gif("05_unigram_segmentation.gif", lambda f, i: figures.unigram(f, data, i), 14)
            page("05_unigram_interactive.html", "unigram")
        elif name == "sentencepiece":
            png("06_sentencepiece_comparison.png", lambda f: figures.sentencepiece(f, data, 2))
            gif("06_sentencepiece_whitespace.gif", lambda f, i: figures.sentencepiece(f, data, i), 3)
        elif name == "unicode":
            png("07_unicode_comparison.png", lambda f: figures.unicode(f, data, 5))
            gif("07_unicode_byte_tokenization.gif", lambda f, i: figures.unicode(f, data, i), len(data["unicode"]))
        elif name == "comparison":
            png("08_tokenizer_comparison.png", lambda f: figures.comparison(f, data))
        elif name == "vocabulary":
            png("09_vocabulary_vs_compression.png", lambda f: figures.vocabulary(f, data))
            gif("09_vocabulary_growth.gif", lambda f, i: figures.vocabulary(f, data, i), 8)
            page("09_vocabulary_tradeoff.html", "vocabulary")
            csv("09_vocabulary_metrics.csv", data["vocabulary"])
        elif name == "embedding":
            page("10_embedding_parameters_3d.html", "embedding")
            png("10_embedding_memory.png", lambda f: figures.embedding(f, data))
            gif("10_embedding_lookup.gif", lambda f, i: figures.embedding(f, data, i), 6)
        elif name == "attention":
            gif("11_attention_complexity.gif", lambda f, i: figures.attention(f, data, i), 6)
            page("11_attention_complexity_3d.html", "attention")
            png("11_attention_scaling.png", lambda f: figures.attention(f, data))
        elif name == "multilingual":
            png("12_multilingual_efficiency.png", lambda f: figures.multilingual(f, data))
            png("12_multilingual_heatmap.png", lambda f: figures.multilingual(f, data, heatmap=True))
            page("12_multilingual_comparison.html", "multilingual")
            csv("12_multilingual_metrics.csv", data["multilingual"])
        elif name == "ocr":
            gif("13_ocr_token_inflation.gif", lambda f, i: figures.ocr(f, data, i), 4)
            png("13_ocr_noise_experiment.png", lambda f: figures.ocr(f, data))
            page("13_ocr_noise_interactive.html", "ocr")
            csv("13_ocr_metrics.csv", data["ocr"])
        elif name == "rag":
            gif("14_rag_chunking.gif", lambda f, i: figures.rag(f, data, i), 3)
            png("14_rag_chunking_comparison.png", lambda f: figures.rag(f, data))
            page("14_rag_chunking_interactive.html", "rag")
        elif name == "context":
            page("15_context_window_simulator.html", "context")
            png("15_context_budget.png", lambda f: figures.context(f, data))
        elif name == "cost":
            page("16_llm_cost_simulator.html", "cost")
            page("16_monthly_cost_3d.html", "monthly")
            png("16_cost_sensitivity.png", lambda f: figures.cost(f, data))
    (output_dir / "experiments.json").write_text(json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False), encoding="utf-8")
    write_gallery(output_dir / "index.html", artifacts)
    manifest = {"run_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
                "mode": "quick" if quick else "full", "sections": list(SECTIONS) if section is None else [section],
                "versions": {"python": platform.python_version(), **{p: version(p) for p in ("numpy", "matplotlib", "Pillow", "plotly", "tiktoken", "tokenizers", "sentencepiece", "streamlit")}},
                "model_metadata": json.loads((MODELS_DIR / "training.json").read_text()),
                "hypotheses": ["stored merge ranks explain inference", "assigned Unigram probabilities determine best path", "encoding and controlled corruption affect token counts", "chunk boundaries affect known-evidence containment"],
                "interpretation_status": "author review required", "artifacts": artifacts,
                "limitations": "synthetic probes; local models differ from pretrained encodings; theoretical proxies are not hardware measurements; no LLM or retrieval evaluation"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Generated {len(artifacts)} artifacts + gallery, experiment record and manifest at {output_dir}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=SECTIONS)
    parser.add_argument("--quick", action="store_true", help="Fewer GIF frames, lower DPI, three OCR trials instead of twelve")
    parser.add_argument("--setup", action="store_true", help="Cache pretrained encodings and train local synthetic tokenizer models only")
    parser.add_argument("--output-dir", type=Path, help="Relative paths resolve against this topic directory")
    args = parser.parse_args()
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT.parents[1] / ".local/tiktoken"))
    if args.setup:
        from visual_core import get_encoding
        for name in ("cl100k_base", "o200k_base"):
            get_encoding(name)
        prepare_tokenizers(MODELS_DIR)
        print("Pretrained resources cached; local synthetic models prepared.")
        return
    folder = args.output_dir or Path("outputs") / ("run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if not folder.is_absolute():
        folder = ROOT / folder
    generate(folder, section=args.section, quick=args.quick)


if __name__ == "__main__":
    main()
