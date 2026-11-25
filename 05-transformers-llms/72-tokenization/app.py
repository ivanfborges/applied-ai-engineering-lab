"""Live local dashboard: streamlit run 05-transformers-llms/72-tokenization/app.py."""

from __future__ import annotations

import html
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parents[2] / ".local/tiktoken"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from example import context_budget, estimated_cost, vocabulary_budget
from visual_core import (ROOT, MODELS_DIR, SEED, PRESETS, ALGORITHMS, UNIGRAM_PROBS, RAG_DOCUMENT,
                         prepare_tokenizers, compare_text, build_experiments,
                         unigram_viterbi, exhaustive_paths, association_scores,
                         chunk_document, chunk_metrics, attention_counts,
                         pipeline_state, synthetic_embeddings, prune_context, corrupt)
from visual_figures import color


SECTIONS = ["Tokenization fundamentals", "BPE training and inference", "WordPiece and Unigram",
            "SentencePiece and Unicode", "Tokenizer comparison", "Vocabulary and computation",
            "Multilingual efficiency", "OCR and document intelligence", "RAG and context", "LLM cost estimation"]


@st.cache_resource(max_entries=1)
def local_tokenizers():
    return prepare_tokenizers(MODELS_DIR)


@st.cache_data(max_entries=1)
def measurements():
    return build_experiments(local_tokenizers(), quick=True)


def token_blocks(pieces):
    # Tokens need colored, wrapping blocks; escaping preserves arbitrary user text.
    badges = "".join(f'<span style="display:inline-block;margin:3px;padding:6px 9px;border-radius:6px;background:{color(p)};color:white;font:14px monospace;overflow-wrap:anywhere;max-width:100%">{html.escape(str(p)).replace(" ", "·").replace(chr(10), "↵")}</span>' for p in pieces)
    st.html('<div aria-label="Token boundaries">' + badges + '</div>')


def table(rows, label):
    st.dataframe(pd.DataFrame(rows), hide_index=True, alt=label)


def chart(fig, label, key=None):
    fig.update_layout(margin=dict(t=50, b=40, l=35, r=35), height=450)
    st.plotly_chart(fig, width="stretch", alt=label, key=key)


def reset_controls():
    for key in list(st.session_state):
        if key != "section":
            del st.session_state[key]


def main():
    st.set_page_config(page_title="Day 72 · Tokenization lab", layout="wide")
    st.title("Tokenization for LLMs")
    st.caption("Day 72 · Local CPU learning lab · Synthetic corpora · Interpretation review pending")
    section = st.sidebar.selectbox("Study section", SECTIONS, key="section")
    st.sidebar.button("Reset this section", on_click=reset_controls, key="reset")
    st.sidebar.caption("Use the standalone gallery for animations and offline Plotly exports. No paid API is called.")
    st.header(section)
    try:
        local = local_tokenizers()
        data = measurements()
    except (ImportError, RuntimeError, ValueError) as exc:
        st.error(str(exc)); st.stop()

    if section == SECTIONS[0]:
        st.markdown("Follow the actual encoding pipeline. Identity normalization preserves this input; embeddings below are **synthetic**, generated deterministically from seed 272 and each real token ID.")
        text = st.selectbox("Pipeline input", [p["text"] for p in data["pipeline"]], key="pipeline_text")
        stage = st.slider("Pipeline stage", 0, 6, 6, key="pipeline_stage")
        record = pipeline_state(text)
        stages = [("Raw text", text), ("Normalization (identity)", record["normalized"]),
                  ("Regex pretokenization", record["pretokenized"]),
                  ("Byte-BPE pieces (hex)", record["pieces"]), ("Token IDs", record["ids"])]
        for i, (label, value) in enumerate(stages):
            if i <= stage:
                st.caption(label); st.code(str(value), language=None)
        if stage >= 3:
            token_blocks(record["pieces"])
        if stage >= 5:
            st.caption(f"Conceptual embedding matrix {record['embedding_shape']}; only selected rows are allocated.")
            chart(go.Figure(go.Heatmap(z=record["selected_rows"], y=[f"ID {i}" for i in record["ids"]], colorscale="RdBu")), "Synthetic embedding rows indexed by actual token IDs")
        if stage >= 6:
            st.info(f"Transformer input has {len(record['ids'])} token rows × 6 synthetic features before any model layers. Position signals are a separate operation; no Transformer is executed here.")

    elif section == SECTIONS[1]:
        st.markdown("Training recomputes weighted corpus frequencies; inference applies the fixed rank table. Colored borders and pair frequencies are also animated in the generated BPE GIF.")
        step = st.slider("Training merges applied", 0, len(data["bpe"])-1, 0, key="bpe_step")
        state = data["bpe"][step]
        for row in state["corpus"]:
            st.caption(f"Frequency × {row['frequency']}"); token_blocks(row["pieces"])
        with st.container(horizontal=True):
            st.metric("Vocabulary entries", state["vocabulary_size"])
            st.metric("Weighted corpus tokens", state["weighted_tokens"])
            st.metric("Initial / current tokens", f"{state['compression']:.2f}")
        table([{"pair": "+".join(pair), "count": count} for pair, count in state["pairs"]], "Actual weighted pair counts at this BPE step")
        word = st.selectbox("Inference word", list(data["inference"]), key="bpe_word")
        trace = data["inference"][word]
        index = st.slider("Inference step", 0, max(1, len(trace)-1), 0, key="bpe_inference")
        chosen = trace[min(index, len(trace)-1)]
        token_blocks(chosen["pieces"]); st.json(chosen)
        st.caption("newest is a seen-word control; lowest/newer/slower are novel combinations. café is intentionally unsupported.")

    elif section == SECTIONS[2]:
        concept = st.segmented_control("Concept", ["WordPiece", "Unigram"], default="WordPiece", key="algorithm")
        if concept == "WordPiece":
            st.warning("The association formula below is an educational reconstruction, not a verified reproduction of the original WordPiece trainer.")
            left = st.slider("Pair B left-symbol frequency", 40, 10000, 40, 40, key="wp_left")
            scores = association_scores(left=(1000, left))
            table([{"pair": label, "joint count": scores["frequency"][i], "association": scores["association"][i]} for i, label in enumerate(["A", "B"])], "Synthetic pair scoring comparison")
            word = st.text_input("WordPiece word", "unbelievable", key="wp_word", max_chars=100)
            if word and not any(c.isspace() for c in word):
                from visual_core import wordpiece_trace
                manual = wordpiece_trace(word, local["WordPiece local"].get_vocab())
                real = local["WordPiece local"].encode(word).tokens
                token_blocks(real); st.json({"educational": manual, "actual_HF_tokens": real, "equal_for_this_word": manual["pieces"] == real})
            else:
                st.info("Enter one nonempty word to inspect greedy matching.")
            st.caption("The HF model was trained locally on a tiny synthetic corpus. Its actual trainer is not the illustrative pair-score simulator.")
        else:
            st.markdown("These are **manually assigned weights**. Sliders are normalized into a valid vocabulary distribution before scoring paths.")
            weights = {p: st.slider(f"Weight: {p}", 1, 100, round(v*100), key=f"ug_{p}") for p, v in UNIGRAM_PROBS.items()}
            probabilities = {p: v/sum(weights.values()) for p, v in weights.items()}
            result = unigram_viterbi("unbelievable", probabilities)
            step = st.slider("Viterbi end position / backtrack", 1, 13, 13, key="ug_step")
            token_blocks(result["pieces"])
            table([{"path": " + ".join(p), "probability": score, "log probability": float(np.log(score))} for p, score in exhaustive_paths("unbelievable", probabilities)], "All valid Unigram paths and actual assigned-probability products")
            st.json(result["steps"][min(step, 12)-1] if step <= 12 else {"backtrack": result["pieces"], "log_probability": result["log_probability"]})
            fig = go.Figure()
            for edge in result["edges"]:
                a, b = edge["start"], edge["end"]
                fig.add_trace(go.Scatter(x=[a, (a+b)/2, b], y=[0, .3+.08*(b-a), 0], mode="lines+text", text=["", edge["piece"], ""], name=f"p={edge['p']:.3f}", line_shape="spline"))
            fig.update_layout(xaxis_title="Character boundary", yaxis_visible=False)
            chart(fig, "Unigram candidate segmentation lattice")

    elif section == SECTIONS[3]:
        text = st.text_area("Normalization and Unicode input", "  Café   ＡＩ é e\u0301 🚀 中文  ", key="sp_text", max_chars=3000)
        for name in ("SentencePiece BPE", "SentencePiece Unigram", "cl100k_base"):
            row = compare_text(name, text, local)
            with st.container(border=True):
                st.subheader(name); token_blocks(row["pieces"])
                st.json({k: row[k] for k in ("normalized", "ids", "decoded", "unknown", "exact_reconstruction")})
        table([{"character": c, "code point": f"U+{ord(c):04X}", "UTF-8 hex": c.encode().hex(" "), "bytes": len(c.encode())} for c in text], "Input Unicode code points and UTF-8 bytes")
        st.caption("SentencePiece models share nmt_nfkc normalization, whitespace collapsing and a dummy prefix; unknown characters remain possible because byte fallback is disabled.")

    elif section == SECTIONS[4]:
        category = st.selectbox("Input category", list(PRESETS), key="category")
        text = st.text_area("Text to tokenize", PRESETS[category], key=f"text_{category}", max_chars=6000)
        names = st.multiselect("Tokenizers", ALGORITHMS, default=list(ALGORITHMS[:2]), key="tokenizers")
        results = []
        for i, name in enumerate(names):
            try:
                row = compare_text(name, text, local)
            except (ValueError, UnicodeError) as exc:
                st.warning(f"{name}: {exc}"); continue
            results.append({k: v for k, v in row.items() if k not in ("text", "normalized", "decoded", "pieces", "ids", "origin")})
            with st.container(border=True):
                st.subheader(name); st.caption(row["origin"]); token_blocks(row["pieces"])
                table([{"piece / byte hex": p, "ID": row["ids"][j] if row["ids"] is not None else None} for j, p in enumerate(row["pieces"])], f"Actual pieces and IDs for {name}")
                st.text(f"Decoded: {row['decoded']!r}")
        table(results, "Side-by-side tokenizer metrics for the current input")
        st.caption("Whitespace-word fertility is omitted for Chinese/Japanese scripts. Tiny local models and pretrained encodings differ in corpus and normalization; this is not a controlled algorithm benchmark.")

    elif section == SECTIONS[5]:
        concept = st.selectbox("Computation view", ["Vocabulary compression", "Embedding parameters", "Attention interactions"], key="computation")
        if concept == "Vocabulary compression":
            rows = pd.DataFrame(data["vocabulary"])
            st.line_chart(rows.pivot_table(index="vocabulary", columns="partition", values="mean_tokens_per_supported_word", aggfunc="first"), alt="Train and held-out tokens per supported word across BPE vocabularies")
            table(data["vocabulary"], "Measured supported coverage, rejections and compression by partition")
            st.caption("The held-out list includes café, rejected at every merge limit. Compression denominators cover supported words only.")
        elif concept == "Embedding parameters":
            vocab = st.slider("Vocabulary entries", 16000, 250000, 50000, 1000, key="vocab")
            width = st.slider("Embedding width", 512, 8192, 4096, 128, key="width")
            tied = st.toggle("Tie input/output weights", True, key="tied")
            precision = st.selectbox("Ideal weight precision", ["FP32", "FP16/BF16", "INT8"], key="precision")
            budget = vocabulary_budget(vocab, width, tied=tied)
            byte_count = budget["parameters"] * {"FP32": 4, "FP16/BF16": 2, "INT8": 1}[precision]
            st.json({"parameters": budget["parameters"], "ideal_weight_bytes": byte_count, "decimal_GB": byte_count/1e9, "GiB": byte_count/2**30})
            x, y = np.linspace(16000, 250000, 20), np.linspace(512, 8192, 20)
            fig = go.Figure(go.Surface(x=x, y=y, z=np.outer(y, x)*(1 if tied else 2)))
            fig.update_layout(scene=dict(xaxis_title="Vocabulary entries", yaxis_title="Embedding width", zaxis_title="Weight parameters"))
            chart(fig, "Theoretical vocabulary and embedding-width parameter surface")
            st.caption("INT8 is ideal weight storage; quantization metadata, gradients, optimizer states, activations and the rest of the model are excluded.")
        else:
            n = st.select_slider("Sequence length", [128, 256, 512, 1024, 2048, 4096], value=1024, key="length")
            head = st.slider("Head dimension", 16, 256, 64, 16, key="head")
            st.json(attention_counts(n, head))
            x, y = np.linspace(128, 4096, 25), np.linspace(16, 256, 17)
            fig = go.Figure(go.Surface(x=x, y=y, z=np.outer(y, x*x)))
            fig.update_layout(scene=dict(xaxis_title="Sequence length", yaxis_title="Head dimension", zaxis_title="n²d proxy"))
            chart(fig, "Theoretical dense attention interaction-times-width proxy")
            st.caption("Causal valid pairs do not establish allocated memory. A single cached query has n interactions; it is a different workload from a full prefill. FlashAttention can avoid storing the full matrix.")

    elif section == SECTIONS[6]:
        language = st.multiselect("Passages", list(PRESETS)[:5], default=list(PRESETS)[:5], key="languages")
        rows = [r for r in data["multilingual"] if r["language"] in language]
        if rows:
            frame = pd.DataFrame(rows)
            st.bar_chart(frame.pivot(index="language", columns="name", values="tokens"), stack=False, alt="Actual pretrained token counts for selected synthetic translations")
            table(rows, "Multilingual count, byte and code-point metrics")
            selected = st.selectbox("Inspect passage", language, key="inspect_language")
            token_blocks(compare_text("cl100k_base", PRESETS[selected], local)["pieces"])
        st.caption("Handwritten approximately equivalent translations require human translation review. Unicode code points are not grapheme clusters; whitespace fertility is not a universal multilingual metric.")

    elif section == SECTIONS[7]:
        mechanism = st.selectbox("Corruption mechanism", ["substitution", "insert_space", "break_word", "remove_space", "deletion"], key="mechanism")
        rate = st.slider("Corruption probability", 0.0, .6, .15, .05, key="rate")
        seed = st.slider("Seeded trial", 0, 11, 0, key="trial")
        text = st.text_area("Clean source baseline", PRESETS["Legal"], key="clean_text", max_chars=4000)
        noisy = corrupt(text, rate, mechanism, SEED+seed)
        st.code(noisy, language=None)
        rows = []
        for name in ALGORITHMS[:2]:
            clean, noise = compare_text(name, text, local), compare_text(name, noisy, local)
            rows.append({"encoding": name, "clean": clean["tokens"], "noisy": noise["tokens"], "inflation_pct": 100*(noise["tokens"]-clean["tokens"])/clean["tokens"] if clean["tokens"] else None})
            token_blocks(noise["pieces"])
        table(rows, "Live synthetic OCR token inflation relative to the retained source")
        measured = pd.DataFrame([r for r in data["ocr"] if r["mechanism"] == mechanism])
        chart(go.Figure([go.Box(x=measured["rate"], y=measured["inflation_pct"], name="two fixed passages, three seeded trials")]), "Measured OCR inflation distributions from the quick reference experiment")
        st.caption("Noise rates mean per-code-point opportunities; substitutions affect only mapped letters. Text changes can reduce token counts. No retrieval accuracy or legal interpretation is measured.")

    elif section == SECTIONS[8]:
        concept = st.segmented_control("RAG view", ["Chunking", "Context window"], default="Chunking", key="rag_view")
        if concept == "Chunking":
            encoding = st.selectbox("Embedding tokenizer", ALGORITHMS[:2], key="rag_encoding")
            generator = st.selectbox("Generator tokenizer", ALGORITHMS[:2], index=1, key="generator")
            strategy = st.selectbox("Chunking rule", ["characters", "tokens", "sentences"], index=1, key="strategy")
            size = st.slider("Chunk limit (strategy units)", 8, 100, 40, key="chunk_size")
            overlap = st.slider("Overlap (strategy units)", 0, min(size-1, 20), 5, key="overlap")
            doc = st.text_area("Synthetic document", RAG_DOCUMENT, key="document", max_chars=6000)
            chunks = chunk_document(doc, encoding, strategy, size, overlap)
            metrics = chunk_metrics(chunks, size, generator)
            st.json(metrics)
            generator_limit = st.selectbox("Hypothetical generator context capacity", [128, 256, 512, 8192], index=2, key="rag_generator_limit")
            reserve = st.number_input("Generator output reservation", 0, 8192, 64, key="rag_output_reserve")
            st.json(context_budget({"serialized_prompt": metrics["generator_tokens_final_prompt"]}, reserve, generator_limit))
            table(chunks, "Original character spans, actual tokens and sentence-boundary controls")
            for chunk in chunks:
                token_blocks([chunk["text"]])
            st.caption("No text is truncated. Oversized sentences fall back to token-safe spans. Overlap can begin inside a sentence; fixed characters do not enforce token caps. Evidence containment is only a known-phrase check.")
        else:
            limit = st.selectbox("Hypothetical context limit", [8192, 32768, 128000, 1000000], index=1, key="context_limit")
            components = {name: st.number_input(label, 0, 1000000, default, key=f"context_{name}")
                          for name, label, default in [("system", "System instructions", 1500), ("history", "Conversation history", 4000), ("query", "User query", 300), ("tools", "Tool definitions / outputs", 500), ("overhead", "Structural overhead", 200)]}
            retrieval = st.slider("Retrieved chunks (1,000 tokens each)", 0, 100, 14, key="retrieval_chunks")
            output = st.number_input("Reserved output tokens", 0, 1000000, 4000, key="reserved_output")
            prune = st.toggle("Remove lower-priority chunks until fit", False, key="prune")
            result = prune_context(components, [1000]*retrieval, output, limit) if prune else {"retained": [1000]*retrieval, "removed": 0, "budget": context_budget({**components, "retrieval": 1000*retrieval}, output, limit)}
            st.json(result)
            names, counts = list(components), list(components.values())
            names += ["retrieval", "reserved output"]; counts += [sum(result["retained"]), output]
            chart(go.Figure([go.Bar(name=name, x=[count], y=["context"], orientation="h") for name, count in zip(names, counts)]).update_layout(barmode="stack", xaxis_title="Hypothetical tokens"), "Allocated context components and output reservation")
            st.caption("These are already-counted hypothetical components. Tokenize the final serialized request in a real system; separately tokenized strings may not add exactly. Irreducible overflow remains visible after all chunks are removed.")

    else:
        st.warning("Hypothetical prices and projections; no provider bill, savings or answer quality is measured.")
        inp = st.number_input("Input tokens/request", 0, 1000000, 6000, key="cost_input")
        out = st.number_input("Output tokens/request", 0, 1000000, 1000, key="cost_output")
        pin = st.number_input("Input USD per million", 0.0, 1000.0, 2.0, key="cost_pin")
        pout = st.number_input("Output USD per million", 0.0, 1000.0, 8.0, key="cost_pout")
        daily = st.number_input("Daily requests", 0, 10000000, 333, key="cost_daily")
        volume = st.number_input("Monthly requests", 0, 100000000, 10000, key="cost_monthly")
        reduction = st.slider("Redundant input-context reduction (%)", 0, 80, 0, key="cost_reduction")
        reduced_input = round(inp*(1-reduction/100))
        per, before = estimated_cost(reduced_input, out, pin, pout), estimated_cost(inp, out, pin, pout)
        with st.container(horizontal=True):
            st.metric("Hypothetical USD/request", f"{per:.5f}")
            st.metric("Hypothetical daily USD", f"{per*daily:.2f}")
            st.metric("Hypothetical monthly USD", f"{per*volume:.2f}")
        st.json({"input_cost": str(estimated_cost(reduced_input, 0, pin, pout)), "output_cost": str(estimated_cost(0, out, pin, pout)), "before_monthly": str(before*volume), "projected_difference": str((before-per)*volume)})
        x, y = np.linspace(0, 32000, 25), np.linspace(0, 100000, 21)
        fig = go.Figure(go.Surface(x=x, y=y, z=np.outer(y, (x*pin+out*pout)/1e6)))
        fig.update_layout(scene=dict(xaxis_title="Input tokens/request", yaxis_title="Monthly requests", zaxis_title="Hypothetical USD/month"))
        chart(fig, "Hypothetical monthly cost surface with fixed output length")
        st.caption("Daily and monthly volumes are independent scenarios; a month is not assumed to contain a fixed number of days. Context reduction changes only input tokens, with no quality guarantee.")


if __name__ == "__main__":
    main()
