"""Measured tokenizer states and explicitly theoretical models for the visual lab."""

from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from from_scratch import SimpleBPE, merge_pair
from example import context_budget, estimated_cost, inspect_text


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "outputs/models-v2"
SEED = 272
WEIGHTS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
BPE_TEXT = " ".join(word for word, count in WEIGHTS.items() for _ in range(count))
PARALLEL = {
    "English": "The report describes how artificial intelligence helps people find information in documents.",
    "Portuguese": "O relatório descreve como a inteligência artificial ajuda as pessoas a encontrar informações em documentos.",
    "Spanish": "El informe describe cómo la inteligencia artificial ayuda a las personas a encontrar información en documentos.",
    "Chinese": "报告描述了人工智能如何帮助人们查找文档中的信息。",
    "Japanese": "この報告書は、人工知能が文書内の情報を見つけるのにどのように役立つかを説明しています。",
}
TRAINING = [
    "Machine learning is powerful. Machine learning supports document analysis.",
    "Inteligência artificial generativa. Aprendizado de máquina e documentos legais.",
    "La inteligencia artificial utiliza información de documentos.",
    "人工智能 文档 信息 学习。 日本語の文書と情報。",
    "Retrieval-augmented generation uses contextual embeddings and token budgets.",
    "unbelievable believable unaffordable affordable belief ability able tokenization tokens",
    "The legal report identifies article 125 and process ORCHID-72.",
    "O relatório jurídico identifica o artigo 125 e o processo ORCHID-72.",
    "def predict(x): return model.predict(x) SELECT id FROM records WHERE valid = 1;",
    '{"model": "demo", "temperature": 0.2, "evidence": "document"}',
    "low lower lowest newer newest widest slower intelligence learning",
    "Café ação informação. Spaces and punctuation: commas, periods; numbers 123.",
]
PRESETS = {
    **PARALLEL,
    "Python": "def predict(x):\n    return model.predict(x)",
    "JSON": '{"model": "demo", "temperature": 0.2}',
    "SQL": "SELECT article_id, title FROM documents WHERE published = 1;",
    "Legal": "Article 125 requires the ORCHID-72 process to retain the original evidence.",
    "Technical": "Retrieval-augmented generation with contextual embeddings.",
    "OCR": "Env1ronmentaI regu1at ion Art ic le 1 2 5",
    "Emojis": "Inteligência 🚀 🧠 é e\u0301 中文 日本語",
}
UNICODE_PROBES = ["A", "é", "ç", "中文", "日本語", "🚀", "🧠", "inteligência", "e\u0301"]
RAG_DOCUMENT = (
    "Overview. This synthetic policy explains document retention and token budgets.\n\n"
    "Evidence. The ORCHID-72 process requires retention for seven years. "
    "Article 125 identifies the original source as the authoritative record.\n\n"
    "Technical details. Retrieval-augmented generation uses an embedding model and a generator. "
    "Their tokenizers and input limits can differ. Preserve citations and identifiers.\n\n"
    "Operations. Remove redundant context only after checking the required evidence. "
    "This is a synthetic document, not legal advice."
)
UNIGRAM_PROBS = {"un": .20, "believable": .12, "believ": .18,
                 "able": .20, "unbeliev": .25, "unused": .05}
ALGORITHMS = ("cl100k_base", "o200k_base", "WordPiece local", "SentencePiece BPE",
              "SentencePiece Unigram", "Character BPE")


def prepare_tokenizers(directory: Path) -> dict:
    """Train small local models once, retaining their actual serialized artifacts."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import sentencepiece as spm
        from tokenizers import Tokenizer, models, trainers, normalizers, pre_tokenizers, decoders
    except ImportError as exc:
        raise RuntimeError('Install the canonical extra: python -m pip install -e ".[dev,nlp]"') from exc
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256("\n".join(TRAINING).encode()).hexdigest()
    metadata = directory / "training.json"
    if metadata.exists() and json.loads(metadata.read_text())["corpus_sha256"] != fingerprint:
        raise ValueError("local model corpus changed; select a new model directory")
    wp_path = directory / "wordpiece.json"
    if not wp_path.exists():
        wp = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        wp.normalizer = normalizers.NFKC()
        wp.pre_tokenizer = pre_tokenizers.Whitespace()
        wp.decoder = decoders.WordPiece(prefix="##", cleanup=False)
        # Seed continuation symbols in lexical order to remove randomized
        # word-iteration effects on numeric IDs used to break trainer ties.
        alphabet = sorted({c for text in TRAINING for c in wp.normalizer.normalize_str(text) if not c.isspace()})
        wp.train_from_iterator(TRAINING, trainers.WordPieceTrainer(
            vocab_size=384, min_frequency=1,
            special_tokens=["[UNK]"] + ["##" + c for c in alphabet], show_progress=False))
        vocabulary = {piece: i for i, piece in enumerate(["[UNK]"] + sorted(p for p in wp.get_vocab() if p != "[UNK]"))}
        # Remove training-only special status from continuation pieces. These
        # are newly trained synthetic models, with no pretrained weights to remap.
        wp = Tokenizer(models.WordPiece(vocab=vocabulary, unk_token="[UNK]"))
        wp.normalizer = normalizers.NFKC()
        wp.pre_tokenizer = pre_tokenizers.Whitespace()
        wp.decoder = decoders.WordPiece(prefix="##", cleanup=False)
        wp.add_special_tokens(["[UNK]"])
        wp.save(str(wp_path))
    wp = Tokenizer.from_file(str(wp_path))
    result = {"WordPiece local": wp}
    settings = {}
    for kind in ("bpe", "unigram"):
        path = directory / f"sentencepiece_{kind}.model"
        if not path.exists():
            model = io.BytesIO()
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(TRAINING), model_writer=model,
                model_type=kind, vocab_size=256, character_coverage=1.0,
                normalization_rule_name="nmt_nfkc", num_threads=1,
                shuffle_input_sentence=False, input_sentence_size=0,
                hard_vocab_limit=False, bos_id=-1, eos_id=-1, pad_id=-1,
                minloglevel=2,
            )
            path.write_bytes(model.getvalue())
        processor = spm.SentencePieceProcessor(model_file=str(path))
        result[f"SentencePiece {'BPE' if kind == 'bpe' else 'Unigram'}"] = processor
        settings[kind] = {"actual_vocabulary_size": processor.vocab_size(),
                          "normalization": "nmt_nfkc; remove_extra_whitespaces; add_dummy_prefix",
                          "requested_vocabulary_size": 256, "byte_fallback": False,
                          "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory / "training.json").write_text(json.dumps({
        "corpus_sha256": fingerprint, "sentences": len(TRAINING),
        "wordpiece_actual_vocabulary_size": wp.get_vocab_size(),
        "wordpiece_normalization": "NFKC; Whitespace pretokenizer; no lowercasing",
        "wordpiece_training": "HF WordPieceTrainer, target 384; ordered continuation-symbol seeding, canonical lexical IDs; only UNK retains special status",
        "wordpiece_sha256": hashlib.sha256(wp_path.read_bytes()).hexdigest(),
        "sentencepiece": settings,
    }, indent=2), encoding="utf-8")
    return result


def get_encoding(name):
    import tiktoken
    return tiktoken.get_encoding(name)


def compare_text(name: str, text: str, local: dict) -> dict:
    if name not in ALGORITHMS or not isinstance(text, str):
        raise ValueError("unknown tokenizer or non-string text")
    text.encode("utf-8", errors="strict")
    if name.endswith("_base"):
        data = inspect_text(get_encoding(name), text)
        ids, pieces = data["ids"], data["token_bytes_hex"]
        decoded, normalized, unknown = get_encoding(name).decode(ids), text, 0
        origin = "pretrained encoding; byte pieces shown in hex"
    elif name == "WordPiece local":
        tokenizer = local[name]
        encoded = tokenizer.encode(text)
        ids, pieces = encoded.ids, encoded.tokens
        decoded, normalized = tokenizer.decode(ids, skip_special_tokens=False), tokenizer.normalizer.normalize_str(text)
        unknown = pieces.count("[UNK]")
        origin = "trained locally on 12 synthetic sentences; NFKC + Whitespace"
    elif name.startswith("SentencePiece"):
        tokenizer = local[name]
        ids, pieces = tokenizer.encode(text, out_type=int), tokenizer.encode(text, out_type=str)
        decoded, normalized = tokenizer.decode(ids), tokenizer.normalize(text)
        unknown = ids.count(tokenizer.unk_id())
        origin = "trained locally on 12 synthetic sentences; nmt_nfkc"
    else:
        tokenizer = SimpleBPE(10).fit(BPE_TEXT)
        pieces = [piece for word in text.split() for piece in tokenizer.encode_word(word)]
        ids, decoded, normalized = None, " ".join("".join(tokenizer.encode_word(w)) for w in text.split()), text
        unknown, origin = 0, "educational character BPE; whitespace collapsed; pieces have no IDs"
    n = len(pieces)
    # Whitespace words are not defined here for scripts lacking reliable spaces.
    words = len(text.split()) if not re.search(r"[\u3400-\u9fff\u3040-\u30ff]", text) else None
    return {"name": name, "origin": origin, "text": text, "normalized": normalized,
            "decoded": decoded, "pieces": pieces, "ids": ids, "tokens": n,
            "code_points": len(text), "utf8_bytes": len(text.encode()),
            "characters_per_token": len(text) / n if n else None,
            "bytes_per_token": len(text.encode()) / n if n else None,
            "tokens_per_whitespace_word": n / words if words else None,
            "unknown": unknown, "exact_reconstruction": decoded == text}


def bpe_states(merges=10):
    model = SimpleBPE(merges).fit(BPE_TEXT)
    corpus = {tuple(w): n for w, n in WEIGHTS.items()}
    alphabet = set("".join(WEIGHTS))
    initial = sum(len(w) * n for w, n in WEIGHTS.items())
    states = []
    for step in range(len(model.history) + 1):
        counts = Counter()
        for symbols, count in corpus.items():
            for pair in zip(symbols, symbols[1:]):
                counts[pair] += count
        selected = model.history[step][0] if step < len(model.history) else None
        total = sum(len(symbols) * n for symbols, n in corpus.items())
        states.append({"step": step, "corpus": [{"pieces": list(s), "frequency": n} for s, n in corpus.items()],
                       "pairs": [(list(p), n) for p, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))],
                       "selected": list(selected) if selected else None, "vocabulary_size": len(alphabet),
                       "weighted_tokens": total, "initial_tokens": initial,
                       "compression": initial / total})
        if selected:
            alphabet.add("".join(selected))
            corpus = {merge_pair(s, selected): n for s, n in corpus.items()}
    return states


def inference_states(word, model=None):
    model = model or SimpleBPE(10).fit(BPE_TEXT)
    if set(word) - model.alphabet:
        return [{"word": word, "pieces": list(word), "error": "unsupported characters; no fallback", "selected": None, "candidates": []}]
    symbols, states = tuple(word), []
    while True:
        candidates = sorted(set(zip(symbols, symbols[1:])) & model.merge_ranks.keys(), key=model.merge_ranks.get)
        selected = candidates[0] if candidates else None
        states.append({"word": word, "pieces": list(symbols), "selected": list(selected) if selected else None,
                       "candidates": [{"pair": list(p), "rank": model.merge_ranks[p]} for p in candidates]})
        if not selected:
            return states
        symbols = merge_pair(symbols, selected)


def wordpiece_trace(word, vocabulary):
    """Greedy maximum matching; a dead end makes the entire word unknown."""
    if not isinstance(word, str) or not word or any(c.isspace() for c in word):
        raise ValueError("expected one nonempty word")
    start, pieces, trace = 0, [], []
    while start < len(word):
        choices = [(word[start:end] if start == 0 else "##" + word[start:end], end)
                   for end in range(len(word), start, -1)]
        found = next(((piece, end) for piece, end in choices if piece in vocabulary), None)
        trace.append({"start": start, "candidates": [p for p, _ in choices],
                      "selected": found[0] if found else "[UNK]", "previous": pieces.copy()})
        if found is None:
            return {"pieces": ["[UNK]"], "trace": trace}
        pieces.append(found[0])
        start = found[1]
    return {"pieces": pieces, "trace": trace}


def association_scores(joint=(100, 30), left=(1000, 40), right=(500, 50)):
    joint, left, right = (np.asarray(x, dtype=float) for x in (joint, left, right))
    if joint.shape != left.shape or joint.shape != right.shape or joint.ndim != 1 or len(joint) == 0:
        raise ValueError("frequency arrays must be matching nonempty vectors")
    if not all(np.all(np.isfinite(x)) and np.all(x > 0) for x in (joint, left, right)):
        raise ValueError("frequencies must be positive and finite")
    if np.any(joint > np.minimum(left, right)):
        raise ValueError("joint frequency cannot exceed a marginal count")
    scores = joint / (left * right)
    return {"frequency": joint.tolist(), "association": scores.tolist(),
            "bpe_winner": int(np.argmax(joint)), "association_winner": int(np.argmax(scores))}


def unigram_viterbi(text, probabilities):
    """Exact best-path decoding for manually supplied normalized piece probabilities."""
    if not text or not probabilities or any(not piece for piece in probabilities):
        raise ValueError("text and token strings must be nonempty")
    if any(not math.isfinite(p) or p <= 0 or p > 1 for p in probabilities.values()):
        raise ValueError("probabilities must be finite in (0, 1]")
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-10):
        raise ValueError("vocabulary probabilities must sum to one")
    best, previous = [-math.inf] * (len(text) + 1), [None] * (len(text) + 1)
    best[0] = 0.0
    edges, steps = [], []
    for end in range(1, len(text) + 1):
        evaluated = []
        for piece in sorted(probabilities):
            start = end - len(piece)
            if start >= 0 and text[start:end] == piece:
                score = best[start] + math.log(probabilities[piece])
                edge = {"start": start, "end": end, "piece": piece, "p": probabilities[piece],
                        "log_p": math.log(probabilities[piece]), "score": score if math.isfinite(score) else None}
                edges.append(edge)
                evaluated.append(edge)
                if score > best[end]:
                    best[end], previous[end] = score, (start, piece)
        steps.append({"end": end, "evaluated": evaluated,
                      "best": [s if math.isfinite(s) else None for s in best],
                      "predecessor": previous[end]})
    if previous[-1] is None:
        raise ValueError("text has no complete segmentation under this vocabulary")
    pieces, position = [], len(text)
    while position:
        position, piece = previous[position]
        pieces.append(piece)
    return {"text": text, "probabilities": probabilities, "pieces": pieces[::-1],
            "log_probability": best[-1], "probability": math.exp(best[-1]),
            "edges": edges, "steps": steps}


def exhaustive_paths(text, probabilities, start=0):
    """Small-example oracle; not a scalable inference algorithm."""
    if start == len(text):
        return [([], 1.0)]
    paths = []
    for piece, probability in sorted(probabilities.items()):
        if text.startswith(piece, start):
            for tail, score in exhaustive_paths(text, probabilities, start + len(piece)):
                paths.append(([piece] + tail, probability * score))
    return paths


def synthetic_embeddings(ids, width=6):
    """Stateless rows indexed by real IDs; no enormous pretrained embedding table."""
    if type(width) is not int or width < 1 or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("width must be positive and IDs nonnegative integers")
    return np.array([np.random.default_rng(np.random.SeedSequence([SEED, i])).normal(size=width) for i in ids]).reshape(len(ids), width)


def pipeline_state(text):
    import regex
    from tiktoken_ext.openai_public import cl100k_base
    encoding = get_encoding("cl100k_base")
    pretokenized = regex.findall(cl100k_base()["pat_str"], text)
    if "".join(pretokenized) != text:
        raise AssertionError("pretokenizer must preserve the identity-normalized input")
    ids = encoding.encode_ordinary(text)
    return {"text": text, "normalized": text, "pretokenized": pretokenized,
            "ids": ids, "pieces": [encoding.decode_single_token_bytes(i).hex() for i in ids],
            "embedding_shape": [encoding.n_vocab, 6],
            "selected_rows": synthetic_embeddings(ids).tolist()}


def vocabulary_experiment():
    rows = []
    held_out = ["lowest", "newer", "slower", "weirdest", "café"]
    for limit in [0, 2, 4, 6, 8, 10, 15, 25]:
        model = SimpleBPE(limit).fit(BPE_TEXT)
        for partition, words in (("train", BPE_TEXT.split()), ("held_out", held_out)):
            supported = [w for w in words if not set(w) - model.alphabet]
            total = sum(len(model.encode_word(w)) for w in supported)
            chars = sum(map(len, supported))
            rows.append({"merge_limit": limit, "merges": len(model.history), "vocabulary": len(model.vocabulary),
                         "partition": partition, "words": len(words), "supported": len(supported),
                         "rejected": len(words) - len(supported), "word_coverage": len(supported) / len(words),
                         "character_coverage": sum(c in model.alphabet for w in words for c in w) / sum(map(len, words)),
                         "tokens": total, "mean_tokens_per_supported_word": total / len(supported),
                         "compression_supported_characters_per_token": chars / total})
    return rows


def attention_counts(length, head_dimension=64):
    if type(length) is not int or length < 1 or type(head_dimension) is not int or head_dimension < 1:
        raise ValueError("length and head dimension must be positive integers")
    return {"length": length, "dense": length * length,
            "causal_valid": length * (length + 1) // 2,
            "one_cached_decode_query": length,
            "dense_proxy": length * length * head_dimension}


def corrupt(text, rate, mechanism, seed=SEED):
    if not 0 <= rate <= 1 or mechanism not in ("substitution", "insert_space", "break_word", "remove_space", "deletion"):
        raise ValueError("invalid OCR rate or mechanism")
    rng = np.random.default_rng(seed)
    replacements = {"o": "0", "O": "0", "i": "1", "l": "1", "e": "3", "a": "4"}
    out = []
    for i, char in enumerate(text):
        active = rng.random() < rate
        if active and mechanism == "substitution":
            out.append(replacements.get(char, char))
        elif active and mechanism == "insert_space":
            out.append(char + " ")
        elif active and mechanism == "break_word" and char.isalpha() and i + 1 < len(text) and text[i+1].isalpha():
            out.append(char + "-\n")
        elif active and ((mechanism == "remove_space" and char.isspace()) or mechanism == "deletion"):
            continue
        else:
            out.append(char)
    return "".join(out)


def ocr_experiment(trials=8):
    rows = []
    passages = {"technical": PRESETS["Technical"], "legal": PRESETS["Legal"]}
    for encoding in ("cl100k_base", "o200k_base"):
        enc = get_encoding(encoding)
        for passage, text in passages.items():
            baseline = len(enc.encode_ordinary(text))
            for method in ("substitution", "insert_space", "break_word", "remove_space", "deletion"):
                for rate in (0, .05, .15, .30):
                    for trial in range(trials):
                        noisy = corrupt(text, rate, method, SEED + trial)
                        count = len(enc.encode_ordinary(noisy))
                        rows.append({"encoding": encoding, "passage": passage, "mechanism": method,
                                     "rate": rate, "trial": trial, "clean_tokens": baseline, "tokens": count,
                                     "inflation_pct": 100 * (count - baseline) / baseline, "text": noisy})
    return rows


def multilingual_experiment():
    rows = []
    for language, text in PARALLEL.items():
        for name in ("cl100k_base", "o200k_base"):
            row = compare_text(name, text, {})
            rows.append({k: v for k, v in {**row, "language": language}.items()
                         if k not in ("pieces", "ids", "normalized", "decoded", "origin")})
    return rows


def chunk_document(text, encoding="cl100k_base", strategy="tokens", size=40, overlap=5):
    """Character-span chunks; token mode never independently decodes partial UTF-8."""
    if strategy not in ("characters", "tokens", "sentences") or type(size) is not int or size < 1:
        raise ValueError("invalid chunk strategy or size")
    if type(overlap) is not int or not 0 <= overlap < size:
        raise ValueError("overlap must be a nonnegative integer below size")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    enc = get_encoding(encoding)
    count = lambda value: len(enc.encode_ordinary(value))
    if strategy != "characters" and any(count(char) > size for char in text):
        raise ValueError("token limit cannot fit an individual Unicode code point")
    sentence_ends = {m.end() for m in re.finditer(r"[.!?](?:\s+|$)|\n\n", text)} | {len(text)}
    chunks, start = [], 0
    while start < len(text):
        if strategy == "characters":
            end = min(len(text), start + size)
        else:
            end = start + 1
            while end < len(text) and count(text[start:end + 1]) <= size:
                end += 1
            if strategy == "sentences":
                candidates = [p for p in sentence_ends if start < p <= end]
                if candidates:
                    end = max(candidates)
        piece = text[start:end]
        chunks.append({"start": start, "end": end, "text": piece, "tokens": count(piece),
                       "overlap_tokens": count(text[start:chunks[-1]["end"]]) if chunks and chunks[-1]["end"] > start else 0,
                       "ends_sentence": end in sentence_ends, "starts_sentence": start == 0 or start in sentence_ends})
        if end == len(text):
            break
        if strategy == "characters":
            next_start = end - overlap
        else:
            next_start = end
            while next_start > start and count(text[next_start - 1:end]) <= overlap:
                next_start -= 1
        start = max(start + 1, next_start)
    return chunks


def chunk_metrics(chunks, embedding_limit=40, generator="o200k_base"):
    lengths = [c["tokens"] for c in chunks]
    intervals = set(i for c in chunks for i in range(c["start"], c["end"]))
    gen = get_encoding(generator)
    return {"chunks": len(chunks), "min": min(lengths, default=0), "max": max(lengths, default=0),
            "mean": float(np.mean(lengths)) if lengths else 0,
            "token_limit_violations": sum(n > embedding_limit for n in lengths),
            "duplicated_code_points": sum(len(c["text"]) for c in chunks) - len(intervals),
            "overlap_tokens_reencoded": sum(c["overlap_tokens"] for c in chunks),
            "summed_chunk_tokens": sum(lengths),
            "generator_tokens_final_prompt": len(gen.encode_ordinary("System: retain cited evidence.\n" + "\n---\n".join(c["text"] for c in chunks))),
            "evidence_contained": any("retention for seven years" in c["text"] for c in chunks),
            "truncated_code_points": 0, "retrieval_evaluated": False}


def prune_context(components, retrieval_counts, output_reserve, limit):
    """Keep higher-priority chunks first; report irreducible overflow explicitly."""
    retained = list(retrieval_counts)
    while retained and not context_budget({**components, "retrieval": sum(retained)}, output_reserve, limit)["fits"]:
        retained.pop()
    return {"retained": retained, "removed": len(retrieval_counts) - len(retained),
            "budget": context_budget({**components, "retrieval": sum(retained)}, output_reserve, limit)}


def build_experiments(local, quick=False):
    inference = {w: inference_states(w) for w in ("lowest", "newer", "slower", "newest", "café")}
    unigram = unigram_viterbi("unbelievable", UNIGRAM_PROBS)
    unigram["paths"] = [{"pieces": p, "probability": score, "log_probability": math.log(score)}
                        for p, score in exhaustive_paths("unbelievable", UNIGRAM_PROBS)]
    sentences = ["Machine learning is powerful.", "Inteligência artificial generativa.",
                 "  Café   ＡＩ  é e\u0301.  "]
    sentencepiece = [compare_text(name, sentence, local) for sentence in sentences
                     for name in ("SentencePiece BPE", "SentencePiece Unigram")]
    comparison = []
    for name in ALGORITHMS:
        text = "lowest newer" if name == "Character BPE" else PRESETS["Portuguese"]
        comparison.append(compare_text(name, text, local))
    wp_vocab = {"un", "##afford", "##able", "[UNK]"}
    hf_word = "unbelievable"
    hf_parity = wordpiece_trace(hf_word, local["WordPiece local"].get_vocab())
    if hf_parity["pieces"] != local["WordPiece local"].encode(hf_word).tokens:
        raise AssertionError("educational greedy WordPiece differs from real local model")
    rag = []
    for strategy in ("characters", "tokens", "sentences"):
        chunks = chunk_document(RAG_DOCUMENT, strategy=strategy, size=40, overlap=5)
        rag.append({"strategy": strategy, "chunks": chunks, "metrics": chunk_metrics(chunks)})
    return {"seed": SEED, "pipeline": [pipeline_state(t) for t in (
                "Machine learning is powerful.", "Inteligência artificial generativa.",
                "Retrieval-augmented generation.", "unbelievable")],
            "bpe": bpe_states(), "inference": inference,
            "wordpiece": wordpiece_trace("unaffordable", wp_vocab), "hf_wordpiece": hf_parity,
            "association": association_scores(), "unigram": unigram,
            "sentencepiece": sentencepiece,
            "unicode": [compare_text("cl100k_base", text, local) for text in UNICODE_PROBES],
            "comparison": comparison, "vocabulary": vocabulary_experiment(),
            "attention": [attention_counts(n) for n in (128, 256, 512, 1024, 2048, 4096)],
            "multilingual": multilingual_experiment(), "ocr": ocr_experiment(3 if quick else 12),
            "rag": rag,
            "context": context_budget({"system": 1500, "history": 4000, "query": 300, "retrieval": 14000, "tools_overhead": 500}, 4000, 32768),
            "hypothetical_cost": str(estimated_cost(6000, 1000, 2, 8))}
