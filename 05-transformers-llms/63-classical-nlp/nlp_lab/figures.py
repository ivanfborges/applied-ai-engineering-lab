"""Bounded Matplotlib/Pillow exports; every plot is built from calculated values."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer

from . import core as c
from . import interactive as web


BLUE, ORANGE, GREEN, PURPLE, GOLD, CYAN = web.COLORS
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14,
                     "axes.labelsize": 11, "figure.facecolor": "#f5f7fb", "axes.facecolor": "white",
                     "savefig.facecolor": "#f5f7fb"})


def canvas(title: str, subtitle: str = "", quick: bool = False, size=(13, 7.5)):
    fig = plt.figure(figsize=size, dpi=80 if quick else 125)
    fig.suptitle(title, x=.04, y=.97, ha="left", weight="bold", fontsize=17)
    if subtitle: fig.text(.04, .90, subtitle, ha="left", va="top", fontsize=11, color="#405469", wrap=True)
    return fig


def save_png(fig, path: Path, quick: bool = False) -> None:
    fig.savefig(path, dpi=110 if quick else 180, bbox_inches="tight", pad_inches=.25)
    plt.close(fig)


def save_gif(path: Path, states: list, draw, quick: bool = False, duration: int = 650) -> None:
    """Store palette frames, not full RGBA arrays, and close each figure promptly."""
    frames = []
    for state in states:
        fig = draw(state)
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
        frames.append(frame)
        plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration,
                   loop=0, optimize=False, disposal=2)


def boxes(ax, tokens: list[str], active: set[int] | None = None, revealed: int | None = None,
          columns: int = 10, offsets: list[str] | None = None):
    active = active or set()
    rows = max(1, int(np.ceil(len(tokens) / columns)))
    ax.set(xlim=(0, columns), ylim=(-rows, .15)); ax.axis("off")
    for i, token in enumerate(tokens):
        x, y = i % columns, -(i // columns) - .85
        selected = i in active
        color = BLUE if selected else "#eaf2f8"
        if revealed is not None and i >= revealed: color = "#eeeeee"
        patch = FancyBboxPatch((x + .04, y), .9, .8, boxstyle="round,pad=.025", facecolor=color, edgecolor=BLUE if selected else "#c5d3df")
        ax.add_patch(patch)
        text = token.replace(" ", "·").replace("\n", "↵")
        if len(text) > 15: text = textwrap.fill(text, 12)
        ax.text(x + .49, y + .48, text, ha="center", va="center", fontsize=max(7, 11 - len(text) // 4), color="white" if selected else "#20384d")
        label = offsets[i] if offsets else str(i)
        ax.text(x + .49, y + .16, label, ha="center", va="center", fontsize=7, color="white" if selected else "#546573")


def tokenization(output: Path, quick: bool, seed: int) -> dict:
    records, unavailable = {}, {}
    for strategy in c.STRATEGIES:
        try: records[strategy] = [c.segment(text, strategy) for text in c.SENTENCES]
        except c.ResourceUnavailable as exc: unavailable[strategy] = str(exc)
    states = [(strategy, i, progress) for strategy in records for i in range(len(c.SENTENCES)) for progress in (.5, 1)]
    def draw(state):
        strategy, i, progress = state
        pieces = records[strategy][i]
        vocab = len({p["token"] for row in records[strategy] for p in row})
        fig = canvas("1 · Same text, different computational units", f"{strategy.upper()} · {len(pieces)} tokens · observed corpus vocabulary = {vocab}\n{c.SENTENCES[i]}", quick)
        ax = fig.add_axes([.04, .20, .92, .58])
        offsets = [f"[{p['start']}:{p['end']}]" for p in pieces]
        revealed = int(np.ceil(len(pieces) * progress))
        boxes(ax, [p["token"] for p in pieces], set(range(revealed)), columns=12, offsets=offsets)
        fig.text(.04, .10, f"Revealed {revealed}/{len(pieces)} boundaries. Offsets: {'UTF-8 bytes' if strategy == 'byte' else 'Unicode code points'}.\nBPE: actual local character-frequency merges; no pretrained model. Linguistic: Treebank EN / Toktok supplied PT.", fontsize=10)
        return fig
    save_gif(output / "01_tokenization.gif", states, draw, quick, 450)
    web.tokenization_html(output / "01_tokenization_explorer.html")
    model = c.trained_bpe()
    return {"unavailable": unavailable, "token_counts": {s: [len(row) for row in rows] for s, rows in records.items()},
            "bpe": {"algorithm": "character BPE", "vocabulary": model.vocabulary, "merges": model.merges,
                    "training_sentences": list(c.SENTENCES), "corpus_sha256": model.corpus_sha256,
                    "unknown_control": model.encode("🦄")}}


def colored_text(ax, text: str, colors: list[str], y: float) -> None:
    for i, (character, color) in enumerate(zip(text, colors)):
        row, col = divmod(i, 65)
        shown = "·" if character == " " else "→" if character == "\t" else character
        ax.text(.01 + col * .0145, y - row * .12, shown, color=color, fontsize=12, family="DejaVu Sans", transform=ax.transAxes)


def normalization(output: Path, quick: bool, seed: int) -> dict:
    pipeline = c.normalization_trace(c.RAW, ["NFC", "casefold", "whitespace"])
    states = [(f"pipeline {i}: {row['operation']}", pipeline[i-1]["text"], row["text"]) for i, row in enumerate(pipeline[1:], 1)]
    # Alternatives are independent so a destructive earlier operation cannot
    # hide what a later operation would have removed from the original input.
    states += [(f"independent: {op}", c.RAW, c.normalization_step(c.RAW, op)) for op in c.NORMALIZATIONS]
    def draw(state):
        operation, before, after = state
        fig = canvas("2 · Normalization chooses what information survives", f"Operation: {operation} · canonical pipeline followed by independent alternatives; no universal recipe", quick)
        ax = fig.add_axes([.04, .23, .92, .57]); ax.axis("off")
        old_colors, new_colors = [BLUE] * len(before), [BLUE] * len(after)
        for tag, a, b, x, y in SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
            if tag != "equal":
                old_colors[a:b] = [ORANGE if tag == "replace" else "#B22222"] * (b - a)
                new_colors[x:y] = [GREEN] * (y - x)
        ax.text(0, 1, f"BEFORE · Python len={len(before)}", transform=ax.transAxes, weight="bold")
        colored_text(ax, before, old_colors, .85)
        ax.text(0, .47, f"AFTER · Python len={len(after)}", transform=ax.transAxes, weight="bold")
        colored_text(ax, after, new_colors, .32)
        fig.text(.04, .13, "Blue: preserved · Orange/red: modified/removed in source · Green: replacement/inserted output", fontsize=11)
        fig.text(.04, .07, "Apple/apple and US/us can collapse; accents and punctuation may carry meaning. Stopword policy contains 'not'.", fontsize=10)
        return fig
    save_gif(output / "02_normalization_pipeline.gif", states, draw, quick, 1000)
    forms = [(form, c.normalization_step("é ① ﬁ", form)) for form in ("NFC", "NFD", "NFKC", "NFKD")]
    fig = canvas("2 · Visually similar text can have different code points", "Canonical forms compose/decompose accents; compatibility forms also fold ① and ﬁ. Length counts code points, not glyphs.", quick)
    ax = fig.add_axes([.05, .13, .9, .66]); ax.axis("off")
    for i, (form, text) in enumerate(forms):
        y = .92 - i * .24
        ax.text(0, y, f"{form}: {text!r} · len={len(text)}", fontsize=14, weight="bold", transform=ax.transAxes)
        points = "  ".join(p["codepoint"] for p in c.codepoints(text))
        names = " | ".join(p["name"] for p in c.codepoints(text) if p["character"] != " ")
        ax.text(0, y - .07, points, fontsize=11, transform=ax.transAxes)
        ax.text(0, y - .13, textwrap.fill(names, 115), fontsize=9, transform=ax.transAxes)
    save_png(fig, output / "02_unicode_normalization.png", quick)
    return {"pipeline_and_alternatives": [{"operation": op, "before": before, "after": after} for op, before, after in states],
            "unicode": [{"form": form, "text": text, "length": len(text), "points": c.codepoints(text)} for form, text in forms]}


def morphology(output: Path, quick: bool, seed: int) -> dict:
    missing = None
    try: rows = c.morphology_rows()
    except c.ResourceUnavailable as exc:
        missing = str(exc)
        rows = c.morphology_rows(include_wordnet=False)
    web.morphology_html(output / "03_stemming_vs_lemmatization.html", rows)
    methods = [m for m in ("porter", "snowball", "lemma", "portuguese") if any(m in r for r in rows)]
    states = [(method, group) for method in methods for group in range(max(1, int(np.ceil(sum(method in r for r in rows) / 14))))]
    def draw(state):
        method, group = state
        selected = [row for row in rows if method in row]
        selected = selected[group * 14:(group + 1) * 14] if len(selected) > 14 else selected
        if not selected: selected = [row for row in rows if method in row]
        targets = sorted({row[method] for row in selected})
        fig = canvas("3 · Surface forms → computed morphological representations", f"Actual NLTK {method} · POS supplied for WordNet · algorithm outputs are never hardcoded", quick)
        ax = fig.add_axes([.05, .15, .9, .64]); ax.axis("off")
        ax.set(xlim=(0, 1), ylim=(-1, max(len(selected), len(targets))))
        y_targets = {word: (i * (len(selected) - 1) / max(1, len(targets) - 1)) for i, word in enumerate(targets)}
        for i, row in enumerate(selected):
            target = row[method]
            ax.text(.02, i, f"{row['word']} [{row['pos']}]", va="center", fontsize=10)
            ax.plot([.30, .73], [i, y_targets[target]], color=BLUE, alpha=.4)
        for target, y in y_targets.items(): ax.text(.76, y, target, va="center", fontsize=11, color=GREEN, weight="bold")
        all_rows = [row for row in rows if method in row]
        before, after = len({r["word"] for r in all_rows}), len({r[method] for r in all_rows})
        fig.text(.05, .07, f"Full set: {before} distinct surface forms → {after} outputs; reduction={100*(1-after/before):.1f}%.\nConflation is not accuracy. Inspect irregulars, nonwords and ambiguous 'saw'. Portuguese uses Portuguese Snowball only.", fontsize=10)
        return fig
    save_gif(output / "03_morphological_transformations.gif", states, draw, quick, 1500)
    return {"rows": rows, "wordnet_unavailable": missing, "pos_source": "explicit teaching annotations", "wordnet_version": c.wordnet_resource().get_version() if missing is None else None}


def ngrams(output: Path, quick: bool, seed: int) -> dict:
    tokens = c.tokenize("machine learning models learn useful patterns")
    states = [(tokens, n, i, "word") for n in (1, 2, 3, 4) for i in range(len(tokens) - n + 1)]
    states += [(["a", "a", "a"], 2, 0, "repeated tokens"), ([], 2, -1, "empty sequence"),
               (["short"], 3, -1, "n > m"), (list("model"), 3, 0, "character")]
    def draw(state):
        sequence, n, i, unit = state
        grams = c.generate_ngrams(sequence, n)
        fig = canvas("4 · A sliding window preserves local order", f"{unit} · m={len(sequence)} units · n={n} · max(0, m−n+1)={len(grams)} occurrences", quick)
        boxes(fig.add_axes([.04, .52, .92, .22]), sequence, set(range(i, i + n)) if i >= 0 else set(), columns=6)
        current = grams[i] if i >= 0 else "No valid window"
        fig.text(.06, .42, f"Active window: {current}", fontsize=14, weight="bold")
        collection = grams[:i + 1] if i >= 0 else []
        fig.text(.06, .34, "Extracted collection:\n" + "\n".join(f"{j + 1}. {' | '.join(g)}" for j, g in enumerate(collection)), fontsize=12, va="top")
        fig.text(.06, .06, f"Extracted {len(collection)} of {len(grams)} · m = sequence length; n = window size. Repeated windows remain occurrences.", fontsize=10)
        return fig
    save_gif(output / "04_ngram_sliding_window.gif", states, draw, quick, 700)
    web.ngram_html(output / "04_ngram_explorer.html")
    return {"tokens": tokens, "counts": {str(n): len(c.generate_ngrams(tokens, n)) for n in (1, 2, 3, 4)}}


def vocabulary(output: Path, quick: bool, seed: int) -> dict:
    docs = c.synthetic_corpus()
    computed = [c.feature_stats(docs, n) for n in (1, 2, 3)]
    stats = [item[0] for item in computed]
    base = stats[0]["features"]
    observed_fixed = [len({g for doc in docs for g in c.generate_ngrams(c.tokenize(doc), n)}) for n in (1, 2, 3)]
    fig = canvas("5 · Observed feature growth is not the combinatorial bound", "45 synthetic documents; min_df=1, no feature cap; feature counts are fitted on the same corpus.", quick)
    left, right = fig.add_axes([.07, .19, .38, .57]), fig.add_axes([.57, .19, .37, .57])
    left.bar(["1", "1–2", "1–3"], [s["features"] for s in stats], color=[BLUE, GREEN, ORANGE])
    left.set(xlabel="Included word orders", ylabel="Observed vocabulary size")
    for i, s in enumerate(stats): left.text(i, s["features"], str(s["features"]), ha="center", va="bottom")
    right.semilogy([1, 2, 3], [base ** n for n in (1, 2, 3)], "o--", label=f"Fixed-order bound: K^n, K={base}", color=ORANGE)
    right.semilogy([1, 2, 3], observed_fixed, "o-", label="Observed fixed-order unique n-grams", color=BLUE)
    right.set(xlabel="Fixed order n", ylabel="Distinct features (log scale)", xticks=[1, 2, 3]); right.legend(fontsize=9)
    fig.text(.05, .07, "Left: cumulative ranges. Right: fixed orders. K^n is a theoretical maximum; it does not predict observed counts or memory.", fontsize=10)
    save_png(fig, output / "05_vocabulary_growth.png", quick)
    fig = canvas("5 · More columns can mean a sparser matrix", "Binary occupancy, not learned relevance. Memory describes count arrays only; CSR storage excludes Python object overhead.", quick)
    for i, (stat, matrix, _) in enumerate(computed):
        ax = fig.add_axes([.055 + .315 * i, .22, .27, .53])
        ax.imshow((matrix.toarray() > 0).astype(int), aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"1–{i+1}: {stat['shape']}\ndensity={stat['density']:.3f}; sparse={stat['sparsity']:.3f}", fontsize=11)
        ax.set(xlabel="Feature index", ylabel="Document ID")
        fig.text(.055 + .315 * i, .12, f"nnz={stat['nnz']}\nDense={stat['dense_bytes']:,} B\nCSR={stat['sparse_bytes']:,} B", fontsize=10)
    save_png(fig, output / "05_feature_sparsity.png", quick)
    web.feature_html(output / "05_feature_space_explorer.html")
    return {"corpus": docs, "stats": stats, "base_vocabulary": base, "observed_fixed_order": observed_fixed}


def matrix_axis(ax, matrix, labels, row_labels, title: str, vmax=None):
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value), ha="center", va="center", fontsize=9,
                    color="white" if value > (vmax or max(1, matrix.max())) * .6 else "#142536")
    return image


def dtm(output: Path, quick: bool, seed: int) -> dict:
    docs = ["Machine learning models", "Machine learning learning", "Deep learning models"]
    states = [(n, stage) for n in (1, 2) for stage in range(6)]
    def draw(state):
        n, stage = state
        vocabulary, matrix = c.document_term_matrix(docs, n)
        labels = [" ".join(g) for g in vocabulary]
        stages = ("Raw documents", "Tokenization", "Vocabulary extraction", "Feature index assignment", "Token counting", "Completed count matrix")
        fig = canvas("6 · Raw text → fixed columns → counts", f"Word order n={n} · Stage {stage+1}/6: {stages[stage]}", quick)
        left = fig.add_axes([.04, .2, .42, .57]); left.axis("off")
        text = "\n\n".join(f"D{i}: {text}" for i, text in enumerate(docs)) if stage == 0 else "\n\n".join(f"D{i}: {c.tokenize(text)}" for i, text in enumerate(docs))
        left.text(0, 1, textwrap.fill(text, 48, replace_whitespace=False), va="top", fontsize=11)
        if stage >= 2: left.text(0, .35, "Vocabulary / column index:\n" + "\n".join(f"{i}: {term}" for i, term in enumerate(labels)), va="top", fontsize=11)
        if stage >= 3:
            shown = np.zeros_like(matrix) if stage == 3 else matrix * np.array([1, 0, 0])[:, None] if stage == 4 else matrix
            ax = fig.add_axes([.53, .30, .42, .44])
            matrix_axis(ax, shown, labels, ["D0", "D1", "D2"], "Rows = documents; columns = fixed features", vmax=2)
        fig.text(.04, .07, "Counts preserve repetitions, not global order. Vocabulary is fitted once; new documents reuse column meanings.", fontsize=10)
        return fig
    save_gif(output / "06_document_term_matrix.gif", states, draw, quick, 1000)
    contrasts = ["dog bites man", "man bites dog"]
    fig = canvas("6 · Identical unigram counts can hide different events", "D0: dog bites man · D1: man bites dog. Same analyzer and vocabulary ordering; different events.", quick)
    for i, n in enumerate((1, 2)):
        vocab, matrix = c.document_term_matrix(contrasts, n)
        ax = fig.add_axes([.08 + .47 * i, .31, .37, .4])
        image = matrix_axis(ax, matrix, [" ".join(g) for g in vocab], ["D0", "D1"], f"n={n}", vmax=1)
        fig.colorbar(image, ax=ax, fraction=.05)
    fig.text(.07, .1, "Unigrams: equal vectors. Bigrams: different adjacency features. Neither represents full compositional semantics.", fontsize=10)
    save_png(fig, output / "06_bag_of_words_comparison.png", quick)
    checks = []
    for n in (1, 2):
        vocab, matrix = c.document_term_matrix(docs, n)
        reference = CountVectorizer(tokenizer=c.tokenize, token_pattern=None, lowercase=False, ngram_range=(n, n)).fit_transform(docs)
        checks.append(bool(np.array_equal(matrix, reference.toarray())))
    return {"documents": docs, "count_parity": checks}


def tfidf(output: Path, quick: bool, seed: int) -> dict:
    docs = ["document license license approved", "document license not approved", "document inspection pollution", "document payment bank"]
    vocabulary, counts = c.document_term_matrix(docs)
    df, idf, unnormalized = c.tfidf_from_counts(counts, l2=False)
    _, _, normalized = c.tfidf_from_counts(counts)
    fig = canvas("7 · Common terms receive less IDF weight", "Same feature ordering throughout. Each panel has its own labeled numeric color scale; heights are not directly comparable.", quick, size=(15, 8))
    labels = [g[0] for g in vocabulary]
    for i, (matrix, title, scale) in enumerate(((counts, "Raw TF counts", "Count"), (unnormalized, "TF × smoothed IDF", "Unnormalized weight"), (normalized, "L2-normalized TF-IDF", "Normalized weight"))):
        ax = fig.add_axes([.045 + .32 * i, .32, .27, .43])
        image = matrix_axis(ax, matrix, labels, ["D0", "D1", "D2", "D3"], title)
        bar = fig.colorbar(image, ax=ax, fraction=.046, pad=.04); bar.set_label(scale, fontsize=9)
    fig.text(.05, .10, "DF: " + ", ".join(f"{term}={value}" for term, value in zip(labels, df)) + "\nIDF: " + ", ".join(f"{term}={value:.2f}" for term, value in zip(labels, idf)), fontsize=10)
    save_png(fig, output / "07_tfidf_heatmaps.png", quick)
    fig = canvas("7 · IDF decreases with document frequency", "D=100 indexed documents; smoothed IDF = ln((1+D)/(1+df)) + 1. df is an integer from 1 to D.", quick)
    ax = fig.add_axes([.09, .20, .83, .56])
    freq = np.arange(1, 101)
    ax.plot(freq, c.tfidf_weight(100, freq, 1), color=BLUE, linewidth=3)
    ax.set(xlabel="Document frequency df", ylabel="Smoothed IDF"); ax.grid(alpha=.2)
    ax.annotate("Present everywhere: IDF=1 (not zero)", xy=(100, 1), xytext=(45, 2.5), arrowprops={"arrowstyle": "->"}, fontsize=11)
    save_png(fig, output / "07_idf_decay.png", quick)
    web.tfidf_html(output / "07_tfidf_surface_3d.html")
    return {"documents": docs, "features": labels, "df": df.tolist(), "idf": idf.tolist(),
            "counts": counts.tolist(), "unnormalized": unnormalized.tolist(), "normalized": normalized.tolist()}


def vectors(output: Path, quick: bool, seed: int) -> dict:
    result = c.vector_space("word", c.SPACE_QUERIES[0], seed)
    fig = canvas("8 · Authoritative pairwise cosine lives in the original space", "15 synthetic documents, five topic groups. Color = sparse unigram TF-IDF cosine; this is not the SVD distance matrix.", quick)
    ax = fig.add_axes([.15, .18, .7, .61])
    image = ax.imshow(result["pairwise"], cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(15), [f"D{i}" for i in range(15)], fontsize=8)
    ax.set_yticks(range(15), [f"D{i}" for i in range(15)], fontsize=8)
    ax.set(xlabel="Document ID", ylabel="Document ID")
    for boundary in (2.5, 5.5, 8.5, 11.5):
        ax.axhline(boundary, color=ORANGE, linewidth=1); ax.axvline(boundary, color=ORANGE, linewidth=1)
    fig.colorbar(image, ax=ax, label="Original-space cosine")
    save_png(fig, output / "08_cosine_similarity_heatmap.png", quick)
    web.space_html(output / "08_document_vector_space_3d.html", seed)
    return {"preprocessing": "shared NFC/casefold/whitespace normalization for words and characters",
            "representations": {rep: {"variance": (r := c.vector_space(rep, c.SPACE_QUERIES[0], seed))["variance"], "features": r["features"], "ranking": r["ranking"]} for rep in ("word", "word12", "char")}}


def cosine_geometry(output: Path, quick: bool, seed: int) -> dict:
    states = [(angle, magnitude) for magnitude in (1, 2) for angle in np.linspace(0, 180, 13)] + [(0, 0)]
    def draw(state):
        angle, magnitude = state
        a, b = np.array([1., 0.]), magnitude * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        value = c.cosine(a, b)
        fig = canvas("9 · Cosine compares direction, not vector magnitude", "Generic signed coordinates. Negative similarity is possible here; ordinary nonnegative TF-IDF cannot have negative cosine.", quick)
        ax = fig.add_axes([.1, .16, .5, .62]); ax.set_aspect("equal")
        for vector, color, label in ((a, BLUE, "a"), (b, ORANGE, "b")):
            ax.annotate("", xy=vector, xytext=(0, 0), arrowprops={"arrowstyle": "->", "lw": 4, "color": color})
            ax.text(vector[0] + .05, vector[1] + .08, label, color=color, fontsize=15)
        ax.set(xlim=(-2.5, 2.5), ylim=(-.5, 2.5), xlabel="x", ylabel="y"); ax.grid(alpha=.2)
        text = f"Angle = {angle:.0f}°\n||a|| = 1\n||b|| = {magnitude}\ndot(a,b) = {np.dot(a,b):.3f}\ncosine = {'undefined' if value is None else f'{value:.3f}'}"
        fig.text(.67, .68, text, va="top", fontsize=15)
        fig.text(.67, .29, "cos(a,b) =\ndot(a,b) / (||a|| ||b||)\n\nZero vector: undefined angle.\nLibrary zero scores are a convention.", fontsize=11)
        return fig
    save_gif(output / "09_cosine_similarity.gif", states, draw, quick, 450)
    web.geometry_html(output / "09_vector_geometry_3d.html")
    return {"same_direction": c.cosine(np.array([1, 0]), np.array([3, 0])), "orthogonal": c.cosine(np.array([1, 0]), np.array([0, 1])), "opposite": c.cosine(np.array([1, 0]), np.array([-1, 0])), "zero": None}


def language_model(output: Path, quick: bool, seed: int) -> dict:
    sentences = [c.tokenize(s) for s in c.LM_SENTENCES]
    model = c.NGramLM(sentences, 2)
    steps = model.generate(seed, .1, 12)
    def draw(step):
        fig = canvas("10 · Next-token generation is a sequence of conditional draws", f"Bigram Counter model · seed={seed} · add-alpha=0.1 · context={step['context']}", quick)
        ax = fig.add_axes([.08, .28, .85, .48])
        colors = [ORANGE if w == step["selected"] else BLUE for w in model.vocabulary]
        ax.bar(model.vocabulary, step["probabilities"], color=colors)
        ax.tick_params(axis="x", labelrotation=35, labelsize=9)
        ax.set(ylabel="P(next token | context)", ylim=(0, 1), title=f"Selected: {step['selected']} · sum={sum(step['probabilities']):.6f}")
        fig.text(.07, .12, "Generated: " + " ".join(step["sequence"]), fontsize=13)
        fig.text(.07, .05, "<s> is context only; </s> ends a sentence. Generation stops at EOS or 12 tokens. Smoothing adds mass, not understanding.", fontsize=10)
        return fig
    save_gif(output / "10_language_generation.gif", steps, draw, quick, 1100)
    web.transition_html(output / "10_ngram_transition_graph.html")
    fig = canvas("10 · Smoothing assigns mass to unseen events", "Same prediction vocabulary, observed context ('was',). An unseen unsmoothed context has an undefined distribution, not an all-zero PMF.", quick)
    ax = fig.add_axes([.08, .27, .86, .48]); x = np.arange(len(model.vocabulary))
    for shift, alpha, color in ((-.2, 0, BLUE), (.2, .5, ORANGE)):
        ax.bar(x + shift, model.distribution(("was",), alpha), .4, label=f"alpha={alpha}", color=color)
    ax.set_xticks(x, model.vocabulary, rotation=35, ha="right"); ax.set(ylabel="Conditional probability"); ax.legend()
    fig.text(.07, .08, "Every valid PMF sums to 1 over tokens + EOS. Trigram and unigram models use the same boundary and smoothing rules.", fontsize=10)
    save_png(fig, output / "10_smoothing_comparison.png", quick)
    return {"corpus": list(c.LM_SENTENCES), "vocabulary": model.vocabulary, "seed": seed, "alpha": .1,
            "generation": steps, "orders": {str(order): {"vocabulary_size": len((m := c.NGramLM(sentences, order)).vocabulary),
                        "max_normalization_error": max(abs(m.distribution(context, .1).sum() - 1) for context in m.counts)} for order in (1, 2, 3)}}


def retrieval(output: Path, quick: bool, seed: int) -> dict:
    runs, unavailable = {}, {}
    for name, stop, morph, n in c.CONFIGS:
        try: runs[name] = c.retrieval_run(stop, morph, n)
        except c.ResourceUnavailable as exc: unavailable[name] = str(exc)
    states = [(name, i) for i in range(len(c.CONTRAST_QUERIES)) for name in runs]
    def draw(state):
        name, i = state
        result = runs[name]["results"][i]
        ranking = sorted(range(6), key=lambda d: (-result["scores"][d], d))
        scores = [result["scores"][d] for d in ranking]
        fig = canvas("11 · Preprocessing changes evidence and retrieval order", f"{name} · query: {result['query']} · relevant IDs fixed in advance: {result['relevant']}", quick)
        ax = fig.add_axes([.40, .24, .53, .53])
        colors = [GREEN if d in result["relevant"] else BLUE for d in ranking]
        ax.barh(range(6), scores, color=colors)
        ax.set_yticks(range(6), [f"D{d}: {c.CONTRAST_DOCS[d]}" for d in ranking], fontsize=9)
        ax.invert_yaxis(); ax.set(xlim=(0, 1.05), xlabel="Original-space cosine (zero bars are not matches)")
        for row, value in enumerate(scores): ax.text(min(1, value) + .01, row, f"{value:.3f}", va="center", fontsize=9)
        m = result["metrics"]
        fig.text(.06, .1, f"Green = labeled relevant; blue = other. Recall@3={m['recall']:.2f}; RR={m['rr']:.2f}; nDCG@3={m['ndcg']:.2f}.\nScores are computed, judgments are hand-specified synthetic fixtures. Positive-score ties use document ID.", fontsize=10)
        return fig
    save_gif(output / "11_preprocessing_ranking_comparison.gif", states, draw, quick, 750)
    fig = canvas("11 · Feature changes need judged-query evaluation", "Seven constructed queries, six contrasting documents, k=3. Metrics use positive-score rankings; MRR is untruncated.", quick, size=(15, 8))
    ax = fig.add_axes([.07, .34, .89, .43]); names = list(runs); x = np.arange(len(names))
    for j, (key, label, color) in enumerate((("recall", "Recall@3", BLUE), ("rr", "MRR", GREEN), ("ndcg", "nDCG@3", ORANGE))):
        ax.bar(x + (j - 1) * .25, [runs[name]["mean_metrics"][key] for name in names], .25, color=color, label=label)
    ax.set_xticks(x, names, rotation=20, ha="right", fontsize=10); ax.set(ylim=(0, 1.1), ylabel="Mean across seven queries"); ax.legend(ncol=3)
    fig.text(.06, .08, "This is a hand-authored teaching challenge, not an unbiased benchmark. Examine individual failures in experiments.json.\nWordNet uses supplied small-vocabulary POS; unlisted tokens default to noun. No automatic POS accuracy is claimed.", fontsize=10)
    save_png(fig, output / "11_retrieval_metrics.png", quick)
    return {"runs": runs, "unavailable": unavailable, "judgments": list(c.CONTRAST_QUERIES)}


def ocr(output: Path, quick: bool, seed: int) -> dict:
    report = c.ocr_experiment(seed, trials=4 if quick else 20)
    report["preprocessing"] = "shared NFC/casefold/whitespace normalization for words and characters"
    fig = canvas("12 · Character overlap can survive some synthetic OCR edits", f"Clean fixed index · six queries · {report['trials']} paired trials per rate · seed={seed} · band = ±1 trial SD, not a confidence interval", quick)
    ax = fig.add_axes([.08, .20, .83, .56])
    for j, rep in enumerate(c.REPRESENTATIONS):
        means, deviations = [], []
        for rate in report["rates"]:
            values = [r["rr"] for r in report["rows"] if r["representation"] == rep and r["rate"] == rate]
            means.append(np.mean(values)); deviations.append(np.std(values, ddof=1) if len(values) > 1 else 0)
        means, deviations = np.array(means), np.array(deviations)
        ax.plot(report["rates"], means, "o-", label=rep, color=web.COLORS[j], linewidth=2)
        ax.fill_between(report["rates"], np.maximum(0, means - deviations), np.minimum(1, means + deviations), color=web.COLORS[j], alpha=.12)
    ax.set(xlabel="Probability of selecting an original alphanumeric character for editing", ylabel="MRR over six queries", ylim=(0, 1.06)); ax.legend(ncol=4); ax.grid(alpha=.2)
    fig.text(.06, .07, "Edits mix substitution, deletion, insertion, visual confusion and accent corruption. Not a calibrated OCR noise model.\nShared corrupted queries support paired comparisons; trials are not independent real-world documents.", fontsize=10)
    save_png(fig, output / "12_ocr_robustness.png", quick)
    web.ocr_html(output / "12_ocr_corruption_explorer.html", report)
    return report


def rag(output: Path, quick: bool, seed: int) -> dict:
    fig = canvas("13 · Classical NLP and model-native representations can coexist", "Conceptual architecture only: no dense embedding service, BM25 engine or LLM is executed. TF-IDF is the implemented lexical example.", quick, size=(15, 9))
    ax = fig.add_axes([.025, .13, .95, .66]); ax.axis("off"); ax.set(xlim=(0, 1), ylim=(0, 1))
    nodes = {
        "source": (.07, .58, "Original source\ndocuments"), "ocr": (.24, .58, "Text extraction\nand OCR"),
        "preserve": (.44, .92, "Preserve original\ncontent + offsets"),
        "lexical": (.44, .70, "Lexical analyzer\nversioned policies"), "sparse": (.65, .70, "Sparse index\nTF-IDF or BM25"),
        "native": (.44, .27, "Model-native\ntokenization"), "dense": (.65, .27, "Dense embeddings\nvector retrieval"),
        "fusion": (.83, .48, "Rank fusion\noptional reranking"), "context": (.83, .07, "Context assembly\n→ LLM generation"),
    }
    for source, target in (("source", "ocr"), ("ocr", "preserve"), ("ocr", "lexical"), ("ocr", "native"),
                           ("lexical", "sparse"), ("native", "dense"), ("sparse", "fusion"), ("dense", "fusion"), ("fusion", "context")):
        a, b = nodes[source], nodes[target]
        ax.add_patch(FancyArrowPatch(a[:2], b[:2], arrowstyle="->", mutation_scale=17, color="#526679", linewidth=2, shrinkA=35, shrinkB=35))
    for key, (x, y, label) in nodes.items():
        color = "#e4f2eb" if key in ("lexical", "sparse", "preserve") else "#eaf2f8"
        ax.text(x, y, label, ha="center", va="center", fontsize=10, bbox={"boxstyle": "round,pad=.6", "facecolor": color, "edgecolor": BLUE})
    fig.text(.05, .08, "Lexical branch: evaluate normalization, language-specific stems and character n-grams for OCR.\nDense branch: preserve meaningful natural language; aggressive stripping can violate the native input distribution.\nFusion combines ranks from real retrievers when implemented. No fabricated dense ranked lists are shown here.", fontsize=11)
    save_png(fig, output / "13_hybrid_rag_architecture.png", quick)
    return {"status": "illustrative architecture", "dense_retrieval_executed": False, "rank_fusion_executed": False}


GENERATORS = {"tokenization": tokenization, "normalization": normalization, "morphology": morphology,
              "ngrams": ngrams, "vocabulary": vocabulary, "dtm": dtm, "tfidf": tfidf, "vectors": vectors,
              "cosine": cosine_geometry, "language_model": language_model, "retrieval": retrieval, "ocr": ocr, "rag": rag}
