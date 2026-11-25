"""CPU raster rendering of actual traces, measurements and theoretical proxies."""

from __future__ import annotations

import hashlib
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

from visual_core import get_encoding, synthetic_embeddings, PRESETS

COLORS = ["#2878a1", "#cc6a33", "#37856b", "#9765a3", "#b18c21", "#b85472"]
INK, BG, MUTED = "#182c3b", "#f7f9fc", "#52687a"


def configure():
    names = ["DejaVu Sans"]
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/seguiemj.ttf")):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            names.append(font_manager.FontProperties(fname=str(path)).get_name())
    plt.rcParams.update({"font.family": names, "font.sans-serif": names,
                         "font.monospace": ["DejaVu Sans Mono", *names[1:]],
                         "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": BG, "axes.facecolor": BG,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})


def color(piece):
    return COLORS[int(hashlib.sha256(str(piece).encode()).hexdigest()[:8], 16) % len(COLORS)]


def title(fig, heading, subtitle):
    fig.suptitle(heading, x=.055, y=.965, ha="left", weight="bold", fontsize=20)
    fig.text(.055, .907, subtitle, fontsize=10, color=MUTED)
    fig.text(.055, .025, "DAY 72  •  CPU educational lab  •  Synthetic data  •  Author review required", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=.075, right=.96, top=.84, bottom=.12, hspace=.65, wspace=.36)


def blocks(ax, pieces, y=.75, selected=None, fontsize=12, height=.16):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    x = .01
    for index, piece in enumerate(pieces):
        label = str(piece).replace(" ", "·").replace("\n", "↵").replace("\t", "⇥")
        if len(label) > 26:
            label = label[:23] + "…"
        width = min(.68, max(.09, len(label) * .016 + .04))
        if x + width > .99:
            x, y = .01, y - .25
        if y < .02:
            ax.text(.01, .01, "More pieces in the full JSON / interactive table", fontsize=9)
            break
        hit = selected and index + 1 < len(pieces) and list(pieces[index:index+2]) == list(selected)
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.008",
                                   facecolor=color(piece), edgecolor="#f4b942" if hit else "white", linewidth=3 if hit else 1))
        ax.text(x + width / 2, y + height/2, label, ha="center", va="center", color="white", fontsize=fontsize)
        x += width + .014


def pipeline(fig, data, frame=6):
    record = data["pipeline"][(frame // 7) % len(data["pipeline"])]
    text = record["text"]
    frame %= 7
    enc = get_encoding("cl100k_base")
    ids = enc.encode_ordinary(text)
    pieces = [enc.decode_single_token_bytes(i).decode("utf-8", errors="backslashreplace") for i in ids]
    embeddings = synthetic_embeddings(ids)
    title(fig, "01 · From text to Transformer input", "Actual cl100k_base IDs and regex pretokenization; identity normalization; synthetic embeddings")
    grid = fig.add_gridspec(4, 2, width_ratios=[1.35, 1])
    stages = ["Raw text", "Normalization: identity", "Actual regex pretokenization",
              "Actual byte-BPE pieces", "Actual token IDs", "Synthetic embedding lookup", "Transformer input"]
    ax = fig.add_subplot(grid[:, 0]); ax.axis("off")
    details = [repr(text), repr(text), repr(record["pretokenized"]), " · ".join(pieces), str(ids),
               f"E[IDs] → {len(ids)} × 6", f"Token rows + position signal: shape ({len(ids)}, 6)"]
    for i, (stage, detail) in enumerate(zip(stages, details)):
        y = .94 - i * .135
        ax.text(0, y, f"{i + 1}. {stage}", weight="bold", color=COLORS[0] if i <= frame else "#aab6c0")
        if i <= frame:
            ax.text(.02, y - .052, detail, fontsize=10, wrap=True)
    ax = fig.add_subplot(grid[0, 1]); blocks(ax, pieces, y=.32, height=.5, fontsize=11)
    ax.set_title(f"Sequence length = {len(ids)}")
    ax = fig.add_subplot(grid[1:, 1])
    im = ax.imshow(embeddings, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set(yticks=range(len(ids)), yticklabels=[f"ID {i}" for i in ids], xlabel="Embedding feature (6 synthetic dimensions)")
    ax.set_title(f"Selected rows only · conceptual E: {enc.n_vocab} × 6")
    fig.colorbar(im, ax=ax, shrink=.7, label="Synthetic weight")


def bpe(fig, data, frame=20):
    states = data["bpe"]
    step = min(frame // 2, len(states) - 1)
    state = states[step]
    title(fig, "02 · BPE learns an ordered merge table", f"State {step}/{len(states)-1} · " + ("select pair → merge all non-overlapping occurrences" if frame % 2 == 0 else "corpus after the preceding merge"))
    grid = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(grid[0, 0]); ax.axis("off")
    ax.set_title(f"Corpus · vocabulary {state['vocabulary_size']} · selected {state['selected']}", loc="left")
    for j, row in enumerate(state["corpus"]):
        ax.text(0, .83 - j*.24, f"× {row['frequency']}", fontsize=10)
        inset = ax.inset_axes([.09, .74 - j*.24, .90, .20])
        blocks(inset, row["pieces"], selected=state["selected"], y=.12, height=.72, fontsize=11)
    ax = fig.add_subplot(grid[0, 1])
    pairs = state["pairs"][:6]
    if pairs:
        ax.barh([" + ".join(p) for p, _ in pairs][::-1], [n for _, n in pairs][::-1], color=COLORS[0])
    ax.set(xlabel="Weighted adjacent-pair count", title="Candidate pair frequencies")
    ax = fig.add_subplot(grid[1, 0])
    history = states[:step+1]
    ax.plot([s["step"] for s in history], [s["vocabulary_size"] for s in history], "o-", color=COLORS[2])
    ax.set(xlabel="Merges applied", ylabel="Vocabulary entries", xlim=(0, 10), ylim=(8, 22))
    ax = fig.add_subplot(grid[1, 1])
    ax.plot([s["step"] for s in history], [s["weighted_tokens"] for s in history], "o-", color=COLORS[1])
    ax.set(xlabel="Merges applied", ylabel="Weighted corpus tokens", xlim=(0, 10), ylim=(0, states[0]["initial_tokens"] + 5))
    ax.set_title(f"Compression: {state['compression']:.2f} code points / token")


def inference(fig, data, frame=0):
    all_frames = [s for trace in data["inference"].values() for s in trace]
    state = all_frames[min(frame, len(all_frames)-1)]
    title(fig, "03 · Inference applies ranks, not fresh frequencies", "Fixed training table; supported novel words plus an unsupported-character control")
    ax = fig.add_subplot(211); blocks(ax, state["pieces"], selected=state["selected"])
    ax.set_title(f"Word: {state['word']} · selected learned pair: {state['selected']}", loc="left")
    ax = fig.add_subplot(212); ax.axis("off")
    candidates = "\n".join(f"rank {c['rank']}: {c['pair']}" for c in state["candidates"])
    ax.text(.02, .95, candidates or state.get("error", "No ranked pair remains: final segmentation"), va="top", fontsize=14)
    ax.text(.55, .9, "No corpus pair counts are recomputed.\nRepeated pairs merge left to right.\nJoining supported pieces reconstructs the word.\nCharacter coverage does not imply byte fallback.", va="top", linespacing=1.8)


def association(fig, data):
    score = data["association"]
    title(fig, "04 · Frequency and relative association can disagree", "Synthetic counts; association rule is an educational reconstruction, not the original unpublished WordPiece trainer")
    axes = fig.subplots(1, 3)
    axes[0].bar(["Pair A", "Pair B"], score["frequency"], color=COLORS[:2]); axes[0].set(title="BPE-style frequency", ylabel="Joint count")
    axes[1].bar(["Pair A", "Pair B"], score["association"], color=COLORS[:2]); axes[1].set(title="WordPiece-style association", ylabel="Joint / (left × right)")
    matrix = [[100, 0], [0, 30]]
    axes[2].imshow(matrix, cmap="Blues")
    axes[2].set(xticks=[0, 1], xticklabels=["b_A", "b_B"], yticks=[0, 1], yticklabels=["a_A", "a_B"], title="Observed candidate adjacency")
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(matrix[i][j]), ha="center", va="center", color="white" if matrix[i][j] > 50 else INK)


def wordpiece(fig, data, frame=0):
    trace = data["wordpiece"]["trace"]
    state = trace[min(frame, len(trace)-1)]
    title(fig, "04 · WordPiece uses greedy longest matching", "Fixed illustrative vocabulary: un, ##afford, ##able; real locally trained HF model checked separately")
    ax = fig.add_subplot(211); blocks(ax, state["previous"] + [state["selected"]])
    ax.set_title(f"unaffordable · search starts at character {state['start']}", loc="left")
    ax = fig.add_subplot(212); ax.axis("off")
    ax.text(.02, .95, "Candidates, longest first:\n" + "\n".join(state["candidates"]), va="top", fontsize=11)
    ax.text(.56, .92, "Select the first vocabulary match.\nContinue with ##-prefixed pieces.\nA failed suffix makes the entire word [UNK].\nNo learned BPE rank table is applied.", va="top", linespacing=1.8)


def unigram(fig, data, frame=13):
    result = data["unigram"]; n = len(result["text"])
    end = min(frame, n)
    title(fig, "05 · Unigram chooses a best path through a lattice", "Manually assigned probabilities sum to one; dynamic programming scores are computed, not learned")
    ax = fig.add_subplot(211); ax.set(xlim=(-.5, n+.5), ylim=(-.25, 2.1)); ax.axis("off")
    path = set(); start = 0
    for piece in result["pieces"]:
        path.add((start, start + len(piece))); start += len(piece)
    for edge in result["edges"]:
        a, b = edge["start"], edge["end"]
        chosen = (a, b) in path and frame >= n
        tint = COLORS[2] if chosen else COLORS[0] if b <= end else "#bcc8d2"
        lift = .4 + .09*(b-a)
        ax.annotate("", (b, .1), (a, .1), arrowprops={"arrowstyle": "->", "color": tint, "connectionstyle": f"arc3,rad={-.18 - .018*(b-a)}", "lw": 3 if chosen else 1.4})
        ax.text((a+b)/2, lift, f"{edge['piece']}\np={edge['p']:.2f}", ha="center", va="center", color=tint, fontsize=10)
    for i in range(n+1):
        ax.plot(i, .1, "o", color=COLORS[1] if i == end else INK)
        ax.text(i, -.16, str(i), ha="center", fontsize=10)
    ax = fig.add_subplot(212); ax.axis("off")
    if frame < n:
        state = result["steps"][max(0, end-1)]
        text = f"Evaluate end={end}; best predecessor={state['predecessor']}\n"
        text += "\n".join(f"{e['piece']}: log p={e['log_p']:.4f}; cumulative={e['score']}" for e in state["evaluated"])
        ax.text(.01, .9, text or "Unreachable prefix", va="top", fontsize=12)
        ax.text(.01, .12, "DP best scores: " + "  ".join("—" if x is None else f"{x:.2f}" for x in state["best"]), fontsize=10)
    else:
        for i, row in enumerate(result["paths"]):
            ax.text(.01, .9-i*.24, f"{' + '.join(row['pieces'])}   P={row['probability']:.5f}   log P={row['log_probability']:.4f}", fontsize=13,
                    color=COLORS[2] if row["pieces"] == result["pieces"] else INK)
        ax.text(.01, .05, "Backtrack the best predecessors → " + " + ".join(result["pieces"]), weight="bold")


def sentencepiece(fig, data, frame=0):
    rows = data["sentencepiece"]; pair = rows[(frame % 3)*2:(frame % 3)*2+2]
    title(fig, "06 · SentencePiece is a framework, with BPE and Unigram", "Same synthetic training corpus; actual normalization, whitespace pieces, IDs and decoded text")
    axes = fig.subplots(2, 1)
    for ax, row in zip(axes, pair):
        ax.axis("off"); ax.set_title(row["name"], loc="left", color=COLORS[0])
        lines = [f"Original: {row['text']!r}", f"Normalized marker stream: {row['normalized']!r}",
                 "Pieces: " + " | ".join(row["pieces"]), f"IDs: {row['ids']}",
                 f"Decoded: {row['decoded']!r} · exact original reconstruction={row['exact_reconstruction']}"]
        ax.text(.01, .97, "\n".join(textwrap.fill(line, 115) for line in lines), va="top", fontsize=11, linespacing=1.35)


def unicode(fig, data, frame=0):
    row = data["unicode"][frame % len(data["unicode"])]
    title(fig, "07 · Byte-token boundaries can split a character", "Actual cl100k_base output; concatenate token bytes before strict UTF-8 decoding")
    ax = fig.add_subplot(211); ax.axis("off")
    codepoints = " ".join(f"U+{ord(c):04X}" for c in row["text"])
    byte_info = " | ".join(f"{c!r}: {c.encode().hex(' ')} ({len(c.encode())} B)" for c in row["text"])
    ax.text(.01, .92, f"Text: {row['text']!r}\nCode points: {codepoints}\nUTF-8 bytes: {row['text'].encode().hex(' ')}\n{byte_info}", va="top", fontsize=12, linespacing=1.65)
    ax = fig.add_subplot(212); ax.axis("off")
    lines = []
    for token, piece in zip(row["ids"], row["pieces"]):
        try:
            independently = repr(bytes.fromhex(piece).decode("utf-8", errors="strict"))
        except UnicodeDecodeError:
            independently = "INVALID UTF-8 ALONE"
        lines.append(f"ID {token:<7} bytes {piece:<24} → {independently}")
    ax.text(.01, .97, "\n".join(lines[:9]) + f"\nFull byte reconstruction: {row['decoded']!r} ✓", va="top", fontsize=12)


def comparison(fig, data):
    title(fig, "08 · Tokenizers define different units", "Actual counts; local tiny models and pretrained encodings are not a controlled algorithm benchmark")
    fig.subplots_adjust(left=.20, wspace=.40)
    rows = data["comparison"]
    ax = fig.add_subplot(121)
    ax.barh([r["name"] for r in rows][::-1], [r["tokens"] for r in rows][::-1], color=COLORS)
    ax.set(xlabel="Tokens / pieces", title="Portuguese probe (character BPE uses 'lowest newer')")
    ax = fig.add_subplot(122); ax.axis("off")
    for i, row in enumerate(rows):
        ax.text(0, .96 - i*.15, f"{row['name']}\n{row['tokens']} units · unknown {row['unknown']} · exact {row['exact_reconstruction']}", fontsize=11)


def vocabulary(fig, data, frame=7):
    rows = data["vocabulary"]
    limit = sorted({r["merge_limit"] for r in rows})[min(frame, 7)]
    title(fig, "09 · More vocabulary does not guarantee held-out compression", "Train and held-out words shown separately; one unsupported held-out word is explicitly rejected")
    axes = fig.subplots(2, 2)
    for partition, tint in (("train", COLORS[0]), ("held_out", COLORS[1])):
        selected = [r for r in rows if r["partition"] == partition and r["merge_limit"] <= limit]
        x = [r["vocabulary"] for r in selected]
        axes[0, 0].plot([r["merges"] for r in selected], x, "o-", label=partition, color=tint)
        axes[0, 1].plot(x, [r["tokens"] for r in selected], "o-", label=partition, color=tint)
        axes[1, 0].plot(x, [r["compression_supported_characters_per_token"] for r in selected], "o-", label=partition, color=tint)
        axes[1, 1].scatter(x, [r["mean_tokens_per_supported_word"] for r in selected], label=partition, color=tint)
    for ax in axes.flat:
        ax.legend(fontsize=9); ax.grid(alpha=.2)
    axes[0, 0].set(xlabel="Actual merges", ylabel="Vocabulary entries")
    axes[0, 1].set(xlabel="Vocabulary entries", ylabel="Total supported tokens")
    axes[1, 0].set(xlabel="Vocabulary entries", ylabel="Supported code points / token")
    twin = axes[1, 0].twinx(); twin.set_ylabel("Vocabulary entries", color=MUTED)
    selected = [r for r in rows if r["partition"] == "train" and r["merge_limit"] <= limit]
    twin.plot([r["vocabulary"] for r in selected], [r["vocabulary"] for r in selected], "--", color=MUTED, alpha=.45)
    axes[1, 1].set(xlabel="Vocabulary entries", ylabel="Tokens / supported word")


def embedding(fig, data, frame=None):
    if frame is not None:
        pipeline(fig, data, frame=5)
        ids = get_encoding("cl100k_base").encode_ordinary("Machine learning is powerful.")
        # Highlight selected rows in the already rendered synthetic lookup matrix.
        axis = fig.axes[2]
        axis.axhspan(frame % len(ids) - .45, frame % len(ids) + .45, fill=False, edgecolor=COLORS[1], lw=3)
        return
    title(fig, "10 · Vocabulary storage is a separate budget", "Theoretical embedding + bias-free output weights; excludes all other model and training memory")
    ax = fig.add_subplot(111)
    sizes = np.array([16000, 32000, 50000, 100000, 250000])
    for width, tint in zip([512, 2048, 4096, 8192], COLORS):
        ax.plot(sizes, sizes*width*2/1e9, "o-", label=f"d={width}, tied FP16/BF16", color=tint)
    ax.set(xlabel="Vocabulary entries", ylabel="Weight storage (decimal GB)")
    ax.legend(); ax.grid(alpha=.2)


def attention(fig, data, frame=5):
    row = data["attention"][min(frame, 5)]
    title(fig, "11 · Attention interaction counts depend on the workload", "Per head: dense n²; causal valid n(n+1)/2; one cached new query n. Counts are not measured FLOPs or latency.")
    ax = fig.add_subplot(121)
    # A 24×24 schematic avoids allocating a 4096×4096 matrix for the illustration.
    ax.imshow(np.tril(np.ones((24, 24))), cmap="Blues", vmin=0, vmax=1)
    ax.set(title=f"Schematic only: {row['length']} × {row['length']} positions", xlabel="Key position (scaled)", ylabel="Query position (scaled)", xticks=[0, 23], xticklabels=[0, row["length"]-1], yticks=[0, 23], yticklabels=[0, row["length"]-1])
    ax = fig.add_subplot(122)
    for key, label, tint in (("dense", "Dense full sequence", COLORS[0]), ("causal_valid", "Causal valid pairs", COLORS[1]), ("one_cached_decode_query", "One cached new query", COLORS[2])):
        ax.plot([r["length"] for r in data["attention"]], [r[key] for r in data["attention"]], "o-", label=label, color=tint)
    ax.set(xscale="log", yscale="log", xlabel="Sequence length n", ylabel="Interactions per head")
    ax.axvline(row["length"], color=MUTED, ls="--"); ax.legend(fontsize=9)
    fig.text(.55, .13, f"Selected dense={row['dense']:,}; causal={row['causal_valid']:,}; cached query={row['length']:,}", fontsize=10)


def multilingual(fig, data, heatmap=False):
    title(fig, "12 · The same content has encoding-dependent length", "Five handwritten approximately equivalent translations; descriptive probes, not language-wide efficiency or fairness")
    languages = list(dict.fromkeys(r["language"] for r in data["multilingual"]))
    matrix = np.array([[next(r["tokens"] for r in data["multilingual"] if r["language"] == language and r["name"] == enc)
                        for enc in ("cl100k_base", "o200k_base")] for language in languages])
    ax = fig.add_subplot(111)
    if heatmap:
        ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set(xticks=[0, 1], xticklabels=["cl100k_base", "o200k_base"], yticks=range(5), yticklabels=languages)
        for i in range(5):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max()*.65 else INK)
    else:
        x = np.arange(5)
        ax.bar(x-.18, matrix[:, 0], .36, label="cl100k_base", color=COLORS[0])
        ax.bar(x+.18, matrix[:, 1], .36, label="o200k_base", color=COLORS[1])
        ax.set(xticks=x, xticklabels=languages, ylabel="Actual tokens for selected passage"); ax.legend()


def ocr(fig, data, frame=None):
    rows = data["ocr"]
    title(fig, "13 · OCR corruption can raise or lower token counts", "Independent mechanisms, preserved clean baselines, matched seed trials; no retrieval or answer-quality evaluation")
    if frame is not None:
        candidates = [r for r in rows if r["encoding"] == "cl100k_base" and r["passage"] == "technical" and r["mechanism"] == "insert_space" and r["trial"] == 0]
        row = candidates[min(frame, len(candidates)-1)]
        ax = fig.add_subplot(211); ax.axis("off")
        ax.text(.01, .9, "Clean: " + PRESETS["Technical"] + f"\nCorrupted (rate={row['rate']:.2f}): " + row["text"], fontsize=13, va="top", wrap=True)
        ax = fig.add_subplot(212)
        pieces = [get_encoding("cl100k_base").decode_single_token_bytes(i).decode("utf-8", errors="backslashreplace") for i in get_encoding("cl100k_base").encode_ordinary(row["text"])]
        blocks(ax, pieces, fontsize=10)
        ax.set_title(f"{row['tokens']} tokens vs {row['clean_tokens']} clean · inflation {row['inflation_pct']:.1f}%", loc="left")
        return
    axes = fig.subplots(1, 2)
    methods = ["substitution", "insert_space", "break_word", "remove_space", "deletion"]
    for method, tint in zip(methods, COLORS):
        selected = [r for r in rows if r["mechanism"] == method and r["encoding"] == "cl100k_base"]
        rates = sorted({r["rate"] for r in selected})
        axes[0].plot(rates, [np.mean([r["inflation_pct"] for r in selected if r["rate"] == rate]) for rate in rates], "o-", label=method, color=tint)
    axes[0].axhline(0, color=MUTED); axes[0].set(xlabel="Corruption probability per code point", ylabel="Mean inflation (%) across two passages"); axes[0].legend(fontsize=9)
    values = [[r["inflation_pct"] for r in rows if r["mechanism"] == m and r["rate"] == .3 and r["encoding"] == "cl100k_base"] for m in methods]
    axes[1].boxplot(values, tick_labels=["sub", "space", "break", "remove", "delete"])
    axes[1].set(ylabel="Inflation (%) at rate 0.30", xlabel="Mechanism (two passages × seeded trials)")
    axes[1].axhline(0, color=MUTED)


def rag(fig, data, frame=2):
    title(fig, "14 · Chunk limits and evidence containment are different checks", "Actual UTF-8-safe character spans; character mode uses code points, other modes use tokens; overlap=5 in each mode's units")
    axes = fig.subplots(3, 1)
    for ax, row in zip(axes, data["rag"]):
        chunks = row["chunks"][:max(1, int(len(row["chunks"])*(min(frame, 2)+1)/3))]
        for j, chunk in enumerate(chunks):
            y = j % 2
            ax.broken_barh([(chunk["start"], chunk["end"]-chunk["start"])], (y, .8), facecolors=COLORS[j % len(COLORS)], alpha=.8)
            if len(row["chunks"]) <= 5:
                ax.text((chunk["start"]+chunk["end"])/2, y+.4, f"{chunk['tokens']} tok", ha="center", va="center", color="white", fontsize=9)
        ax.set(xlim=(0, 540), ylim=(0, 2), yticks=[], xlabel="Original document code-point offset")
        m = row["metrics"]
        ax.set_title(f"{row['strategy']}: {m['chunks']} chunks · max={m['max']} tok · duplicate points={m['duplicated_code_points']} · evidence contained={m['evidence_contained']}", loc="left", fontsize=11)


def context(fig, data):
    title(fig, "15 · Context capacity is an admission constraint", "Hypothetical shared decoder window; reserved generation is not measured output or a reasoning-quality guarantee")
    ax = fig.add_subplot(111)
    labels = ["System", "History", "Query", "Retrieval", "Tools/overhead", "Output reserve", "Remaining"]
    counts = [1500, 4000, 300, 14000, 500, 4000, 8468]
    left = 0
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax.barh([0], [count], left=left, color=COLORS[i % len(COLORS)] if i < 6 else "#dce5ed", label=f"{label}: {count:,}")
        left += count
    ax.set(xlim=(0, 32768), xlabel="Allocated + remaining tokens", yticks=[], title="24,300 allocated / 32,768 hypothetical capacity")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, fontsize=10)


def cost(fig, data):
    title(fig, "16 · Cost projections use explicit hypothetical rates", "USD 2/million input, USD 8/million output; 1,000 output tokens/request; excludes cache, tools and infrastructure")
    axes = fig.subplots(1, 2)
    tokens = np.linspace(0, 32000, 40)
    for volume, tint in zip([1000, 10000, 100000], COLORS):
        axes[0].plot(tokens, (tokens*2+1000*8)/1e6*volume, label=f"{volume:,} monthly requests", color=tint)
    axes[0].set(xlabel="Input tokens/request", ylabel="Estimated monthly USD"); axes[0].legend()
    reduction = np.linspace(0, .8, 30)
    axes[1].plot(reduction*100, (6000*(1-reduction)*2+8000)/1e6*10000, color=COLORS[2])
    axes[1].set(xlabel="Hypothetical input-context reduction (%)", ylabel="Estimated monthly USD (10,000 requests)")
    axes[1].text(.05, .12, "Output fixed; answer quality unmeasured.\nProjected savings are not production savings.", transform=axes[1].transAxes, fontsize=11)


def save_png(path, draw, quick=False):
    fig = plt.figure(figsize=(14, 8.5))
    try:
        draw(fig)
        fig.savefig(path, dpi=100 if quick else 190)
    finally:
        plt.close(fig)


def save_gif(path, draw, frames, quick=False):
    indices = list(range(frames))
    if quick and len(indices) > 8:
        indices = np.unique(np.linspace(0, frames-1, 8).astype(int)).tolist()
    images = []
    for index in indices:
        fig = plt.figure(figsize=(14, 8.5), dpi=75 if quick else 100)
        try:
            draw(fig, index)
            fig.canvas.draw()
            images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        finally:
            plt.close(fig)
    images[0].save(path, save_all=True, append_images=images[1:], duration=950, loop=0, optimize=False)
