"""Sequence, mask, multi-head and architecture renderers with computed weights."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from attention_figures import decorate, matrix, planar_arrows, save_gif, save_png
from attention_lab import (
    COLORS, cache_fixture, context_fixture, cross_fixture, leakage_fixture,
    mask_fixture, memory_estimates, multihead_fixture, rag_fixture,
)


def token_graph(ax, queries, keys, weights, active=0):
    ax.set(xlim=(-.6, max(len(queries), len(keys))-.3), ylim=(-.6, 2.7))
    ax.axis("off")
    for i, word in enumerate(queries):
        ax.text(i, 2.1, word, ha="center", fontsize=8, color=COLORS["query"],
                bbox=dict(facecolor="#dbeafe" if i == active else "#f1f5f9", edgecolor="none", pad=4))
    for j, word in enumerate(keys):
        ax.text(j, 0, word, ha="center", fontsize=8, color=COLORS["key"])
        ax.add_patch(FancyArrowPatch((active, 1.9), (j, .25), arrowstyle="->",
                                    mutation_scale=8, linewidth=.3+6*weights[active, j],
                                    alpha=.15+.85*weights[active, j], color=COLORS["output"]))
    ax.text(0, 2.55, f"Query row {active} → available keys · width encodes α", fontsize=10)


def contextual(path, quick=False):
    contexts = [context_fixture(False), context_fixture(True)]
    frames = 12 if quick else 24
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    def update(index):
        blend = index/(frames-1)
        chosen = contexts[int(blend >= .5)]
        t, bank = chosen["attention"], chosen["bank"]
        for ax in axes.flat:
            ax.clear()
        token_graph(axes[0, 0], chosen["words"], chosen["words"], t.weights, bank)
        axes[0, 0].set_title("Endpoint context shown: " + ("finance" if blend >= .5 else "river"), loc="left", fontsize=11)
        matrix(axes[0, 1], t.weights, "Actual endpoint A", limits=(0, 1))
        for fixture, marker, label in zip(contexts, ("o", "s"), ("river", "finance")):
            point = fixture["attention"].output[fixture["bank"]]
            axes[1, 0].scatter(*point, color=COLORS["output"], marker=marker, label=label+" output", s=70)
        first, second = [c["attention"].output[c["bank"]] for c in contexts]
        interpolated = (1-blend)*first + blend*second
        axes[1, 0].scatter(*chosen["x"][bank], color=COLORS["value"], label="same bank input", s=70)
        axes[1, 0].plot([first[0], second[0]], [first[1], second[1]], "--", color=COLORS["mask"])
        axes[1, 0].scatter(*interpolated, facecolors="none", edgecolors=COLORS["query"], s=160, label="display interpolation")
        axes[1, 0].set(xlim=(-.1, 2.1), ylim=(-.1, 2.1), xlabel="Engineered water feature", ylabel="Engineered finance feature")
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].axis("off")
        axes[1, 1].text(0, .95, "Engineered, untrained features\n\nQᵢ = [1,1] for every token\nK = 2X; V = X\n\n"+
                            f"River bank O = {np.round(first,3)}\nFinance bank O = {np.round(second,3)}\n\n"+
                            "Dashed transition is display interpolation,\nnot model attention on a blended sentence.", fontsize=11, va="top")
        decorate(fig, "07 · Same bank input, two different contextual mixtures", "Surrounding features explicitly encode water or finance. Actual endpoint matrices use their own sentence lengths. Links show bank-query weights. No trained linguistic understanding or semantic accuracy is demonstrated; the moving circle only interpolates endpoint outputs.")
    save_gif(path, fig, update, frames, fps=3)


def masking(gif_path, png_path, leakage_path):
    cases = mask_fixture()
    fig, axes = plt.subplots(1, 5, figsize=(17, 6))
    names = list(cases)
    def update(index):
        mode, row = divmod(index, 6)
        name = names[mode]
        t = cases[name]
        arrays = (t.raw, t.allowed.astype(float), t.masked, t.weights, t.output)
        labels = ("Raw QKᵀ", "True=allowed", "Scaled + mask", "Normalized A", "Output O")
        colors = ("key", "mask", "key", "output", "output")
        for ax, array, label, color in zip(axes, arrays, labels, colors):
            ax.clear()
            shown = array.copy()
            if label in ("Normalized A", "Output O"):
                shown[row+1:] = np.nan
            matrix(ax, shown, label, color, active=(row, 0), limits=(0, 1) if label in ("Normalized A", "True=allowed") else None)
            ax.set_ylabel("Query position" if label == "Raw QKᵀ" else "")
        decorate(fig, f"08 · {name} attention · reveal query row {row}", "Orange: scores; gray cells: blocked −∞ or unrevealed rows; purple: weights/output. Padding excludes key positions 4,5 but still computes padded-query outputs. Causal forbids j>i. Every allowed row sums to one; fully blocked rows raise ValueError. Future information must also be absent upstream.")
    save_gif(gif_path, fig, update, 18, fps=2)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, name in enumerate(("Full", "Causal")):
        t = cases[name]
        matrix(axes[row, 0], t.allowed, name+" mask", "mask", limits=(0, 1))
        matrix(axes[row, 1], t.weights, name+" weights", limits=(0, 1))
        matrix(axes[row, 2], t.output, name+" output")
    decorate(fig, "08 · Full visibility versus causal visibility", "Same Q/K/V in both rows. Gray mask: 1 permits, 0 blocks. All prohibited weights are exactly zero. Causal outputs only depend on permitted representations; the diagonal is included. This cannot repair leakage already encoded into earlier Q/K/V.")
    save_png(png_path, fig)
    original, changed = leakage_fixture()
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    def leak(index):
        use_changed = index % 2 == 1
        for ax in axes:
            ax.clear()
        for ax, name in zip(axes[:2], ("Full", "Causal")):
            before = original[name].output
            after = changed[name].output if use_changed else before
            ax.plot(range(6), before[:, 0], "o--", color=COLORS["mask"], label="Original x output")
            ax.plot(range(6), after[:, 0], "s-", color=COLORS["output"], label="Current x output")
            ax.axvspan(-.3, 4.6, color=COLORS["query"], alpha=.05)
            ax.set(xlim=(-.3, 5.3), ylim=(-12, 14), xlabel="Query position", ylabel="Output coordinate x", title=name)
            ax.legend(fontsize=8)
        axes[2].axis("off")
        full_error = np.max(abs(original["Full"].output[:5]-changed["Full"].output[:5])) if use_changed else 0
        causal_error = np.max(abs(original["Causal"].output[:5]-changed["Causal"].output[:5])) if use_changed else 0
        axes[2].text(0, .9, ("FUTURE K/V CHANGED" if use_changed else "ORIGINAL K/V")+
                     f"\n\nFirst 5 outputs: max |Δ|\nFull = {full_error:.6f}\nCausal = {causal_error:.6f}\n\n"+
                     "Last key/value changed.\nEarlier queries and permitted\nkeys/values remain fixed.\n\nUnmasked future access can leak\ninformation during prediction.", va="top", fontsize=12)
        decorate(fig, "08 · Future-token intervention with earlier representations fixed", "Synthetic last K becomes [9,−7,6]; last V becomes [12,−10]. Both output coordinates are included in the reported max difference; plots show x only. No causal claim about a complete trained network follows from this numerical control.")
    save_gif(leakage_path, fig, leak, 6, fps=1)
    return {name: float(np.max(abs(original[name].output[:5]-changed[name].output[:5]))) for name in ("Full", "Causal")}


def self_cross(gif_path, png_path):
    fixture = cross_fixture()
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    def update(index):
        cross = index >= 5
        t = fixture["cross_attention" if cross else "self_attention"]
        row = (index-5) if cross else index
        panels = [(fixture["source"], "Encoder source Y", "mask"), (fixture["target"], "Decoder target X", "mask"),
                  (fixture["wq"], "W_Q", "query"), (fixture["wk"], "W_K", "key"), (fixture["wv"], "W_V", "value"),
                  (t.query, "Q from X" if cross else "Q from Y", "query"), (t.key, "K from Y", "key"), (t.value, "V from Y", "value"),
                  (t.weights, "Rectangular A" if cross else "Square A", "output"), (t.output, "Contextual O", "output")]
        for ax in axes.flat:
            ax.clear()
        for ax, (data, title, color) in zip(axes.flat, panels):
            matrix(ax, data, title, color, limits=(0, 1) if title.endswith(" A") else None)
        token_graph(axes.flat[10], [f"q{i}" for i in range(len(t.query))], [f"s{i}" for i in range(5)], t.weights, row)
        axes.flat[11].axis("off")
        axes.flat[11].text(0, .95, ("CROSS-ATTENTION" if cross else "SELF-ATTENTION")+
                           f"\n\nQ: {t.query.shape}\nK/V: {t.key.shape}\nA: {t.weights.shape}\nO: {t.output.shape}\n\nUntrained projections.\nEncoder-decoder analogy;\nno model is trained.", va="top", fontsize=11)
        decorate(fig, "09 · Query origin changes; key/value source stays paired", "Five source tokens; three target tokens. Synthetic projections have dₖ=dᵥ=2. Self: Q/K/V from Y. Cross: Q from X; K/V from Y. Thick purple links show the active query row. All matrices and outputs are computed, with no padding or causal mask in this comparison.")
    save_gif(gif_path, fig, update, 8, fps=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    t = fixture["cross_attention"]
    matrix(axes[0], t.weights, "Cross attention · targets × source", limits=(0, 1), rows=["target 0", "target 1", "target 2"], columns=[f"src {i}" for i in range(5)])
    matrix(axes[1], t.output, "Output · targets × dᵥ")
    decorate(fig, "09 · Different sequence lengths produce rectangular attention", "Q: 3×2; K: 5×2; V: 5×2 → A: 3×5 and O: 3×2. Each target query independently normalizes over all five source keys. These synthetic matrices are not machine-translation predictions.")
    save_png(png_path, fig)


def multihead(path):
    fixture = multihead_fixture()
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    def update(index):
        for ax in axes.flat:
            ax.clear()
        for i, (t, projections) in enumerate(zip(fixture["heads"], fixture["projections"])):
            packed = np.hstack(projections)
            matrix(axes[i, 0], packed, f"Head {i+1}: WQ | WK | WV", "query", columns=["Q0", "Q1", "K0", "K1", "V0", "V1"])
            matrix(axes[i, 1], t.weights, f"Head {i+1}: A", limits=(0, 1))
            matrix(axes[i, 2], t.output, f"Head {i+1}: O")
            ax = axes[i, 3]
            if i == 0:
                matrix(ax, fixture["x"], "Same input X", "mask")
            elif i == 1:
                matrix(ax, fixture["concatenated"] if index >= 3 else np.full_like(fixture["concatenated"], np.nan), "Concat: [O1 | O2 | O3]")
            else:
                matrix(ax, fixture["output"] if index >= 4 else np.full_like(fixture["output"], np.nan), "Final concat @ W_O")
            if index == i:
                axes[i, 1].add_patch(FancyBboxPatch((-.03, -.03), 1.06, 1.06, boxstyle="square,pad=0", transform=axes[i, 1].transAxes, fill=False, color=COLORS["output"], lw=3, clip_on=False))
        stages = ["Head 1: projected matching", "Head 2: another projection space", "Head 3: a third matching pattern", "Concatenate: (4×2) × 3 → 4×6", "Output projection: (4×6) @ (6×3) → 4×3"]
        decorate(fig, "10 · Independent heads, concatenation and output projection", stages[index]+". W_O (synthetic) = "+np.array2string(fixture["wo"], precision=1).replace("\n", " ")+". Heads are intentionally designed; no semantic/syntactic specialization is learned or claimed.")
    save_gif(path, fig, update, 5, fps=1)


def complexity(png_path, gif_path, timings):
    theory = memory_estimates()
    n = theory["lengths"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    axes[0].loglog(n, theory["interactions"], "o-", color=COLORS["output"], label="n² pair interactions")
    axes[0].loglog(n, n*n[0], "--", color=COLORS["mask"], label="linear reference, aligned at n=128")
    axes[0].set(xlabel="Tokens", ylabel="Pairs per head/example", title="Theoretical dense growth")
    for dtype, color in (("fp32", COLORS["query"]), ("fp16", COLORS["key"])):
        axes[1].loglog(n, theory[dtype+"_mib"], "o-", color=color, label=dtype.upper())
    axes[1].set(xlabel="Tokens", ylabel="MiB", title="One score OR weight matrix")
    measured = [row for row in timings if row["median_seconds"] is not None]
    if measured:
        sizes = [r["length"] for r in measured]
        times = np.array([r["median_seconds"] for r in measured])*1000
        axes[2].loglog(sizes, times, "o-", label="Observed NumPy median", color=COLORS["output"])
        axes[2].loglog(sizes, times[0]*(np.array(sizes)/sizes[0])**2, "--", label="Quadratic reference", color=COLORS["mask"])
        axes[2].set(xlabel="Tokens", ylabel="Milliseconds", title="CPU float64 · dₖ=dᵥ=32")
        axes[2].set_xticks(sizes, [str(size) for size in sizes], rotation=25, ha="right", fontsize=9)
        from matplotlib.ticker import NullLocator
        axes[2].xaxis.set_minor_locator(NullLocator())
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.15)
    decorate(fig, "11 · Quadratic interactions, matrix bytes and observed CPU time", "Theory up to 16,384 tokens allocates no corresponding score matrices. Timings only for listed lengths; one warm-up and median of repeated complete NumPy calls, including checks/softmax/allocations. FP16 is a byte estimate, not measured arithmetic. Environment and samples are in run.json.")
    fig.subplots_adjust(bottom=.23)
    save_png(png_path, fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    def update(index):
        size = int(n[index])
        for ax in axes:
            ax.clear()
        side = size/n[-1]
        axes[0].add_patch(FancyBboxPatch((0, 0), side, side, boxstyle="square,pad=0", color=COLORS["output"], alpha=.65))
        axes[0].set(xlim=(0, 1.05), ylim=(0, 1.05), aspect="equal", xlabel="Key axis / 16,384", ylabel="Query axis / 16,384", title=f"n={size:,}: {size*size:,} pairs")
        axes[1].bar(["FP16", "FP32"], [theory["fp16_mib"][index], theory["fp32_mib"][index]], color=[COLORS["key"], COLORS["query"]])
        axes[1].set(ylim=(0, 1100), ylabel="MiB", title="One matrix only · theory")
        decorate(fig, "11 · Matrix area grows as n²", "The square's side scales with sequence length and its area with interaction count. Fixed axes across all frames. Larger theoretical lengths are not allocated or timed. FlashAttention reduces materialization/traffic without removing dense quadratic arithmetic.")
    save_gif(gif_path, fig, update, len(n), fps=1)


def flow_boxes(ax, labels, active):
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    for i, label in enumerate(labels):
        x = .02+(i%4)*.25
        y = .70-(i//4)*.44
        face = "#dbeafe" if i == active else "#f1f5f9"
        ax.add_patch(FancyBboxPatch((x, y), .21, .24, boxstyle="round,pad=.008", facecolor=face, edgecolor=COLORS["query"] if i == active else "#cbd5e1"))
        ax.text(x+.105, y+.12, label, ha="center", va="center", fontsize=9)
        if i%4 < 3 and i < len(labels)-1:
            ax.annotate("", (x+.24, y+.12), (x+.215, y+.12), arrowprops=dict(arrowstyle="->", color="#64748b"))
    if len(labels)>4:
        ax.annotate("", (.10, .50), (.88, .65), arrowprops=dict(arrowstyle="->", color="#64748b", connectionstyle="arc3,rad=-.1"))


def rag_cache(rag_path, cache_path):
    fixture = rag_fixture()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    labels = ["User query", "External vector\nretrieval", "Top-2 document\nchunks", "Prompt assembly",
              "Token vectors", "Internal neural\nattention", "Response stage\n(no text model)"]
    def update(index):
        for ax in axes.flat:
            ax.clear()
        flow_boxes(axes[0, 0], labels, index)
        axes[0, 1].bar(["Doc 0", "Doc 1", "Doc 2"], fixture["cosine"], color=[COLORS["key"] if i in fixture["selected"] else COLORS["mask"] for i in range(3)])
        axes[0, 1].set(ylim=(0, 1.1), ylabel="Cosine similarity", title="External retrieval · three synthetic document vectors")
        supplied_labels = ["query"] + [f"doc {i}" for i in fixture["selected"]]
        matrix(axes[1, 0], fixture["attention"].weights, "Internal attention on supplied vectors", limits=(0, 1), rows=supplied_labels, columns=supplied_labels)
        matrix(axes[1, 1], fixture["attention"].output, "Contextual representations · not generated text")
        decorate(fig, "12A · Retrieval supplies candidates; attention aggregates context", f"Actual cosine ranking selects document IDs {fixture['selected'].tolist()}. Internal attention uses [query, selected doc vectors] as a tiny token-proxy sequence. No tokenizer, LLM, citation engine or generated answer is implemented. Architecture is conceptual; weights are calculated.")
    save_gif(rag_path, fig, update, len(labels), fps=1)
    cache = cache_fixture()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    def cached(index):
        step = max(0, index-1)
        state = cache["states"][min(step, 2)]
        t = state["cache"]
        for ax in axes.flat:
            ax.clear()
        if index == 0:
            matrix(axes[0, 0], cache["x"][:3], "Prefill prompt X · 3 positions", "mask")
            matrix(axes[0, 1], cache["x"][:3]@cache["wk"], "Store prompt K", "key")
            matrix(axes[0, 2], cache["x"][:3]@cache["wv"], "Store prompt V", "value")
            prompt = cache["x"][:3]
            from attention_lab import trace
            prefill = trace(prompt@cache["wq"], prompt@cache["wk"], prompt@cache["wv"], np.tri(3, dtype=bool))
            matrix(axes[1, 0], prefill.weights, "Causal prefill attention", limits=(0, 1))
            matrix(axes[1, 1], prefill.output, "Prefill outputs")
        else:
            matrix(axes[0, 0], state["query"], "New query only", "query")
            matrix(axes[0, 1], t.key, "Cached + appended K", "key", active=(len(t.key)-1, 0))
            matrix(axes[0, 2], t.value, "Cached + appended V", "value", active=(len(t.value)-1, 0))
            matrix(axes[1, 0], t.weights, "New query attends to full prefix", limits=(0, 1))
            matrix(axes[1, 1], t.output, "Cached-path output")
        axes[1, 2].axis("off")
        axes[1, 2].text(0, .95, ("PREFILL" if index == 0 else f"DECODE STEP {index}")+
                           ("\n\nProject 3 prompt positions.\nStore their K/V for reuse." if index == 0 else
                            f"\n\nCache length: {len(t.key)}\nNew K/V projection rows: 1\nFull-prefix alternative: {len(t.key)}\nMax output error: {state['error']:.2e}")+
                           "\n\nAttention over the prefix\nis still computed.\nNo text is sampled here.", va="top", fontsize=11)
        decorate(fig, "12B · A growing KV cache reuses earlier projections", "One frozen synthetic layer; six predetermined input vectors, first three as prompt. New K/V are appended once. Cached outputs are compared numerically with full causal prefix recomputation. Token sampling, positions, residuals, multiple layers and a real decoder are deliberately omitted.")
    save_gif(cache_path, fig, cached, 4, fps=1)
    return dict(retrieved_ids=fixture["selected"].tolist(), cosine=fixture["cosine"].tolist(),
                cache_errors=[s["error"] for s in cache["states"]],
                cache_lengths=[len(s["cache"].key) for s in cache["states"]])
