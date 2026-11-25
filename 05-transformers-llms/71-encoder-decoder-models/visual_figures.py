"""Matplotlib figures and Pillow animations for the computed Day 71 data."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
import numpy as np
from matplotlib.colors import ListedColormap
import textwrap

from visual_experiments import COLORS, SOURCE, TARGET, connection_counts, shared_pca
from from_scratch import visibility

FAMILIES = ("encoder", "decoder", "cross")
NAMES = {"encoder": "Encoder-only", "decoder": "Decoder-only", "cross": "Encoder-decoder"}


def configure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7f9fc", "axes.facecolor": "#ffffff",
                         "text.color": COLORS["ink"], "axes.labelcolor": COLORS["ink"]})


def caption(fig, text):
    fig.text(0.025, 0.015, textwrap.fill(text, width=170), fontsize=9, color="#4d5665", va="bottom")


def heatmap(ax, matrix, keys, queries, title, *, probability=False, allowed=None, cmap="viridis", limit=None):
    values = np.asarray(matrix, dtype=float)
    masked = np.ma.masked_where(~allowed, values) if allowed is not None else np.ma.masked_invalid(values)
    palette = plt.get_cmap(cmap).with_extremes(bad=COLORS["masked"])
    image = ax.imshow(masked, cmap=palette, vmin=0 if probability else (-limit if limit else None),
                      vmax=1 if probability else limit, aspect="auto")
    ax.set_xticks(range(len(keys)), keys, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(queries)), queries, fontsize=8)
    ax.set(xlabel="Key position / token", ylabel="Query position / token", title=title)
    if allowed is not None:
        for i, j in zip(*np.where(~allowed)):
            ax.text(j, i, "×", ha="center", va="center", color="#303744", fontsize=10)
    return image


def save_png(path, draw, quick=False):
    fig = plt.figure(figsize=(13, 7), layout="constrained")
    try:
        draw(fig)
        fig.get_layout_engine().set(rect=(0, 0.09, 1, 0.84))
        fig.savefig(path, dpi=95 if quick else 135, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)


def save_gif(path, draw, frames, quick=False):
    fig = plt.figure(figsize=(12, 7))
    try:
        def update(index):
            fig.clear()
            draw(fig, index)
            fig.subplots_adjust(left=0.07, right=0.94, bottom=0.17, top=0.85, wspace=0.4, hspace=0.6)
            return []
        animation = FuncAnimation(fig, update, frames=frames, interval=900, repeat=True)
        animation.save(path, writer=PillowWriter(fps=1.1), dpi=80 if quick else 110)
    finally:
        plt.close(fig)


def architecture(fig, models, frame=None):
    fig.set_size_inches(14, 10)
    axes = fig.subplots(1, 3)
    for ax, family in zip(axes, FAMILIES):
        ax.set(xlim=(0, 1), ylim=(-2.8, 12.2))
        ax.axis("off")
        ax.set_title(NAMES[family], color=COLORS[family], weight="bold", pad=18)
        trace = models["traces"][family]
        labels = ["Input embeddings", "Positional information", *list(trace)[1:]]
        labels[2] = "Multi-head " + ("causal " if family != "encoder" else "bidirectional ") + "self-attention"
        labels[-1] = "Final normalization → contextual states"
        if family == "cross":
            labels.insert(0, "Source encoder → memory Hₑ")
        if family != "encoder":
            labels += ["Output projection → probabilities", "Append selected token; repeat"]
        spacing = 11.0 / max(len(labels)-1, 1)
        for step, label in enumerate(labels):
            y = 11-step*spacing
            active = frame is not None and step == min(frame, len(labels)-1)
            base_color = COLORS["decoder"] if family == "cross" else COLORS[family]
            color = COLORS["active"] if active else COLORS["encoder"] if label.startswith("Source encoder") else (COLORS["cross"] if "Cross-attention" == label else base_color)
            width = 0.82 if family == "cross" else 0.94
            ax.add_patch(FancyBboxPatch((0.02, y-0.31), width, 0.62, boxstyle="round,pad=0.02",
                                       edgecolor=color, facecolor=color if active else "white", linewidth=2))
            ax.text(0.43 if family == "cross" else 0.49, y, label, ha="center", va="center", fontsize=8,
                    color="white" if active else COLORS["ink"], wrap=True)
            if step and not (family == "cross" and step == 1):
                ax.annotate("", xy=(0.49, y+0.34), xytext=(0.49, y+spacing-0.34),
                            arrowprops={"arrowstyle": "->", "color": base_color})
        if family == "cross":
            cross_y = 11-labels.index("Cross-attention")*spacing
            ax.plot([0.86, 0.97, 0.97], [11, 11, cross_y], color=COLORS["encoder"], lw=2)
            ax.annotate("", xy=(0.85, cross_y), xytext=(0.97, cross_y),
                        arrowprops={"arrowstyle":"->", "color":COLORS["cross"], "lw":2})
        if family == "encoder":
            ax.text(0.02, -1.2, "Output: one contextual vector per input token\nNo autoregressive text decoder", fontsize=9)
            ax.text(0.02, -2.3, "Input: The bank approved\nthe customer's loan.", fontsize=9)
        else:
            step = 0 if frame is None else max(0, frame-(len(labels)-2)) % len(models["generated"][family])
            records = models["generated"][family]
            chosen = " ".join(record["token"] for record in records[:step+1])
            ax.text(0.02, -1.05, "Untrained greedy output: " + chosen, color=COLORS["active"], fontsize=8, wrap=True)
            best = sorted(zip(models["vocab"], records[step]["probabilities"]), key=lambda x: -x[1])[:3]
            ax.text(0.02, -1.8, "Next-token P: " + "; ".join(f"{t}={p:.2f}" for t, p in best), fontsize=8, wrap=True)
            if family == "cross":
                ax.text(0.02, -2.6, "Source: English encoder memory → cross-attention\nReference only: O banco aprovou o empréstimo do cliente.", fontsize=8, wrap=True)
    fig.suptitle("Transformer information pathways · active component in orange" if frame is not None else
                 "Transformer architectures · shared building blocks, different information paths", fontsize=16)
    caption(fig, "Schematic Post-LN blocks. Numerical states and greedy outputs use random CPU weights; no learned language or translation is claimed.")


def attention_panels(fig, data, kind):
    axes = fig.subplots(1, 3)
    limit = max(abs(item["logits"]).max() for item in data.values())
    for ax, family in zip(axes, FAMILIES):
        item = data[family]
        matrix = item[{"masks": "allowed", "logits": "logits", "probabilities": "weights"}[kind]]
        image = heatmap(ax, matrix, item["key_labels"], item["query_labels"], NAMES[family],
                        probability=kind != "logits", allowed=item["allowed"] if kind == "probabilities" else None,
                        cmap="coolwarm" if kind == "logits" else {"encoder":"Greens","decoder":"Blues","cross":"Purples"}[family], limit=limit if kind == "logits" else None)
        if kind == "masks":
            image.set_cmap(ListedColormap([COLORS["masked"], COLORS[family]]))
        fig.colorbar(image, ax=ax, shrink=0.65, label="Scaled score" if kind == "logits" else "Visibility (1 = permitted)" if kind == "masks" else "Probability")
    fig.suptitle(f"Computed attention {kind} · shared source projections", fontsize=16)
    caption(fig, "Cross-attention uses decoder Q and encoded source K/V. Gray × means blocked; valid small probabilities remain colored. Synthetic single head.")


def masking_frame(fig, data, frame):
    axes = fig.subplots(1, 3)
    stages = ["logits", "allowed", "masked_logits", "weights"]
    stage = stages[frame % 4]
    for ax, family in zip(axes, FAMILIES):
        item = data[family]
        image = heatmap(ax, item[stage], item["key_labels"], item["query_labels"], NAMES[family],
                        probability=stage in ("weights", "allowed"),
                        allowed=item["allowed"] if stage in ("weights", "masked_logits") else None)
        fig.colorbar(image, ax=ax, shrink=0.7)
    fig.suptitle(f"Attention computation: {stage.replace('_', ' ')}", fontsize=16)
    caption(fig, "Scores = QKᵀ / √d. Add 0 to valid pairs and −∞ to blocked pairs; softmax runs across keys. Gray × is inaccessible.")


def cross_frame(fig, item, index=0):
    axes = fig.subplots(2, 2)
    ax = axes[0, 0]
    image = heatmap(ax, item["weights"], item["key_labels"], item["query_labels"], "Rectangular target-to-source probabilities",
                    probability=True, cmap="Purples")
    fig.colorbar(image, ax=ax, shrink=0.7)
    ax.axhspan(index-0.45, index+0.45, fill=False, edgecolor=COLORS["active"], lw=2)
    ax = axes[0, 1]
    ax.axis("off")
    for j, token in enumerate(item["key_labels"]):
        ax.text(j, 1, token, rotation=25, ha="center", fontsize=8, color=COLORS["encoder"])
        weight = item["weights"][index, j]
        ax.plot([2.5, j], [0, 0.85], color=COLORS["cross"], lw=0.5+10*weight, alpha=0.15+0.85*weight)
    ax.text(2.5, -0.15, f"Decoder query {index}: {item['query_labels'][index]}", ha="center", color=COLORS["decoder"])
    ax.set(xlim=(-0.5, 5.5), ylim=(-0.4, 1.3), title="Line width / opacity = computed probability")
    features = np.arange(item["q"].shape[1])
    axes[1, 0].plot(features, item["q"][index], "o-", color=COLORS["decoder"], label="Q from target")
    axes[1, 0].plot(features, item["k"].mean(axis=0), "o-", color=COLORS["encoder"], label="Mean source K")
    axes[1, 0].plot(features, item["v"].mean(axis=0), "o-", color=COLORS["cross"], label="Mean source V")
    axes[1, 0].set(xlabel="Projected feature", ylabel="Value", title="Projected vectors (mean K/V for context)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].bar(features, item["output"][index], color=COLORS["cross"])
    axes[1, 1].set(xlabel="Value feature", ylabel="Weighted output", title="Output = Σ probability × source V")
    fig.suptitle("Cross-attention: decoder queries read encoder keys and values", fontsize=15)
    caption(fig, "English source / Portuguese reference target. Random projections do not produce learned linguistic alignments or translation quality.")


def perturbation(fig, model, frame=None, invariance=False):
    if invariance:
        axes = fig.subplots(1, 2)
        for ax, values, name in zip(axes, [model["deltas"]["decoder"], model["target_deltas"]],
                                     ["Future source input → causal self-attention", "Future target input → conditional decoder"]):
            ax.bar(range(len(values)), values, color=[COLORS["masked"]]*(len(values)-1)+[COLORS["active"]])
            ax.set(xlabel="Output position", ylabel="L2 displacement", title=name)
            ax.text(0.03, 0.9, f"Prefix max = {max(values[:-1]):.2e}\nTolerance = 1e-12", transform=ax.transAxes)
        fig.suptitle("Causal invariance · protected prefixes remain unchanged", fontsize=16)
    else:
        axes = fig.subplots(2, 3)
        fraction = 1 if frame is None else frame / 5
        for column, family in enumerate(FAMILIES):
            baseline = model["baseline"][family]
            modified = baseline + fraction*(model["perturbed"][family]-baseline)
            # Stack baseline and intervention states in one shared feature scale.
            axes[0, column].imshow(np.vstack([baseline, modified]), aspect="auto", vmin=-3, vmax=3, cmap="coolwarm")
            axes[0, column].axhline(len(baseline)-0.5, color="black", lw=1)
            axes[0, column].set(title=NAMES[family]+" states", xlabel="Hidden feature", ylabel="Baseline / modified positions")
            values = np.linalg.norm(modified-baseline, axis=1)
            labels = SOURCE if family != "cross" else ["<start>", *TARGET[:-1]]
            axes[1, column].bar(np.arange(len(values))-(0.18 if family == "cross" else 0), values,
                                width=0.36 if family == "cross" else 0.7,
                                color=[COLORS["masked"]]*(len(values)-1)+[COLORS[family]] if family == "decoder" else COLORS[family], label="Source perturbation")
            if family == "cross":
                axes[1, column].bar(np.arange(len(values))+0.18, fraction*model["target_deltas"], width=0.36,
                                    color=COLORS["active"], label="Separate target perturbation")
            axes[1, column].set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
            axes[1, column].set(ylabel="L2 displacement", xlabel="Output position / input token")
            axes[1, column].set_ylim(0, max(5, max(model["deltas"][family])*1.15))
            axes[1, column].legend(fontsize=8)
            if frame is not None:
                for position, token in enumerate(labels):
                    axes[1, column].text(position, 4.7, "●", color=COLORS["masked"] if family == "decoder" and position < len(values)-1 else COLORS["active"] if position == len(values)-1 else COLORS[family], ha="center", fontsize=12)
        fig.suptitle("Final-token intervention: measured hidden-state displacement", fontsize=16)
        if frame is not None:
            fig.text(0.5, 0.91, f"Display interpolation: {fraction:.0%} of endpoint displacement", ha="center", color=COLORS["active"])
    caption(fig, "Endpoints are actual CPU float64 forwards, +10 in last input feature 0. Animation interpolates endpoints, not intermediate model runs. Availability permits influence.")


def geometry_frame(fig, model, frame=0):
    arrays = [model["embeddings"][f] for f in FAMILIES] + [model["baseline"][f] for f in FAMILIES] + [model["perturbed"][f] for f in FAMILIES]
    pca = shared_pca(arrays)
    coordinates = pca["coordinates"]
    for i, family in enumerate(FAMILIES):
        ax = fig.add_subplot(1, 3, i+1, projection="3d")
        a, b, c = coordinates[i], coordinates[i+3], coordinates[i+6]
        fraction = frame/5
        current = a+(b-a)*fraction if frame <= 5 else b+(c-b)*(frame-5)/5
        ax.scatter(*a.T, color=COLORS["masked"], s=25, label="Input embedding")
        ax.scatter(*current.T, color=COLORS[family] if frame <= 5 else COLORS["active"], s=35, label="Display state")
        for start, end in zip(a if frame <= 5 else b, current):
            ax.plot(*np.stack([start, end]).T, color=COLORS[family], alpha=0.5)
        all_xyz = np.concatenate(coordinates)
        for axis in "xyz":
            j = "xyz".index(axis)
            getattr(ax, f"set_{axis}lim")(all_xyz[:, j].min()-0.3, all_xyz[:, j].max()+0.3)
        ax.set(xlabel="PC1", ylabel="PC2", zlabel="PC3", title=NAMES[family])
        ax.legend(fontsize=7)
    fig.suptitle(f"Shared PCA · first 3 PCs retain {sum(pca['explained_ratio']):.1%} of fitted variance", fontsize=15)
    caption(fig, "One PCA basis for all inputs and endpoint states. Frames interpolate projected endpoints; distances can shrink. Random clusters have no semantic interpretation.")


def objective(fig, data, name, frame=0):
    item = data[name]
    selected = item["selected"] if name == "mlm" else list(range(len(item["labels"])))
    index = selected[frame % len(selected)]
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.8], width_ratios=[1.4, 1])
    ax = fig.add_subplot(grid[0, :])
    probability_axis = fig.add_subplot(grid[1, 0])
    mask_axis = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    def row(label, tokens, y, chosen=None):
        ax.text(-1.1, y, label, fontsize=9)
        for j, token in enumerate(tokens):
            ax.text(j, y, token, ha="center", bbox={"boxstyle": "round", "facecolor": COLORS["active"] if j == chosen else "#e8eef5", "edgecolor": "white"}, fontsize=9)
    row("Original", data["original"], 3)
    if name == "span":
        row("Encoder", data["span_source"], 2.1)
    row("Input", item["inputs"], 1.2, index)
    row("Label", item["labels"], 0.3, index)
    ax.set(xlim=(-1.2, max(len(item["inputs"]), 7)), ylim=(-0.5, 3.8))
    probabilities = item["probabilities"][index]
    colors = [COLORS["active"] if token == item["labels"][index] else COLORS[{"mlm":"encoder","causal":"decoder","span":"cross"}[name]] for token in data["vocab"]]
    probability_axis.bar(range(len(probabilities)), probabilities, color=colors)
    probability_axis.set_xticks(range(len(probabilities)), data["vocab"], rotation=40, ha="right", fontsize=7)
    probability_axis.set(ylabel="Toy prediction probability", xlabel="Vocabulary token", ylim=(0, 1),
                title=f"Predict position {index}; true label = {item['labels'][index]}; NLL = {item['losses'][index]:.4f}")
    if name == "causal":
        probability_axis.plot(range(len(probabilities)), data["unmasked"]["probabilities"][index], "o--",
                              color=COLORS["active"], ms=3,
                              label=f"Unmasked control; NLL {data['unmasked']['losses'][index]:.3f}")
        probability_axis.legend(fontsize=7)
    mask = visibility(len(item["inputs"]), len(item["inputs"]), causal=name != "mlm")
    heatmap(mask_axis, mask, item["inputs"], item["inputs"], "Bidirectional context" if name == "mlm" else "Causal visibility · all training rows in parallel", probability=True)
    mask_axis.images[0].set_cmap(ListedColormap([COLORS["masked"], COLORS["encoder"] if name == "mlm" else COLORS["decoder"]]))
    mask_axis.axhspan(index-0.45, index+0.45, fill=False, edgecolor=COLORS["active"], lw=2)
    if name == "causal" and index+1 < len(item["inputs"]):
        mask_axis.text(index+1, index, "LEAK\nif unmasked", ha="center", va="center", fontsize=6, color="black")
    titles = {"mlm": "MLM: corruption is distinct from bidirectional attention",
              "causal": "CLM: shifted labels + causal mask allow parallel training",
              "span": "Span corruption: sentinels delimit the missing spans"}
    fig.suptitle(titles[name], fontsize=15)
    notes = {"mlm": "Selected positions 1/3/5 force mask/random/unchanged cases for display; original BERT uses 80/10/10 among selected positions. Loss only on selected positions.",
             "causal": "Position i predicts input i+1. Removing the causal mask exposes the label at the next input position; inference must append unknown tokens sequentially.",
             "span": "Two manually selected spans; distinct sentinels in encoder input and reconstruction target. Causal decoder reads the source. All probabilities are from random toy projections."}
    caption(fig, notes[name]+" No pretrained model predictions.")


def objectives_comparison(fig, data):
    axes = fig.subplots(1, 3)
    for ax, name, color in zip(axes, ["mlm", "causal", "span"], [COLORS[f] for f in FAMILIES]):
        item = data[name]
        losses = item["losses"]
        indices = item["selected"] if name == "mlm" else list(range(len(losses)))
        ax.bar(range(len(losses)), [losses[i] if i in indices else 0 for i in range(len(losses))], color=color)
        ax.set(xlabel="Prediction position", ylabel="Token NLL (nats)", title=f"{name.upper()}: mean selected NLL {np.mean(losses[indices]):.3f}")
    fig.suptitle("Different prediction targets · toy computed losses", fontsize=16)
    caption(fig, "Different conditioning and target populations: loss magnitudes do not rank objectives. Unselected MLM positions contribute zero to the displayed objective.")


def generation_frame(fig, data, frame):
    run = next(r for r in data["runs"] if r["mode"] == "sampling" and r["temperature"] == 1)
    step = run["steps"][min(frame, len(run["steps"])-1)]
    axes = fig.subplots(1, 2)
    for ax, key, title in zip(axes, ["logits", "probabilities"], ["Next-token logits", "Temperature-softmax probabilities"]):
        ax.bar(range(len(data["vocab"])), step[key], color=[COLORS["active"] if i == step["chosen"] else COLORS["decoder"] for i in range(len(data["vocab"]))])
        ax.set_xticks(range(len(data["vocab"])), data["vocab"], rotation=30, ha="right")
        ax.set(title=title, xlabel="Vocabulary token", ylabel="Logit" if key == "logits" else "Probability")
    fig.suptitle("Autoregressive toy generation · context: " + " ".join(step["context"]), fontsize=14)
    fig.text(0.5, 0.9, "Select → append: " + step["token"], ha="center", color=COLORS["active"])
    caption(fig, "Fixed random bigram logits, seeded sampling, temperature 1. Generation temperature rescales vocabulary logits; it is separate from attention's 1/√d scaling.")


def cache_figure(fig, data, frame=None):
    axes = fig.subplots(1, 3)
    stop = len(data["lengths"]) if frame is None else min(frame+1, len(data["lengths"]))
    lengths = data["lengths"][:stop]
    axes[0].plot(lengths, data["uncached_projection_vectors"][:stop], "o-", color=COLORS["active"], label="Recompute prefix K/V")
    axes[0].plot(lengths, data["cached_projection_vectors"][:stop], "o-", color=COLORS["decoder"] ,label="Prefill + append K/V")
    axes[0].set(xlabel="Available prefix length", ylabel="Cumulative projected K/V token-vectors", title="Projection work count")
    axes[0].legend(fontsize=8)
    axes[1].plot(lengths, data["cache_bytes"][:stop], "o-", color=COLORS["cross"])
    axes[1].set(xlabel="Cached tokens", ylabel="Bytes", title="One-layer / one-head float64 cache")
    length = int(lengths[-1])
    axes[2].imshow(np.ones((2, length)), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    prefill = length == data["prompt"]
    axes[2].set(yticks=[0, 1], yticklabels=["K", "V"], xlabel="Absolute token position", title="Prefill: create all prompt K/V" if prefill else f"Reuse entries 0…{length-2}; append {length-1}")
    axes[2].axvspan(-0.5 if prefill else length-1.5, length-0.5, color=COLORS["active"], alpha=0.7)
    fig.suptitle("KV cache: prefill, reuse and append"+f" · output max error {data['max_abs_error']:.2e}", fontsize=15)
    caption(fig, "Counts are projected token-vectors, not FLOPs or measured speed. Absolute positional inputs are identical. Each new query still attends to the entire available prefix.")


def complexity(fig, log=False):
    axes = fig.subplots(1, 2)
    lengths = np.arange(1, 4097, 64)
    for ax, vary in zip(axes, ["source", "target"]):
        data = connection_counts(lengths, 256) if vary == "source" else connection_counts(1024, lengths)
        for family in FAMILIES:
            ax.plot(lengths, data[family], label=NAMES[family], color=COLORS[family])
        ax.set(xlabel=f"{vary.title()} length", ylabel="Permitted connections", title="Target m=256" if vary == "source" else "Source n=1024")
        if log:
            ax.set_yscale("log")
            ax.set_xscale("log")
        ax.legend(fontsize=8)
    fig.suptitle("Idealized dense attention connectivity" + (" · log scales" if log else ""), fontsize=16)
    caption(fig, "One instance of each attention type; not FLOPs, latency or an equivalent task comparison. Encoder-only does not generate targets. Dense kernels may still compute blocked scores.")


def rag_figure(fig, data, frame=None):
    axes = fig.subplots(2, 1, height_ratios=[1, 1.2])
    labels = ["User query", "Bi-encoder\nindependent vectors", "Vector retrieval", "Candidates", "Cross-encoder\ntoy pair scoring", "Generative LLM\nconceptual", "Response checks\nconceptual"]
    ax = axes[0]
    ax.axis("off")
    for i, label in enumerate(labels):
        color = COLORS["active"] if frame == i else COLORS["encoder"] if i in (1, 2) else COLORS["decoder"] if i == 5 else COLORS["cross"] if i == 4 else COLORS["masked"]
        ax.text(i, 0, label, ha="center", va="center", fontsize=8, bbox={"boxstyle":"round,pad=0.5","facecolor":color,"edgecolor":"white"}, color="white")
        if i:
            ax.annotate("", xy=(i-0.35, 0), xytext=(i-0.65, 0), arrowprops={"arrowstyle":"->"})
    ax.set(xlim=(-0.6, 6.6), ylim=(-1, 1), title="Query: " + data["query"])
    candidates = data["retrieved"]
    axes[1].bar(np.arange(len(candidates))-0.18, data["cosine"][candidates], width=0.36, color=COLORS["encoder"], label="Independent cosine")
    axes[1].bar(np.arange(len(candidates))+0.18, data["joint_scores"][candidates], width=0.36, color=COLORS["cross"], label="Toy joint pair score")
    axes[1].set_xticks(range(len(candidates)), [f"D{i}: {data['documents'][i]}" for i in candidates], rotation=12, ha="right", fontsize=8)
    axes[1].set(ylabel="Score (different definitions)", xlabel="Retrieved candidate order", title="Reranked IDs: " + " → ".join(f"D{i}" for i in data["reranked"]))
    axes[1].legend()
    fig.suptitle("RAG uses distinct retrieval, pair scoring and generation roles", fontsize=16)
    caption(fig, "Lexical vectors simulate independent encodings; pair scores are a toy rule. All six fixture scores are precomputed for offline controls. No trained reranker, LLM or response checks run.")
