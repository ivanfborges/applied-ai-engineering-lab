"""Scientific PNG/GIF views; bounded real frames, shared coordinates, no training."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import torch
from scipy.special import erf

from from_scratch import sinusoidal_positions
from transformer_lab import COLORS, capture_attention, shared_pca, memory_bytes

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                     "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False})


def heading(fig, title, classification, footer="Synthetic educational example · author interpretation review required"):
    fig.suptitle(title, fontsize=18, fontweight="bold", y=.98, color="#172033")
    fig.text(.5, .928, classification, ha="center", fontsize=11, color=COLORS["muted"])
    fig.text(.5, .025, footer, ha="center", fontsize=10, color=COLORS["muted"])


def layout(fig):
    fig.tight_layout(rect=(.015, .055, .985, .90), h_pad=2, w_pad=2)


def png(path, draw, dpi=110, figsize=(13, 7.5)):
    fig = plt.figure(figsize=figsize)
    try:
        draw(fig, 0)
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


def gif(path, draw, frames, fps=2, dpi=90, figsize=(13, 7.5)):
    if not frames or fps <= 0 or dpi < 40:
        raise ValueError("animations require frames, positive fps and dpi >= 40")
    fig = plt.figure(figsize=figsize)
    try:
        def update(index):
            fig.clear()
            draw(fig, index)
            return []
        animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=True)
        animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(fig)


def heat(ax, values, title, query=None, labels_x=None, labels_y=None, probability=False, annotate=False):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    limit = max(float(abs(finite).max()), 1e-6) if finite.size else 1
    color_map = plt.get_cmap("viridis" if probability else "RdBu_r").with_extremes(bad="#cbd5e1")
    image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=color_map,
                      vmin=0 if probability else -limit, vmax=1 if probability else limit)
    ax.set(title=title, xlabel="Key / feature", ylabel="Query / token")
    if labels_x is not None:
        ax.set_xticks(range(len(labels_x)), labels_x, rotation=45, ha="right", fontsize=8)
    if labels_y is not None:
        ax.set_yticks(range(len(labels_y)), labels_y, fontsize=8)
    if query is not None:
        ax.add_patch(Rectangle((-.5, query - .5), values.shape[1], 1, fill=False, ec="#f97316", lw=3))
    if annotate and values.size <= 100:
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, "blocked" if not np.isfinite(value) else f"{value:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if np.isfinite(value) and abs(value) > limit * .6 else "#172033")
    ax.figure.colorbar(image, ax=ax, fraction=.045, pad=.02)


def bars(ax, values, title, color=None, labels=None, limit=None):
    ax.bar(range(len(values)), values, color=color or COLORS["attention"])
    ax.set(title=title, xlabel="Feature / token ID", ylabel="Value")
    if labels:
        ax.set_xticks(range(len(values)), labels, rotation=30, ha="right", fontsize=8)
    if limit:
        ax.set_ylim(*limit)


def token_graph(ax, row, labels, query):
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    for key, weight in enumerate(row):
        if weight > 0:
            ax.add_patch(FancyArrowPatch(points[query], points[key], arrowstyle="->",
                         mutation_scale=12, lw=.4 + 8 * weight, alpha=.15 + .85 * weight,
                         color=COLORS["attention"], connectionstyle="arc3,rad=.12"))
    for i, (point, label) in enumerate(zip(points, labels)):
        ax.scatter(*point, s=450, c=COLORS["decoder"] if i == query else COLORS["encoder"], zorder=3)
        ax.text(*(point * 1.23), label, ha="center", va="center", fontsize=9)
    ax.set(xlim=(-1.6, 1.6), ylim=(-1.5, 1.5), title=f"Query {query}: edge width = actual probability")
    ax.set_aspect("equal")
    ax.axis("off")


def architecture(folder, config, **style):
    n, m, d, v = config.sequence_length, config.sequence_length + 1, config.d_model, config.vocab_size
    blocks = [("Source token IDs", f"(B,{n})", "encoder"), ("Token embedding", f"(B,{n},{d})", "encoder"),
              ("Add positional information", f"(B,{n},{d})", "encoder"), ("Encoder self-attention", f"(B,{n},{d})", "attention"),
              ("Residual + layer norm", f"(B,{n},{d})", "norm"), ("FFN + residual + norm", f"{d} -> {config.d_ff} -> {d}", "ffn"),
              ("Final encoder memory", f"(B,{n},{d})", "encoder"), ("Shifted target + embedding + position", f"(B,{m},{d})", "decoder"),
              ("Causal self-attention + Add/Norm", f"(B,{m},{d})", "attention"), ("Cross-attention + Add/Norm", f"Q: target; K,V: memory", "attention"),
              ("FFN + residual + norm", f"(B,{m},{d})", "ffn"), ("Vocabulary projection", f"(B,{m},{v})", "decoder"),
              ("Softmax over vocabulary", f"(B,{m},{v})", "decoder"), ("Next-token choice", "append prediction to prefix", "decoder")]

    def draw(fig, frame):
        ax = fig.add_subplot()
        ax.set(xlim=(0, 12), ylim=(-.6, 8.2))
        ax.axis("off")
        for i, (label, shape, kind) in enumerate(blocks):
            column, row = i // 7, i % 7
            x, y = .4 + column * 6.3, 7.25 - row * 1.02
            active = i == frame
            ax.add_patch(FancyBboxPatch((x, y - .64), 4.8, .80, boxstyle="round,pad=.08",
                        facecolor=COLORS[kind] if active else "#e2e8f0", edgecolor=COLORS[kind], lw=3 if active else 1,
                        alpha=1 if i <= frame else .4))
            ax.text(x + 2.4, y - .09, f"{i + 1}. {label}", ha="center", va="center", fontsize=10,
                    color="white" if active else "#172033", weight="bold" if active else "normal")
            ax.text(x + 2.4, y - .40, shape, ha="center", va="center", fontsize=9, color="white" if active else "#475569")
            if row < 6:
                ax.annotate("", (x + 2.4, y - .85), (x + 2.4, y - .70), arrowprops=dict(arrowstyle="->", color="#64748b"))
        ax.annotate("", (6.5, 5.0), (5.35, .90), arrowprops=dict(arrowstyle="->", lw=3, color=COLORS["encoder"], connectionstyle="angle,angleA=0,angleB=90,rad=10"))
        ax.text(5.95, 3.0, "K,V\nfrom final\nencoder\nmemory", ha="center", color=COLORS["encoder"], fontsize=10)
        ax.text(9.1, -.24, "Generated prefix returns to shifted-target input", ha="center", fontsize=9, color=COLORS["decoder"])
        heading(fig, "01 · How do blocks become a conditional sequence model?", "Conceptual illustration · original post-norm topology; stacked layers condensed",
                "Blue encoder · orange decoder · purple attention · green FFN · cyan normalization · Add = identity residual")
        layout(fig)
    gif(folder / "01_transformer_architecture.gif", draw, list(range(14)), **style)


def positional(folder, demo, **style):
    pe = sinusoidal_positions(32, 16)
    cosine = pe @ pe.T / np.outer(np.linalg.norm(pe, axis=1), np.linalg.norm(pe, axis=1))
    def static(fig, _):
        axes = fig.subplots(1, 2)
        heat(axes[0], pe, "Original sinusoidal encoding: 32 positions x 16 features")
        heat(axes[1], cosine, "Cosine similarity between positional vectors", probability=False)
        axes[0].set(xlabel="Encoding dimension", ylabel="Position")
        axes[1].set(xlabel="Position", ylabel="Position")
        heading(fig, "02 · How do features encode position?", "Mathematical visualization · sinusoidal positions, not the model's learned position table")
        layout(fig)
    png(folder / "02_positional_encoding_heatmap.png", static, dpi=style["dpi"])
    def draw(fig, frame):
        axes = fig.subplots(2, 2)
        for dimension in (0, 1, 2, 3, 6, 7):
            axes[0, 0].plot(pe[:, dimension], label=f"feature {dimension}")
        axes[0, 0].axvline(frame, color="black", ls="--")
        axes[0, 0].set(title=f"Different frequencies · active position {frame}", xlabel="Position", ylabel="Encoding value", ylim=(-1.15, 1.15))
        axes[0, 0].legend(ncol=3, fontsize=8)
        bars(axes[0, 1], pe[frame], "Position vector", limit=(-1.2, 1.2))
        heat(axes[1, 0], demo["embeddings"], "Fixed token vectors without positions")
        positioned = demo["embeddings"] + sinusoidal_positions(frame + 5, 8)[frame:frame + 5]
        heat(axes[1, 1], positioned, f"Same vectors + PE at positions {frame}–{frame + 4}")
        heading(fig, "02 · Position changes input; unrestricted attention alone is equivariant", "Mathematical visualization + untrained numerical control",
                f"Permutation control at fixed positions 0–4: error without PE {demo['without_positions_error']:.2e}; with PE {demo['with_positions_change']:.3f} · no semantic claim")
        layout(fig)
    gif(folder / "02_positional_encoding_waves.gif", draw, list(range(0, 32, 2)), **style)


def self_attention(folder, snapshot, words, **style):
    probability = snapshot["probability"][0]
    n = len(words)
    def draw(fig, frame):
        query, operation = divmod(frame, 4)
        axes = fig.subplots(2, 3)
        token_graph(axes[0, 0], probability[query], words, query)
        vectors = np.stack([snapshot[key][0, query] for key in ("q", "k", "v")])
        heat(axes[0, 1], vectors, "Head 0 projected Q / K / V at active token", labels_y=["Q", "K", "V"])
        bars(axes[0, 2], snapshot["raw"][0, query], "Q dot every K (unscaled)")
        bars(axes[1, 0], snapshot["scores"][0, query], "Divide by sqrt(head width)")
        heat(axes[1, 1], probability, "Softmax over keys · head 0", query=query, probability=True)
        bars(axes[1, 2], snapshot["head_output"][0, query], "Sum of weighted value vectors")
        stage = ("Project Q/K/V", "Compute scaled compatibility", "Normalize row", "Aggregate values")[operation]
        heading(fig, f"03 · Encoder self-attention: {stage} · query '{words[query]}'", "Untrained numerical demonstration · sentence labels are aliases for synthetic token IDs",
                "No learned coreference or linguistic relationship is implied · graph and matrix use exactly the same probabilities")
        layout(fig)
    gif(folder / "03_encoder_self_attention.gif", draw, list(range(n * 4)), **style)
    def static(fig, _):
        ax = fig.add_subplot()
        heat(ax, probability, "Actual untrained head 0 attention probabilities", labels_x=words, labels_y=words, probability=True)
        heading(fig, "03 · Which source positions receive each query's weight?", "Untrained numerical demonstration · no linguistic interpretation")
        layout(fig)
    png(folder / "03_encoder_attention_heatmap.png", static, dpi=style["dpi"])


def causal(folder, snapshot, decoder_tokens, **style):
    scores = snapshot["scores"][0]
    n = len(scores)
    blocked = np.triu(np.ones((n, n), dtype=bool), 1)
    masked = np.where(blocked, np.nan, scores)
    from transformer_lab import masked_softmax
    weights, unmasked = masked_softmax(scores, blocked), masked_softmax(scores)
    labels = ["BOS" if value == 0 else str(value) for value in decoder_tokens]
    def draw(fig, query):
        axes = fig.subplots(2, 3)
        heat(axes[0, 0], scores, "Scaled logits before masking", query=query)
        heat(axes[0, 1], blocked.astype(float), "Causal mask: 1 = blocked", query=query, probability=True)
        heat(axes[0, 2], masked, "Masked logits: gray = -infinity", query=query)
        heat(axes[1, 0], unmasked, "Unmasked softmax (same Q/K)", query=query, probability=True)
        heat(axes[1, 1], weights, "Masked softmax: future probability = 0", query=query, probability=True)
        aggregation = weights[query] @ snapshot["v"][0]
        bars(axes[1, 2], aggregation, "Causal weighted value aggregation")
        leaked = labels[query + 1] if query < n - 1 else "terminal slot"
        heading(fig, f"04 · Query {query}: which future labels become visible without masking?", "Numerical demonstration · same projections, different visibility",
                f"Decoder input: {' '.join(labels)} · next input {leaked} contains the current label when present · masked scores are not probabilities")
        layout(fig)
    gif(folder / "04_decoder_causal_attention.gif", draw, list(range(n)), **style)


def cross_attention(folder, trace, initial_trace=None, **style):
    snapshot = next(a for a in trace["attention"] if a["mechanism"] == "cross")
    weights = snapshot["probability"][0]
    source, decoder = list(map(str, trace["source"])), ["BOS" if v == 0 else str(v) for v in trace["decoder_input"]]
    def draw(fig, query):
        axes = fig.subplots(1, 2)
        ax = axes[0]
        n, m = len(source), len(decoder)
        for key, label in enumerate(source):
            x = key / max(1, n - 1)
            ax.scatter(x, 1, s=500, c=COLORS["encoder"])
            ax.text(x, 1.12, label, ha="center")
            for target in range(m):
                if target == query:
                    weight = weights[target, key]
                    ax.plot([x, target / max(1, m - 1)], [1, 0], color=COLORS["attention"], lw=.3 + 10 * weight, alpha=.2 + .8 * weight)
        for i, label in enumerate(decoder):
            x = i / max(1, m - 1)
            ax.scatter(x, 0, s=500, c=COLORS["decoder"] if i == query else "#cbd5e1")
            ax.text(x, -.17, label, ha="center")
        ax.text(.5, 1.4, "K,V: final encoder memory", ha="center", color=COLORS["encoder"])
        ax.text(.5, -.4, "Q: decoder states after self-attention", ha="center", color=COLORS["decoder"])
        ax.set(xlim=(-.2, 1.2), ylim=(-.65, 1.65), title="Actual cross-attention edges · head 0")
        ax.axis("off")
        heat(axes[1], weights, f"Rectangular matrix: {m} queries x {n} keys", query=query, labels_x=source, labels_y=decoder, probability=True, annotate=True)
        heading(fig, "05 · How does each decoder query consult encoder memory?", trace["classification"],
                "Weights are observations, not an alignment target or explanation · correct reversal need not yield an anti-diagonal pattern")
        layout(fig)
    gif(folder / "05_cross_attention.gif", draw, list(range(len(decoder))), **style)
    def static(fig, _):
        axes = fig.subplots(1, 2)
        before = next(a for a in (initial_trace or trace)["attention"] if a["mechanism"] == "cross")["probability"][0]
        heat(axes[0], before, "Initialized checkpoint · head 0", probability=True)
        heat(axes[1], weights, "Displayed checkpoint · head 0", probability=True)
        heading(fig, "05 · What changed in cross-attention after optimization?", trace["classification"], "Same source and shifted targets · no imposed diagonal/anti-diagonal criterion")
        layout(fig)
    png(folder / "05_cross_attention_heatmap.png", static, dpi=style["dpi"])


def multihead(folder, trace, **style):
    snapshot = trace["attention"][0]
    heads = len(snapshot["probability"])
    def static(fig, _):
        columns = min(4, heads)
        axes = fig.subplots(int(np.ceil(heads / columns)), columns, squeeze=False).ravel()
        for h, ax in enumerate(axes):
            if h < heads:
                heat(ax, snapshot["probability"][h], f"Encoder layer 1 · head {h}", probability=True)
            else:
                ax.axis("off")
        heading(fig, "06 · Different projections produce different probability matrices", trace["classification"], "No linguistic roles assigned to heads · head number is an index, not a ranking")
        layout(fig)
    png(folder / "06_attention_heads.png", static, dpi=style["dpi"])
    def draw(fig, frame):
        head = frame % heads
        axes = fig.subplots(2, 4)
        heat(axes[0, 0], snapshot["input"], "Input features shared across projections")
        heat(axes[0, 1], snapshot["q"][head], f"Head {head}: learned Q projection")
        heat(axes[0, 2], snapshot["k"][head], f"Head {head}: learned K projection")
        heat(axes[0, 3], snapshot["v"][head], f"Head {head}: learned V projection")
        heat(axes[1, 0], snapshot["probability"][head], f"Head {head}: scaled row softmax", probability=True)
        heat(axes[1, 1], snapshot["head_output"][head], f"Head {head}: probabilities @ V")
        heat(axes[1, 2], snapshot["concat"], f"Concatenate {heads} head outputs")
        heat(axes[1, 3], snapshot["output"], "Final output projection W_O")
        heading(fig, f"06 · Parallel subspaces, concatenation and output mixing · head {head}", trace["classification"], "Each head uses its own Q/K/V projection; heads do not simply repeat one calculation")
        layout(fig)
    gif(folder / "06_multi_head_attention.gif", draw, list(range(heads * 2)), **style)


def feed_forward(folder, seed=67, **style):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(5, 8))
    w1, w2 = rng.normal(0, .3, (8, 32)), rng.normal(0, .2, (32, 8))
    expanded = x @ w1
    activated = np.maximum(0, expanded)
    output = activated @ w2
    changed = x.copy()
    changed[0] += 2
    invariant = np.max(abs((np.maximum(0, changed @ w1) @ w2)[1:] - output[1:]))
    def draw(fig, frame):
        token, stage = divmod(frame, 4)
        axes = fig.subplots(2, 2)
        for ax, values, title in zip(axes.ravel(), (x[token], expanded[token], activated[token], output[token]),
                                     ("Input: 8 features", "Linear expansion: 32 features", "ReLU: 32 features", "Linear contraction: 8 features")):
            bars(ax, values, title, color=COLORS["ffn"])
        axes.ravel()[stage].set_facecolor("#dcfce7")
        heading(fig, f"07 · Token {token}: same FFN at every position, stage {stage + 1}", "Untrained numerical demonstration · fixed 8 -> 32 -> 8 matrices",
                f"Changing token 0 changes other token outputs by {invariant:.1f} · attention mixes positions; this FFN mixes only features")
        layout(fig)
    gif(folder / "07_feed_forward_network.gif", draw, list(range(12)), **style)
    def static(fig, _):
        axes = fig.subplots(1, 3)
        grid = np.linspace(-4, 4, 200)
        axes[0].plot(grid, np.maximum(0, grid), label="ReLU")
        axes[0].plot(grid, .5 * grid * (1 + erf(grid / np.sqrt(2))), label="GELU")
        axes[0].set(title="Nonlinear activation functions", xlabel="Input", ylabel="Output")
        axes[0].legend()
        heat(axes[1], activated, "ReLU activation values · all tokens")
        gelu = .5 * expanded * (1 + erf(expanded / np.sqrt(2)))
        heat(axes[2], gelu, "GELU on same pre-activations")
        heading(fig, "07 · Nonlinear features differ while weights and inputs stay fixed", "Untrained numerical demonstration · no activation performance comparison")
        layout(fig)
    png(folder / "07_ffn_activations.png", static, dpi=style["dpi"])


def residual(folder, trace, **style):
    values = trace["residuals"]
    limit = max(float(max(abs(row[key][0, :2]).max() for row in values for key in ("input", "update", "sum"))) * 1.4, 1)
    def draw(fig, frame):
        layer, fraction = frame // 5, (frame % 5) / 4
        row = values[layer]
        ax = fig.add_subplot()
        x, update = row["input"][0, :2], row["update"][0, :2] * fraction
        for start, vector, label, color in ((np.zeros(2), x, "x", COLORS["encoder"]),
                  (x, update, "F(x), translated to tip of x", COLORS["residual"]),
                  (np.zeros(2), x + update, "x + F(x)", COLORS["attention"])):
            ax.arrow(*start, *vector, width=.012 * limit, length_includes_head=True, color=color)
            ax.plot([], [], color=color, label=label)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="Feature 0", ylabel="Feature 1",
               title=f"Encoder layer {layer + 1}, token 0: geometric update fraction {fraction:.2f} before layer norm")
        ax.axhline(0, color="#cbd5e1"); ax.axvline(0, color="#cbd5e1")
        ax.legend(loc="upper left")
        ax.set_aspect("equal")
        heading(fig, "08 · A sublayer adds an update instead of replacing its input", trace["classification"], "Geometric interpolation of actual endpoint vectors · fixed axes · derivative contains an identity term; LayerNorm subsequently changes the geometry")
        layout(fig)
    gif(folder / "08_residual_connections.gif", draw, list(range(len(values) * 5)), **style)


def normalization(folder, trace, **style):
    x = trace["residuals"][0]["sum"]
    mean, variance = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    centered, normalized = x - mean, (x - mean) / np.sqrt(variance + 1e-5)
    gamma, beta = np.linspace(.7, 1.3, x.shape[1]), np.linspace(-.2, .2, x.shape[1])
    affine = normalized * gamma + beta
    def draw(fig, token):
        axes = fig.subplots(2, 4)
        for i, (name, values) in enumerate(zip(("Before LN", "Subtract token mean", "Divide by sqrt(var + eps)", "Illustrative affine gamma/beta"),
                                               (x, centered, normalized, affine))):
            bars(axes[0, i], values[token], name, color=COLORS["norm"], limit=(-4, 4))
            axes[1, i].hist(values[token], bins=10, color=COLORS["norm"], range=(-4, 4))
            axes[1, i].set(title=f"mean={values[token].mean():.3f}; std={values[token].std():.3f}", xlabel="Feature value", ylabel="Feature count", xlim=(-4, 4))
        heading(fig, f"09 · Normalize feature values within token {token}, never across tokens", "Model activation + mathematical visualization; affine example is explicitly illustrative",
                "Moments use population variance · epsilon prevents division by zero · affine output need not have zero mean/unit variance")
        layout(fig)
    gif(folder / "09_layer_normalization.gif", draw, list(range(min(5, len(x)))), **style)
    png(folder / "09_activation_distributions.png", draw, dpi=style["dpi"])
    def diagrams(fig, _):
        axes = fig.subplots(1, 2)
        for ax, label, operations in zip(axes, ("Post-norm", "Pre-norm"),
                                        (("x", "F(x)", "x + F(x)", "LayerNorm -> output"), ("x", "LayerNorm(x)", "F(LayerNorm(x))", "x + branch -> output"))):
            ax.axis("off")
            for i, text in enumerate(operations):
                y = .85 - i * .22
                ax.text(.5, y, text, ha="center", va="center", bbox=dict(boxstyle="round", fc="#cffafe", ec=COLORS["norm"]), transform=ax.transAxes)
                if i < 3:
                    ax.annotate("", (.5, y - .14), (.5, y - .05), xycoords="axes fraction", arrowprops=dict(arrowstyle="->"))
            ax.annotate("identity path", (.73, .19 if label == "Pre-norm" else .41), (.73, .83),
                        xycoords="axes fraction", arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.3", color=COLORS["residual"]), fontsize=9)
            ax.set_title(label)
        heading(fig, "09 · Where is normalization relative to the residual addition?", "Conceptual illustration · no measured pre-norm/post-norm training comparison")
        layout(fig)
    png(folder / "09_pre_norm_vs_post_norm.png", diagrams, dpi=style["dpi"])


def representations(folder, trace, projection=None, **style):
    projection = projection or shared_pca(trace["stages"])
    coordinates = projection["coordinates"]
    limits = (coordinates[:, :, :2].min() - .5, coordinates[:, :, :2].max() + .5)
    def draw(fig, frame):
        ax = fig.add_subplot()
        for token in range(coordinates.shape[1]):
            point = coordinates[frame, token]
            ax.plot(coordinates[:frame + 1, token, 0], coordinates[:frame + 1, token, 1], alpha=.45)
            ax.scatter(point[0], point[1], c=[plt.cm.tab10(token % 10)], s=120)
            ax.text(point[0] + .06, point[1] + .06, f"pos {token}: {trace['source'][token]}", fontsize=9)
        ax.set(xlim=limits, ylim=limits, xlabel="Shared PC1", ylabel="Shared PC2", title=projection["names"][frame])
        ax.set_aspect("equal")
        heading(fig, "10 · How do actual token representations move through encoder blocks?", trace["classification"],
                f"One PCA fit across all shown stages · displayed 2D variance: {sum(projection['variance_ratio'][:2]):.1%} · proximity is not guaranteed semantic similarity")
        layout(fig)
    gif(folder / "10_representation_evolution.gif", draw, list(range(len(coordinates))), **style)


def training_inference(folder, trace, row, **style):
    generated = row["generated_examples"][0]
    source = row["source_examples"][0]
    target = row["target_examples"][0]
    distributions = np.array(row["first_probabilities"])
    def draw(fig, frame):
        axes = fig.subplots(2, 3)
        teacher = [0] + target[:-1]
        heat(axes[0, 0], np.triu(np.ones((len(target), len(target))), 1), "Teacher forcing: 1 = blocked future input", probability=True)
        heat(axes[0, 1], trace["logits"], "Parallel teacher-forced vocabulary logits")
        axes[0, 1].set(xlabel="Vocabulary token ID", ylabel="Target slot")
        axes[1, 0].axis("off")
        axes[1, 0].text(.03, .8, f"Source: {source}\nCorrect shifted input:\n{teacher}\nPrediction targets:\n{target}", fontsize=10, va="top", linespacing=2)
        axes[1, 1].axis("off")
        axes[1, 1].text(.03, .8, f"All target positions computed together.\nCausal visibility still blocks future labels.\n\nHeld-out mean cross-entropy:\n{row['teacher_forced_loss']:.4f}\n\nLogits become probabilities via softmax.", fontsize=10, va="top", linespacing=1.8)
        prefix = [0] + generated[:frame]
        bars(axes[0, 2], distributions[frame], f"Inference step {frame + 1}: next token", limit=(0, 1))
        axes[1, 2].axis("off")
        axes[1, 2].text(.03, .8, f"Available generated prefix:\n{prefix}\nNext greedy token: {generated[frame]}\nGenerated so far:\n{generated[:frame + 1]}\nGround truth is unavailable.", fontsize=10, va="top", linespacing=2)
        heading(fig, "11 · Parallel teacher forcing and sequential free-running decoding", "Trained model observation · aggregate CE is from held-out sequences, distributions from first example",
                "BOS=0, EOS=1 · fixed source length + 1 generation horizon; no early EOS stopping · EOS included in metrics")
        layout(fig)
    gif(folder / "11_training_vs_inference.gif", draw, list(range(len(generated))), **style)


def experiments(folder, comparison, **style):
    history = comparison["history"]
    def charts(fig, _):
        axes = fig.subplots(2, 2)
        for name, color in (("masked", COLORS["encoder"]), ("unmasked", COLORS["decoder"])):
            rows = history[name]
            steps = [row["step"] for row in rows]
            axes[0, 0].plot(steps, [row["train_loss"] for row in rows], "o-", label=name, color=color)
            axes[0, 1].plot(steps, [row["teacher_forced_loss"] for row in rows], "o-", label=name, color=color)
            axes[1, 0].plot(steps, [row["autoregressive"]["token_accuracy"] for row in rows], "o-", label=name, color=color)
            axes[1, 1].plot(steps, [row["autoregressive"]["exact_match"] for row in rows], "o-", label=name, color=color)
        for ax, title in zip(axes.ravel(), ("Training CE (eval mode, full training set)", "Held-out teacher-forced CE", "Held-out autoregressive token accuracy", "Held-out autoregressive sequence exact match")):
            ax.set(title=title, xlabel="Optimizer updates", ylabel="Cross-entropy" if "CE" in title else "Fraction correct")
            ax.legend()
            if "accuracy" in title or "match" in title:
                ax.set_ylim(-.03, 1.03)
        heading(fig, "12 · Does missing masking create a teacher-forcing advantage?", "Measured experimental result · paired initialization, data, batches, architecture and budget",
                "Single seed, fixed length · held-out curves inspected descriptively, no checkpoint selection · EOS included; no forced expected outcome")
        layout(fig)
    png(folder / "12_causal_mask_experiment.png", charts, dpi=style["dpi"])
    def leak(fig, frame):
        ax = fig.add_subplot()
        ax.axis("off")
        n = comparison["config"]["sequence_length"]
        row = history["masked"][-1]
        target = row["target_examples"][0]
        decoder = [0] + target[:-1]
        position = frame % n
        ax.text(.5, .90, "Shifted decoder input (ground truth during teacher forcing)", ha="center", fontsize=15)
        for i, token in enumerate(decoder):
            x = .07 + i * .85 / len(decoder)
            color = COLORS["decoder"] if i == position + 1 else COLORS["encoder"]
            ax.text(x, .65, str(token), ha="center", fontsize=18, bbox=dict(fc=color, ec="white", boxstyle="round"), color="white")
            ax.text(x, .48, f"label: {target[i]}", ha="center", fontsize=11)
        x = .07 + position * .85 / len(decoder)
        next_x = .07 + (position + 1) * .85 / len(decoder)
        ax.annotate("Future input contains the answer to this prediction", (next_x, .72), (x, .32),
                    arrowprops=dict(arrowstyle="->", color=COLORS["residual"], lw=3), fontsize=12)
        ax.text(.5, .13, "Correct mask blocks this edge. Unmasked model can exploit it during training.\nAt generation time this future ground-truth token does not exist.", ha="center", fontsize=13, linespacing=1.8)
        heading(fig, "12 · Why is future-target access a shortcut?", "Conceptual illustration with actual synthetic teacher-forcing inputs",
                "Connectivity controls test numerical sensitivity; the paired training chart reports the measured outcome")
        layout(fig)
    gif(folder / "12_target_leakage.gif", leak, list(range(comparison["config"]["sequence_length"])), **style)


def recurrent(folder, **style):
    def draw(fig, frame):
        axes = fig.subplots(3, 1)
        n = 7
        for ax, kind, color in zip(axes, ("RNN", "LSTM", "Transformer encoder"), (COLORS["encoder"], COLORS["ffn"], COLORS["attention"])):
            for token in range(n):
                active = token <= frame if kind != "Transformer encoder" else True
                ax.scatter(token, 0, s=400, c=color if active else "#cbd5e1")
                ax.text(token, -.3, f"token {token}", ha="center", fontsize=9)
                if token < n - 1 and kind != "Transformer encoder":
                    ax.annotate("", (token + .8, .03), (token + .15, .03), arrowprops=dict(arrowstyle="->", color=color))
                    if kind == "LSTM":
                        ax.plot([token, token + 1], [.35, .35], color=color, lw=2)
            if kind == "Transformer encoder":
                for token in range(1, n):
                    ax.add_patch(FancyArrowPatch((0, .1), (token, .1), connectionstyle=f"arc3,rad={-.25 - .03 * token}", arrowstyle="->", color=color))
            ax.set(xlim=(-.5, n - .3), ylim=(-.65, 1.0), title=kind + (" · sequential hidden state (+ cell state)" if kind == "LSTM" else " · sequential hidden state" if kind == "RNN" else " · all source positions available together"))
            ax.axis("off")
        heading(fig, "13 · How long is the computational path between distant tokens?", "Conceptual illustration · recurrent path grows with distance; dense attention connects positions directly",
                "Transformer autoregressive inference remains sequential · no universal latency or hardware speed claim")
        layout(fig)
    gif(folder / "13_transformer_vs_rnn.gif", draw, list(range(7)), **style)


def complexity(folder, config, **style):
    lengths = 2 ** np.arange(7, 16)
    d, ff = config.d_model, config.d_ff
    def draw(fig, frame):
        axes = fig.subplots(1, 2)
        for name, work in (("Recurrent hidden-state work n*d^2", lengths * d * d), ("Encoder pair work n^2*d", lengths ** 2 * d),
                           ("Decoder pair work m^2*d (m=n)", lengths ** 2 * d), ("Cross pair work m*n*d (m=n/2)", lengths ** 2 * d / 2),
                           ("FFN work n*d*d_ff", lengths * d * ff)):
            axes[0].loglog(lengths, work, "o-", label=name)
        axes[0].axvline(lengths[frame], color="black", ls="--")
        axes[0].set(xlabel="Source length n", ylabel="Leading term (arbitrary operation count)", title=f"Theoretical work · d={d}, d_ff={ff}")
        axes[0].legend(fontsize=8)
        for heads, batch, precision in ((1, 1, 4), (4, 1, 4), (4, 4, 2)):
            sizes = [memory_bytes(int(n), int(n), heads, batch, precision)["encoder"] / 2 ** 20 for n in lengths]
            axes[1].loglog(lengths, sizes, "o-", label=f"B={batch}, heads={heads}, FP{precision * 8}")
        axes[1].axvline(lengths[frame], color="black", ls="--")
        axes[1].set(xlabel="Self-attention length", ylabel="MiB for materialized scores only", title="No attention matrices allocated for this estimate")
        axes[1].legend(fontsize=9)
        heading(fig, f"14 · What changes when length reaches {lengths[frame]:,} tokens?", "Theoretical complexity estimate · not measured latency, FLOPs or total process memory",
                "Projection costs omitted from pair terms · fused exact attention can avoid full score storage; dense pair arithmetic remains quadratic")
        layout(fig)
    png(folder / "14_attention_complexity.png", lambda f, _: draw(f, len(lengths) - 1), dpi=style["dpi"])
    gif(folder / "14_sequence_length_scaling.gif", draw, list(range(len(lengths))), **style)


def training_progress(folder, comparison, **style):
    rows = comparison["history"]["masked"]
    def draw(fig, frame):
        axes = fig.subplots(1, 2)
        shown = rows[:frame + 1]
        axes[0].plot([r["step"] for r in shown], [r["train_loss"] for r in shown], "o-", label="Training CE", color=COLORS["encoder"])
        axes[0].plot([r["step"] for r in shown], [r["teacher_forced_loss"] for r in shown], "o-", label="Held-out teacher-forced CE", color=COLORS["decoder"])
        axes[0].set(xlim=(0, comparison["config"]["steps"]), ylim=(0, max(r["train_loss"] for r in rows) * 1.1), xlabel="Update", ylabel="Cross-entropy", title="Fixed-budget training snapshots")
        axes[0].legend()
        bars(axes[1], [shown[-1]["autoregressive"]["token_accuracy"], shown[-1]["autoregressive"]["exact_match"]],
             f"Autoregressive held-out quality · update {shown[-1]['step']}", labels=["Token accuracy", "Exact match"], limit=(0, 1))
        heading(fig, "15 · What did this model actually learn at each checkpoint?", "Measured experimental result · same fixed synthetic held-out split; no checkpoint selection")
        layout(fig)
    gif(folder / "15_training_progress.gif", draw, list(range(len(rows))), **style)
    png(folder / "15_training_metrics.png", lambda f, _: draw(f, len(rows) - 1), dpi=style["dpi"])
    def examples(fig, frame):
        ax = fig.add_subplot()
        ax.axis("off")
        row = rows[frame]
        for i, (source, target, generated) in enumerate(zip(row["source_examples"], row["target_examples"], row["generated_examples"])):
            ax.text(.04, .89 - i * .23, f"Source {source}\nTarget {target}\nGreedy {generated}", fontsize=12, va="top", linespacing=1.6,
                    color=COLORS["ffn"] if target == generated else COLORS["decoder"])
        heading(fig, f"15 · Fixed held-out examples · optimizer update {row['step']}", "Trained model observation" if row["step"] else "Untrained numerical demonstration",
                "First four examples, never selected for success · fixed length + EOS slot; EOS=1 is included in exact match")
        layout(fig)
    gif(folder / "15_generation_examples.gif", examples, list(range(len(rows))), **style)
