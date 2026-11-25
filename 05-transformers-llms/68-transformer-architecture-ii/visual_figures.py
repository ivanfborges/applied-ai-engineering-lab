"""Matplotlib/Pillow figures using actual traced tensors and labeled controls."""

from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image

from from_scratch import MultiHeadAttention
from visual_core import (HEAD_COLORS, TOKENS, architecture_counts, constructed_routing,
                         default_case, diversity_case, kv_bytes, online_attention,
                         scaling_samples, subspace_cases)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 12, "axes.labelsize": 10, "figure.facecolor": "#f4f7fa",
                     "axes.spines.top": False, "axes.spines.right": False})


def heat(ax, matrix, title, *, probability=False, annotate=True, colorbar=False):
    matrix = np.asarray(matrix)
    finite = matrix[np.isfinite(matrix)]
    bound = max(float(np.max(abs(finite))), 1e-9) if finite.size else 1
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="viridis" if probability else "RdBu_r",
                      vmin=0 if probability else -bound, vmax=1 if probability else bound,
                      aspect="auto", interpolation="nearest")
    ax.set_title(title, pad=12)
    ax.set_xlabel("Key / feature column")
    ax.set_ylabel("Query / token row")
    if annotate and matrix.size <= 144:
        for (i, j), value in np.ndenumerate(matrix):
            text = "" if np.isnan(value) else "−∞" if np.isneginf(value) else f"{value:.2f}"
            dark_cell = value < .6 if probability else np.isfinite(value) and abs(value) > bound * .52
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if dark_cell else "#132334")
    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=.035, pad=.03)
    return image


def footer(fig, text):
    fig.text(.04, .025, text, fontsize=9, color="#42566b", wrap=True)


def export_animation(folder, stem, count, draw, caption, *, dpi=90, duration=1200):
    """Keep every explanatory stage as a PNG so it can be paused or exported."""
    directory = folder / "frames" / stem
    directory.mkdir(parents=True, exist_ok=True)
    images, paths = [], []
    fig = plt.figure(figsize=(12, 7), dpi=dpi)
    try:
        for i in range(count):
            fig.clear()
            draw(fig, i)
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=dpi)
            buffer.seek(0)
            with Image.open(buffer) as source:
                frame = source.convert("RGB")
            path = directory / f"stage-{i + 1:02d}.png"
            frame.save(path)
            images.append(frame)
            paths.append(path.relative_to(folder).as_posix())
        gif = folder / f"{stem}.gif"
        images[0].save(gif, save_all=True, append_images=images[1:], loop=0,
                       duration=duration, optimize=False, disposal=2)
    finally:
        plt.close(fig)
        for image in images:
            image.close()
    return {"gif": gif.name, "frames": paths, "caption": caption}


def pipeline(folder, seed, quick, dpi):
    _, traces = default_case(seed)
    t = traces["causal"]
    stages = [
        ("Input embeddings", "(B,T,D) = (1,6,16)", t["x"][0], "Each row is a synthetic token representation."),
        ("Q/K/V projections", "3 × (1,6,16)", t["q"][0].transpose(1, 0, 2).reshape(6, 16), "Every projection uses all 16 input features."),
        ("Reshape + transpose", "3 × (1,4,6,4)", t["q"][0, 0], "Packed output columns become four head axes; no raw feature partition."),
        ("Parallel QKᵀ", "(1,4,6,6)", t["raw"][0, 0], "Four independent query-key score matrices, computed in parallel."),
        ("Scale by sqrt(4)", "(1,4,6,6)", t["scaled"][0, 0], "Each head uses its own query/key width, d_head=4."),
        ("Causal mask", "(1,4,6,6)", t["masked"][0, 0], "Future columns are blocked before softmax; −∞ becomes zero probability."),
        ("Row-wise softmax", "(1,4,6,6)", t["attention"][0, 0], "Normalize keys separately for each head and query; rows sum to one."),
        ("Attention × V", "(1,4,6,4)", t["heads"][0, 0], "Each head retrieves a different weighted value representation."),
        ("Concatenate heads", "(1,6,16)", t["concat"][0], "Transpose back to token-major order, preserving every head's channels."),
        ("Output projection", "(1,6,16)", t["output"][0], "W_O mixes all head channels into the model-width output."),
    ]
    def draw(fig, step):
        left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.05, 1.3]})
        left.set_xlim(0, 1); left.set_ylim(0, 11); left.axis("off")
        for j, (name, shape, _, _) in enumerate(stages):
            y = 10 - j
            left.add_patch(Rectangle((.03, y - .32), .92, .8, facecolor="#d7edf8" if j == step else "white",
                                     edgecolor="#0072B2" if j == step else "#bccad4", linewidth=2 if j == step else 1))
            left.text(.06, y + .12, f"{j + 1:02d}  {name}", fontsize=10, weight="bold")
            left.text(.06, y - .15, shape, fontsize=9)
            if j < 9:
                left.annotate("", xy=(.49, y - .55), xytext=(.49, y - .32), arrowprops={"arrowstyle": "->", "color": "#5a6d7e"})
            if 2 <= j <= 7:
                for head in range(4):
                    left.plot(.72 + head * .055, y + .12, marker="s", color=HEAD_COLORS[head], markersize=7)
        name, shape, matrix, explanation = stages[step]
        if 2 <= step <= 7:
            right.remove()
            tensors = {2: "q", 3: "raw", 4: "scaled", 5: "masked", 6: "attention", 7: "heads"}
            for head in range(4):
                ax = fig.add_axes([.56 + (head % 2) * .22, .24 + (1 - head // 2) * .35, .16, .19])
                heat(ax, t[tensors[step]][0, head], f"Head {head}", probability=step == 6, annotate=False)
                ax.title.set_color(HEAD_COLORS[head])
                ax.tick_params(labelsize=8); ax.set_xlabel("Key / channel", fontsize=8); ax.set_ylabel("Query", fontsize=8)
            fig.text(.76, .87, f"{name}: four parallel heads\n{shape}", ha="center", fontsize=12)
        else:
            heat(right, matrix, f"{name}: actual values\n{shape}", probability=False)
        fig.suptitle(f"Multi-head attention forward pass · stage {step + 1}/10", fontsize=17, weight="bold")
        footer(fig, explanation + "\nHead colors H0 blue · H1 orange · H2 green · H3 purple. Random, untrained computation.")
        fig.subplots_adjust(bottom=.14, top=.88, wspace=.3)
    return export_animation(folder, "01_multi_head_attention_pipeline", len(stages), draw,
                            "One actual causal forward pass. Colored markers identify parallel heads; all shapes include batch.", dpi=dpi)


def subspaces(folder, seed, quick, dpi):
    cases = subspace_cases(seed)
    sequence = [(mode, head, kind) for mode in ("direct", "pca") for head in range(4) for kind in (("Q",) if quick else ("Q", "K", "V"))]
    def draw(fig, step):
        mode, head, kind = sequence[step]
        case = cases[mode]
        entry = next(e for e in case["entries"] if e["head"] == head and e["kind"] == kind)
        for col, (points, title) in enumerate(((case["original"], "Input tokens"), (entry["view"], f"Head {head} {kind}: X @ W"))):
            ax = fig.add_subplot(1, 2, col + 1, projection="3d")
            for i, point in enumerate(points):
                ax.scatter(*point, color=HEAD_COLORS[i % len(HEAD_COLORS)], s=50)
                ax.text(*point, f" {i}:{TOKENS[i]}", fontsize=8)
            ax.set_title(title + (" (separate PCA)" if mode == "pca" else " (direct 3D)"))
            prefix = "PC" if mode == "pca" else "coordinate"
            ax.set_xlabel(prefix + " 1"); ax.set_ylabel(prefix + " 2"); ax.set_zlabel(prefix + " 3")
            ax.view_init(elev=22, azim=-55 + head * 5)
        fig.suptitle("Full-input projections into different coordinate spaces", fontsize=17, weight="bold")
        footer(fig, f"Exact token 0↔1 distance: {case['distances'][0,1]:.3f} → {entry['distances'][0,1]:.3f}. "
                    f"Projected variance shown: {entry['retained']:.1%}.\n" +
                    ("Separately fitted PCA: plotted cross-space distances and axes are NOT comparable." if mode == "pca" else
                     "Constructed random 3→3 maps illustrate geometry; they are not a D/H head-width configuration.") +
                    " Random projections have no learned semantics.")
        fig.subplots_adjust(bottom=.18, top=.85, left=.02, right=.95)
    return export_animation(folder, "02_representation_subspaces", len(sequence), draw,
                            "Direct 3D maps followed by separately fitted PCA. Token colors persist; exact distances use full coordinates.", dpi=dpi)


def matrices(folder, seed, quick, dpi):
    rng = np.random.default_rng(seed + 4)
    x = rng.normal(size=(1, 4, 8))
    layer = MultiHeadAttention(8, 2, seed)
    t = layer.forward(x, causal=True, return_trace=True)[2]
    projections = [(x[0], w, x[0] @ w, f"X @ W_{name}", "(4,8) × (8,8) → (4,8)")
                   for name, w in (("Q", layer.wq), ("K", layer.wk), ("V", layer.wv))]
    stages = projections + [
        (x[0], layer.wq[:, :4], t["q"][0, 0], "Separate head 0 projection = packed columns 0:4", "(4,8) × (8,4) → (4,4)"),
        (t["q"][0, 0], t["k"][0, 0].T, t["raw"][0, 0], "Head 0 Q @ Kᵀ", "(4,4) × (4,4) → (4,4)"),
        (t["raw"][0, 0], np.full((1, 1), .5), t["scaled"][0, 0], "Divide raw scores by sqrt(d_head)=2", "(1,2,4,4) remains unchanged in shape"),
        (t["scaled"][0, 0], t["blocked"][0, 0].astype(float), t["masked"][0, 0], "Apply True=blocked causal mask", "(1,2,4,4) · inaccessible entries become −∞"),
        (t["masked"][0, 0], np.ones((4, 1)), t["attention"][0, 0], "Softmax across KEYS, not heads or queries", "Each probability row sums to 1"),
        (t["attention"][0, 0], t["v"][0, 0], t["heads"][0, 0], "Head 0 attention @ V", "(4,4) × (4,4) → (4,4)"),
        (t["heads"][0, 0], t["heads"][0, 1], t["concat"][0], "Concatenate head outputs after transpose", "(4,4) + (4,4) → (4,8)"),
        (t["concat"][0], layer.wo, t["output"][0], "Concatenated representation @ W_O", "(4,8) × (8,8) → (4,8)"),
    ]
    def draw(fig, step):
        a, b, c, title, dimensions = stages[step]
        axes = fig.subplots(1, 3)
        heat(axes[0], a, f"Left input {a.shape}")
        heat(axes[1], b, f"Right input / mask {b.shape}")
        heat(axes[2], c, f"Result {c.shape}", probability=step == 7)
        if step in (0, 1, 2, 3, 4, 8, 10):
            row, column = 1, min(1, c.shape[1] - 1)
            axes[0].add_patch(Rectangle((-.5, row - .5), a.shape[1], 1, fill=False, edgecolor="#E69F00", linewidth=3))
            axes[1].add_patch(Rectangle((column - .5, -.5), 1, b.shape[0], fill=False, edgecolor="#E69F00", linewidth=3))
            axes[2].add_patch(Rectangle((column - .5, row - .5), 1, 1, fill=False, edgecolor="#009E73", linewidth=3))
            detail = f"Selected dot product: row {row} · column {column} = {a[row] @ b[:,column]:.6f}."
        else:
            detail = "All displayed values are computed; blank cells show blocked −∞ logits."
        fig.suptitle(f"{step + 1:02d} · {title}", fontsize=15, weight="bold")
        footer(fig, dimensions + "\n" + detail + " B=1, T=4, D=8, H=2. Random untrained weights; no bias/dropout.")
        fig.subplots_adjust(bottom=.2, top=.82, wspace=.4)
    return export_animation(folder, "04_attention_matrix_operations", len(stages), draw,
                            "Actual 4-token, 8-feature, 2-head arithmetic. Highlighted row/column terms compute the selected result cell.", dpi=dpi)


def attention_png(folder, seed, dpi):
    _, traces = default_case(seed)
    t = traces["causal"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=dpi)
    for h in range(4):
        ax = axes.flat[h]
        heat(ax, t["attention"][0, h], f"Head {h} · probabilities", probability=True, colorbar=True)
        ax.title.set_color(HEAD_COLORS[h])
    heat(axes.flat[4], t["attention"][0].mean(0), "Mean over heads · hides differences", probability=True, colorbar=True)
    heat(axes.flat[5], t["output"][0], "Final output · T×D", annotate=False, colorbar=True)
    fig.suptitle("Causal attention · synthetic vectors, randomly initialized heads", fontsize=16, weight="bold")
    footer(fig, "Labels: 0 The · 1 curious · 2 engineer · 3 checks · 4 every · 5 head. Probabilities share scale [0,1].\nAveraging maps is an inspection operation; the model concatenates head outputs. No learned linguistic semantics.")
    fig.subplots_adjust(bottom=.15, top=.88, hspace=.55, wspace=.4)
    path = folder / "05_attention_heads.png"
    fig.savefig(path, dpi=dpi); plt.close(fig)
    return path.name


def graph(folder, seed, quick, dpi):
    _, traces = default_case(seed)
    t = traces["bidirectional"]
    points = np.column_stack((np.cos(np.arange(6) * 2 * np.pi / 6), np.sin(np.arange(6) * 2 * np.pi / 6)))
    def draw(fig, head):
        left, right = fig.subplots(1, 2)
        query = 2
        for key, weight in enumerate(t["attention"][0, head, query]):
            if key == query:
                left.add_patch(FancyArrowPatch(points[query] + [-.07, .03], points[query] + [.07, .03], connectionstyle="arc3,rad=-2.4", arrowstyle="-|>", mutation_scale=15, color=HEAD_COLORS[head], linewidth=1+10*weight, alpha=.3+.7*weight))
            else:
                left.add_patch(FancyArrowPatch(points[query], points[key], arrowstyle="-|>", shrinkA=15, shrinkB=15, mutation_scale=18, color=HEAD_COLORS[head], linewidth=1+10*weight, alpha=.3+.7*weight))
        for i, point in enumerate(points):
            left.scatter(*point, s=280, color=HEAD_COLORS[i], zorder=3)
            left.text(point[0], point[1] - .2, f"{i}: {TOKENS[i]}", ha="center")
        left.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal", title=f"Head {head}: query 2 → key")
        left.axis("off")
        right.bar(np.arange(6), t["attention"][0, head, query], color=HEAD_COLORS[head])
        right.set(ylim=(0, 1), xlabel="Candidate key", ylabel="Probability", title="Exact routing weights")
        fig.suptitle("One query, multiple directed routing patterns", fontsize=17, weight="bold")
        footer(fig, "Thickness/opacity encode computed attention. Arrows point query→key; value information flows into the query output.\nRandom untrained layer. Edges are not verified linguistic relations or causal explanations.")
        fig.subplots_adjust(bottom=.18, top=.85, wspace=.3)
    return export_animation(folder, "06_token_attention_graph", 4, draw,
                            "Cycle through heads for the same query. Directed arrows encode actual computed attention weights.", dpi=dpi)


def advantage(folder, seed, quick, dpi):
    case = constructed_routing()
    steps = [0., .25, .5, .75, 1.]
    def draw(fig, step):
        alpha = steps[step]
        left, right = fig.subplots(1, 2)
        left.plot(case["shared_curve"][:, 0], case["shared_curve"][:, 1], color="#7b8b98", label="Single shared distribution")
        shared = np.array([alpha, 10 * (1 - alpha)])
        left.scatter(*shared, s=110, color=HEAD_COLORS[0], label=f"Shared α={alpha:.2f}")
        left.scatter(*case["routed"], s=150, marker="*", color=HEAD_COLORS[1], label="Two independently routed channels")
        left.set(xlabel="Entity code channel", ylabel="Numeric attribute channel", xlim=(-.1, 1.2), ylim=(-1, 12), title="Reachable representations with fixed value encoding")
        left.legend(fontsize=8, loc="lower left")
        right.bar([-.18, .82], shared, width=.35, label="Shared weighted representation", color=HEAD_COLORS[0])
        right.bar([.18, 1.18], case["routed"], width=.35, label="Head 0 entity + head 1 attribute", color=HEAD_COLORS[1])
        right.set(xticks=[0, 1], xticklabels=["Entity channel", "Numeric channel"], ylim=(0, 11), title="Concatenation preserves different retrievals")
        right.legend(fontsize=8)
        fig.suptitle("Constructed advantage: separate retrieval distributions", fontsize=17, weight="bold")
        footer(fig, "Sources: entity=(1,0), attribute=(0,10); extra position/context fields identify the two sources.\nManual logits [6,−6] and [−6,6] route distinct scalar channels. Restricted fixed-value comparison; not a universal MHA superiority proof.")
        fig.subplots_adjust(bottom=.2, top=.85, wspace=.3)
    return export_animation(folder, "07_single_vs_multi_head", len(steps), draw,
                            "Constructed two-source payload example. A shared scalar mixture traces a line; independent channels retrieve both selected attributes.", dpi=dpi)


def scaling(folder, seed, quick, dpi):
    samples = scaling_samples(seed, 1000 if quick else 4000)
    def draw(fig, index):
        sample = samples[index]
        axes = fig.subplots(2, 2)
        axes[0, 0].hist(sample["raw"], bins=40, density=True, color=HEAD_COLORS[0], alpha=.65, label="raw")
        axes[0, 0].hist(sample["scaled"], bins=40, density=True, color=HEAD_COLORS[1], alpha=.65, label="scaled")
        axes[0, 0].set(title="Same random components, paired comparison", xlabel="Dot product", ylabel="Density"); axes[0, 0].legend()
        raw = sample["logits"]
        unscaled = np.exp(raw - raw.max()); unscaled /= unscaled.sum()
        scaled = np.exp((raw - raw.max()) / np.sqrt(sample["dimension"])); scaled /= scaled.sum()
        for ax, probabilities, title in ((axes[0, 1], unscaled, "Unscaled six-key row"), (axes[1, 1], scaled, "Scaled six-key row")):
            ax.bar(range(6), probabilities, color=HEAD_COLORS[0 if title.startswith("Un") else 1])
            entropy = -np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))
            ax.set(title=f"{title} · max={probabilities.max():.3f}, H={entropy:.3f}", ylim=(0, 1), xlabel="Key", ylabel="Probability")
        dimensions = [s["dimension"] for s in samples]
        axes[1, 0].plot(dimensions, [s["raw_variance"] for s in samples], "o-", label="empirical raw")
        axes[1, 0].plot(dimensions, dimensions, "--", label="expected d")
        axes[1, 0].plot(dimensions, [s["scaled_variance"] for s in samples], "o-", label="empirical scaled")
        axes[1, 0].axvline(sample["dimension"], color="#888888", linestyle=":")
        axes[1, 0].set(xlabel="Head dimension", ylabel="Variance", title="Raw variance ∝ d; scaled ≈1"); axes[1, 0].legend(fontsize=8)
        fig.suptitle(f"Scaling attention logits · d_head={sample['dimension']}", fontsize=17, weight="bold")
        footer(fig, "Nested paired IID N(0,1) coordinates. Empirical variance is a Monte Carlo measurement. Entropy is in nats.\nScaling counters dimension-dependent saturation; it does not force uniform attention or prove task quality.")
        fig.subplots_adjust(bottom=.17, top=.85, hspace=.65, wspace=.3)
    return export_animation(folder, "08_attention_scaling", len(samples), draw,
                            "Empirical raw/scaled variance and actual softmax probabilities for paired random coordinates.", dpi=dpi)


def causal(folder, seed, quick, dpi):
    layer, traces = default_case(seed)
    original = traces["causal"]["x"]
    changed = original.copy(); changed[:, 4] += np.random.default_rng(seed + 3).normal(size=16) * 5
    control = {mode: layer.forward(changed, causal=mode == "causal")[0] for mode in traces}
    def draw(fig, step):
        axes = fig.subplots(2, 3)
        prefix = step + 1
        full, masked = traces["bidirectional"], traces["causal"]
        # Recompute on the available prefix: visible probability rows must normalize
        # over available keys, rather than hiding columns of a full-context softmax.
        prefix_traces = {mode: layer.forward(original[:, :prefix], causal=mode == "causal", return_trace=True)[2]
                         for mode in traces}
        def available_matrix(values):
            padded = np.full((6, 6), np.nan)
            padded[:prefix, :prefix] = values
            return padded
        heat(axes[0, 0], full["raw"][0, 0], "Original raw QKᵀ · head 0")
        heat(axes[0, 1], masked["blocked"][0, 0].astype(float), "Causal mask · 1 = blocked", probability=True)
        heat(axes[0, 2], available_matrix(prefix_traces["causal"]["masked"][0, 0]), "Current-prefix masked logits · future = −∞")
        for ax, mode, title in ((axes[1, 0], "bidirectional", "Bidirectional"), (axes[1, 1], "causal", "Causal")):
            heat(ax, available_matrix(prefix_traces[mode]["attention"][0, 0]), title + f" probabilities · reveal {prefix}/6", probability=True)
        differences = [np.max(abs(control[mode][:, :4] - t["output"][:, :4])) for mode, t in traces.items()]
        axes[1, 2].bar(["Bidirectional", "Causal"], differences, color=HEAD_COLORS[:2])
        axes[1, 2].set(title="Perturb token 4: prefix 0–3 change", ylabel="Maximum absolute change")
        fig.suptitle("Causal visibility during progressive token reveal", fontsize=17, weight="bold")
        footer(fig, "Each prefix is actually recomputed; unavailable token rows/columns are blank. This is not trained text generation.\nThe perturbation experiment uses the full sequence. Future influence vanishes in this layer under the correct mask.")
        fig.subplots_adjust(bottom=.18, top=.86, hspace=.65, wspace=.5)
    return export_animation(folder, "10_causal_attention", 6, draw,
                            "Raw logits, mask and probabilities with progressive reveal, plus an executed future-token perturbation control.", dpi=dpi)


def concatenation(folder, seed, quick, dpi):
    layer, traces = default_case(seed)
    t = traces["bidirectional"]
    query = 2
    contributions = np.stack([t["heads"][0, h, query] @ layer.wo[h*4:(h+1)*4] for h in range(4)])
    def draw(fig, feature):
        axes = fig.subplots(2, 2)
        for head in range(4):
            axes[0, 0].bar(np.arange(4) + head * 4, t["heads"][0, head, query], color=HEAD_COLORS[head], label=f"Head {head}")
        axes[0, 0].set(title="Concatenated vector: 4 heads × 4 channels", xlabel="Concatenated feature", ylabel="Activation"); axes[0, 0].legend(fontsize=8)
        heat(axes[0, 1], layer.wo, "W_O (16×16): each row block belongs to a head", annotate=False)
        for h in range(4):
            axes[0, 1].add_patch(Rectangle((-.5, h*4-.5), 16, 4, fill=False, edgecolor=HEAD_COLORS[h], linewidth=2))
        axes[0, 1].axvline(feature, color="#111111", linewidth=2)
        axes[1, 0].bar(range(4), contributions[:, feature], color=HEAD_COLORS[:4])
        axes[1, 0].axhline(t["output"][0, query, feature], color="#111111", linestyle="--", label="Sum = final feature")
        axes[1, 0].set(title=f"Output feature {feature}: per-head contributions", xlabel="Head", ylabel="Contribution"); axes[1, 0].legend(fontsize=8)
        heat(axes[1, 1], contributions, "Head contributions across all output features", annotate=False)
        mean = t["heads"][0, :, query].mean(0)
        fig.suptitle("Concatenation + W_O mixes features across heads", fontsize=17, weight="bold")
        footer(fig, f"Query 2. Feature {feature}: sum_i (Z_i @ W_O_i) = {contributions[:,feature].sum():.6f}.\n"
                    f"Simply averaging head outputs gives only {len(mean)} channels, not the 16D projected output. Random untrained layer.")
        fig.subplots_adjust(bottom=.18, top=.85, hspace=.6, wspace=.4)
    return export_animation(folder, "11_head_concatenation", 4 if quick else 8, draw,
                            "Track color-coded head channels through actual W_O row blocks; inspect the sum of head contributions for selected output features.", dpi=dpi)


def sharing(folder, seed, quick, dpi):
    settings = [8, 4, 2, 1]
    def draw(fig, step):
        kv_heads = settings[step]
        c = architecture_counts(8, kv_heads, 64)
        left, right = fig.subplots(1, 2)
        for head in range(8):
            target = c["assignment"][head]
            y = target * 7 / max(1, kv_heads - 1)
            left.plot([0, 1], [head, y], color=HEAD_COLORS[target % 8], linewidth=2)
            left.scatter([0, 1], [head, y], color=HEAD_COLORS[target % 8], s=90)
            left.text(-.1, head, f"Q{head}", ha="right")
        for target in range(kv_heads):
            left.text(1.1, target * 7 / max(1, kv_heads - 1), f"K{target} / V{target}")
        left.set(xlim=(-.4, 1.8), ylim=(-1, 8), title="Same 8 query heads; different K/V sharing"); left.axis("off")
        sizes = [kv_bytes(1, 32, 8192, n, 64) / 2**30 for n in settings]
        right.bar([str(n) for n in settings], sizes, color=[HEAD_COLORS[1] if n == kv_heads else "#aabcca" for n in settings])
        right.set(xlabel="KV heads", ylabel="Theoretical GiB", title="Cache: B=1, L=32, T=8192, FP16")
        fig.suptitle(f"{'MHA' if kv_heads == 8 else 'MQA' if kv_heads == 1 else 'GQA'} · 8 query heads / {kv_heads} KV heads", fontsize=17, weight="bold")
        footer(fig, f"D=512, d_head=64. Bias-free Q/K/V/O parameters: {c['parameters']:,}.\nArchitectural counts only; no trained conversion, quality equivalence, latency or GPU memory measurement.")
        fig.subplots_adjust(bottom=.18, top=.85, wspace=.3)
    return export_animation(folder, "13_mha_gqa_mqa_comparison", len(settings), draw,
                            "Architectural transition from independent KV heads to groups and one shared KV head. Payload/parameter counts are theoretical.", dpi=dpi)


def flash(folder, seed, quick, dpi):
    _, traces = default_case(seed)
    t = traces["bidirectional"]
    output, stages = online_attention(t["q"][0, 0], t["k"][0, 0], t["v"][0, 0])
    def draw(fig, step):
        s = stages[step]
        axes = fig.subplots(1, 3)
        positions = np.zeros((6, 6)); positions[:, s["start"]:s["end"]] = 1
        heat(axes[0], positions, f"Current key block {s['start']}:{s['end']}", probability=True)
        heat(axes[1], np.column_stack((s["maximum"], s["denominator"])), "Online max m and normalizer l (6×2)")
        heat(axes[2], s["partial_output"], "Partial normalized context (6×4)")
        error = np.max(abs(output - t["heads"][0, 0]))
        fig.suptitle("Tiled online softmax · educational FlashAttention concept", fontsize=17, weight="bold")
        footer(fig, "Update m'=max(m,max(S_block)); rescale old l and value accumulator by exp(m−m'). Then add current block.\n"
                    f"Final tiled-vs-dense max error: {error:.3e}. All dense pairs are still visited. This CPU demonstration is not a GPU kernel or speed benchmark.")
        fig.subplots_adjust(bottom=.22, top=.82, wspace=.45)
    return export_animation(folder, "15_flash_attention_concept", len(stages), draw,
                            "Executed blockwise online-softmax accumulator for one actual head. Dense equivalence is measured; GPU execution is conceptual only.", dpi=dpi)


ANIMATIONS = {"architecture": pipeline, "subspaces": subspaces, "matrices": matrices,
              "graph": graph, "single-vs-multi": advantage, "scaling": scaling,
              "causal": causal, "concatenation": concatenation, "sharing": sharing, "flash": flash}
