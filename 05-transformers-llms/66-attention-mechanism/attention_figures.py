"""Matrix, vector and scaling animations for Day 66 (headless Matplotlib)."""

from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from attention_lab import (
    COLORS, entropy, pipeline_fixture, projection_fixture, scaling_experiment,
    softmax_state, value_fixture, weighted_values,
)


def decorate(fig, title, caption):
    if not hasattr(fig, "_attention_title"):
        fig._attention_title = fig.text(.04, .95, title, fontsize=17, weight="bold", va="top", color="#172033")
        fig._attention_caption = fig.text(.04, .025, "", fontsize=10, va="bottom", color="#475569")
    fig._attention_title.set_text(title)
    fig._attention_caption.set_text(fill(caption, 145))
    fig.subplots_adjust(left=.07, right=.96, top=.84, bottom=.18, hspace=.65, wspace=.45)


def matrix(ax, data, title, color="output", active=None, limits=None, rows=None, columns=None):
    data = np.asarray(data, dtype=float)
    cmap_name = {"query": "Blues", "key": "Oranges", "value": "Greens", "output": "Purples", "mask": "Greys"}[color]
    cmap = plt.get_cmap(cmap_name).with_extremes(bad="#dbe3e9")
    finite = data[np.isfinite(data)]
    low, high = limits if limits is not None else ((float(finite.min()), float(finite.max())) if len(finite) else (0, 1))
    if low == high:
        high = low + 1
    image = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, aspect="auto", vmin=low, vmax=high)
    ax.set_title(f"{title}  ({data.shape[0]}×{data.shape[1]})", fontsize=10, loc="left", color=COLORS[color], pad=8)
    ax.set_xticks(range(data.shape[1]), columns if columns is not None else range(data.shape[1]), fontsize=8)
    ax.set_yticks(range(data.shape[0]), rows if rows is not None else range(data.shape[0]), fontsize=8)
    for (i, j), value in np.ndenumerate(data):
        label = "−∞" if np.isneginf(value) else "·" if np.isnan(value) else f"{value:.2f}"
        ink = "white" if np.isfinite(value) and (value-low)/(high-low) > .6 else "#172033"
        ax.text(j, i, label, ha="center", va="center", fontsize=8, color=ink)
    if active is not None:
        i, j = active
        ax.add_patch(Rectangle((j-.49, i-.49), .98, .98, fill=False, edgecolor="#e11d48", lw=2.5))
    return image


def save_png(path, fig):
    try:
        fig.savefig(path, dpi=150, facecolor="white")
    finally:
        plt.close(fig)


def save_gif(path, fig, update, frames, fps=2):
    animation = FuncAnimation(fig, update, frames=frames, interval=1000/fps, repeat=True)
    try:
        animation.save(path, writer=PillowWriter(fps=fps), dpi=85)
    finally:
        plt.close(fig)


def projections(path, quick=False):
    x, matrices, result = projection_fixture()
    products = [result.query, result.key, result.value]
    operations = [(g, i, j) for g, product in enumerate(products) for i, j in np.ndindex(product.shape)]
    if quick:
        operations = [operations[i] for i in np.linspace(0, len(operations)-1, 12, dtype=int)]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    def update(frame):
        g, row, col = operations[frame]
        for group, (w, product) in enumerate(zip(matrices, products)):
            color = ("query", "key", "value")[group]
            for ax in axes[group]:
                ax.clear()
            matrix(axes[group, 0], x, "Input X", color="mask")
            matrix(axes[group, 1], w, f"W_{'QKV'[group]} · synthetic", color=color)
            shown = product.copy() if group < g else np.full_like(product, np.nan)
            if group == g:
                flat_index = row*product.shape[1]+col
                shown.flat[:flat_index+1] = product.flat[:flat_index+1]
            matrix(axes[group, 2], shown, f"{'QKV'[group]} = XW_{'QKV'[group]}", color=color,
                   active=(row, col) if group == g else None, limits=(product.min(), product.max()))
            if group == g:
                axes[group, 0].add_patch(Rectangle((-.5, row-.5), 3, 1, fill=False, edgecolor="#e11d48", lw=2))
                axes[group, 1].add_patch(Rectangle((col-.5, -.5), 1, 3, fill=False, edgecolor="#e11d48", lw=2))
        terms = " + ".join(f"({a:.1f}×{b:.1f})" for a, b in zip(x[row], matrices[g][:, col]))
        decorate(fig, "01 · One embedding, three separate projections", f"Active {'QKV'[g]}[{row},{col}] = {terms} = {products[g][row,col]:.3f}. Red outlines: input row, projection column, resulting cell. Seed 66; rounded random parameters, not trained. Quick mode samples cells.")
    save_gif(path, fig, update, len(operations), fps=2)


def pipeline(gif_path, png_path):
    t = pipeline_fixture()
    panels = [(t.query, "Q", "query"), (t.key, "K", "key"), (t.key.T, "Kᵀ", "key"),
              (t.raw, "QKᵀ", "key"), (t.scaled, "Scores / √2", "key"), (t.weights, "A = row softmax", "output"),
              (t.value, "V", "value"), (t.output, "O = AV", "output")]
    stages = [(0, "Read Q, K and V"), (2, "Transpose K: each column is one key"),
              (3, "Raw score [0,0] = 1×1 + 0×0 = 1"), (4, "Scale [0,0]: 1 / √2 = 0.7071"),
              (5, "Normalize row 0: exp(shifted score) / sum(exp)"),
              (5, "Row 0 weights: 0.575975 + 0.283995 + 0.140029 = 1"),
              (7, "O[0,0] = 0.575975×2 + 0.283995×0 + 0.140029×1"),
              (7, "O[0,1] = 0.575975×0 + 0.283995×2 + 0.140029×1"),
              (7, "Final O[0] = [1.291980, 0.708020]")]
    def draw(fig, axes, index):
        active, annotation = stages[index]
        for j, (data, title, color) in enumerate(panels):
            ax = axes.flat[j]
            ax.clear()
            matrix(ax, data, title, color, active=(0, 1 if index == 7 else 0) if j == active else None)
            for spine in ax.spines.values():
                spine.set_linewidth(2.5 if j == active else .6)
                spine.set_edgecolor(COLORS[color] if j == active else "#cbd5e1")
        ax = axes.flat[8]
        ax.clear(); ax.axis("off")
        ax.text(0, .9, f"Step {index+1}/{len(stages)}\n\n{fill(annotation, 29)}", va="top", fontsize=12)
        decorate(fig, "03 · Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V", "Rows are queries; columns of A are keys. Blue Q · orange K/scores · green V · purple weights/output. Red cell marks active arithmetic. All matrices are synthetic and computed in float64.")
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    save_gif(gif_path, fig, lambda i: draw(fig, axes, i), len(stages), fps=1)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    draw(fig, axes, len(stages)-1)
    save_png(png_path, fig)


def planar_arrows(ax, vectors, colors, labels, limit=2.6, label_offsets=None):
    ax.axhline(0, color="#cbd5e1", lw=.7); ax.axvline(0, color="#cbd5e1", lw=.7)
    for index, (v, color, label) in enumerate(zip(vectors, colors, labels)):
        ax.annotate("", xy=v[:2], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2.3))
        offset = label_offsets[index] if label_offsets is not None else ((6, -18) if label.startswith(("Q", "O")) else (6, 9))
        ax.annotate(label, xy=v[:2], xytext=offset, textcoords="offset points", color=color, fontsize=10)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", xlabel="x", ylabel="y")
    ax.grid(alpha=.15)


def alignment(path, quick=False):
    angles = np.linspace(0, 2*np.pi, 17 if quick else 41)
    keys = np.array([[1., 0.], [-.5, np.sqrt(3)/2], [-.5, -np.sqrt(3)/2]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    def update(frame):
        regime, index = divmod(frame, len(angles))
        q = np.array([np.cos(angles[index]), np.sin(angles[index])])
        k = keys * (np.array([2., .6, 1.])[:, None] if regime else 1)
        raw = k@q
        p = softmax_state(raw, dimension=2)["weights"]
        for ax in axes:
            ax.clear()
        planar_arrows(axes[0], [q, *k], [COLORS["query"]]+[COLORS["key"]]*3, ["Q", "K1", "K2", "K3"])
        axes[0].set_title("Unit-norm keys" if not regime else "Unequal key norms: 2.0, 0.6, 1.0")
        axes[1].bar(["K1", "K2", "K3"], p, color=COLORS["output"])
        axes[1].set(ylim=(0, 1), ylabel="Attention weight", title=f"Query angle {np.rad2deg(angles[index]):.0f}° · dₖ=2")
        for i, weight in enumerate(p):
            axes[1].text(i, weight+.04, f"dot {raw[i]:.2f}\nα {weight:.3f}", ha="center", fontsize=10)
        decorate(fig, "02 · Rotate a query; recompute the attention distribution", "Two complete rotations on fixed axes: unit keys, then unequal magnitudes. All queries have unit norm. Orange key length affects dot products; cosine would depend only on direction. The separate HTML is truly 3D.")
    save_gif(path, fig, update, 2*len(angles), fps=5)


def scaling(folder, quick=False):
    folder = Path(folder)
    results = scaling_experiment(quick)
    raw_bound = max(abs(row["raw_scores"]).max() for row in results)*1.02
    scaled_bound = max(abs(row["scaled_scores"]).max() for row in results)*1.02
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    def update(index):
        row = results[index]
        for ax, name, bound in zip(axes, ("raw", "scaled"), (raw_bound, scaled_bound)):
            ax.clear()
            scores = row[name+"_scores"]
            ax.hist(scores.ravel(), bins=np.linspace(-bound, bound, 65), density=True,
                    color=COLORS["key" if name == "raw" else "query"], alpha=.85)
            ax.set(xlim=(-bound, bound), ylim=(0, .55), xlabel="Score", ylabel="Empirical density",
                   title=f"{name.title()} · dₖ={row['dimension']} · variance {scores.var():.2f}")
            ax.text(.03, .9, f"Theoretical variance: {row['metrics'][name]['theory_variance']:.0f}", transform=ax.transAxes, fontsize=11)
        decorate(fig, "04 · Score dispersion as the key dimension grows", f"Seed 66; {results[0]['queries']} standard-normal queries × 64 standard-normal keys per dimension. Each panel keeps its own fixed score range across frames; y limits match. Raw/scaled x ranges differ and are labeled. No trained vectors.")
    save_gif(folder/"04_attention_scaling.gif", fig, update, len(results), fps=1)
    fig, axes = plt.subplots(2, len(results), figsize=(16, 7))
    for i, row in enumerate(results):
        for j, name in enumerate(("raw", "scaled")):
            ax = axes[j, i]
            bound = raw_bound if name == "raw" else scaled_bound
            ax.hist(row[name+"_scores"].ravel(), bins=np.linspace(-bound, bound, 45), density=True, color=COLORS["key" if j == 0 else "query"])
            ax.set(xlim=(-bound, bound), ylim=(0, .55), title=f"{name} · dₖ={row['dimension']}")
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_ylabel("Empirical density")
    decorate(fig, "04 · Raw and scaled score distributions", "Fixed within-row x limits; different between raw and scaled rows. Independent Gaussian coordinates; empirical matrix entries share query/key factors. No uncertainty interval or trained-model claim.")
    save_png(folder/"04_score_distributions.png", fig)
    dimensions = [r["dimension"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    for name, color in (("raw", COLORS["key"]), ("scaled", COLORS["query"])):
        for ax, metric, title in zip(axes, ("variance", "entropy", "max_probability"), ("Empirical vs theoretical variance", "Mean row entropy (nats)", "Mean row maximum probability")):
            ax.plot(dimensions, [r["metrics"][name][metric] for r in results], "o-", color=color, label=name)
            ax.set(xscale="log", xlabel="dₖ", title=title)
            if metric == "variance":
                ax.plot(dimensions, [r["metrics"][name]["theory_variance"] for r in results], "--", color=color, label=name+" theory")
                ax.set_yscale("log")
            else:
                ax.set_ylim((0, np.log(64)+.1) if metric == "entropy" else (0, 1))
            ax.legend(fontsize=8); ax.grid(alpha=.15)
    decorate(fig, "04 · Scaling controls dispersion under Gaussian assumptions", "New visual protocol: 64 keys; seed 66; variance ddof=0; entropy averaged over queries. Dashed curves are theoretical, solid curves empirical. This is not a training-quality benchmark; author interpretation review required.")
    save_png(folder/"04_attention_entropy.png", fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for name, color in (("raw", COLORS["key"]), ("scaled", COLORS["query"])):
        for ax, metric, title in zip(axes, ("jacobian_norm", "diagonal_mean"), ("Mean ||diag(p) − ppᵀ||F", "Mean pᵢ(1−pᵢ) over rows and keys")):
            ax.plot(dimensions, [r["metrics"][name][metric] for r in results], "o-", color=color, label=name)
            ax.set(xscale="log", xlabel="dₖ", ylabel="Derivative magnitude", title=title)
            ax.legend(); ax.grid(alpha=.15)
    decorate(fig, "04 · Softmax saturation: derivatives measured at its input", "These are derivatives with respect to the logits entering softmax. Derivatives with respect to unscaled dot products include the extra 1/√dₖ factor for scaled attention. Lower local sensitivity is not proof that all network gradients vanish.")
    save_png(folder/"04_softmax_gradient_saturation.png", fig)
    return [{k: v for k, v in row.items() if not k.endswith("_scores")} for row in results]


def temperature(path, quick=False):
    temperatures = np.geomspace(.1, 5, 16 if quick else 32)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    def update(index):
        state = softmax_state(temperature=temperatures[index])
        for ax in axes:
            ax.clear()
        axes[0].bar(range(4), state["scaled"], color=COLORS["key"])
        axes[0].set(ylim=(-6, 11), xlabel="Key", ylabel="Scaled logit / T", title="Fixed raw logits [2,1,0,−1] · dₖ=4")
        axes[1].bar(range(4), state["weights"], color=COLORS["output"])
        axes[1].set(ylim=(0, 1), xlabel="Key", ylabel="Weight", title=f"T={temperatures[index]:.2f} · H={state['entropy']:.3f} nats")
        decorate(fig, "05 · Temperature changes all normalized weights", f"Max probability {state['maximum']:.4f}; softmax Jacobian Frobenius norm {state['jacobian_norm']:.4f}. Temperature is an additional educational control; standard SDPA needs only dimensional scaling. Stable exponentials use maximum subtraction.")
    save_gif(path, fig, update, len(temperatures), fps=3)


def value_animations(aggregation_path, changing_path, quick=False):
    values = value_fixture()[:3, :2]
    frames = 18 if quick else 36
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    def update(index):
        logits = np.array([np.cos(index/frames*2*np.pi+t) for t in (0, 2*np.pi/3, 4*np.pi/3)])*3
        p = softmax_state(logits, dimension=1)["weights"]
        _, weighted, output = weighted_values(p, values)
        for ax in axes:
            ax.clear()
        axes[0].add_patch(Polygon(values, color=COLORS["value"], alpha=.1))
        planar_arrows(axes[0], [*values, *weighted, output], [COLORS["value"]]*3+[COLORS["mask"]]*3+[COLORS["output"]],
                      ["V1", "V2", "V3", "α1V1", "α2V2", "α3V3", "O"],
                      label_offsets=[(-34, -3), (8, -2), (5, 9), (7, -20), (9, -14), (9, 13), (-24, 16)])
        axes[0].set_title("Green values · gray weighted values · purple output")
        axes[1].bar(["V1", "V2", "V3"], p, color=COLORS["output"])
        axes[1].set(ylim=(0, 1), ylabel="Mixture coefficient", title=f"O = [{output[0]:.3f}, {output[1]:.3f}] · Σα=1")
        decorate(fig, "06 · Weighted values stay inside their convex hull", "Weights are computed by softmax from smoothly varying synthetic logits; they stay nonnegative and sum to one. O=ΣαᵢVᵢ. The convex-combination property applies before attention dropout and later output projections.")
    save_gif(aggregation_path, fig, update, frames, fps=4)
    t = pipeline_fixture()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    def changing(index):
        shift = -1.5 + 3*index/(frames-1)
        v = t.value + [shift, 0]
        output = t.weights@v
        for ax in axes:
            ax.clear()
        planar_arrows(axes[0], [*v, output[0]], [COLORS["value"]]*3+[COLORS["output"]], ["V1", "V2", "V3", "O(query 0)"], limit=4)
        axes[0].set_title(f"Only V moves · x shift {shift:.2f}")
        matrix(axes[1], t.weights, "A fixed: Q and K unchanged", limits=(0, 1))
        decorate(fig, "06 · Changing values changes output without changing attention", f"Q and K are held fixed; raw scores and weights are identical at every frame. First output = [{output[0,0]:.4f}, {output[0,1]:.4f}]. Values are translated along x; this is a controlled intervention on one operation.")
    save_gif(changing_path, fig, changing, frames, fps=4)
