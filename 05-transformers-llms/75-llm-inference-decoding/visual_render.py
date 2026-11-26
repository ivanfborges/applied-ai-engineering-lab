"""Headless Matplotlib/Pillow exports and offline, stepped Plotly landscapes."""

from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import plotly.graph_objects as go

from from_scratch import softmax
from visual_core import TOKENS, PRESETS, binary_logits, sequence_score, stages

BLUE, GREEN, GRAY, ORANGE, PURPLE = "#2772a1", "#27856a", "#c9d0d8", "#dc7e20", "#7956a2"
TOKEN_COLORS = ["#eeeeee", BLUE, ORANGE, GREEN]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.titleweight": "bold"})


def save_png(fig, root, name):
    fig.savefig(Path(root) / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def save_animation(root, name, count, draw, size=(12, 7), fps=5):
    fig = plt.figure(figsize=size, layout="constrained")

    def update(index):
        fig.clear()
        draw(fig, int(index))

    animation = FuncAnimation(fig, update, frames=count, interval=1000 / fps, repeat=True)
    animation.save(Path(root) / name, writer=PillowWriter(fps=fps), dpi=85)
    plt.close(fig)
    return name


def bars(ax, values, title, labels=TOKENS, colors=BLUE, probability=True):
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.set(xticks=np.arange(len(values)), xticklabels=labels, title=title,
           ylabel="Probability" if probability else "Logit")
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    if probability:
        ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)


def temperature(root, report):
    def draw(fig, index):
        row = report[index]
        axes = fig.subplots(2, 2)
        bars(axes[0, 0], row["logits"], "Original logits (fixed)", probability=False)
        if row["scaled_logits"] is None:
            axes[0, 1].text(0.5, 0.5, "T = 0: explicit argmax branch\nNo division by zero", ha="center", va="center")
            axes[0, 1].axis("off")
        else:
            bars(axes[0, 1], row["scaled_logits"], "Scaled logits z / T", probability=False)
        bars(axes[1, 0], row["original"], "Original probabilities, T = 1")
        bars(axes[1, 1], row["final"], "Temperature probabilities", colors=GREEN)
        fig.suptitle(f"Synthetic next-token distribution | T={row['temperature']:.2f}\n"
                     f"H={row['entropy']:.3f} bits | max p={row['maximum']:.3f} | effective vocabulary={row['effective']:.2f}")

    names = [save_animation(root, "temperature_softmax.gif", len(report), draw)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    positive = [r for r in report if r["temperature"] > 0]
    x = [r["temperature"] for r in positive]
    axes[0].plot(x, [r["entropy"] for r in positive], color=BLUE, label="Entropy (bits)")
    axes[0].plot(x, [r["effective"] for r in positive], color=GREEN, label="Effective vocabulary (tokens)")
    axes[0].set(xlabel="Temperature", ylabel="Value in indicated units", title="Unfiltered concentration"); axes[0].legend()
    axes[1].plot(x, [r["maximum"] for r in positive], color=ORANGE)
    axes[1].set(xlabel="Temperature", ylabel="Maximum token probability", title="Same finite logits; no filters", ylim=(0, 1.05))
    names.append(save_png(fig, root, "temperature_entropy.png"))
    return names


def top_k(root, report):
    def draw(fig, index):
        row = report[index]; state = row["state"]
        order = np.argsort(-np.asarray(state["logits"]), kind="stable")
        selected = row["selected"]
        labels = [TOKENS[i] for i in order]
        final = np.asarray(state["final"])[order]
        colors = [ORANGE if i == selected else GREEN if state["final"][i] > 0 else GRAY for i in order]
        axes = fig.subplots(2, 2)
        bars(axes[0, 0], np.asarray(state["original"])[order], "1. Rank original probabilities", labels)
        bars(axes[0, 1], np.asarray(state["raw_k"])[order], "2. Exclude tail (not normalized)", labels, colors)
        bars(axes[1, 0], final, "3. Renormalize retained candidates", labels, colors)
        axes[1, 1].axis("off")
        axes[1, 1].text(0.02, 0.9, f"k = {state['top_k']}\nRetained = {state['candidates']}\n"
                              f"Original mass = {state['retained_original_mass']:.4f}\n"
                              f"Final H = {state['entropy']:.3f} bits\n\n4. Sample: {TOKENS[selected]}\n"
                              "Green: retained | gray: excluded\nOrange: sampled\nTies: lower vocabulary index first",
                              va="top", fontsize=12, linespacing=1.7)
        fig.suptitle("Top-k filtering and seeded categorical selection | synthetic")
    return [save_animation(root, "top_k_sampling.gif", len(report), draw, fps=3)]


def top_p(root, report):
    def draw(fig, index):
        axes = fig.subplots(2, 3)
        for column, (preset, records) in enumerate(report.items()):
            row = records[index]
            order = np.argsort(-np.asarray(row["original"]), kind="stable")
            sorted_p = np.asarray(row["original"])[order]
            retained = np.asarray(row["final"])[order] > 0
            labels = [TOKENS[i] for i in order]
            colors = [GREEN if keep else GRAY for keep in retained]
            ax = axes[0, column]
            ax.bar(range(10), sorted_p, color=colors, label="Original probability")
            ax.plot(range(10), np.cumsum(sorted_p), marker=".", color=PURPLE, label="Cumulative mass")
            ax.axhline(row["top_p"], color=ORANGE, linestyle="--", label=f"p={row['top_p']:.2f}")
            crossing = row["candidates"] - 1
            ax.scatter([crossing], [np.cumsum(sorted_p)[crossing]], color=ORANGE, s=70, zorder=5)
            ax.set(ylim=(0, 1.05), title=f"{preset}: {row['candidates']} candidates",
                   xticks=range(10), xticklabels=labels, ylabel="Probability / cumulative mass")
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            if column == 0:
                ax.legend(fontsize=7)
            bars(axes[1, column], np.asarray(row["final"])[order], "Nucleus after renormalization", labels, colors)
        fig.suptitle("One threshold, different nucleus sizes | crossing token is included | synthetic")
    names = [save_animation(root, "top_p_sampling.gif", len(next(iter(report.values()))), draw)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, (preset, logits) in zip(axes, PRESETS.items()):
        first, second = stages(logits, top_k=3), stages(logits, top_p=0.9)
        ax.bar(np.arange(10) - 0.19, first["final"], width=0.38, color=BLUE, label=f"k=3: {first['candidates']} tokens")
        ax.bar(np.arange(10) + 0.19, second["final"], width=0.38, color=GREEN, label=f"p=.9: {second['candidates']} tokens")
        ax.set(title=preset, xlabel="Vocabulary token ID", ylabel="Filtered probability", ylim=(0, 1.05)); ax.legend(fontsize=8)
    names.append(save_png(fig, root, "top_k_vs_top_p.png"))
    return names


def stepped_landscape(grids, metric="entropy"):
    """Each computed grid value is a flat independent tile; no smoothing across cells."""
    figure = go.Figure()
    for position, grid in enumerate(grids):
        ts, ps = np.asarray(grid["temperatures"]), np.asarray(grid["thresholds"])
        t_edges = np.r_[ts[0], (ts[:-1] + ts[1:]) / 2, ts[-1]]
        p_edges = np.r_[ps[0], (ps[:-1] + ps[1:]) / 2, ps[-1]]
        x, y, z, custom, triangles = [], [], [], [], []
        for row, p in enumerate(ps):
            for column, t in enumerate(ts):
                value = grid[metric][row][column]
                start = len(x)
                x.extend([t_edges[column], t_edges[column + 1], t_edges[column + 1], t_edges[column]])
                y.extend([p_edges[row], p_edges[row], p_edges[row + 1], p_edges[row + 1]])
                z.extend([value] * 4)
                info = [t, p, "off" if grid["top_k"] is None else grid["top_k"],
                        grid["entropy"][row][column], grid["candidates"][row][column], grid["maximum"][row][column]]
                custom.extend([info] * 4)
                triangles.extend([(start, start + 1, start + 2), (start, start + 2, start + 3)])
        faces = np.array(triangles)
        figure.add_trace(go.Mesh3d(x=x, y=y, z=z, i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                   customdata=custom, intensity=z, colorscale="Viridis", flatshading=True,
                                   visible=position == 0, colorbar=dict(title="bits" if metric == "entropy" else "tokens"),
                                   cmin=0 if metric == "entropy" else 1, cmax=np.log2(10) if metric == "entropy" else 10,
                                   hovertemplate="Computed T=%{customdata[0]:.3f}<br>Computed p=%{customdata[1]:.3f}"
                                   "<br>top-k=%{customdata[2]}<br>H=%{customdata[3]:.4f} bits"
                                   "<br>Candidates=%{customdata[4]}<br>max p=%{customdata[5]:.4f}<extra></extra>"))
    buttons = [dict(label=f"top-k: {g['top_k'] or 'off'}", method="update",
                    args=[{"visible": [j == i for j in range(len(grids))]}]) for i, g in enumerate(grids)]
    figure.update_layout(title=f"Computed decoding {metric} | flat grid tiles, no interpolation",
                         scene=dict(xaxis_title="Temperature", yaxis_title="Top-p threshold",
                                    zaxis_title="Entropy (bits)" if metric == "entropy" else "Positive-mass candidates"),
                         updatemenus=[dict(buttons=buttons, x=0, y=1.15)], height=700,
                         margin=dict(l=20, r=20, t=120, b=70), template="plotly_white",
                         annotations=[dict(text="Synthetic logits. Each tile shows its computed grid point (hover). Parameter space, not physical geometry or latency.",
                                           x=0, y=-0.07, xref="paper", yref="paper", showarrow=False)])
    return figure


def interactions(root, report):
    names = []
    for metric, filename in (("entropy", "temperature_top_p_entropy_heatmap.png"),
                             ("candidates", "temperature_top_p_candidate_heatmap.png"),
                             ("maximum", "temperature_top_p_max_probability_heatmap.png"),
                             ("effective", "temperature_top_p_effective_vocabulary_heatmap.png")):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        maximum = max(np.max(grid[metric]) for grid in report)
        for ax, grid in zip(axes, report):
            values = np.asarray(grid[metric])
            options = dict(cmap="viridis", vmin=0, vmax=maximum)
            if metric == "candidates":
                options = dict(cmap=plt.get_cmap("viridis", 10), norm=BoundaryNorm(np.arange(0.5, 11.5), 10))
            image = ax.imshow(values, origin="lower", aspect="auto", interpolation="none", **options)
            ticks = np.linspace(0, values.shape[0] - 1, 5).astype(int)
            ax.set(xticks=ticks, xticklabels=[f"{grid['temperatures'][i]:.2f}" for i in ticks],
                   yticks=ticks, yticklabels=[f"{grid['thresholds'][i]:.2f}" for i in ticks],
                   xlabel="Temperature", ylabel="Top-p", title=f"Top-k = {grid['top_k'] or 'off'}")
            colorbar = fig.colorbar(image, ax=ax, label="bits" if metric == "entropy" else "probability" if metric == "maximum" else "tokens")
            if metric == "candidates":
                colorbar.set_ticks(range(1, 11))
        fig.suptitle(f"Computed {metric} | nearest grid cells; no smoothing | synthetic")
        names.append(save_png(fig, root, filename))
    for metric, filename in (("entropy", "decoding_entropy_landscape.html"), ("candidates", "decoding_candidate_landscape.html")):
        stepped_landscape(report, metric).write_html(Path(root) / filename, include_plotlyjs=True, auto_open=False,
                                                   config={"displaylogo": False}, div_id=f"decoding-{metric}")
        names.append(filename)
    return names


def tree_positions(depth=3):
    coordinates = {(): (0.5, 0)}
    for level in range(1, depth + 1):
        for index, ids in enumerate(product(range(2), repeat=level)):
            coordinates[ids] = ((index + 0.5) / 2 ** level, -level)
    return coordinates


def draw_tree(ax, path, visible_depth=3, exact=()):
    coordinates = tree_positions()
    for ids, (x, y) in coordinates.items():
        selected = len(ids) <= visible_depth and tuple(path[:len(ids)]) == ids
        if ids:
            px, py = coordinates[ids[:-1]]
            probability = softmax(binary_logits(ids[:-1]))[ids[-1]]
            ax.plot([px, x], [py, y], color=ORANGE if selected else GRAY, lw=3 if selected else 1)
            ax.text((px + x) / 2, (py + y) / 2, f"{probability:.2f}", fontsize=8, ha="center", backgroundcolor="white")
        ax.scatter([x], [y], s=350, color=ORANGE if selected else BLUE, zorder=4)
        label = "<BOS>" if not ids else "".join("AB"[i] for i in ids)
        score = sequence_score(ids)
        ax.text(x, y - 0.13, f"{label}\nln P={score:.3f}", ha="center", va="top", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.1))
    if exact:
        points = [coordinates[tuple(exact[:level])] for level in range(4)]
        ax.plot([p[0] for p in points], [p[1] for p in points], "--", color=GREEN, alpha=0.6, label="Exact best model path")
        ax.legend(loc="upper left", fontsize=8)
    ax.set(xlim=(-0.04, 1.04), ylim=(-3.8, 0.6)); ax.axis("off")


def generation(root, report):
    strategies = list(report["paths"].items()) + [(f"Beam width {report['beam_width']}", report["beam"]["best"])]
    exact = report["exact"][0]["ids"]

    def tree_frame(fig, index):
        strategy, path = strategies[index // 4]
        ax = fig.subplots(); draw_tree(ax, path["ids"], index % 4, exact)
        beam_text = ""
        if index // 4 == len(strategies) - 1 and index % 4:
            active = report["beam"]["history"][index % 4 - 1]
            beam_text = " | active beam: " + ", ".join("".join("AB"[v] for v in r["ids"]) for r in active)
        fig.suptitle(f"{strategy} | step {index % 4}/3 | model joint log P={path['log_probability']:.3f}\n"
                     "Edge: conditional model probability | node: prefix joint log probability" + beam_text)

    names = [save_animation(root, "autoregressive_generation_tree.gif", len(strategies) * 4, tree_frame, fps=2)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    for ax, (name, path) in zip(axes, (strategies[0], strategies[-1])):
        draw_tree(ax, path["ids"], 3, exact)
        ax.set_title(f"{name}: P={np.exp(path['log_probability']):.4f}")
    names.append(save_png(fig, root, "greedy_vs_beam_search.png"))

    def loop_frame(fig, index):
        axes = fig.subplots(2, 2)
        for column, name in enumerate(("Greedy", "Temperature sampling")):
            path = report["paths"][name]
            row = path["steps"][index]
            bars(axes[0, column], row["state"]["logits"], "Raw conditional logits", ("A", "B"), probability=False)
            bars(axes[1, column], row["state"]["final"], "Transformed policy q", ("A", "B"),
                 [ORANGE if token == row["selected"] else GREEN for token in range(2)])
            before = "".join("AB"[i] for i in row["context"]) or "<BOS>"
            after = "".join("AB"[i] for i in path["ids"][:index + 1])
            axes[0, column].set_title(f"{name}: context {before} -> selected {'AB'[row['selected']]}")
            axes[1, column].set_title(f"Output {after} | remaining budget {2 - index}\n"
                                       f"model log P={row['model_log_probability']:.3f}; policy log q={row['policy_log_probability']:.3f}")
        fig.suptitle(f"Token-by-token conditional generation | step {index + 1}/3 | same initial prompt | synthetic")
    names.append(save_animation(root, "autoregressive_decoding_loop.gif", 3, loop_frame, fps=1))
    return names


def stopping(root, report):
    runs = list(report["runs"].items())

    def draw(fig, index):
        axes = fig.subplots(3, 2, gridspec_kw={"width_ratios": [1.1, 1.5]})
        for line, (name, run) in enumerate(runs):
            row = run["trace"][min(index, len(run["trace"]) - 1)]
            step = min(index + 1, len(run["trace"]))
            bars(axes[line, 0], row["state"]["final"], f"{name} | step {step}", report["vocabulary"],
                 [ORANGE if i == row["selected"] else BLUE for i in range(6)])
            axes[line, 1].axis("off")
            generated = run["result"]["sampled_ids"][:step]
            labels = " | ".join(report["vocabulary"][i] for i in generated)
            axes[line, 1].text(0, 0.85, f"Generated IDs: {list(generated)}\nFragments: {labels}\n"
                              f"Returned text: {row['text']!r}\nRemaining budget: {row['remaining']}\nReason: {row['reason']}",
                              va="top", fontsize=10, linespacing=1.5)
        fig.suptitle("EOS vs output budget vs cross-token stop '<END>' (excluded from returned text) | synthetic")
    names = [save_animation(root, "generation_stopping_conditions.gif", max(len(r[1]["trace"]) for r in runs), draw, size=(13, 9), fps=1)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    texts = [report["runs"]["Output budget"]["result"]["text"], report["complete_budget"]["result"]["text"]]
    for ax, text, title in zip(axes, texts, ("Budget 2: syntactically incomplete JSON", "Budget 3: complete JSON; still no semantic guarantee")):
        ax.axis("off"); ax.set_title(title, fontsize=10)
        ax.text(0.5, 0.55, text, ha="center", fontsize=20, family="monospace")
        ax.text(0.5, 0.2, "Handwritten fragments; output cap is separate from context-window capacity", ha="center", wrap=True, fontsize=9)
    names.append(save_png(fig, root, "structured_output_truncation.png"))
    return names


def draw_sequence_matrix(ax, matrix, title):
    matrix = np.asarray(matrix)
    image = ax.imshow(matrix + 1, interpolation="none", aspect="auto", vmin=0, vmax=3,
                      cmap=ListedColormap(TOKEN_COLORS))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            label = "pad" if value < 0 else ("A", "B", "EOS")[value]
            ax.text(column, row, label, ha="center", va="center", fontsize=6,
                    color="black" if value < 0 else "white")
    ax.set(title=title, xlabel="Generated step (EOS included)", ylabel="Run")
    return image


def reproducibility(root, report):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for ax, (name, row) in zip(axes.ravel(), report.items()):
        draw_sequence_matrix(ax, row["matrix"], f"{name} | unique={row['unique']} | pairwise exact={row['exact_agreement']:.3f}\n"
                             f"mean pairwise token agreement={row['token_agreement']:.3f}")
    fig.suptitle("Seed replay vs shared RNG advancement | A blue, B orange, EOS green, padding gray")
    names = [save_png(fig, root, "seed_reproducibility.png")]
    count = max(len(row["matrix"][0]) for row in report.values())

    def draw(fig, index):
        axes = fig.subplots(2, 2)
        for ax, (name, row) in zip(axes.ravel(), report.items()):
            matrix = np.asarray(row["matrix"]).copy()
            matrix[:, index + 1:] = -1
            draw_sequence_matrix(ax, matrix, name)
        fig.suptitle(f"Reveal generation step {index + 1} | future cells and padding gray | synthetic NumPy only")
    names.append(save_animation(root, "decoding_sequence_comparison.gif", count, draw, size=(13, 8), fps=3))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name, row in report.items():
        lengths, counts = np.unique(row["lengths"], return_counts=True)
        axes[0].step(lengths, counts, where="mid", marker="o", label=name)
    axes[0].set(xlabel="Sampled IDs including EOS", ylabel="Runs", title="Observed length distributions"); axes[0].legend(fontsize=8)
    axes[1].bar(range(4), [row["token_agreement"] for row in report.values()], color=BLUE)
    axes[1].set(xticks=range(4), xticklabels=list(report), ylim=(0, 1.05), ylabel="Mean pairwise token agreement")
    axes[1].tick_params(axis="x", labelrotation=15)
    names.append(save_png(fig, root, "sequence_lengths_and_agreement.png"))
    return names


def sampling(root, report):
    def draw(fig, index):
        axes = fig.subplots(2, 3)
        for ax, row in zip(axes.ravel(), report):
            snapshot = row["snapshots"][min(index, len(row["snapshots"]) - 1)]
            ax.bar(np.arange(10) - 0.2, row["probabilities"], width=0.4, color=BLUE, label="Theoretical")
            ax.bar(np.arange(10) + 0.2, snapshot["empirical"], width=0.4, color=ORANGE, label="Empirical")
            ax.set(ylim=(0, 1), xlabel="Token ID", ylabel="Probability / frequency",
                   title=f"{row['name']}: n={snapshot['samples']} | TV={snapshot['total_variation']:.4f}")
            ax.legend(fontsize=8)
        fig.suptitle("Nested categorical samples from fixed policies | finite-sample error need not decrease at every step")
    names = [save_animation(root, "sampling_convergence.gif", len(report[0]["snapshots"]), draw, size=(13, 7))]
    fig = plt.figure(figsize=(13, 7), layout="constrained"); draw(fig, len(report[0]["snapshots"]) - 1)
    names.append(save_png(fig, root, "theoretical_vs_empirical_sampling.png"))
    return names


def cache(root, report):
    theory = report["theory"]

    def draw(fig, index):
        axes = fig.subplots(1, 2)
        length = theory["lengths"][index]
        mask = np.tri(length, dtype=int)
        if index > 0:
            mask[:-1] = 0
        axes[0].imshow(mask, interpolation="none", cmap="Blues", vmin=0, vmax=1)
        axes[0].set(xlabel="Visible keys/values", ylabel="Queries evaluated now",
                    title="Prefill: causal prompt Q/K/V" if index == 0 else "Decode: one new query over all cached K/V")
        ax = axes[1]
        old, new = theory["reused_positions"][index], theory["new_positions"][index]
        ax.barh([0, 1], [old, 0], color=GREEN, label="Previously cached K/V positions")
        ax.barh([0, 1], [new, length], left=[old, 0], color=ORANGE, label="Newly projected K/V positions")
        ax.set(yticks=[0, 1], yticklabels=["With cache", "Without cache"], xlim=(0, theory["lengths"][-1] + 1),
               xlabel="K/V source positions computed or reused this step", title=f"Prediction {index + 1}: context length {length}")
        ax.legend(fontsize=8, loc="lower right")
        fig.suptitle(f"Prefill predicts token 1; feeding previous outputs predicts later tokens\n"
                     f"K/V retained; old Q not retained | cache={theory['cached_kv_bytes'][index]} bytes (assumed architecture)")
    names = [save_animation(root, "prefill_decode_kv_cache.gif", len(theory["lengths"]), draw, size=(12, 6), fps=2)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    step = np.arange(1, theory["output"] + 1)
    axes[0].plot(step, np.cumsum(theory["new_positions"]), label="Cached", color=GREEN)
    axes[0].plot(step, np.cumsum(theory["uncached_projection_positions"]), label="Recompute", color=ORANGE)
    axes[0].set(ylabel="Cumulative K/V projection positions per layer", title="Theoretical projection work"); axes[0].legend()
    axes[1].plot(step, theory["cached_kv_bytes"], color=BLUE)
    axes[1].set(ylabel="Retained K/V bytes", title="2 x layers x KV heads x dim x bytes x length")
    axes[2].plot(step, np.cumsum(theory["cached_attention_pairs"]), label="Cached", color=GREEN)
    axes[2].plot(step, np.cumsum(theory["uncached_attention_pairs"]), label="Recompute", color=ORANGE)
    axes[2].set(ylabel="Cumulative causal attention pairs per head/layer", title="Attention still reads growing context"); axes[2].legend()
    for ax in axes:
        ax.set_xlabel("Output prediction index (first uses prefill)")
    fig.suptitle("Explicit theoretical counts, not measured latency | batch 1, 2 layers, 2 KV heads, dim 8, float32")
    names.append(save_png(fig, root, "inference_complexity.png"))
    return names


RENDERERS = {"temperature": temperature, "top-k": top_k, "top-p": top_p, "interactions": interactions,
             "generation": generation, "stopping": stopping, "reproducibility": reproducibility, "sampling": sampling, "cache": cache}


def render_group(name, root, report):
    return RENDERERS[name](root, report)
