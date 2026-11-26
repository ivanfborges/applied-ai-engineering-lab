"""Bounded Matplotlib/Pillow exports and offline interactive views of recorded runs."""

from dataclasses import asdict
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import plotly.graph_objects as go

from from_scratch import bigram_loss_gradient
from visual_core import (Scaling, cosine_matrix, cross_entropy_logits, logit_descent,
                         optimal_allocation, pooled_pca, scaling_loss, softmax)
from visual_html import (allocation_html, attention_html, compute_html, corpus_html,
                         cross_entropy_html, embedding_html, perplexity_html,
                         probability_html, scaling_html, write_plotly)


COLORS = {"train": "#287da8", "validation": "#d87c31", "subject": "#287da8", "verb": "#ae67a5",
          "object": "#d87c31", "special": "#788897", "function": "#328c74", "future": "#e4e9ef"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#f4f7fb", "axes.facecolor": "white",
                     "axes.labelcolor": "#183247", "text.color": "#183247", "savefig.facecolor": "#f4f7fb"})


def finish(figure, title, footer):
    figure.suptitle(title, fontsize=17, fontweight="bold", y=.98)
    figure.text(.025, .023, footer, fontsize=9, color="#52687b", va="bottom")
    bottom = .075 + max(0, footer.count("\n") - 1) * .035
    figure.tight_layout(rect=(.01, bottom, .99, .93))
    return figure


def save_png(figure, path):
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_gif(path, frames, draw, duration=600):
    images = []
    for frame in frames:
        figure = draw(frame)
        figure.set_dpi(90)
        figure.canvas.draw()
        images.append(Image.fromarray(np.asarray(figure.canvas.buffer_rgba()).copy()).convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=192))
        plt.close(figure)
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0, optimize=False)


def heatmap(ax, matrix, tokens, title, minimum=0, maximum=1, cmap="Blues"):
    image = ax.imshow(matrix, vmin=minimum, vmax=maximum, cmap=cmap)
    positions = np.arange(len(tokens))
    labels = [f"{i}: {token}" for i, token in enumerate(tokens)]
    ax.set_xticks(positions, labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(positions, labels, fontsize=8)
    ax.set_xlabel("Key / context position")
    ax.set_ylabel("Query position")
    ax.set_title(title, fontsize=11)
    return image


def snapshot_xy(report, split, metric):
    return ([s["step"] for s in report["snapshots"]], [s[split][metric] for s in report["snapshots"]])


def render_objectives(folder, reports):
    folder = Path(folder)
    words = ["<bos>", "The", "model", "learns", "patterns", "from", "data.", "<eos>"]
    vocabulary = sorted(words)
    ids = np.array([vocabulary.index(word) for word in words])
    weights = np.zeros((len(vocabulary), len(vocabulary)))
    trajectory = []
    for step in range(121):
        loss, gradient = bigram_loss_gradient(weights, ids[:-1], ids[1:])
        trajectory.append({"step": step, "mean_nll": loss})
        if step < 120:
            weights -= gradient
    predictions = softmax(weights[ids[:-1]])
    (folder / "objectives_measurements.json").write_text(json.dumps({
        "hypothesis": "A toy bigram table can fit a single sentence's observed transitions.",
        "configuration": {"steps": 120, "learning_rate": 1, "initialization": "zero", "sentence": words},
        "trace": trajectory, "vocabulary": vocabulary, "probabilities": predictions.tolist(),
        "interpretation_status": "author review required", "limitation": "Training fit to one sentence; no generalization."}, indent=2), encoding="utf-8")

    def next_frame(frame):
        generation, position = divmod(frame, len(words) - 1)
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [1.3, 1]})
        ax = axes[0]
        ax.set(xlim=(-.5, len(words) - .5), ylim=(-.4, 3.6))
        ax.axis("off")
        for i, word in enumerate(words):
            color = COLORS["train"] if i <= position else COLORS["future"]
            if i == position + 1:
                color = COLORS["validation"]
            displayed = "?" if generation and i > position else word
            ax.text(i, 3, displayed, ha="center", va="center", fontsize=11,
                    bbox={"boxstyle": "round,pad=.4", "facecolor": color, "edgecolor": "white"})
            if i < len(words) - 1:
                ax.text(i, 1.9, f"input {ids[i]}", ha="center", color=COLORS["train"])
                ax.text(i, 1.25, f"target {ids[i+1]}", ha="center", color=COLORS["validation"])
        ax.annotate("", xy=(position + 1, 2.65), xytext=(position, 2.65), arrowprops={"arrowstyle": "->", "color": COLORS["validation"], "lw": 2})
        ax.text(0, .4, "Visible context: " + " ".join(words[:position + 1]), fontsize=12)
        phase = "SEQUENTIAL GENERATION: future tokens are unknown; shifted IDs below are evaluation references only" if generation else "PARALLEL TRAINING: all ground-truth prefixes are known; one row is highlighted"
        ax.text(0, -.12, phase, fontsize=10, fontweight="bold")
        p = predictions[position]
        target = ids[position + 1]
        axes[1].bar(vocabulary, p, color=[COLORS["validation"] if j == target else COLORS["train"] for j in range(len(vocabulary))])
        axes[1].set(ylim=(0, 1), ylabel="Measured next-token probability", title=f"Reference target: {words[position+1]} · NLL = {-np.log(p[target]):.3f} nats")
        return finish(fig, "Text supplies its own labels: input IDs → one-token-shifted target IDs",
                      "Measured NumPy bigram fit to this one sentence, not Transformer probabilities. Blue: context · orange: next target · gray: future.\nNo manually annotated labels: every target is copied from the next position in the original text.")

    save_gif(folder / "next_token_prediction.gif", range(2 * (len(words) - 1)), next_frame)
    clm, mlm = reports["clm"], reports["mlm"]
    c_tokens = [clm["corpus"]["vocabulary"][i] for i in clm["snapshots"][-1]["probe_inputs"][0]]
    m_tokens = [mlm["corpus"]["vocabulary"][i] for i in mlm["snapshots"][-1]["probe_inputs"][0]]
    allowed_clm, allowed_mlm = np.tril(np.ones((5, 5))), np.ones((6, 6))

    def mask_frame(position):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        heatmap(axes[0], allowed_clm, c_tokens, "CLM: lower-triangular permitted connections")
        heatmap(axes[1], allowed_mlm, m_tokens, "MLM: bidirectional corrupted-input connections")
        axes[0].add_patch(Rectangle((-.5, position - .5), 5, 1, fill=False, ec=COLORS["validation"], lw=3))
        axes[1].add_patch(Rectangle((-.5, 3.5), 6, 1, fill=False, ec=COLORS["validation"], lw=3))
        return finish(fig, "Causal prediction versus masked reconstruction",
                      "CLM input: <bos> the cat eats fish → targets: the cat eats fish <eos>. Blue=allowed; white=blocked.\nMLM original: <bos> the cat eats fish <eos>; object fish becomes <mask>, and only selected positions enter the loss.\nAll-mask corruption here differs from BERT's original selected-token replacement mixture.")

    save_png(mask_frame(3), folder / "attention_masks.png")
    save_gif(folder / "clm_vs_mlm.gif", range(5), mask_frame)
    examples = mlm["snapshots"][-1]["reconstruction_examples"]
    vocab = mlm["corpus"]["vocabulary"]

    def reconstruction_frame(index):
        fig, axes = plt.subplots(2, 1, figsize=(13, 7))
        inputs, targets = examples["inputs"][index], examples["targets"][index]
        selected = [i for i, target in enumerate(targets) if target != -100]
        axes[0].axis("off")
        axes[0].set(xlim=(-.5, 5.5), ylim=(-.2, 2.5))
        for i, token in enumerate(inputs):
            color = COLORS["validation"] if i in selected else COLORS["train"]
            axes[0].text(i, 1.8, vocab[token], ha="center", bbox={"boxstyle": "round,pad=.5", "fc": color, "ec": "white"})
            axes[0].text(i, .65, "target: " + vocab[targets[i]] if i in selected else "loss excluded", ha="center", fontsize=10)
        active = selected[0]
        p = np.asarray(examples["probabilities"][index][active])
        for i in range(6):
            if i != active:
                axes[0].annotate("", xy=(active, 1.45), xytext=(i, 1.45), arrowprops={"arrowstyle": "->", "alpha": .4, "color": COLORS["train"], "connectionstyle": "arc3,rad=.15"})
        axes[1].bar(vocab, p, color=[COLORS["validation"] if j == targets[active] else COLORS["train"] for j in range(len(vocab))])
        axes[1].tick_params(axis="x", labelrotation=55, labelsize=8)
        axes[1].set(ylim=(0, 1), ylabel="Measured probability", title=f"First selected position {active} · target={vocab[targets[active]]} · NLL={-np.log(max(p[targets[active]], 1e-30)):.3f}")
        return finish(fig, "Actual held-out MLM reconstruction: selected targets only",
                      "Original tokens at orange positions become targets; the corrupted input contains <mask>. Other positions have target −100.\nPredictions are recorded from the final trained MLM. Reconstruction can be wrong; attention may read context in both directions.")

    save_gif(folder / "masked_token_reconstruction.gif", range(len(examples["inputs"])), reconstruction_frame, 1000)


def render_probabilities(folder, report):
    folder = Path(folder)
    vocab = report["corpus"]["vocabulary"]
    fish = vocab.index("fish")

    def frame(index):
        snap = report["snapshots"][index]
        fig, axes = plt.subplots(2, 1, figsize=(13, 8))
        for probe, ax in enumerate(axes):
            p = np.asarray(snap["probabilities"][probe])
            ax.bar(vocab, p, color=[COLORS["validation"] if j == fish else COLORS["train"] for j in range(len(vocab))])
            ax.tick_params(axis="x", labelrotation=50, labelsize=8)
            ax.set(ylim=(0, 1), ylabel="Probability", title=f"{report['probe_labels'][probe]} · reference fish p={p[fish]:.3f} · NLL={snap['reference_token_nll'][probe]:.3f} · H={snap['entropy'][probe]:.3f}")
        return finish(fig, f"Actual CLM probability evolution · update {snap['step']}",
                      "Fixed vocabulary order and y-axis. Orange marks one admissible reference, not a unique correct answer.\nBoth distributions are recorded from this training run; confidence can increase for an incorrect or unsupported continuation.")

    save_gif(folder / "token_probability_evolution.gif", range(len(report["snapshots"])), frame)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, label in enumerate(report["probe_labels"]):
        x = [s["step"] for s in report["snapshots"]]
        axes[0].plot(x, [s["entropy"][i] for s in report["snapshots"]], marker="o", label=label)
        axes[1].plot(x, [s["probabilities"][i][fish] for s in report["snapshots"]], marker="o", label=label)
    axes[0].set(xlabel="Training updates", ylabel="Entropy (nats)", ylim=(0, np.log(len(vocab)) + .1))
    axes[1].set(xlabel="Training updates", ylabel="P(reference fish)", ylim=(0, 1))
    axes[0].legend(fontsize=8)
    save_png(finish(fig, "Measured uncertainty and reference-token mass", "Entropy describes the entire distribution; low entropy does not imply correctness. One synthetic seed; author review required."), folder / "prediction_entropy_evolution.png")
    probability_html(report, folder / "prediction_distributions.html")


def render_softmax(folder):
    folder = Path(folder)
    records = logit_descent()
    (folder / "logit_descent_measurements.json").write_text(json.dumps({
        "hypothesis": "Explicit logit gradients reduce a fixed-target toy NLL.",
        "configuration": {"target": 0, "third_logit_fixed": 0, "learning_rate": .2, "steps": 30},
        "records": records, "interpretation_status": "author review required",
        "limitation": "Two free logits, not a Transformer loss landscape."}, indent=2), encoding="utf-8")

    def frame(index):
        row = records[index]
        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        axes[0].bar(["target", "rival", "fixed"], row["logits"], color=[COLORS["validation"], COLORS["train"], COLORS["special"]])
        axes[0].set(ylim=(-3, 4), ylabel="Logit")
        axes[1].bar(["target", "rival", "fixed"], row["probabilities"], color=[COLORS["validation"], COLORS["train"], COLORS["special"]])
        axes[1].set(ylim=(0, 1), ylabel="Softmax probability")
        axes[2].bar(["target", "rival", "fixed"], row["gradient"], color=[COLORS["validation"], COLORS["train"], COLORS["special"]])
        axes[2].set(ylim=(-1, 1), ylabel="Full analytical dL/dlogit = p − y")
        return finish(fig, f"NumPy gradient descent · step {row['step']} · NLL={row['loss']:.3f}",
                      "Updates: z[0:2] ← z[0:2] − 0.2 × gradient[0:2]. Third logit stays zero despite a nonzero full gradient.\nThese are actual computed updates on a three-token objective, not measured Transformer gradients.")

    save_gif(folder / "softmax_gradient_descent.gif", range(0, len(records), 2), frame, 350)
    cross_entropy_html(folder / "cross_entropy_curve.html")
    x = np.linspace(-4, 4, 65)
    xx, yy = np.meshgrid(x, x)
    logits = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
    loss, _ = cross_entropy_logits(logits, 0)
    figure = go.Figure(go.Surface(x=x, y=x, z=loss, colorscale="Viridis", opacity=.85, colorbar={"title": "NLL"}))
    figure.add_trace(go.Scatter3d(x=[r["logits"][0] for r in records], y=[r["logits"][1] for r in records],
                                  z=[r["loss"] for r in records], mode="lines+markers", name="Actual gradient steps",
                                  line={"color": "#e77732", "width": 7}, marker={"size": 3}))
    figure.update_layout(scene={"xaxis_title": "Target logit", "yaxis_title": "Rival logit", "zaxis_title": "NLL"})
    write_plotly(figure, folder / "cross_entropy_gradient_descent_3d.html", "Two free logits: surface and actual optimization trajectory", "Third logit fixed at zero. Simplified convex slice, not the Transformer landscape.")


def render_embeddings(folder, report):
    folder = Path(folder)
    matrices = np.asarray([s["embedding"] for s in report["snapshots"]])
    coordinates, variance = pooled_pca(matrices)
    vocabulary = report["corpus"]["vocabulary"]
    categories = report["corpus"]["categories"]
    colors = [COLORS[c] for c in categories]
    low = coordinates.min(axis=(0, 1)) - .5
    high = coordinates.max(axis=(0, 1)) + .5

    def frame(index):
        fig, ax = plt.subplots(figsize=(12, 7))
        for token in range(len(vocabulary)):
            ax.plot(coordinates[:index + 1, token, 0], coordinates[:index + 1, token, 1], color=colors[token], alpha=.55)
            ax.scatter(*coordinates[index, token, :2], color=colors[token], s=32)
            ax.annotate(vocabulary[token], coordinates[index, token, :2], fontsize=8, xytext=(4, 3), textcoords="offset points")
        for category in dict.fromkeys(categories):
            ax.scatter([], [], color=COLORS[category], label=category)
        ax.legend(loc="upper left", fontsize=8)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel=f"PC1 ({variance[0]:.1%})", ylabel=f"PC2 ({variance[1]:.1%})")
        return finish(fig, f"Measured token embedding trajectories · update {report['snapshots'][index]['step']}",
                      f"One PCA fitted on pooled snapshots; 2D explained variance {sum(variance[:2]):.1%}. Fixed axes, original token identities.\nCategory colors are predefined. Visible proximity does not establish semantic similarity or downstream quality.")

    save_gif(folder / "embedding_evolution_2d.gif", range(len(matrices)), frame)
    embedding_html(report, coordinates, variance, folder / "embedding_evolution_3d.html")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, matrix, label in zip(axes, (matrices[0], matrices[-1]), ("Initialization", "Final update")):
        image = heatmap(ax, cosine_matrix(matrix), vocabulary, label + " · original-dimensional cosine", -1, 1, "RdBu_r")
        ax.set(xlabel="Token", ylabel="Token")
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Cosine similarity")
        ax.tick_params(labelsize=6)
    save_png(finish(fig, "Embedding cosine similarities before and after training", "Same color scale −1..1. These are static input embeddings; contextual hidden states can differ for the same token."), folder / "embedding_cosine_similarity.png")
    neighbors = {}
    for label, matrix in (("initial", matrices[0]), ("final", matrices[-1])):
        cosine = cosine_matrix(matrix)
        np.fill_diagonal(cosine, -np.inf)
        neighbors[label] = {token: [{"token": vocabulary[j], "cosine": float(cosine[i, j])}
                                    for j in np.argsort(-cosine[i], kind="stable")[:3]] for i, token in enumerate(vocabulary)}
    (folder / "embedding_geometry.json").write_text(json.dumps({"pca_explained_variance": variance.tolist(),
        "neighbors_original_dimensions": neighbors, "interpretation_status": "author review required"}, indent=2), encoding="utf-8")


def render_attention(folder, reports):
    folder = Path(folder)
    def frame(index):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, name in zip(axes, ("clm", "mlm")):
            report = reports[name]
            snap = report["snapshots"][index]
            tokens = [report["corpus"]["vocabulary"][i] for i in snap["probe_inputs"][0]]
            heatmap(ax, np.asarray(snap["attention"])[0, 0], tokens, f"{name.upper()} · layer 0 / head 0 · step {snap['step']}")
        return finish(fig, "Actual attention during CLM and MLM training", "Captured from the trained models, on fixed probes; weights are not causal feature importance. Shared color scale 0..1.")
    save_gif(folder / "attention_evolution.gif", range(len(reports["clm"]["snapshots"])), frame)
    heads = reports["clm"]["configuration"]["heads"]
    fig, axes = plt.subplots(2, heads, figsize=(max(12, heads * 4), 9), squeeze=False)
    for row, name in enumerate(("clm", "mlm")):
        report, snap = reports[name], reports[name]["snapshots"][-1]
        tokens = [report["corpus"]["vocabulary"][i] for i in snap["probe_inputs"][0]]
        for head in range(heads):
            heatmap(axes[row, head], np.asarray(snap["attention"])[0, head], tokens, f"{name.upper()} · final head {head}")
    save_png(finish(fig, "Final attention heads from each actual model", "Layer 0, observed probe. Axes show token positions; same 0..1 color range. Head differences do not imply specialization."), folder / "attention_heads_comparison.png")
    attention_html(reports, folder / "interactive_attention_explorer.html")


def render_training(folder, reports):
    folder = Path(folder)
    fig, axes = plt.subplots(4, 2, figsize=(14, 15))
    axes = axes.ravel()
    for column, name in enumerate(("clm", "mlm")):
        report = reports[name]
        trace = report["trace"]
        ax = axes[column]
        ax.plot([r["step"] for r in trace], [r["batch_nll"] for r in trace], alpha=.25, color=COLORS["train"], label="Actual stochastic batch")
        for split in ("train", "validation"):
            ax.plot(*snapshot_xy(report, split, "nll"), marker="o", color=COLORS[split], label="Fixed evaluation: " + split)
        ax.set(xlabel="Updates", ylabel="NLL (nats)", title=name.upper() + " · objective-specific loss")
        ax.legend(fontsize=8)
        axes[4].plot([r["step"] for r in trace], [r["gradient_norm"] for r in trace], label=name.upper())
        axes[6].plot([r["step"] for r in trace], [r["elapsed_seconds"] for r in trace], label=name.upper())
        axes[7].plot([r["step"] for r in trace], [r["processed_tokens"] for r in trace], label=name.upper())
        x = [s["step"] for s in report["snapshots"]]
        for split in ("train", "validation"):
            axes[3].plot(x, [s[split]["accuracy"] for s in report["snapshots"]], label=name.upper() + " " + split)
        axes[5].plot(x, [s["validation"]["nll"] - s["train"]["nll"] for s in report["snapshots"]], marker="o", label=name.upper())
    for split in ("train", "validation"):
        axes[2].plot(*snapshot_xy(reports["clm"], split, "perplexity"), color=COLORS[split], marker="o", label=split)
    labels = [(2, "Autoregressive perplexity", "CLM only: exp(mean NLL)"),
              (3, "Token accuracy", "Distinct prediction tasks; not a model ranking"),
              (4, "Global gradient L2 norm", "Actual pre-update gradient norms; no clipping"),
              (5, "Validation − train NLL", "Fixed-evaluation gaps within each objective"),
              (6, "Elapsed seconds", "Elapsed wall time includes snapshot diagnostics"),
              (7, "Processed input tokens", "Input exposure differs from MLM supervised tokens")]
    for index, ylabel, title in labels:
        axes[index].set(xlabel="Updates", ylabel=ylabel, title=title)
        axes[index].legend(fontsize=8)
    axes[3].set_ylim(0, 1)
    save_png(finish(fig, "Measured optimization and generalization dynamics", "Single-seed synthetic experiments. Curves are recorded values, not schematic overfitting examples; interpretation requires author review.\nCLM NLL/perplexity and MLM reconstruction loss use different context and scoring policies."), folder / "training_dynamics.png")
    figure = go.Figure()
    for name in ("clm", "mlm"):
        for split in ("train", "validation"):
            x, y = snapshot_xy(reports[name], split, "nll")
            figure.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name.upper() + " " + split))
    figure.update_layout(xaxis={"title": "Updates", "rangeslider": {"visible": True}}, yaxis_title="Objective-specific NLL (nats)", hovermode="x unified")
    write_plotly(figure, folder / "training_dynamics_interactive.html", "Recorded training dynamics", "Toggle traces to inspect each objective. Values across CLM and MLM do not rank capabilities.")
    perplexity_html(folder / "perplexity_vs_loss.html")


def render_generalization(folder, reports):
    folder = Path(folder)
    names = ["clm", "large_restricted", "longer", "more_unique"]
    labels = ["Small / diverse", "Larger / restricted", "Small / more updates", "Small / more unique"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for name, label in zip(names, labels):
        report = reports[name]
        count = report["configuration"]["batch_size"] * 5
        axes[0, 0].plot([s["step"] * count for s in report["snapshots"]],
                        [s["validation"]["nll"] for s in report["snapshots"]], marker="o", label=label)
    axes[0, 0].set(xlabel="Processed training input tokens", ylabel="Common validation NLL", title="Measured loss versus token exposure")
    axes[0, 0].legend(fontsize=8)
    x = np.arange(len(names))
    axes[0, 1].bar(x, [reports[n]["parameters"] for n in names], color=COLORS["train"])
    axes[0, 1].set(ylabel="Total model parameters", title="Capacity is visible")
    for offset, split in ((-.18, "train"), (.18, "validation")):
        axes[1, 0].bar(x + offset, [reports[n]["snapshots"][-1][split]["nll"] for n in names], .36, label=split, color=COLORS[split])
    axes[1, 0].legend()
    axes[1, 0].set(ylabel="Final NLL (nats)", title="Training pools differ; validation is common")
    axes[1, 1].bar(x, [reports[n]["dataset_statistics"]["unique_examples"] for n in names], color=COLORS["function"])
    axes[1, 1].set(ylabel="Unique training sentences", title="New examples versus repeated exposure")
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xticks(x, labels, rotation=15, ha="right", fontsize=8)
    save_png(finish(fig, "Measured capacity, diversity and optimization scenarios", "Same seed, tokenizer and common held-out distribution. Larger/restricted changes two factors; these are scenarios, not isolated causal effects.\nLonger training spends more compute. Other scenarios match token exposure but not parameter-dependent computation."), folder / "generalization_comparison.png")


def render_leakage(folder, reports):
    folder = Path(folder)
    names = ["clm", "unmasked"]
    title = {"clm": "Correct causal mask", "unmasked": "INTENTIONALLY INCORRECT unmasked CLM"}
    maxima = max(s[split]["nll"] for name in names for s in reports[name]["snapshots"] for split in ("train", "validation"))

    def frame(index):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for column, name in enumerate(names):
            report = reports[name]
            snap = report["snapshots"][index]
            tokens = [report["corpus"]["vocabulary"][i] for i in snap["probe_inputs"][0]]
            heatmap(axes[0, column], np.asarray(snap["attention"])[0, 0], tokens, title[name] + f" · step {snap['step']}")
            for split in ("train", "validation"):
                history = report["snapshots"][:index + 1]
                axes[1, column].plot([s["step"] for s in history], [s[split]["nll"] for s in history], marker="o", color=COLORS[split], label=split)
            axes[1, column].set(xlim=(0, report["configuration"]["steps"]), ylim=(0, maxima * 1.05), xlabel="Updates", ylabel="Teacher-forced NLL")
            axes[1, column].legend()
        return finish(fig, "Future-token leakage can make prediction scores misleading", "Unmasked full-sequence inputs expose many next-token targets at later input positions. The causal model blocks those connections.\nBoth final models are also evaluated with only the available prefix; see the ablation plot and measured JSON for actual results.")

    save_gif(folder / "future_token_leakage.gif", range(len(reports["clm"]["snapshots"])), frame)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for name in names:
        report = reports[name]
        for split, style in (("train", "--"), ("validation", "-")):
            x, y = snapshot_xy(report, split, "nll")
            axes[0, 0].plot(x, y, linestyle=style, label=title[name] + " " + split)
        x, y = snapshot_xy(report, "validation", "accuracy")
        axes[0, 1].plot(x, y, label=title[name])
    axes[0, 0].set(xlabel="Updates", ylabel="Full-sequence teacher-forced NLL")
    axes[0, 1].set(xlabel="Updates", ylabel="Full-sequence teacher-forced accuracy", ylim=(0, 1))
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=7)
    for i, name in enumerate(names):
        report = reports[name]
        axes[1, 0].bar(i - .17, report["snapshots"][-1]["validation"]["accuracy"], .34, color=COLORS["train"], label="Full-sequence teacher forced" if i == 0 else None)
        axes[1, 0].bar(i + .17, report["prefix_only_validation"]["accuracy"], .34, color=COLORS["validation"], label="Prefix only" if i == 0 else None)
    axes[1, 0].set_xticks([0, 1], ["Causal", "Incorrect unmasked"])
    axes[1, 0].set(ylim=(0, 1), ylabel="Final validation accuracy")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].axis("off")
    descriptions = []
    for name in names:
        report = reports[name]
        vocab = report["corpus"]["vocabulary"]
        generated = [" ".join(vocab[i] for i in row) for row in report["generated_from_probe_prefixes"]]
        descriptions.append(title[name] + "\n" + "\n".join(generated) + f"\nPrefix-only validation NLL: {report['prefix_only_validation']['nll']:.3f}")
    axes[1, 1].text(0, .95, "Actual greedy continuations (fixed four-token prefixes):\n\n" + "\n\n".join(descriptions), va="top", fontsize=9, wrap=True)
    save_png(finish(fig, "Causal-mask ablation: identical initialization and training budget", "Final prefix-only evaluation scores every target with its preceding prefix for both models. Unmasked training is intentionally invalid for CLM.\nGreedy examples are individual observations; one seed and small synthetic grammar do not establish a general effect size."), folder / "causal_mask_ablation.png")


def render_scaling(folder, scaling=Scaling()):
    folder = Path(folder)
    parameters = np.logspace(6, 13, 100)
    tokens = np.logspace(7, 15, 100)
    nn, dd = np.meshgrid(parameters, tokens)
    loss = scaling_loss(nn, dd, scaling)
    fig, ax = plt.subplots(figsize=(11, 7))
    image = ax.contourf(np.log10(parameters), np.log10(tokens), loss, levels=30, cmap="viridis")
    contours = ax.contour(np.log10(parameters), np.log10(tokens), loss, levels=10, colors="white", linewidths=.6, alpha=.6)
    ax.clabel(contours, fontsize=7)
    for budget in (1e20, 1e22, 1e24):
        d = budget / (6 * parameters)
        valid = (d >= 1e7) & (d <= 1e15)
        ax.plot(np.log10(parameters[valid]), np.log10(d[valid]), label=f"C={budget:.0e} FLOPs", lw=2)
    ax.legend(fontsize=9)
    ax.set(xlabel="log10 model parameters", ylabel="log10 training tokens", title="Capacity-limited left · data-limited bottom · diminishing returns toward top-right")
    fig.colorbar(image, ax=ax, label="Illustrative loss")
    save_png(finish(fig, "Illustrative scaling law and fixed-compute contours", "Coefficients: " + str(asdict(scaling)) + "; N and D expressed in billions. No empirical fit.\nLoss = E + A/(N/1e9)^alpha + B/(D/1e9)^beta. Contours assume dense C≈6ND."), folder / "scaling_laws_heatmap.png")
    scaling_html(folder / "scaling_laws_3d.html", scaling)
    scaling_html(folder / "fixed_compute_tradeoffs.html", scaling, surface=False)


def render_compute(folder, scaling=Scaling()):
    folder = Path(folder)
    budgets = np.logspace(19, 25, 13)
    parameters = np.logspace(6, 13, 250)
    all_losses = np.array([scaling_loss(parameters, budget / (6 * parameters), scaling) for budget in budgets])
    y_max = float(all_losses.max()) * 1.03

    def frame(index):
        budget = budgets[index]
        optimum = optimal_allocation(float(budget), scaling)
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(parameters, all_losses[index], color=COLORS["train"])
        ax.scatter(optimum["parameters"], optimum["loss"], s=85, color=COLORS["validation"], label="Derived minimum")
        ax.set(xscale="log", xlim=(parameters.min(), parameters.max()), ylim=(scaling.E, y_max), xlabel="Model parameters", ylabel="Illustrative loss")
        ax.legend()
        ax.text(.03, .9, f"C={budget:.2e} FLOPs\nN*={optimum['parameters']/1e9:.2f}B · D*={optimum['tokens']/1e9:.2f}B\nD*/N*={optimum['tokens']/optimum['parameters']:.1f}", transform=ax.transAxes, va="top")
        return finish(fig, "Compute-optimal allocation under an assumed loss law", "D=C/(6N); minimum from differentiation of the normalized illustrative law, independently checked numerically.\nLeft: capacity limited. Right: token limited. This is analytical, not a result from miniature training.")

    save_gif(folder / "compute_optimal_allocation.gif", range(len(budgets)), frame, 550)
    allocation_html(folder / "compute_optimal_allocation.html", scaling)
    compute_html(folder / "training_compute_simulator.html")
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    labels = ["Gopher", "Chinchilla"]
    axes[0].bar(labels, [280, 70], color=[COLORS["train"], COLORS["validation"]])
    axes[0].set(ylabel="Parameters (billions)")
    axes[1].bar(labels, [300, 1400], color=[COLORS["train"], COLORS["validation"]])
    axes[1].set(ylabel="Training tokens (billions)")
    axes[2].bar(labels, [300 / 280, 1400 / 70], color=[COLORS["train"], COLORS["validation"]])
    axes[2].set(ylabel="Training tokens per parameter")
    save_png(finish(fig, "Historical allocation: published Gopher versus Chinchilla counts", "Source: Hoffmann et al. (2022), Table 1 (arXiv:2203.15556). Model parameter convention follows the paper.\nHistorical counts, not this lab's measurements; ~20 tokens/parameter is not a universal optimum. Dense 6ND is only approximate."), folder / "chinchilla_comparison.png")
    (folder / "analytical_assumptions.json").write_text(json.dumps({
        "scaling_coefficients": asdict(scaling), "coefficient_units": "N and D in billions", "empirical_fit": False,
        "optima": [{"compute_flops": float(c), **optimal_allocation(float(c), scaling)} for c in budgets],
        "historical_source": "https://arxiv.org/html/2203.15556v1#S1.T1",
        "historical_counts": {"Gopher": {"parameters": 280e9, "tokens": 300e9}, "Chinchilla": {"parameters": 70e9, "tokens": 1.4e12}}}, indent=2), encoding="utf-8")


def render_data(folder, reports):
    folder = Path(folder)
    names = ["clm", "duplicated", "single_grammar", "conflicting", "domain_imbalanced"]
    labels = ["Diverse", "Duplicated", "One grammar", "Role conflicts", "Domain imbalance"]
    x = np.arange(len(names))
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    stats = [reports[name]["dataset_statistics"] for name in names]
    axes[0, 0].bar(x - .18, [s["examples"] for s in stats], .36, label="Training pool rows", color=COLORS["special"])
    axes[0, 0].bar(x + .18, [s["unique_examples"] for s in stats], .36, label="Unique sentences", color=COLORS["train"])
    axes[0, 0].set(ylabel="Sentences", title="Equal pool size can contain less unique information")
    axes[0, 0].legend(fontsize=8)
    for offset, split in ((-.18, "train"), (.18, "validation")):
        axes[0, 1].bar(x + offset, [reports[n]["snapshots"][-1][split]["nll"] for n in names], .36, label=split, color=COLORS[split])
    axes[0, 1].set(ylabel="Final NLL", title="Common held-out validation distribution")
    axes[0, 1].legend()
    for offset, domain, color in ((-.18, "animal_examples", COLORS["train"]), (.18, "technical_examples", COLORS["validation"])):
        axes[1, 0].bar(x + offset, [s[domain] / s["examples"] for s in stats], .36, label=domain.replace('_examples', ''), color=color)
    axes[1, 0].set(ylabel="Fraction of corpus rows", ylim=(0, 1), title="Domain mixture")
    axes[1, 0].legend()
    counts = np.asarray([s["token_counts"] for s in stats])
    image = axes[1, 1].imshow(counts, aspect="auto", cmap="Blues")
    axes[1, 1].set_xticks(range(counts.shape[1]), reports["clm"]["corpus"]["vocabulary"], rotation=65, ha="right", fontsize=7)
    axes[1, 1].set_yticks(x, labels, fontsize=8)
    axes[1, 1].set_title("Actual corpus token frequencies")
    fig.colorbar(image, ax=axes[1, 1], label="Occurrences")
    for name, label in zip(names, labels):
        for ax, split in ((axes[2, 0], "train"), (axes[2, 1], "validation")):
            ax.plot(*snapshot_xy(reports[name], split, "nll"), marker=".", label=label)
            ax.set(xlabel="Updates", ylabel="NLL", title="Measured " + split + " loss")
            ax.legend(fontsize=7)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xticks(x, labels, rotation=15, ha="right", fontsize=8)
    save_png(finish(fig, "Synthetic data quality and diversity at matched token exposure", "Same architecture, seed, optimizer, batch size, steps and validation. Conditions alter duplication, grammar, role associations or domain mixture.\nAll corpora remain artificial; these single-seed observations do not identify internet-scale data-quality effects."), folder / "data_quality_comparison.png")
    corpus_html(reports, folder / "corpus_diversity.html")


def write_gallery(folder):
    folder = Path(folder)
    assets = sorted(p for p in folder.iterdir() if p.suffix in (".gif", ".png", ".html") and p.name != "index.html")
    cards = []
    for path in assets:
        name = html.escape(path.name)
        preview = f'<img src="{name}" loading="lazy" alt="{name}">' if path.suffix in (".gif", ".png") else '<span class="open">Open local interactive Plotly view →</span>'
        cards.append(f'<article><a href="{name}">{preview}<strong>{name}</strong></a></article>')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Day 73 · Pretraining laboratory</title>
<style>body{font:16px system-ui;background:#f1f5f9;color:#183247;margin:30px}h1{font-size:30px}p{max-width:1000px;line-height:1.6}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:20px}article{background:white;border:1px solid #dbe4ed;border-radius:10px;padding:15px}img{width:100%;height:240px;object-fit:contain}a{color:#246889;text-decoration:none}strong{display:block;margin:12px 0;overflow-wrap:anywhere}.open{display:block;padding:70px 20px;background:#e4f0f7}</style></head>
<body><h1>Day 73 · Interactive pretraining laboratory</h1><p>Follow the learning loop: objective → measured probabilities → embeddings and attention → generalization and leakage → analytical scale and compute. All training is synthetic. Scaling coefficients and hardware scenarios are illustrative. Interpretation: author review required. Generated artifacts remain local.</p><div class="grid">'''
    (folder / "index.html").write_text(page + "".join(cards) + "</div></body></html>", encoding="utf-8")
    manifest = [{"file": p.name, "bytes": p.stat().st_size, "classification": "regenerable artifact"} for p in assets]
    (folder / "assets_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
