"""Matplotlib exports and offline Plotly replay of recorded CPU computations."""

from __future__ import annotations

from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Circle, FancyArrowPatch
    import torch
except ImportError as error:
    raise ImportError('From the repository root: python -m pip install -e ".[dev,deep-learning]"') from error


COLORS = {"zero": "#277DA8", "one": "#DF7861", "train": "#277DA8", "validation": "#9B72AA",
          "test": "#798795", "loss": "#CF5D44", "gradient": "#9D5CB0", "update": "#178F83",
          "ink": "#213547", "background": "#F7F9FC"}
CMAP = LinearSegmentedColormap.from_list("binary_probability", ["#D0E6F2", "#FFFFFF", "#F4CFC5"])


def theme() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": COLORS["background"], "axes.facecolor": "white",
                         "text.color": COLORS["ink"], "axes.labelcolor": COLORS["ink"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": COLORS["background"]})


def finish(fig, title: str, operation: str, takeaway: str) -> None:
    if not hasattr(fig, "lab_captions"):
        fig.lab_captions = [fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98),
                            fig.text(0.5, 0.055, operation, ha="center", fontsize=10, color=COLORS["ink"]),
                            fig.text(0.5, 0.022, takeaway, ha="center", fontsize=10, color=COLORS["update"])]
    for artist, value in zip(fig.lab_captions, (title, operation, takeaway)):
        artist.set_text(value)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.86, bottom=0.18, wspace=0.33, hspace=0.48)


def save_png(fig, path: Path) -> None:
    fig.savefig(path, dpi=135)


def save_gif(fig, update, count: int, path: Path, fps: int = 2) -> None:
    animation = FuncAnimation(fig, update, frames=count, interval=1000 / fps,
                              repeat=True, blit=False, cache_frame_data=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=95)
    plt.close(fig)


def scatter_data(ax, data: dict, ids: np.ndarray, *, alpha: float = 0.7, size: int = 20) -> None:
    ax.scatter(data["x"][ids, 0], data["x"][ids, 1],
               c=np.where(data["y"][ids, 0] == 1, COLORS["one"], COLORS["zero"]),
               s=size, alpha=alpha, edgecolors="none")
    ax.set(xlabel="Feature 1 (training standard deviations)", ylabel="Feature 2 (training standard deviations)",
           xlim=(-3.1, 3.1), ylim=(-3.1, 3.1))
    ax.text(0.02, 0.02, "Blue: y=0 · Coral: y=1", transform=ax.transAxes, fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})


def probability_field(ax, data: dict, probability: np.ndarray) -> None:
    ax.contourf(data["axis"], data["axis"], probability, levels=np.linspace(0, 1, 12), cmap=CMAP, vmin=0, vmax=1)
    if probability.min() < 0.5 < probability.max():
        ax.contour(data["axis"], data["axis"], probability, levels=[0.5], colors=[COLORS["ink"]], linewidths=1.5)


def dataset_visual(lab: dict, folder: Path, frames: int, *, animate: bool = True) -> None:
    data, record = lab["data"], lab["mlp"]
    batch_size = lab["configuration"]["batch_size"]
    first_order = record["orders"][0]
    sequences = [(0, "Original training order", data["ids"]["train"], None),
                 (0, "DataLoader shuffle: epoch 1", first_order, None)]
    batches = int(np.ceil(len(first_order) / batch_size))
    for batch in np.unique(np.linspace(0, batches - 1, max(2, frames - 2), dtype=int)):
        sequences.append((int(batch), "Active training mini-batch", first_order,
                          first_order[batch * batch_size:(batch + 1) * batch_size]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.1))

    def draw(index):
        for ax in axes:
            ax.clear()
        batch, label, order, active = sequences[index]
        for name, marker in (("train", "o"), ("validation", "^"), ("test", "s")):
            ids = data["ids"][name]
            axes[0].scatter(data["x"][ids, 0], data["x"][ids, 1], color=COLORS[name], marker=marker,
                            alpha=0.35, s=22, label=f"{name}: {len(ids)}" + (" — no updates" if name != "train" else ""))
        if active is not None:
            axes[0].scatter(data["x"][active, 0], data["x"][active, 1], facecolors="none",
                            edgecolors=COLORS["update"], linewidths=2, s=80, label=f"active: {len(active)}")
        axes[0].set(xlabel="Feature 1 (standardized)", ylabel="Feature 2 (standardized)",
                    title="Partitions: only training rows enter backward()", xlim=(-3.2, 3.2), ylim=(-3.2, 3.2))
        axes[0].legend(fontsize=8, loc="upper left")
        # A tile is an observation ID, not a tensor element or an invented sample.
        width = 12
        start = 0 if active is None else min((batch * batch_size // width) * width, max(0, len(order) - 96))
        shown = order[start:start + 96]
        padded = np.full(int(np.ceil(len(shown) / width)) * width, np.nan)
        padded[:len(shown)] = shown
        image = padded.reshape(-1, width)
        colors = np.full_like(padded, 0.25)
        colors[len(shown):] = np.nan
        if active is not None:
            colors[np.isin(padded, active)] = 1
        axes[1].imshow(colors.reshape(image.shape), cmap="GnBu", vmin=0, vmax=1, aspect="auto")
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                if np.isfinite(image[row, col]):
                    axes[1].text(col, row, str(int(image[row, col])), ha="center", va="center", fontsize=8,
                                 color="white" if colors.reshape(image.shape)[row, col] == 1 else COLORS["ink"])
        axes[1].set(title=f"{label}\nEpoch 1 | batch {batch + 1 if active is not None else '—'}/{batches} | size {batch_size}",
                    xlabel=f"Slots {start + 1}–{start + len(shown)} of {len(order)} (read left to right)", ylabel="Consecutive slots in DataLoader order")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        finish(fig, "01 · Dataset → DataLoader → active batch", "TensorDataset → DataLoader(shuffle=True) → for X, y, ids in train_loader",
               "Shuffling changes training order; validation and test IDs stay outside gradient updates.")

    draw(min(2, len(sequences) - 1))
    save_png(fig, folder / "01_dataset_overview.png")
    if animate:
        save_gif(fig, draw, len(sequences), folder / "01_dataset_batches.gif")
    else:
        plt.close(fig)


def forward_visual(lab: dict, folder: Path) -> None:
    event = lab["mlp"]["events"][0]
    sample, state = event["sample"], event["state_before"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})
    stages = ["Input x", "z₁ = W₁x + b₁", "h = ReLU(z₁)", "z₂ = W₂h + b₂", "p = sigmoid(z₂)"]

    def draw(frame):
        stage = frame // 2
        ax, detail = axes
        ax.clear(); detail.clear()
        positions = [[(0, y) for y in (0.35, -0.35)], [(1, y) for y in np.linspace(1.35, -1.35, 8)], [(2, 0)]]
        for i, source in enumerate(positions[0]):
            for j, target in enumerate(positions[1]):
                weight = state["hidden.weight"][j, i]
                ax.plot([source[0], target[0]], [source[1], target[1]],
                        color=COLORS["zero"] if weight >= 0 else COLORS["one"],
                        linewidth=0.5 + abs(weight) * 2, alpha=0.4)
        for j, source in enumerate(positions[1]):
            weight = state["output.weight"][0, j]
            ax.plot([1, 2], [source[1], 0], color=COLORS["zero"] if weight >= 0 else COLORS["one"],
                    linewidth=0.5 + abs(weight) * 3, alpha=0.5)
        values = [sample["x"], sample["z1"] if stage == 1 else sample["hidden"], [sample["logit"]]]
        for layer, points in enumerate(positions):
            highlighted = layer == (0 if stage == 0 else 1 if stage < 3 else 2)
            for j, (x, y) in enumerate(points):
                ax.add_patch(Circle((x, y), 0.12, color=COLORS["update"] if highlighted else "#DEE6EF", zorder=5))
                ax.text(x, y, f"{values[layer][j]:.2f}", fontsize=8, ha="center", va="center", zorder=6)
        ax.set(xlim=(-0.35, 2.35), ylim=(-1.65, 1.65), title=f"2 → 8 → 1 MLP | {stages[stage]}")
        ax.axis("off")
        detail.imshow(state["hidden.weight"], cmap="coolwarm_r", vmin=-0.8, vmax=0.8, aspect="auto")
        for j in range(8):
            for i in range(2):
                detail.text(i, j, f"{state['hidden.weight'][j, i]:+.3f}", ha="center", va="center", fontsize=9)
        detail.set(xticks=[0, 1], xticklabels=["x₁", "x₂"], yticks=range(8),
                   ylabel="Hidden neuron", title="Measured W₁ values (edge sign / thickness)")
        finish(fig, "02 · Forward propagation: measured tensors, conceptual highlighting",
               f"{stages[stage]} | logit={sample['logit']:.4f} | p(y=1)={sample['probability']:.4f} | target={sample['y']:.0f}",
               "Each highlighted layer denotes a tensor operation; connections do not transmit particles.")

    save_gif(fig, draw, 10, folder / "02_forward_pass.gif", fps=2)


def loss_visual(folder: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.7))
    logits = torch.linspace(-7, 7, 300, dtype=torch.float64)
    probabilities = torch.sigmoid(logits)
    for y, color in ((0., COLORS["zero"]), (1., COLORS["one"])):
        losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.full_like(logits, y), reduction="none")
        axes[0].plot(probabilities, losses, color=color, linewidth=2, label=f"target y={y:.0f}")
    examples = torch.tensor([-4., 0., 4.], dtype=torch.float64)
    p = torch.sigmoid(examples).numpy()
    measured = torch.nn.functional.binary_cross_entropy_with_logits(examples, torch.ones_like(examples), reduction="none").numpy()
    axes[0].scatter(p, measured, c=COLORS["one"], s=60, zorder=4)
    axes[0].set(xlabel="Predicted p(y=1) = sigmoid(logit)", ylabel="Binary cross-entropy (nats / example)",
                title="Confident errors receive a large penalty", ylim=(0, 7))
    axes[0].legend()
    axes[1].axis("off")
    rows = [[label, f"{z:+.0f}", f"{prob:.3f}", f"{loss:.3f}", f"{prob-1:+.3f}"]
            for label, z, prob, loss in zip(["Wrong", "Uncertain", "Correct"], examples, p, measured)]
    table = axes[1].table(cellText=rows, colLabels=["y=1", "logit", "p", "BCE", "dℓ/dz"], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.1, 2.2)
    axes[1].set_title("Values evaluated with PyTorch (raw logits)")
    finish(fig, "03 · Binary cross-entropy: probability, confidence and error",
           "loss = BCEWithLogitsLoss(logits, targets) | dℓ/dz = sigmoid(z) − y",
           "The optimizer minimizes logit loss; sigmoid is shown for interpretation, not applied before this loss.")
    save_png(fig, folder / "03_binary_cross_entropy.png"); plt.close(fig)


def backward_visual(lab: dict, folder: Path) -> None:
    event = lab["mlp"]["events"][0]
    sample = event["sample"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), gridspec_kw={"height_ratios": [1.2, 1]})
    labels = ["x → W₁x+b₁", "h = ReLU(z₁)", "z = W₂h+b₂", "mean BCE"]

    def draw(frame):
        stage = frame // 2
        graph, bars = axes
        graph.clear(); bars.clear()
        graph.set(xlim=(-0.5, 3.5), ylim=(-0.8, 0.9)); graph.axis("off")
        for i, label in enumerate(labels):
            graph.text(i, 0.3, label, ha="center", fontsize=11,
                       bbox={"boxstyle": "round,pad=0.6", "fc": "#E5ECF4", "ec": "none"})
            if i < 3:
                graph.add_patch(FancyArrowPatch((i + 0.28, 0.3), (i + 0.7, 0.3), arrowstyle="->", mutation_scale=15,
                                                color=COLORS["update"] if stage == 0 else "#B8C6D4"))
        if stage >= 1:
            for i in range(3, 0, -1):
                graph.add_patch(FancyArrowPatch((i - 0.2, -0.15), (i - 0.8, -0.15), arrowstyle="->", mutation_scale=17,
                                                color=COLORS["gradient"]))
        explanation = (f"Measured logit z₀={sample['logit']:.4f} | batch mean BCE={event['batch_loss_before']:.4f}\n"
                       "Forward computes values; no backward gradients have been calculated in this stage.") if stage == 0 else (
                       f"Actual batch-mean adjoint: ∂L/∂z₀ = {sample['dlogit']:+.5f} = (p₀−y₀)/{len(event['ids'])}\n"
                       f"||∂L/∂h₀||₂={np.linalg.norm(sample['dhidden']):.5f} | ||∂L/∂z₁,₀||₂={np.linalg.norm(sample['dz1']):.5f}")
        graph.text(1.5, -0.6, explanation, ha="center", fontsize=10)
        norms = event["gradient_norms"] if stage < 3 else event["update_norms"]
        if stage == 0:
            bars.axis("off")
            bars.text(0.5, 0.5, "Next: loss.backward() populates parameter.grad\noptimizer.step() has not run", ha="center", transform=bars.transAxes)
        else:
            bars.bar(list(norms), list(norms.values()), color=COLORS["gradient"] if stage < 3 else COLORS["update"])
            bars.set(ylabel="L2 norm (loss / parameter units)" if stage < 3 else "L2 norm (parameter units)",
                     title="Measured gradient norms" if stage < 3 else "Measured parameter-update norms (Adam)")
        phase = ["Forward: parameters fixed", "loss.backward(): reverse-mode chain rule",
                 "Gradients exist; parameters are still unchanged", "optimizer.step(): parameters now change"][stage]
        finish(fig, "04 · Backpropagation computes gradients; the optimizer applies updates", phase,
               "Arrows explain graph direction; displayed numbers are actual autograd values from one mini-batch.")

    save_gif(fig, draw, 8, folder / "04_backward_pass.gif", fps=1)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    rows = lab["mlp"]["gradient_history"]
    for layer in ("hidden", "output"):
        ax.plot([r["step"] for r in rows], [r[layer] for r in rows], label=layer, alpha=0.8)
    ax.set(xlabel="Actual optimizer step", ylabel="L2 norm of weight + bias gradients", title="Measured gradients over training (pre-update)")
    ax.legend()
    finish(fig, "04 · Gradient magnitudes across layers", "loss.backward() → parameter.grad → L2 norm per layer",
           "A small gradient alone does not diagnose vanishing gradients or guarantee convergence.")
    save_png(fig, folder / "04_gradient_norms.png"); plt.close(fig)


def landscape_contour(lab: dict, folder: Path) -> None:
    record = lab["logistic"]
    run = record["runs"][record["primary"]]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    contour = ax.contourf(record["weights"], record["biases"], record["surface"], levels=25, cmap="Blues_r")
    fig.colorbar(contour, ax=ax, label="Full fixed-dataset BCE (nats / example)")
    path = run["path"]
    ax.plot(path[:, 0], path[:, 1], color=COLORS["update"], linewidth=1.5)
    selected = np.unique(np.linspace(0, len(path) - 2, 14, dtype=int))
    delta = np.diff(path, axis=0)
    ax.quiver(path[selected, 0], path[selected, 1], delta[selected, 0], delta[selected, 1],
              angles="xy", scale_units="xy", scale=1, color=COLORS["update"], width=0.005)
    ax.scatter(*path[0], c=COLORS["loss"], marker="s", s=80, label="initial")
    ax.scatter(*path[-1], c=COLORS["update"], marker="*", s=170, label="final")
    ax.set(xlabel="Weight w", ylabel="Bias b", title=f"{record['primary']} | {run['uphill_steps']} full-loss increases retained")
    ax.legend()
    finish(fig, "05 · Actual two-parameter logistic loss landscape", "z = w·x + b | surface and path use the same 80-row mean BCE objective",
           "Arrow lengths are actual parameter deltas; mini-batch updates need not lower the full-dataset loss.")
    save_png(fig, folder / "05_loss_contour.png"); plt.close(fig)


def optimizer_visual(lab: dict, folder: Path) -> None:
    record = lab["logistic"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    palette = plt.get_cmap("tab10")
    for i, (name, run) in enumerate(record["runs"].items()):
        color = palette(i // 2)
        style = "-" if run["batch_size"] == 8 else "--"
        axes[0].plot(run["path"][:, 0], run["path"][:, 1], style, color=color, label=name)
        axes[1].plot(run["loss"], style, color=color)
        axes[2].plot(np.arange(1, len(run["path"])), np.linalg.norm(np.diff(run["path"], axis=0), axis=1), style, color=color)
    axes[0].set(xlabel="Weight w", ylabel="Bias b", title="Identical initialization: w=−1.5, b=1")
    axes[1].set(xlabel="Optimizer steps", ylabel="Full-dataset mean BCE (nats)", title="Actual objective after every step")
    axes[2].set(xlabel="Optimizer steps", ylabel="||parameter delta||₂", title="Actual update length")
    axes[0].legend(fontsize=7, loc="best")
    finish(fig, "06 · Optimizer, learning rate and batch size change the path",
           "SGD and Adam | 2 rates each × batches 8 / 80 | 64 updates each | solid=mini-batch; dashed=full batch",
           "Equal update budgets process different numbers of examples; these rates do not establish a universal winner.")
    save_png(fig, folder / "06_optimizer_comparison.png"); plt.close(fig)


def boundary_visual(lab: dict, folder: Path, frames: int, *, animate: bool = True) -> None:
    data, model = lab["data"], lab["mlp"]
    selected = np.unique(np.linspace(0, len(model["checkpoints"]) - 1, frames, dtype=int))
    fig, ax = plt.subplots(figsize=(9.5, 6.6))

    def draw(frame):
        ax.clear()
        index = selected[frame]
        point, metrics = model["checkpoints"][index], model["history"][index]
        probability_field(ax, data, point["grid_probability"])
        scatter_data(ax, data, data["ids"]["train"])
        ax.set_title(f"Epoch {point['epoch']:d} | step {point['step']:d} | frozen train BCE {metrics['train_loss']:.3f} | val {metrics['validation_loss']:.3f}")
        finish(fig, "07 · Training changes the represented decision function", "model(X_grid) → sigmoid(logits) | dark contour: p(y=1)=0.5 | dots: true training labels",
               "Backgrounds are actual checkpoint predictions on a fixed grid; no curve or boundary interpolation.")

    if animate:
        save_gif(fig, draw, len(selected), folder / "07_boundary_evolution.gif", fps=2)
    else:
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, name in zip(axes, ("linear", "mlp")):
        run = lab[name]
        epoch = run["best_epoch"]
        probability_field(ax, data, run["checkpoints"][epoch]["grid_probability"])
        scatter_data(ax, data, data["ids"]["train"], size=16)
        ax.set_title(f"{name.upper()} | selected epoch {epoch}\nvalidation BCE {run['history'][epoch]['validation_loss']:.3f}")
    finish(fig, "07 · Linear and nonlinear models on the same split and update budget",
           "Training dots only | each model restored to its own minimum-validation-BCE epoch",
           "The models differ in capacity and initial functions; one synthetic run is not a general architecture ranking.")
    save_png(fig, folder / "07_final_comparison.png"); plt.close(fig)


def training_visual(lab: dict, folder: Path) -> None:
    run = lab["mlp"]
    rows, best = run["history"], run["best_epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7))
    epoch = [r["epoch"] for r in rows]
    axes[0].plot(epoch, [r["train_loss"] for r in rows], color=COLORS["train"], label="frozen end-of-epoch train")
    axes[0].plot(epoch, [r["validation_loss"] for r in rows], color=COLORS["validation"], label="validation")
    axes[0].plot(epoch[1:], [r["online_loss"] for r in rows[1:]], "--", color=COLORS["loss"], label="online train (changing parameters)", alpha=0.7)
    axes[0].axvline(best, color=COLORS["update"], linestyle=":", label=f"selected checkpoint: epoch {best}")
    axes[0].set(xlabel="Epoch (0 = initialization)", ylabel="Mean BCE (nats / example)", title="Actual training and validation losses")
    axes[0].legend(fontsize=8)
    gap = [r["validation_loss"] - r["train_loss"] for r in rows]
    axes[1].plot(epoch, gap, color=COLORS["gradient"])
    axes[1].axhline(0, color="grey", linewidth=1)
    axes[1].scatter(best, gap[best], c=COLORS["update"], s=70)
    axes[1].set(xlabel="Epoch", ylabel="Validation BCE − frozen train BCE (nats)", title="Observed population gap (signed)")
    finish(fig, "08 · Validation selects a checkpoint; online training loss has different semantics",
           "train() + updates during epoch → eval() + inference_mode() at a fixed checkpoint",
           "A positive gap can have several causes; these observations alone do not establish overfitting.")
    save_png(fig, folder / "08_training_validation.png"); plt.close(fig)


def mode_visual(lab: dict, folder: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    rows = []
    for i, (name, values) in enumerate(lab["dropout"].items()):
        axes[0].plot(np.arange(1, 13), values["probability"], marker="o", markersize=4, label=name,
                     linestyle="--" if values["inference"] else "-", alpha=0.85)
        rows.append([name, "yes" if values["training"] else "no", "yes" if values["requires_grad"][0] else "no"])
    axes[0].set(xlabel="Repeated pass of the same input", ylabel="Predicted p(y=1)",
                title="A seeded, untrained Dropout(p=0.5) network", ylim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[1].axis("off")
    table = axes[1].table(cellText=rows, colLabels=["Mode + context", "Dropout active", "logits require grad"], loc="center", cellLoc="center",
                          colWidths=[0.5, 0.3, 0.35])
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 2.2)
    axes[1].set_title("Two independent switches")
    finish(fig, "09 · Evaluation mode and autograd recording control different mechanisms",
           "model.train()/eval() controls Dropout | torch.inference_mode() disables gradient recording",
           "Inference mode does not turn Dropout off; eval mode does not disable gradients.")
    save_png(fig, folder / "09_train_eval_modes.png"); plt.close(fig)


def plotly_modules():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise ImportError('Plotly is a canonical base dependency: python -m pip install -e ".[dev,deep-learning]"') from error
    return go, make_subplots


def html_export(fig, path: Path, explanation: str) -> None:
    # Plotly rejects interrupted animation promises without an error value.
    # Own only the replay calls, so expected cancellation is handled locally
    # while actual errors remain visible. No global rejection filter is used.
    replay_script = """
    const recordedPlot = document.getElementById('{plot_id}');
    function replayFailure(error) {
        if (error !== undefined) {
            recordedPlot.dataset.replayError = String(error);
            console.error('Recorded replay failed:', error);
        }
    }
    function recordedAnimate(args) {
        Plotly.animate(recordedPlot, ...args).catch(replayFailure);
    }
    recordedPlot.on('plotly_buttonclicked', event => {
        if (event.button.method === 'animate') recordedAnimate(event.button.args);
    });
    recordedPlot.on('plotly_sliderchange', event => {
        if (event.interaction && event.step.method === 'animate') recordedAnimate(event.step.args);
    });
    recordedPlot.on('plotly_animatingframe', event => {
        const slider = recordedPlot.layout.sliders && recordedPlot.layout.sliders[0];
        if (!slider) return;
        const index = slider.steps.findIndex(step => step.label === event.name);
        if (index >= 0) Plotly.relayout(recordedPlot, {'sliders[0].active': index}).catch(replayFailure);
    });
    """
    body = fig.to_html(full_html=False, include_plotlyjs=True, auto_play=False,
                       config={"displaylogo": False, "responsive": True, "scrollZoom": True,
                               "toImageButtonOptions": {"format": "png"}},
                       div_id="training-lab", post_script=replay_script)
    payload = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width,initial-scale=1">'
               '<title>Day 62 · PyTorch training lab</title><style>'
               'body{font-family:Arial,sans-serif;margin:20px;background:#F7F9FC;color:#213547}'
               'p{max-width:1150px;line-height:1.5} .plotly-graph-div{min-height:760px}'
               '</style></head><body><h1>Day 62 · Recorded training mechanics</h1><p>'
               + explanation + '</p>' + body + '</body></html>')
    if len(payload.encode("utf-8")) > 12 * 1024**2:
        raise ValueError("HTML exceeds 12 MiB; reduce --frames or --grid-size")
    path.write_text(payload, encoding="utf-8")


def replay_controls(names: list[str], label: str) -> dict:
    return {"updatemenus": [{"type": "buttons", "direction": "left", "x": 0, "y": -0.08,
                             "buttons": [{"label": "Play", "method": "animate", "execute": False, "args": [None, {"frame": {"duration": 650, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]},
                                         {"label": "Pause", "method": "animate", "execute": False, "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]}]}],
            "sliders": [{"active": 0, "currentvalue": {"prefix": label}, "pad": {"t": 45},
                         "steps": [{"label": name, "method": "animate", "execute": False, "args": [[name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]} for name in names]}]}


def landscape_html(lab: dict, folder: Path, frames: int) -> None:
    go, _ = plotly_modules()
    record = lab["logistic"]
    run = record["runs"][record["primary"]]
    path, losses = run["path"], run["loss"]
    selected = np.unique(np.linspace(0, len(path) - 1, frames, dtype=int))
    fig = go.Figure(go.Surface(x=record["weights"], y=record["biases"], z=record["surface"], opacity=0.72,
                              colorscale="Blues", colorbar={"title": "Full BCE"}, name="fixed objective"))
    fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=losses, mode="lines+markers", name="actual optimizer path",
                               marker={"size": 3, "color": np.arange(len(path)), "colorscale": "Viridis"}, line={"color": COLORS["update"], "width": 5},
                               text=[f"step {i} | full BCE {value:.5f}" for i, value in enumerate(losses)], hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=[path[0, 0]], y=[path[0, 1]], z=[losses[0]], mode="markers", name="initial",
                               marker={"size": 7, "color": COLORS["loss"], "symbol": "square"}))
    fig.add_trace(go.Scatter3d(x=[path[-1, 0]], y=[path[-1, 1]], z=[losses[-1]], mode="markers", name="final",
                               marker={"size": 7, "color": COLORS["update"]}))
    fig.add_trace(go.Scatter3d(x=[path[0, 0]], y=[path[0, 1]], z=[losses[0]], mode="markers", name="selected update",
                               marker={"size": 9, "color": COLORS["gradient"]}))
    frames_list = []
    for step in selected:
        delta = np.zeros(2) if step == 0 else path[step] - path[step - 1]
        frames_list.append(go.Frame(name=str(step), traces=[4], data=[go.Scatter3d(
            x=[path[step, 0]], y=[path[step, 1]], z=[losses[step]],
            text=[f"step {step}: Δw={delta[0]:+.4f}, Δb={delta[1]:+.4f}, ||Δθ||={np.linalg.norm(delta):.4f}"], hoverinfo="text")]))
    fig.frames = frames_list
    fig.update_layout(title=f"05 · {record['primary']} · {run['uphill_steps']} full-loss increases retained",
                      scene={"xaxis_title": "Weight w", "yaxis_title": "Bias b", "zaxis_title": "Full fixed-dataset mean BCE (nats)", "uirevision": "keep-camera"},
                      legend={"orientation": "h", "x": 0, "y": 1.02, "font": {"size": 11}},
                      height=760, margin={"b": 130, "r": 100}, **replay_controls([f.name for f in fig.frames], "Actual update: "))
    html_export(fig, folder / "05_loss_landscape.html",
                "Rotate and zoom the surface, then drag the step slider or press Play. "
                "The surface and every trajectory height use the same 80-row logistic BCE objective. "
                "The path is measured mini-batch SGD; full-loss increases are retained. "
                "This two-parameter teaching model is separate from the MLP. Hover over the selected marker for update size.")


def optimizer_html(lab: dict, folder: Path) -> None:
    go, make_subplots = plotly_modules()
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Parameter trajectory", "Full-dataset loss after each update"])
    names = list(lab["logistic"]["runs"])
    for name, run in lab["logistic"]["runs"].items():
        visible = name == lab["logistic"]["primary"]
        fig.add_trace(go.Scatter(x=run["path"][:, 0], y=run["path"][:, 1], mode="lines+markers", name=name, visible=visible), row=1, col=1)
        fig.add_trace(go.Scatter(y=run["loss"], mode="lines", name=name, visible=visible, showlegend=False), row=1, col=2)
    buttons = []
    for i, name in enumerate(names):
        visible = [j // 2 == i for j in range(2 * len(names))]
        buttons.append({"label": name, "method": "update", "args": [{"visible": visible}, {"title": {"text": f"06 · {name}"}}]})
    fig.update_layout(height=740, title=f"06 · {lab['logistic']['primary']}",
                      updatemenus=[{"buttons": buttons, "active": names.index(lab["logistic"]["primary"]),
                                    "x": 1.0, "xanchor": "right", "y": 1.16}], margin={"t": 140})
    record = lab["logistic"]
    fig.update_xaxes(title_text="Weight w", range=[float(record["weights"][0]), float(record["weights"][-1])], row=1, col=1)
    fig.update_yaxes(title_text="Bias b", range=[float(record["biases"][0]), float(record["biases"][-1])], row=1, col=1)
    fig.update_xaxes(title_text="Optimizer step", range=[0, 64], row=1, col=2)
    fig.update_yaxes(title_text="Mean BCE (nats)", range=[0, max(float(r["loss"].max()) for r in record["runs"].values()) * 1.05], row=1, col=2)
    html_export(fig, folder / "06_optimizer_controls.html",
                "Choose optimizer, learning rate and batch size from the menu. All eight prerecorded configurations "
                "start at w=−1.5, b=1 and use the same fixed synthetic data. Equal update counts process different "
                "numbers of examples. These illustrative rates establish no universal optimizer ranking.")


def dashboard_html(lab: dict, folder: Path) -> None:
    go, make_subplots = plotly_modules()
    data, events = lab["data"], lab["mlp"]["events"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=["Post-update decision function", "Active batch: true labels", "Batch probabilities: before → after",
                                                                  "Pre-update autograd norms", "Actual parameter-update norms", "Same-step loss measurements"],
                        horizontal_spacing=0.09, vertical_spacing=0.18)
    train = data["ids"]["train"]
    fig.add_trace(go.Contour(x=data["axis"], y=data["axis"], z=events[0]["grid_probability"], zmin=0, zmax=1,
                             colorscale=[[0, "#D0E6F2"], [0.5, "white"], [1, "#F4CFC5"]], showscale=False,
                             contours={"start": 0, "end": 1, "size": 0.1}, name="p(y=1) after step"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["x"][train, 0], y=data["x"][train, 1], mode="markers", showlegend=False,
                             marker={"size": 4, "color": data["y"][train, 0], "colorscale": [[0, COLORS["zero"]], [1, COLORS["one"]]], "cmin": 0, "cmax": 1}), row=1, col=1)
    first = events[0]
    fig.add_trace(go.Scatter(x=first["x"][:, 0], y=first["x"][:, 1], mode="markers", name="active training batch",
                             marker={"size": 9, "color": first["y"][:, 0], "colorscale": [[0, COLORS["zero"]], [1, COLORS["one"]]], "cmin": 0, "cmax": 1}), row=1, col=2)
    sigmoid = lambda z: torch.sigmoid(torch.as_tensor(z)).numpy()
    fig.add_trace(go.Bar(x=list(range(len(first["ids"]))), y=sigmoid(first["logits_before"][:, 0]), name="before step", marker_color=COLORS["gradient"]), row=1, col=3)
    fig.add_trace(go.Bar(x=list(range(len(first["ids"]))), y=sigmoid(first["logits_after"][:, 0]), name="after step", marker_color=COLORS["update"]), row=1, col=3)
    fig.add_trace(go.Bar(x=list(first["gradient_norms"]), y=list(first["gradient_norms"].values()), name="gradient", marker_color=COLORS["gradient"]), row=2, col=1)
    fig.add_trace(go.Bar(x=list(first["update_norms"]), y=list(first["update_norms"].values()), name="parameter delta", marker_color=COLORS["update"]), row=2, col=2)
    keys = [("batch_loss_before", COLORS["loss"], "batch before"), ("batch_loss_after", COLORS["update"], "batch after"),
            ("train", COLORS["train"], "full train after"), ("validation", COLORS["validation"], "validation after")]
    for _, color, label in keys:
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name=label, line={"color": color}), row=2, col=3)
    frames = []
    for i, event in enumerate(events):
        traces = [go.Contour(z=event["grid_probability"]),
                  go.Scatter(x=event["x"][:, 0], y=event["x"][:, 1], marker={"color": event["y"][:, 0]}),
                  go.Bar(x=list(range(len(event["ids"]))), y=sigmoid(event["logits_before"][:, 0])),
                  go.Bar(x=list(range(len(event["ids"]))), y=sigmoid(event["logits_after"][:, 0])),
                  go.Bar(y=list(event["gradient_norms"].values())), go.Bar(y=list(event["update_norms"].values()))]
        for key, _, _ in keys:
            values = [e[key]["loss"] if isinstance(e[key], dict) else e[key] for e in events[:i+1]]
            traces.append(go.Scatter(x=[e["step"] for e in events[:i+1]], y=values))
        frames.append(go.Frame(name=str(event["step"]), data=traces, traces=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                               layout={"title": {"text": f"10 · Epoch {event['epoch']} · update {event['step']} · batch {event['batch_index']}/{event['batches_in_epoch']} · BCE {event['batch_loss_before']:.4f} → {event['batch_loss_after']:.4f}"}}))
    fig.frames = frames
    fig.update_layout(title=f"10 · Epoch {first['epoch']} · update {first['step']} · batch {first['batch_index']}/{first['batches_in_epoch']} · BCE {first['batch_loss_before']:.4f} → {first['batch_loss_after']:.4f}", height=850,
                      barmode="group", legend={"orientation": "h", "y": -0.22}, margin={"t": 100, "b": 200},
                      **replay_controls([f.name for f in frames], "Recorded optimizer step: "))
    for col in (1, 2):
        fig.update_xaxes(title_text="Standardized feature 1", range=[-3, 3], row=1, col=col)
        fig.update_yaxes(title_text="Standardized feature 2", range=[-3, 3], row=1, col=col)
    fig.update_xaxes(title_text="Slot in active batch", range=[-1, lab["configuration"]["batch_size"]], row=1, col=3)
    fig.update_yaxes(title_text="p(y=1)", range=[0, 1], row=1, col=3)
    for col in (1, 2):
        field = "gradient_norms" if col == 1 else "update_norms"
        maximum = max(max(e[field].values()) for e in events)
        fig.update_yaxes(title_text="L2 norm", range=[0, maximum * 1.15], row=2, col=col)
    fig.update_xaxes(title_text="Actual optimizer step", range=[0, events[-1]["step"]], row=2, col=3)
    max_loss = max(max(e["batch_loss_before"], e["batch_loss_after"], e["train"]["loss"], e["validation"]["loss"]) for e in events)
    fig.update_yaxes(title_text="Mean BCE (nats)", range=[0, max_loss * 1.1], row=2, col=3)
    # Populate the initial curves so the page is meaningful even before playback.
    for trace, (key, _, _) in zip(fig.data[7:], keys):
        trace.x = [first["step"]]
        trace.y = [first[key]["loss"] if isinstance(first[key], dict) else first[key]]
    html_export(fig, folder / "10_training_dashboard.html",
                "Drag the step slider or press Play. Every frame is ONE recorded update: batch predictions and gradients "
                "come from before optimizer.step(); parameter deltas, the boundary, full training loss and validation loss "
                "come from immediately after that same step. Validation is inference-only. Colors: blue=class 0 / training, "
                "coral=class 1 / loss, purple=gradients / validation, teal=updates. The slider skips unrecorded steps; "
                "it never interpolates a training result. Use the separate epoch chart for online aggregated loss and model selection.")


def dashboard_static(lab: dict, folder: Path) -> None:
    event = lab["mlp"]["events"][-1]
    data = lab["data"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    probability_field(axes[0, 0], data, event["grid_probability"])
    scatter_data(axes[0, 0], data, event["ids"], size=50)
    axes[0, 0].set_title("Post-update function + active batch")
    axes[0, 1].bar(event["gradient_norms"].keys(), event["gradient_norms"].values(), color=COLORS["gradient"])
    axes[0, 1].set(title="Pre-update autograd gradient norms", ylabel="L2 norm")
    axes[1, 0].bar(event["update_norms"].keys(), event["update_norms"].values(), color=COLORS["update"])
    axes[1, 0].set(title="Actual parameter deltas", ylabel="L2 norm")
    events = lab["mlp"]["events"]
    for key, color in (("train", COLORS["train"]), ("validation", COLORS["validation"])):
        axes[1, 1].plot([e["step"] for e in events], [e[key]["loss"] for e in events], color=color, label=f"{key}: post-update")
    axes[1, 1].set(xlabel="Recorded optimizer step", ylabel="Mean BCE (nats)", title="Fixed-state evaluations after recorded updates")
    axes[1, 1].legend(fontsize=8)
    finish(fig, f"10 · Same-update dashboard | epoch {event['epoch']} | update {event['step']}",
           f"Active batch BCE: {event['batch_loss_before']:.4f} before → {event['batch_loss_after']:.4f} after | n={len(event['ids'])}",
           "This is one actual event; open the standalone HTML to scrub earlier updates.")
    save_png(fig, folder / "10_dashboard_overview.png"); plt.close(fig)
