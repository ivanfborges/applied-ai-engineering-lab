"""Matplotlib views and real widget callbacks; backend is selected by the CLI."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.widgets import Button, Slider
from sklearn.metrics import precision_recall_curve, roc_curve

from visual_core import LabExperiments, infer, metrics_at


NORMAL, ANOMALY, RECON, LATENT = "#087f8c", "#d65b32", "#3766aa", "#8058a5"
GROUPS = {"normal": NORMAL, "off_subspace": ANOMALY, "on_subspace": "#a94072",
          "normal_like_anomaly": "#b58900", "unusual_normal": "#555e69", "off_manifold": ANOMALY}


def style():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "figure.facecolor": "#f8fafc", "axes.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.16, "legend.fontsize": 8,
                         "savefig.facecolor": "#f8fafc"})


def figure(title: str, footer: str, rows=1, columns=2, size=(11, 6.5)):
    fig, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.14, top=0.85, hspace=0.6, wspace=0.36)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97)
    fig.text(0.5, 0.035, footer, ha="center", fontsize=8, color="#475569")
    return fig, axes.ravel()


def limits(values, padding=0.12):
    values = np.asarray(values)
    lo, hi = float(np.min(values)), float(np.max(values))
    spread = max(hi - lo, 0.1)
    return lo - spread * padding, hi + spread * padding


def export_animation(fig, update, frames: int, path: Path, *, fps=3, dpi=90):
    """Pillow buffers a bounded number of frames; no GUI or FFmpeg is required."""
    if frames < 2 or path.exists():
        raise ValueError("Use at least two frames and a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = PillowWriter(fps=fps)
        with writer.saving(fig, str(path), dpi=dpi):
            for frame in range(frames):
                update(frame)
                writer.grab_frame()
    finally:
        plt.close(fig)


def export_png(fig, path):
    if path.exists():
        raise ValueError(f"Output already exists: {path}")
    try:
        fig.savefig(path, dpi=110)
    finally:
        plt.close(fig)


def groups_2d(ax, points, kinds, xlabel="Latent coordinate 1", ylabel="Latent coordinate 2"):
    for kind in np.unique(kinds):
        mask = kinds == kind
        ax.scatter(*points[mask, :2].T, s=13, alpha=0.65, c=GROUPS[kind], label=kind.replace("_", " "))
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(loc="best", framealpha=0.85)


def demo_network(lab: LabExperiments, output: Path):
    base = lab.base()
    data, model = lab.data, base["model"]
    choices = [int(np.flatnonzero(data.kinds == name)[0]) for name in ("normal", "off_subspace")]
    fig, axes = figure("01 | Encoder → latent space → decoder",
        "Synthetic inputs • Highlighted evaluation order is conceptual, not physical motion or semantic neuron identity.", 2, 1, size=(11, 9))
    ax, bars = axes
    cmap = plt.get_cmap("coolwarm")
    nodes, positions = [], []
    for layer, count in enumerate((12, 8, 2, 8, 12)):
        positions.append(np.column_stack((np.full(count, layer), np.linspace(0.04, 0.96, count))))
    for left, right in zip(positions[:-1], positions[1:]):
        for a in left:
            for b in right:
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#aebccc", lw=0.35, alpha=0.35, zorder=1)
    for p in positions:
        nodes.append(ax.scatter(*p.T, c=np.zeros(len(p)), s=110, edgecolors="white", zorder=3, cmap=cmap, vmin=-3, vmax=3))
    texts = [[ax.text(*p, "", ha="center", va="center", fontsize=6, zorder=4) for p in ps] for ps in positions]
    ax.set(xlim=(-0.4, 4.4), ylim=(-0.17, 1.12))
    ax.axis("off")
    for i, label in enumerate(("Input: 12", "Encoder: 8", "Latent: 2", "Decoder: 8", "Reconstruction: 12")):
        ax.text(i, 1.08, label, ha="center", fontsize=8)
    stage_text = ax.text(2, -0.12, "", ha="center", fontsize=10)
    x = np.arange(12)
    original = bars.bar(x - 0.2, np.zeros(12), width=0.4, color=NORMAL, label="Original input")
    rebuilt = bars.bar(x + 0.2, np.zeros(12), width=0.4, color=RECON, label="Reconstruction")
    bars.set(xlabel="Feature index", ylabel="Standardized feature value", xticks=x,
             ylim=limits(np.r_[data.test[choices].ravel(), base["reconstruction"][choices].ravel()]))
    bars.legend(loc="upper right")

    def update(frame):
        sample = choices[min(len(choices)-1, frame * len(choices) // lab.frames)]
        x0 = data.test[sample]
        hidden1 = np.tanh(x0 @ model.encoder[0].weight.detach().numpy().T + model.encoder[0].bias.detach().numpy())
        z = base["latent"][sample]
        hidden2 = np.tanh(z @ model.decoder[0].weight.detach().numpy().T + model.decoder[0].bias.detach().numpy())
        values = [x0, hidden1, z, hidden2, base["reconstruction"][sample]]
        within_sample = frame % (lab.frames // 2)
        stage = min(4, within_sample * 5 // (lab.frames // 2))
        for layer, (node, vector) in enumerate(zip(nodes, values)):
            node.set_array(vector)
            node.set_linewidth(2.2 if layer == stage else 0.6)
            node.set_edgecolor(ANOMALY if layer == stage else "white")
            for text, value in zip(texts[layer], vector):
                text.set_text(f"{value:.1f}")
        for a, b, v, r in zip(original, rebuilt, x0, values[-1]):
            a.set_height(v)
            b.set_height(r)
        stage_text.set_text(f"z = encoder(x) = [{z[0]:.2f}, {z[1]:.2f}]   |   x̂ = decoder(z)")
        bars.set_title(f"Sample {sample}: {data.kinds[sample]} | Reconstruction MSE = {base['scores'][sample]:.4f}")

    export_animation(fig, update, lab.frames, output / "01_encoder_decoder.gif")


def latent_view(lab: LabExperiments, interactive=False):
    result = lab.geometry()
    raw, scaler = result["raw"], result["scaler"]
    rec = scaler.inverse_transform(result["reconstruction"])
    fig = plt.figure(figsize=(12, 6.5))
    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.30 if interactive else 0.24, top=0.84, wspace=0.23)
    fig.suptitle("02 | Curved 2D structure in 3D → learned coordinates → reconstruction", fontsize=13, y=0.96)
    original = fig.add_subplot(131, projection="3d")
    latent = fig.add_subplot(132)
    reconstructed = fig.add_subplot(133, projection="3d")
    groups_2d(latent, result["latent"], raw.kinds)
    latent.set_title("Actual encoder outputs (not generator variables)")
    for kind in np.unique(raw.kinds):
        mask = raw.kinds == kind
        original.scatter(*raw.test[mask].T, c=GROUPS[kind], s=10, alpha=0.6, label=kind.replace("_", " "))
        reconstructed.scatter(*rec[mask].T, c=GROUPS[kind], marker="x", s=12, alpha=0.7, label=kind.replace("_", " "))
    for i in range(0, len(rec), 9):
        reconstructed.plot(*np.vstack((raw.test[i], rec[i])).T, color="#7c8795", lw=0.8, alpha=0.65)
    # A faint generating surface gives geometric context, never a claimed learned projection.
    grid = np.linspace(-1.8, 1.8, 15)
    u, v = np.meshgrid(grid, grid)
    original.plot_wireframe(u, v, 0.8 * (u**2 + 0.6*v**2), color="#a7b2c0", alpha=0.25, lw=0.5)
    extent = [limits(np.r_[raw.test[:, j], rec[:, j]]) for j in range(3)]
    for ax in (original, reconstructed):
        ax.set(xlabel="x₁ (synthetic units)", ylabel="x₂ (synthetic units)", zlabel="x₃ (synthetic units)",
               xlim=extent[0], ylim=extent[1], zlim=extent[2])
        ax.legend(loc="upper left", fontsize=7)
    original.set_title("Original observations + generating surface")
    reconstructed.set_title("Reconstructed × and sampled residual segments")
    fig.text(.5, .055, "Synthetic geometry • Lines are residuals, not guaranteed nearest-point projections. Latent axes have no fixed semantics.", ha="center", fontsize=8)
    state = {"azimuth": -60.0, "sample": 0}
    controls = {}
    sample_text = fig.text(.5, .12, "", ha="center", fontsize=9)
    marker = latent.scatter([], [], s=100, facecolors="none", edgecolors="black", linewidths=1.5)

    def rotate(angle):
        state["azimuth"] = float(angle)
        for ax in (original, reconstructed):
            ax.view_init(elev=24, azim=angle)
        fig.canvas.draw_idle()

    def select(index):
        i = int(index)
        state["sample"] = i
        marker.set_offsets(result["latent"][[i], :2])
        sample_text.set_text(f"Test {i}: {raw.kinds[i]} | standardized MSE = {result['scores'][i]:.4f}")
        fig.canvas.draw_idle()

    if interactive:
        controls["camera"] = Slider(fig.add_axes([.18, .16, .62, .025]), "Azimuth", -180, 180, valinit=-60)
        controls["sample"] = Slider(fig.add_axes([.18, .12, .62, .025]), "Test sample", 0, len(raw.test)-1, valinit=0, valstep=1)
        controls["camera"].on_changed(rotate)
        controls["sample"].on_changed(select)
        sample_text.set_position((.5, .215))
    select(0)
    rotate(-60)
    return fig, {"rotate": rotate, "select": select, "state": state, "controls": controls}


def demo_latent(lab, output):
    fig, api = latent_view(lab)
    export_animation(fig, lambda i: api["rotate"](-65 + 300 * i / lab.frames), lab.frames, output / "02_manifold_3d.gif")
    result = lab.geometry()
    fig, (ax,) = figure("02 | Learned 2D latent representation",
        "Actual encoder outputs • OOD labels need not separate • Coordinates are not identifiable semantic concepts.", 1, 1, size=(8, 6))
    groups_2d(ax, result["latent"], result["raw"].kinds)
    export_png(fig, output / "02_latent_space.png")


def demo_reconstruction(lab, output):
    result, data = lab.base(), lab.data
    threshold = result["thresholds"]["0.99"]
    indices = [int(np.flatnonzero(data.kinds == k)[0]) for k in ("normal", "off_subspace", "on_subspace", "unusual_normal")]
    missed = np.flatnonzero((data.labels == 1) & (result["scores"] <= threshold))
    if len(missed):
        indices.append(int(missed[np.argmin(result["scores"][missed])]))
    fig, axes = figure("03 | Reconstruction error is a sum of feature contributions",
        "Synthetic cases • Selection is descriptive after evaluation, never model tuning • A low score does not certify normality.")
    ax, errors = axes
    x = np.arange(12)
    lo, hi = limits(np.r_[data.test[indices].ravel(), result["reconstruction"][indices].ravel()])
    max_error = np.max((data.test[indices] - result["reconstruction"][indices])**2)

    def update(frame):
        i = indices[min(len(indices)-1, frame * len(indices) // lab.frames)]
        for panel in axes:
            panel.clear()
        ax.bar(x-.2, data.test[i], width=.4, color=NORMAL, label="Original input")
        ax.bar(x+.2, result["reconstruction"][i], width=.4, color=RECON, label="Reconstruction")
        ax.set(xlabel="Feature index", ylabel="Standardized value", ylim=(lo, hi), xticks=x,
               title=f"Test {i}: {data.kinds[i]} | label={data.labels[i]}")
        ax.legend()
        squared = (data.test[i] - result["reconstruction"][i])**2
        colors = [ANOMALY if j == np.argmax(squared) else "#94a8c1" for j in x]
        errors.bar(x, squared, color=colors)
        errors.set(xlabel="Feature index", ylabel="Squared standardized residual", xticks=x,
                   ylim=(0, max_error * 1.13), title=f"MSE(x) = mean((x − x̂)²) = {squared.mean():.4f}")
        errors.text(.03, .95, f"99% calibration τ = {threshold:.4f}\nDecision: {'alert' if squared.mean() > threshold else 'no alert'}\nOrange: largest contribution", transform=errors.transAxes, va="top")

    export_animation(fig, update, lab.frames, output / "03_reconstruction_error.gif")


def demo_training(lab, output):
    base, data = lab.base(), lab.data
    checkpoints, history = base["checkpoints"], base["history"]
    normal = np.flatnonzero(data.kinds == "normal")[:70]
    sample = int(normal[0])
    fig, axes = figure("04 | Actual checkpoints: learning a reconstruction",
        "Adam/backprop update both networks • Fixed final epoch, no test-based selection • Latent bases may rotate during training.", 1, 3, size=(12, 6))
    loss_ax, rec_ax, latent_ax = axes
    loss_ax.plot(history[:, 0], color=NORMAL, alpha=.2)
    loss_ax.plot(history[:, 1], color=ANOMALY, alpha=.2)
    train_line, = loss_ax.plot([], [], c=NORMAL, label="Train normal MSE")
    val_line, = loss_ax.plot([], [], c=ANOMALY, label="Validation normal MSE")
    loss_ax.set(xlabel="Epoch", ylabel="Standardized MSE", yscale="log", xlim=(0, len(history)-1),
                ylim=(max(history.min()*.8, 1e-7), history.max()*1.15))
    loss_ax.legend()
    rec_ax.plot(data.test[sample], c=NORMAL, label="Fixed original")
    rec_line, = rec_ax.plot([], [], c=RECON, marker="o", ms=3, label="Checkpoint reconstruction")
    rec_ax.set(xlabel="Feature index", ylabel="Standardized value", xlim=(0, 11),
               ylim=limits(np.r_[data.test[sample], np.concatenate([c["reconstruction"][sample] for c in checkpoints])]))
    rec_ax.legend()
    all_z = np.concatenate([c["latent"][normal] for c in checkpoints])
    points = latent_ax.scatter([], [], s=15, c=LATENT)
    latent_ax.set(xlabel="Learned coordinate 1", ylabel="Learned coordinate 2",
                  xlim=limits(all_z[:, 0]), ylim=limits(all_z[:, 1]))

    def update(frame):
        cp = checkpoints[min(len(checkpoints)-1, frame * len(checkpoints) // lab.frames)]
        epoch = cp["epoch"]
        xs = np.arange(epoch+1)
        train_line.set_data(xs, history[:epoch+1, 0])
        val_line.set_data(xs, history[:epoch+1, 1])
        rec_line.set_data(np.arange(12), cp["reconstruction"][sample])
        points.set_offsets(cp["latent"][normal])
        loss_ax.set_title(f"Epoch {epoch} | train {history[epoch, 0]:.4f}\nvalidation {history[epoch, 1]:.4f}")
        rec_ax.set_title(f"Fixed test sample {sample} (never optimized)")
        latent_ax.set_title(f"Same {len(normal)} normal samples, checkpoint {epoch}")

    export_animation(fig, update, lab.frames, output / "04_training_dynamics.gif")


def threshold_view(lab, interactive=True):
    result, data = lab.base(), lab.data
    scores, labels = result["scores"], data.labels
    thresholds = result["thresholds"]
    fig, axes = figure("05 | A score becomes a decision only after choosing a threshold",
        "Exploratory test slider is NOT threshold selection • Calibration ≠ guaranteed future FPR • Scores are not probabilities.", 2, 2, size=(11, 8))
    fig.subplots_adjust(bottom=.30, top=.86, hspace=.75, wspace=.35)
    distribution, confusion, roc_ax, pr_ax = axes
    max_score = max(scores.max(), max(thresholds.values())) * 1.05 + 1e-8
    bins = np.r_[0., np.geomspace(max(scores.min() * .8, 1e-6), max_score, 36)]
    for label, color, text in ((0, NORMAL, "Normal"), (1, ANOMALY, "Anomaly")):
        distribution.hist(scores[labels == label], bins=bins, color=color, alpha=.55, label=text)
    distribution.set(xlabel="Reconstruction MSE (symlog x, standardized²)", ylabel="Test count", xlim=(0, max_score),
                     xscale="symlog")
    distribution.set_xscale("symlog", linthresh=.1)
    distribution.legend()
    threshold_line = distribution.axvline(thresholds["0.99"], color="black", lw=1.5, label="Threshold")
    regions = []
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    roc_ax.plot(fpr, tpr, c=RECON)
    roc_ax.plot([0, 1], [0, 1], "--", c="gray", lw=.7)
    pr_ax.plot(recall, precision, c=RECON)
    pr_ax.axhline(labels.mean(), c="gray", ls="--", label=f"Prevalence {labels.mean():.1%}")
    pr_ax.legend(loc="lower left")
    roc_ax.set(xlabel="False-positive rate", ylabel="True-positive rate / recall", xlim=(-.03, 1.03), ylim=(-.03, 1.03),
               title=f"ROC-AUC = {result['metrics']['0.99']['roc_auc']:.3f}")
    pr_ax.set(xlabel="Recall", ylabel="Precision", xlim=(-.03, 1.03), ylim=(-.03, 1.03),
              title=f"Average precision = {result['metrics']['0.99']['average_precision']:.3f}")
    roc_point, = roc_ax.plot([], [], "o", color=ANOMALY, ms=8)
    pr_point, = pr_ax.plot([], [], "o", color=ANOMALY, ms=8)
    image = confusion.imshow(np.zeros((2, 2)), cmap="Blues", vmin=0, vmax=len(labels))
    confusion.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"],
                  xlabel="Predicted label", ylabel="True label")
    cell_text = [[confusion.text(j, i, "", ha="center", va="center", fontsize=14) for j in range(2)] for i in range(2)]
    metrics_text = fig.text(.5, .18, "", ha="center", fontsize=10)
    state, controls = {}, {}

    def update(threshold):
        threshold = float(threshold)
        metrics = metrics_at(labels, scores, threshold)
        state.update(threshold=threshold, metrics=metrics)
        threshold_line.set_xdata([threshold, threshold])
        for patch in regions:
            patch.remove()
        regions[:] = [distribution.axvspan(0, threshold, color=NORMAL, alpha=.06),
                      distribution.axvspan(threshold, max_score, color=ANOMALY, alpha=.06)]
        distribution.set_title(f"No alert ≤ τ | Alert > τ = {threshold:.4f}")
        cm = np.asarray(metrics["confusion_matrix"])
        image.set_data(cm)
        for i in range(2):
            for j in range(2):
                cell_text[i][j].set_text(str(cm[i, j]))
        confusion.set_title(f"Confusion matrix | FP = {metrics['fp']}")
        roc_point.set_data([metrics["false_positive_rate"]], [metrics["recall"]])
        pr_point.set_data([metrics["recall"]], [metrics["precision"]])
        metrics_text.set_text(f"τ={threshold:.4f}    Precision={metrics['precision']:.3f}    Recall={metrics['recall']:.3f}    F1={metrics['f1']:.3f}    FP={metrics['fp']}")
        fig.canvas.draw_idle()
        return metrics

    if interactive:
        slider = Slider(fig.add_axes([.16, .12, .68, .025]), "Threshold", 0, max_score, valinit=thresholds["0.99"], valfmt="%.4f")
        slider.on_changed(update)
        controls["threshold"] = slider
        for i, q in enumerate(("0.95", "0.975", "0.99")):
            button = Button(fig.add_axes([.22 + .20*i, .065, .16, .03]), f"Calibration {float(q):.1%}")
            button.on_clicked(lambda event, quantile=q: slider.set_val(thresholds[quantile]))
            controls[q] = button
        fig.text(.5, .225, "Buttons use independent NORMAL calibration scores; dragging explores already measured test outcomes.", ha="center", fontsize=8)
    update(thresholds["0.99"])
    return fig, {"update": update, "state": state, "controls": controls, "max_score": max_score}


def demo_threshold(lab, output):
    fig, _ = threshold_view(lab, interactive=True)
    export_png(fig, output / "05_threshold_interactive.png")
    fig, api = threshold_view(lab, interactive=False)
    # Predeclared calibration quantiles, repeated across frames for readability.
    thresholds = list(lab.base()["thresholds"].values())
    export_animation(fig, lambda i: api["update"](thresholds[min(2, i * 3 // lab.frames)]),
                     lab.frames, output / "05_threshold_sweep.gif", fps=2)


def demo_comparison(lab, output):
    results = lab.comparison()
    fig = plt.figure(figsize=(11, 6.5))
    fig.subplots_adjust(left=.06, right=.96, bottom=.19, top=.84, wspace=.3)
    fig.suptitle("06 | PCA, linear AE and nonlinear AE on the same held-out rows", fontsize=14, y=.96)
    space = fig.add_subplot(131, projection="3d")
    latent_ax, mse_ax = fig.add_subplot(132), fig.add_subplot(133)
    foot = fig.text(.5, .05, "", ha="center", fontsize=8)
    title = fig.text(.5, .89, "", ha="center", fontsize=11)
    all_raw = np.concatenate([r["raw"].test for r in results])
    all_rec = np.concatenate([r["scaler"].inverse_transform(r["reconstruction"]) for r in results])
    xyz_limits = [limits(np.r_[all_raw[:, j], all_rec[:, j]]) for j in range(3)]
    all_z = np.concatenate([r["latent"] for r in results])
    ymax = max(r["normal_mse"] for r in results) * 1.3

    def update(frame):
        current = results[min(5, frame * 6 // lab.frames)]
        for ax in (space, latent_ax, mse_ax):
            ax.clear()
        normal = current["data"].labels == 0
        raw = current["raw"].test[normal][:80]
        rec = current["scaler"].inverse_transform(current["reconstruction"])[normal][:80]
        space.scatter(*raw.T, c=NORMAL, s=8, alpha=.5, label="Original normal")
        space.scatter(*rec.T, c=RECON, marker="x", s=12, label="Reconstruction")
        for i in range(0, len(raw), 8):
            space.plot(*np.vstack((raw[i], rec[i])).T, c="gray", alpha=.6, lw=.7)
        space.set(xlabel="x₁", ylabel="x₂", zlabel="x₃", xlim=xyz_limits[0], ylim=xyz_limits[1], zlim=xyz_limits[2])
        space.view_init(elev=24, azim=-60)
        space.legend(fontsize=7)
        groups_2d(latent_ax, current["latent"][normal], current["data"].kinds[normal])
        latent_ax.set(xlim=limits(all_z[:, 0]), ylim=limits(all_z[:, 1]), title=current["name"] + " coordinates")
        subset = [r for r in results if r["dataset"] == current["dataset"]]
        mse_ax.bar(np.arange(3), [r["normal_mse"] for r in subset], color=[RECON, LATENT, NORMAL])
        mse_ax.set(xticks=np.arange(3), xticklabels=["PCA", "Linear AE", "Nonlinear\nAE"], ylabel="Normal test MSE (standardized²)",
                   ylim=(0, ymax), title="Measured held-out reconstruction")
        for j, record in enumerate(subset):
            mse_ax.text(j, record["normal_mse"] + ymax*.025, f"{record['normal_mse']:.4f}", ha="center", fontsize=8)
        title.set_text(f"Dataset: {current['dataset']} | displayed reconstruction: {current['name']}")
        foot.set_text("Synthetic • Same scaler and rows per dataset • PCA gives the linear TRAINING optimum; finite-step AE need not reach it.\nLatent bases differ; lower test MSE here establishes no universal advantage.")

    export_animation(fig, update, lab.frames, output / "06_pca_vs_autoencoders.gif", fps=2)


def demo_bottleneck(lab, output):
    results = lab.bottlenecks()
    fig, axes = figure("07 | More latent coordinates change both capacity and detection",
        "Same rows/epochs/optimizer, different parameter counts • Points are initialization seeds, not confidence intervals.", 2, 2, size=(11, 8))
    loss_ax, ap_ax, dist_ax, rec_ax = axes
    seeds = sorted({r["config"]["seed"] for r in results})
    for seed in seeds:
        rows = [r for r in results if r["config"]["seed"] == seed]
        loss_ax.plot([r["config"]["latent"] for r in rows], [r["normal_test_mse"] for r in rows], "o-", label=f"Init seed {seed}")
        ap_ax.plot([r["config"]["latent"] for r in rows], [r["metrics"]["0.99"]["average_precision"] for r in rows], "o-", label=f"Init seed {seed}")
    loss_ax.set(xlabel="Latent dimension k", ylabel="Normal test MSE", xticks=[1, 2, 4, 8])
    ap_ax.set(xlabel="Latent dimension k", ylabel="Average precision (all test rows)", xticks=[1, 2, 4, 8], ylim=(0, 1))
    for ax in (loss_ax, ap_ax):
        ax.legend()
    active_loss = loss_ax.axvline(1, c="black", alpha=.4)
    active_ap = ap_ax.axvline(1, c="black", alpha=.4)
    displayed = [r for r in results if r["config"]["seed"] == seeds[0]]
    bins = np.linspace(0, max(r["scores"].max() for r in displayed)*1.02, 28)
    hist_ymax = max(np.histogram(r["scores"][lab.data.labels == label], bins)[0].max() for r in displayed for label in (0, 1))*1.2
    sample = int(np.flatnonzero(lab.data.kinds == "on_subspace")[0])
    value_limits = limits(np.r_[lab.data.test[sample], np.concatenate([r["reconstruction"][sample] for r in displayed])])

    def update(frame):
        record = displayed[min(3, frame * 4 // lab.frames)]
        k = record["config"]["latent"]
        active_loss.set_xdata([k, k])
        active_ap.set_xdata([k, k])
        dist_ax.clear()
        rec_ax.clear()
        for label, color, text in ((0, NORMAL, "Normal"), (1, ANOMALY, "Anomaly")):
            dist_ax.hist(record["scores"][lab.data.labels == label], bins, alpha=.6, color=color, label=text)
        dist_ax.set(xlabel="Reconstruction MSE", ylabel="Test count", xlim=(bins[0], bins[-1]), ylim=(0, hist_ymax),
                    title=f"12→8→{k}→8→12 | {record['parameters']} parameters")
        dist_ax.legend()
        rec_ax.plot(lab.data.test[sample], color=NORMAL, label="Original on-subspace anomaly")
        rec_ax.plot(record["reconstruction"][sample], color=RECON, label=f"Reconstructed, k={k}")
        rec_ax.set(xlabel="Feature index", ylabel="Standardized value", ylim=value_limits, title=f"Same test sample {sample}, init seed {seeds[0]}")
        rec_ax.legend(fontsize=7)

    export_animation(fig, update, lab.frames, output / "07_bottleneck_comparison.gif", fps=2)


def demo_failures(lab, output):
    results = lab.failures()
    fig, axes = figure("08 | Capacity × contamination: residuals versus intended labels",
        "Synthetic factorial control • Normal-like anomalies have no observable distinguishing signal • Proposed failures need not occur.", 1, 2, size=(11, 6.5))
    fig.subplots_adjust(left=.18, right=.97, wspace=.55, bottom=.23)
    rates, scores_ax = axes
    kinds = list(np.unique(lab.data.kinds))
    rates.imshow([[r["alert_rate_by_kind"][k] for k in kinds] for r in results], vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    rates.set(yticks=range(4), yticklabels=[r["name"].replace(" / ", "\n") for r in results],
              xticks=range(len(kinds)), xticklabels=[k.replace("_", "\n") for k in kinds],
              title="Alert fraction at 99% calibration τ", xlabel="Externally specified test group")
    for i, r in enumerate(results):
        for j, kind in enumerate(kinds):
            rates.text(j, i, f"{r['alert_rate_by_kind'][kind]:.2f}", ha="center", va="center", fontsize=8)
    for label in rates.get_xticklabels():
        label.set_fontsize(7)
    ymax = max(r["scores"].max() for r in results)*1.1
    rng = np.random.default_rng(3)
    jitter = rng.uniform(-.15, .15, len(lab.data.test))

    def update(frame):
        record = results[min(3, frame * 4 // lab.frames)]
        scores_ax.clear()
        for j, kind in enumerate(kinds):
            mask = lab.data.kinds == kind
            scores_ax.scatter(j + jitter[mask], record["scores"][mask], c=GROUPS[kind], s=12, alpha=.6)
        scores_ax.axhline(record["thresholds"]["0.99"], c="black", ls="--", label="Independent calibration τ")
        scores_ax.set(xticks=range(len(kinds)), xticklabels=[k.replace("_", "\n") for k in kinds],
                      xlabel="Externally specified test group", ylabel="Reconstruction MSE", ylim=(0, ymax),
                      title=f"{record['name']}\nAP={record['metrics']['0.99']['average_precision']:.3f}, {record['parameters']} parameters")
        scores_ax.tick_params(axis="x", labelsize=7)
        scores_ax.legend(fontsize=7)

    export_animation(fig, update, lab.frames, output / "08_failure_modes.gif", fps=2)


def demo_denoising(lab, output):
    results = lab.denoising()
    fig, axes = figure("09 | Denoising learns corrupted input → clean target",
        "Actual neural outputs on the SAME corrupted test inputs • Improvement is measured, never imposed by smoothing.", 1, 2)
    left, right = axes
    value_limits = limits(np.concatenate([results[0][key].ravel() for key in ("clean", "noisy", "reconstruction")]
                                         + [results[1]["reconstruction"].ravel()]))

    def update(frame):
        sample = frame % 8
        for panel, record, name in zip(axes, results, ("Standard AE", "Denoising AE")):
            panel.clear()
            panel.plot(record["clean"][sample], c=NORMAL, label="Clean target")
            panel.plot(record["noisy"][sample], c="gray", ls="--", label="Corrupted input")
            panel.plot(record["reconstruction"][sample], c=RECON, marker="o", ms=3, label="Actual reconstruction")
            panel.set(xlabel="Ordered synthetic feature index", ylabel="Standardized value", ylim=value_limits,
                      title=f"{name} | test clean-target MSE={record['clean_target_mse']:.4f}\nSample {sample}")
            panel.legend()

    export_animation(fig, update, lab.frames, output / "09_denoising.gif")


def demo_scaling(lab, output):
    results = lab.scaling()
    fig, axes = figure("10 | Feature scale changes the reconstruction objective",
        "Same raw observations and fixed training budget • Standardization does not encode business importance • Mixed units need care.", 1, 2)
    objective_ax, physical_ax = axes
    objective_max = max(r["objective_errors"].max() for r in results) * 1.2
    physical_max = max(r["physical_errors"].max() for r in results) * 1.2

    def update(frame):
        record = results[min(1, frame * 2 // lab.frames)]
        for ax in axes:
            ax.clear()
        x = np.arange(12)
        objective_ax.bar(x, record["objective_errors"], color=RECON)
        objective_ax.set(xlabel="Feature index (scale multipliers rise from 1 to 100)", ylabel="Per-feature training-objective squared residual",
                         yscale="symlog", ylim=(0, objective_max), title=f"{record['name']}: objective units {'standardized²' if record['name'] == 'Standardized' else 'raw²'}")
        physical_ax.bar(x, record["physical_errors"], color=NORMAL)
        physical_ax.set(xlabel="Feature index", ylabel="Per-feature raw squared residual (mixed units)", yscale="symlog",
                        ylim=(0, physical_max), title="Both models evaluated in the SAME raw units")
        objective_ax.text(.03, .93, "Mean across normal test rows\nSymmetric-log scale preserves zero", transform=objective_ax.transAxes, va="top")

    export_animation(fig, update, lab.frames, output / "10_feature_scaling.gif", fps=2)


DEMOS = {
    "network": (demo_network, ["01_encoder_decoder.gif"]),
    "latent": (demo_latent, ["02_manifold_3d.gif", "02_latent_space.png"]),
    "reconstruction": (demo_reconstruction, ["03_reconstruction_error.gif"]),
    "training": (demo_training, ["04_training_dynamics.gif"]),
    "threshold": (demo_threshold, ["05_threshold_interactive.png", "05_threshold_sweep.gif"]),
    "comparison": (demo_comparison, ["06_pca_vs_autoencoders.gif"]),
    "bottleneck": (demo_bottleneck, ["07_bottleneck_comparison.gif"]),
    "failures": (demo_failures, ["08_failure_modes.gif"]),
    "denoising": (demo_denoising, ["09_denoising.gif"]),
    "scaling": (demo_scaling, ["10_feature_scaling.gif"]),
}
