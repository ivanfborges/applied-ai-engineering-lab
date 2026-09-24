"""Generate the Day 21 Logistic Regression Visual Lab offline.

Run without arguments for all 14 numbered views. Outputs are relative to this
script, independent of the working directory. See VISUAL_GUIDE.md.
"""

import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import plotly
import plotly.graph_objects as go
import sklearn
from sklearn.metrics import log_loss

from from_scratch import sigmoid, logit
from visual_experiments import (
    SEED, boundary_segment, calibration_experiment, classification_case,
    decision_counts, feature_grid, finite_json, gradient_trace, loss_landscape,
    nonlinear_experiment, raw_parameters, regularization_experiment,
)

BLUE, ORANGE, INK, MUTED, PURPLE = "#2476a3", "#d47731", "#172d41", "#5c6f7d", "#963b78"
BACKGROUND = "#fbfaf6"
FIELD = LinearSegmentedColormap.from_list("probability", [BLUE, "#faf7ef", ORANGE])
COLORS = np.array([BLUE, ORANGE])
OUTPUTS = (
    "01_sigmoid.png", "02_probability_odds_logit.png", "03_cross_entropy.png",
    "04_decision_boundary.png", "05_threshold_animation.gif",
    "06_probability_surface_3d.html", "07_gradient_descent.gif",
    "08_regularization.png", "09_nonlinear_limitation.png",
    "10_polynomial_features.png", "11_calibration.png", "12_full_pipeline.png",
    "13_loss_surface_3d.html", "14_sigmoid_parameters.gif",
)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 13, "axes.labelsize": 11, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#b2bcc1", "axes.spines.top": False,
        "axes.spines.right": False, "axes.facecolor": BACKGROUND,
        "figure.facecolor": BACKGROUND, "savefig.facecolor": BACKGROUND,
        "grid.alpha": 0.18, "legend.frameon": False,
    })


def heading(fig, number, title, subtitle, footer):
    fig.text(0.065, 0.955, f"DAY 21  /  {number:02d}", color=MUTED, fontsize=10, weight="bold")
    fig.text(0.065, 0.908, title, fontsize=21, weight="bold")
    fig.text(0.065, 0.864, subtitle, fontsize=10.5, color=MUTED)
    fig.text(0.065, 0.035, footer, fontsize=10, color=MUTED)


def save_png(fig, out, name):
    fig.savefig(out / name, dpi=150)
    plt.close(fig)


def save_gif(fig, update, frames, out, name):
    animation = FuncAnimation(fig, update, frames=frames, interval=125, blit=False)
    animation.save(out / name, writer=PillowWriter(fps=8), dpi=100)
    plt.close(fig)


def grid_for(case, resolution=150):
    return feature_grid(np.vstack((case.X_train, case.X_test)), resolution)


def bounds_for(xx, yy):
    return (xx.min(), xx.max(), yy.min(), yy.max())


def field_plot(ax, model, xx, yy, grid, *, contours=True):
    probabilities = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    image = ax.imshow(probabilities, origin="lower", extent=bounds_for(xx, yy),
                      cmap=FIELD, vmin=0, vmax=1, aspect="auto", interpolation="bilinear")
    if contours:
        levels = [t for t in (0.25, 0.5, 0.75) if probabilities.min() < t < probabilities.max()]
        lines = ax.contour(xx, yy, probabilities, levels=levels,
                           colors=INK, linewidths=1.3)
        locations = []
        for index, segments in enumerate(lines.allsegs):
            segment = max(segments, key=len)
            target = np.array([xx.min() + (0.3 + 0.2*index)*np.ptp(xx), yy.mean()])
            normalized = (segment - target) / [np.ptp(xx), np.ptp(yy)]
            locations.append(segment[np.argmin(np.sum(normalized**2, axis=1))])
        ax.clabel(lines, fmt=lambda v: f"p={v:.2f}", fontsize=9, manual=locations)
    ax.set(xlabel="Feature 1", ylabel="Feature 2")
    return image


def sample_points(ax, case, training=True):
    if training:
        ax.scatter(*case.X_train.T, c=COLORS[case.y_train], s=16, alpha=0.25,
                   edgecolors="none", label="Training (faint)")
    for label, marker in ((0, "o"), (1, "^")):
        selected = case.y_test == label
        ax.scatter(*case.X_test[selected].T, color=COLORS[label], marker=marker,
                   s=43, edgecolors="white", linewidths=0.7, label=f"Test: true y={label}")


def create_sigmoid_visualization(out):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(left=0.1, right=0.94, bottom=0.2, top=0.79)
    heading(fig, 1, "A score becomes a probability",
            "The sigmoid maps an unrestricted linear score to P(y=1 | x).",
            "Near p=0.5 the binary model is least decisive; this is not a measure of epistemic uncertainty.")
    z = np.linspace(-10, 10, 600)
    ax.axvspan(-10, 0, color=BLUE, alpha=0.07)
    ax.axvspan(0, 10, color=ORANGE, alpha=0.07)
    ax.plot(z, sigmoid(z), color=INK, linewidth=3)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.axvline(0, color=MUTED, ls="--", lw=1)
    for value, offset in [(-5, (8, 28)), (-2, (-65, 30)), (0, (14, -38)), (2, (15, -36)), (5, (5, -38))]:
        p = float(sigmoid(value))
        ax.scatter(value, p, color=PURPLE, s=48, zorder=5)
        ax.annotate(f"z={value}\np={p:.3f}", (value, p), xytext=offset,
                    textcoords="offset points", fontsize=10, color=INK)
    ax.text(-8, 0.74, "Negative evidence\nprobability near 0", color=BLUE, fontsize=13)
    ax.text(5, 0.28, "Positive evidence\nprobability near 1", color=ORANGE, fontsize=13, ha="center")
    ax.set(xlim=(-10, 10), ylim=(-0.04, 1.07), xlabel=r"Linear score  $z=w^\top x+b$",
           ylabel=r"$P(y=1 \mid x)=\sigma(z)$")
    save_png(fig, out, OUTPUTS[0])


def create_logit_visualization(out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.8), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    fig.subplots_adjust(left=0.065, right=0.97, bottom=0.2, top=0.77, wspace=0.36)
    heading(fig, 2, "Probability → odds → logit",
            "The linear model lives in log-odds space. Sigmoid is the inverse transformation.",
            "As p approaches 0 or 1, the logit tends to −∞ or +∞. Endpoints are excluded.")
    p = np.linspace(0.01, 0.99, 500)
    samples = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    axes[0].plot(p, p / (1 - p), color=BLUE, lw=2.5)
    axes[0].set(yscale="log", xlabel="Probability p", ylabel="Odds p / (1−p)", title="Odds: ratio of probabilities")
    axes[1].plot(p, logit(p), color=PURPLE, lw=2.5)
    axes[1].set(xlabel="Probability p", ylabel="Logit (natural log)", title="Logit: unbounded score")
    axes[1].axhline(0, color=MUTED, lw=1, ls="--")
    axes[0].scatter(samples, samples / (1-samples), color=ORANGE, zorder=3)
    axes[1].scatter(samples, logit(samples), color=ORANGE, zorder=3)
    for ax in axes[:2]:
        ax.axvline(0.5, color=MUTED, ls=":", lw=1)
        ax.grid(True)
    axes[2].axis("off")
    rows = [[f"{v:.1f}", f"{v/(1-v):.3f}", f"{logit(v):+.3f}"] for v in samples]
    table = axes[2].table(cellText=rows, colLabels=["p", "odds", "logit"],
                          loc="center", cellLoc="center", bbox=[0, 0.2, 1, 0.65])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce1e2")
        cell.set_facecolor("#f1e0ce" if row == 3 else BACKGROUND)
    axes[2].text(0.5, 0.02, "p=0.5  ↔  odds=1  ↔  logit=0", ha="center", fontsize=11)
    save_png(fig, out, OUTPUTS[1])


def create_cross_entropy_visualization(out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.2, top=0.77, wspace=0.3)
    heading(fig, 3, "Confident mistakes cost more",
            "Binary cross-entropy scores probability estimates, before a threshold is applied.",
            "Loss uses natural logarithms (nats). Both classes use the same predicted positive-class probability p.")
    p = np.linspace(0.001, 0.999, 700)
    axes[0].plot(p, -np.log(p), color=ORANGE, lw=2.5, label="True y=1: −log(p)")
    axes[0].plot(p, -np.log1p(-p), color=BLUE, lw=2.5, label="True y=0: −log(1−p)")
    for value, xytext in [(0.01, (0.18, 5.1)), (0.5, (0.42, 2.1)), (0.9, (0.67, 1.35))]:
        loss = -np.log(value)
        axes[0].scatter(value, loss, color=INK, zorder=3)
        axes[0].annotate(f"y=1, p={value:g}\nloss={loss:.3f}", (value, loss), xytext=xytext,
                         arrowprops=dict(arrowstyle="-", color=MUTED), fontsize=10)
    axes[0].set(xlabel="Predicted P(y=1)", ylabel="Per-observation log loss", ylim=(0, 7.2))
    axes[0].legend(loc="upper center", fontsize=9)
    mistakes = np.array([0.49, 0.1, 0.001])
    bars = axes[1].bar(["p=0.49", "p=0.10", "p=0.001"], -np.log(mistakes),
                      color=[BLUE, ORANGE, PURPLE], width=0.55)
    axes[1].bar_label(bars, fmt="%.3f", padding=5)
    axes[1].set(title="True y=1; threshold=0.5\nAll three decisions are wrong",
                ylabel="Log loss", ylim=(0, 8))
    save_png(fig, out, OUTPUTS[2])


def create_probability_surface(out, case):
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=0.09, right=0.89, bottom=0.21, top=0.79)
    heading(fig, 4, "A continuous probability field",
            "One fitted model assigns probabilities throughout the feature plane.",
            "Synthetic make_classification data; scaling fits training rows only. Contours are model probabilities.")
    xx, yy, grid = grid_for(case)
    image = field_plot(ax, case.model, xx, yy, grid)
    sample_points(ax, case)
    ax.legend(loc="upper left", fontsize=9)
    fig.colorbar(image, ax=ax, label="P(y=1)", pad=0.03)
    save_png(fig, out, OUTPUTS[3])
    return {"test_log_loss": float(log_loss(
        case.y_test, case.model.predict_proba(case.X_test)[:, 1]))}


def create_threshold_animation(out, case):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_axes([0.075, 0.19, 0.53, 0.60])
    counts_ax = fig.add_axes([0.72, 0.34, 0.23, 0.34])
    fig.text(0.075, 0.815, "Background: model P(y=1), blue=0, orange=1", color=MUTED, fontsize=10)
    heading(fig, 5, "Same model. Different decisions.",
            "The probability field stays fixed while the decision threshold moves.",
            "Synthetic test set. Shape = true class (○ 0, △ 1); fill = predicted class. No threshold is selected here.")
    xx, yy, grid = grid_for(case)
    field_plot(ax, case.model, xx, yy, grid, contours=False)
    coef, intercept = raw_parameters(case.model)
    line, = ax.plot([], [], color=PURPLE, lw=3)
    probabilities = case.model.predict_proba(case.X_test)[:, 1]
    scatters = []
    for label, marker in [(0, "o"), (1, "^")]:
        mask = case.y_test == label
        scatters.append((mask, ax.scatter(*case.X_test[mask].T, s=40, marker=marker,
                         edgecolors=INK, linewidths=0.5, zorder=3)))
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", color=BLUE, label="Predict 0"),
        Line2D([], [], marker="s", ls="", color=ORANGE, label="Predict 1"),
        Line2D([], [], color=PURPLE, lw=3, label="p = threshold"),
    ], loc="upper left", fontsize=9)
    bars = counts_ax.barh(["TN", "FP", "FN", "TP"], [0] * 4, color=[BLUE, PURPLE, PURPLE, ORANGE])
    counts_ax.set(xlim=(0, len(case.y_test)), xlabel="Test observations")
    count_labels = [counts_ax.text(1, i, "", va="center", fontsize=12) for i in range(4)]
    status = fig.text(0.70, 0.755, "", fontsize=16, weight="bold")
    score = fig.text(0.70, 0.18, "", fontsize=11)
    thresholds = np.round(np.r_[np.linspace(0.05, 0.95, 21), np.linspace(0.95, 0.05, 21)[1:-1]], 3)

    def update(frame):
        threshold = thresholds[frame]
        segment = boundary_segment(coef, intercept, threshold, bounds_for(xx, yy))
        line.set_data(segment[:, 0], segment[:, 1])
        predicted = (probabilities >= threshold).astype(int)
        for mask, scatter in scatters:
            scatter.set_facecolors(COLORS[predicted[mask]])
        counts = decision_counts(case.y_test, probabilities, threshold)
        for bar, label, name in zip(bars, count_labels, ("TN", "FP", "FN", "TP")):
            value = counts[name]
            bar.set_width(value)
            label.set_text(str(value))
            label.set_x(min(value + 3, len(case.y_test) - 12))
        status.set_text(f"Threshold = {threshold:.3f}\nBoundary score = {logit(threshold):+.2f}")
        score.set_text(f"Precision  {counts['precision']:.3f}\nRecall       {counts['recall']:.3f}\nPositive decisions  {counts['positives']}")
    save_gif(fig, update, len(thresholds), out, OUTPUTS[4])
    return {str(t): decision_counts(case.y_test, probabilities, t) for t in (0.05, 0.5, 0.95)}


def write_interactive(fig, out, name, title):
    fig.update_layout(template="plotly_white", title=dict(text=title, x=0.04),
        font=dict(family="Arial", size=13, color=INK), height=760,
        margin=dict(l=20, r=20, t=100, b=45),
        scene=dict(camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))),
        legend=dict(x=0.01, y=0.96), paper_bgcolor=BACKGROUND)
    # One local JS bundle serves both HTML files; no CDN or server is needed.
    fig.write_html(out / name, include_plotlyjs="directory", full_html=True,
                   config={"responsive": True, "displaylogo": False, "scrollZoom": True},
                   div_id=name.split(".")[0])


def create_3d_probability_surface(out, case):
    xx, yy, grid = grid_for(case, 75)
    p = case.model.predict_proba(grid)[:, 1].reshape(xx.shape)
    fig = go.Figure(go.Surface(x=xx, y=yy, z=p, colorscale=[[0, BLUE], [0.5, "#faf7ef"], [1, ORANGE]],
        cmin=0, cmax=1, opacity=0.9, colorbar=dict(title="P(y=1)", len=0.6),
        hovertemplate="Feature 1=%{x:.2f}<br>Feature 2=%{y:.2f}<br>P(y=1)=%{z:.3f}<extra></extra>"))
    coef, intercept = raw_parameters(case.model)
    segment = boundary_segment(coef, intercept, 0.5, bounds_for(xx, yy))
    line_points = np.linspace(segment[0], segment[-1], 80)
    fig.add_trace(go.Scatter3d(x=line_points[:, 0], y=line_points[:, 1], z=np.full(80, 0.5),
        mode="lines", line=dict(color=PURPLE, width=8), name="p=0.5 on the sheet"))
    for label in (0, 1):
        mask = case.y_test == label
        fig.add_trace(go.Scatter3d(x=case.X_test[mask, 0], y=case.X_test[mask, 1],
            z=np.full(mask.sum(), -0.08), mode="markers",
            marker=dict(color=COLORS[label], size=4, symbol="circle" if label == 0 else "diamond"),
            name=f"Test y={label} (projected)",
            hovertemplate="Feature 1=%{x:.2f}<br>Feature 2=%{y:.2f}<extra>Projected test sample</extra>"))
    fig.update_layout(scene=dict(xaxis_title="Feature 1", yaxis_title="Feature 2",
        zaxis=dict(title="P(y=1)", range=[-0.13, 1.03])))
    write_interactive(fig, out, OUTPUTS[5],
        "06 · The sigmoid probability sheet<br><sup>Drag to rotate · scroll to zoom · hover to inspect. Samples are projected below the sheet; synthetic data.</sup>")


def create_gradient_descent_animation(out, case):
    scale = case.model.steps[0][1]
    X = scale.transform(case.X_train)
    xx, yy, grid = feature_grid(X)
    trace = gradient_trace(X, case.y_train, [-1.0, 0.8, 0.4])
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_axes([0.075, 0.20, 0.52, 0.58])
    loss_ax = fig.add_axes([0.70, 0.27, 0.25, 0.43])
    fig.text(0.075, 0.81, "Background: model P(y=1), blue=0, orange=1", color=MUTED, fontsize=10)
    heading(fig, 7, "Learning moves the probability field",
            "NumPy batch gradient descent: errors → gradient → weights and bias.",
            "Unpenalized training BCE on standardized synthetic training data. Unevenly spaced iterations; loop restarts training.")
    image = ax.imshow(np.full(xx.shape, 0.5), origin="lower", extent=bounds_for(xx, yy),
                      vmin=0, vmax=1, cmap=FIELD, aspect="auto", interpolation="bilinear")
    for label, marker in ((0, "o"), (1, "^")):
        mask = case.y_train == label
        ax.scatter(*X[mask].T, color=COLORS[label], s=22, marker=marker,
                   edgecolors="white", linewidths=0.5, label=f"True y={label}")
    line, = ax.plot([], [], color=PURPLE, lw=3)
    ax.set(xlabel="Standardized feature 1", ylabel="Standardized feature 2")
    ax.legend(loc="upper left", fontsize=9)
    loss_line, = loss_ax.plot([], [], color=PURPLE, lw=2.5)
    marker, = loss_ax.plot([], [], "o", color=ORANGE)
    losses = trace["losses"]
    loss_ax.set(xlim=(0, len(losses) - 1), ylim=(0, losses.max() * 1.08),
                xlabel="Gradient updates", ylabel="Training BCE (nats)")
    loss_ax.grid(True)
    status = fig.text(0.69, 0.75, "", fontsize=14, weight="bold")
    parameters = fig.text(0.69, 0.15, "", fontsize=11)

    def update(frame):
        step = int(trace["iterations"][frame])
        theta = trace["parameters"][frame]
        image.set_data(sigmoid(grid @ theta[:-1] + theta[-1]).reshape(xx.shape))
        segment = boundary_segment(theta[:-1], theta[-1], 0.5, bounds_for(xx, yy))
        line.set_data(segment[:, 0], segment[:, 1])
        loss_line.set_data(np.arange(step + 1), losses[:step + 1])
        marker.set_data([step], [losses[step]])
        status.set_text(f"Iteration {step:3d} / {len(losses)-1}\nBCE = {losses[step]:.4f}")
        parameters.set_text(f"w = [{theta[0]:+.2f}, {theta[1]:+.2f}]\nb = {theta[2]:+.2f}  |  learning rate = 0.3")
    save_gif(fig, update, len(trace["iterations"]), out, OUTPUTS[6])
    return dict(initial_loss=float(losses[0]), final_loss=float(losses[-1]),
                steps=len(losses)-1, saved_frames=len(trace["iterations"]),
                max_loss_increase=float(np.diff(losses).max()))


def create_regularization_visualization(out):
    strengths, paths = regularization_experiment()
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.23, top=0.77, wspace=0.22)
    heading(fig, 8, "Regularization constrains coefficients",
            "Smaller C means stronger regularization. Coefficients are in standardized feature units.",
            "Synthetic: 3 informative, 2 redundant, 3 noise features. Individual correlated coefficients need not shrink monotonically.")
    palette = plt.get_cmap("tab10")
    for ax, (name, path) in zip(axes, paths.items()):
        for j in range(path.shape[1]):
            ax.plot(strengths, path[:, j], "-o", lw=1.7, ms=4, color=palette(j), label=f"x{j+1}")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set(xscale="log", xlabel="C  (stronger ← regularization → weaker)", title=name)
        ax.grid(True)
        zeros = np.sum(path == 0, axis=1)
        ax.text(0.04, 0.96, f"Exact zeros at C=0.01: {zeros[0]}/8", transform=ax.transAxes, va="top", fontsize=10)
    axes[0].set_ylabel("Coefficient")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.10), ncol=8)
    save_png(fig, out, OUTPUTS[7])
    return {name: dict(C=strengths, coefficients=path, exact_zeros=(path == 0).sum(axis=1),
                       l2_norm=np.linalg.norm(path, axis=1)) for name, path in paths.items()}


def create_nonlinear_comparison(out):
    case, models, metrics = nonlinear_experiment()
    xx, yy, grid = grid_for(case)
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=0.09, right=0.89, bottom=0.21, top=0.77)
    heading(fig, 9, "A straight boundary meets curved structure",
            "The raw feature representation restricts the model's decision geometry.",
            "Synthetic moons, noise=0.23. One fixed train/test split; this illustrates representation capacity, not a universal ranking.")
    image = field_plot(ax, case.model, xx, yy, grid)
    sample_points(ax, case)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Raw features · test accuracy {metrics['Raw features']['accuracy']:.3f}")
    fig.colorbar(image, ax=ax, label="P(y=1)")
    save_png(fig, out, OUTPUTS[8])

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.92, bottom=0.21, top=0.75, wspace=0.15)
    heading(fig, 10, "Change the representation, change the boundary",
            "Both classifiers are linear in their supplied features. Degree 3 allows cubic geometry in the original plane.",
            "Same synthetic data, split, and C=1; degree fixed before evaluation. Transformations fit training data only.")
    for ax, (name, model) in zip(axes, models.items()):
        image = field_plot(ax, model, xx, yy, grid)
        sample_points(ax, case, training=False)
        m = metrics[name]
        ax.set_title(f"{name}\nTest accuracy {m['accuracy']:.3f} · log loss {m['log_loss']:.3f}", fontsize=11)
    axes[1].set_ylabel("")
    color_ax = fig.add_axes([0.94, 0.22, 0.015, 0.50])
    fig.colorbar(image, cax=color_ax, label="P(y=1)")
    save_png(fig, out, OUTPUTS[9])
    return metrics


def create_calibration_visualization(out):
    series = calibration_experiment()
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.23, top=0.76, wspace=0.32)
    heading(fig, 11, "Ranking and probability reliability differ",
            "Multiplying logits by 2.5 preserves ordering while changing confidence. Neither curve is a calibration guarantee.",
            "Synthetic logistic data; 1,200 held-out rows; 10 equal-width bins. Markers show nonempty bins; histogram shows support.")
    axes[0].plot([0, 1], [0, 1], "--", color=MUTED, label="Ideal reliability")
    for (name, result), color in zip(series.items(), (BLUE, ORANGE)):
        axes[0].plot(result["mean"], result["fraction"], "-o", color=color, lw=2, label=name)
        axes[1].hist(result["probabilities"], bins=np.linspace(0, 1, 11), histtype="step",
                     lw=2, color=color, label=name)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability in bin",
                ylabel="Observed fraction positive", title="Reliability diagram")
    axes[0].legend(fontsize=9)
    axes[0].grid(True)
    axes[1].set(xlabel="Predicted probability", ylabel="Held-out observations", title="Bin support")
    for i, ((name, result), color) in enumerate(zip(series.items(), (BLUE, ORANGE))):
        fig.text(0.08, 0.125 - i * 0.035,
            f"{name}:  ROC-AUC {result['auc']:.3f}  |  Brier {result['brier']:.3f}  |  Log loss {result['log_loss']:.3f}",
            color=color, fontsize=11)
    save_png(fig, out, OUTPUTS[10])
    return {name: {key: value for key, value in result.items() if key != "probabilities"}
            for name, result in series.items()}


def create_pipeline_visualization(out):
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_position([0, 0.08, 1, 0.72])
    ax.set(xlim=(0, 13), ylim=(0, 5))
    ax.axis("off")
    heading(fig, 12, "From features to decisions — and back through learning",
            "Separate the probability model from the threshold policy.",
            "Training adjusts weights and bias using labeled training data. Threshold selection uses validation data; final evaluation uses test data.")
    def box(x, y, width, text, color):
        patch = FancyBboxPatch((x, y), width, 0.95, boxstyle="round,pad=0.06,rounding_size=0.09",
                               facecolor=color, edgecolor="none")
        ax.add_patch(patch)
        ax.text(x + width / 2, y + 0.475, text, ha="center", va="center", fontsize=12, color=INK)
    def arrow(start, end, color=MUTED, **kwargs):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                    color=color, lw=1.8, **kwargs))
    positions = [0.3, 2.4, 4.7, 7.0, 9.5, 11.3]
    widths = [1.6, 1.8, 1.8, 2.0, 1.3, 1.4]
    labels = ["Features\nx", "Linear score\nz = wᵀx + b", "Sigmoid\n1 / (1 + exp(−z))",
              "Probability\nP(y=1 | x)", "Threshold\nτ", "Decision\np ≥ τ"]
    for x, width, label in zip(positions, widths, labels):
        box(x, 3.35, width, label, "#e2edf2" if x < 9 else "#f3e1d1")
    for j in range(5):
        arrow((positions[j] + widths[j] + 0.06, 3.82), (positions[j+1] - 0.08, 3.82))
    ax.text(4.7, 4.62, "PROBABILITY MODEL", fontsize=11, color=BLUE, ha="center", weight="bold")
    ax.text(10.8, 4.62, "DECISION POLICY", fontsize=11, color=ORANGE, ha="center", weight="bold")
    box(0.5, 1.2, 1.45, "True label\ny", "#f3e1d1")
    box(3.0, 1.2, 2.35, "Binary cross-entropy\n−y log p − (1−y) log(1−p)", "#ebe2eb")
    box(6.2, 1.2, 2.35, "Gradient\nXᵀ(p−y)/n, mean(p−y)", "#ebe2eb")
    box(9.5, 1.2, 2.55, "Update parameters\nw ← w − η dw\nb ← b − η db", "#ebe2eb")
    for start, end in [((1.95, 1.67), (2.92, 1.67)), ((5.35, 1.67), (6.12, 1.67)),
                       ((8.55, 1.67), (9.42, 1.67))]:
        arrow(start, end, PURPLE)
    ax.plot([8.0, 8.0, 4.2], [3.28, 2.55, 2.55], color=PURPLE, lw=1.8)
    arrow((4.2, 2.55), (4.2, 2.22), PURPLE)
    ax.text(6.0, 2.62, "training probabilities", fontsize=10, color=PURPLE)
    ax.plot([10.8, 10.8, 2.5, 2.5], [1.12, 0.42, 0.42, 3.1], color=PURPLE, lw=1.5)
    arrow((2.5, 3.1), (2.5, 3.28), PURPLE)
    ax.text(6.3, 0.18, "repeat on training data · unpenalized gradient shown", ha="center", color=PURPLE, fontsize=10)
    save_png(fig, out, OUTPUTS[11])


def create_loss_surface(out):
    weights, biases, loss, trace = loss_landscape()
    theta = trace["parameters"]
    values = trace["losses"][trace["iterations"]]
    fig = go.Figure(go.Surface(x=weights, y=biases, z=loss, colorscale="Tealgrn",
        opacity=0.88, colorbar=dict(title="BCE", len=0.6),
        hovertemplate="Weight=%{x:.3f}<br>Bias=%{y:.3f}<br>BCE=%{z:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=theta[:, 0], y=theta[:, 1], z=values, mode="lines+markers",
        marker=dict(size=3, color=PURPLE), line=dict(color=PURPLE, width=6),
        customdata=trace["iterations"], name="NumPy gradient descent",
        hovertemplate="Iteration %{customdata}<br>Weight=%{x:.3f}<br>Bias=%{y:.3f}<br>BCE=%{z:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=theta[[0, -1], 0], y=theta[[0, -1], 1], z=values[[0, -1]],
        mode="markers+text", text=["Start", "End"], textposition="top center",
        marker=dict(size=6, color=[ORANGE, BLUE]), name="Optimization endpoints"))
    fig.update_layout(scene=dict(xaxis_title="Weight w", yaxis_title="Bias b",
                                  zaxis_title="Mean binary cross-entropy"))
    write_interactive(fig, out, OUTPUTS[12],
        "13 · A convex loss surface and a learning path<br><sup>One weight + bias · 120 synthetic Bernoulli observations · unpenalized BCE · finite grid illustrates, rather than proves, convexity.</sup>")
    return dict(initial_loss=float(values[0]), final_loss=float(values[-1]),
                final_parameters=theta[-1], steps=160)


def create_sigmoid_animation(out):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.2, top=0.75)
    heading(fig, 14, "Weight controls orientation; bias shifts the score",
            "First vary w with b=0, then vary b with w=1. These are parameter changes, not fitted models.",
            "When w ≠ 0, the p=0.5 location is x=−b/w. At w=0 the prediction is constant.")
    x = np.linspace(-6, 6, 400)
    curve, = ax.plot(x, sigmoid(x), color=PURPLE, lw=3)
    boundary, = ax.plot([], [], color=MUTED, ls="--", lw=1.5)
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.set(xlim=(-6, 6), ylim=(-0.03, 1.03), xlabel="Feature x", ylabel="σ(wx+b)")
    label = ax.text(0.04, 0.91, "", transform=ax.transAxes, fontsize=15,
                    bbox=dict(facecolor=BACKGROUND, edgecolor="none", alpha=0.9))
    # Each phase returns to w=1, b=0, avoiding an abrupt switch in the curve.
    weights = 1 + 2.5 * np.sin(np.linspace(0, 2*np.pi, 24, endpoint=True))
    biases = 3 * np.sin(np.linspace(0, 2*np.pi, 24, endpoint=True))
    configurations = [(w, 0.0, "Vary weight") for w in weights] + [(1.0, b, "Vary bias") for b in biases]
    def update(frame):
        w, b, phase = configurations[frame]
        curve.set_ydata(sigmoid(w*x+b))
        if abs(w) > 1e-12:
            boundary.set_data([-b/w, -b/w], [0, 1])
        else:
            boundary.set_data([], [])
        label.set_text(f"{phase}   |   w={w:+.2f}, b={b:+.2f}")
    save_gif(fig, update, len(configurations), out, OUTPUTS[13])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Generate all views (default).")
    group.add_argument("--only", nargs="+", choices=[
        "sigmoid", "logit", "loss", "boundary", "threshold", "3d", "gradient",
        "regularization", "nonlinear", "calibration", "pipeline", "landscape", "parameters"])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "visuals")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    style()
    case = classification_case()
    jobs = {
        "sigmoid": lambda: create_sigmoid_visualization(out),
        "logit": lambda: create_logit_visualization(out),
        "loss": lambda: create_cross_entropy_visualization(out),
        "boundary": lambda: create_probability_surface(out, case),
        "threshold": lambda: create_threshold_animation(out, case),
        "3d": lambda: create_3d_probability_surface(out, case),
        "gradient": lambda: create_gradient_descent_animation(out, case),
        "regularization": lambda: create_regularization_visualization(out),
        "nonlinear": lambda: create_nonlinear_comparison(out),
        "calibration": lambda: create_calibration_visualization(out),
        "pipeline": lambda: create_pipeline_visualization(out),
        "landscape": lambda: create_loss_surface(out),
        "parameters": lambda: create_sigmoid_animation(out),
    }
    selected = args.only or list(jobs)
    results = {"seed": SEED, "synthetic": True,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                     "plotly": plotly.__version__},
        "selection": selected, "experiments": {}}
    for index, name in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] Creating {name} visualization...", flush=True)
        result = jobs[name]()
        if result is not None:
            results["experiments"][name] = finite_json(result)
    # Separate partial-run records so they never replace a complete run's evidence.
    record = "experiment_results.json" if not args.only else "experiment_results_selected.json"
    (out / record).write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    if not args.only:
        missing = [name for name in OUTPUTS if not (out / name).is_file()]
        if missing:
            raise RuntimeError(f"Missing expected outputs: {missing}")
    print(f"Done. Visualizations saved to {out}", flush=True)


if __name__ == "__main__":
    main()