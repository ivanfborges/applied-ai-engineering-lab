"""Matplotlib scenes and a self-contained Plotly explorer; no app server."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import plotly.graph_objects as go

from pca_visual_core import (covariance_ellipse, direction_variance, fit_view,
                             geometry_data, outlier_states, plane_data,
                             predictive_experiment, reconstruct, scaling_views,
                             spectrum_data, threshold_count, wine_weights)


INK, MUTED, BLUE, ORANGE, PURPLE = "#172B45", "#77889A", "#0072B2", "#D55E00", "#7759A5"
PC_COLORS = [BLUE, ORANGE, PURPLE]


def style():
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13,
                         "axes.labelsize": 11, "figure.facecolor": "#F4F7FB",
                         "axes.facecolor": "white", "text.color": INK,
                         "axes.labelcolor": INK, "axes.edgecolor": "#B4C0CC",
                         "grid.color": "#DDE4EB", "grid.alpha": .65,
                         "savefig.facecolor": "#F4F7FB"})


def decorate(ax, xlabel, ylabel, equal=False):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.4)
    ax.set_axisbelow(True)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def footer(fig, text):
    fig.text(.5, .035, text, ha="center", va="center", fontsize=10, color=INK)


def finish(fig, output, stem, update=None, frames=None, fast=False, snapshot=-1, show=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if update is not None:
        animation = FuncAnimation(fig, update, frames=frames, interval=100,
                                  repeat=True, blit=False, cache_frame_data=False)
        animation.save(output / f"{stem}.gif", writer=PillowWriter(fps=10), dpi=65 if fast else 90)
        update(frames[snapshot])
    fig.savefig(output / f"{stem}.png", dpi=110 if fast else 180)
    if show:
        plt.show()
    plt.close(fig)


def geometry(output, fast=False, show=False):
    view = fit_view(geometry_data())
    X, model = view["X"], view["model"]
    mean, components = model.mean_, model.components_
    target = np.arctan2(components[0, 1], components[0, 0])
    projected = reconstruct(view, 1)["Xhat"]
    ellipse = covariance_ellipse(view)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.15, 1, .85]})
    fig.subplots_adjust(left=.06, right=.97, bottom=.19, top=.78, wspace=.4)
    fig.suptitle("PCA finds the direction of maximum variance", fontsize=19, y=.96)
    subtitle = fig.text(.5, .865, "", ha="center", fontsize=12)
    footer(fig, "Synthetic n=100, seed=42 | ellipse: two standard deviations | fixed points during the basis change")
    count = 10 if fast else 22
    states = [("original", 0.)] * 7 + [("basis", t) for t in np.linspace(0, 1, count)]
    states += [("basis", 1.)] * 7 + [("projection", t) for t in np.linspace(0, 1, count)]
    states += [("projection", 1.)] * 7
    extent = max(np.abs(X - mean).max(), 4.) + 1

    def update(state):
        phase, t = state
        for ax in axes:
            ax.clear()
        ax, coord, diagnostic = axes
        ax.scatter(*X.T, s=16, color=MUTED, alpha=.6, label="Original observations")
        ax.plot(*ellipse.T, color=MUTED, lw=1.5, label="Covariance ellipse")
        ax.axhline(0, color="#B4C0CC", lw=1)
        ax.axvline(0, color="#B4C0CC", lw=1)
        ax.scatter(*mean, marker="*", color=INK, s=130, zorder=5, label="Training mean")
        angle = target * t if phase == "basis" else (target if phase == "projection" else 0)
        basis = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        if phase != "original":
            for j in range(2):
                direction = basis[j] * 3
                ax.quiver(*mean, *direction, angles="xy", scale_units="xy", scale=1,
                          color=PC_COLORS[j], width=.009)
                ax.text(*(mean + direction * 1.05), f"PC{j+1}" if phase == "projection" or t == 1 else f"axis {j+1}",
                        color=PC_COLORS[j], fontsize=11)
        centered_scores = (X - mean) @ basis.T
        if phase == "projection":
            moving = X + t * (projected - X)
            ax.scatter(*moving.T, marker="x", color=BLUE, s=22, label="Projection motion")
            for i in range(0, len(X), 8):
                ax.plot([X[i, 0], projected[i, 0]], [X[i, 1], projected[i, 1]], "--", color=BLUE, alpha=.45, lw=.8)
            centered_scores[:, 1] *= 1 - t
        coord.scatter(*centered_scores.T, s=17, color=BLUE)
        coord.axhline(0, color=INK, lw=.8)
        coord.axvline(0, color=INK, lw=.8)
        coord.set(xlim=(-extent, extent), ylim=(-extent, extent))
        ax.set(xlim=(mean[0]-extent, mean[0]+extent), ylim=(mean[1]-extent, mean[1]+extent))
        decorate(ax, "Original x", "Original y", equal=True)
        decorate(coord, "Coordinate along axis 1", "Coordinate along axis 2", equal=True)
        ax.set_title("Original measurement space")
        coord.set_title("Coordinates in the displayed basis")
        angles = np.linspace(-np.pi/2, np.pi/2, 150)
        variances = [direction_variance(X, a) for a in angles]
        diagnostic.plot(np.rad2deg(angles), variances, color=BLUE, lw=2)
        diagnostic.axvline(np.rad2deg(target), color=ORANGE, ls="--", label="Fitted PC1")
        diagnostic.scatter([np.rad2deg(angle)], [direction_variance(X, angle)], color=INK, zorder=5)
        diagnostic.set_title("Variance along a unit direction")
        decorate(diagnostic, "Direction angle (degrees)", "Sample variance")
        diagnostic.text(.04, .96, f"PC1 variance = {model.explained_variance_[0]:.3f}\n"
                        f"PC2 variance = {model.explained_variance_[1]:.3f}\n" + r"PC1 $\perp$ PC2", transform=diagnostic.transAxes,
                        va="top", fontsize=10)
        diagnostic.legend(loc="lower right", fontsize=9)
        if phase == "original":
            subtitle.set_text("1 / Center the data: the mean anchors the new coordinate system")
        elif phase == "basis":
            subtitle.set_text("2 / Rotate the basis toward PC1 — original observations stay fixed")
        else:
            subtitle.set_text(r"3 / Projection discards PC2:  $z_1=(x-\mu)^\top v_1$,   $\hat{x}=\mu+z_1v_1$")
        ax.legend(loc="lower left", fontsize=8)
        return []

    finish(fig, output, "01_pca_geometry", update, states, fast, show=show)
    return {"sample_count": len(X), "seed": 42, "eigenvalues": model.explained_variance_.tolist(),
            "pc1_angle_degrees": float(np.rad2deg(target)), "frames_requested": len(states)}


def interactive_3d_figure():
    view = fit_view(plane_data())
    X, model = view["X"], view["model"]
    mean = model.mean_
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers",
                              name="Original", marker=dict(size=4, color=MUTED, opacity=.65),
                              customdata=np.arange(len(X)),
                              hovertemplate="row %{customdata}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>Original</extra>"))
    fig.add_trace(go.Scatter3d(x=[mean[0]], y=[mean[1]], z=[mean[2]], mode="markers",
                              name="Training mean", marker=dict(size=7, color=INK, symbol="diamond")))
    for j in range(3):
        end = mean + 2 * np.sqrt(model.explained_variance_[j]) * model.components_[j]
        fig.add_trace(go.Scatter3d(x=[mean[0], end[0]], y=[mean[1], end[1]], z=[mean[2], end[2]],
                                  mode="lines+markers+text", text=["", f"PC{j+1}"],
                                  textposition="bottom right" if j == 2 else "top center",
                                  textfont=dict(color=PC_COLORS[j]),
                                  marker=dict(size=[0, 5], color=PC_COLORS[j], symbol="diamond"),
                                  line=dict(width=7, color=PC_COLORS[j]), name=f"PC{j+1} (+2 SD)"))
    u, v = np.meshgrid(np.linspace(-5, 5, 8), np.linspace(-3, 3, 8))
    plane = mean + u[..., None] * model.components_[0] + v[..., None] * model.components_[1]
    fig.add_trace(go.Surface(x=plane[..., 0], y=plane[..., 1], z=plane[..., 2],
                             surfacecolor=np.zeros_like(u), colorscale=[[0, BLUE], [1, BLUE]],
                             opacity=.13, showscale=False, name="PC1–PC2 plane", hoverinfo="skip"))
    records = []
    for k in (1, 2, 3):
        state = reconstruct(view, k)
        records.append({key: state[key] for key in ("k", "evr", "mse")})
        R = state["Xhat"]
        fig.add_trace(go.Scatter3d(x=R[:, 0], y=R[:, 1], z=R[:, 2], mode="markers",
                                  name="Reconstructed", marker=dict(size=5, color=ORANGE, symbol="diamond"),
                                  customdata=np.arange(len(X)), visible=k == 2,
                                  hovertemplate="row %{customdata}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>Reconstructed</extra>"))
        segments = np.stack([X, R, np.full_like(X, np.nan)], axis=1).reshape(-1, 3)
        fig.add_trace(go.Scatter3d(x=segments[:, 0], y=segments[:, 1], z=segments[:, 2], mode="lines",
                                  name="Orthogonal residuals", line=dict(color=ORANGE, width=2, dash="dash"),
                                  hoverinfo="skip", visible=k == 2))

    def title(record):
        return (f"Compress a 3D cloud: k={record['k']} | retained variance={record['evr']:.2%}"
                f" | MSE={record['mse']:.5f}")

    steps = []
    for i, record in enumerate(records):
        visible = [True] * 5 + [i == 1] + [j // 2 == i for j in range(6)]
        steps.append(dict(method="update", label=str(i + 1),
                          args=[{"visible": visible}, {"title.text": title(record)}]))
    low, high = X.min(axis=0) - 1, X.max(axis=0) + 1
    eye = 2 * model.components_[2] + .8 * model.components_[0] + .6 * model.components_[1]
    fig.update_layout(title=dict(text=title(records[1]), x=.04, font=dict(size=21)),
                      template="plotly_white", height=900, margin=dict(l=10, r=10, t=170, b=130),
                      font=dict(family="Arial", size=14, color=INK),
                      legend=dict(orientation="h", y=1.03, x=0),
                      scene=dict(aspectmode="data", dragmode="orbit",
                                 xaxis=dict(title="Original x", range=[low[0], high[0]]),
                                 yaxis=dict(title="Original y", range=[low[1], high[1]]),
                                 zaxis=dict(title="Original z", range=[low[2], high[2]]),
                                 camera=dict(eye=dict(zip(("x", "y", "z"), eye)))),
                      uirevision="preserve-camera", meta=dict(states=records, seed=43, sample_count=len(X)),
                      sliders=[dict(active=1, x=.1, len=.8, y=-.04,
                                    currentvalue=dict(prefix="Retained components: "), steps=steps)],
                      annotations=[dict(x=.5, y=1.12, xref="paper", yref="paper", showarrow=False,
                                        text="Drag: rotate · wheel: zoom · hover: row/coordinates · slider: exact PCA reconstruction<br>"
                                             "Synthetic n=100, seed=43 · blue plane appears at k=2 · errors are in original units")])
    return fig, records


def interactive_3d(output, fast=False, show=False):
    fig, records = interactive_3d_figure()
    path = Path(output) / "03_pca_3d_interactive.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=True, full_html=True, div_id="pca-3d",
                   config={"responsive": True, "displaylogo": False, "scrollZoom": True})
    return {"seed": 43, "sample_count": 100, "states": records, "offline_plotly_embedded": True}


def explained_variance(output, fast=False, show=False):
    view = fit_view(spectrum_data())
    ratios = view["model"].explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    retained = threshold_count(view)
    p = len(ratios)
    fig, (ax, total) = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(bottom=.2, top=.76, left=.08, right=.96, wspace=.3)
    fig.suptitle("Explained variance answers a reconstruction question", fontsize=18, y=.96)
    caption = fig.text(.5, .85, "", ha="center", fontsize=13)
    footer(fig, "Synthetic n=180, p=6, seed=44 | full SVD, raw units | 95% is a heuristic, not a prediction guarantee")

    def update(k):
        ax.clear()
        total.clear()
        x = np.arange(1, k + 1)
        ax.bar(x, ratios[:k] * 100, color=[BLUE if j < retained else MUTED for j in range(k)])
        for j in range(k):
            ax.text(j + 1, ratios[j] * 100 + 1, f"{ratios[j]:.1%}", ha="center", fontsize=10)
        total.plot(x, cumulative[:k] * 100, "o-", color=BLUE, lw=2)
        total.axhline(95, color=ORANGE, ls="--", label="95% threshold")
        total.axvline(retained, color=ORANGE, ls=":", label=f"First crossing: k={retained}")
        for panel in (ax, total):
            panel.set(xlim=(.4, p + .6), xticks=range(1, p + 1), ylim=(0, 105))
            decorate(panel, "Principal component", "Variance (%)")
        ax.set_title("Individual variance explained")
        total.set_title("Cumulative variance explained")
        total.legend(loc="lower right", fontsize=10)
        caption.set_text(f"Revealed {k}/{p} components: cumulative {cumulative[k-1]:.2%}"
                         + (f"  |  keep {retained} of {p} for >95%" if k >= retained else ""))

    frames = [k for k in range(1, p + 1) for _ in range(8)]
    finish(fig, output, "02_explained_variance", update, frames, fast, show=show)
    return {"seed": 44, "sample_count": 180, "feature_count": p, "selected_k": retained,
            "explained_variance_ratio": ratios.tolist(), "cumulative": cumulative.tolist()}


def reconstruction(output, fast=False, show=False):
    view = fit_view(plane_data(), standardize=True)
    X = view["X"]
    states = [reconstruct(view, k) for k in (1, 2, 3)]
    raw_basis = fit_view(X)["model"].components_
    eye = 2 * raw_basis[2] + .8 * raw_basis[0] + .6 * raw_basis[1]
    elevation = np.rad2deg(np.arctan2(eye[2], np.linalg.norm(eye[:2])))
    azimuth = np.rad2deg(np.arctan2(eye[1], eye[0]))
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(121, projection="3d")
    error_ax = fig.add_subplot(122)
    fig.subplots_adjust(bottom=.21, top=.79, left=.04, right=.96, wspace=.25)
    fig.suptitle("Retaining more components reduces reconstruction error", fontsize=18, y=.96)
    caption = fig.text(.5, .865, "", ha="center", fontsize=12)
    footer(fig, "Synthetic n=100, seed=43 | standardized PCA → inverse scaling | residuals drawn in original units\n"
           "Orthogonality applies in standardized space; fractional frames are display interpolation between exact ranks")
    count = 6 if fast else 16
    frames = [(0, 0.)] * 7
    for j in (1, 2):
        frames += [(j, t) for t in np.linspace(0, 1, count)] + [(j, 1.)] * 7

    def update(frame):
        j, t = frame
        state = states[j]
        previous = states[max(0, j-1)]["Xhat"]
        R = previous * (1-t) + state["Xhat"] * t if j else state["Xhat"]
        working = view["scaler"].transform(R)
        mse = np.mean((view["working"] - working) ** 2)
        raw_mse = np.mean((X - R) ** 2)
        ax.clear()
        error_ax.clear()
        ax.scatter(*X.T, s=10, color=MUTED, alpha=.5, label="Original")
        ax.scatter(*R.T, s=15, color=ORANGE, marker="x", label="Reconstructed")
        for a, b in zip(X[::4], R[::4]):
            ax.plot(*np.vstack([a, b]).T, "--", color=ORANGE, alpha=.55, lw=.8)
        lo, hi = X.min(axis=0)-.6, X.max(axis=0)+.6
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
               xlabel="Original x", ylabel="Original y", zlabel="Original z")
        ax.set_box_aspect(hi-lo)
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set_title("Projection → inverse scaling → original space")
        ax.legend(loc="upper left", fontsize=9)
        errors = [s["mse"] for s in states]
        error_ax.plot([1, 2, 3], errors, "o-", color=BLUE, lw=2, label="Exact PCA ranks")
        error_ax.scatter([j+t if j else 1], [mse], color=ORANGE, s=75, zorder=5)
        error_ax.set(xticks=[1, 2, 3], xlim=(.8, 3.2), ylim=(-.01, max(errors)*1.15))
        decorate(error_ax, "Retained components (transition shown between integers)", "MSE in standardized feature units")
        error_ax.set_title(r"$\hat{X}=ZV_k^\top+\mu$ before reversing scaling")
        error_ax.legend(loc="upper right", fontsize=9)
        exact = j == 0 or t == 1
        label = f"Exact k={j+1}" if exact else f"Adding PC{j+1}: display interpolation {t:.0%}"
        caption.set_text(f"{label} | standardized MSE={mse:.5f} | original-unit MSE={raw_mse:.5f}")

    # Save an informative nonzero-error endpoint rather than the overlapping full-rank view.
    snapshot = next(i for i, state in enumerate(frames) if state == (1, 1.))
    finish(fig, output, "04_pca_reconstruction", update, frames, fast, snapshot=snapshot, show=show)
    return {"seed": 43, "sample_count": 100, "standardized": True,
            "states": [{key: s[key] for key in ("k", "evr", "mse", "raw_mse")} for s in states]}


def scaling(output, fast=False, show=False):
    raw, standardized = scaling_views()
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.subplots_adjust(bottom=.24, top=.76, left=.08, right=.96, wspace=.3)
    fig.suptitle("Feature scale changes what PCA calls important", fontsize=19, y=.96)
    fig.text(.5, .87, "Identical 130 rows, seed=45 — each panel uses equal distance per displayed unit", ha="center", fontsize=12)
    for ax, view, label in zip(axes, (raw, standardized), ("Raw measurement units", "Training-standardized units")):
        X, model = view["working"], view["model"]
        ax.scatter(*X.T, s=19, color=MUTED, alpha=.6)
        extent = max(np.abs(X - model.mean_).max(), 1) * 1.15
        ax.set(xlim=(model.mean_[0]-extent, model.mean_[0]+extent),
               ylim=(model.mean_[1]-extent, model.mean_[1]+extent))
        for j in (0, 1):
            # Unit directions, with equal display length; lengths here are not eigenvalues.
            vector = model.components_[j] * extent * .65
            ax.quiver(*model.mean_, *vector, angles="xy", scale_units="xy", scale=1,
                      color=PC_COLORS[j], width=.008, label=f"PC{j+1}: {model.explained_variance_ratio_[j]:.2%} EVR")
        ax.set_title(label)
        decorate(ax, "Feature 1" if view is raw else "Standardized feature 1",
                 "Feature 2" if view is raw else "Standardized feature 2", equal=True)
        ax.legend(loc="lower left", fontsize=10)
    footer(fig, "The thin raw cloud is real: feature 1 has SD≈100, feature 2 SD≈1.\n"
           "Arrow lengths are equal for visibility; compare directions and EVR. Scaling is a modeling choice, not a universal requirement.")
    finish(fig, output, "05_scaling_comparison", fast=fast, show=show)
    return {"seed": 45, "sample_count": 130,
            "raw_pc1_evr": float(raw["model"].explained_variance_ratio_[0]),
            "standardized_pc1_evr": float(standardized["model"].explained_variance_ratio_[0]),
            "raw_pc1": raw["model"].components_[0].tolist(),
            "standardized_pc1": standardized["model"].components_[0].tolist()}


def predictive_information(output, fast=False, show=False):
    experiment = predictive_experiment()
    view, test = experiment["view"], experiment["test"]
    X, y, model = experiment["X"][test], experiment["y"][test], view["model"]
    scores = model.transform(X)
    fig, axes = plt.subplots(1, 4, figsize=(16, 6.5))
    fig.subplots_adjust(bottom=.27, top=.77, left=.06, right=.98, wspace=.45)
    fig.suptitle("High explained variance does not guarantee predictive power", fontsize=19, y=.96)
    fig.text(.5, .865, f"Raw PC1 retains {model.explained_variance_ratio_[0]:.4%} of training variance; the label is in the small direction", ha="center", fontsize=12)
    markers = ["o", "^"]
    for cls in (0, 1):
        mask = y == cls
        color = PC_COLORS[cls]
        axes[0].scatter(*X[mask].T, s=18, color=color, marker=markers[cls], alpha=.7, label=f"Class {cls}")
        # Vertical row offset is only to reveal class overlap on the one-dimensional axis.
        offset = np.full(mask.sum(), cls * .1)
        axes[1].scatter(scores[mask, 0], offset, s=19, color=color, marker=markers[cls], alpha=.6)
        axes[2].scatter(scores[mask, 1], offset, s=19, color=color, marker=markers[cls], alpha=.6)
    vector = model.components_[0] * 35
    axes[0].quiver(*model.mean_, *vector, angles="xy", scale_units="xy", scale=1,
                   color=INK, width=.012, label="PC1 (raw units)")
    axes[0].set_title("Held-out original observations")
    decorate(axes[0], "High-variance nuisance", "Low-variance label feature")
    axes[0].legend(loc="upper left", fontsize=8)
    for j, title in ((1, "Keep PC1: class overlap"), (2, "Inspect PC2: low-variance signal")):
        axes[j].set_title(title)
        axes[j].set(yticks=[], ylim=(-.1, .2))
        decorate(axes[j], f"PC{j} score", "Class rows for visibility")
    names = ["Original", "PCA k=1", "PCA k=2"]
    values = list(experiment["balanced_accuracy"].values())
    axes[3].bar(range(3), values, color=[MUTED, ORANGE, BLUE])
    for i, value in enumerate(values):
        axes[3].text(i, value + .025, f"{value:.3f}", ha="center")
    axes[3].set(xticks=range(3), xticklabels=names, ylim=(0, 1.1), title="Actual held-out classification")
    axes[3].tick_params(axis="x", labelrotation=20)
    decorate(axes[3], "Fixed model", "Balanced accuracy")
    footer(fig, "Synthetic n=500, seed=46; 70/30 stratified split. PCA fitted in raw units on training rows only.\n"
           "Displayed scatter uses unequal axis scales to reveal the small feature. PC score panels use independent horizontal scales.\n"
           "Baseline and retained scores are standardized before LR (C=1); deliberately constructed failure case, not a general PCA ranking.")
    finish(fig, output, "06_variance_vs_predictive_information", fast=fast, show=show)
    return {"seed": 46, "sample_count": 500, "train_count": len(experiment["train"]),
            "test_count": len(test), "pc1_training_evr": float(model.explained_variance_ratio_[0]),
            "balanced_accuracy": experiment["balanced_accuracy"]}


def outliers(output, fast=False, show=False):
    states = outlier_states()
    baseline = states[0][0]["model"]
    fig, (ax, diagnostic) = plt.subplots(1, 2, figsize=(12, 7))
    fig.subplots_adjust(bottom=.21, top=.76, left=.07, right=.97, wspace=.3)
    fig.suptitle("Extreme observations can redirect principal components", fontsize=18, y=.96)
    caption = fig.text(.5, .865, "", ha="center", fontsize=12)
    footer(fig, "Synthetic clean n=100, seed=42 | add [3,18], [2,22], [1,25], [-1,28] in order | no scaling\n"
           "Every displayed fit uses an actual augmented dataset. Outliers are intentionally strong; ordinary PCA is not robust.")
    records = []
    for view, count in states:
        model = view["model"]
        angle = np.rad2deg(np.arccos(np.clip(np.abs(model.components_[0] @ baseline.components_[0]), 0, 1)))
        records.append({"added_outliers": count, "pc1_evr": float(model.explained_variance_ratio_[0]),
                        "mean": model.mean_.tolist(), "unsigned_pc1_angle_change": float(angle)})

    def update(index):
        view, count = states[index]
        X, model = view["X"], view["model"]
        ax.clear()
        diagnostic.clear()
        ax.scatter(*X[:100].T, s=15, color=MUTED, alpha=.6, label="Clean observations")
        if count:
            ax.scatter(*X[100:].T, s=90, color=ORANGE, marker="X", label="Added outliers")
        for label, fitted, color, linestyle in (("Clean PC1", baseline, BLUE, "--"),
                                                ("Refitted PC1", model, ORANGE, "-")):
            line = fitted.mean_ + np.array([[-8], [8]]) * fitted.components_[0]
            ax.plot(*line.T, color=color, ls=linestyle, lw=2.5, label=label)
            ax.scatter(*fitted.mean_, color=color, marker="*", s=150, zorder=5)
        ax.set(xlim=(-12, 18), ylim=(-2, 30))
        decorate(ax, "Original x", "Original y", equal=True)
        ax.set_title("PC1 lines pass through their own fitted means")
        x = np.arange(index + 1)
        diagnostic.plot(x, [r["pc1_evr"]*100 for r in records[:index+1]], "o-", color=BLUE)
        diagnostic.set(xlim=(-.2, 4.2), ylim=(50, 101), xticks=range(5))
        decorate(diagnostic, "Number of added outliers", "PC1 explained variance (%)")
        diagnostic.set_title("Both the mean and covariance are refitted")
        handles, labels = ax.get_legend_handles_labels()
        diagnostic.legend(handles, labels, loc="lower left", fontsize=9)
        record = records[index]
        caption.set_text(f"Added {count} outliers | mean=({model.mean_[0]:.2f}, {model.mean_[1]:.2f})"
                         f" | unsigned PC1 change={record['unsigned_pc1_angle_change']:.1f}°")

    frames = [i for i in range(len(states)) for _ in range(12)]
    finish(fig, output, "07_outlier_sensitivity", update, frames, fast, show=show)
    return {"clean_seed": 42, "clean_count": 100, "outliers": [[3, 18], [2, 22], [1, 25], [-1, 28]],
            "states": records}


def loadings(output, fast=False, show=False):
    view, names = wine_weights()
    weights = view["model"].components_[:3]
    fig, ax = plt.subplots(figsize=(14, 7.8))
    fig.subplots_adjust(left=.07, right=.89, top=.74, bottom=.35)
    fig.suptitle("Component weights describe a direction, not a causal factor", fontsize=19, y=.96)
    fig.text(.5, .865, "Wine: standardized training features only (133 rows) | cell = signed direction coefficient vⱼᵢ", ha="center", fontsize=12)
    bound = max(abs(weights.min()), abs(weights.max()))
    image = ax.imshow(weights, cmap="PuOr", vmin=-bound, vmax=bound, aspect="auto")
    ax.set(xticks=range(len(names)), xticklabels=[name.replace("_", " ") for name in names],
           yticks=range(3), yticklabels=[f"PC{i+1}" for i in range(3)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    for j in range(3):
        for i in range(len(names)):
            value = weights[j, i]
            ax.text(i, j, f"{value:+.2f}", ha="center", va="center",
                    color="white" if abs(value) > bound*.65 else INK, fontsize=10)
    cax = fig.add_axes([.92, .35, .015, .39])
    fig.colorbar(image, cax=cax, label="Direction weight (not feature-score correlation)")
    footer(fig, "Positive / negative: features move together / oppositely along the chosen axis sign. Magnitude depends on scaling.\n"
           "Flipping a component AND its scores leaves reconstruction unchanged. Display sign: largest absolute coefficient is positive.\n"
           "These are eigenvector coefficients, not v√λ loadings or correlations. No causal or chemical-factor interpretation is validated.")
    finish(fig, output, "08_component_loadings", fast=fast, show=show)
    return {"dataset": "UCI Wine via sklearn", "training_rows": 133, "split_seed": 42,
            "standardized": True, "features": names, "direction_weights": weights.tolist(),
            "explained_variance_ratio": view["model"].explained_variance_ratio_[:3].tolist()}
