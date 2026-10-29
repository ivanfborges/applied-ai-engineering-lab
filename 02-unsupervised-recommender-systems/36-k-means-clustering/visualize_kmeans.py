"""Export a local K-Means teaching lab: Matplotlib/Pillow and offline Plotly."""

from __future__ import annotations

import argparse
import html
import json
import platform
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import plotly
import plotly.graph_objects as go
import sklearn
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from from_scratch import predict_labels
from kmeans_visual_core import (
    convergence_data, fit_reference, initialization_experiment,
    limitations_experiment, lloyd_trace, scaling_experiment,
    selection_experiment, three_dimensional_data,
)


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
          "#56B4E9", "#7B61A8", "#8C6D31", "#5C6B73", "#A23B72"]
INK, MUTED, PAPER = "#16283F", "#546478", "#F5F7FB"
OUTPUT_NAMES = {
    "convergence": "01_kmeans_convergence.gif",
    "scaling": "02_feature_scaling_comparison.png",
    "elbow": "03_k_selection.png",
    "boundaries": "04_decision_boundaries.png",
    "limitations": "05_kmeans_limitations.png",
    "3d": "06_interactive_kmeans_3d.html",
    "initialization": "07_initialization_comparison.png",
}
SCENE_GUIDE = {
    "convergence": ("Follow the exact updates", "Watch assignments change only when points choose their nearest center.",
                    "The mean minimizes squared loss for fixed assignments.", "Convergence alone does not certify a useful or globally optimal partition."),
    "scaling": ("Question the representation", "Compare the partitions in the same original coordinates and inspect the distance shares.",
                "StandardScaler applies inverse-variance weights to squared differences.", "Choose feature weights deliberately and preserve the fitted transformation."),
    "elbow": ("Evaluate K", "Inspect diminishing inertia reductions, silhouette and cluster sizes together.",
              "More centers improve compactness; silhouette also accounts for separation.", "Internal metrics need stability, profile and downstream validation."),
    "boundaries": ("Inspect the geometry", "Compare regions before and after the controlled edit of one center.",
                   "Equal Euclidean distances form bisectors and convex Voronoi cells.", "A compactness objective may split a nonconvex population."),
    "limitations": ("Challenge the assumptions", "Inspect moons, unequal densities and distant observations beside a compact control.",
                    "Centroid loss and density connectivity encode different grouping rules.", "DBSCAN also needs justified parameters and can fail at varying densities."),
    "3d": ("Explore three real features", "Rotate the cloud, hover over points and move the exact-iteration slider.",
           "The same Lloyd updates work in three dimensions; centers are inverse-transformed for display.",
           "Embedding geometry and feature scaling need validation before assigning semantic names."),
    "initialization": ("Inspect local solutions", "Compare actual starting squares, final means and reported objectives across seeds.",
                       "Different starts can reach different local solutions; restarts retain the best observed objective.",
                       "Specify and evaluate a restart budget rather than assuming an initializer guarantees success."),
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "figure.facecolor": PAPER, "axes.facecolor": "white",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.edgecolor": "#CBD3DF", "axes.spines.top": False,
        "axes.spines.right": False, "grid.color": "#DFE5EE", "grid.alpha": 0.7,
        "savefig.facecolor": PAPER,
    })


def page(fig, title, subtitle, footer):
    fig.suptitle(title, x=0.065, y=0.98, ha="left", fontsize=21, weight="bold")
    fig.text(0.065, 0.92, subtitle, fontsize=11, color=MUTED, va="top")
    fig.text(0.065, 0.035, footer, fontsize=10, color=MUTED, va="bottom", linespacing=1.5)


def limits(X, centers=None, padding=0.12):
    values = X if centers is None else np.vstack((X, centers))
    lower, upper = values.min(axis=0), values.max(axis=0)
    margin = np.maximum(upper - lower, 1) * padding
    return lower - margin, upper + margin


def scatter(ax, X, labels, centers=None, *, size=22):
    colors = [COLORS[int(label) % len(COLORS)] if label >= 0 else "#A8B0BC" for label in labels]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=size, alpha=0.78, edgecolors="white", linewidths=0.25)
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c=COLORS[:len(centers)], marker="X", s=190,
                   edgecolors=INK, linewidths=1.4, zorder=5)
        for index, center in enumerate(centers):
            ax.annotate(f"C{index}", center, xytext=(7, 8), textcoords="offset points",
                        fontsize=9, weight="bold", color=INK)


def legend(ax, k=3, *, extra=None):
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=COLORS[j],
                      markeredgecolor="white", label=f"Cluster {j}", markersize=7) for j in range(k)]
    handles.append(Line2D([], [], marker="X", color="none", markerfacecolor="white",
                          markeredgecolor=INK, label="Centroid", markersize=9))
    if extra:
        handles.extend(extra)
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.92)


def save_png(fig, path):
    fig.savefig(path, dpi=140)


def convergence_scene(path, seed):
    X, initial = convergence_data(seed)
    states = lloyd_trace(X, initial)
    fig, (ax, objective) = plt.subplots(1, 2, figsize=(12, 7.8), gridspec_kw={"width_ratios": [1.7, 1]})
    fig.subplots_adjust(left=0.075, right=0.96, top=0.81, bottom=0.32, wspace=0.32)
    page(fig, "01 / Watch Lloyd's algorithm", "One run, fixed center identities, exact assignment and mean-update states.",
         "Colored points = displayed assignments. X = centroid. Arrows = exact mean updates.\n"
         "No interpolated optimization steps; empty centers retain their previous position. Cluster IDs are arbitrary.")
    lower, upper = limits(X, initial)
    phase_text = {
        "initial": "Initial centers\nPoints are not colored yet.\nJ is the nearest-center baseline.",
        "assignment": "ASSIGN\nChoose the nearest center.\nCenters stay fixed.",
        "update": "UPDATE\nMove centers to member means.\nAssignments stay fixed for this step.",
    }

    def draw(index):
        state = states[index]
        ax.clear()
        objective.clear()
        if state.phase == "initial":
            ax.scatter(X[:, 0], X[:, 1], c="#A8B0BC", s=22, edgecolors="white", linewidths=0.25)
            scatter(ax, np.empty((0, 2)), [], state.centers)
        else:
            scatter(ax, X, state.labels, state.centers)
        updates = [states[0]] + [s for s in states[1:index + 1] if s.phase == "update"]
        for previous, current in zip(updates, updates[1:]):
            for j, (old, new) in enumerate(zip(previous.centers, current.centers)):
                ax.annotate("", xy=new, xytext=old,
                            arrowprops={"arrowstyle": "->", "color": COLORS[j], "lw": 1.6, "alpha": 0.75})
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
               xlabel="Feature 1 (synthetic units)", ylabel="Feature 2 (synthetic units)")
        ax.set_aspect("equal", adjustable="box")
        status = "CONVERGED: assignments unchanged" if state.converged else f"{state.phase.upper()} step"
        ax.set_title(f"Mean updates: {state.iteration}  |  {status}", fontsize=11, pad=14)
        legend(ax)
        objective.plot(np.arange(len(states)), [s.inertia for s in states], color="#D8DFE9", lw=2)
        for phase, marker, color in (("assignment", "o", COLORS[0]), ("update", "s", COLORS[1])):
            indices = [i for i, s in enumerate(states[:index + 1]) if s.phase == phase]
            objective.scatter(indices, [states[i].inertia for i in indices], marker=marker, c=color, s=36, label=phase.title())
        objective.set(xlim=(-0.4, len(states) - 0.5), ylim=(0, states[0].inertia * 1.12),
                      xlabel="Exact state index (assignment / update)", ylabel="J for displayed centers and labels")
        objective.grid(axis="y")
        objective.set_title(f"Current inertia: {state.inertia:,.2f}", fontsize=12, weight="bold", pad=14)
        objective.legend(loc="upper right", fontsize=9)
        explanation.set_text(phase_text[state.phase] if not state.converged else
                             "CONVERGED\nNo assignments changed after the mean update.\nThis is a local solution, not a global guarantee.")

    explanation = fig.text(0.075, 0.205, "", fontsize=10, color=INK, linespacing=1.5, va="top")
    schedule = [0, 0] + list(range(1, len(states))) + [len(states) - 1] * 4
    animation = FuncAnimation(fig, draw, frames=schedule, interval=650, repeat=True)
    animation.save(path, writer=PillowWriter(fps=1.5), dpi=90)
    draw(len(states) - 1)
    return fig, {"n": len(X), "k": 3, "initial": initial.tolist(), "states": len(states),
                 "iterations": states[-1].iteration, "initial_inertia": states[0].inertia,
                 "final_inertia": states[-1].inertia, "converged": bool(states[-1].converged)}


def scaling_scene(path, seed):
    X, truth, scaler, raw, standardized, shares = scaling_experiment(seed)
    fig = plt.figure(figsize=(13, 8.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.0], left=0.075, right=0.95,
                           top=0.82, bottom=0.19, hspace=0.65, wspace=0.24)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    lower, upper = limits(X)
    for ax, model, centers, title in zip(axes, (raw, standardized),
            (raw.cluster_centers_, scaler.inverse_transform(standardized.cluster_centers_)),
            ("Fit: raw units", "Fit: standardized space; centers inverse-transformed")):
        scatter(ax, X, model.labels_, centers)
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
               xlabel="processing_time (minutes)", ylabel="interaction_count (continuous proxy)")
        ax.set_title(title, fontsize=11, pad=12)
    bars = fig.add_subplot(grid[1, :])
    bars.barh([1, 0], shares[:, 0] * 100, color=COLORS[0], label="processing_time contribution")
    bars.barh([1, 0], shares[:, 1] * 100, left=shares[:, 0] * 100,
              color=COLORS[1], label="interaction_count contribution")
    bars.set(yticks=[1, 0], yticklabels=["Raw", "Scaled"], xlim=(0, 100),
             xlabel="Share of squared distance for a fixed difference: 60 minutes and 1 interaction (%)")
    bars.legend(loc="upper center", bbox_to_anchor=(0.5, -0.65), ncol=2, fontsize=9, frameon=False)
    page(fig, "02 / Preprocessing defines geometry", "Identical synthetic observations, K=3, n_init=10. Both panels use ORIGINAL coordinates.",
         "StandardScaler weights squared feature differences by inverse variance; the fitting objective changes.\n"
         "Scaling is a domain choice: it can also amplify noise. Numeric cluster IDs need not match across fits.")
    save_png(fig, path)
    return fig, {"n": len(X), "k": 3, "raw_inertia_raw_space": float(raw.inertia_),
                 "standardized_inertia_standardized_space": float(standardized.inertia_),
                 "raw_generator_ari": float(adjusted_rand_score(truth, raw.labels_)),
                 "scaled_generator_ari": float(adjusted_rand_score(truth, standardized.labels_)),
                 "squared_distance_shares": shares.tolist(), "scaler_mean": scaler.mean_.tolist(),
                 "scaler_scale": scaler.scale_.tolist()}


def elbow_scene(path, seed, k_min=2, k_max=10):
    X, rows = selection_experiment(seed, k_min, k_max)
    valid = [row for row in rows if row["silhouette"] is not None]
    candidate = max(valid, key=lambda row: row["silhouette"])["k"] if valid else None
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.8))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.8, bottom=0.28, wspace=0.38)
    ks = [row["k"] for row in rows]
    axes[0].plot(ks, [row["inertia"] for row in rows], "o-", color=COLORS[0], lw=2)
    axes[0].set(xlabel="Number of centroids K", ylabel="Inertia (squared synthetic units)", title="Compactness keeps improving")
    axes[1].plot(ks, [row["silhouette"] for row in rows], "o-", color=COLORS[2], lw=2)
    axes[1].set(xlabel="Number of centroids K", ylabel="Mean Euclidean silhouette", title="Cohesion AND separation")
    if candidate is not None:
        axes[1].axvline(candidate, color=COLORS[1], ls="--", label=f"Candidate K={candidate}\n(maximum in this sweep)")
        axes[1].legend(fontsize=9)
    selected = sorted(set([ks[0], ks[-1]] + ([candidate] if candidate is not None else [])))
    for position, k in enumerate(selected):
        row = next(row for row in rows if row["k"] == k)
        left = 0
        for j, count in enumerate(sorted(row["sizes"], reverse=True)):
            axes[2].barh(position, count, left=left, color=COLORS[j % len(COLORS)], edgecolor="white")
            left += count
    axes[2].set(yticks=range(len(selected)), yticklabels=[f"K={k}" for k in selected],
                xlabel="Observations (sizes sorted within each fit)", title="Inspect cluster sizes too")
    for ax in axes[:2]:
        ax.set_xticks(ks)
        ax.grid(axis="y")
    page(fig, "03 / Choose K with evidence", "360 synthetic points; fixed representation and fitting population across the entire sweep.",
         "More centers can reduce optimal inertia without discovering more useful groups. An elbow is a heuristic, not a guarantee.\n"
         "Silhouette: s = (b - a) / max(a, b); a = within-cluster mean distance; b = nearest other cluster mean distance.\n"
         "Internal scores need profile, stability and downstream review. A candidate is not the true number of real-world clusters.")
    save_png(fig, path)
    return fig, {"n": len(X), "k_min": k_min, "k_max": k_max, "candidate_by_silhouette": candidate, "sweep": rows}


def three_d_figure(seed):
    X, scaler, scaled, initial = three_dimensional_data(seed)
    states = [s for s in lloyd_trace(scaled, initial) if s.phase == "assignment"]
    labels = ["processing_time (h)", "interaction_count", "rework_score"]

    def traces(state):
        centers = scaler.inverse_transform(state.centers)
        data = []
        for j in range(3):
            members = X[state.labels == j]
            data.append(go.Scatter3d(x=members[:, 0].tolist(), y=members[:, 1].tolist(), z=members[:, 2].tolist(),
                mode="markers", name=f"Cluster {j}", legendgroup=str(j), uid=f"cluster-{j}",
                marker={"color": COLORS[j], "size": 4, "opacity": 0.8},
                hovertemplate=f"Cluster {j}<br>processing_time=%{{x:.2f}} h<br>interaction_count=%{{y:.2f}}<br>rework_score=%{{z:.2f}}<extra></extra>"))
            data.append(go.Scatter3d(x=[centers[j, 0]], y=[centers[j, 1]], z=[centers[j, 2]],
                mode="markers+text", text=[f"C{j}"], textposition="top center", name=f"Centroid {j}",
                legendgroup=str(j), uid=f"center-{j}",
                marker={"color": COLORS[j], "size": 10, "symbol": "diamond", "line": {"color": INK, "width": 2}},
                hovertemplate=f"Centroid {j}<br>processing_time=%{{x:.2f}} h<br>interaction_count=%{{y:.2f}}<br>rework_score=%{{z:.2f}}<extra></extra>"))
        return data

    def title(state):
        status = " | converged" if state.converged else ""
        return f"06 / K-Means in three dimensions<br><sup>Exact iteration {state.iteration}{status}; standardized-space inertia = {state.inertia:.3f}</sup>"

    frames = [go.Frame(name=str(s.iteration), data=traces(s), traces=list(range(6)),
                       layout=go.Layout(title={"text": title(s)})) for s in states]
    fig = go.Figure(data=traces(states[0]), frames=frames)
    lower, upper = limits(X, scaler.inverse_transform(initial))
    scene = {axis: {"title": labels[i], "range": [float(lower[i]), float(upper[i])],
                    "backgroundcolor": "white", "gridcolor": "#DFE5EE"}
             for i, axis in enumerate(("xaxis", "yaxis", "zaxis"))}
    scene.update({"aspectmode": "cube", "uirevision": "keep-camera", "dragmode": "orbit",
                  "camera": {"eye": {"x": 1.8, "y": 1.8, "z": 1.4}}})
    fig.update_layout(title={"text": title(states[0]), "x": 0.04}, template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 13, "color": INK}, paper_bgcolor=PAPER,
        height=860, margin={"l": 25, "r": 25, "t": 100, "b": 210}, scene=scene,
        legend={"title": "Arbitrary cluster IDs", "x": 1, "xanchor": "right", "y": 1},
        sliders=[{"active": 0, "currentvalue": {"prefix": "Exact mean updates: "}, "pad": {"t": 25},
                  "steps": [{"label": str(s.iteration), "method": "animate", "args": [[str(s.iteration)],
                             {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]} for s in states]}],
        annotations=[{"text": "Fit: StandardScaler space. Display: original feature units. Slider states are exact nearest-center assignments.<br>Drag to rotate; scroll to zoom; use the modebar to pan. No server, CDN or artificial third coordinate.",
                      "xref": "paper", "yref": "paper", "x": 0, "y": -0.27, "yanchor": "top", "showarrow": False, "align": "left", "font": {"size": 12}}])
    return fig, {"n": len(X), "dimensions": 3, "fit_space": "StandardScaler", "display_space": "original units",
                 "initial_centers_standardized": initial.tolist(), "slider_iterations": [s.iteration for s in states],
                 "inertias": [s.inertia for s in states], "converged": bool(states[-1].converged)}


def three_d_scene(path, seed):
    fig, result = three_d_figure(seed)
    fig.write_html(path, include_plotlyjs=True, full_html=True, auto_open=False, auto_play=False,
                   config={"responsive": True, "scrollZoom": True, "displaylogo": False},
                   div_id="kmeans-3d-lab")
    return None, result


def boundaries_scene(path, seed):
    X, _ = convergence_data(seed)
    model = fit_reference(X, 3, seed=seed)
    centers = model.cluster_centers_
    edited = centers.copy()
    edited[0] += [2, 1.5]
    lower, upper = limits(X, np.vstack((centers, edited)))
    xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], 260), np.linspace(lower[1], upper[1], 240))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    fig.subplots_adjust(left=0.075, right=0.96, top=0.81, bottom=0.21, wspace=0.22)
    assignments = []
    for ax, current, title in zip(axes, (centers, edited),
            ("Fitted nearest-center regions", "Controlled edit: move C0 by (+2, +1.5)")):
        regions = predict_labels(grid, current).reshape(xx.shape)
        labels = predict_labels(X, current)
        assignments.append(labels)
        ax.contourf(xx, yy, regions, levels=np.arange(4) - 0.5,
                    cmap=ListedColormap(COLORS[:3]), alpha=0.14)
        ax.contour(xx, yy, regions, levels=[0.5, 1.5], colors=INK, linewidths=0.8, alpha=0.65)
        scatter(ax, X, labels, current)
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
               xlabel="Feature 1 (synthetic units)", ylabel="Feature 2 (synthetic units)", title=title)
        ax.set_aspect("equal", adjustable="box")
    axes[1].annotate("", xy=edited[0], xytext=centers[0],
                     arrowprops={"arrowstyle": "->", "color": INK, "lw": 2})
    page(fig, "04 / Distance creates decision regions", "Dense-grid predictions reveal convex Euclidean Voronoi cells; both panels use the same axes.",
         "Between two centers, equal squared distances define a straight bisector. Other centers clip the resulting regions.\n"
         "The right panel changes one center deliberately; it is not a new optimized solution or an extra Lloyd iteration.")
    save_png(fig, path)
    return fig, {"n": len(X), "k": 3, "grid_shape": list(xx.shape), "center_0_shift": [2, 1.5],
                 "changed_assignments": int(np.sum(assignments[0] != assignments[1]))}


def limitations_scene(path, seed):
    cases = limitations_experiment(seed)
    fig, axes = plt.subplots(5, 2, figsize=(12, 16))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.89, bottom=0.14, hspace=0.48, wspace=0.19)
    rows = []
    for (left, right), case in zip(axes, cases):
        X, km, db = case["X"], case["kmeans"], case["dbscan"]
        lower, upper = limits(X)
        scatter(left, X, km.labels_, km.cluster_centers_, size=15)
        scatter(right, X, db.labels_, size=15)
        occupied = len(set(db.labels_) - {-1})
        left.set_title(f"{case['name']} | K-Means K={case['k']}", fontsize=11)
        right.set_title(f"DBSCAN: eps={case['eps']}, min_samples=6 | groups={occupied}", fontsize=11)
        for ax in (left, right):
            ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                   xlabel="Feature 1", ylabel="Feature 2")
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=8)
        left.text(0, -0.27, case["note"], transform=left.transAxes, fontsize=9, color=MUTED)
        rows.append({"case": case["name"], "n": len(X), "k": case["k"],
                     "kmeans_inertia": float(km.inertia_), "dbscan_eps": case["eps"],
                     "dbscan_min_samples": 6, "dbscan_groups": occupied,
                     "dbscan_noise": int(np.sum(db.labels_ == -1))})
    page(fig, "05 / Where the model can mislead", "Five synthetic controls. Grey DBSCAN points are noise; IDs and colors are local to each fit.",
         "K-Means uses compact nearest-mean regions; DBSCAN uses density connectivity and can reject points.\n"
         "DBSCAN radii are illustrative per-dataset choices, not a tuned comparison or a claim of universal superiority.\n"
         "Coordinate systems match within each row. Different datasets have different scales and populations.")
    save_png(fig, path)
    return fig, {"cases": rows}


def initialization_scene(path, seed):
    X, runs, budgets = initialization_experiment(seed)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.subplots_adjust(left=0.065, right=0.97, top=0.81, bottom=0.24, hspace=0.44, wspace=0.22)
    lower, upper = limits(X)
    rows = []
    for ax, run in zip(axes.flat, runs):
        model, initial = run["model"], run["initial"]
        scatter(ax, X, model.labels_, model.cluster_centers_, size=13)
        ax.scatter(initial[:, 0], initial[:, 1], marker="s", s=140, facecolors="none", edgecolors=INK,
                   linewidths=1.5, zorder=6)
        for old, new in zip(initial, model.cluster_centers_):
            ax.plot([old[0], new[0]], [old[1], new[1]], "--", color=INK, alpha=0.5, lw=1)
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), xlabel="Feature 1", ylabel="Feature 2")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{run['strategy']} | seed {run['seed']}\nFinal inertia = {model.inertia_:,.2f}", fontsize=11)
        rows.append({"strategy": run["strategy"], "seed": run["seed"], "n_init": 1,
                     "initial": initial.tolist(), "final_inertia": float(model.inertia_), "iterations": int(model.n_iter_)})
    page(fig, "07 / Initialization explores local solutions", "Four synthetic lobes, K=3, one run per panel. Initial centers are shown explicitly, not inferred.",
         "Hollow squares = initial centers; X = final means; dashed lines = net displacement (not an optimization trace).\n"
         f"Separate ten-restart fits: random J={budgets['random']:,.2f}; K-Means++ J={budgets['k-means++']:,.2f}.\n"
         "Multiple starts retain the smallest observed objective; neither initializer guarantees a global optimum.\n"
         "Cluster colors identify centers within each run; numeric IDs have no shared meaning across panels.")
    save_png(fig, path)
    return fig, {"n": len(X), "k": 3, "runs": rows, "ten_restart_inertias": budgets}


def write_gallery(output_dir):
    """A static local front door connecting the exports; no JS or server needed."""
    cards = []
    for scene, filename in OUTPUT_NAMES.items():
        title, observe, why, implication = SCENE_GUIDE[scene]
        href = html.escape(filename, quote=True)
        preview = (f'<a class="launch" href="{href}">Open the offline 3D explorer &rarr;</a>' if scene == "3d"
                   else f'<a href="{href}"><img src="{href}" alt="{html.escape(title)}" loading="lazy"></a>')
        cards.append(f'<section id="{scene}"><h2>{html.escape(title)}</h2>{preview}'
                     f'<p><b>What to observe:</b> {html.escape(observe)}</p>'
                     f'<p><b>Why it happens:</b> {html.escape(why)}</p>'
                     f'<p><b>Production implication:</b> {html.escape(implication)}</p></section>')
    nav = " ".join(f'<a href="#{scene}">{i:02d} {scene}</a>' for i, scene in enumerate(OUTPUT_NAMES, 1))
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Day 36 | K-Means visual lab</title><style>'
                'body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui,sans-serif;'
                'color:#16283f;background:#f5f7fb}h1{font-size:38px;line-height:1.2}h2{font-size:27px}'
                'nav{display:flex;flex-wrap:wrap;gap:12px}a{color:#0072b2}section{background:white;'
                'padding:28px;margin:28px 0;border-radius:12px;border:1px solid #dfe5ee}'
                'img{width:100%;height:auto}p{max-width:90ch}.launch{display:inline-block;padding:20px;'
                'background:#e4f1f9;border-radius:8px;font-size:23px;font-weight:600}</style></head><body>'
                '<h1>K-Means: from exact updates to modeling judgment</h1>'
                '<p>Day 36 / Seven connected scenes / Synthetic data / CPU only / Offline</p>'
                '<p>Follow the optimization, question the geometry, then challenge the result. '
                'These experiments support technical discussion; interpretations await author review.</p>'
                f'<nav>{nav}</nav>{"".join(cards)}'
                '<p>Artifacts are generated locally. No network, account, Python server or cloud service is needed.</p>'
                '</body></html>')
    path = output_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Export all seven scenes")
    group.add_argument("--scene", choices=OUTPUT_NAMES, help="One scene (default: convergence)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-min", type=int, default=2, help="Lower K for the elbow scene")
    parser.add_argument("--k-max", type=int, default=10, help="Upper K for the elbow scene")
    parser.add_argument("--no-show", action="store_true", help="Use Agg and do not open windows or a browser")
    parser.add_argument("--save", action="store_true", help="Explicit export request (export is also the default)")
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2 ** 32 - 2:
        parser.error("--seed must be between 0 and 2**32-3 (room for the two additional seeds)")
    if not 2 <= args.k_min <= args.k_max < 360:
        parser.error("Require 2 <= --k-min <= --k-max < 360")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.no_show:
        matplotlib.use("Agg")
    style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = list(OUTPUT_NAMES) if args.all else [args.scene or "convergence"]
    renderers = {"convergence": convergence_scene, "scaling": scaling_scene, "elbow": elbow_scene,
                 "boundaries": boundaries_scene, "limitations": limitations_scene,
                 "3d": three_d_scene, "initialization": initialization_scene}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
              "synthetic_data": True, "interpretations": "pending author review",
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                           "plotly": plotly.__version__, "pillow": PIL.__version__},
              "reference_fits": {"algorithm": "lloyd", "tol": 0, "max_iter": 300, "n_init": 10},
              "scenes": {}}
    with threadpool_limits(limits=1):
        for scene in scenes:
            path = args.output_dir / OUTPUT_NAMES[scene]
            kwargs = {"k_min": args.k_min, "k_max": args.k_max} if scene == "elbow" else {}
            fig, metrics = renderers[scene](path, args.seed, **kwargs)
            report["scenes"][scene] = {"file": path.name, "bytes": path.stat().st_size, **metrics}
            print(f"Exported {path.name} ({path.stat().st_size / 1024:.1f} KiB)", flush=True)
            if fig is not None:
                plt.close(fig)
    if args.all:
        gallery = write_gallery(args.output_dir)
        print(f"Local lab entry: {gallery}", flush=True)
    # Scene-specific reports prevent one-scene reruns from replacing the all-scene record.
    report_name = "visual_report_all.json" if args.all else f"visual_report_{scenes[0]}.json"
    (args.output_dir / report_name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.no_show:
        entry = args.output_dir / ("index.html" if args.all else OUTPUT_NAMES[scenes[0]])
        webbrowser.open(entry.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
