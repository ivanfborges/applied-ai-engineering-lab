"""Ten deterministic hierarchical-clustering scenes; offline exports, no server."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import html
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import PIL
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy
import sklearn
from scipy.cluster.hierarchy import cut_tree, leaves_list, linkage
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
INK, MUTED, PAPER, ACCENT = "#172B45", "#51647B", "#F3F6FA", "#D97706"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
          "#6854A3", "#56B4E9", "#264653", "#B5446E", "#6A7F23",
          "#427D92", "#A95C27", "#7182BD", "#2B8C7E", "#96589A",
          "#765A45", "#185B9A", "#9B7134", "#657085", "#BD4773"]
SCENES = {
    1: ("Bottom-up agglomeration", ("01_agglomerative_process.gif",)),
    2: ("Four definitions of a merge", ("02_linkage_methods.gif",)),
    3: ("A dendrogram built in merge order", ("03_dendrogram_construction.gif",)),
    4: ("One tree, several cuts", ("04_dendrogram_cuts.gif", "04_dendrogram_explorer.html")),
    5: ("The chaining effect", ("05_chaining_effect.gif",)),
    6: ("Feature units change geometry", ("06_feature_scaling.gif",)),
    7: ("Ward measures variance increase", ("07_ward_variance.gif",)),
    8: ("Genuine three-feature hierarchy", ("08_hierarchical_3d.html",)),
    9: ("Synthetic document-vector distances", ("09_embedding_distances.html",)),
    10: ("Quadratic pairwise storage", ("10_complexity.png",)),
}


def small_data():
    """Fifteen 2D observations; no inferred real-world categories."""
    X, _ = make_blobs(n_samples=15, centers=[[-2, -0.5], [0, 2], [2, -0.3]],
                      cluster_std=0.36, random_state=42)
    return X


def bridge_data():
    """Dense ends connected by intentionally short, slightly unequal links."""
    left = np.array([[-4.4, 0], [-4.2, .12], [-4, -.12], [-3.8, .02]])
    middle = np.column_stack([[-3, -2.3, -1.55, -.75, .1, .95, 1.85, 2.8],
                              [0, .04, -.03, .02, -.04, .01, .03, -.01]])
    right = np.array([[3.8, 0], [4, .13], [4.2, -.1], [4.4, .01]])
    return np.vstack([left, middle, right])


def node_members(Z):
    n = len(Z) + 1
    members = {i: [i] for i in range(n)}
    for step, (left, right, _, _) in enumerate(Z):
        members[n + step] = members[int(left)] + members[int(right)]
    return members


def tree_states(Z):
    """Exact active node IDs after each merge, including the singleton state."""
    n = len(Z) + 1
    active = set(range(n))
    states = [sorted(active)]
    for step, row in enumerate(Z):
        active.difference_update(map(int, row[:2]))
        active.add(n + step)
        states.append(sorted(active))
    return states


def merge_labels(Z, completed):
    if not isinstance(completed, (int, np.integer)) or not 0 <= completed <= len(Z):
        raise ValueError("completed must be an integer from 0 to n-1")
    members = node_members(Z)
    labels = np.empty(len(Z) + 1, dtype=int)
    for node in tree_states(Z)[completed]:
        labels[members[node]] = min(members[node])
    return labels


def exact_partition(Z, k):
    """Cut by merge order even at tied or zero heights; labels are anchor rows."""
    n = len(Z) + 1
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= n:
        raise ValueError("k must be an integer from 1 to n")
    return merge_labels(Z, n - int(k))


def interior_height(Z, k):
    if not 2 <= k < len(Z) + 1:
        raise ValueError("interior cuts require 2 <= k < n")
    n = len(Z) + 1
    low, high = Z[n-k-1, 2], Z[n-k, 2]
    mid = low + (high - low) / 2
    return float(mid) if low < mid < high else None


def tree_geometry(Z):
    """U-shaped branches indexed in actual merge order, using SciPy leaf order."""
    n = len(Z) + 1
    order = leaves_list(Z).astype(int).tolist()
    positions = {leaf: (float(position), 0.) for position, leaf in enumerate(order)}
    branches = []
    for step, (left, right, height, _) in enumerate(Z):
        xl, yl = positions[int(left)]
        xr, yr = positions[int(right)]
        branches.append(([xl, xl, xr, xr], [yl, height, height, yr]))
        positions[n + step] = ((xl + xr) / 2, float(height))
    return order, branches


def ward_delta(X, a, b):
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    return float(len(a) * len(b) / (len(a) + len(b)) *
                 np.square(X[a].mean(axis=0) - X[b].mean(axis=0)).sum())


def ward_control():
    # A small group can favor a farther small group over a nearer large one.
    X = np.vstack([[[0., -.1], [0., .1]],
                   np.array([[x, y] for x in (1.85, 1.95, 2.05, 2.15) for y in (-.08, .08)]),
                   [[-2.3, -.1], [-2.3, .1]]])
    Z = linkage(X, method="ward")
    members = node_members(Z)
    active = tree_states(Z)[len(X)-3]
    candidates = sorted((ward_delta(X, members[a], members[b]), a, b)
                        for a, b in combinations(active, 2))
    return X, Z, active, candidates


def scaling_control():
    raw = small_data() * [1., 100.]
    standardized = StandardScaler().fit_transform(raw)
    alphas = np.linspace(0, 1, 9)
    points = [(1-alpha)*raw + alpha*standardized for alpha in alphas]
    trees = [linkage(X, method="ward") for X in points]
    return raw, standardized, alphas, points, trees


def embedding_control():
    X = np.array([[1., .1], [8., .5], [.8, .7], [6., 6.], [-1., .5], [-5., 3.], [.1, 2.]])
    unit = X / np.linalg.norm(X, axis=1, keepdims=True)
    settings = [("Raw / Euclidean", X, "euclidean"), ("Raw / cosine", X, "cosine"),
                ("Unit / Euclidean", unit, "euclidean"), ("Unit / cosine", unit, "cosine"),
                ("Unit / squared Euclidean", unit, "sqeuclidean")]
    return [(name, values, metric, linkage(values, method="average", metric=metric),
             squareform(pdist(values, metric=metric))) for name, values, metric in settings]


def pair_storage(n):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer >= 2")
    pairs = int(n) * (int(n)-1) // 2
    return pairs, pairs * 8, int(n)**2 * 8


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": PAPER, "axes.facecolor": "white", "savefig.facecolor": PAPER,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#CCD6E0",
        "axes.spines.top": False, "axes.spines.right": False, "grid.alpha": .18})


def page(fig, title, subtitle, footer):
    fig.suptitle(title, x=.055, y=.98, ha="left", weight="bold", fontsize=19)
    fig.text(.055, .91, subtitle, fontsize=10, color=MUTED, va="top")
    fig.text(.055, .04, footer, fontsize=9, color=MUTED, va="bottom", linespacing=1.5)


def color(anchor):
    return COLORS[int(anchor) % len(COLORS)]


def scatter(ax, X, labels, *, annotate=True, equal=True):
    ax.scatter(*X.T, c=[color(label) for label in labels], s=88, edgecolors="white", linewidths=1.2, zorder=4)
    if annotate:
        for row, point in enumerate(X):
            ax.annotate(str(row), point, xytext=(5, 5), textcoords="offset points", fontsize=8, zorder=6)
    span = np.maximum(np.ptp(X, axis=0), .5)
    ax.set(xlim=(X[:, 0].min()-.18*span[0], X[:, 0].max()+.18*span[0]),
           ylim=(X[:, 1].min()-.24*span[1], X[:, 1].max()+.24*span[1]),
           xlabel="Feature 1", ylabel="Feature 2")
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.grid(True)


def dendrogram_axis(ax, Z, *, completed=None, labels=None, threshold=None):
    order, branches = tree_geometry(Z)
    completed = len(Z) if completed is None else completed
    for step, (xs, ys) in enumerate(branches[:completed]):
        ax.plot(xs, ys, color=ACCENT if step == completed-1 else "#62768C", lw=2)
    ax.set(xticks=range(len(order)), xticklabels=order,
           xlim=(-.7, len(order)-.3), ylim=(-.035*max(Z[-1, 2], 1), max(Z[-1, 2], 1)*1.12),
           xlabel="Original observation IDs (rotations preserve the tree)", ylabel="Linkage height")
    if labels is not None:
        for tick, leaf in zip(ax.get_xticklabels(), order):
            tick.set_color(color(labels[leaf]))
    if threshold is not None:
        ax.axhline(threshold, color="#BE324A", ls="--", lw=1.8, label=f"Cut t={threshold:.3f}")
        ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y")


def merge_schedule(n):
    return [(0, "start")] + [(step, phase) for step in range(n-1) for phase in ("select", "merge")]


def highlight_merge(ax, X, Z, step, method):
    left, right = map(int, Z[step, :2])
    members = node_members(Z)
    a, b = members[left], members[right]
    for ids, edge in ((a, ACCENT), (b, "#B73766")):
        ax.scatter(*X[ids].T, s=220, facecolors="none", edgecolors=edge, linewidths=2.3, zorder=5)
    distances = cdist(X[a], X[b])
    if method in ("single", "complete"):
        index = np.argmin(distances) if method == "single" else np.argmax(distances)
        i, j = np.unravel_index(index, distances.shape)
        points = X[[a[i], b[j]]]
        ax.plot(*points.T, color=ACCENT, lw=2.5, zorder=3)
    elif method == "average":
        for i in a:
            for j in b:
                ax.plot(*X[[i, j]].T, color=ACCENT, lw=.8, alpha=.28, zorder=2)
    else:
        centers = np.array([X[a].mean(axis=0), X[b].mean(axis=0)])
        ax.plot(*centers.T, "D--", color=ACCENT, lw=2, markersize=8, zorder=7)
    text = f"nodes {left} + {right} | height {Z[step, 2]:.3f}"
    if method == "ward":
        text += f"\nΔSSE {ward_delta(X, a, b):.3f}; height = sqrt(2 ΔSSE)"
    else:
        text += f"\n{len(a)*len(b)} cross-pairs; {method} distance"
    return text


def save_gif(fig, draw, frames, path, *, fast=False, force=False, fps=1.5):
    try:
        if path.exists() and not force:
            return
        animation = FuncAnimation(fig, draw, frames=frames, repeat=True, blit=False, cache_frame_data=False)
        animation.save(path, writer=PillowWriter(fps=fps*(1.6 if fast else 1)), dpi=65 if fast else 100)
    finally:
        plt.close(fig)


def save_html(fig, path, *, force=False, intro=""):
    if path.exists() and not force:
        return
    body = fig.to_html(full_html=False, include_plotlyjs=True, div_id=f"hierarchical-{path.stem}",
                       config={"responsive": True, "displaylogo": False, "scrollZoom": True})
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Hierarchical clustering visual lab</title><style>'
                'body{font:16px system-ui;color:#172b45;background:#f3f6fa;margin:0;padding:18px}'
                'p{max-width:1100px;line-height:1.55;margin:12px auto}.plotly-graph-div{margin:auto}'
                '</style></head><body><p>' + html.escape(intro) + '</p>' + body +
                '<p>Synthetic teaching data. Group IDs are arbitrary; geometric structure does not '
                'establish semantic or business validity. Interpretation candidates require author review.</p></body></html>')
    path.write_text(document, encoding="utf-8")


def render_process(output, *, fast=False, force=False):
    X = small_data()
    Z = linkage(X, method="average")
    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.subplots_adjust(left=.08, right=.96, top=.73, bottom=.22)
    page(fig, "01 / Watch the groups become a hierarchy", "15 synthetic observations • average linkage • Euclidean distance • every merge is shown",
         "Numbers identify observations; node IDs identify groups. A merged group keeps its lowest-row member's color.\nAmber and rose rings select the next pair. Earlier merges are irreversible; labels are not semantic categories.")

    def draw(frame):
        step, phase = frame
        ax.clear()
        completed = 0 if phase == "start" else step + (phase == "merge")
        scatter(ax, X, merge_labels(Z, completed))
        if phase == "start":
            ax.set_title("Start: 15 singleton clusters • zero merges", loc="left")
        else:
            detail = highlight_merge(ax, X, Z, step, "average")
            ax.set_title(f"Step {step+1}/14: {'select' if phase == 'select' else 'merge'} {detail}\n"
                         f"Remaining clusters: {len(X)-completed}", loc="left", fontsize=11)
    save_gif(fig, draw, merge_schedule(len(X)), output / SCENES[1][1][0], fast=fast, force=force)
    return {"n": len(X), "method": "average", "heights": Z[:, 2].tolist(), "merges_shown": len(Z)}


def render_linkages(output, *, fast=False, force=False):
    X = bridge_data()[[0, 1, 3, 4, 6, 8, 10, 11, 12, 14, 15]]
    methods = ("single", "complete", "average", "ward")
    trees = {method: linkage(X, method=method) for method in methods}
    criteria = {"single": "Minimum cross-pair distance", "complete": "Maximum cross-pair distance",
                "average": "Mean of ALL cross-pair distances", "ward": "Minimum increase in within-group SSE"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.subplots_adjust(left=.06, right=.97, top=.77, bottom=.18, hspace=.65, wspace=.23)
    page(fig, "02 / Same coordinates, four definitions of closeness", "Synchronized merge count, not synchronized distance • lines show the selected criterion • 11 synthetic points",
         r"Ward: $\Delta SSE=\frac{n_A n_B}{n_A+n_B}\|\mu_A-\mu_B\|^2$; height $=\sqrt{2\Delta SSE}$, not raw SSE."
         "\nWard needs Euclidean geometry. Average shows every contributing cross-pair; ringed groups are the selected candidates.")

    def draw(frame):
        step, phase = frame
        selected = []
        for method in methods:
            members = node_members(trees[method])
            selected.append(tuple(sorted(tuple(sorted(members[int(node)]))
                                         for node in trees[method][step, :2])))
        for ax, method in zip(axes.flat, methods):
            ax.clear()
            completed = 0 if phase == "start" else step + (phase == "merge")
            Z = trees[method]
            scatter(ax, X, merge_labels(Z, completed), annotate=False)
            ax.set_ylim(-2.1, 2.1)
            detail = "Start: one group per point"
            if phase != "start":
                detail = highlight_merge(ax, X, Z, step, method)
            ax.set_title(f"{method.upper()} / {criteria[method]}\n{detail}", loc="left", fontsize=9)
            ax.text(.02, .04, f"{len(X)-completed} groups | " +
                    ("different member pairs" if len(set(selected)) > 1 and phase != "start" else "same step count"),
                    transform=ax.transAxes, fontsize=8, bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
    save_gif(fig, draw, merge_schedule(len(X)), output / SCENES[2][1][0], fast=fast, force=force)
    return {"n": len(X), "trees": {m: trees[m].tolist() for m in methods}, "merges_shown_per_method": len(X)-1}


def render_construction(output, *, fast=False, force=False):
    X = small_data()[:12]
    Z = linkage(X, method="average")
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(left=.06, right=.97, top=.72, bottom=.23, wspace=.25)
    page(fig, "03 / Each merge creates exactly one branch", "One average-linkage tree • scatter and dendrogram share the same SciPy merge rows",
         "Leaves are original observations. A branch joins its two child groups at their actual linkage distance.\nEarly branches stay inside every later union. Left/right rotations change leaf order, not the hierarchy.")

    def draw(frame):
        step, phase = frame
        completed = 0 if phase == "start" else step + (phase == "merge")
        labels = merge_labels(Z, completed)
        for ax in axes:
            ax.clear()
        scatter(axes[0], X, labels)
        dendrogram_axis(axes[1], Z, completed=completed, labels=labels)
        if phase == "start":
            axes[0].set_title("12 singleton clusters", loc="left")
            axes[1].set_title("Leaves first; no branches yet", loc="left")
        else:
            highlight_merge(axes[0], X, Z, step, "average")
            axes[0].set_title(f"Step {step+1}: {phase} nodes " + " + ".join(str(int(v)) for v in Z[step, :2]), loc="left", fontsize=11)
            axes[1].set_title(f"{'Next' if phase == 'select' else 'New'} branch: height {Z[step, 2]:.3f}\n"
                              f"{completed} branches / {len(X)-completed} groups", loc="left", fontsize=11)
    save_gif(fig, draw, merge_schedule(len(X)), output / SCENES[3][1][0], fast=fast, force=force)
    return {"n": len(X), "Z": Z.tolist(), "merges_shown": len(Z)}


def cut_controls():
    X = small_data()
    Z = linkage(X, method="ward")
    settings = [(f"Blobs K={k}", X, Z, k) for k in (2, 3, 4, 5)]
    tiny = np.column_stack([[1., 2., 8., 9.], np.zeros(4)])
    tied = linkage(tiny, method="average")
    settings += [("Tie control K=2", tiny, tied, 2), ("Tie control K=3", tiny, tied, 3)]
    return settings


def cut_description(Z, k):
    threshold = interior_height(Z, k)
    return (f"K={k} | merge-order cut only: tied heights, no horizontal separator"
            if threshold is None else f"K={k} | interior height t={threshold:.3f}")


def add_tree_trace(fig, Z, *, row=1, col=1):
    order, branches = tree_geometry(Z)
    xs, ys = [], []
    for x, y in branches:
        xs.extend(x + [None])
        ys.extend(y + [None])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#62768C", width=2),
                             showlegend=False, hoverinfo="skip"), row=row, col=col)
    fig.update_xaxes(tickvals=list(range(len(order))), ticktext=[str(v) for v in order],
                     title_text="Observation IDs", row=row, col=col)
    fig.update_yaxes(title_text="Linkage height", row=row, col=col)


def explorer_figure():
    settings = cut_controls()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Full hierarchy", "Partition from this tree"))
    groups, metadata = [], []
    for name, X, Z, k in settings:
        start = len(fig.data)
        labels = exact_partition(Z, k)
        threshold = interior_height(Z, k)
        add_tree_trace(fig, Z)
        fig.add_trace(go.Scatter(x=[-.5, len(X)-.5] if threshold is not None else [],
            y=[threshold, threshold] if threshold is not None else [], mode="lines", name="Horizontal cut",
            line=dict(color="#BE324A", dash="dash"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=X[:, 0].tolist(), y=X[:, 1].tolist(), mode="markers+text",
            text=[str(i) for i in range(len(X))], textposition="top right", showlegend=False,
            marker=dict(size=13, color=[color(v) for v in labels], line=dict(color="white", width=1)),
            hovertemplate="row %{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"), row=1, col=2)
        groups.append(list(range(start, len(fig.data))))
        metadata.append(dict(name=name, k=k, labels=labels.tolist(), threshold=threshold, Z=Z.tolist()))
    steps = []
    for group, (name, X, Z, k) in zip(groups, settings):
        order, _ = tree_geometry(Z)
        visible = [i in group for i in range(len(fig.data))]
        layout = {"title.text": "04 / " + name + "<br><sup>" + cut_description(Z, k) + "</sup>",
                  "xaxis.tickvals": list(range(len(order))), "xaxis.ticktext": [str(v) for v in order],
                  "xaxis.range": [-.7, len(X)-.3], "yaxis.range": [-.1, max(Z[-1, 2], 1)*1.1]}
        steps.append(dict(method="update", label=name, args=[{"visible": visible}, layout]))
    for i, trace in enumerate(fig.data):
        trace.visible = i in groups[0]
    order, _ = tree_geometry(settings[0][2])
    fig.update_xaxes(tickvals=list(range(len(order))), ticktext=[str(v) for v in order], row=1, col=1)
    fig.update_layout(title=dict(text="04 / Blobs K=2<br><sup>"+cut_description(settings[0][2], 2)+"</sup>"),
        template="plotly_white", height=720, margin=dict(t=115, b=140),
        sliders=[dict(active=0, currentvalue=dict(prefix="Choose granularity: "), steps=steps)],
        meta=dict(states=metadata, fit="One Ward tree for blobs; one average tree for the explicit tie control"))
    fig.update_xaxes(title_text="Feature 1", row=1, col=2)
    fig.update_yaxes(title_text="Feature 2", scaleanchor="x2", scaleratio=1, row=1, col=2)
    return fig


def render_cuts(output, *, fast=False, force=False):
    settings = cut_controls()
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    fig.subplots_adjust(left=.06, right=.97, top=.73, bottom=.23, wspace=.25)
    page(fig, "04 / A cut chooses granularity, not truth", "Choose K on one fitted tree • the final two states isolate a separate tied-height control",
         "A horizontal cut joins branches below its height. At ties, several merges occur at the same height.\nAn exact-K merge-order partition may exist without a horizontal separator. There is no claimed optimal K.")

    def draw(index):
        name, X, Z, k = settings[index]
        labels = exact_partition(Z, k)
        for ax in axes:
            ax.clear()
        dendrogram_axis(axes[0], Z, labels=labels, threshold=interior_height(Z, k))
        scatter(axes[1], X, labels)
        if index >= 4:
            axes[1].set_ylim(-1.5, 1.5)
        axes[0].set_title(name + "\n" + ("Average linkage tie control" if index >= 4 else "Same Ward tree"), loc="left", fontsize=11)
        axes[1].set_title(cut_description(Z, k).replace(" | ", "\n"), loc="left", fontsize=10)
    save_gif(fig, draw, list(range(6)), output / SCENES[4][1][0], fast=fast, force=force, fps=.65)
    save_html(explorer_figure(), output / SCENES[4][1][1], force=force,
              intro="Move the slider to change K on the same fitted tree. Two explicit tie controls show why merge-order cuts and horizontal thresholds differ. Hover or zoom in both 2D panels.")
    return {"cuts": [{"name": name, "k": k, "threshold": interior_height(Z, k),
                       "sizes": sorted(np.unique(exact_partition(Z, k), return_counts=True)[1].tolist())}
                      for name, _, Z, k in settings]}


def render_chaining(output, *, fast=False, force=False):
    X = bridge_data()
    methods = ("single", "complete", "average")
    trees = {m: linkage(X, method=m) for m in methods}
    fig, axes = plt.subplots(1, 3, figsize=(14, 7))
    fig.subplots_adjust(left=.045, right=.98, top=.71, bottom=.24, wspace=.18)
    page(fig, "05 / Short local links can connect distant groups", "Intentionally constructed dense ends + a sparse bridge • same 16 points and units • every merge",
         "Single linkage needs only one short cross-pair. A chain can span both ends without a long local link.\nComplete and average linkage account for other cross-pairs. This is a grouping assumption, not a universal failure.")

    def draw(frame):
        step, phase = frame
        for ax, method in zip(axes, methods):
            ax.clear()
            Z = trees[method]
            completed = 0 if phase == "start" else step + (phase == "merge")
            scatter(ax, X, merge_labels(Z, completed), annotate=False)
            ax.set_ylim(-2.1, 2.1)
            if phase != "start":
                highlight_merge(ax, X, Z, step, method)
            ax.set_title(f"{method.upper()} / {len(X)-completed} groups\n" +
                         ("Singletons" if phase == "start" else f"{phase} {step+1}/15; h={Z[step,2]:.3f}"),
                         loc="left", fontsize=10)
    save_gif(fig, draw, merge_schedule(len(X)), output / SCENES[5][1][0], fast=fast, force=force)
    return {"n": len(X), "final_heights": {m: float(trees[m][-1, 2]) for m in methods},
            "groups_at_threshold_1_1": {m: int((trees[m][:, 2] > 1.1).sum()+1) for m in methods}}


def render_scaling(output, *, fast=False, force=False):
    raw, standard, alphas, points, trees = scaling_control()
    baseline = exact_partition(trees[0], 3)
    pairs = [(0, 1), (0, 7), (3, 12)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    fig.subplots_adjust(left=.07, right=.97, top=.72, bottom=.25, wspace=.25)
    page(fig, "06 / Changing units changes the merge problem", "Same 15 row IDs • feature 2 multiplied by 100 • raw → standardized via a linear coordinate path",
         "Plots fit their axes independently; read the printed units, not screen lengths. Both panels cut at K=3.\nWard is refitted for each changing geometry. Standardization changes the question; it is not always appropriate.")

    def draw(index):
        for ax in axes:
            ax.clear()
        current = points[index]
        scatter(axes[0], raw, baseline, equal=False)
        scatter(axes[1], current, exact_partition(trees[index], 3), equal=False)
        axes[0].set_title("Raw coordinates (fixed)\nFeature 2 uses 100× the source units", loc="left", fontsize=11)
        axes[1].set_title(f"Coordinate path α={alphas[index]:.3f}\n" +
                          ("Standardized endpoint (z-scores)" if index == len(alphas)-1 else "Interpolated units; hierarchy recomputed"), loc="left", fontsize=11)
        for ax, values in zip(axes, (raw, current)):
            text = "  |  ".join(f"d({a},{b})={np.linalg.norm(values[a]-values[b]):.2f}" for a, b in pairs)
            ax.text(0, -.2, text, transform=ax.transAxes, fontsize=8, color=MUTED)
    save_gif(fig, draw, list(range(len(alphas))), output / SCENES[6][1][0], fast=fast, force=force, fps=.85)
    return {"n": len(raw), "alpha": alphas.tolist(), "metric": "euclidean", "linkage": "ward",
            "endpoint_partition_ARI": float(adjusted_rand_score(baseline, exact_partition(trees[-1], 3))),
            "pair_distances": [{"pair": list(pair), "raw": float(np.linalg.norm(raw[pair[0]]-raw[pair[1]])),
                                "standardized": float(np.linalg.norm(standard[pair[0]]-standard[pair[1]]))} for pair in pairs]}


def render_ward(output, *, fast=False, force=False):
    X, Z, active, candidates = ward_control()
    members = node_members(Z)
    step = len(X)-3
    winner, left, right = candidates[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.06, right=.97, top=.72, bottom=.23, wspace=.2)
    page(fig, "07 / Ward is not just the nearest pair of centroids", "12 synthetic points • the three groups are actual active nodes of the fitted Ward tree",
         r"$\Delta SSE(A,B)=\frac{n_A n_B}{n_A+n_B}\|\mu_A-\mu_B\|^2$     |     SciPy height $h=\sqrt{2\Delta SSE}$"
         "\nSquared deviations sum to SSE. Cluster sizes matter. This is a local greedy decision, not a globally optimal partition.")

    def draw(stage):
        for ax in axes:
            ax.clear()
        completed = step + (stage >= 4)
        labels = merge_labels(Z, completed)
        scatter(axes[0], X, labels, annotate=False)
        axes[0].set_ylim(-1.4, 1.4)
        nodes = tree_states(Z)[completed]
        totals = []
        for node in nodes:
            ids = members[node]
            center = X[ids].mean(axis=0)
            for point in X[ids]:
                axes[0].plot(*np.array([center, point]).T, color=color(min(ids)), alpha=.85, lw=1.8)
            axes[0].scatter(*center, marker="D", s=100, c=color(min(ids)), edgecolors=INK, zorder=7)
            sse = float(np.square(X[ids]-center).sum())
            totals.append(sse)
            axes[0].annotate(f"node {node}: n={len(ids)}\nSSE={sse:.3f}", center,
                             xytext=(3, 15), textcoords="offset points", fontsize=8)
        headings = ["Three preformed groups: deviations → SSE", "Inspect candidate 1", "Inspect candidate 2",
                    "Compare all candidates; choose minimum ΔSSE", "Apply the selected merge; update the centroid", "Read the same merge on Ward's height scale"]
        axes[0].set_title(headings[stage]+f"\nTotal SSE now = {sum(totals):.3f}", loc="left", fontsize=11)
        if stage in (1, 2, 3):
            shown = candidates[:2] if stage == 3 else [candidates[stage-1]]
            for delta, a, b in shown:
                centers = np.array([X[members[a]].mean(axis=0), X[members[b]].mean(axis=0)])
                axes[0].plot(*centers.T, "--", color=ACCENT if delta == winner else MUTED, lw=2)
        axes[1].axis("off")
        lines = ["ACTUAL CANDIDATE MERGES", ""]
        for delta, a, b in candidates:
            distance = np.linalg.norm(X[members[a]].mean(axis=0)-X[members[b]].mean(axis=0))
            lines += [f"Nodes {a} + {b}: n={len(members[a])} and {len(members[b])}",
                      f"centroid distance={distance:.3f}; ΔSSE={delta:.3f}", ""]
        lines += [f"Selected nodes {left} + {right}", f"Smallest ΔSSE = {winner:.3f}",
                  f"SciPy Ward height = {Z[step, 2]:.3f}", f"h² / 2 = {Z[step,2]**2/2:.3f}", "",
                  "A smaller centroid distance alone", "does not determine the Ward merge."]
        axes[1].text(.05, .98, "\n".join(lines), transform=axes[1].transAxes, va="top", fontsize=11, linespacing=1.45)
    save_gif(fig, draw, list(range(6)), output / SCENES[7][1][0], fast=fast, force=force, fps=.6)
    return {"n": len(X), "preformed_counts": sorted(len(members[a]) for a in active),
            "candidate_delta_SSE": [float(v[0]) for v in candidates], "selected_nodes": [left, right],
            "selected_height": float(Z[step, 2]), "selected_delta_SSE": winner}


def three_dimensional_figure():
    X, _ = make_blobs(n_samples=48, centers=[[0, 0, 0], [3, 0, 1], [0, 3, 2], [3, 3, -1]],
                      cluster_std=.38, random_state=42)
    Z = linkage(X, method="ward")
    fig = go.Figure()
    groups, states = [], []
    for k in (2, 3, 4, 5, 6):
        labels = exact_partition(Z, k)
        start = len(fig.data)
        for label in np.unique(labels):
            ids = np.flatnonzero(labels == label)
            fig.add_trace(go.Scatter3d(x=X[ids, 0].tolist(), y=X[ids, 1].tolist(), z=X[ids, 2].tolist(),
                mode="markers", marker=dict(size=5, color=color(label)), name=f"Group {label} (n={len(ids)})",
                text=[f"row {i}" for i in ids], hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra>%{fullData.name}</extra>"))
        groups.append(list(range(start, len(fig.data))))
        states.append(dict(k=k, labels=labels.tolist()))
    steps = []
    for group, state in zip(groups, states):
        k = state["k"]
        steps.append(dict(method="update", label=str(k), args=[{"visible": [i in group for i in range(len(fig.data))]},
                     {"title.text": f"08 / Same Ward hierarchy: K={k}<br><sup>48 observations, three original features. "
                                    "Rotate, zoom or pan; granularity does not validate categories.</sup>"}]))
    for i, trace in enumerate(fig.data):
        trace.visible = i in groups[0]
    fig.update_layout(template="plotly_white", height=790, title=dict(text="08 / Genuine three-feature hierarchy: K=2"),
        margin=dict(t=115, b=110), sliders=[dict(currentvalue=dict(prefix="Clusters K="), steps=steps)],
        scene=dict(xaxis_title="Original feature 1", yaxis_title="Original feature 2", zaxis_title="Original feature 3",
                   xaxis=dict(range=[float(X[:,0].min()-.5), float(X[:,0].max()+.5)]),
                   yaxis=dict(range=[float(X[:,1].min()-.5), float(X[:,1].max()+.5)]),
                   zaxis=dict(range=[float(X[:,2].min()-.5), float(X[:,2].max()+.5)]),
                   aspectmode="data", uirevision="fixed-geometry"), uirevision="same-tree",
        meta=dict(fit_dimensions=3, X=X.tolist(), Z=Z.tolist(), states=states))
    return fig


def render_three_dimensional(output, *, fast=False, force=False):
    fig = three_dimensional_figure()
    save_html(fig, output / SCENES[8][1][0], force=force,
              intro="Drag to rotate the original three-feature space, scroll to zoom, and use pan in the toolbar. The K slider cuts one fitted Ward tree; camera position is preserved. No projection or dimensionality reduction is used.")
    return {"n": 48, "features": 3, "k_values": [2, 3, 4, 5, 6], "seed": 42}


def embedding_figure():
    settings = embedding_control()
    fig = make_subplots(rows=1, cols=3, column_widths=[.34, .35, .31],
                        subplot_titles=("Vectors: direction and length", "Pairwise dissimilarities", "Average-linkage hierarchy"))
    groups, states = [], []
    for name, X, metric, Z, distances in settings:
        start = len(fig.data)
        labels = exact_partition(Z, 3)
        line_x, line_y = [], []
        for point in X:
            line_x += [0., float(point[0]), None]
            line_y += [0., float(point[1]), None]
        fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", line=dict(color="#96A5B6"),
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=X[:,0].tolist(), y=X[:,1].tolist(), mode="markers+text", showlegend=False,
            text=[f"D{i}" for i in range(len(X))],
            textposition=["top right", "bottom right", "bottom right", "top left", "bottom left", "top left", "top right"],
            customdata=np.linalg.norm(X, axis=1).tolist(), marker=dict(size=12, color=[color(v) for v in labels]),
            hovertemplate="%{text}<br>norm=%{customdata:.3f}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Heatmap(z=distances.tolist(), x=[f"D{i}" for i in range(len(X))],
            y=[f"D{i}" for i in range(len(X))], colorscale="Blues", showscale=False,
            hovertemplate="%{x} / %{y}<br>distance=%{z:.4f}<extra></extra>"), row=1, col=2)
        add_tree_trace(fig, Z, row=1, col=3)
        groups.append(list(range(start, len(fig.data))))
        states.append(dict(name=name, metric=metric, X=X.tolist(), Z=Z.tolist(), labels=labels.tolist(), distances=distances.tolist()))
    steps = []
    for group, (name, X, _, Z, _) in zip(groups, settings):
        order, _ = tree_geometry(Z)
        steps.append(dict(method="update", label=name, args=[{"visible": [i in group for i in range(len(fig.data))]},
            {"title.text": "09 / Synthetic document vectors: " + name + "<br><sup>Average linkage, K=3. "
                           "Angles and lengths answer different similarity questions.</sup>",
             "xaxis3.tickvals": list(range(len(X))), "xaxis3.ticktext": [str(v) for v in order]}]))
    for i, trace in enumerate(fig.data):
        trace.visible = i in groups[0]
    order, _ = tree_geometry(settings[0][3])
    fig.update_xaxes(tickvals=list(range(len(order))), ticktext=[str(v) for v in order], row=1, col=3)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title_text="Vector coordinate 2", row=1, col=1)
    fig.update_xaxes(title_text="Vector coordinate 1", row=1, col=1)
    fig.update_layout(template="plotly_white", height=800, margin=dict(t=175, b=145),
        title=dict(text="09 / Hypothetical document vectors: raw Euclidean", y=.98, yanchor="top"),
        updatemenus=[dict(type="dropdown", x=.0, y=1.12, xanchor="left", buttons=steps)],
        meta=dict(synthetic=True, real_embedding_model=False, linkage="average", states=states))
    fig.add_annotation(x=0, y=-.28, xref="paper", yref="paper", xanchor="left", showarrow=False, align="left",
        text="Unit vectors: ||x-y||² = 2(1-cos(x,y)). Means of square roots differ from square roots of means.<br>"
             "Cosine and squared Euclidean are proportional here; ordinary Euclidean need not give the same average-linkage order.")
    return fig


def render_embeddings(output, *, fast=False, force=False):
    fig = embedding_figure()
    save_html(fig, output / SCENES[9][1][0], force=force,
        intro="Seven small, nonzero synthetic 2D vectors stand in for hypothetical documents; they are not outputs of an embedding model. Compare metric and normalization in the dropdown. This could motivate human-reviewed document-taxonomy exploration in RAG, but establishes no retrieval benefit. Heatmaps rescale with each metric; read the hover values.")
    settings = embedding_control()
    return {"vectors": 7, "features": 2, "settings": [s[0] for s in settings],
            "raw_euclidean_vs_cosine_K3_ARI": float(adjusted_rand_score(exact_partition(settings[0][3],3), exact_partition(settings[1][3],3))),
            "unit_cosine_vs_squared_euclidean_heights_error": float(np.max(np.abs(2*settings[3][3][:,2]-settings[4][3][:,2])))}


def render_complexity(output, *, fast=False, force=False):
    sizes = [100, 1000, 10000, 50000, 100000, 1000000]
    storage = [pair_storage(n) for n in sizes]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(left=.07, right=.97, top=.77, bottom=.22, wspace=.23)
    page(fig, "10 / Pairwise relationships grow quadratically", r"Exact arithmetic: $N_{pairs}=n(n-1)/2$ • float64 = 8 bytes • no distances allocated and no timings measured",
         "Condensed storage retains one value per unordered pair. A full square matrix also stores the mirrored entries and diagonal.\nThese are theoretical raw-storage calculations, excluding features and workspace; sparse/approximate methods change the problem.")
    axes[0].loglog(sizes, [row[0] for row in storage], "o-", lw=2, color=COLORS[0])
    axes[0].set(xlabel="Observations n", ylabel="Unique point pairs", title="Double n → approximately four times as many pairs")
    axes[0].grid(True, which="both")
    axes[1].axis("off")
    table = axes[1].table(cellText=[[f"{n:,}", f"{row[0]:,}", f"{row[1]/2**30:,.3f}", f"{row[2]/2**30:,.3f}"]
                                   for n, row in zip(sizes, storage)],
                         colLabels=["n", "Unique pairs", "Condensed GiB", "Square GiB"], loc="center", cellLoc="right",
                         colWidths=[.19, .36, .23, .22])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.4)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D4DEE9")
        cell.set_facecolor("#E3EDF7" if row == 0 else "white")
    path = output / SCENES[10][1][0]
    try:
        if force or not path.exists():
            fig.savefig(path, dpi=80 if fast else 120)
    finally:
        plt.close(fig)
    return {"n": sizes, "unique_pairs": [row[0] for row in storage],
            "condensed_bytes": [row[1] for row in storage], "square_bytes": [row[2] for row in storage],
            "measured_runtime": False}


RENDERERS = {1: render_process, 2: render_linkages, 3: render_construction, 4: render_cuts,
             5: render_chaining, 6: render_scaling, 7: render_ward, 8: render_three_dimensional,
             9: render_embeddings, 10: render_complexity}


def write_gallery(output):
    cards = []
    for number, (title, names) in SCENES.items():
        available = [name for name in names if (output/name).is_file()]
        if not available:
            continue
        media = next((name for name in available if name.endswith(("gif", "png"))), None)
        picture = f'<img loading="lazy" src="{media}" alt="{html.escape(title)}">' if media else ""
        links = " · ".join(f'<a href="{name}">{name}</a>' for name in available)
        cards.append(f'<article><h2>{number:02d} / {html.escape(title)}</h2>{picture}<p>{links}</p></article>')
    content = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
               'content="width=device-width,initial-scale=1"><title>Hierarchical clustering lab</title>'
               '<style>body{font:17px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;background:#f3f6fa;'
               'color:#172b45}article{padding:24px;margin:24px 0;background:white;border:1px solid #d4dee9;'
               'border-radius:12px}img{width:100%;height:auto}p{line-height:1.55}a{color:#0072b2}</style>'
               '<h1>Hierarchical clustering: inspect every merge</h1><p>Ten synthetic teaching scenes. '
               'All HTML explorers work offline. Interpretation candidates require author review.</p>' +
               "".join(cards) + '</html>')
    (output/"index.html").write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Generate all ten scenes; skip complete existing outputs")
    group.add_argument("--scene", type=int, choices=sorted(SCENES), help="Generate one independently executable scene")
    group.add_argument("--list", action="store_true", help="List concepts and artifact filenames")
    parser.add_argument("--fast", action="store_true", help="Lower GIF/PNG resolution and speed playback; preserve every mathematical state")
    parser.add_argument("--force", action="store_true", help="Explicitly regenerate existing outputs (including after code changes)")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs", help="Destination, relative to working directory if not absolute")
    args = parser.parse_args(argv)
    if args.list:
        for number, (title, names) in SCENES.items():
            print(f"{number:02d}: {title}: {', '.join(names)}")
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    style()
    now = datetime.now(timezone.utc)
    report = dict(generated_at_utc=now.isoformat(), fast=args.fast,
        interpretation_status="author review required", synthetic=True,
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                      matplotlib=matplotlib.__version__, pillow=PIL.__version__, plotly=plotly.__version__), scenes={})
    with threadpool_limits(limits=1):
        for number in (sorted(SCENES) if args.all else [args.scene]):
            title, names = SCENES[number]
            if not args.force and all((output/name).is_file() for name in names):
                result = {"status": "skipped_existing"}
                print(f"Skipped {number:02d}: existing complete scene (use --force to replace)", flush=True)
            else:
                result = RENDERERS[number](output, fast=args.fast, force=args.force)
                result["status"] = "generated"
                print(f"Generated {number:02d}: {', '.join(names)}", flush=True)
            result["artifacts"] = [{"path": name, "bytes": (output/name).stat().st_size} for name in names]
            report["scenes"][str(number)] = result
    write_gallery(output)
    report_path = output / f"visual_run_{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Gallery: {output/'index.html'}\nRun record: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
