"""Export independently reproducible DBSCAN teaching scenes; never starts a server."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import PIL
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sklearn
import scipy
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from dbscan_visual_core import (
    classification_names, expansion_experiment, fit_density, make_data,
    neighborhood_data, summarize, parameter_grid, scaling_experiment,
    EPS_VALUES, MIN_SAMPLES_VALUES, kth_neighbor_distances,
    three_dimensional_data, fixed_radius_density, variable_density_experiment,
    distance_concentration,
)


ROOT = Path(__file__).resolve().parent
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#6A51A3", "#56B4E9"]
CORE, BORDER, NOISE = "#0072B2", "#D97700", "#64748B"
INK, MUTED, PAPER = "#16283F", "#536278", "#F4F7FB"
SCENES = {
    1: ("Epsilon neighborhoods", ("01_epsilon_neighborhood.gif",)),
    2: ("Core, border and noise", ("02_core_border_noise.png", "02_core_border_noise.gif")),
    3: ("Exact DBSCAN expansion", ("03_dbscan_expansion.gif", "03_dbscan_final.png")),
    4: ("K-Means versus DBSCAN", ("04_kmeans_vs_dbscan.png",)),
    5: ("Parameter sensitivity", ("05_parameter_sensitivity.html", "05_cluster_count_heatmap.png", "05_noise_fraction_heatmap.png")),
    6: ("Self-inclusive k-distance", ("06_k_distance_plot.png",)),
    7: ("Genuine 3D clustering", ("07_dbscan_3d_interactive.html",)),
    8: ("Illustrative density landscape", ("08_density_landscape_3d.html",)),
    9: ("Variable-density limitation", ("09_variable_density_limitation.png",)),
    10: ("Feature scaling and distance", ("10_feature_scaling.png",)),
    11: ("High-dimensional distance contrast", ("11_high_dimensional_distances.png",)),
}


def style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": PAPER, "axes.facecolor": "white", "savefig.facecolor": PAPER,
        "axes.edgecolor": "#CBD5E1", "axes.spines.top": False, "axes.spines.right": False,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "grid.alpha": 0.3})


def page(fig, title: str, subtitle: str, footer: str) -> None:
    fig.suptitle(title, x=0.06, y=0.98, ha="left", fontsize=19, weight="bold")
    fig.text(0.06, 0.91, subtitle, color=MUTED, va="top", fontsize=10)
    fig.text(0.06, 0.035, footer, color=MUTED, va="bottom", fontsize=9, linespacing=1.4)


def setup_axis(ax, X: np.ndarray, *, equal: bool = True) -> None:
    lower, upper = X.min(axis=0), X.max(axis=0)
    margin = np.maximum(upper - lower, 1) * 0.13
    ax.set(xlim=(lower[0] - margin[0], upper[0] + margin[0]),
           ylim=(lower[1] - margin[1], upper[1] + margin[1]), xlabel="Feature x", ylabel="Feature y")
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.grid(True)


def label_colors(labels: np.ndarray) -> list[str]:
    return [PALETTE[int(label) % len(PALETTE)] if label >= 0 else NOISE for label in labels]


def plot_partition(ax, X: np.ndarray, result: dict, *, by_type: bool = False, size: int = 30) -> None:
    labels, core = result["labels"], result["core"]
    for name, mask, marker in (("Core", core, "o"), ("Border", (labels >= 0) & ~core, "s"),
                               ("Noise", labels == -1, "x")):
        colors = {"Core": CORE, "Border": BORDER, "Noise": NOISE}[name] if by_type else label_colors(labels[mask])
        ax.scatter(*X[mask].T, s=size, c=colors, marker=marker,
                   linewidths=0.8, label=f"{name} ({int(mask.sum())})", zorder=3)
    setup_axis(ax, X)


def save_png(fig, path: Path) -> None:
    try:
        fig.savefig(path, dpi=130)
    finally:
        plt.close(fig)


def save_gif(fig, draw, frames, path: Path, *, fps: float = 2) -> None:
    animation = FuncAnimation(fig, draw, frames=frames, repeat=True, blit=False, cache_frame_data=False)
    try:
        animation.save(path, writer=PillowWriter(fps=fps), dpi=90)
    except Exception as exc:
        raise RuntimeError(f"GIF export failed for {path.name}; check Matplotlib/Pillow and output permissions: {exc}") from exc
    finally:
        plt.close(fig)


def render_epsilon(output: Path) -> dict:
    X = neighborhood_data()
    distances = np.linalg.norm(X - X[0], axis=1)
    radii = np.r_[np.linspace(0.12, 1.25, 22), np.linspace(1.25, 0.12, 22)[1:]]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(left=0.08, right=0.69, bottom=0.20, top=0.81)
    page(fig, "01 / How epsilon defines a neighborhood", "Synthetic points • fixed min_samples = 7 • the reference point counts itself",
         "Mechanism: distance(x, y) <= epsilon. Only the radius changes; the density requirement stays fixed.\nInterview: epsilon has units and a local meaning; it is not a maximum cluster diameter.")
    info = fig.text(0.73, 0.68, "", va="top", fontsize=12, linespacing=1.8)
    def draw(frame):
        eps = radii[frame]
        inside = distances <= eps
        ax.clear()
        ax.scatter(*X[~inside].T, c="#CBD5E1", s=40)
        ax.scatter(*X[inside].T, c=CORE, s=52, label="Inside, including self")
        ax.scatter(*X[0], marker="*", s=250, c=BORDER, edgecolors=INK, zorder=6, label="Reference point")
        ax.add_patch(Circle(X[0], eps, facecolor=CORE, alpha=0.08))
        ax.add_patch(Circle(X[0], eps, fill=False, edgecolor=CORE, lw=2))
        setup_axis(ax, X)
        ax.set(xlim=(-1.65, 1.65), ylim=(-1.45, 1.45))
        ax.legend(loc="upper left", fontsize=8)
        info.set_text(f"epsilon = {eps:.2f}\nNeighbors = {inside.sum()}\n= {inside.sum()-1} others + self\nmin_samples = 7\n\n{'CORE' if inside.sum() >= 7 else 'NOT CORE'}\n{inside.sum()} {'>=' if inside.sum() >= 7 else '<'} 7")
        info.set_color(CORE if inside.sum() >= 7 else INK)
    save_gif(fig, draw, range(len(radii)), output / SCENES[1][1][0], fps=5)
    return {"seed": 42, "n": len(X), "min_samples": 7, "epsilon_range": [0.12, 1.25],
            "frames": len(radii), "neighbor_count_range": [int((distances <= radii.min()).sum()), int((distances <= radii.max()).sum())]}


def classification_figure(result: dict, visible: int | None = None):
    X = result["X"]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(left=0.08, right=0.74, bottom=0.21, top=0.79)
    page(fig, "02 / Core, border and noise are different roles", "A fixed completed DBSCAN fit • epsilon = 0.64 • min_samples = 4, including self",
         "Border: not core, but within epsilon of a core. Noise: no core neighbor.\nInterview: non-core does not mean noise. These are density roles, not verified anomaly labels.")
    return fig, ax


def draw_classification(ax, result: dict, visible: int | None = None) -> None:
    X = result["X"]
    ax.clear()
    if visible is None:
        visible = len(X)
    if visible < len(X):
        ax.scatter(*X[visible:].T, c="#CBD5E1", s=38)
    partial = {"labels": result["labels"][:visible], "core": result["core"][:visible]}
    plot_partition(ax, X[:visible], partial, by_type=True, size=55)
    setup_axis(ax, X)
    selected_core = int(np.flatnonzero(result["core"])[0])
    for i, color, text in ((selected_core, CORE, "Core neighborhood"), (0, BORDER, "Border neighborhood")):
        ax.add_patch(Circle(X[i], result["eps"], fill=False, edgecolor=color, ls="--", lw=1.7))
        if visible == len(X):
            count = int((np.linalg.norm(X - X[i], axis=1) <= result["eps"]).sum())
            ax.annotate(f"point {i}: {count} incl. self", X[i], xytext=(8, -22 if i else -37),
                        textcoords="offset points", fontsize=8, color=color)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.68), frameon=False)
    ax.set_title(f"Inspect {min(visible, len(X))}/{len(X)} observations • final-fit roles revealed", fontsize=10)


def render_classification(output: Path) -> dict:
    result = expansion_experiment()
    fig, ax = classification_figure(result)
    draw_classification(ax, result)
    save_png(fig, output / SCENES[2][1][0])
    fig, ax = classification_figure(result)
    def draw(frame):
        draw_classification(ax, result, visible=frame)
        if frame:
            i = frame - 1
            count = int((np.linalg.norm(result["X"] - result["X"][i], axis=1) <= result["eps"]).sum())
            name = classification_names(result["labels"], result["core"])[i]
            ax.set_title(f"Point {i}: {count} neighbors including self / need 4 → {name}", fontsize=10)
            ax.scatter(*result["X"][i], s=170, facecolors="none", edgecolors=INK, lw=2)
    save_gif(fig, draw, [*range(1, len(result["X"]) + 1), *([len(result["X"])] * 4)],
             output / SCENES[2][1][1], fps=3)
    return {"eps": 0.64, "min_samples": 4, "n": len(result["X"]), **summarize(result["labels"], result["core"])}


def render_expansion(output: Path) -> dict:
    result = expansion_experiment()
    X, core, steps = result["X"], result["core"], result["steps"]
    fig, ax = plt.subplots(figsize=(11, 6.7))
    fig.subplots_adjust(left=0.07, right=0.77, bottom=0.24, top=0.77)
    page(fig, "03 / Follow actual DBSCAN expansion", "Small code-defined curves • epsilon = 0.64 • min_samples = 4 • exact states, no interpolation",
         "Only core points enter the expansion queue. A previously unassigned/noise point can become border.\nInterview: a shared border may attach to the first component, but cannot join two core components.")
    caption = fig.text(0.06, 0.84, "", fontsize=10, va="top")
    info = fig.text(0.79, 0.72, "", fontsize=9, va="top", linespacing=1.6)
    def draw(frame):
        state = steps[frame]
        ax.clear()
        pending = state.labels == -2
        noise = state.labels == -1
        ax.scatter(*X[pending].T, c="#CBD5E1", s=42)
        ax.scatter(*X[noise].T, c=NOISE, marker="x", s=65)
        for mask, marker, size in (((state.labels >= 0) & core, "o", 58),
                                   ((state.labels >= 0) & ~core, "s", 60)):
            ax.scatter(*X[mask].T, c=label_colors(state.labels[mask]), marker=marker, s=size)
        ax.scatter(*X[state.expanded].T, facecolors="none", edgecolors=INK, s=90, linewidths=1)
        if state.current is not None:
            i = state.current
            ax.add_patch(Circle(X[i], result["eps"], color="#E69F00", alpha=0.12))
            ax.add_patch(Circle(X[i], result["eps"], fill=False, edgecolor="#B7791F", lw=1.5))
            ax.scatter(*X[state.neighborhood].T, s=105, facecolors="none", edgecolors="#B7791F", lw=1)
            ax.scatter(*X[i], marker="*", s=250, color="#F5B841", edgecolors=INK, zorder=10)
        for i, point in enumerate(X):
            ax.annotate(str(i), point, xytext=(4, 5), textcoords="offset points", fontsize=7)
        setup_axis(ax, X)
        caption.set_text(f"{frame+1}/{len(steps)}  |  {state.caption}")
        info.set_text("COLOR = cluster ID\nSHAPE = role\n\nGrey dot: pending\nGrey x: noise for now\nCircle: assigned core\nSquare: assigned border\nDark ring: expanded core\nGold star: current point\nGold ring: neighbor\n\nCompleted components\nretain their colors.")
    save_gif(fig, draw, [*range(len(steps)), *([len(steps)-1] * 4)], output / SCENES[3][1][0], fps=2)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.22, top=0.78, right=0.78)
    page(fig, "03 / Final density-connected components", "Same observations and same parameters as the exact execution trace",
         "Colors identify clusters; circles are core, squares are border, grey crosses are noise.\nNo centroid is optimized. Labels are arbitrary; geometry and membership carry the meaning.")
    plot_partition(ax, X, result, size=60)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_png(fig, output / SCENES[3][1][1])
    return {"eps": 0.64, "min_samples": 4, "n": len(X), "exact_states": len(steps),
            "noise_to_border_points": [int(s.current) for s in steps if s.operation == "noise_to_border"],
            **summarize(result["labels"], core)}


def render_comparison(output: Path) -> dict:
    X, _ = make_data()
    result = fit_density(X, 0.18, 7)
    km = KMeans(n_clusters=2, n_init=10, random_state=42, algorithm="lloyd").fit(X)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.23, top=0.76, wspace=0.22)
    page(fig, "04 / Two definitions of a cluster", "Synthetic moons + uniform background • identical raw-coordinate axes and observations",
         "K-Means: nearest-centroid regions assign every observation. DBSCAN: chains of dense neighborhoods can curve.\nThe known K=2 favors this teaching baseline; no universal winner or validated anomaly detection is claimed.")
    low, high = X.min(axis=0) - 0.2, X.max(axis=0) + 0.2
    gx, gy = np.meshgrid(np.linspace(low[0], high[0], 160), np.linspace(low[1], high[1], 120))
    from matplotlib.colors import ListedColormap
    region = km.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    axes[0].contourf(gx, gy, region, levels=[-0.5, 0.5, 1.5], cmap=ListedColormap(PALETTE[:2]), alpha=0.10)
    axes[0].contour(gx, gy, region, levels=[0.5], colors=[MUTED], linewidths=1)
    axes[0].scatter(*X.T, c=label_colors(km.labels_), s=16)
    axes[0].scatter(*km.cluster_centers_.T, c=PALETTE[:2], marker="X", s=180, edgecolors=INK)
    axes[0].set_title("K-Means: K=2, n_init=10, seed=42\n2 groups • 0 noise • X = centroid", fontsize=11)
    plot_partition(axes[1], X, result, size=19)
    axes[1].set_title(f"DBSCAN: eps=0.18, min_samples=7\n{result['clusters']} groups • {result['noise_count']} noise • no centroids", fontsize=11)
    for ax in axes:
        setup_axis(ax, X)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]))
    axes[1].legend(loc="upper right", fontsize=8)
    save_png(fig, output / SCENES[4][1][0])
    return {"seed": 42, "n": len(X), "dbscan": summarize(result["labels"], result["core"]),
            "eps": 0.18, "min_samples": 7, "kmeans": {"K": 2, "n_init": 10, "noise_count": 0}}


def point_traces(X: np.ndarray, result: dict, *, three_dimensional: bool = False, heights=None) -> list:
    labels, core = result["labels"], result["core"]
    names = classification_names(labels, core)
    custom = [[i, int(labels[i]), str(names[i]), *map(float, X[i])] for i in range(len(X))]
    traces = []
    for name, mask, symbol in (("Core", core, "circle"),
                              ("Border", (labels >= 0) & ~core, "square"),
                              ("Noise", labels == -1, "x")):
        kwargs = dict(x=X[mask, 0].tolist(), y=X[mask, 1].tolist(), mode="markers", name=name,
                      marker=dict(size=4 if three_dimensional else 6, symbol=symbol,
                                  color=label_colors(labels[mask])),
                      customdata=[custom[i] for i in np.flatnonzero(mask)],
                      hovertemplate="Point %{customdata[0]}<br>Cluster %{customdata[1]}<br>%{customdata[2]}<br>x=%{customdata[3]:.3f}<br>y=%{customdata[4]:.3f}")
        if three_dimensional:
            kwargs["z"] = X[mask, 2].tolist() if heights is None else heights[mask].tolist()
            if X.shape[1] == 3:
                kwargs["hovertemplate"] += "<br>z=%{customdata[5]:.3f}"
            else:
                kwargs["hovertemplate"] += "<br>illustrative density=%{z:.3f}"
            traces.append(go.Scatter3d(**kwargs, hoverlabel=dict(namelength=-1)))
        else:
            traces.append(go.Scatter(**kwargs))
        traces[-1].hovertemplate += "<extra></extra>"
    return traces


def save_html(fig, path: Path, div_id: str, *, post_script=None) -> None:
    fig.write_html(path, include_plotlyjs=True, full_html=True, auto_play=False,
                   div_id=div_id, post_script=post_script,
                   config={"displaylogo": False, "responsive": True, "scrollZoom": True})


def sensitivity_title(result: dict) -> str:
    return ("05 / Density parameters change the neighborhood graph<br><sup>"
            f"eps={result['eps']:.2f} · min_samples={result['min_samples']} · "
            f"clusters={result['clusters']} · core={result['core_count']} · border={result['border_count']} · "
            f"noise={result['noise_count']} ({result['noise_fraction']:.1%})</sup>")


def sensitivity_figure(grid: dict):
    X, results = grid["X"], grid["results"]
    ei, mi = 2, 2
    fig = make_subplots(rows=2, cols=2, column_widths=[0.62, 0.38],
                        specs=[[{"rowspan": 2}, {}], [None, {}]],
                        subplot_titles=("Point color = cluster; shape = role", "Cluster count", "Noise fraction"),
                        horizontal_spacing=0.12, vertical_spacing=0.20)
    for trace in point_traces(X, results[mi][ei]):
        fig.add_trace(trace, row=1, col=1)
    for key, row, scale in (("cluster_counts", 1, "Blues"), ("noise_fractions", 2, "YlOrBr")):
        fig.add_trace(go.Heatmap(z=grid[key].tolist(), x=list(range(6)), y=list(range(5)), colorscale=scale,
                      colorbar=dict(len=0.34, y=0.82 if row == 1 else 0.19, thickness=12),
                      hovertemplate="eps index=%{x}<br>m index=%{y}<br>value=%{z}<extra></extra>"), row=row, col=2)
    for row in (1, 2):
        fig.add_trace(go.Scatter(x=[ei], y=[mi], mode="markers", showlegend=False, hoverinfo="skip",
                      marker=dict(symbol="square-open", size=24, color=INK, line=dict(width=2))), row=row, col=2)
        fig.update_xaxes(tickvals=list(range(6)), ticktext=[str(v) for v in EPS_VALUES], title="epsilon", row=row, col=2)
        fig.update_yaxes(tickvals=list(range(5)), ticktext=[str(v) for v in MIN_SAMPLES_VALUES], title="min_samples", row=row, col=2)
    frames = []
    for m_index, rows in enumerate(results):
        for e_index, result in enumerate(rows):
            data = point_traces(X, result)
            data += [go.Scatter(x=[e_index], y=[m_index]), go.Scatter(x=[e_index], y=[m_index])]
            frames.append(go.Frame(name=f"{m_index}-{e_index}", data=data, traces=[0, 1, 2, 5, 6],
                layout=dict(title=dict(text=sensitivity_title(result)),
                            meta=dict(eps=result["eps"], min_samples=result["min_samples"],
                                      clusters=result["clusters"], noise_count=result["noise_count"]))))
    fig.frames = frames
    fig.update_layout(template="plotly_white", height=900, font=dict(family="Arial", color=INK),
        title=dict(text=sensitivity_title(results[mi][ei]), x=0.04, y=0.97),
        margin=dict(t=160, b=210, l=65, r=80), uirevision="keep-point-view",
        legend=dict(orientation="h", x=0, y=1.09),
        meta=dict(eps=0.18, min_samples=7, clusters=results[mi][ei]["clusters"], noise_count=results[mi][ei]["noise_count"]),
        sliders=[dict(active=ei, currentvalue=dict(prefix="epsilon: "), pad=dict(t=42),
                      steps=[dict(method="skip", value=str(i), label=str(eps), args=[]) for i, eps in enumerate(EPS_VALUES)])],
        updatemenus=[dict(type="dropdown", active=mi, x=0.63, y=1.15, xanchor="left", yanchor="bottom",
            buttons=[dict(method="skip", name=str(i), label=f"min_samples = {m}", args=[]) for i, m in enumerate(MIN_SAMPLES_VALUES)])])
    fig.update_xaxes(title="Feature x", range=[-1.6, 2.6], row=1, col=1)
    fig.update_yaxes(title="Feature y", range=[-1.1, 1.6], scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.add_annotation(text="Fixed synthetic corpus · 30 precomputed fits · cluster IDs may change between fits<br>"
                           "Small epsilon can fragment; large epsilon can bridge. Counts need not vary monotonically.",
                       xref="paper", yref="paper", x=0, y=-0.32, showarrow=False, xanchor="left", font=dict(size=11))
    return fig


def render_sensitivity(output: Path) -> dict:
    grid = parameter_grid()
    fig = sensitivity_figure(grid)
    controls = """
const graph = document.getElementById('{plot_id}');
let epsilonIndex = 2, minimumIndex = 2;
const redraw = () => Plotly.animate(graph, [`${minimumIndex}-${epsilonIndex}`], {
  mode:'immediate', frame:{duration:0,redraw:true}, transition:{duration:0}});
graph.on('plotly_sliderchange', event => {epsilonIndex=Number(event.step.value); redraw();});
graph.on('plotly_buttonclicked', event => {minimumIndex=Number(event.button.name); redraw();});
"""
    save_html(fig, output / SCENES[5][1][0], "dbscan-sensitivity", post_script=controls)
    for index, key, title, cmap in ((1, "cluster_counts", "Detected cluster count", "Blues"),
                                    (2, "noise_fractions", "Fraction left as noise", "YlOrBr")):
        figure, ax = plt.subplots(figsize=(9.5, 6.5))
        figure.subplots_adjust(left=0.12, right=0.90, bottom=0.24, top=0.79)
        page(figure, f"05 / {title}", "Same 485-point synthetic corpus • independent fits at every grid cell",
             "Mechanism: epsilon changes neighbors; min_samples changes the core requirement.\nInterview: inspect coverage and connectivity together; this grid does not identify an optimal setting.")
        values = grid[key]
        heatmap = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap)
        ax.set(xticks=range(6), xticklabels=EPS_VALUES, yticks=range(5), yticklabels=MIN_SAMPLES_VALUES,
               xlabel="epsilon (raw coordinate units)", ylabel="min_samples (including self)")
        for mi in range(5):
            for ei in range(6):
                value = values[mi, ei]
                contrast = "white" if value > (values.min() + values.max()) / 2 else INK
                ax.text(ei, mi, f"{value:.1%}" if index == 2 else str(int(value)), ha="center", va="center", color=contrast)
        figure.colorbar(heatmap, ax=ax, pad=0.03, fraction=0.04)
        save_png(figure, output / SCENES[5][1][index])
    return {"n": len(grid["X"]), "eps_values": list(EPS_VALUES), "min_samples_values": list(MIN_SAMPLES_VALUES),
            "cluster_counts": grid["cluster_counts"].tolist(), "noise_fractions": grid["noise_fractions"].tolist(),
            "fits": [{"eps": r["eps"], "min_samples": r["min_samples"], **summarize(r["labels"], r["core"])} for row in grid["results"] for r in row]}


def render_k_distance(output: Path) -> dict:
    X, _ = make_data()
    radii = kth_neighbor_distances(X, 7)
    candidates = [0.12, 0.18, 0.25]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.24, top=0.79)
    page(fig, "06 / A radius diagnostic, not an optimum", "7th observation includes the query point itself • 485 raw-coordinate observations",
         "Mechanism: sort each observation's radius needed to include 7 observations, including self.\nInterview: slope changes suggest candidates; elbows can be ambiguous, especially with unequal densities.")
    ax.plot(np.arange(1, len(X)+1), radii, color=CORE, lw=2, label="Sorted self-inclusive 7-distance")
    for eps, color in zip(candidates, PALETTE[1:4]):
        ax.axhline(eps, color=color, ls="--", lw=1.3, label=f"Candidate eps = {eps:.2f}")
    ax.set(xlabel="Sorted observation rank (not point identity)", ylabel="Distance in raw coordinate units", ylim=(0, radii.max()*1.08))
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=9)
    save_png(fig, output / SCENES[6][1][0])
    return {"n": len(X), "min_samples": 7, "includes_self": True, "candidate_eps": candidates,
            "quantiles": np.quantile(radii, [0.5, 0.9, 0.95, 0.99, 1]).tolist()}


def render_scaling(output: Path) -> dict:
    experiment = scaling_experiment()
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.28, top=0.72, wspace=0.34)
    page(fig, "10 / Feature units define the neighborhood", "Identical observation identities • synthetic spatial axes • epsilon belongs to each fitting space",
         "Left: comparable units. Middle: x multiplied by 30. Right: StandardScaler fitted on the distorted corpus.\nPanels use auto aspect for legibility; circles can appear elliptical on screen. Numeric distances use the axis units.\nInterview: scaling chooses weights. Standardization is useful here by construction, not universally correct.")
    names = ["Comparable synthetic units", "Distorted x (30x units)", "Standardized (dimensionless)"]
    for ax, X, result, name in zip(axes, experiment["spaces"], experiment["results"], names):
        plot_partition(ax, X, result, size=17)
        setup_axis(ax, X, equal=False)
        ax.set_aspect("auto")
        ax.set_title(f"{name}\neps=0.30, m=7\n{result['clusters']} clusters / {result['noise_count']} noise", fontsize=10)
        ax.add_patch(Circle(X[0], 0.3, fill=False, lw=1.4, edgecolor=INK))
        ax.set_xlabel("x coordinate" if name != names[2] else "standardized x")
        ax.set_ylabel("y coordinate" if name != names[2] else "standardized y")
    save_png(fig, output / SCENES[10][1][0])
    return {"n": 300, "seed": 42, "centers": [[-2, -1], [2, 1]], "cluster_std": 0.27,
            "x_multiplier": 30, "eps_in_each_space": 0.3, "min_samples": 7,
            "fitted_scales": experiment["fitted_scales"], "fitted_means": experiment["fitted_means"],
            "fits": [summarize(r["labels"], r["core"]) for r in experiment["results"]]}


def three_dimensional_figure(X: np.ndarray, result: dict):
    fig = go.Figure(point_traces(X, result, three_dimensional=True))
    fig.update_layout(template="plotly_white", height=820, font=dict(color=INK),
        title=dict(text="07 / Density connectivity in three actual features<br><sup>"
             f"Euclidean DBSCAN on x, y, z · eps=0.26 · min_samples=7 · {result['clusters']} clusters · {result['noise_count']} noise</sup>", x=0.04),
        margin=dict(l=30, r=30, t=100, b=90), legend=dict(orientation="h", x=0.05, y=1.02),
        scene=dict(xaxis_title="Feature x", yaxis_title="Feature y", zaxis_title="Feature z",
                   aspectmode="data", camera=dict(eye=dict(x=1.6, y=1.5, z=1.1), projection=dict(type="orthographic"))),
        meta=dict(fit_dimensions=3, n=len(X), eps=0.26, min_samples=7), uirevision="3d-camera")
    fig.add_annotation(text="Rotate / zoom / hover · circles=core, squares=border, grey x=noise<br>"
                           "Synthetic curved tubes + background. Full 3D vectors, not a 2D embedding. Noise is not an anomaly verdict.",
                       xref="paper", yref="paper", x=0, y=-0.06, showarrow=False, xanchor="left", font=dict(size=12))
    return fig


def render_three_dimensional(output: Path) -> dict:
    X = three_dimensional_data()
    result = fit_density(X, 0.26, 7)
    save_html(three_dimensional_figure(X, result), output / SCENES[7][1][0], "dbscan-3d")
    return {"seed": 42, "n": len(X), "dimensions": 3, "tube_points_each": 140, "jitter_std": 0.045,
            "background_count": 24, "eps": 0.26, "min_samples": 7, **summarize(result["labels"], result["core"])}


def density_landscape_figure():
    X, _ = make_data()
    eps, m = 0.18, 7
    result = fit_density(X, eps, m)
    gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 75), np.linspace(-1.1, 1.6, 65))
    counts, density = fixed_radius_density(X, np.c_[gx.ravel(), gy.ravel()], eps)
    point_counts, point_density = fixed_radius_density(X, X, eps)
    threshold = m/(len(X)*np.pi*eps**2)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=gx.tolist(), y=gy.tolist(), z=density.reshape(gx.shape).tolist(),
        customdata=counts.reshape(gx.shape).tolist(), colorscale="Cividis", opacity=0.8,
        colorbar=dict(title="Count / (n pi eps²)", thickness=15),
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>count=%{customdata}<br>illustrative density=%{z:.3f}<extra></extra>", name="Count landscape"))
    fig.add_trace(go.Surface(x=[-1.6, 2.6], y=[-1.1, 1.6], z=[[threshold]*2]*2,
        showscale=False, colorscale=[[0, "#E69F00"], [1, "#E69F00"]], opacity=0.2,
        name="Core count threshold", showlegend=True, hovertemplate=f"Illustrative core threshold {threshold:.3f}<extra></extra>"))
    for trace in point_traces(X, result, three_dimensional=True, heights=point_density):
        fig.add_trace(trace)
    fig.update_layout(template="plotly_white", height=840, font=dict(color=INK),
        title=dict(text="08 / A count landscape explains density, not a DBSCAN objective<br><sup>"
                        "2D features below; height = fixed-radius count / (n pi eps²). Amber plane: min_samples / (n pi eps²).</sup>", x=0.04),
        margin=dict(l=25, r=100, t=115, b=100), legend=dict(orientation="h", x=0, y=1.02),
        scene=dict(xaxis_title="Feature x (DBSCAN input)", yaxis_title="Feature y (DBSCAN input)",
                   zaxis_title="Illustrative empirical density", aspectmode="manual", aspectratio=dict(x=1.6, y=1, z=0.8),
                   camera=dict(eye=dict(x=1.5, y=-1.7, z=1.4), projection=dict(type="orthographic"))),
        meta=dict(fit_dimensions=2, n=len(X), eps=eps, min_samples=m, threshold=threshold), uirevision="landscape-camera")
    fig.add_annotation(text="DBSCAN uses observed-point counts and core connectivity; it does not fit this surface.<br>"
                           "Grid interpolation is illustrative. Point colors/roles are actual DBSCAN output; height is not a third input feature.",
                       xref="paper", yref="paper", x=0, y=-0.08, showarrow=False, xanchor="left", font=dict(size=12))
    return fig, {"n": len(X), "fit_dimensions": 2, "grid_shape": list(gx.shape), "eps": eps, "min_samples": m,
                 "illustrative_density_threshold": threshold, "point_count_range": [int(point_counts.min()), int(point_counts.max())],
                 "grid_count_range": [int(counts.min()), int(counts.max())], **summarize(result["labels"], result["core"])}


def render_landscape(output: Path) -> dict:
    figure, record = density_landscape_figure()
    save_html(figure, output / SCENES[8][1][0], "dbscan-landscape")
    return record


def render_variable_density(output: Path) -> dict:
    experiment = variable_density_experiment()
    X, provenance = experiment["X"], experiment["provenance"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 6.3))
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.26, top=0.74, wspace=0.30)
    page(fig, "09 / One radius must serve unlike densities", "Fixed synthetic populations • first panel is provenance, later panels are DBSCAN outputs",
         "A tight radius can discard the broad group; a wider radius can create bridges and absorb background.\nInterview: review sparse coverage and connectivity together. A global threshold can work on a sample, but offers no universal guarantee.\nGaussian source labels describe generation, not verified clusters or anomaly truth. min_samples=7, including self, for all fits.")
    for i, name in enumerate(("Dense source", "Sparse source", "Background source")):
        axes[0].scatter(*X[provenance == i].T, c=[PALETTE[0], PALETTE[1], NOISE][i],
                        s=15, marker="x" if i == 2 else "o", label=name)
    setup_axis(axes[0], X)
    axes[0].set_title("Generation provenance\n120 compact + 120 broad + 15 background", fontsize=9)
    axes[0].legend(loc="upper left", fontsize=6.5)
    fits = []
    for ax, result in zip(axes[1:], experiment["results"]):
        plot_partition(ax, X, result, size=17)
        assigned = result["labels"] >= 0
        sparse_retained = int(assigned[120:240].sum())
        ax.set_title(f"eps={result['eps']:.2f}, m=7\n{result['clusters']} groups / {result['noise_count']} noise\nsparse assigned: {sparse_retained}/120", fontsize=10)
        fits.append({"eps": result["eps"], **summarize(result["labels"], result["core"]),
                     "dense_assigned": int(assigned[:120].sum()), "sparse_assigned": sparse_retained,
                     "background_assigned": int(assigned[240:].sum())})
    save_png(fig, output / SCENES[9][1][0])
    return {"seed": 42, "n": len(X), "centers": [[-0.65, 0], [0.65, 0]], "standard_deviations": [0.055, 0.32],
            "population_sizes": [120, 120, 15], "min_samples": 7, "fits": fits}


def render_high_dimension(output: Path) -> dict:
    result = distance_concentration()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.25, top=0.77, wspace=0.28)
    page(fig, "11 / Does distance still provide useful contrast?", "Supplement: independent standard-Gaussian coordinates • 180 observations per dimension • seed 42",
         "Mechanism: in this isotropic noise control, pair distances concentrate relative to their overall scale.\nInterview: investigate feature relevance and neighborhood contrast; this does not prove real high-dimensional embeddings are unclusterable.")
    bins = np.linspace(0, max(float(d.max()) for d in result["relative_distances"])*1.01, 65)
    for dimension, values, color in zip(result["dimensions"], result["relative_distances"], PALETTE):
        axes[0].hist(values, bins=bins, density=True, histtype="step", linewidth=1.8, color=color, label=f"D={dimension}")
    axes[0].set(xlabel="Pair distance / sqrt(2D)", ylabel="Empirical histogram density", title="Relative distance spread")
    axes[0].legend()
    axes[1].plot(result["dimensions"], result["coefficient_of_variation"], "o-", color=CORE, label="Pair-distance coefficient of variation")
    axes[1].plot(result["dimensions"], result["median_nearest_farthest_ratio"], "s-", color=BORDER, label="Median nearest / farthest distance")
    axes[1].set(xlabel="Number of independent features D", ylabel="Unitless statistic", title="Local versus overall distance contrast", ylim=(0, 1))
    axes[1].set_xscale("log")
    axes[1].set_xticks(result["dimensions"], labels=result["dimensions"])
    axes[1].legend(loc="center right", fontsize=8)
    for ax in axes:
        ax.grid(True)
    save_png(fig, output / SCENES[11][1][0])
    return {key: value for key, value in result.items() if key != "relative_distances"}


RENDERERS = {1: render_epsilon, 2: render_classification, 3: render_expansion, 4: render_comparison,
             5: render_sensitivity, 6: render_k_distance, 7: render_three_dimensional,
             8: render_landscape, 9: render_variable_density, 10: render_scaling, 11: render_high_dimension}


def write_gallery(output: Path) -> None:
    cards = []
    for number, (title, names) in SCENES.items():
        available = [name for name in names if (output / name).is_file()]
        if not available:
            continue
        media = next((name for name in available if name.endswith((".gif", ".png"))), None)
        links = " · ".join(f'<a href="{html.escape(name)}">{html.escape(name)}</a>' for name in available)
        picture = f'<img loading="lazy" src="{media}" alt="{html.escape(title)}">' if media else ""
        cards.append(f"<article><h2>{number:02d} / {html.escape(title)}</h2>{picture}<p>{links}</p></article>")
    content = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DBSCAN visual lab</title><style>body{font:17px system-ui;color:#16283f;background:#f4f7fb;max-width:1120px;margin:40px auto;padding:0 20px}article{background:white;padding:20px;margin:22px 0;border:1px solid #dce3ec;border-radius:12px}img{width:100%;height:auto}a{color:#006ba7}p{line-height:1.6}</style><h1>DBSCAN: observe the geometry</h1><p>Deterministic synthetic teaching controls. Open the HTML exports to rotate or change parameters; no server or network is needed. Interpretation candidates require author review.</p>' + "".join(cards) + '</html>'
    (output / "index.html").write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--visualization", type=int, choices=sorted(SCENES))
    group.add_argument("--list", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args(argv)
    if args.list:
        for number, (title, names) in SCENES.items():
            print(f"{number:2d}  {title}: {', '.join(names)}")
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = sorted(SCENES) if args.all else [args.visualization]
    style()
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                           "plotly": plotly.__version__, "pillow": PIL.__version__, "scipy": scipy.__version__},
              "interpretation_status": "author review required", "scenes": {}}
    with threadpool_limits(limits=1):
        for number in selected:
            try:
                result = RENDERERS[number](output)
            except Exception as exc:
                raise RuntimeError(f"Visualization {number} ({SCENES[number][0]}) failed: {exc}") from exc
            result["artifacts"] = [{"path": name, "bytes": (output / name).stat().st_size} for name in SCENES[number][1]]
            report["scenes"][str(number)] = result
            print(f"Generated {number:02d}: {', '.join(SCENES[number][1])}", flush=True)
    write_gallery(output)
    suffix = "all" if args.all else str(args.visualization).zfill(2)
    (output / f"visual_report_{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
