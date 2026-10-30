"""Offline Plotly scenes and Matplotlib/Pillow exports for the manifold lab."""

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

import manifold_visual_core as core


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
DIGIT_SCALE = [(bound, color) for i, color in enumerate(COLORS)
               for bound in (i / 10, (i + 1) / 10)]
CSS = """
body{margin:0;background:#edf2f7;color:#192d45;font:16px/1.6 system-ui,sans-serif}
main{max-width:1320px;margin:auto;padding:28px}h1{font-size:2.1rem;line-height:1.2}
h2{font-size:1.3rem}a{color:#006ba6}nav{margin-bottom:24px}code{background:#e5edf5;padding:2px 5px}
.card{background:white;border:1px solid #d9e2ec;border-radius:10px;padding:22px;margin:18px 0}
.warning{border-left:5px solid #c88300}.takeaway{border-left:5px solid #008679}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
img{max-width:100%;height:auto}small{color:#52657a}.plotly-graph-div{min-height:560px}
table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #d9e2ec}
"""


def style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white",
                         "axes.labelcolor": "#19324d", "text.color": "#19324d",
                         "savefig.facecolor": "white"})


def prepare(root):
    root = Path(root)
    for folder in ("interactive", "animations", "figures", "cache"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    if not (root / "plotly.min.js").exists():
        (root / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    style()


def plot_page(root, filename, title, concept, math, figure, parameters, observe, mistake, takeaway):
    # Preserve geometric units instead of stretching scatter axes independently.
    for axis in figure.select_yaxes():
        axis.update(scaleanchor=axis.plotly_name.replace("yaxis", "x"), scaleratio=1, constrain="domain")
    figure.update_scenes(aspectmode="data")
    figure.update_layout(template="plotly_white", height=700, margin=dict(t=115, b=75, l=65, r=45),
                         font=dict(family="Arial, sans-serif", color="#19324d", size=13),
                         title_x=0.5, title_y=0.98, title_xanchor="center", title_yanchor="top")
    for menu in figure.layout.updatemenus:
        menu.y = 1.10
    body = figure.to_html(full_html=False, include_plotlyjs="../plotly.min.js",
                          div_id=Path(filename).stem,
                          config={"scrollZoom": True, "responsive": True, "displaylogo": False})
    text = (f'<!doctype html><html lang="en"><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style><main>'
            f'<nav><a href="../index.html">← Learning path</a></nav><h1>{html.escape(title)}</h1>'
            f'<p>{html.escape(concept)}</p><div class="card"><strong>Mathematical idea</strong><br>'
            f'<code>{html.escape(math)}</code><p>{html.escape(parameters)}</p></div>' + body +
            f'<div class="card"><strong>Inspect</strong><p>{html.escape(observe)}</p></div>'
            f'<div class="card warning"><strong>Common mistake and limitation</strong>'
            f'<p>{html.escape(mistake)}</p></div><div class="card takeaway"><strong>Takeaway</strong>'
            f'<p>{html.escape(takeaway)}</p></div><small>Computed exploratory evidence. '
            'Interpretation candidates require author review. No external scripts or APIs.</small></main></html>')
    (Path(root) / "interactive" / filename).write_text(text, encoding="utf-8")


def save_figure(fig, root, filename):
    fig.savefig(Path(root) / "figures" / filename, dpi=120, bbox_inches="tight")
    plt.close(fig)


def model_fit(root, settings, X, method, components=2, seed=42, **parameters):
    Y, info = core.fit_embedding(X, method, Path(root) / "cache", n_components=components,
                                seed=seed, iterations=settings["iterations"], epochs=settings["epochs"],
                                **parameters)
    info["scores"] = core.evaluate(X, Y)
    return Y, info


def scatter(Y, colors, *, digits=False, original_rows=None, visible=True, name=""):
    marker = dict(size=6, color=colors, colorscale=DIGIT_SCALE if digits else "Viridis",
                  showscale=True, opacity=0.85)
    if digits:
        marker.update(cmin=-0.5, cmax=9.5, colorbar=dict(title="Digit", tickvals=list(range(10))))
    else:
        marker.update(cmin=float(np.min(colors)), cmax=float(np.max(colors)),
                      colorbar=dict(title="Roll t"))
    custom = np.column_stack([np.arange(len(Y)), colors,
                              np.arange(len(Y)) if original_rows is None else original_rows])
    hover = "sample row=%{customdata[0]}<br>" + ("digit" if digits else "roll t") + "=%{customdata[1]}"
    hover += "<br>source row=%{customdata[2]}<extra></extra>"
    common = dict(mode="markers", marker=marker, customdata=custom, hovertemplate=hover,
                  visible=visible, name=name, showlegend=False)
    if Y.shape[1] == 3:
        return go.Scatter3d(x=Y[:, 0], y=Y[:, 1], z=Y[:, 2], **common)
    return go.Scatter(x=Y[:, 0], y=Y[:, 1], **common)


def swiss(root, settings):
    X, t = core.swiss_sample(settings["swiss_n"])
    fits = {m: model_fit(root, settings, X, m) for m in ("PCA", "t-SNE", "UMAP")}
    titles = ["Original synthetic 3D geometry"] + [
        f'{m}: T10={fits[m][1]["scores"]["T@10"]:.3f}; R10={fits[m][1]["scores"]["R@10"]:.3f}'
        for m in fits]
    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}],
                                              [{"type": "xy"}, {"type": "xy"}]], subplot_titles=titles)
    fig.add_trace(scatter(X, t), row=1, col=1)
    for (m, (Y, _)), (row, col) in zip(fits.items(), ((1, 2), (2, 1), (2, 2))):
        trace = scatter(Y, t)
        trace.marker.showscale = False
        fig.add_trace(trace, row=row, col=col)
        fig.update_xaxes(title_text="PC1" if m == "PCA" else "embedding coordinate 1", row=row, col=col)
        fig.update_yaxes(title_text="PC2" if m == "PCA" else "embedding coordinate 2", row=row, col=col)
    fig.update_layout(meta={"input_shape": list(X.shape), "metric": "euclidean", "seed": 42},
                      scene=dict(xaxis_title="original x", yaxis_title="original y", zaxis_title="original z"))
    plot_page(root, "01_swiss_roll.html", "Unfold a curved surface", "A linear plane can overlap layers of a curved surface. Compare actual nonlinear fits with the original Swiss Roll.",
              "X ∈ R^(n×3) → Y ∈ R^(n×2); original distance = Euclidean in 3D",
              fig, f'Synthetic Swiss Roll: n={len(X)}, noise=0.05, seed=42; no scaling. t-SNE p=30; UMAP k=15, min_dist=0.1.',
              "Rotate the original; follow the same roll-t colors in each panel. Hover to identify rows. Color continuity and local fidelity are different questions.",
              "This toy starts in 3D, while real embeddings may have hundreds of dimensions. Euclidean shortcuts across roll layers differ from intrinsic surface distances; nonlinear fits need not recover a perfect rectangle.",
              "A nonlinear map can reveal curved structure missed by a linear projection, while still distorting other relationships.")
    return {"files": ["interactive/01_swiss_roll.html"], "hypothesis": "Linear and nonlinear objectives retain different relationships on a curved surface.",
            "data": {"kind": "synthetic Swiss Roll", "n": len(X), "sha256": core.array_hash(X)},
            "fits": {m: info for m, (_, info) in fits.items()}}


def digits_3d(root, settings):
    X, y, rows = core.digits_sample(settings["digits_n"])
    fits = {m: model_fit(root, settings, X, m, components=3) for m in ("PCA", "t-SNE", "UMAP")}
    fig = go.Figure()
    for i, (m, (Y, _)) in enumerate(fits.items()):
        fig.add_trace(scatter(Y, y, digits=True, original_rows=rows, visible=i == 0, name=m))
    buttons = [dict(label=m, method="update", args=[{"visible": [j == i for j in range(3)]},
               {"title.text": f'{m}: real 3D fit · T10={info["scores"]["T@10"]:.3f}; R10={info["scores"]["R@10"]:.3f}'}])
               for i, (m, (_, info)) in enumerate(fits.items())]
    fig.update_layout(title="PCA: real 3D fit", updatemenus=[dict(buttons=buttons, x=0, y=1.14)],
                      scene=dict(xaxis_title="coordinate 1", yaxis_title="coordinate 2", zaxis_title="coordinate 3"),
                      uirevision="keep-camera", meta={"dimensions": 3, "samples": len(X), "methods": list(fits)})
    plot_page(root, "02_digits_3d.html", "Explore genuine three-dimensional embeddings",
              "Each dropdown entry is a separate PCA, t-SNE or UMAP model fitted directly with n_components=3. Labels only color the outputs.",
              "64 pixel-count features / 16 → 3 fitted coordinates", fig,
              f'Public scikit-learn Digits, n={len(X)}; sample and embedding seed=42. Euclidean metric; no standardization or pre-PCA.',
              "Switch method, drag to rotate, scroll to zoom, and hover for the digit and original dataset row. Inspect depth rather than assuming overlap from one viewing angle.",
              "PCA axes are variance directions; t-SNE/UMAP axes have no independent semantic units. There is no injected noise dimension and no shared coordinate frame between fits.",
              "A third fitted dimension changes the optimization problem; it does not certify semantic or predictive quality.")
    return {"files": ["interactive/02_digits_3d.html"], "hypothesis": "A third fitted dimension permits spatial inspection but does not establish task quality.",
            "data": {"kind": "public digits /16", "n": len(X), "sha256": core.array_hash(X)},
            "fits": {m: info for m, (_, info) in fits.items()}}


def probabilities(root, settings):
    X, _ = core.probability_sample()
    reference = 0
    others = np.arange(len(X)) != reference
    distances2 = np.sum((X[others] - X[reference]) ** 2, axis=1)
    solutions, frames = {}, []
    for target in (5, 15, 30):
        solution, search = core.match_perplexity(distances2, target)
        solutions[target] = solution
        chosen = np.unique(np.linspace(0, len(search) - 1, min(10, len(search))).astype(int))
        frames.extend((target, int(i), search[i]) for i in chosen)
        frames.extend([(target, len(search) - 1, solution)] * 3)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), layout="constrained")

    def draw(ax, state, title):
        p = state["probabilities"]
        ax.scatter(X[others, 0], X[others, 1], s=25 + 700 * p,
                   c=p, cmap="viridis", vmin=0, vmax=0.3, alpha=0.85)
        segments = np.array([[X[reference], q] for q in X[others]])
        ax.add_collection(LineCollection(segments, colors="#2785a4", linewidths=0.2 + 18 * p, alpha=0.35))
        ax.scatter(*X[reference], marker="*", c="black", s=180, zorder=5)
        ax.add_patch(Circle(X[reference], state["sigma"], fill=False, ls="--", color="#d48100", lw=2))
        ax.set(xlim=(-4, 4), ylim=(-4, 4), title=title, xlabel="original x", ylabel="original y")
        ax.set_aspect("equal")

    for ax, (target, state) in zip(axes, solutions.items()):
        draw(ax, state, f'Target perplexity {target}\nσ={state["sigma"]:.3f}; achieved={state["perplexity"]:.3f}')
    fig.suptitle("Perplexity is effective neighborhood size · 72 synthetic points\nStar: reference; dashed circle: σ, not a cutoff; edge width / marker area encode p(j|i)")
    save_figure(fig, root, "03_probability_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
    order = np.argsort(distances2)
    def update(frame):
        target, iteration, state = frame
        for ax in axes:
            ax.clear()
        draw(axes[0], state, "Original Euclidean geometry")
        raw = np.exp(-distances2 / (2 * state["sigma"] ** 2))
        axes[1].plot(np.sqrt(distances2[order]), raw[order], color="#d48100", label="Gaussian weight (unnormalized)")
        axes[1].plot(np.sqrt(distances2[order]), state["probabilities"][order], color="#0072b2", label="Conditional probability p(j|i)")
        axes[1].set(xlabel="distance from reference (original units)", ylabel="weight / probability",
                    ylim=(-0.02, 1.05), title=f'H={state["entropy_bits"]:.3f} bits; 2^H={state["perplexity"]:.2f}')
        axes[1].legend(fontsize=8)
        fig.suptitle(f'Gaussian bandwidth search · target perplexity={target} · actual search iteration={iteration + 1}\n'
                     'p(j|i) = exp(−d²/2σ²) / sum(weights); self excluded; no hard neighbor count', fontsize=12)
    animation = FuncAnimation(fig, update, frames=frames, interval=250, repeat=True, cache_frame_data=False)
    animation.save(Path(root) / "animations/03_neighborhood_probabilities.gif", writer=PillowWriter(fps=4), dpi=95)
    plt.close(fig)
    return {"files": ["figures/03_probability_comparison.png", "animations/03_neighborhood_probabilities.gif"],
            "hypothesis": "Entropy matching changes local Gaussian scale rather than selecting a fixed neighbor count.",
            "configuration": {"kind": "synthetic 2D blobs", "n": 72, "reference_row": 0, "seed": 42},
            "solutions": {str(p): {k: v for k, v in state.items() if k != "probabilities"}
                          for p, state in solutions.items()}, "frames": len(frames)}


def optimization(root, settings):
    X, labels = make_blobs(n_samples=60, n_features=6, centers=3, cluster_std=1.3, random_state=42)
    trace = core.optimization_trace(X, steps=settings["toy_steps"], stride=10)
    positions = trace["positions"]
    bounds = max(0.1, float(np.abs(positions).max()) * 1.2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout="constrained")
    def update(frame):
        for ax in axes:
            ax.clear()
        Y = positions[frame]
        step = int(trace["iterations"][frame])
        axes[0].scatter(Y[:, 0], Y[:, 1], c=labels, cmap="tab10", vmin=0, vmax=9, s=35)
        attraction, repulsion = core.descent_forces(Y, trace["P"])
        vectors = (attraction[0], repulsion[0])
        scale = 0.25 * bounds / max(np.linalg.norm(v) for v in vectors)
        for vector, color, label in zip(vectors, ("#0072b2", "#d55e00"), ("attraction", "repulsion")):
            end = Y[0] + vector * scale
            axes[0].annotate("", xy=end, xytext=Y[0], arrowprops=dict(arrowstyle="->", color=color, lw=2))
            axes[0].plot([], [], color=color, label=label)
        axes[0].scatter(*Y[0], marker="*", s=180, color="black", zorder=5)
        axes[0].set(xlim=(-bounds, bounds), ylim=(-bounds, bounds), xlabel="embedding coordinate 1",
                    ylabel="embedding coordinate 2", title=f'Real saved state at step {step}; KL={trace["losses"][frame]:.4f}')
        axes[0].set_aspect("equal")
        axes[0].legend(fontsize=8, loc="upper right")
        inset = axes[0].inset_axes([0.03, 0.03, 0.31, 0.31])
        initial = positions[0]
        inset.scatter(initial[:, 0], initial[:, 1], c=labels, cmap="tab10", vmin=0, vmax=9, s=7)
        inset.set_title("Initial state (zoom)", fontsize=7)
        inset.set_aspect("equal")
        inset.tick_params(labelsize=5)
        axes[1].plot(np.arange(len(trace["full_losses"])), trace["full_losses"], color="#aab7c4", lw=1)
        axes[1].plot(np.arange(step + 1), trace["full_losses"][:step + 1], color="#0072b2", lw=2)
        axes[1].scatter(step, trace["full_losses"][step], color="#d55e00")
        axes[1].set(xlabel="actual gradient steps", ylabel="KL(P || Q)", title="Synchronized exact objective")
        fig.suptitle("Educational t-SNE: 60 synthetic points, 6D → 2D · perplexity=10\n"
                     "Arrows: exact contributions at point 0, common DISPLAY scale; updates use their sum and backtracking", fontsize=11)
    frames = list(range(len(positions))) + [len(positions) - 1] * 5
    anim = FuncAnimation(fig, update, frames=frames, interval=200, repeat=True, cache_frame_data=False)
    anim.save(Path(root) / "animations/04_tsne_optimization.gif", writer=PillowWriter(fps=5), dpi=100)
    update(len(positions) - 1)
    save_figure(fig, root, "04_tsne_final.png")
    np.savez_compressed(Path(root) / "cache/toy_optimization_states.npz", **trace, X=X, labels=labels)
    return {"files": ["animations/04_tsne_optimization.gif", "figures/04_tsne_final.png"],
            "hypothesis": "Exact KL gradient steps can lower a fixed perplexity-matched objective on this toy input.",
            "configuration": {"synthetic_shape": [60, 6], "seed": 42, "perplexity": 10,
                              "steps": settings["toy_steps"], "initial_rate": 100, "stride": 10},
            "initial_kl": float(trace["losses"][0]), "final_kl": float(trace["losses"][-1]),
            "rate_range": [float(trace["accepted_rates"].min()), float(trace["accepted_rates"].max())],
            "max_perplexity_error": float(np.max(np.abs(trace["achieved_perplexities"] - 10))),
            "frames": len(frames), "limitations": "Dense pairs; no momentum, exaggeration, scalable approximations or global-minimum guarantee."}


def kernels(root, settings):
    distances = np.linspace(0, 6, 300)
    gaussian = np.exp(-distances ** 2 / 2)
    student = 1 / (1 + distances ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for ax in axes:
        ax.plot(distances, gaussian, label="Gaussian: exp(−r²/2)", color="#0072b2")
        ax.plot(distances, student, label="Student-t: 1/(1+r²)", color="#d55e00")
        ax.set(xlabel="distance r (illustrative units)", ylabel="unnormalized kernel weight")
        ax.legend(fontsize=9)
    axes[0].set(title="Same peak; different tail weights")
    axes[1].set(yscale="log", title="Log scale reveals moderate-distance tails")
    fig.suptitle("Crowding: many moderately close high-D points compete for limited room in 2D\n"
                 "Heavy tails give separated map points appreciable affinity. These are kernels, not normalized densities.", fontsize=11)
    save_figure(fig, root, "05_crowding_kernels.png")
    return {"files": ["figures/05_crowding_kernels.png"], "kind": "analytical kernel illustration; no embedding fit or measured crowding statistic"}


def edge_traces(X, weights, symmetric=True):
    weights = np.triu(weights, 1) if symmetric else weights
    traces = []
    for low, high in zip(np.arange(0, 1, 0.2), np.arange(0.2, 1.21, 0.2)):
        a, b = np.where((weights > low) & (weights <= min(high, 1.000001)))
        if not len(a):
            continue
        xs = np.column_stack([X[a, 0], X[b, 0], np.full(len(a), np.nan)]).ravel()
        ys = np.column_stack([X[a, 1], X[b, 1], np.full(len(a), np.nan)]).ravel()
        traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=0.5 + 3 * high, color="#2785a4"),
                                opacity=0.2 + 0.6 * min(high, 1), showlegend=False, hoverinfo="skip"))
    return traces


def umap_graph(root, settings):
    stage_titles = ["1 · Original synthetic moons (exact 2D)", "2 · kNN connectivity (edges omit arrows)",
                    "3 · Outgoing local memberships from row 0", "4 · Symmetric fuzzy union (fitted graph)",
                    "5 · Final UMAP embedding (separate fit)", ""]
    fig = make_subplots(rows=2, cols=3, subplot_titles=stage_titles)
    groups, records = [], {}
    for k in (5, 15, 30):
        data = core.umap_graph_stages(k, epochs=settings["epochs"])
        X, Y = data["X"], data["Y"]
        start = len(fig.data)
        connectivity = np.zeros_like(data["graph"])
        for i, row in enumerate(data["indices"]):
            connectivity[i, row[1:]] = 1
        local = np.zeros_like(data["directed"])
        local[0] = data["directed"][0]
        panels = [(X, None), (X, np.maximum(connectivity, connectivity.T)), (X, local),
                  (X, data["graph"]), (Y, None)]
        for panel, (coordinates, matrix) in enumerate(panels):
            row, col = panel // 3 + 1, panel % 3 + 1
            if matrix is not None:
                for trace in edge_traces(X, matrix, symmetric=panel != 2):
                    if panel == 1:
                        trace.line.width = 0.6
                        trace.opacity = 0.16
                    fig.add_trace(trace, row=row, col=col)
            fig.add_trace(go.Scatter(x=coordinates[:, 0], y=coordinates[:, 1], mode="markers",
                                    marker=dict(size=7, color=[COLORS[i] for i in data["labels"]]),
                                    customdata=np.arange(len(X)), hovertemplate="row %{customdata}<extra></extra>",
                                    showlegend=False), row=row, col=col)
            if panel == 2:
                fig.add_trace(go.Scatter(x=[X[0, 0]], y=[X[0, 1]], mode="markers",
                                        marker=dict(symbol="star", color="black", size=14),
                                        showlegend=False, name="reference 0"), row=row, col=col)
        stop = len(fig.data)
        groups.append((k, start, stop))
        records[str(k)] = {"fit_seconds": data["fit_seconds"], "max_graph_error": data["max_graph_error"],
                           "warnings": data["warnings"], "scores": core.evaluate(X, Y),
                           "scheduling_prune_max_change": data["scheduling_prune_max_change"],
                           "reference_rho": float(data["rho"][0]), "reference_sigma": float(data["sigma"][0])}
    for i, trace in enumerate(fig.data):
        trace.visible = groups[0][1] <= i < groups[0][2]
    buttons = [dict(label=f"n_neighbors={k}", method="update", args=[
        {"visible": [start <= i < stop for i in range(len(fig.data))]},
        {"title.text": f"Actual graph and final fit · n_neighbors={k}"}]) for k, start, stop in groups]
    fig.update_layout(title="Actual graph and final fit · n_neighbors=5", updatemenus=[dict(buttons=buttons, x=0, y=1.17)],
                      meta={"n_neighbors": [5, 15, 30], "n": 75, "graph_matches_model": True})
    plot_page(root, "06_umap_graph.html", "Build UMAP's fuzzy neighborhood graph",
              "Observe graph construction stages, then a genuine final embedding. The control selects independently computed graphs and fits, not optimization snapshots.",
              "μ(i→j)=exp[−max(0,d−ρ_i)/σ_i]; union = μ + μᵀ − μ·μᵀ", fig,
              "75 synthetic 2D moons; Euclidean metric, seed=42, local_connectivity=1, set_op_mix_ratio=1. Exact supplied neighbors include self in column zero; self edges are then removed.",
              "Compare kNN edges, row 0's outgoing strengths, and the full union. Thickness/opacity encode binned edge weights. Hover for row IDs; compare k=5,15,30.",
              "No UMAP optimization path is animated. The displayed union is captured before finite-epoch scheduling can prune weak edges. The supplied kNN table includes self. Row-specific scales do not preserve original density; connectivity arrows are omitted.",
              "UMAP embeds locally scaled connectivity. A larger graph neighborhood changes its objective, not just its display.")
    return {"files": ["interactive/06_umap_graph.html"],
            "hypothesis": "Library local scales and fuzzy union reproduce the graph used by the fitted model.",
            "configuration": {"synthetic_shape": [75, 2], "seed": 42, "metric": "euclidean", "epochs": settings["epochs"]},
            "configurations": records}


def parameter_fits(root, settings):
    X, y, rows = core.digits_sample(settings["digits_n"])
    families = [
        ("t-SNE perplexity", "perplexity", [5, 15, 30, 50], "t-SNE", {}),
        ("UMAP n_neighbors", "n_neighbors", [5, 15, 50, 100], "UMAP", {"min_dist": 0.1}),
        ("UMAP min_dist", "min_dist", [0.0, 0.1, 0.5, 0.9], "UMAP", {"n_neighbors": 15}),
    ]
    results = []
    for title, parameter, values, method, fixed in families:
        states = []
        for value in values:
            Y, info = model_fit(root, settings, X, method, **{**fixed, parameter: value})
            states.append({"value": value, "Y": Y, "info": info})
        results.append({"title": title, "parameter": parameter, "states": states})
    return X, y, rows, results


def parameters(root, settings):
    X, y, rows, families = parameter_fits(root, settings)
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), layout="constrained")
    for axrow, family in zip(axes, families):
        for ax, state in zip(axrow, family["states"]):
            Y, scores = state["Y"], state["info"]["scores"]
            ax.scatter(Y[:, 0], Y[:, 1], c=[COLORS[c] for c in y], s=10, alpha=0.8)
            ax.set(title=f'{family["parameter"]}={state["value"]}\nT10={scores["T@10"]:.3f}; R10={scores["R@10"]:.3f}',
                   xlabel="coordinate 1", ylabel="coordinate 2")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
    fig.suptitle(f'Parameter sensitivity · same {len(X)} public digits /16 · Euclidean metric · seed=42\n'
                 'Rows: t-SNE perplexity; UMAP n_neighbors (min_dist=0.1); UMAP min_dist (n_neighbors=15). Each panel is a separate fit.', fontsize=12)
    save_figure(fig, root, "07_parameter_grid.png")

    figure = go.Figure()
    flattened = [(family, state) for family in families for state in family["states"]]
    buttons = []
    for i, (family, state) in enumerate(flattened):
        title = f'{family["title"]}={state["value"]}'
        figure.add_trace(scatter(state["Y"], y, digits=True, original_rows=rows, visible=i == 0, name=title))
        scores = state["info"]["scores"]
        buttons.append(dict(label=title, method="update", args=[{"visible": [j == i for j in range(len(flattened))]},
                       {"title.text": f'{title} · T10={scores["T@10"]:.3f}; R10={scores["R@10"]:.3f}'}]))
    figure.update_layout(title="Select a computed parameter configuration", updatemenus=[dict(buttons=buttons, x=0, y=1.16)],
                         xaxis_title="embedding coordinate 1", yaxis_title="embedding coordinate 2",
                         meta={"separate_fits": True, "states": len(flattened), "input_sha256": core.array_hash(X)})
    plot_page(root, "07_parameters.html", "Choose a computed parameter fit",
              "The dropdown changes among actual fits on one unchanged digits sample. Digit colors and original row IDs remain fixed.",
              "perplexity ≠ n_neighbors ≠ evaluation k; each controls a different scale", figure,
              f'n={len(X)}, 64 features /16, Euclidean metric; seeds=42; t-SNE init=PCA, UMAP init=spectral; iterations={settings["iterations"]}, UMAP epochs={settings["epochs"]}.',
              "Inspect lost neighbors and metric changes rather than choosing the most separated picture. All twelve configurations were computed; shared baseline fits can be loaded from cache.",
              "The control switches separate nonconvex models. Coordinates are not interpolated, and different fits have no shared spatial frame. No hyperparameter was selected on these exploratory labels.",
              "Parameter sensitivity is evidence about this representation and sample, rather than a universal ranking of settings.")

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout="constrained")
    for row, family in enumerate(families):
        values = [s["value"] for s in family["states"]]
        for col, metric in enumerate(("T", "R")):
            for k in (5, 10, 30):
                axes[row, col].plot(values, [s["info"]["scores"][f"{metric}@{k}"] for s in family["states"]],
                                    marker="o", label=f"k={k}")
            axes[row, col].set(xlabel=family["parameter"], ylabel="trustworthiness" if metric == "T" else "neighbor recall", ylim=(0, 1.02))
            axes[row, col].legend(fontsize=8)
        axes[row, 2].plot(values, [s["info"]["fit_seconds"] for s in family["states"]], color="#d55e00", marker="o")
        axes[row, 2].set(xlabel=family["parameter"], ylabel="recorded original fit seconds")
        axes[row, 0].set_title(family["title"])
    fig.suptitle("Measured neighborhood fidelity and single-run fit times\n"
                 "Same sample and original Euclidean neighborhoods. Cached fits retain original timing; JIT/warmup and fit order matter. Not a speed benchmark.", fontsize=11)
    save_figure(fig, root, "08_neighborhood_metrics.png")

    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    states = families[0]["states"]
    def update(index):
        ax.clear()
        state = states[index]
        Y, scores = state["Y"], state["info"]["scores"]
        ax.scatter(Y[:, 0], Y[:, 1], c=[COLORS[c] for c in y], s=20, alpha=0.85)
        ax.set(xlabel="embedding coordinate 1", ylabel="embedding coordinate 2",
               title=f't-SNE perplexity={state["value"]} · T10={scores["T@10"]:.3f}; R10={scores["R@10"]:.3f}')
        ax.set_aspect("equal", adjustable="box")
        fig.suptitle("Parameter sweep: separate fitted models\nNo interpolated coordinates; this is NOT an optimization trajectory", fontsize=12)
    anim = FuncAnimation(fig, update, frames=[i for i in range(4) for _ in range(4)], interval=250,
                         cache_frame_data=False)
    anim.save(Path(root) / "animations/07_parameter_sweep.gif", writer=PillowWriter(fps=4), dpi=100)
    plt.close(fig)
    records = {family["title"]: [{"value": state["value"], **state["info"]} for state in family["states"]]
               for family in families}
    return {"files": ["figures/07_parameter_grid.png", "interactive/07_parameters.html",
                      "figures/08_neighborhood_metrics.png", "animations/07_parameter_sweep.gif"],
            "hypothesis": "Changing neighborhood or packing controls alters local fidelity at different evaluation scales.",
            "data": {"kind": "public digits /16", "n": len(X), "sha256": core.array_hash(X)}, "families": records}


def geometry_traps(root, settings):
    X, labels = core.geometry_sample()
    representations = {"Original 2D": (X, None)}
    for method in ("t-SNE", "UMAP"):
        representations[method] = model_fit(root, settings, X, method)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.7), layout="constrained")
    metrics = {}
    for col, (name, (Y, _)) in enumerate(representations.items()):
        colors = [COLORS[c] for c in labels]
        axes[0, col].scatter(Y[:, 0], Y[:, 1], c=colors, s=18, alpha=0.8)
        centers = np.asarray([Y[labels == c].mean(axis=0) for c in range(3)])
        axes[0, col].scatter(centers[:, 0], centers[:, 1], marker="x", color="black", s=80)
        for c, center in enumerate(centers):
            axes[0, col].annotate(f"group {c}", center, xytext=(5, 5), textcoords="offset points", fontsize=9)
        axes[0, col].set(title=name, xlabel="original x" if col == 0 else "embedding coordinate 1",
                         ylabel="original y" if col == 0 else "embedding coordinate 2")
        axes[0, col].set_aspect("equal", adjustable="box")
        geometry = core.group_geometry(Y, labels)
        metrics[name] = geometry
        axes[1, col].axis("off")
        pairs = geometry["relative_centroid_distances"]
        radii = geometry["relative_5nn_radius"]
        text = ("Measured centroid distances / largest pair:\n"
                f"  groups 0–1: {pairs[0]:.3f}\n  groups 0–2: {pairs[1]:.3f}\n  groups 1–2: {pairs[2]:.3f}\n\n"
                "Within-group median 5NN radius / group 0:\n"
                f"  group 0: {radii[0]:.2f}\n  group 1: {radii[1]:.2f}\n  group 2: {radii[2]:.2f}")
        axes[1, col].text(0.03, 0.96, text, va="top", transform=axes[1, col].transAxes, fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.6", facecolor="#eff4f8", edgecolor="#d6e2ec"))
    fig.suptitle("Two interpretation traps: inter-group gaps and local radii\n"
                 "Synthetic 2D Gaussians: equal counts (60/group), centers x=0,4,12; std=0.15,0.5,1.1. Euclidean metric; seeds=42.", fontsize=11)
    fig.supxlabel("Distances and radii have different units across spaces. Ratios are computed separately within each representation.\n"
                  "A larger plotted gap or compact group does not establish original distance/density. Validate in the original features.", fontsize=10)
    save_figure(fig, root, "09_geometry_traps.png")
    return {"files": ["figures/09_geometry_traps.png"],
            "hypothesis": "Nonlinear maps can alter centroid-distance ratios and relative within-group neighbor radii.",
            "configuration": {"synthetic_shape": [180, 2], "group_counts": [60, 60, 60], "std": [0.15, 0.5, 1.1],
                              "centers_x": [0, 4, 12], "seed": 42, "metric": "euclidean"},
            "geometry": metrics, "fits": {name: info for name, (_, info) in representations.items() if info is not None}}


def seed_stability(root, settings):
    X, y, _ = core.digits_sample(settings["digits_n"])
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for row, method in enumerate(("t-SNE", "UMAP")):
        states = []
        for seed in (0, 21, 42):
            extra = {"init": "random"} if method == "t-SNE" else {}
            Y, info = model_fit(root, settings, X, method, seed=seed, **extra)
            states.append((seed, Y, info))
        baseline = {k: core.neighbor_indices(states[-1][1], k) for k in (10, 30)}
        records = []
        for ax, (seed, Y, info) in zip(axes[row], states):
            overlap = {f"overlap@{k}_vs_seed42": core.neighborhood_overlap(baseline[k], core.neighbor_indices(Y, k))
                       for k in (10, 30)}
            ax.scatter(Y[:, 0], Y[:, 1], c=[COLORS[c] for c in y], s=12, alpha=0.85)
            ax.set(title=f'{method} seed={seed}\nT10={info["scores"]["T@10"]:.3f}; overlap10={overlap["overlap@10_vs_seed42"]:.3f}',
                   xlabel="coordinate 1", ylabel="coordinate 2")
            ax.set_aspect("equal", adjustable="box")
            records.append({"seed": seed, **info, **overlap})
        results[method] = records
    fig.suptitle("Perturb the actual initialization, then compare neighbor sets\n"
                 "Same digits and preprocessing. t-SNE uses init=random (p=30); UMAP uses init=spectral (k=15, min_dist=0.1).", fontsize=11)
    fig.supxlabel("Layouts have separate coordinate systems. Neighbor agreement avoids alignment, but does not establish correct original-space neighbors.", fontsize=10)
    save_figure(fig, root, "10_seed_stability.png")
    return {"files": ["figures/10_seed_stability.png"],
            "hypothesis": "Actual randomized initializations can change neighborhoods even when aggregate fidelity is similar.",
            "data": {"kind": "public digits /16", "n": len(X), "sha256": core.array_hash(X)}, "fits": results}


NEIGHBOR_COLORS = ["#c7d0d9", "#d55e00", "#0072b2", "#009e73", "#161d2d"]
NEIGHBOR_NAMES = ["other", "true neighbor lost", "introduced neighbor", "shared neighbor", "reference"]


def neighbor_categories(true_neighbors, projected_neighbors, reference, n):
    categories = np.zeros(n, dtype=int)
    categories[true_neighbors[reference]] = 1
    categories[projected_neighbors[reference]] = 2
    shared = set(true_neighbors[reference]) & set(projected_neighbors[reference])
    categories[list(shared)] = 3
    categories[reference] = 4
    return categories, len(shared) / true_neighbors.shape[1]


def neighbor_errors(root, settings):
    X, y, source_rows = core.digits_sample(settings["digits_n"])
    display, display_info = model_fit(root, settings, X, "PCA")
    Y, info = model_fit(root, settings, X, "UMAP")
    true_neighbors, projected_neighbors = core.neighbor_indices(X, 10), core.neighbor_indices(Y, 10)
    recalls = [len(set(a) & set(b)) / 10 for a, b in zip(true_neighbors, projected_neighbors)]
    worst = int(np.argmin(recalls))
    selected = list(dict.fromkeys([0, 1, worst]))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["PCA used ONLY to display the original 64D rows", "UMAP: embedded neighbors defined in 2D"])
    groups, records = [], []
    for reference in selected:
        start = len(fig.data)
        categories, recall = neighbor_categories(true_neighbors, projected_neighbors, reference, len(X))
        for col, coordinates in enumerate((display, Y), 1):
            for category in range(5):
                mask = categories == category
                fig.add_trace(go.Scatter(x=coordinates[mask, 0], y=coordinates[mask, 1], mode="markers",
                                        marker=dict(color=NEIGHBOR_COLORS[category], size=14 if category == 4 else (9 if category else 5),
                                                    symbol="star" if category == 4 else "circle"),
                                        name=NEIGHBOR_NAMES[category], legendgroup=NEIGHBOR_NAMES[category], showlegend=col == 1,
                                        customdata=np.column_stack([np.arange(len(X))[mask], source_rows[mask], y[mask]]),
                                        hovertemplate="sample row=%{customdata[0]}<br>source=%{customdata[1]}<br>digit=%{customdata[2]}<extra></extra>"),
                              row=1, col=col)
        stop = len(fig.data)
        groups.append((reference, recall, start, stop))
        records.append({"sample_row": reference, "source_row": int(source_rows[reference]), "recall@10": recall,
                        "true_neighbor_rows": true_neighbors[reference].tolist(),
                        "projected_neighbor_rows": projected_neighbors[reference].tolist()})
    for i, trace in enumerate(fig.data):
        trace.visible = groups[0][2] <= i < groups[0][3]
    buttons = [dict(label=f"row {reference}" + (" (lowest recall)" if reference == worst else ""), method="update", args=[
        {"visible": [start <= i < stop for i in range(len(fig.data))]},
        {"title.text": f"Reference row {reference}: original-neighbor recall@10={recall:.1f}"}])
        for reference, recall, start, stop in groups]
    fig.update_layout(title=f"Reference row {selected[0]}: inspect true and introduced neighbors",
                      updatemenus=[dict(buttons=buttons, x=0, y=1.16)], legend=dict(orientation="h", y=-0.12),
                      meta={"reference_rows": selected, "original_dimensions": 64, "evaluation_k": 10})
    plot_page(root, "11_neighbor_errors.html", "Inspect neighbors that the map loses",
              "True neighbors are computed in the original 64D digits representation. The left PCA panel only locates those rows for inspection; the right panel shows the UMAP fit.",
              "recall_i@10 = |N_X^10(i) ∩ N_Y^10(i)| / 10", fig,
              f'{len(X)} unchanged public digits /16; Euclidean metric in both spaces; UMAP k=15, min_dist=0.1, seed=42.',
              "Select a reference. Orange points were original neighbors but were lost; blue points were introduced; green points agree. Hover for exact row IDs.",
              "The PCA display does not preserve all original distances. The lowest-recall row is deliberately selected as a diagnostic extreme, not a typical example or an independently selected test case.",
              "Good aggregate trustworthiness can coexist with individual neighbor errors. Check original-space pairs before interpreting semantic groups.")
    categories, recall = neighbor_categories(true_neighbors, projected_neighbors, worst, len(X))
    static, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, coordinates, title in zip(axes, (display, Y), ("PCA display of 64D rows (not exact distances)", "UMAP: neighbor geometry in 2D")):
        for category in range(5):
            mask = categories == category
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], c=NEIGHBOR_COLORS[category],
                       s=120 if category == 4 else (35 if category else 12), marker="*" if category == 4 else "o", label=NEIGHBOR_NAMES[category])
        ax.set(title=title, xlabel="coordinate 1", ylabel="coordinate 2")
        ax.set_aspect("equal", adjustable="box")
    axes[1].legend(fontsize=8)
    static.suptitle(f'Diagnostic extreme: sample row {worst}, recall@10={recall:.1f}\nTrue neighbors calculated in 64D, never in the PCA display; not a representative-row claim.', fontsize=11)
    save_figure(static, root, "11_neighbor_errors.png")
    return {"files": ["interactive/11_neighbor_errors.html", "figures/11_neighbor_errors.png"],
            "hypothesis": "Aggregate metrics can conceal specific lost and introduced neighbors.",
            "selected_rows": records, "worst_row": worst, "fits": {"PCA_display": display_info, "UMAP": info}}


MODULES = {
    "swiss": (swiss, "Curvature and linear projections", "Can a nonlinear fit reveal the Swiss Roll's curved structure?"),
    "digits-3d": (digits_3d, "Real three-dimensional embeddings", "How do actual PCA, t-SNE and UMAP fits use a third coordinate?"),
    "probabilities": (probabilities, "Gaussian similarities and perplexity", "How does entropy matching adapt each Gaussian bandwidth?"),
    "optimization": (optimization, "Watch the exact KL gradient descend", "How do attraction and repulsion change saved coordinates?"),
    "kernels": (kernels, "Crowding and heavy tails", "How do Student-t tails differ from Gaussian weights?"),
    "graph": (umap_graph, "Construct the fuzzy UMAP graph", "What changes between kNN edges, local scales and fuzzy union?"),
    "parameters": (parameters, "Parameter sensitivity and quantitative fidelity", "Do stronger visual gaps correspond to better neighborhood preservation?"),
    "geometry-traps": (geometry_traps, "Misleading gaps and density", "Which centroid-distance and local-radius ratios survive projection?"),
    "stability": (seed_stability, "Perturb real initializations", "How much do embedded neighbor sets agree across seeds?"),
    "neighbors": (neighbor_errors, "Inspect individual neighborhood errors", "Which original 64D neighbors were lost or introduced?"),
}


LESSONS = {
    "swiss": ("Linear projection versus a curved surface", "X in 3D → actual fitted Y in 2D",
              "Follow identical roll-position colors, inspect neighbors and rotate the original.",
              "A color-ordered map need not preserve intrinsic distances or every original Euclidean neighbor."),
    "digits-3d": ("Three real fitted coordinates", "n_components=3 for each library model",
                  "Switch method, rotate, zoom and hover for original row and digit IDs.",
                  "Independent nonlinear axes have no semantic units or shared coordinate frame."),
    "probabilities": ("Entropy controls effective neighborhood scale", "p(j|i) ∝ exp(−d²/2σ²); perplexity=2^H",
                      "Observe actual sigma-search iterations for targets 5,15,30; the star is the reference.",
                      "The sigma circle is not a cutoff; perplexity is not a hard neighbor count."),
    "optimization": ("A fixed probability objective moves the map", "KL(P||Q); attraction + repulsion = −gradient",
                     "Follow saved point positions and synchronized KL; arrows use a common display scale.",
                     "The updates use unscaled forces and backtracking. This omits production acceleration and early exaggeration."),
    "kernels": ("Heavy tails help make room for moderate-distance pairs", "Gaussian exp(−r²/2) versus Student-t 1/(1+r²)",
                "Compare the tails on linear and logarithmic axes.",
                "These are unnormalized analytical kernels, not a measured claim that crowding has disappeared."),
    "graph": ("Local scales become a fuzzy connectivity graph", "union=μ+μᵀ−μ·μᵀ",
              "Compare actual graphs and final fits at n_neighbors=5,15,30.",
              "Stages are not UMAP optimization snapshots; graph weights do not estimate original density."),
    "parameters": ("Different controls change the neighborhood problem", "T(k): rank-weighted false neighbors; R(k): retained original neighbors",
                   "Compare p=[5,15,30,50], k_graph=[5,15,50,100], min_dist=[0,.1,.5,.9].",
                   "Every panel/GIF state is a separate fit. Single recorded fit times include possible warmup; no speed ranking is established."),
    "geometry-traps": ("Visual gaps and local radii can change independently", "centroid distance / max pair; within-group r5 / group-0 r5",
                       "Compare measured ratios in exact original 2D, t-SNE and UMAP; all group counts equal 60.",
                       "A compact map group does not establish original density or prevalence. Validate against original features."),
    "stability": ("Actually perturb initialization before claiming stability", "overlap@k = shared embedded neighbors / k",
                  "Compare seeds 0,21,42; t-SNE init=random, UMAP init=spectral.",
                  "Neighbor agreement needs no alignment, but does not prove those neighbors are correct or generalize."),
    "neighbors": ("An aggregate score can hide row-level errors", "recall_i@10 = shared original/projected neighbors / 10",
                  "Inspect lost orange, introduced blue, shared green and black reference points.",
                  "Original neighbors are computed in 64D. PCA only displays the rows; the worst-recall example is deliberately extreme."),
}


def write_index(root, catalog):
    entries = []
    for key, (_, title, question) in MODULES.items():
        if key not in catalog:
            continue
        files = catalog[key]["files"]
        links = " · ".join(f'<a href="{html.escape(f)}">{html.escape(Path(f).name)}</a>' for f in files)
        images = "".join(f'<img loading="lazy" src="{html.escape(f)}" alt="{html.escape(title)}">'
                         for f in files if f.endswith((".png", ".gif")))
        concept, math, observe, mistake = LESSONS[key]
        explanation = (f'<p><strong>Concept:</strong> {html.escape(concept)}</p>'
                       f'<p><code>{html.escape(math)}</code></p><p><strong>Observe:</strong> {html.escape(observe)}</p>'
                       f'<p class="warning"><strong>Limitation:</strong> {html.escape(mistake)}</p>'
                       '<p><strong>Practical takeaway:</strong> inspect original-space relationships and '
                       'validate the intended ML or retrieval task before making decisions.</p>')
        entries.append(f'<section class="card" id="{key}"><h2>{html.escape(title)}</h2>'
                       f'<p>{html.escape(question)}</p>{explanation}<p>{links}</p>{images}</section>')
    nav = " · ".join(f'<a href="#{key}">{html.escape(MODULES[key][1])}</a>' for key in MODULES if key in catalog)
    intro = ('<h1>Neighborhoods, maps and misleading intuition</h1><p>Day 40 · t-SNE and UMAP · '
             'CPU-only offline laboratory</p><div class="card takeaway"><strong>Learning path</strong>'
             '<p>Begin with geometry and 3D fits. Follow the probability and gradient mechanisms, '
             'then compare graphs, parameters and interpretation errors. Every plotted metric is computed.</p></div>'
             '<div class="card warning"><strong>Read the map as a hypothesis</strong><p>Original-space metric: Euclidean. '
             'Synthetic data are labeled in each module; digits are real bundled public data. Separate fits do not share coordinates. '
             'Axes and gaps do not establish semantic relevance, original density or held-out model quality. '
             'Interpretation candidates require author review.</p></div>')
    text = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Day 40 · t-SNE and UMAP lab</title><style>{CSS}</style><main>' + intro +
            f'<nav>{nav}</nav>' + "".join(entries) +
            '<section class="card"><h2>Connect the display to a real task</h2>'
            '<p>For document and multimodal embeddings, investigate original-space neighbors, source quality and model versions. '
            'A visually separated map does not establish held-out Recall@k, MRR or nDCG. '
            'Validate anomaly candidates and clustering in the features used for decisions.</p>'
            '<p>Read VISUAL_GUIDE.md in the topic directory for mathematics, commands, experiment records and limitations.</p></section></main></html>')
    (Path(root) / "index.html").write_text(text, encoding="utf-8")
