"""Bounded Matplotlib/Pillow and self-contained Plotly renderers."""

from functools import lru_cache
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve

import anomaly_visual_core as core
from example import anomaly_scores
from from_scratch import RobustMAD, average_path_length, path_length


NORMAL, ANOMALY, FALSE_ALERT, ROBUST = "#2874a6", "#df7c20", "#a348a6", "#23856d"
COLORS = {"TN": NORMAL, "TP": ANOMALY, "FP": FALSE_ALERT, "FN": "#252c35"}
EXPLANATIONS = {
    "robust_statistics": (
        "Why does contamination mask unusual values?",
        "The same 100 synthetic normal latencies receive up to ten increasingly extreme observations.",
        "Mean/std and median/MAD are re-estimated on each contaminated sample; neither uses future frames.",
        "Robust scale = 1.4826 × raw MAD: a Gaussian-consistency factor, not a Gaussianity claim.",
        "The selector increases both contamination count and severity. Compare absolute classical and robust Z-scores.",
        "A 3.5 cutoff is heuristic; zero MAD raises an explicit error rather than dividing by epsilon."),
    "relationships": (
        "Can normal marginal values form an unusual pair?",
        "Blue normal observations and orange synthetic injections share latency/queue coordinates.",
        "The rectangle uses featurewise training medians/MAD; the tilted contour uses training mean/covariance.",
        "Squared Mahalanobis distance accounts for covariance; the marked star stays inside the marginal rectangle.",
        "Click legend entries to toggle the rectangle, covariance contour and points; hover the star.",
        "Covariance is estimated only on normal training data. Elliptical/Gaussian assumptions motivate contours; estimated parameters do not guarantee tail coverage."),
    "isolation_forest": (
        "How do random partitions turn into an anomaly score?",
        "The GIF replays one actual NumPy isolation tree, then averages corrected paths across trees.",
        "Every split is clipped to its parent cell; only a varying feature and a random threshold are used.",
        "s(x) = 2^(-mean corrected path / c(sample size)); terminal leaves add c(leaf size).",
        "The HTML switches real scikit-learn fits across sample/tree counts; higher -score_samples means more anomalous.",
        "One tree need not isolate an anomaly faster. Dense-region failures and absent context can remain invisible; scores are not probabilities."),
    "one_class_svm": (
        "What region does an RBF One-Class SVM learn?",
        "A nonlinear normal ring, held-out orange injections, model contours and open support-vector circles.",
        "All fields and zero boundaries come from the fitted model; preprocessing is fitted only on normal training rows.",
        "The RBF kernel exp(-gamma × squared distance) makes gamma sensitive to feature units.",
        "Choose among 18 scaled/raw × gamma × nu combinations. The boundary is -decision_function = 0.",
        "Nu bounds training errors/support-vector fractions in the formulation, not future anomaly prevalence. Changing units changes the meaning of numeric gamma."),
    "landscape": (
        "How does a detector map two features to one score?",
        "Three selectable native-score surfaces with observed synthetic points at their actual scores.",
        "MAD, Isolation Forest and the scaled SVM are fitted once on the same normal training features.",
        "Height is max robust deviation, -score_samples or -decision_function, respectively: higher means more anomalous.",
        "Switch detectors, rotate by dragging, zoom with the wheel and hover observations or surface locations.",
        "Each surface has its own native Z scale. Heights/colors are not calibrated probabilities or numerically comparable across models."),
    "thresholds": (
        "Why is an anomaly score different from an alert?",
        "Coordinated score histograms, TP/FP/FN/TN points, a PR curve and metrics all follow the selected policy.",
        "Fit on normal training rows, calibrate on separate normal rows, then evaluate held-out synthetic labels.",
        "AP measures ranking; strict score > threshold determines alerts. A calibration quantile is an empirical reference policy.",
        "Switch three calibration quantiles per detector without refitting. Two extra IF fits vary only the built-in contamination cutoff.",
        "Future FPR is not guaranteed. Identical AP can coexist with different alert counts; artificial positive prevalence affects precision."),
    "monitoring": (
        "When should a historical reference change?",
        "A chronological synthetic stream: stable operation, temporary incident, gradual benign mean shift, then a plateau.",
        "A fixed MAD reference is compared with scripted operator-approved refits and recalibration using disjoint past windows.",
        "This demonstrates covariate/data drift, not proof of concept drift in an unknown incident-label relationship.",
        "The GIF reveals the stream progressively; the HTML compares fixed/recalibrated policies. Updates activate at 300 and 330, with at least 12 steps of delay.",
        "Approval windows are illustrative, not autonomously inferred. Latest-data refits can absorb incidents; no model here establishes root cause or guarantees alert quality."),
}


def style(fig, height=790):
    fig.update_layout(template="plotly_white", height=height, font=dict(family="Arial", size=13),
                      margin=dict(l=70, r=70, t=135, b=85), hovermode="closest",
                      legend=dict(orientation="h", y=-0.14), uirevision="day41")
    return fig


def html_export(fig, root, filename, section, extra=""):
    question, *answers = EXPLANATIONS[section]
    labels = ["What you see", "Internal mechanism", "Mathematics", "Try this", "Limitation"]
    details = "".join(f"<dt>{a}</dt><dd>{escape(b)}</dd>" for a, b in zip(labels, answers))
    graph = fig.to_html(full_html=False, include_plotlyjs=True, include_mathjax=False,
                        div_id=Path(filename).stem, config=dict(responsive=True, displaylogo=False, scrollZoom=True))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(question)}</title>
<style>body{{font:16px/1.55 Arial,sans-serif;color:#243447;margin:0;background:#f1f5f9}}
main{{max-width:1250px;margin:auto;padding:26px;background:white}}h1{{font-size:29px;line-height:1.25}}
dt{{font-weight:700}}dd{{margin:0 0 8px}}.badge{{color:#596778;font-size:14px}}.plotly-graph-div{{min-width:650px}}
.chart{{overflow-x:auto}}a{{color:#21618c}}</style></head><body><main>
<p class="badge">DAY 41 · SYNTHETIC DATA · OFFLINE LEARNING LAB · AUTHOR REVIEW REQUIRED</p>
<h1>{escape(question)}</h1><dl>{details}</dl><p>{escape(extra)}</p>
<div class="chart">{graph}</div><p class="badge">Model evidence is conditional on the documented synthetic generator. No production benchmark.</p>
</main></body></html>"""
    (root / filename).write_text(page, encoding="utf-8")


def switches(fig, states, section, slider=False):
    """One coordinated control avoids independent dropdowns with conflicting state."""
    annotations = [a.to_plotly_json() for a in fig.layout.annotations]
    buttons = []
    for i, state in enumerate(states):
        visible = [j in state["indices"] for j in range(len(fig.data))]
        layout = dict(meta=dict(section=section, state=i, label=state["label"]),
                      annotations=annotations + [dict(text=state.get("summary", state["label"]),
                          x=0, y=1.12 if slider else 1.06, xref="paper", yref="paper", showarrow=False,
                          xanchor="left", font=dict(size=13))])
        layout.update(state.get("layout", {}))
        buttons.append(dict(label=state["label"], method="update", args=[dict(visible=visible), layout]))
    for j, trace in enumerate(fig.data):
        trace.visible = j in states[0]["indices"]
    fig.update_layout(**buttons[0]["args"][1])
    if slider:
        fig.update_layout(sliders=[dict(active=0, steps=buttons, y=-0.12, pad=dict(t=40),
                                       currentvalue=dict(prefix="Contamination state: "))],
                          legend=dict(orientation="h", y=-0.3), margin=dict(b=155))
    else:
        fig.update_layout(updatemenus=[dict(buttons=buttons, x=0, y=1.21, xanchor="left", yanchor="top")])


def save_gif(fig, draw, nframes, path, fps=3):
    animation = FuncAnimation(fig, draw, frames=nframes, interval=1000 / fps, repeat=True)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=85)
    plt.close(fig)


@lru_cache(maxsize=2)
def bundle(profile):
    train, cal, test, groups = core.telemetry_data(profile)
    models = core.fit_visual_detectors(train, profile)
    x, y, grid = core.mesh_for(np.vstack((train, test)), core.PROFILES[profile]["grid"])
    return dict(train=train, cal=cal, test=test, groups=groups, models=models, x=x, y=y, grid=grid)


def observations(fig, X, groups, row=None, col=None, z=None):
    for name, symbol in (("normal", "circle"), ("relationship", "diamond"), ("extreme", "x")):
        mask = groups == name
        coords = dict(x=X[mask, 0], y=X[mask, 1], name=f"Synthetic {name}",
                      marker=dict(color=NORMAL if name == "normal" else ANOMALY, size=6, symbol=symbol))
        if z is None:
            trace = go.Scatter(**coords, mode="markers", hovertemplate="latency=%{x:.1f} ms<br>queue=%{y:.1f}<extra>%{fullData.name}</extra>")
        else:
            coords["marker"]["symbol"] = "circle" if name == "normal" else "diamond"
            trace = go.Scatter3d(**coords, z=z[mask], mode="markers",
                hovertemplate="latency=%{x:.1f}<br>queue=%{y:.1f}<br>native score=%{z:.4f}<extra>%{fullData.name}</extra>")
        fig.add_trace(trace, row=row, col=col)


def robust_statistics(root, profile):
    states = core.contamination_states(core.PROFILES[profile]["robust_frames"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw=dict(width_ratios=[1.6, 1]))

    def draw(i):
        a, b = axes
        a.clear(); b.clear()
        s = states[i]
        values = s["values"]
        a.scatter(values[:100], np.linspace(0.1, 0.45, 100), s=15, c=NORMAL, label="Normal reference")
        a.scatter(values[100:], np.linspace(0.55, 0.7, len(values) - 100), s=30, c=ANOMALY, label="Added outliers")
        a.axvspan(s["mean"] - s["std"], s["mean"] + s["std"], color=FALSE_ALERT, alpha=0.16, label="Mean +/- 1 std")
        a.axvspan(s["median"] - s["robust_scale"], s["median"] + s["robust_scale"], color=ROBUST, alpha=0.25, label="Median +/- 1.4826 MAD")
        a.axvline(s["mean"], color=FALSE_ALERT, lw=2, label="Mean")
        a.axvline(s["median"], color=ROBUST, lw=2, label="Median")
        a.set(xlim=(140, 1500), ylim=(0, 1), xlabel="Synthetic request latency (ms)", yticks=[], title=f"{s['n_outliers']} added outliers | fixed normal sample")
        a.legend(loc="upper right", fontsize=8)
        counts = [j["n_outliers"] for j in states[:i + 1]]
        b.plot(counts, [j["std"] for j in states[:i + 1]], color=FALSE_ALERT, marker=".", label="Std")
        b.plot(counts, [j["robust_scale"] for j in states[:i + 1]], color=ROBUST, marker=".", label="1.4826 MAD")
        b.set(xlim=(-0.2, 10.2), ylim=(0, 330), xlabel="Added outlier count (severity also rises)", ylabel="Estimated scale (ms)", title="Contaminating the baseline can mask extremes")
        b.legend()
        fig.suptitle("01 | Mean/std move with extremes; median/MAD resist this contamination", fontsize=13)
        # A stable figure-level text artist is updated instead of accumulating captions.
        caption.set_text(f"mean={s['mean']:.1f}   median={s['median']:.1f}   std={s['std']:.1f}   raw MAD={s['raw_mad']:.1f}   robust scale={s['robust_scale']:.1f} ms\nSynthetic seed 141; two-sided robust thresholds remain heuristics; zero MAD is an explicit error.")

    caption = fig.text(0.5, 0.015, "", ha="center", fontsize=9)
    fig.subplots_adjust(bottom=0.19, top=0.85, wspace=0.3)
    save_gif(fig, draw, len(states), root / "01_robust_statistics.gif")
    plot = style(make_subplots(rows=1, cols=2, subplot_titles=["Absolute classical Z-score", "Absolute robust Z-score"]))
    controls = []
    for i, s in enumerate(states):
        first = len(plot.data)
        for col, center, scale in ((1, s["mean"], s["std"]), (2, s["median"], s["robust_scale"])):
            values = s["values"]
            plot.add_trace(go.Scatter(x=values, y=np.abs((values - center) / scale), mode="markers",
                marker=dict(color=[NORMAL] * 100 + [ANOMALY] * s["n_outliers"], size=7),
                name="Normal / added outliers", showlegend=False,
                hovertemplate="latency=%{x:.1f}<br>absolute Z=%{y:.2f}<extra></extra>"), row=1, col=col)
        controls.append(dict(label=str(i), indices=list(range(first, len(plot.data))),
            summary=f"Outliers={s['n_outliers']} | mean={s['mean']:.1f}, median={s['median']:.1f} | std={s['std']:.1f}, robust scale={s['robust_scale']:.1f} ms"))
    plot.add_hline(y=3.5, line_dash="dash", line_color=FALSE_ALERT)
    plot.update_xaxes(title_text="Latency (ms)", range=[140, 1500])
    plot.update_yaxes(title_text="Absolute standardized deviation", range=[0, 60])
    plot.update_yaxes(range=[0, 6], row=1, col=1)
    switches(plot, controls, "robust_statistics", slider=True)
    html_export(plot, root, "01_robust_zscores.html", "robust_statistics", "Blue = original normal rows; orange = added outliers. Axes use different fixed ranges: compare numeric scores. Dashed 3.5 line is a heuristic, not a guaranteed error rate.")
    return dict(files=["01_robust_statistics.gif", "01_robust_zscores.html"],
                observations={k: states[-1][k] for k in ("mean", "median", "std", "raw_mad", "robust_scale", "n_outliers")})


def relationships(root, profile):
    b = bundle(profile)
    train, test, groups = b["train"], b["test"], b["groups"]
    mad = b["models"]["MAD"]
    fig = style(go.Figure())
    observations(fig, test, groups)
    lo, hi = mad.center_ - 3 * mad.scale_, mad.center_ + 3 * mad.scale_
    fig.add_trace(go.Scatter(x=[lo[0], hi[0], hi[0], lo[0], lo[0]], y=[lo[1], lo[1], hi[1], hi[1], lo[1]],
        mode="lines", line=dict(color=ROBUST, dash="dash"), name="Marginal robust +/- 3 scales"))
    d2 = core.covariance_distance(b["grid"], train).reshape(len(b["y"]), -1)
    fig.add_trace(go.Contour(x=b["x"], y=b["y"], z=d2,
        contours=dict(start=9.21034, end=9.21034, size=1, coloring="lines"),
        line=dict(color=FALSE_ALERT, width=3), colorscale=[[0, FALSE_ALERT], [1, FALSE_ALERT]],
        showscale=False, name="Covariance contour D²=9.21", showlegend=True,
        hovertemplate="latency=%{x:.1f}<br>queue=%{y:.1f}<br>D²=%{z:.2f}<extra></extra>"))
    marked = mad.center_ + [1.5, -1.5] * mad.scale_
    joint = float(core.covariance_distance(marked[None, :], train)[0])
    fig.add_trace(go.Scatter(x=[marked[0]], y=[marked[1]], mode="markers",
        marker=dict(symbol="star", color=ANOMALY, size=18, line=dict(color="black", width=1)),
        name="Plausible marginals; unusual pair", text=[f"Marginal deviations=1.5 each; D²={joint:.2f}"], hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(meta=dict(section="relationships", joint_d2=joint))
    fig.update_xaxes(title="Latency (ms)")
    fig.update_yaxes(title="Queue depth (synthetic continuous proxy)")
    html_export(fig, root, "02_relationships.html", "relationships", "The star is an extra constructed teaching point, excluded from all metric populations. D²=9.21 is a Gaussian population-reference contour, not a finite-sample coverage guarantee.")
    return dict(files=["02_relationships.html"], observations=dict(marked_marginal_deviations=[1.5, -1.5], marked_d2=joint))


def isolation_forest(root, profile):
    X, selected = core.isolation_data()
    model, events = core.tree_trace(X)
    count = 30 if profile == "quick" else 80
    paths, means, scores = core.ensemble_paths(X, selected, count)
    ensemble_steps = np.unique(np.linspace(0, count - 1, 8 if profile == "quick" else 14).astype(int))
    fig, (ax, bar) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    _, _, grid = core.mesh_for(X, 3)
    bounds = [grid[:, 0].min(), grid[:, 0].max(), grid[:, 1].min(), grid[:, 1].max()]
    caption = fig.text(0.5, 0.02, "", ha="center", fontsize=9)

    def draw(frame):
        ax.clear(); bar.clear()
        seen = min(frame, len(events))
        ax.scatter(X[:-2, 0], X[:-2, 1], c=NORMAL, s=25, label="Normal clusters")
        ax.scatter(X[-2:, 0], X[-2:, 1], c=ANOMALY, s=50, marker="D", label="Isolated observations")
        for e in events[:seen]:
            x0, x1, y0, y1 = e["bounds"]
            if e["feature"] == 0:
                ax.plot([e["threshold"]] * 2, [y0, y1], color="#76889a", lw=1)
            else:
                ax.plot([x0, x1], [e["threshold"]] * 2, color="#76889a", lw=1)
        if 0 < frame <= len(events):
            event = events[frame - 1]
            x0, x1, y0, y1 = event["bounds"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#f6c85f", alpha=0.2))
            ax.scatter(X[event["indices"], 0], X[event["indices"], 1], facecolors="none", edgecolors="black", s=65)
            title = f"Split {frame}/{len(events)}: feature {event['feature']} <= {event['threshold']:.2f}\nParent n={len(event['indices'])}; children {len(event['left'])}/{len(event['right'])}"
        else:
            title = "Start: all observations" if frame == 0 else "Same generator; averaging independent random trees"
        ax.scatter(X[list(selected), 0], X[list(selected), 1], s=180, marker="o", facecolors="none", edgecolors=[ROBUST, ANOMALY], lw=2)
        ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="Synthetic feature 0", ylabel="Synthetic feature 1", title=title)
        ax.legend(loc="lower left", fontsize=8)
        if frame <= len(events):
            reached = [sum(i in e["indices"] for e in events[:seen]) for i in selected]
            finished = [sum(i in e["indices"] for e in events) == n for i, n in zip(selected, reached)]
            corrected = [path_length(X[i], model.trees_[0]) if done else n for i, n, done in zip(selected, reached, finished)]
            bar.bar(["Typical", "Isolated"], corrected, color=[ROBUST, ANOMALY])
            info = " / ".join(f"depth {n}" + (f" + leaf correction = {v:.2f}" if done else " (partial)") for n, v, done in zip(reached, corrected, finished))
            bar.set_title("Observed depths; correction when terminal", fontsize=10)
        else:
            k = ensemble_steps[frame - len(events) - 1]
            bar.bar(["Typical", "Isolated"], means[k], color=[ROBUST, ANOMALY])
            bar.axhline(average_path_length(len(X)), color="gray", ls="--", label="c(sample size)")
            bar.legend(fontsize=8)
            bar.set_title(f"{k + 1} trees | corrected mean path", fontsize=10)
            info = f"Original scores: typical={scores[k, 0]:.3f}, isolated={scores[k, 1]:.3f} | higher = more anomalous"
        bar.set(ylim=(0, 12), ylabel="Corrected path length (or partial depth)")
        caption.set_text(info + "\nTerminal c(n) handles unsplit rows. One tree is not a performance guarantee; score is not a probability.")
        fig.suptitle("03 | Actual recursive isolation, then ensemble averaging", fontsize=14)

    fig.subplots_adjust(bottom=0.21, top=0.81, wspace=0.3)
    save_gif(fig, draw, len(events) + 1 + len(ensemble_steps), root / "03_isolation_tree.gif", fps=3)
    x, y, grid = core.mesh_for(X, core.PROFILES[profile]["grid"])
    plot, states = style(go.Figure()), []
    for samples in (16, 40):
        for trees in (25, 100):
            fitted = IsolationForest(n_estimators=trees, max_samples=samples, random_state=core.SEED, n_jobs=1).fit(X)
            start = len(plot.data)
            plot.add_trace(go.Contour(x=x, y=y, z=(-fitted.score_samples(grid)).reshape(len(y), -1),
                colorscale="Cividis", colorbar=dict(title="-score_samples"),
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>score=%{z:.3f}<extra></extra>"))
            plot.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", name="Training rows (including isolated rows)",
                marker=dict(color=[NORMAL] * (len(X) - 2) + [ANOMALY] * 2, line=dict(color="white", width=1), size=8)))
            states.append(dict(label=f"max_samples={samples} | n_estimators={trees}", indices=list(range(start, len(plot.data))),
                summary="Library score = -score_samples; this outlier-detection demo includes isolated training rows."))
    plot.update_xaxes(title="Synthetic feature 0")
    plot.update_yaxes(title="Synthetic feature 1")
    switches(plot, states, "isolation_forest")
    html_export(plot, root, "03_isolation_scores.html", "isolation_forest", "GIF uses 40 rows and exact harmonic sums. The score map uses scikit-learn, whose tree construction and path approximation differ. An operationally faulty point inside a dense region may still score as ordinary.")
    return dict(files=["03_isolation_tree.gif", "03_isolation_scores.html"],
        observations=dict(split_events=len(events), selected_indices=list(selected),
            first_tree_corrected_paths=paths[0].tolist(), ensemble_trees=count,
            ensemble_mean_paths=means[-1].tolist(), ensemble_original_scores=scores[-1].tolist()))


def one_class_svm(root, profile):
    train, anomaly = core.ring_data(profile == "quick")
    x, y, grid = core.mesh_for(np.vstack((train, anomaly)), core.PROFILES[profile]["grid"])
    fig, states, supports = style(go.Figure()), [], []
    for config in core.svm_configurations(train):
        model = config["model"]
        field = -model.decision_function(grid).reshape(len(y), -1)
        first = len(fig.data)
        fig.add_trace(go.Contour(x=x, y=y, z=field, colorscale="RdBu_r", opacity=0.7,
            colorbar=dict(title="-decision_function"), hovertemplate="latency=%{x:.1f}<br>queue=%{y:.1f}<br>anomaly score=%{z:.4f}<extra></extra>"))
        fig.add_trace(go.Contour(x=x, y=y, z=field, contours=dict(start=0, end=0, size=1, coloring="lines"),
            line=dict(width=3), colorscale=[[0, "#111111"], [1, "#111111"]], showscale=False, showlegend=True, name="Actual zero boundary"))
        fig.add_trace(go.Scatter(x=train[:, 0], y=train[:, 1], mode="markers", name="Normal training ring",
            marker=dict(color=NORMAL, size=5), text=(-model.decision_function(train)).round(5),
            hovertemplate="latency=%{x:.1f}<br>queue=%{y:.1f}<br>direct anomaly score=%{text}<extra></extra>"))
        sv = config["support"]
        fig.add_trace(go.Scatter(x=sv[:, 0], y=sv[:, 1], mode="markers", name="Support vectors",
            marker=dict(color=ROBUST, symbol="circle-open", size=11, line=dict(width=2))))
        fig.add_trace(go.Scatter(x=anomaly[:, 0], y=anomaly[:, 1], mode="markers", name="Held-out synthetic anomalies",
            marker=dict(color=ANOMALY, symbol="diamond", size=11), text=(-model.decision_function(anomaly)).round(5),
            hovertemplate="latency=%{x:.1f}<br>queue=%{y:.1f}<br>anomaly score=%{text}<extra></extra>"))
        supports.append(dict(label=config["label"], support_vectors=len(sv), anomaly_scores=(-model.decision_function(anomaly)).tolist()))
        states.append(dict(label=config["label"], indices=list(range(first, len(fig.data))),
            summary=f"Support vectors: {len(sv)}/{len(train)} | black contour: -decision_function = 0; positive side is anomalous"))
    fig.update_xaxes(title="Latency-like feature (ms)")
    fig.update_yaxes(title="Queue-like feature (different units)")
    switches(fig, states, "one_class_svm")
    html_export(fig, root, "04_one_class_svm.html", "one_class_svm", "Gamma values are 0.03, 0.3 and 3; nu values are 0.01, 0.05 and 0.15. Scaled/raw configurations keep numeric gamma fixed to expose unit sensitivity. The finite mesh can miss narrow inlier islands at high raw-unit gamma; point hover scores are evaluated directly on the fitted model.")
    return dict(files=["04_one_class_svm.html"], observations=dict(configurations=supports))


def landscape(root, profile):
    b = bundle(profile)
    fig, states = style(go.Figure(), 850), []
    for name, model in b["models"].items():
        first = len(fig.data)
        scores = anomaly_scores(name, model, b["grid"])
        fig.add_trace(go.Surface(x=b["x"], y=b["y"], z=scores.reshape(len(b["y"]), -1), colorscale="Cividis",
            opacity=0.8, colorbar=dict(title="Native score"),
            hovertemplate="latency=%{x:.1f}<br>queue=%{y:.1f}<br>native score=%{z:.4f}<extra></extra>"))
        observations(fig, b["test"], b["groups"], z=anomaly_scores(name, model, b["test"]))
        states.append(dict(label=name, indices=list(range(first, len(fig.data))),
            summary=f"{name} | native score units; higher = more anomalous; no cross-model height comparison",
            layout={"scene.zaxis.title.text": f"{name} native anomaly score", "scene.zaxis.autorange": True}))
    fig.update_layout(scene=dict(xaxis_title="Latency (ms)", yaxis_title="Queue depth", zaxis_title="Native anomaly score",
        aspectratio=dict(x=1, y=1, z=0.7), camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))))
    switches(fig, states, "landscape")
    html_export(fig, root, "05_score_landscape.html", "landscape", "No normalization is applied. Model surfaces are separate and the Z axis rescales on selection. Surface interpolation is a rendering approximation; point scores are evaluated directly.")
    return dict(files=["05_score_landscape.html"], observations=dict(grid_shape=[len(b["y"]), len(b["x"])], points=len(b["test"])))


def thresholds(root, profile):
    b = bundle(profile)
    policies = core.threshold_policies(b["models"], b["train"], b["cal"], b["test"], b["groups"], profile)
    fig = style(make_subplots(rows=2, cols=2, specs=[[{}, {}], [{}, {"type": "table"}]],
        subplot_titles=["Calibration and test score distributions", "Test decisions vs synthetic truth", "Ranking and selected operating point", "Decision metrics"],
        horizontal_spacing=0.13, vertical_spacing=0.18), 1020)
    states = []
    truth = b["groups"] != "normal"
    for policy in policies:
        first = len(fig.data)
        scores, t = policy["scores"], policy["threshold"]
        for label, values, color in (("Calibration normal", policy["cal_scores"], "#85929e"),
                ("Test normal", scores[~truth], NORMAL), ("Test anomaly", scores[truth], ANOMALY)):
            fig.add_trace(go.Histogram(x=values, histnorm="probability", opacity=0.55, nbinsx=25,
                marker_color=color, name=label), row=1, col=1)
        fig.add_trace(go.Scatter(x=[t, t], y=[0, 0.6], mode="lines", name="Selected threshold",
            line=dict(color="black", dash="dash")), row=1, col=1)
        alerts = scores > t
        categories = np.where(truth, np.where(alerts, "TP", "FN"), np.where(alerts, "FP", "TN"))
        for label in ("TN", "TP", "FP", "FN"):
            mask = categories == label
            fig.add_trace(go.Scatter(x=b["test"][mask, 0], y=b["test"][mask, 1], mode="markers", name=label,
                marker=dict(color=COLORS[label], symbol="x" if label == "FN" else "circle", size=7),
                text=scores[mask], hovertemplate="latency=%{x:.1f}<br>queue=%{y:.1f}<br>score=%{text:.4f}<extra>%{fullData.name}</extra>"), row=1, col=2)
        precision, recall, _ = precision_recall_curve(truth, scores)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", line_shape="hv", name="PR ranking curve", line_color=NORMAL), row=2, col=1)
        fig.add_trace(go.Scatter(x=[policy["recall"]], y=[policy["precision"]], mode="markers", name="Selected policy",
            marker=dict(color=ANOMALY, size=12)), row=2, col=1)
        keys = ["ap", "threshold", "precision", "recall", "fpr", "alerts", "tp", "fp", "fn", "tn"]
        values = [f"{policy[k]:.4f}" if isinstance(policy[k], float) else str(policy[k]) for k in keys]
        fig.add_trace(go.Table(header=dict(values=["Metric", "Observed value"], fill_color="#e7edf4"),
            cells=dict(values=[[k.upper() if len(k) <= 3 else k for k in keys], values], height=24)), row=2, col=2)
        states.append(dict(label=policy["label"], indices=list(range(first, len(fig.data))),
            summary=f"AP={policy['ap']:.4f} | alerts={policy['alerts']} | TP={policy['tp']}, FP={policy['fp']}, FN={policy['fn']} | scoring stays frozen"))
    fig.update_xaxes(title_text="Native anomaly score", row=1, col=1)
    fig.update_yaxes(title_text="Within-group fraction per bin", range=[0, 0.6], row=1, col=1)
    fig.update_xaxes(title_text="Latency (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Queue depth", row=1, col=2)
    fig.update_xaxes(title_text="Recall", range=[0, 1.02], row=2, col=1)
    fig.update_yaxes(title_text="Precision", range=[0, 1.02], row=2, col=1)
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=-0.13))
    switches(fig, states, "thresholds")
    pair = policies[-2:]
    extra = (f"Same-ranking comparison: two IF configurations with contamination 0.01/0.10 have AP "
             f"{pair[0]['ap']:.6f}/{pair[1]['ap']:.6f} but {pair[0]['alerts']}/{pair[1]['alerts']} alerts "
             "at their respective built-in zero boundaries. These two extra policies use training offsets, not the normal-reference quantile protocol. FN markers are black crosses; FP markers are purple.")
    html_export(fig, root, "06_thresholds.html", "thresholds", extra)
    measurements = [{k: v for k, v in p.items() if k not in ("scores", "cal_scores")} for p in policies]
    return dict(files=["06_thresholds.html"], observations=dict(policies=measurements))


def monitoring(root, profile):
    time, latency, incident, mean = core.monitoring_data()
    policies = core.monitoring_policies(latency)
    ends = np.unique(np.linspace(120, len(time), core.PROFILES[profile]["monitoring_frames"]).astype(int))
    # Always show the incident and both activation instants, also in quick mode.
    ends = np.unique(np.r_[ends, 181, 188, 240, 300, 330, 360])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), gridspec_kw=dict(height_ratios=[1, 1, 0.8]))
    caption = fig.text(0.5, 0.015, "", ha="center", fontsize=9)

    def draw(frame):
        end = ends[frame]
        for ax in axes:
            ax.clear()
        visible = slice(0, end)
        axes[0].plot(time[visible], latency[visible], color=NORMAL, lw=1, label="Observed latency")
        axes[0].plot(time[visible], mean[visible], color=ROBUST, ls=":", label="Generator mean (display only)")
        mask = incident[:end]
        axes[0].scatter(time[:end][mask], latency[:end][mask], color=ANOMALY, s=25, label="Injected incident truth")
        axes[0].set(ylabel="Latency (ms)", ylim=(210, 455))
        axes[0].legend(loc="upper left", ncol=3, fontsize=8)
        for name, color in (("fixed", NORMAL), ("adaptive", ROBUST)):
            s, t = policies[f"{name}_scores"][:end], policies[f"{name}_threshold"][:end]
            axes[1].plot(time[:end], s, color=color, lw=1, alpha=0.8, label=f"{name} score")
            axes[1].plot(time[:end], t, color=color, ls="--", label=f"{name} threshold")
            flag = (s > t) & (time[:end] >= 120)
            axes[1].scatter(time[:end][flag], s[flag], marker="x", color=color, s=20)
        axes[1].set(ylabel="Absolute robust deviation", ylim=(0, 22))
        axes[1].legend(loc="upper left", ncol=2, fontsize=8)
        for ax in axes[:2]:
            ax.set(xlim=(0, 360))
            ax.axvspan(180, min(188, end), color=ANOMALY, alpha=0.12) if end >= 180 else None
            ax.axvspan(240, min(end, 330), color=ROBUST, alpha=0.08) if end >= 240 else None
            ax.axvspan(330, end, color="#8b9baa", alpha=0.12) if end >= 330 else None
            for update in policies["updates"]:
                if update["activate"] < end:
                    ax.axvline(update["activate"], color=ROBUST, ls=":", alpha=0.8)
        axes[1].set_xlabel("Chronological observation index | incident 180–187; drift 240–329; plateau 330+")
        bins = np.linspace(210, 455, 32)
        axes[2].hist(latency[:80], bins=bins, density=True, alpha=0.5, color=NORMAL, label="Historical fit window")
        axes[2].hist(latency[max(0, end - 40):end], bins=bins, density=True, alpha=0.5, color=ROBUST, label="Most recent 40 available rows")
        axes[2].set(xlim=(210, 455), ylim=(0, 0.06), xlabel="Latency (ms)", ylabel="Density")
        axes[2].legend(fontsize=8)
        fig.suptitle(f"07 | Monitoring through t={end - 1}: fixed vs incident-gated illustrative reference", fontsize=12)
        caption.set_text("Updates at 300/330 use only earlier approved fit/calibration windows, with a 12-step delay.\nThis is synthetic benign data drift, not an automatic diagnosis of incident vs concept drift.")

    fig.subplots_adjust(bottom=0.14, top=0.9, hspace=0.4)
    save_gif(fig, draw, len(ends), root / "07_monitoring.gif", fps=3)
    plot = style(make_subplots(rows=2, cols=1, subplot_titles=["Telemetry and synthetic truth", "Scoring and alerts"], vertical_spacing=0.18), 900)
    states, observations_record = [], {}
    for name, color in (("fixed", NORMAL), ("adaptive", ROBUST)):
        first = len(plot.data)
        s, t = policies[f"{name}_scores"], policies[f"{name}_threshold"]
        evaluation = time >= 120
        flags = (s > t) & evaluation
        plot.add_trace(go.Scatter(x=time, y=latency, mode="lines", line_color=NORMAL, name="Observed latency"), row=1, col=1)
        plot.add_trace(go.Scatter(x=time[incident], y=latency[incident], mode="markers", name="Injected incident truth", marker_color=ANOMALY), row=1, col=1)
        plot.add_trace(go.Scatter(x=time[flags], y=latency[flags], mode="markers", name="Policy alerts", marker=dict(color=FALSE_ALERT, symbol="circle-open", size=10)), row=1, col=1)
        plot.add_trace(go.Scatter(x=time[evaluation], y=s[evaluation], name=f"{name} score", mode="lines", line_color=color), row=2, col=1)
        plot.add_trace(go.Scatter(x=time[evaluation], y=t[evaluation], name="Threshold", mode="lines", line=dict(color=color, dash="dash")), row=2, col=1)
        plot.add_trace(go.Scatter(x=time[flags], y=s[flags], mode="markers", name="Flagged scores", marker=dict(color=FALSE_ALERT, symbol="x")), row=2, col=1)
        observations_record[name] = dict(alerts=int(flags.sum()), incident_detected=int(np.sum(flags & incident)),
            benign_drift_alerts=int(np.sum(flags & (time >= 240))), stable_false_alerts=int(np.sum(flags & ~incident & (time < 240))))
        states.append(dict(label=f"{name} baseline", indices=list(range(first, len(plot.data))),
            summary=f"{name}: {flags.sum()} alerts; known synthetic incident vs benign shift is generator truth, not model diagnosis"))
    plot.update_xaxes(title_text="Chronological observation index", range=[0, 360])
    plot.update_yaxes(title_text="Latency (ms)", row=1, col=1)
    plot.update_yaxes(title_text="Absolute robust deviation", row=2, col=1)
    for row in (1, 2):
        plot.add_vrect(x0=180, x1=188, fillcolor=ANOMALY, opacity=0.1, line_width=0, row=row, col=1)
        plot.add_vrect(x0=240, x1=330, fillcolor=ROBUST, opacity=0.08, line_width=0, row=row, col=1)
    switches(plot, states, "monitoring")
    html_export(plot, root, "07_monitoring.html", "monitoring", "Training: [0,80), calibration: [80,120), evaluation: [120,360). Update 300: fit [240,272), calibrate [272,288). Update 330: fit [270,302), calibrate [302,318). Operator approval is scripted; labels are display/evaluation-only.")
    return dict(files=["07_monitoring.gif", "07_monitoring.html"], observations=dict(policies=observations_record, approved_updates=policies["updates"], animation_frames=len(ends)))


SECTIONS = {name: globals()[name] for name in EXPLANATIONS}


def write_index(root, records):
    cards = []
    for name, record in records.items():
        question = EXPLANATIONS[name][0]
        links = " ".join(f'<a href="{escape(f)}">{escape(f)}</a>' for f in record["files"] if f.endswith(".html"))
        images = "".join(f'<img src="{escape(f)}" alt="{escape(question)}" loading="lazy" style="max-width:100%">' for f in record["files"] if f.endswith(".gif"))
        cards.append(f'<section><h2>{escape(question)}</h2><p>{links}</p>{images}</section>')
    (root / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Day 41 visual lab</title><body style="max-width:1150px;margin:auto;padding:24px;font:16px/1.6 Arial;color:#243447">'
        '<h1>Day 41 — Anomaly Detection</h1><p>Offline synthetic learning lab. Interpretation candidates require author review. '
        'Open each HTML for controls and explanations; each file includes Plotly.js.</p>' + "".join(cards) + '</body></html>', encoding="utf-8")
