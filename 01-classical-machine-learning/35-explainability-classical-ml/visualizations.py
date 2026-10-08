"""Plotly views and Matplotlib export panels for the explainability lab."""

from __future__ import annotations

from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import visual_core as core

COLORS = ("#0072B2", "#CC79A7", "#009E73", "#777777")
CLASS_COLORS = ("#0072B2", "#D55E00")
PDP_COLOR = "#E69F00"


def style(fig, title, *, height=460):
    fig.update_layout(
        template="plotly_white", title=dict(text=title, font=dict(size=19)),
        height=height, margin=dict(l=55, r=35, t=80, b=65),
        font=dict(size=13), legend=dict(orientation="h", y=-0.18),
        uirevision=title
    )
    return fig


def animation_controls(fig, labels=None):
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="left", x=0, y=1.18,
        buttons=[
            dict(label="Play", method="animate", args=[None, {
                "frame": {"duration": 300, "redraw": True}, "fromcurrent": True,
                "transition": {"duration": 150}}]),
            dict(label="Pause", method="animate", args=[[None], {
                "mode": "immediate", "frame": {"duration": 0, "redraw": False},
                "transition": {"duration": 0}}]),
        ]
    )])
    if labels:
        fig.update_layout(sliders=[dict(
            active=0, currentvalue=dict(prefix="Step: "), pad=dict(t=40),
            steps=[dict(label=label, method="animate", args=[[str(index)], {
                "mode": "immediate", "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0}}]) for index, label in enumerate(labels)]
        )])
    return fig


def dataset_scatter(lab):
    X, y = lab["X_test"], lab["y_test"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Signal and proxy", "Signal and context"))
    for panel, second in enumerate((1, 2), 1):
        for label in (0, 1):
            mask = y == label
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, second], mode="markers",
                marker=dict(color=CLASS_COLORS[label], size=6, symbol=("circle", "x")[label]),
                name=f"Observed y={label}", legendgroup=str(label), showlegend=panel == 1
            ), row=1, col=panel)
        fig.update_xaxes(title_text="signal", row=1, col=panel)
        fig.update_yaxes(title_text=core.FEATURES[second], row=1, col=panel)
    return style(fig, "Held-out synthetic observations")


def split_plot(X, y, feature, threshold, animate=False):
    stats = core.split_statistics(X, y, feature, threshold)
    fig = go.Figure()
    for label in (0, 1):
        mask = y == label
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode="markers",
            marker=dict(color=CLASS_COLORS[label], size=7, symbol=("circle", "x")[label]),
            name=f"Observed y={label}"))
    limits = [(float(X[:, j].min() - 0.2), float(X[:, j].max() + 0.2)) for j in (0, 1)]

    def boundary(value):
        return go.Scatter(
            x=[value, value] if feature == 0 else limits[0],
            y=limits[1] if feature == 0 else [value, value],
            mode="lines", line=dict(color="#222222", width=3, dash="dash"),
            name="Candidate split"
        )

    fig.add_trace(boundary(threshold))
    if animate:
        thresholds = np.linspace(*np.quantile(X[:, feature], [0.05, 0.95]), 15)
        fig.frames = [go.Frame(name=str(i), traces=[2], data=[boundary(t)],
            layout=dict(title=dict(text=f"Candidate threshold {t:.2f}; Gini decrease {core.split_statistics(X, y, feature, t)['gain']:.4f}")))
            for i, t in enumerate(thresholds)]
        animation_controls(fig)
    fig.update_xaxes(title="signal", range=limits[0], autorange=False)
    fig.update_yaxes(title="context", range=limits[1], autorange=False)
    return style(fig, f"Candidate split; weighted impurity reduction {stats['gain']:.4f}")


def split_distributions(stats):
    fig = go.Figure()
    for label in (0, 1):
        fig.add_trace(go.Bar(x=["Parent", "Left", "Right"],
            y=[int(count[label]) for count in stats["counts"]],
            name=f"Class {label}", marker_color=CLASS_COLORS[label]))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Observation count")
    return style(fig, "Class distributions before and after the candidate split", height=330)


def tree_diagram(stats, feature_name, threshold):
    fig = go.Figure(go.Scatter(
        x=[0.5, 0.2, None, 0.5, 0.8], y=[0.85, 0.2, None, 0.85, 0.2],
        mode="lines", line=dict(color="#888888"), showlegend=False
    ))
    for x, y, text in [
        (0.5, 0.85, f"{feature_name} <= {threshold:.2f}<br>Gini {stats['parent']:.3f}"),
        (0.2, 0.2, f"Left: n={sum(stats['counts'][1])}<br>Gini {stats['left']:.3f}"),
        (0.8, 0.2, f"Right: n={sum(stats['counts'][2])}<br>Gini {stats['right']:.3f}"),
    ]:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, bgcolor="#EAF1F8",
                           bordercolor="#0072B2", borderpad=12)
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1.1])
    return style(fig, "One candidate split, represented as a stump", height=270)


def importance_plot(results, metric, names=core.FEATURES):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Tree impurity importance", f"Held-out {metric} decrease"))
    colors = list(COLORS)[:len(names)]
    fig.add_trace(go.Bar(x=names, y=results["mdi"], marker_color=colors, name="MDI"),
                  row=1, col=1)
    library = results["library"]
    fig.add_trace(go.Bar(x=names, y=library.importances_mean, marker_color=colors,
                        error_y=dict(type="data", array=library.importances_std),
                        name="Permutation +/- repeat SD"), row=1, col=2)
    fig.update_yaxes(title="Normalized impurity reduction", row=1, col=1)
    fig.update_yaxes(title=f"Baseline minus shuffled {metric}", row=1, col=2, zeroline=True)
    return style(fig, "Two definitions of importance; axes have different units")


def group_plot(results, metric):
    own = results["scratch"]
    ids = [0, 1, 4]
    fig = go.Figure(go.Bar(
        x=["signal alone", "proxy alone", "signal + proxy"],
        y=own["importances_mean"][ids],
        error_y=dict(type="data", array=own["importances_std"][ids]),
        marker_color=[COLORS[0], COLORS[1], "#56B4E9"]
    ))
    fig.update_yaxes(title=f"{metric} decrease; frozen model", zeroline=True)
    return style(fig, "One shared shuffle for the pair; error bars show repeat SD", height=360)


def permutation_animation(lab, path, metric, feature):
    X, y = lab["X_test"], lab["y_test"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original: frozen rows", "Transport to a shuffled column"))
    for panel in (1, 2):
        for label in (0, 1):
            mask = y == label
            fig.add_trace(go.Scatter(
                x=X[mask, feature], y=path["probabilities"][0][mask], mode="markers",
                ids=[str(i) for i in np.flatnonzero(mask)],
                customdata=np.flatnonzero(mask),
                marker=dict(color=CLASS_COLORS[label], symbol=("circle", "x")[label], size=7),
                name=f"Observed y={label}", legendgroup=str(label), showlegend=panel == 1,
                hovertemplate="Row %{customdata}<br>feature %{x:.3f}<br>P(y=1) %{y:.3f}<extra></extra>"
            ), row=1, col=panel)
    baseline = core.metric_score(y, path["probabilities"][0], metric)
    frames = []
    for i, (rows, probability, alpha) in enumerate(zip(path["states"], path["probabilities"], path["alphas"])):
        score = core.metric_score(y, probability, metric)
        data = [go.Scatter(x=rows[y == label, feature], y=probability[y == label]) for label in (0, 1)]
        phase = "Actual row permutation" if i == len(path["states"]) - 1 else "Illustrative transport, not a permutation"
        frames.append(go.Frame(name=str(i), traces=[2, 3], data=data, layout=dict(
            title=dict(text=f"{phase}<br><sup>{metric}: baseline {baseline:.4f}, current {score:.4f}, decrease {baseline-score:.4f}</sup>")
        )))
    fig.frames = frames
    values = X[:, feature]
    for panel in (1, 2):
        fig.update_xaxes(title_text=core.FEATURES[feature],
                         range=[values.min() - 0.2, values.max() + 0.2], autorange=False, row=1, col=panel)
        fig.update_yaxes(title_text="Predicted P(y=1)", range=[-0.04, 1.04], autorange=False, row=1, col=panel)
    style(fig, f"Original {metric} {baseline:.4f}; Play transports feature values", height=520)
    return animation_controls(fig, [f"{alpha:.0%}" for alpha in path["alphas"]])


def replacement_scatter(lab, curves, feature, grid_index):
    X = curves["reference"]
    modified = core.replacement_rows(X, feature, curves["grid"][grid_index])
    counterpart = 1 if feature == 0 else 0
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original reference inputs", "Feature replaced; other inputs fixed"))
    for col, rows in enumerate((X, modified), 1):
        probability = core.positive_probability(lab["model"], rows)
        fig.add_trace(go.Scatter(
            x=rows[:, feature], y=rows[:, counterpart], mode="markers",
            marker=dict(color=probability, colorscale="Viridis", cmin=0, cmax=1, size=7,
                        showscale=col == 2, colorbar=dict(title="P(y=1)")),
            name="Reference rows"
        ), row=1, col=col)
        limits = [min(X[:, feature].min(), curves["grid"].min()) - 0.2,
                  max(X[:, feature].max(), curves["grid"].max()) + 0.2]
        fig.update_xaxes(title_text=core.FEATURES[feature], range=limits, row=1, col=col)
        fig.update_yaxes(title_text=core.FEATURES[counterpart],
                        range=[X[:, counterpart].min() - 0.2, X[:, counterpart].max() + 0.2], row=1, col=col)
    return style(fig, "Replacement changes input geometry and predictions", height=430)


def prediction_histogram(curves, grid_index):
    predictions = curves["ice"][:, grid_index]
    fig = go.Figure(go.Histogram(x=predictions, xbins=dict(start=0, end=1, size=0.05),
                                marker_color=COLORS[0], name="Replaced-row predictions"))
    fig.add_vline(x=float(predictions.mean()), line_color=PDP_COLOR, line_width=3,
                  annotation_text=f"Mean = {predictions.mean():.3f}")
    fig.update_xaxes(title="Replaced-row P(y=1)", range=[0, 1])
    fig.update_yaxes(title="Reference row count")
    return style(fig, "Average predictions, not labels or feature values", height=320)


def pdp_plot(curves, feature, grid_index=None):
    fig = go.Figure(go.Scatter(x=curves["grid"], y=curves["pdp"], mode="lines",
                              line=dict(color=PDP_COLOR, width=4), name="Mean of all reference ICE"))
    if grid_index is not None:
        fig.add_trace(go.Scatter(x=[curves["grid"][grid_index]], y=[curves["pdp"][grid_index]],
            mode="markers", marker=dict(size=14, color="#222222"), name="Selected replacement"))
    fig.update_xaxes(title=core.FEATURES[feature])
    fig.update_yaxes(title="Mean predicted P(y=1)", range=[0, 1])
    return style(fig, "Partial dependence: the average replacement response", height=360)


def ice_plot(curves, feature, selected=0, count=25, centered=False):
    ice = curves["ice"].copy()
    if centered:
        ice -= ice[:, :1]
    mean = ice.mean(axis=0)
    fig = go.Figure()
    context = curves["reference"][:, 2]
    low, high = float(context.min()), float(context.max())
    from plotly.colors import sample_colorscale
    for row in range(min(count, len(ice))):
        color = sample_colorscale("Viridis", (float(context[row]) - low) / max(high - low, 1e-9))[0]
        fig.add_trace(go.Scatter(x=curves["grid"], y=ice[row], mode="lines",
            line=dict(color=color, width=1), opacity=0.35,
            name=f"Row {row}; context={context[row]:.2f}", showlegend=False))
    fig.add_trace(go.Scatter(x=curves["grid"], y=mean, mode="lines",
                            line=dict(color=PDP_COLOR, width=5), name=f"PDP: all {len(ice)} rows"))
    fig.add_trace(go.Scatter(x=curves["grid"], y=ice[selected], mode="lines",
                            line=dict(color="#111111", width=3, dash="dash"), name=f"Selected row {selected}"))
    fig.update_xaxes(title=core.FEATURES[feature])
    fig.update_yaxes(title="Probability change from grid anchor" if centered else "P(y=1)",
                    range=[-1, 1] if centered else [0, 1])
    return style(fig, "ICE: individual responses; thin-line colors encode context")


def surface_plot(surface, kind="pdp", lab=None, selected=0):
    z = surface[kind]
    label = "Mean replacement P(y=1)" if kind == "pdp" else "Fixed-input P(y=1)"
    fig = go.Figure(go.Surface(
        x=surface["signal"], y=surface["context"], z=z, colorscale="Viridis", cmin=0, cmax=1,
        colorbar=dict(title="P(y=1)"),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        hovertemplate="signal=%{x:.2f}<br>context=%{y:.2f}<br>probability=%{z:.3f}<extra></extra>"
    ))
    if lab is not None and kind == "fixed":
        rows = lab["X_test"][:60]
        probabilities = core.positive_probability(lab["model"], rows)
        fig.add_trace(go.Scatter3d(x=rows[:, 0], y=rows[:, 2], z=probabilities, mode="markers",
            marker=dict(color=lab["y_test"][:60], colorscale=[[0, CLASS_COLORS[0]], [1, CLASS_COLORS[1]]],
                        size=3, cmin=0, cmax=1),
            name="Actual rows: their own proxy/noise"))
        row = lab["X_test"][selected]
        probability = core.positive_probability(lab["model"], row.reshape(1, -1))[0]
        fig.add_trace(go.Scatter3d(x=[row[0]], y=[row[2]], z=[probability], mode="markers",
            marker=dict(size=8, color="#D55E00", symbol="diamond"), name="Selected actual prediction"))
    fig.update_layout(scene=dict(
        xaxis_title="signal", yaxis_title="context", zaxis=dict(title=label, range=[0, 1]),
        aspectratio=dict(x=1, y=1, z=0.7)
    ))
    return style(fig, f"{label}: rotate, zoom and hover", height=570)


def surface_slices(surface, context_index):
    fig = go.Figure()
    for kind, color in (("pdp", PDP_COLOR), ("fixed", COLORS[0])):
        fig.add_trace(go.Scatter(x=surface["signal"], y=surface[kind][context_index],
            mode="lines", line=dict(color=color, width=3), name=f"{kind} at context={surface['context'][context_index]:.2f}"))
    fig.update_xaxes(title="signal")
    fig.update_yaxes(title="P(y=1)", range=[0, 1])
    return style(fig, "Same context slice: averaging versus fixed nuisance inputs", height=330)


def support_plot(lab, replacement=None):
    X = lab["X_test"]
    rho = lab["config"][1]
    grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    margin = 1.96 * np.sqrt(1 - rho**2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=rho*grid-margin, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=grid, y=rho*grid+margin, mode="lines", fill="tonexty",
        fillcolor="rgba(0,114,178,0.12)", line=dict(width=0), name="Generator's 95% proxy|signal band"))
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
        marker=dict(color=COLORS[0], size=5), name="Observed held-out inputs"))
    if replacement is not None:
        flags = core.proxy_outside_band(replacement, rho)
        for outside in (False, True):
            mask = flags == outside
            fig.add_trace(go.Scatter(x=replacement[mask, 0], y=replacement[mask, 1], mode="markers",
                marker=dict(color="#D55E00" if outside else "#009E73", symbol="x", size=9),
                name="Replacement outside band" if outside else "Replacement inside band"))
    fig.update_xaxes(title="signal")
    fig.update_yaxes(title="signal_proxy")
    return style(fig, "Known synthetic joint geometry; outside-band is a diagnostic")


def shapley_animation(game):
    steps = [(order, step) for order, path in enumerate(game["paths"]) for step in path["steps"]]
    fig = go.Figure()
    for feature in (0, 1):
        fig.add_trace(go.Scatter(x=[feature], y=[0.65], mode="markers+text",
            marker=dict(size=58, color=COLORS[feature]), text=[f"x{feature+1}"],
            textfont=dict(color="white"), showlegend=False))
    fig.add_annotation(x=0.5, y=0.2, text="Empty coalition: f(0,0)=0", showarrow=False)
    frames = []
    for i, (order, step) in enumerate(steps):
        data = [go.Scatter(marker=dict(size=58, color=COLORS[j],
                    opacity=1 if j in step["members"] else 0.2)) for j in (0, 1)]
        member_text = ",".join(f"x{j+1}" for j in step["members"]) or "empty"
        text = f"Coalition: {member_text}; v(S)={step['value']:.1f}<br>Joining contribution={step['marginal']:.1f}"
        frames.append(go.Frame(name=str(i), traces=[0, 1], data=data, layout=dict(
            title=dict(text=f"Order {order+1}: features join; attribution is averaged over BOTH orders"),
            annotations=[dict(x=0.5, y=0.2, text=text, showarrow=False)]
        )))
    fig.frames = frames
    fig.update_xaxes(visible=False, range=[-0.5, 1.5], autorange=False)
    fig.update_yaxes(visible=False, range=[0, 1.1], autorange=False)
    style(fig, "Two orders: marginal contributions, then average", height=340)
    return animation_controls(fig, [f"Order {order+1}, step {i%3}" for i, (order, _) in enumerate(steps)])


def waterfall_plot(result, selected=0):
    values = result["values"][selected]
    baseline = float(result["baseline"][selected])
    fig = go.Figure(go.Waterfall(
        x=["Background", *core.FEATURES, "Prediction"], measure=["absolute", *["relative"]*4, "total"],
        y=[baseline, *values, 0], text=[f"{baseline:.3f}", *[f"{v:+.3f}" for v in values],
            f"{result['predictions'][selected]:.3f}"], textposition="outside",
        increasing=dict(marker_color="#D55E00"), decreasing=dict(marker_color=COLORS[0]),
        totals=dict(marker_color="#777777"), connector=dict(line=dict(color="#999999"))
    ))
    lower = min(0, float((baseline + np.cumsum(values)).min())) - 0.08
    upper = max(1, float((baseline + np.cumsum(values)).max())) + 0.08
    fig.update_yaxes(title="Positive-class probability units", range=[lower, upper])
    return style(fig, f"Row {selected}: baseline + contributions = P(y=1)")


def beeswarm_plot(result):
    values, rows = result["values"], result["rows"]
    ranking = np.argsort(np.abs(values).mean(axis=0))
    fig = go.Figure()
    for rank, feature in enumerate(ranking):
        contribution = values[:, feature]
        # Pack nearby contributions symmetrically; jitter has no numerical meaning.
        span = max(float(np.ptp(contribution)), 1e-6)
        bins = np.floor((contribution - contribution.min()) / span * 18).astype(int)
        offsets = np.zeros(len(rows))
        for group in np.unique(bins):
            ids = np.flatnonzero(bins == group)
            offsets[ids] = (np.arange(len(ids)) - (len(ids)-1)/2) * min(0.09, 0.65/max(len(ids), 1))
        feature_value = rows[:, feature]
        normalized = (feature_value - feature_value.min()) / max(float(np.ptp(feature_value)), 1e-9)
        fig.add_trace(go.Scatter(x=contribution, y=rank+offsets, mode="markers",
            marker=dict(color=normalized, colorscale="RdBu_r", cmin=0, cmax=1, size=7,
                        showscale=rank == len(ranking)-1,
                        colorbar=dict(title="Feature value", tickvals=[0, 1], ticktext=["Low", "High"])),
            customdata=feature_value, name=core.FEATURES[feature], showlegend=False,
            hovertemplate="Contribution %{x:.3f}<br>Feature value %{customdata:.3f}<extra></extra>"
        ))
    fig.add_vline(x=0, line_color="#888888")
    fig.update_yaxes(tickvals=list(range(4)), ticktext=[core.FEATURES[j] for j in ranking],
                     title="Ranked by mean |SHAP| on these explained rows")
    fig.update_xaxes(title="Contribution to P(y=1) relative to the background")
    return style(fig, "SHAP distribution: each dot is an explained observation")


def dependence_plot(result, feature=0, color_feature=2):
    fig = go.Figure(go.Scatter(x=result["rows"][:, feature], y=result["values"][:, feature],
        mode="markers", marker=dict(color=result["rows"][:, color_feature], colorscale="Viridis",
                                    size=8, showscale=True, colorbar=dict(title=core.FEATURES[color_feature]))))
    fig.update_xaxes(title=f"Observed {core.FEATURES[feature]}")
    fig.update_yaxes(title=f"{core.FEATURES[feature]} SHAP in probability units", zeroline=True)
    return style(fig, "Attribution versus actual feature values; color suggests heterogeneity")


def cardinality_plot(result):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Impurity credit on random labels", "Held-out AUC decrease"))
    names = ["Continuous noise", "Binary noise"]
    fig.add_trace(go.Bar(x=names, y=result["mdi"], name="MDI", marker_color=[COLORS[0], COLORS[3]]), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=result["pi"], name="Permutation",
        error_y=dict(type="data", array=result["pi_std"]), marker_color=[COLORS[0], COLORS[3]]), row=1, col=2)
    fig.update_yaxes(title_text="Normalized split credit", row=1, col=1)
    fig.update_yaxes(title_text="AUC decrease", row=1, col=2)
    return style(fig, "Deliberate random-label control: many split candidates can fit chance", height=370)


def leakage_plot(result):
    names = [*core.FEATURES, "post_outcome (INVALID)"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Normalized impurity importance", "INVALID model: held-out permutation"))
    fig.add_trace(go.Bar(x=names, y=[*result["clean_mdi"], 0], name="Leakage-free model", marker_color=COLORS[0]), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=result["invalid_mdi"], name="Deliberately invalid model", marker_color="#D55E00"), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=result["invalid_pi"], name="INVALID model AUC decrease", marker_color="#D55E00"), row=1, col=2)
    fig.update_yaxes(title_text="MDI", row=1, col=1)
    fig.update_yaxes(title_text="AUC decrease", row=1, col=2)
    return style(fig, "A convincing explanation cannot repair unavailable input information")


def matplotlib_ice(curves, feature=0, count=None, show_mean=True):
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    n = len(curves["ice"]) if count is None else count
    for row in range(n):
        ax.plot(curves["grid"], curves["ice"][row], color=COLORS[0], alpha=0.22, linewidth=1)
    if show_mean:
        ax.plot(curves["grid"], curves["ice"].mean(axis=0), color=PDP_COLOR, linewidth=3,
                label=f"PDP: mean of all {len(curves['ice'])} reference rows")
    ax.set(xlabel=core.FEATURES[feature], ylabel="Predicted P(y=1)", ylim=(0, 1),
           xlim=(curves["grid"].min(), curves["grid"].max()), title="ICE responses and their average PDP")
    ax.plot([], [], color=COLORS[0], alpha=0.6, label="Individual ICE")
    ax.legend(loc="upper left")
    return fig


def matplotlib_waterfall(result, selected=0):
    baseline, values = result["baseline"][selected], result["values"][selected]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(0, baseline, color="#777777")
    current = baseline
    for index, value in enumerate(values, 1):
        ax.bar(index, abs(value), bottom=min(current, current+value),
               color="#D55E00" if value >= 0 else COLORS[0])
        ax.text(index, max(current, current+value)+0.025, f"{value:+.3f}", ha="center", fontsize=10)
        ax.plot([index-1, index], [current, current], color="#888888", linewidth=1)
        current += value
    ax.bar(5, result["predictions"][selected], color="#777777")
    ax.set_xticks(range(6), ["Background", *core.FEATURES, "Prediction"])
    ax.set(ylabel="Positive-class probability units",
           title=f"Row {selected}: SHAP additive reconstruction",
           ylim=(min(-0.05, float((baseline+np.cumsum(values)).min())-0.1),
                 max(1.05, float((baseline+np.cumsum(values)).max())+0.1)))
    return fig


def matplotlib_importance(results, metric="ROC-AUC"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    axes[0].bar(core.FEATURES, results["mdi"], color=COLORS)
    axes[0].set(title="MDI: normalized split credit", ylabel="Normalized impurity reduction")
    axes[1].bar(core.FEATURES, results["library"].importances_mean, color=COLORS,
                yerr=results["library"].importances_std, capsize=4)
    axes[1].axhline(0, color="#555555", linewidth=1)
    axes[1].set(title=f"Held-out {metric} permutation", ylabel=f"{metric} decrease +/- repeat SD")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    return fig


def png_bytes(fig):
    data = BytesIO()
    fig.savefig(data, format="png", dpi=200)
    plt.close(fig)
    return data.getvalue()