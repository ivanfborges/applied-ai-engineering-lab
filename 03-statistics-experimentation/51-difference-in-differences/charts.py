"""Plotly teaching figures with a shared, statistically meaningful visual language."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from estimation import trajectories
from simulation import adoption_schedule

COLORS = {"control": "#2563eb", "treated": "#ea7c19", "cf": "#64748b",
          "truth": "#7c3aed", "effect": "#14804a", "bias": "#c73535"}


def finish(fig: go.Figure, title: str, ytitle: str = "Productivity points") -> go.Figure:
    fig.update_layout(
        template="plotly_white", title=dict(text=title, font=dict(size=20)),
        font=dict(family="Arial, sans-serif", size=14, color="#172b4d"),
        height=480, margin=dict(l=55, r=25, t=65, b=85),
        legend=dict(orientation="h", y=-0.20, x=0),
        hovermode="x unified", uirevision="did-lab",
        xaxis_title="Period relative to rollout", yaxis_title=ytitle,
    )
    return fig


def trend_chart(panel: pd.DataFrame, show_truth: bool = False, selected: int = 5) -> go.Figure:
    data = trajectories(panel)
    fig = go.Figure()
    for column, name, color, dash in [
        ("control", "Control observed", COLORS["control"], "solid"),
        ("treated", "Treated observed", COLORS["treated"], "solid"),
        ("estimated_cf", "Estimated untreated counterfactual", COLORS["cf"], "dash"),
        ("true_y0", "True untreated Y(0) — synthetic only", COLORS["truth"], "dot"),
    ]:
        if column == "true_y0" and not show_truth:
            continue
        frame = data.loc[data.index >= 0] if column == "estimated_cf" else data
        fig.add_trace(go.Scatter(x=frame.index, y=frame[column], mode="lines+markers",
                                name=name, line=dict(color=color, dash=dash, width=3),
                                hovertemplate="Period %{x}<br>%{y:.2f} points<extra>%{fullData.name}</extra>"))
    row = data.loc[selected]
    fig.add_trace(go.Scatter(x=[selected, selected], y=[row.estimated_cf, row.treated],
                            mode="lines+markers", line=dict(color=COLORS["effect"], width=7),
                            name=f"Estimated gap at {selected}: {row.estimated_gap:+.2f}"))
    fig.add_vline(x=-0.5, line_dash="dot", line_color=COLORS["cf"], annotation_text="Rollout at 0")
    return finish(fig, "Observed outcomes and the missing untreated path")


def geometry(tpre: float = 100, tpost: float = 125, cpre: float = 100,
             cpost: float = 110, stage: int = 5) -> go.Figure:
    """Six browser-animated stages; every frame keeps the same axes."""
    if stage not in range(6):
        raise ValueError("stage must be in 0..5.")
    cf = tpre + cpost - cpre
    did = tpost - cf
    low, high = min(tpre, tpost, cpre, cpost, cf), max(tpre, tpost, cpre, cpost, cf)
    pad = max(5, 0.2 * (high - low))
    names = ["1 · Before", "2 · After", "3 · Treated change", "4 · Control change",
             "5 · Subtract control change", "6 · Difference-in-Differences"]

    def traces(s):
        return [
            go.Scatter(x=[0, 1], y=[tpre, tpost if s >= 1 else None], mode="lines+markers",
                       name="Treated", line=dict(color=COLORS["treated"], width=3), marker=dict(size=12)),
            go.Scatter(x=[0, 1], y=[cpre, cpost if s >= 1 else None], mode="lines+markers",
                       name="Control", line=dict(color=COLORS["control"], width=3), marker=dict(size=9, symbol="diamond")),
            go.Scatter(x=[0, 1], y=[tpre, cf] if s >= 4 else [None, None], mode="lines+markers",
                       name="Treated pre + control change", line=dict(color=COLORS["cf"], dash="dash")),
            go.Scatter(x=[1, 1], y=[cf, tpost] if s >= 5 else [None, None], mode="lines+markers",
                       name=f"DiD = {did:+.2f}", line=dict(color=COLORS["effect"], width=7)),
        ]

    def annotations(s):
        items = []
        if s >= 2:
            items.append(dict(x=0.36, y=(tpre+tpost)/2, text=f"ΔT = {tpost-tpre:+.2f}",
                              showarrow=False, font=dict(color=COLORS["treated"]), bgcolor="white"))
        if s >= 3:
            items.append(dict(x=0.72, y=(cpre+cpost)/2, text=f"ΔC = {cpost-cpre:+.2f}",
                              showarrow=False, font=dict(color=COLORS["control"]), bgcolor="white"))
        if s >= 5:
            items.append(dict(x=1.06, y=(cf+tpost)/2, text=f"DiD {did:+.2f}",
                              showarrow=False, font=dict(color=COLORS["effect"]), xanchor="left"))
        return items

    fig = go.Figure(traces(stage))
    fig.frames = [go.Frame(name=str(s), data=traces(s),
                          layout=go.Layout(annotations=annotations(s), title_text=names[s])) for s in range(6)]
    finish(fig, names[stage])
    fig.update_layout(
        annotations=annotations(stage), xaxis=dict(tickvals=[0, 1], ticktext=["Before", "After"], range=[-0.1, 1.5]),
        yaxis_range=[low-pad, high+pad], xaxis_title="Window means",
        updatemenus=[dict(type="buttons", x=1, y=1.15, xanchor="right", buttons=[dict(
            label="▶ Derive DiD", method="animate", args=[None, dict(
                frame=dict(duration=1000, redraw=True), transition=dict(duration=250), fromcurrent=False)])])],
    )
    return fig


def coefficient_plot(summary: dict, truth: float) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=[summary["twfe"]], y=["TWFE"], mode="markers", marker=dict(size=14, color=COLORS["effect"]),
        error_x=dict(type="data", symmetric=False,
                     array=[summary["high"]-summary["twfe"]],
                     arrayminus=[summary["twfe"]-summary["low"]]),
        name="Estimate ± marginal 95% CI"))
    fig.add_vline(x=0, line_color=COLORS["cf"], line_dash="dash", annotation_text="Zero")
    fig.add_vline(x=truth, line_color=COLORS["truth"], line_dash="dot", annotation_text="True synthetic ATT")
    finish(fig, "Sampling uncertainty does not bound identification bias", "")
    fig.update_layout(xaxis_title="Effect in productivity points", height=330)
    return fig


def sampling_plot(draws: pd.DataFrame, expected: float) -> go.Figure:
    fig = go.Figure(go.Histogram(x=draws.estimate, nbinsx=30, marker_color=COLORS["control"],
                                 name="Independent-dataset DiD estimates"))
    fig.add_vline(x=float(draws.truth.iloc[0]), line_color=COLORS["truth"],
                  line_dash="dot", annotation_text="True ATT")
    fig.add_vline(x=expected, line_color=COLORS["bias"], line_dash="dash", annotation_text="DGP expected DiD")
    finish(fig, "Sampling distribution across independently seeded panels", "Number of datasets")
    fig.update_layout(xaxis_title="DiD estimate (points)", bargap=0.04, showlegend=False)
    return fig


def event_plot(events: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for pre, label, color in [(True, "Pre: change in group gap vs -1", COLORS["control"]),
                              (False, "Post: relative group-gap estimate", COLORS["treated"])]:
        frame = events.loc[(events.period < 0) == pre].loc[lambda x: ~x.reference]
        fig.add_trace(go.Scatter(x=frame.period, y=frame.estimate, mode="markers",
                                error_y=dict(type="data", array=frame.high-frame.estimate),
                                marker=dict(color=color, size=10), name=label))
    reference = int(events.loc[events.reference, "period"].iloc[0])
    fig.add_trace(go.Scatter(x=[reference], y=[0], mode="markers", marker=dict(symbol="diamond", size=11,
                            color=COLORS["cf"]), name="Reference -1: normalized to zero, no interval"))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["cf"])
    fig.add_vline(x=-0.5, line_dash="dot", annotation_text="Rollout at 0")
    return finish(fig, "Event study · pointwise 95% team-clustered intervals", "Gap change relative to period -1 (points)")


def plot_3d(panel: pd.DataFrame, selected: int = 5, show_truth: bool = False) -> go.Figure:
    data = trajectories(panel)
    fig = go.Figure()
    for column, group, label, color, dash in [
        ("control", 0, "Control observed", COLORS["control"], "solid"),
        ("treated", 1, "Treated observed", COLORS["treated"], "solid"),
        ("estimated_cf", 1, "Estimated untreated counterfactual", COLORS["cf"], "dash"),
        ("true_y0", 1, "True Y(0) — synthetic only", COLORS["truth"], "dot"),
    ]:
        if column == "true_y0" and not show_truth:
            continue
        frame = data.loc[data.index >= 0] if column == "estimated_cf" else data
        fig.add_trace(go.Scatter3d(x=frame.index, y=[group]*len(frame), z=frame[column], mode="lines+markers",
                                  name=label, line=dict(color=color, dash=dash, width=5), marker=dict(size=4),
                                  hovertemplate="Time %{x}<br>Group %{y}<br>%{z:.2f} points<extra>%{fullData.name}</extra>"))
    row = data.loc[selected]
    fig.add_trace(go.Scatter3d(x=[selected]*2, y=[1, 1], z=[row.estimated_cf, row.treated],
                              mode="lines+markers", name=f"Estimated gap {row.estimated_gap:+.2f}",
                              line=dict(color=COLORS["effect"], width=9), marker=dict(size=5)))
    levels = data[["control", "treated", "estimated_cf", "true_y0"]]
    bounds = [float(levels.min().min()-2), float(levels.max().max()+2)]
    fig.add_trace(go.Scatter3d(x=[-0.5]*4, y=[0, 0, 1, 1], z=[bounds[0], bounds[1], bounds[1], bounds[0]],
                              mode="lines", name="Rollout boundary (period 0)",
                              line=dict(color=COLORS["cf"], dash="dot")))
    finish(fig, "Optional 3D view · time × group × outcome")
    fig.update_layout(height=620, hovermode="closest", scene=dict(
        xaxis=dict(title="Period", range=[-6.5, 5.5]),
        yaxis=dict(title="Group", tickvals=[0, 1], ticktext=["Control", "Treated"], range=[-0.2, 1.2]),
        zaxis=dict(title="Productivity points", range=bounds),
        camera=dict(eye=dict(x=1.9, y=-2.2, z=1.3)), aspectratio=dict(x=1.8, y=0.8, z=1),
    ))
    return fig


def staggered_plot() -> go.Figure:
    data = adoption_schedule()
    fig = go.Figure(go.Heatmap(z=data.values, x=data.columns, y=data.index,
                              colorscale=[[0, "#e2e8f0"], [1, COLORS["treated"]]],
                              zmin=0, zmax=1, showscale=False, text=data.values,
                              texttemplate="%{text}", hovertemplate="%{y}<br>Time %{x}<br>Treated=%{z}<extra></extra>"))
    finish(fig, "Concept only · cohort-specific adoption dates", "Cohort")
    fig.update_layout(height=300)
    return fig
