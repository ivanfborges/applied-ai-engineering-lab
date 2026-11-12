"""Shared Plotly figures; no I/O, simulations or Streamlit state."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from statsmodels.stats.proportion import power_proportions_2indep

from power_math import required_n, sampling_model, theoretical_power, validate_design

BLUE = "#235789"
TEAL = "#12877F"
AMBER = "#CE8737"
ROSE = "#A65865"
INK = "#172E40"


def style(fig, title, x=None, y=None, height=490):
    fig.update_layout(
        title=dict(text=title, font=dict(size=19)), height=height,
        template="plotly_white", paper_bgcolor="#FAFAF7", plot_bgcolor="#FAFAF7",
        font=dict(family="Arial, sans-serif", color=INK, size=13),
        margin=dict(l=55, r=30, t=105, b=55),
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11)),
        hovermode="closest",
    )
    fig.update_xaxes(title_text=x, gridcolor="#E0E7E4", zeroline=False)
    fig.update_yaxes(title_text=y, gridcolor="#E0E7E4", zeroline=False)
    return fig


def distribution_figure(baseline, effect, n, alpha):
    model = sampling_model(baseline, effect, n, alpha)
    se0, se1, c = model["se_null"], model["se_alt"], model["critical"]
    x = np.sort(np.unique(np.r_[np.linspace(min(-4.5 * se0, effect - 4.5 * se1),
                                           max(4.5 * se0, effect + 4.5 * se1), 900), -c, c]))
    null = norm.pdf(x, 0, se0) / 100
    alt = norm.pdf(x, effect, se1) / 100
    fig = go.Figure()
    for mask, density, name, color in (
        (x <= -c, null, f"Type I regions · α={model['alpha']:.1%}", "rgba(166,88,101,0.28)"),
        (x >= c, null, "Type I right tail", "rgba(166,88,101,0.28)"),
        ((x >= -c) & (x <= c), alt, f"Type II region · β={model['beta']:.1%}", "rgba(206,135,55,0.32)"),
        (x <= -c, alt, f"Detection regions · power={model['power']:.1%}", "rgba(18,135,127,0.28)"),
        (x >= c, alt, "Detection right tail", "rgba(18,135,127,0.28)"),
    ):
        fig.add_trace(go.Scatter(x=x[mask] * 100, y=density[mask], mode="lines",
                                 line=dict(width=0), fill="tozeroy", fillcolor=color,
                                 name=name, showlegend="right tail" not in name,
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x * 100, y=null, name="H0 · common planning rate",
                             line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=x * 100, y=alt, name="H1 · assumed population effect",
                             line=dict(color=TEAL, width=3)))
    for cutoff in (-c, c):
        fig.add_vline(x=cutoff * 100, line=dict(color=ROSE, dash="dash", width=1.5))
    fig.add_vline(x=effect * 100, line=dict(color=TEAL, dash="dot"))
    return style(fig, f"Sampling distributions · {n:,} independent users per arm",
                 "Estimated treatment − control difference (percentage points)", "Density per percentage point", 540)


def power_curve_figure(baseline, effect, n, alpha, target):
    needed = required_n(baseline, effect, alpha, target)
    maximum = min(max(needed * 2.2, n * 1.2, 1000), 2_000_000)
    ns = np.unique(np.r_[np.geomspace(10, maximum, 140).astype(int), n])
    values = [theoretical_power(baseline, effect, size, alpha) for size in ns]
    fig = go.Figure(go.Scatter(x=ns, y=values, name="Calculated prospective power",
                               line=dict(color=TEAL, width=3), fill="tozeroy", fillcolor="rgba(18,135,127,0.08)"))
    for value in sorted(set((0.8, 0.9, target))):
        fig.add_hline(y=value, line_dash="dot", line_color=AMBER,
                      annotation_text=f"{value:.0%} target", annotation_position="bottom right")
    fig.add_vline(x=needed, line_dash="dash", line_color=BLUE,
                  annotation_text=f"Required n = {needed:,}", annotation_position="top left")
    fig.add_trace(go.Scatter(x=[n], y=[theoretical_power(baseline, effect, n, alpha)],
                             mode="markers", marker=dict(size=13, color=ROSE), name="Selected sample"))
    fig.update_yaxes(range=[0, 1.03], tickformat=".0%")
    return style(fig, "Power grows as uncertainty shrinks", "Independent users per arm", "Statistical power")


def surface_figure(baseline=0.10, alpha=0.05, target=0.80, max_n=60000, max_effect=0.03):
    validate_design(baseline, 0, alpha)
    max_effect = min(max_effect, (1 - baseline) * 0.95)
    ns = np.geomspace(100, max_n, 55)
    effects = np.linspace(max_effect / 80, max_effect, 45)
    nn, dd = np.meshgrid(ns, effects)
    zz = power_proportions_2indep(dd, baseline, nn, ratio=1, alpha=alpha,
                                 alternative="two-sided", return_results=False)
    colors = [[0, "#E8EBED"], [max(0, target - 0.001), "#E4B67B"],
              [target, "#55A99B"], [1, "#075D63"]]
    fig = go.Figure(go.Surface(
        x=nn, y=dd * 100, z=zz, colorscale=colors, cmin=0, cmax=1,
        name="Calculated power", colorbar=dict(title="Power", tickformat=".0%", len=0.7),
        hovertemplate="n/arm=%{x:,.0f}<br>Effect=%{y:.3f} pp<br>Power=%{z:.2%}<extra></extra>",
        contours_z=dict(show=True, start=target, end=target, size=0.05,
                        color=INK, project_z=True),
    ))
    fig.add_trace(go.Surface(x=nn, y=dd * 100, z=np.full_like(nn, target),
                             showscale=False, opacity=0.22,
                             colorscale=[[0, BLUE], [1, BLUE]], name="Target plane",
                             hovertemplate=f"Target plane: {target:.0%}<extra></extra>"))
    fig.update_layout(scene=dict(
        xaxis_title="Independent users per arm", yaxis_title="Positive absolute effect (pp)",
        zaxis=dict(title="Power", range=[0, 1], tickformat=".0%"),
        camera=dict(eye=dict(x=1.5, y=1.6, z=1.1)),
        aspectratio=dict(x=1.3, y=1, z=0.8),
    ))
    return style(fig, f"Sensitivity landscape · target plane at {target:.0%}", height=650)


def sensitivity_data(baseline, alpha, target):
    maximum = min(0.05, (1 - baseline) * 0.95)
    effects = np.geomspace(maximum / 30, maximum, 45)
    ns = np.array([required_n(baseline, effect, alpha, target) for effect in effects])
    return effects, ns


def sensitivity_figure(baseline, alpha, target):
    effects, ns = sensitivity_data(baseline, alpha, target)
    reference = ns[-1] * (effects[-1] / effects) ** 2
    fig = go.Figure(go.Scatter(x=effects * 100, y=ns, mode="lines", name="Bernoulli planning calculation",
                               line=dict(color=TEAL, width=3), customdata=effects / baseline * 100,
                               hovertemplate="Effect=%{x:.3f} pp<br>n/arm=%{y:,.0f}<br>Relative uplift=%{customdata:.2f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=effects * 100, y=reference, line=dict(color=AMBER, dash="dash"),
                             name="1 / Δ² reference · anchored at largest effect"))
    fig.update_yaxes(type="log")
    return style(fig, "Small effects demand disproportionately more data",
                 "Positive absolute effect (percentage points)", "Required users per arm · logarithmic scale")


def monte_carlo_figure(result):
    outcomes, config = result["outcomes"], result["config"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Synthetic estimated effects", "Synthetic p-values"))
    fig.add_trace(go.Histogram(x=outcomes["effect"] * 100, nbinsx=55,
                               marker_color=TEAL, name="Estimated effect"), row=1, col=1)
    fig.add_trace(go.Histogram(x=outcomes["p_value"], xbins=dict(start=0, end=1, size=0.04),
                               marker_color=BLUE, name="P-value"), row=1, col=2)
    fig.add_vline(x=config["effect"] * 100, line_dash="dash", line_color=INK, row=1, col=1)
    fig.add_vline(x=config["alpha"], line_dash="dash", line_color=ROSE, row=1, col=2)
    fig.update_xaxes(title_text="Estimated difference (pp)", row=1, col=1)
    fig.update_xaxes(title_text="Two-sided pooled Z p-value", range=[0, 1], row=1, col=2)
    return style(fig, f"Monte Carlo · {config['repetitions']:,} independent synthetic experiments", y="Experiments")


def rejection_figure(result):
    fig = go.Figure(go.Scatter(
        x=[result["rejection_rate"]], y=["Empirical rejection rate"], mode="markers",
        marker=dict(size=14, color=TEAL),
        error_x=dict(type="data", symmetric=False,
                     array=[result["ci_high"] - result["rejection_rate"]],
                     arrayminus=[result["rejection_rate"] - result["ci_low"]]),
        name="Monte Carlo estimate · 95% Wilson interval",
    ))
    fig.add_vline(x=result["analytical"], line_dash="dash", line_color=BLUE,
                  annotation_text=f"Analytical approximation {result['analytical']:.2%}")
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    return style(fig, "Simulation uncertainty around the rejection probability", "Probability", height=290)


def cluster_figure(users, conversations, icc):
    rhos = np.linspace(0, 1, 101)
    raw = users * conversations
    effective = raw / (1 + (conversations - 1) * rhos)
    fig = go.Figure(go.Scatter(x=rhos, y=effective, line=dict(color=TEAL, width=3),
                               name="Approximate independent information"))
    fig.add_hline(y=raw, line_dash="dot", line_color=BLUE, annotation_text="Raw conversation count")
    fig.add_trace(go.Scatter(x=[icc], y=[raw / (1 + (conversations - 1) * icc)],
                             mode="markers", marker=dict(size=13, color=AMBER), name="Selected ICC"))
    return style(fig, "More dependence, less independent information", "Intraclass correlation (ICC)", "Equivalent independent conversations")


def conversation_figure(users, conversations):
    shown_users, shown_m = min(users, 12), min(conversations, 12)
    fig = go.Figure()
    for user in range(shown_users):
        fig.add_trace(go.Scatter(x=np.arange(shown_m), y=np.full(shown_m, user + 1),
                                 mode="markers+lines", marker=dict(size=11, color=TEAL if user % 2 else BLUE),
                                 line=dict(color="#D9E2DF"), showlegend=False,
                                 hovertemplate=f"User {user + 1}<br>Conversation %{{x}}<extra></extra>"))
    return style(fig, "Shared users form clusters · schematic subset", "Conversation within user (zero indexed)", "User", 330)


def feasibility_figure(plan, n_required, window):
    fig = go.Figure()
    for name, values, color in (("Assigned users per arm", plan["assigned"], BLUE),
                                 ("Mature outcomes per arm", plan["mature"], TEAL)):
        fig.add_trace(go.Scatter(x=plan["days"], y=values, name=name, line=dict(color=color, width=3)))
    fig.add_hline(y=n_required, line_dash="dash", line_color=AMBER, annotation_text="Required n per arm")
    fig.add_vline(x=window, line_dash="dot", line_color=ROSE, annotation_text="Business deadline")
    return style(fig, "Assignment is earlier than outcome availability", "Calendar day from first enrollment", "Expected users per arm")


def interval_figure(rows, threshold, alpha):
    fig = go.Figure()
    for row in rows:
        excludes_zero = row["ci_low"] > 0 or row["ci_high"] < 0
        fig.add_trace(go.Scatter(
            x=[row["effect"] * 100], y=[f"{row['n']:,} / arm"], mode="markers",
            marker=dict(size=12, color=TEAL if excludes_zero else AMBER),
            error_x=dict(type="data", symmetric=False,
                         array=[(row["ci_high"] - row["effect"]) * 100],
                         arrayminus=[(row["effect"] - row["ci_low"]) * 100]),
            name=f"n={row['n']:,}", showlegend=False,
            hovertemplate=f"Synthetic estimate: {row['effect'] * 100:.3f} pp<br>CI: [{row['ci_low'] * 100:.3f}, {row['ci_high'] * 100:.3f}] pp<extra></extra>",
        ))
    fig.add_vline(x=0, line_color=INK, line_dash="dot", annotation_text="Zero effect")
    fig.add_vline(x=threshold * 100, line_color=ROSE, line_dash="dash", annotation_text="Business threshold")
    return style(fig, f"Synthetic experiments · {(1 - alpha):.0%} unpooled Wald intervals",
                 "Estimated difference (percentage points)", "Independent sample size", 430)
