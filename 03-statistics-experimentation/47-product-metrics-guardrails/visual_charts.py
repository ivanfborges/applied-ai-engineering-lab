"""Plotly figures: synthetic populations, observed estimates and teaching animations."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import binom, lognorm, norm

from metrics import binomial_interval, zero_incident_upper_bound

COLORS = {"control": "#0072B2", "treatment": "#E69F00"}


def finish(fig, title, x=None, y=None, height=430):
    fig.update_layout(title=title, height=height, margin=dict(l=35, r=35, t=65, b=45),
                      legend=dict(orientation="h", y=-.18), font=dict(size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    if x:
        fig.update_xaxes(title=x)
    if y:
        fig.update_yaxes(title=y)
    return fig


def conversion_chart(users):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "funnel"}, {"type": "xy"}]],
                        subplot_titles=("All assigned users stay in the denominator", "Observed conversion + exact 95% intervals"))
    for arm, color in COLORS.items():
        g = users.loc[users.arm == arm]
        k, n = int(g.converted.sum()), len(g)
        lo, hi = binomial_interval(k, n)
        fig.add_trace(go.Funnel(name=arm, y=["Eligible", "Request served", "Converted"],
                               x=[n, int(g.served_ok.sum()), k], marker_color=color,
                               textinfo="value+percent initial"), row=1, col=1)
        fig.add_trace(go.Bar(name=arm, x=[arm], y=[100*k/n], marker_color=color,
                             error_y=dict(type="data", symmetric=False, array=[100*(hi-k/n)],
                                          arrayminus=[100*(k/n-lo)]), showlegend=False), row=1, col=2)
    fig.update_yaxes(title="Conversion (%)", row=1, col=2)
    return finish(fig, "Synthetic first-request conversion")


def repeated_chart(table, truth):
    fig = go.Figure(go.Scatter(x=table.experiment, y=100*table.lift, mode="markers",
                              error_y=dict(type="data", symmetric=False, array=100*(table.high-table.lift),
                                           arrayminus=100*(table.lift-table.low)), name="Independent fixed-horizon experiment",
                              marker=dict(color=COLORS["treatment"], size=5)))
    fig.add_hline(y=100*truth, line_dash="dash", annotation_text="Generating effect")
    fig.add_hline(y=0, line_dash="dot")
    return finish(fig, "Repeated experiments vary even with identical generating probabilities",
                  "Independent repetition", "Conversion lift (pp)")


def _play(fig, names, prefix):
    fig.update_layout(updatemenus=[dict(type="buttons", showactive=False, buttons=[
        dict(label="Play", method="animate", args=[None, {"frame": {"duration": 180, "redraw": True},
                                                         "transition": {"duration": 0}, "fromcurrent": True}]),
        dict(label="Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0}}])])],
        sliders=[dict(active=0, currentvalue={"prefix": prefix}, steps=[
            dict(label=n, method="animate", args=[[n], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}])
            for n in names])])
    return fig


def uncertainty_chart(path, truth):
    def trace(part):
        return go.Scatter(x=part.n, y=100*part.lift, mode="lines+markers", name="Nested synthetic sample",
                          error_y=dict(type="data", symmetric=False, array=100*(part.high-part.lift),
                                       arrayminus=100*(part.lift-part.low)), line_color=COLORS["treatment"])
    names = [str(n) for n in path.n]
    fig = go.Figure(data=[trace(path.iloc[:1])], frames=[
        go.Frame(name=names[i], data=[trace(path.iloc[:i+1])]) for i in range(len(path))])
    fig.add_hline(y=100*truth, line_dash="dash", annotation_text="Fixed true effect")
    fig.add_hline(y=0, line_dash="dot")
    fig.update_xaxes(range=[0, float(path.n.max())*1.03])
    fig.update_yaxes(range=[100*float(path.low.min())-1, 100*float(path.high.max())+1])
    return _play(finish(fig, "Sample size and uncertainty — fixed-horizon illustrations, not sequential inference",
                        "Users per arm", "Conversion lift (pp)", 500), names, "n = ")


def retention_chart(table):
    fig = go.Figure()
    for arm, color in COLORS.items():
        g = table.loc[table.arm == arm]
        for col, dash, label in (("exact_activity", "solid", "exact-day activity"),
                                  ("rolling_to_d30", "dot", "any return [day, D30]"),
                                  ("subscription_survival", "dash", "subscription KM survival")):
            fig.add_trace(go.Scatter(x=g.day, y=100*g[col], name=f"{arm}: {label}",
                                    line=dict(color=color, dash=dash), connectgaps=False))
    return finish(fig, "Activity, finite rolling returns and subscription survival are different estimands",
                  "Elapsed day since entry", "Percent", 490)


def latency_charts(users, systems, threshold):
    distribution = make_subplots(rows=1, cols=2, subplot_titles=("All first-attempt durations", "Generating density (not a fitted KDE)"))
    ecdf = go.Figure()
    xmax = max(float(users.latency_s.max()), threshold)
    grid = np.linspace(.01, xmax, 500)
    for arm, color in COLORS.items():
        values = np.sort(users.loc[users.arm == arm, "latency_s"].to_numpy())
        cfg = systems[arm]
        distribution.add_trace(go.Histogram(x=values, histnorm="probability density", nbinsx=60,
                                           opacity=.55, marker_color=color, name=arm), row=1, col=1)
        distribution.add_trace(go.Scatter(x=grid, y=lognorm.pdf(grid, cfg.sigma, scale=cfg.median),
                                         line_color=color, name=f"{arm} population PDF"), row=1, col=2)
        ecdf.add_trace(go.Scatter(x=values, y=np.arange(1,len(values)+1)/len(values), name=arm,
                                 line=dict(color=color, shape="hv")))
        probs = np.array([.5, .9, .95, .99])
        quantiles = np.quantile(values, probs, method="inverted_cdf")
        ecdf.add_trace(go.Scatter(x=quantiles, y=probs, text=["p50", "p90", "p95", "p99"],
                                 mode="markers+text", textposition="top center", name=f"{arm} percentiles",
                                 marker=dict(color=color, size=9), hovertemplate="%{text}: %{x:.3f}s<extra></extra>"))
    distribution.update_layout(barmode="overlay")
    for col in (1, 2):
        distribution.add_vline(x=threshold, line_dash="dash", row=1, col=col)
    ecdf.add_vline(x=threshold, line_dash="dash", annotation_text="Interactive duration threshold")
    ecdf.add_hline(y=.95, line_dash="dot")
    return (finish(distribution, "Synthetic latency: median and tail spread have separate roles", "Seconds", "Density"),
            finish(ecdf, "Empirical CDF — all attempts, including failures", "First-attempt duration (seconds)", "Fraction at or below duration"))


def tail_animation(median=1.4):
    sigmas = np.linspace(.2, 1.1, 24)
    grid = np.linspace(.02, median*np.exp(norm.ppf(.995)*sigmas[-1]), 450)
    def traces(sigma):
        q = median * np.exp(norm.ppf([.5, .95, .99])*sigma)
        return [go.Scatter(x=grid, y=lognorm.pdf(grid,sigma,scale=median), mode="lines",
                           name="Population density", line_color=COLORS["treatment"]),
                go.Scatter(x=q, y=lognorm.pdf(q,sigma,scale=median), mode="markers+text",
                           text=["median", "p95", "p99"], textposition="top center", name="Percentiles",
                           marker=dict(size=10, color=COLORS["control"]))]
    names = [f"{s:.2f}" for s in sigmas]
    fig = go.Figure(data=traces(sigmas[0]), frames=[go.Frame(name=n,data=traces(s)) for n,s in zip(names,sigmas)])
    fig.update_xaxes(range=[0,float(grid.max())])
    fig.update_yaxes(range=[0,2.2/median])
    return _play(finish(fig, "Fixed median, growing tail — analytical lognormal distributions", "Seconds", "Density", 510), names, "Log-scale sigma = ")


def economics_heatmap(control_contribution, revenue, max_cost=.25):
    costs, conversion = np.linspace(0,max_cost,90), np.linspace(0,.4,90)
    value = conversion[:, None]*revenue-costs[None,:]
    fig = go.Figure(go.Heatmap(x=costs, y=100*conversion, z=value, colorscale="Cividis", colorbar=dict(title="Contribution/user")))
    if float(value.min()) <= control_contribution <= float(value.max()):
        fig.add_trace(go.Contour(x=costs,y=100*conversion,z=value,
                                contours=dict(start=control_contribution,end=control_contribution,size=1,coloring="none"),
                                line=dict(color="#CC79A7",width=3),showscale=False,name="Matches control contribution",
                                hoverinfo="skip"))
    return finish(fig, "Population economics — magenta contour matches control contribution", "Mean cost/request", "Marginal conversion (%)")


def safety_chart(users, confidence):
    fig = go.Figure()
    for arm, color in COLORS.items():
        g = users.loc[users.arm == arm]
        k,n = int(g.unsafe_detected.sum()),len(g)
        low,high = binomial_interval(k,n,confidence)
        fig.add_trace(go.Bar(x=[arm],y=[100*k/n],name=arm,marker_color=color,
                             error_y=dict(type="data",symmetric=False,array=[100*(high-k/n)],arrayminus=[100*(k/n-low)])))
    return finish(fig, "Detected unsafe outputs / all first requests — exact binomial intervals", "Arm", "Detected incidence (%)")


def rare_event_chart(p,n,confidence):
    ns = np.unique(np.geomspace(10,50000,100).astype(int))
    alpha = 1-confidence
    fig = make_subplots(rows=1,cols=2,subplot_titles=("If zero incidents are detected", "Sampling distribution of detected counts"))
    fig.add_trace(go.Scatter(x=ns,y=100*np.array([zero_incident_upper_bound(int(x),alpha) for x in ns]),
                             name="Exact one-sided bound"),row=1,col=1)
    fig.add_trace(go.Scatter(x=ns,y=100*(-np.log(alpha)/ns),name="-log(alpha)/n; ~3/n at 95%",line_dash="dot"),row=1,col=1)
    fig.add_vline(x=n,line_dash="dash",row=1,col=1)
    lower,upper = binom.ppf([.0001,.9999],n,p).astype(int)
    xs = np.arange(lower, upper+1)
    fig.add_trace(go.Bar(x=xs,y=binom.pmf(xs,n,p),name="Binomial(n,p)",marker_color=COLORS["treatment"]),row=1,col=2)
    fig.update_xaxes(type="log",title="Independent evaluation opportunities",row=1,col=1)
    fig.update_yaxes(title="Upper incidence bound (%)",row=1,col=1)
    fig.update_xaxes(title="Detected incident count",row=1,col=2)
    fig.update_yaxes(title="Probability mass",row=1,col=2)
    return finish(fig,"Rare events — a conditional zero-count bound is not proof of zero risk",height=480)


def tradeoff_chart(table):
    fig = go.Figure()
    for symbol, label, mask in (("circle","Within population guardrails",table.feasible),
                                 ("x","Violates population guardrails",~table.feasible)):
        group=table.loc[mask]
        fig.add_trace(go.Scatter3d(x=100*group.conversion,y=group.p95,z=group.cost,mode="markers",name=label,
            marker=dict(size=group["size"],symbol=symbol,color=100*group.unsafe,colorscale="Cividis",cmin=0,cmax=1.5,
                        showscale=symbol=="circle",colorbar=dict(title="Unsafe/request (%)"),opacity=.85),
            customdata=group[["name","contribution","reason","pareto"]].to_numpy(),
            hovertemplate="%{customdata[0]}<br>conversion=%{x:.2f}%<br>p95=%{y:.2f}s<br>cost=%{z:.4f}<br>contribution=%{customdata[1]:.4f}<br>%{customdata[2]}<br>Pareto=%{customdata[3]}<extra></extra>"))
    for arm, symbol in (("control","diamond"),("treatment","square")):
        g=table.loc[table.name==arm]
        fig.add_trace(go.Scatter3d(x=100*g.conversion,y=g.p95,z=g.cost,mode="markers+text",text=g.name,
                                   name=arm,marker=dict(size=12,color=COLORS[arm],symbol=symbol)))
    frontier=table.loc[table.pareto]
    fig.add_trace(go.Scatter3d(x=100*frontier.conversion,y=frontier.p95,z=frontier.cost,mode="markers",
                               name="Feasible Pareto set",marker=dict(size=5,color="#CC79A7",symbol="diamond")))
    fig.update_layout(scene=dict(xaxis_title="Conversion (%)",yaxis_title="p95 duration (s)",zaxis_title="Mean cost/request"))
    return finish(fig,"Synthetic population trade-off space — rotate, zoom and hover",height=650)
