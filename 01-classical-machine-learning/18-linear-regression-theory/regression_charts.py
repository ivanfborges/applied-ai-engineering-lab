"""Plotly figures shared by the interactive lab and offline HTML exports."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import regression_math as rm

BLUE = "#2563eb"
TEAL = "#0f766e"
ORANGE = "#d97706"
RED = "#be123c"
COLORS = [BLUE, TEAL, ORANGE, RED]


def style(fig, title="", x="x", y="y", height=470):
    fig.update_layout(
        title=dict(text=title, x=.02), template="plotly_white",
        colorway=COLORS, height=height,
        margin=dict(l=55, r=25, t=65, b=65),
        legend=dict(orientation="h", y=-.18),
        font=dict(family="Arial, sans-serif", size=13),
        hovermode="closest",
    )
    fig.update_xaxes(title_text=x)
    fig.update_yaxes(title_text=y)
    return fig


def trace(x, y, name, color=BLUE, mode="lines", **kwargs):
    return go.Scatter(x=x, y=y, name=name, mode=mode,
                      line=dict(color=color), marker=dict(color=color, size=6), **kwargs)


def segments(x, observed, predicted, name, color):
    xs, ys = [], []
    for a, b, c in zip(x, observed, predicted):
        xs.extend([a, a, None])
        ys.extend([b, c, None])
    return trace(xs, ys, name, color)


def line_chart(x, y, beta, title="A line estimated from noisy observations",
               true_beta=None, residual_lines=False, grid=None):
    grid = np.linspace(min(x), max(x), 180) if grid is None else grid
    pred = rm.design_matrix(x)@beta
    fig = go.Figure()
    if residual_lines:
        for label, mask, color in (("Positive residual", y >= pred, TEAL),
                                   ("Negative residual", y < pred, RED)):
            fig.add_trace(segments(x[mask], y[mask], pred[mask], label, color))
        fig.add_trace(trace(x, pred, "Predictions", ORANGE, "markers",
                            marker_symbol="x"))
    fig.add_trace(trace(x, y, "Observed", BLUE, "markers"))
    fig.add_trace(trace(grid, beta[0]+beta[1]*grid, "Fitted mean", TEAL))
    if true_beta is not None:
        fig.add_trace(trace(grid, true_beta[0]+true_beta[1]*grid,
                            "Generating mean", ORANGE, line_dash="dash"))
    return style(fig, title, "Predictor x", "Target y")


def diagnostics(y, pred):
    e = y-pred
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Actual vs predicted", "Residual vs predicted", "Residual distribution"))
    fig.add_trace(trace(pred, y, "Observed", BLUE, "markers"), row=1, col=1)
    limits = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    fig.add_trace(trace(limits, limits, "Ideal prediction", TEAL), row=1, col=1)
    fig.add_trace(trace(pred, e, "Residuals", RED, "markers"), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", row=1, col=2)
    fig.add_trace(go.Histogram(x=e, nbinsx=18, name="Count", marker_color=BLUE),
                  row=1, col=3)
    style(fig, "Different views of the same training errors", "", "", 450)
    for col, x, yaxis in ((1, "Predicted", "Actual"), (2, "Predicted", "Residual"),
                         (3, "Residual", "Count")):
        fig.update_xaxes(title_text=x, row=1, col=col)
        fig.update_yaxes(title_text=yaxis, row=1, col=col)
    return fig


def loss_axes(x, y, path=None):
    beta = rm.fit_ols(x, y)["beta"]
    low, high = beta-4, beta+4
    if path is not None:
        # Keep an unstable trajectory from flattening the entire loss bowl.
        visible = path[np.all(np.abs(path-beta) < 15, axis=1)]
        if len(visible):
            low = np.minimum(low, visible.min(axis=0)-.5)
            high = np.maximum(high, visible.max(axis=0)+.5)
    a, b = np.linspace(low[0], high[0], 65), np.linspace(low[1], high[1], 65)
    return a, b, rm.loss_grid(x, y, a, b), beta


def loss_figures(x, y, true_beta=(2., 1.5)):
    a, b, z, beta = loss_axes(x, y)
    surface = go.Figure(go.Surface(x=a, y=b, z=z, colorscale="Blues",
                                  colorbar=dict(title="RSS")))
    contour = go.Figure(go.Contour(x=a, y=b, z=z, colorscale="Blues",
                                  contours=dict(showlabels=True), showscale=False))
    for label, point, color in (("OLS / global minimum", beta, RED),
                                ("Generating coefficients", true_beta, ORANGE)):
        value = rm.scores(y, rm.design_matrix(x)@point)["RSS"]
        surface.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[value],
                          mode="markers", name=label, marker=dict(color=color, size=6)))
        contour.add_trace(trace([point[0]], [point[1]], label, color, "markers"))
    style(surface, "OLS minimizes RSS over coefficient space", height=540)
    surface.update_layout(scene=dict(xaxis_title="Intercept beta0",
                                     yaxis_title="Slope beta1", zaxis_title="RSS"))
    style(contour, "Contours of equal RSS", "Intercept beta0", "Slope beta1")
    return surface, contour


def descent_animation(x, y, result):
    path = result["path"]
    a, b, z, optimum = loss_axes(x, y, path)
    grid = np.linspace(x.min(), x.max(), 100)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Line in data space", "Path in parameter space"))
    fig.add_trace(trace(x, y, "Observed", BLUE, "markers"), row=1, col=1)
    fig.add_trace(trace(grid, path[0, 0]+path[0, 1]*grid, "Current line", ORANGE), row=1, col=1)
    fig.add_trace(go.Contour(x=a, y=b, z=z, showscale=False, colorscale="Blues"),
                  row=1, col=2)
    fig.add_trace(trace([optimum[0]], [optimum[1]], "OLS", TEAL, "markers"), row=1, col=2)
    fig.add_trace(trace(path[:1, 0], path[:1, 1], "Gradient descent", RED, "lines+markers"),
                  row=1, col=2)
    indices = np.unique(np.linspace(0, len(path)-1, min(70, len(path))).astype(int))
    fig.frames = [
        go.Frame(name=str(i), data=[
            trace(grid, path[i, 0]+path[i, 1]*grid, "Current line", ORANGE),
            trace(path[:i+1, 0], path[:i+1, 1], "Gradient descent", RED, "lines+markers"),
        ], traces=[1, 4]) for i in indices
    ]
    style(fig, "Play the same optimizer in two spaces", "", "", 500)
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", range=[min(y.min(), -8), max(y.max(), 12)], row=1, col=1)
    fig.update_xaxes(title_text="Intercept beta0", range=[a.min(), a.max()], row=1, col=2)
    fig.update_yaxes(title_text="Slope beta1", range=[b.min(), b.max()], row=1, col=2)
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0, y=1.18, buttons=[
            dict(label="Play", method="animate", args=[None, dict(
                frame=dict(duration=100, redraw=True), transition=dict(duration=0),
                fromcurrent=True)]),
            dict(label="Pause", method="animate", args=[[None], dict(
                mode="immediate", frame=dict(duration=0, redraw=False))]),
        ])],
        sliders=[dict(active=0, y=-.12, x=.1, len=.85, currentvalue=dict(prefix="Step "), steps=[
            dict(label=str(i), method="animate", args=[[str(i)], dict(
                mode="immediate", frame=dict(duration=0, redraw=True),
                transition=dict(duration=0))]) for i in indices
        ])],
    )
    fig.update_layout(height=620, margin=dict(l=55, r=25, t=90, b=150),
                      legend=dict(orientation="h", y=-.4))
    return fig


def plane(X, y, beta):
    axes = [np.linspace(X[:, j].min(), X[:, j].max(), 25) for j in (0, 1)]
    a, b = np.meshgrid(*axes)
    fig = go.Figure(go.Surface(x=a, y=b, z=beta[0]+beta[1]*a+beta[2]*b,
                               opacity=.6, colorscale="Blues", showscale=False,
                               name="Fitted plane"))
    fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=y, mode="markers",
                              name="Observed", marker=dict(size=4, color=RED)))
    style(fig, "An OLS plane in feature–target space", height=590)
    fig.update_layout(scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="Target y"))
    return fig


def projection_chart(fit):
    D, y, predicted, residual = fit["design"], fit["y"], fit["pred"], fit["residual"]
    q = np.linalg.qr(D)[0]
    span = max(np.linalg.norm(y), 2.)
    a, b = np.meshgrid(np.linspace(-span, span, 8), np.linspace(-span, span, 8))
    points = a[..., None]*q[:, 0]+b[..., None]*q[:, 1]
    fig = go.Figure(go.Surface(x=points[..., 0], y=points[..., 1], z=points[..., 2],
                              opacity=.22, showscale=False, colorscale="Blues",
                              name="Column space of D"))
    for label, start, end, color in (
        ("Target y", np.zeros(3), y, BLUE),
        ("Projection y_hat", np.zeros(3), predicted, TEAL),
        ("Residual e", predicted, y, RED),
    ):
        fig.add_trace(go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]],
                                  z=[start[2], end[2]], mode="lines+markers",
                                  name=label, line=dict(color=color, width=7),
                                  marker=dict(size=4)))
    if np.linalg.norm(residual) > 1e-10:
        u, v = q[:, 0]*.35, residual/np.linalg.norm(residual)*.35
        square = np.array([predicted+u, predicted+u+v, predicted+v])
        fig.add_trace(go.Scatter3d(x=square[:, 0], y=square[:, 1], z=square[:, 2],
                                  mode="lines", name="Right angle", line=dict(color=ORANGE, width=5)))
    style(fig, "Three observations: projection in response space", height=600)
    fig.update_layout(scene=dict(aspectmode="cube", camera=dict(eye=dict(x=-1.8, y=-.5, z=.8)),
                                 xaxis_title="Observation 1",
                                 yaxis_title="Observation 2", zaxis_title="Observation 3"))
    return fig


def curvature_chart(x, y, linear, quadratic):
    grid = np.linspace(-3, 3, 180)
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Only x included", "x and x² included", "Residuals vs x", "Residuals vs x"))
    for col, fit, features in (
        (1, linear, grid[:, None]),
        (2, quadratic, np.column_stack((grid, grid*grid))),
    ):
        fig.add_trace(trace(x, y, "Observed", BLUE, "markers", showlegend=col == 1),
                      row=1, col=col)
        fig.add_trace(trace(grid, rm.design_matrix(features)@fit["beta"], "Fitted", TEAL,
                            showlegend=col == 1), row=1, col=col)
        fig.add_trace(trace(x, fit["residual"], "Residual", RED, "markers",
                            showlegend=col == 1), row=2, col=col)
        fig.add_hline(y=0, row=2, col=col, line_dash="dash")
    style(fig, "A nonlinear relationship can use a linear parameter model", "x", "", 700)
    fig.update_yaxes(title_text="Target y", row=1)
    fig.update_yaxes(title_text="Residual", row=2)
    fig.update_yaxes(matches="y3", row=2, col=2)
    return fig


def variance_chart(x, yconstant, ychanging):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Constant conditional variance",
                                                       "Increasing conditional variance"))
    for col, y in enumerate((yconstant, ychanging), 1):
        fitted = rm.fit_ols(x, y)
        fig.add_trace(trace(fitted["pred"], fitted["residual"], "Residual",
                            BLUE if col == 1 else RED, "markers", showlegend=False),
                      row=1, col=col)
        fig.add_hline(y=0, row=1, col=col, line_dash="dash")
    style(fig, "Paired noise draws; matched average conditional variance",
          "Predicted", "Residual")
    fig.update_yaxes(matches="y")
    return fig