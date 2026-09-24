"""Generate the offline Day 20 regularization visual laboratory.

Run from any directory:
    python day20_regularization_visuals.py --no-show

Dependencies are centralized in the repository's pyproject.toml.
See VISUAL_GUIDE.md for conventions, experiments, and output policy.
"""

from pathlib import Path

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "day20"
SHOW_PLOTS = True
SAVE_PLOTS = True
N_REPEATS = 100
N_FRAMES = 48
DPI = 150
GIF_FPS = 8

import argparse
import json
import platform
import sys
import warnings

import matplotlib
import numpy as np
import sklearn
from matplotlib import animation
from matplotlib.patches import Circle, Polygon
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge

import visual_math as math

# pyplot is imported in main after selecting a headless backend when requested.
plt = None
COLORS = {"ridge": "#176B9B", "lasso": "#C95336", "elastic": "#168477", "bias": "#A361AF"}
ARTIFACTS = []


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "#F8FAFC",
        "axes.edgecolor": "#CBD5E1", "text.color": "#172B4D",
        "axes.labelcolor": "#172B4D", "xtick.color": "#334155", "ytick.color": "#334155",
        "grid.alpha": 0.22, "grid.color": "#94A3B8", "lines.linewidth": 2,
        "savefig.facecolor": "white",
    })


def figure(title, subtitle, ncols=1, figsize=(12, 6)):
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.085, right=0.95, wspace=0.32)
    fig.suptitle(title, fontsize=19, weight="bold", x=0.07, ha="left", y=0.96)
    fig.text(0.07, 0.875, subtitle, fontsize=10.5, va="top")
    return fig, axes[0]


def footer(fig, text):
    fig.text(0.07, 0.055, text, fontsize=9, color="#526279", va="bottom")


def register(filename, what, takeaway):
    ARTIFACTS.append({
        "filename": filename, "what_it_shows": what,
        "main_takeaway": takeaway, "classification": "regenerable artifact",
    })


def finish(fig, filename, what, takeaway):
    if SAVE_PLOTS:
        fig.savefig(OUTPUT_DIR / filename, dpi=DPI)
    register(filename, what, takeaway)
    if not SHOW_PLOTS:
        plt.close(fig)


def save_animation(fig, update, filename, what, takeaway):
    movie = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / GIF_FPS, blit=False)
    if SAVE_PLOTS:
        movie.save(OUTPUT_DIR / filename, writer=animation.PillowWriter(fps=GIF_FPS), dpi=100)
    # Retain the animation while an interactive Matplotlib window is open.
    fig._day20_animation = movie
    register(filename, what, takeaway)
    if not SHOW_PLOTS:
        plt.close(fig)


def plot_constraint_geometry(data):
    fig, axes = figure(
        "01 / Why a corner can set a coefficient to zero",
        "Same computed quadratic loss; L2 radius = 1 versus L1 budget = 1. These are different feasible sets.",
        ncols=2,
    )
    grid = np.linspace(-1.4, 3.1, 241)
    b1, b2 = np.meshgrid(grid, grid)
    loss = math.objective_grid(data["X"], data["y"], b1, b2)
    for ax, method in zip(axes, ["ridge", "lasso"]):
        optimum = data[method]
        touch = float(math.objective_grid(data["X"], data["y"], *optimum))
        ax.contour(b1, b2, loss, levels=np.linspace(0.1, 7, 12), colors="#A6B4C4", linewidths=0.8)
        ax.contour(b1, b2, loss, levels=[touch], colors=COLORS[method], linewidths=2.5)
        shape = (
            Circle((0, 0), 1, facecolor=COLORS[method], edgecolor=COLORS[method], alpha=0.18)
            if method == "ridge" else
            Polygon([(1, 0), (0, 1), (-1, 0), (0, -1)], facecolor=COLORS[method], edgecolor=COLORS[method], alpha=0.18)
        )
        ax.add_patch(shape)
        ax.axhline(0, color="#CBD5E1", linewidth=0.8)
        ax.axvline(0, color="#CBD5E1", linewidth=0.8)
        ax.scatter(*data["ols"], marker="*", s=140, c="#172B4D", zorder=5)
        ax.annotate("OLS optimum", data["ols"], xytext=(1.75, 1.0), arrowprops={"arrowstyle": "->"})
        ax.scatter(*optimum, s=70, c=COLORS[method], zorder=6)
        ax.annotate(
            f"Constrained optimum\n({optimum[0]:.2f}, {optimum[1]:.2f})",
            optimum, xytext=(-1.15, 2.1), arrowprops={"arrowstyle": "->", "color": COLORS[method]},
        )
        ax.set(xlim=(-1.4, 3.1), ylim=(-1.4, 3.1), xlabel=r"$\beta_1$", ylabel=r"$\beta_2$")
        ax.set_aspect("equal")
        ax.set_title("Ridge / smooth L2 boundary" if method == "ridge" else "Lasso / L1 corner on the axis")
    footer(fig, "Colored contour = first contact as loss contours expand from OLS. Intercept removed by centering.\nThe optima are solved from this dataset, not placed by hand. Loss = SSE / (2n).")
    finish(fig, "01_l1_l2_geometry.png", "Quadratic loss tangent to computed L1 and L2 constraint optima.", "L1's axis corners can produce exact-zero coefficients.")


def plot_coefficient_path(data, method, number):
    fig, (ax,) = figure(
        f"{number:02d} / {method.title()} coefficient path",
        "Three direct signals, three noise predictors, and two correlated proxies; one fixed training-fitted scaler.",
    )
    feature_colors = ["#176B9B", "#C95336", "#168477", "#9865AB", "#8C7553", "#747D8C", "#176B9B", "#C95336"]
    for j in range(8):
        kind = "signal" if j < 3 else ("proxy" if j >= 6 else "noise")
        ax.semilogx(data["alphas"], data[method][:, j], color=feature_colors[j],
                    linestyle="--" if j >= 6 else "-", label=f"x{j + 1} ({kind})", alpha=0.9)
    ax.axhline(0, color="#64748B", linewidth=0.8)
    ax.set(xlabel=r"Normalized $\alpha$ (log scale)  →  stronger regularization",
           ylabel="Coefficient per training standard deviation")
    ax.grid(True)
    ax.legend(loc="upper right", ncols=2, fontsize=8)
    if method == "lasso":
        ax.axvline(data["alpha_max_lasso"], color="#64748B", linestyle=":", linewidth=1.2)
        ax.text(0.60, 0.07, "All coefficients zero beyond\ncomputed alpha_max", transform=ax.transAxes, fontsize=9)
        zeros = np.flatnonzero(data["lasso"][:, 2] == 0)
        if len(zeros):
            index = zeros[0]
            ax.scatter(data["alphas"][index], 0, s=65, c=feature_colors[2], zorder=5)
            ax.annotate("x3 reaches zero", (data["alphas"][index], 0),
                        xytext=(0.39, 0.30), textcoords="axes fraction", arrowprops={"arrowstyle": "->"})
        note = "Lasso: SSE/(2n) + alpha × L1. Individual variables can re-enter for correlated designs.\nThe all-zero threshold is computed from the centered feature–target correlations."
    else:
        error = np.max(np.abs(data["ridge"][0] - data["ols"]))
        ax.text(0.02, 0.05, f"Small alpha: near OLS\nmax coefficient difference = {error:.3g}",
                transform=ax.transAxes, fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        note = "Ridge: SSE/(2n) + alpha × L2²/2; sklearn alpha = n × normalized alpha.\nSmooth direction-wise shrinkage does not imply every individual coefficient is monotone."
    footer(fig, note)
    finish(fig, f"{number:02d}_{method}_path.png", f"Computed {method.title()} coefficients over 100 penalty values.",
           "Compare continuous shrinkage with L1's exact zeros; the axes use an explicit normalized penalty.")


def animate_soft_thresholding():
    fig, (ax,) = figure(
        "04 / Soft thresholding: a growing interval maps to zero",
        r"$S(z,\lambda)=\mathrm{sign}(z)\max(|z|-\lambda,0)$   •   Exact Lasso solution when $X^\top X/n=I$",
    )
    z = np.linspace(-4, 4, 401)
    ax.plot(z, z, "--", color="#A6B4C4", label="Unregularized z")
    line, = ax.plot(z, z, color=COLORS["lasso"], label="Soft-thresholded coefficient")
    left = ax.axvline(0, linestyle=":", color=COLORS["lasso"])
    right = ax.axvline(0, linestyle=":", color=COLORS["lasso"])
    dead, = ax.plot([0, 0], [0, 0], color=COLORS["lasso"], linewidth=7, solid_capstyle="round")
    label = ax.text(0.03, 0.90, "", transform=ax.transAxes, fontsize=12,
                    bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    samples = np.array([-3.2, -1.3, 0.6, 2.5])
    dots = ax.scatter(samples, samples, c=COLORS["lasso"], s=40, zorder=5)
    ax.set(xlim=(-4, 4), ylim=(-4, 4), xlabel="Unregularized coordinate z", ylabel=r"$S(z,\lambda)$")
    ax.grid(True)
    ax.legend(loc="lower right")
    footer(fig, "The flat segment is exact zero, not a rounding threshold.\nFor correlated designs this operator is used within iterative optimization, not once on OLS coefficients.")

    def update(frame):
        value = 3.5 * frame / (N_FRAMES - 1)
        line.set_ydata(math.soft_threshold(z, value))
        left.set_xdata([-value, -value])
        right.set_xdata([value, value])
        dead.set_xdata([-value, value])
        dots.set_offsets(np.column_stack([samples, math.soft_threshold(samples, value)]))
        label.set_text(f"lambda = {value:.2f}\n|z| ≤ {value:.2f} → coefficient = 0")
        return line, left, right, dead, dots, label

    save_animation(fig, update, "04_soft_thresholding.gif", "The L1 proximal map as its zero interval grows.",
                   "The interval |z| <= lambda becomes exactly zero.")


def plot_bias_variance(data):
    fig, axes = figure(
        "05 / Bias and variance measured across fresh training samples",
        f"{data['repeats']} independent samples × {data['n_train']} rows; degree-{data['degree']} Ridge; known sine-based truth.",
        ncols=2,
    )
    ax = axes[0]
    for key, label, color in [
        ("bias2", "Empirical squared bias", COLORS["bias"]),
        ("variance", "Empirical variance", COLORS["ridge"]),
        ("risk", "Estimated noisy-target MSE", "#172B4D"),
    ]:
        ax.loglog(data["alphas"], data[key], label=label, color=color)
    ax.axhline(data["noise_variance"], color="#94A3B8", linestyle=":", label="Known noise variance")
    best = int(np.argmin(data["risk"]))
    ax.scatter(data["alphas"][best], data["risk"][best], s=60, c="#172B4D", zorder=5)
    ax.set(xlabel="Normalized alpha (log scale)", ylabel="Mean squared error component (log scale)")
    ax.grid(True)
    ax.legend(fontsize=8)
    axes[1].plot(data["grid"], math.truth(data["grid"]), c="#172B4D", label="True mean function", linewidth=3)
    for index, label, color in [
        (0, "Weak penalty", COLORS["lasso"]),
        (best, "Minimum on empirical risk curve", COLORS["ridge"]),
        (-1, "Strong penalty", COLORS["bias"]),
    ]:
        axes[1].plot(data["grid"], data["mean_prediction"][index], label=label, color=color)
    axes[1].set(xlabel="Fixed evaluation x", ylabel="Mean prediction across training samples")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)
    footer(fig, f"Risk = empirical bias² + empirical variance (ddof=0) + known noise variance {data['noise_variance']:.2f}.\nFinite Monte Carlo estimates; the marked minimum is diagnostic, not a CV-selected deployment parameter.")
    finish(fig, "05_bias_variance.png", "Monte Carlo prediction decomposition against a known synthetic mean function.",
           "Variance reduction can offset bias; the finite-sample curves need not be monotone.")


def plot_multicollinearity(data):
    fig, axes = figure(
        "06 / Stable predictions can hide unstable coefficients",
        f"{data['repeats']} new training samples per setting; x2 = x1 + noise; true raw coefficients = (1, 1).",
        ncols=2,
    )
    fig.subplots_adjust(bottom=0.27)
    x = np.arange(len(data["sigmas"]))
    for j, (name, color) in enumerate([("OLS", "#C95336"), ("Ridge", "#176B9B")]):
        for feature in range(2):
            axes[0].semilogy(x, data["coefficients"][:, :, j, feature].std(axis=1),
                             marker="o", linestyle="-" if feature == 0 else "--",
                             color=color, label=f"{name}: beta{feature + 1}")
        means = data["rmse"][:, :, j].mean(axis=1)
        spreads = data["rmse"][:, :, j].std(axis=1)
        axes[1].errorbar(x + (j - 0.5) * 0.08, means, yerr=spreads, color=color, marker="o",
                        capsize=4, label=name)
    labels = [f"{sigma:g}\nr={corr:.5f}" for sigma, corr in zip(data["sigmas"], data["correlations"].mean(axis=1))]
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Noise SD in x2; mean training correlation\n→ higher multicollinearity")
        ax.grid(True)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Raw coefficient SD across samples (log scale)")
    axes[1].set_ylabel("Noisy-reference RMSE: mean ± SD")
    footer(fig, f"Ridge uses a fixed illustrative normalized alpha = {data['alpha']}; train-only standardization, then raw-unit coefficients.\nRMSE uses {data['n_eval']} fixed independent reference rows per setting. Error bars are SD, not confidence intervals.")
    finish(fig, "06_multicollinearity.png", "Coefficient variability and prediction RMSE as predictors become redundant.",
           "Assess coefficient variability separately from prediction accuracy.")


def plot_lasso_stability(data):
    fig, axes = figure(
        "07 / Sparse does not automatically mean stable",
        f"{data['repeats']} bootstrap fits of two noisy proxies for the same latent signal; observed correlation = {data['correlation']:.4f}.",
        ncols=2,
    )
    labels = ["Only x1", "Only x2", "Both", "Neither"]
    bars = axes[0].bar(labels, data["categories"], color=[COLORS["lasso"], COLORS["ridge"], COLORS["elastic"], "#94A3B8"])
    axes[0].bar_label(bars, labels=[f"{v:.0%}" for v in data["categories"]], padding=4)
    axes[0].set(ylim=(0, 1.12), ylabel="Fraction of bootstrap fits")
    axes[0].text(0.02, 0.97, f"Selected: x1 {data['frequency'][0]:.0%}  |  x2 {data['frequency'][1]:.0%}",
                 transform=axes[0].transAxes, va="top")
    weights = data["coefficients"]
    axes[1].scatter(weights[:, 0], weights[:, 1], alpha=0.65, c=COLORS["lasso"], edgecolors="white", linewidths=0.5)
    axes[1].axhline(0, c="#94A3B8", linewidth=0.8)
    axes[1].axvline(0, c="#94A3B8", linewidth=0.8)
    axes[1].set(xlabel="x1 coefficient", ylabel="x2 coefficient", title="Each point is one bootstrap fit")
    axes[1].grid(True)
    footer(fig, f"Fixed illustrative Lasso alpha = {data['alpha']}; nonzero means exactly nonzero. Scaling is re-fitted in each bootstrap.\nThese frequencies describe this finite sample and penalty; dropping a proxy does not prove irrelevance or causality.")
    finish(fig, "07_lasso_stability.png", "Bootstrap selection categories and the joint coefficient distribution.",
           "Correlated predictors can exchange weight or disappear across resamples.")


def plot_elasticnet_grouping(data):
    fig, axes = figure(
        "08 / What happens to a correlated group as the L1 share changes?",
        f"{data['repeats']} paired bootstrap samples; fixed normalized alpha = {data['alpha']}; x1–x3 share a latent signal.",
        ncols=2,
    )
    fig.subplots_adjust(bottom=0.25)
    names = ["Ridge\n0", ".1", ".3", ".5", ".7", ".9", "Lasso\n1"]
    features = ["x1 / group", "x2 / group", "x3 / group", "x4 / signal", "x5 / noise", "x6 / noise", "x7 / noise", "x8 / noise"]
    bound = np.abs(data["mean"]).max()
    images = [
        axes[0].imshow(data["mean"].T, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto"),
        axes[1].imshow(data["frequency"].T, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto"),
    ]
    for i, ax in enumerate(axes):
        ax.set_xticks(np.arange(7), names, fontsize=8)
        ax.set_yticks(np.arange(8), features, fontsize=8)
        ax.set_xlabel("L1 ratio")
        ax.axhline(2.5, color="#172B4D", linewidth=1.5)
        fig.colorbar(images[i], ax=ax, fraction=0.045, pad=0.03)
    axes[0].set_title("Mean signed coefficient")
    axes[1].set_title("Fraction selected")
    for row in range(8):
        for column in range(7):
            value = data["frequency"][column, row]
            axes[1].text(column, row, f"{value:.0%}", ha="center", va="center", fontsize=7.5,
                         color="white" if value > 0.65 else "#172B4D")
    footer(fig, f"All three group members retained: Ridge {data['group_all_frequency'][0]:.0%}, ratio .5 {data['group_all_frequency'][3]:.0%}, Lasso {data['group_all_frequency'][-1]:.0%}.\nObserved grouping depends on this generator and penalty. Ridge retention alone is not evidence of useful feature selection.")
    finish(fig, "08_elasticnet_grouping.png", "Mean coefficients and selection frequencies over L1 ratios.",
           "Compare correlated-group retention with sparsity among noise features.")


def surface_components(data, alpha, grid):
    b1, b2 = np.meshgrid(grid, grid)
    losses, optima = [], []
    for method in ["ols", "ridge", "lasso"]:
        ratio = 1.0 if method == "lasso" else 0.0
        strength = 0.0 if method == "ols" else alpha
        losses.append(math.objective_grid(data["X"], data["y"], b1, b2, strength, ratio))
        if method == "ols":
            optimum = data["ols"]
        elif method == "ridge":
            optimum = Ridge(alpha=len(data["X"]) * alpha, solver="svd").fit(data["X"], data["y"]).coef_
        else:
            optimum = Lasso(alpha=alpha, tol=1e-12, max_iter=100_000).fit(data["X"], data["y"]).coef_
        value = float(math.objective_grid(data["X"], data["y"], *optimum, strength, ratio))
        optima.append((optimum, value))
    return losses, optima


def plot_3d_loss_surface(data, force_fallback=False):
    grid = np.linspace(-1.5, 3.1, 45)
    alphas = np.geomspace(0.01, 10, 20)
    try:
        if force_fallback:
            raise ImportError("Requested Matplotlib fallback")
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        fig = plt.figure(figsize=(13, 5))
        fig.suptitle("09 / Centered regression loss + coefficient penalty; alpha = 1", fontsize=15)
        losses, optima = surface_components(data, 1.0, grid)
        b1, b2 = np.meshgrid(grid, grid)
        for j, name in enumerate(["OLS", "Ridge", "Lasso"]):
            ax = fig.add_subplot(1, 3, j + 1, projection="3d")
            ax.plot_surface(b1, b2, losses[j], cmap="viridis", alpha=0.85)
            optimum, value = optima[j]
            ax.scatter(*optimum, value, c="red", s=45)
            ax.set(xlabel="beta1", ylabel="beta2", zlabel="Objective", title=name)
        footer(fig, "OLS: SSE/(2n). Ridge adds alpha × L2²/2. Lasso adds alpha × L1. Intercept is unpenalized.")
        finish(fig, "09_loss_surface_3d.png", "Matplotlib fallback with computed objective surfaces and optima.",
               "Each penalty changes both the landscape and its optimum.")
        return "09_loss_surface_3d.png"

    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "surface"}] * 3],
                        subplot_titles=["OLS: half-MSE", "Ridge: half-MSE + alpha × L2²/2", "Lasso: half-MSE + alpha × L1"])
    frames = []
    all_payloads = []
    for index, alpha in enumerate(alphas):
        losses, optima = surface_components(data, float(alpha), grid)
        payload = []
        for j in range(3):
            optimum, value = optima[j]
            payload.extend([
                go.Surface(x=grid.tolist(), y=grid.tolist(), z=losses[j].tolist(), colorscale="Viridis",
                           showscale=False, opacity=0.9, name="Objective",
                           hovertemplate="beta1=%{x:.2f}<br>beta2=%{y:.2f}<br>objective=%{z:.3f}<extra></extra>"),
                go.Scatter3d(x=[optimum[0]], y=[optimum[1]], z=[value], mode="markers",
                             marker={"color": "#E4572E", "size": 6}, name=f"{['OLS', 'Ridge', 'Lasso'][j]} optimum",
                             showlegend=False, hovertemplate="beta1=%{x:.3f}<br>beta2=%{y:.3f}<br>objective=%{z:.3f}<extra>Optimum</extra>"),
            ])
        all_payloads.append(payload)
        frames.append(go.Frame(name=str(index), data=payload, traces=list(range(6))))
    for j in range(3):
        for trace in all_payloads[0][2 * j:2 * j + 2]:
            fig.add_trace(trace, row=1, col=j + 1)
    fig.frames = frames
    for j in range(1, 4):
        fig.update_layout(**{f"scene{j if j > 1 else ''}": {
            "xaxis_title": "beta1", "yaxis_title": "beta2", "zaxis_title": "Objective",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}}, "uirevision": "keep-camera",
            "aspectmode": "cube",
        }})
    fig.update_layout(
        title="09 / Rotate the landscapes; move alpha to recompute both penalized optima",
        height=700, width=1350, margin={"l": 20, "r": 20, "t": 100, "b": 180},
        paper_bgcolor="white", font={"family": "Arial", "color": "#172B4D"},
        sliders=[{
            "active": 0, "currentvalue": {"prefix": "Normalized alpha = "},
            "pad": {"t": 45},
            "steps": [{"label": f"{a:.3g}", "method": "animate",
                       "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                         "transition": {"duration": 0}}]} for i, a in enumerate(alphas)],
        }],
    )
    fig.add_annotation(text="Centered data; intercept unpenalized. Ridge native alpha = n * slider alpha. Each panel rescales z independently. Offline HTML.",
                       x=0.5, y=-0.35, xref="paper", yref="paper", showarrow=False, font={"size": 11})
    if SAVE_PLOTS:
        fig.write_html(OUTPUT_DIR / "09_loss_surface_3d.html", include_plotlyjs=True, full_html=True, auto_open=False, auto_play=False)
    register("09_loss_surface_3d.html", "Offline rotatable loss surfaces with a computed alpha slider.",
             "Ridge and Lasso move the optimum in different ways; hover to inspect coefficients.")
    return "09_loss_surface_3d.html"


def animate_regularized_surface(data):
    fig, (ax,) = figure(
        "10 / Watch the Ridge optimum move as the penalty grows",
        r"Objective = SSE/(2n) + $\lambda\|\beta\|_2^2/2$; contours are recomputed at every frame.",
    )
    grid = np.linspace(-1.1, 3.1, 141)
    b1, b2 = np.meshgrid(grid, grid)
    alphas = np.r_[0, np.geomspace(0.01, 40, N_FRAMES - 1)]
    gram = data["X"].T @ data["X"] / len(data["X"])
    cross = data["X"].T @ data["y"] / len(data["X"])
    optima = np.array([np.linalg.solve(gram + a * np.eye(2), cross) for a in alphas])
    contour = [None]
    trail, = ax.plot([], [], color=COLORS["ridge"], linewidth=2)
    point, = ax.plot([], [], "o", color=COLORS["ridge"], markersize=9)
    ax.scatter(*data["ols"], marker="*", c="#172B4D", s=100, label="OLS solution")
    ax.scatter(0, 0, marker="+", c="#64748B", s=100, label="Zero coefficients")
    label = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top",
                    bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    ax.set(xlim=(-1.1, 3.1), ylim=(-1.1, 3.1), xlabel="beta1", ylabel="beta2")
    ax.legend(loc="lower right", fontsize=9)
    footer(fig, "Same contour levels in every frame; intercept remains unpenalized. The path and minima are computed from the data.\nCoefficients approach zero gradually as lambda grows; no claim that each coordinate must always shrink monotonically.")

    def update(frame):
        if contour[0] is not None:
            old = contour[0]
            if hasattr(old, "remove"):
                old.remove()
            else:
                for collection in old.collections:
                    collection.remove()
        loss = math.objective_grid(data["X"], data["y"], b1, b2, alphas[frame])
        contour[0] = ax.contour(b1, b2, loss, levels=[0.1, 0.3, 0.6, 1, 2, 3, 4, 6, 10, 20, 40], colors="#6887A6", linewidths=1)
        trail.set_data(optima[:frame + 1, 0], optima[:frame + 1, 1])
        point.set_data([optima[frame, 0]], [optima[frame, 1]])
        label.set_text(f"lambda = {alphas[frame]:.3f}\nbeta1 = {optima[frame, 0]:.3f}\nbeta2 = {optima[frame, 1]:.3f}")
        return trail, point, label

    save_animation(fig, update, "10_ridge_landscape.gif", "Changing Ridge contours and their moving optimum.",
                   "The penalty changes the objective, not just the displayed coefficients.")


def plot_predictions(data):
    fig, axes = figure(
        "11 / Regularization changes the fitted function",
        "Same 32 observations, degree-12 polynomial features, train-only scaling; the middle penalty is chosen by five-fold CV.",
        ncols=3, figsize=(14, 6),
    )
    # The weak-penalty panel includes all predictions; the other panels keep readable local scales.
    low = min(data["predictions"].min(), data["y"].min(), data["truth"].min())
    high = max(data["predictions"].max(), data["y"].max(), data["truth"].max())
    span = high - low
    for j, (ax, title) in enumerate(zip(axes, ["Very weak penalty", "CV-selected penalty", "Very strong penalty"])):
        ax.scatter(data["x"], data["y"], s=20, c="#94A3B8", alpha=0.8, label="Observations")
        ax.plot(data["grid"], data["truth"], c="#172B4D", linestyle="--", label="True function")
        ax.plot(data["grid"], data["predictions"][j], c=COLORS["ridge"], label="Ridge prediction")
        ax.set(xlabel="x", ylabel="y", title=f"{title}\nnative alpha = {data['native_alphas'][j]:.3g}")
        # Separate y scales preserve detail; state this explicitly and report full errors.
        if j == 0:
            ax.set_ylim(low - 0.05 * span, high + 0.05 * span)
        ax.text(0.03, 0.04, f"Function RMSE = {data['function_rmse'][j]:.3f}", transform=ax.transAxes, fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
        ax.grid(True)
    axes[1].legend(fontsize=8, loc="upper left")
    footer(fig, f"Panel y scales differ to show the weak-penalty excursions without hiding the other curves. Selected CV RMSE = {data['cv_rmse']:.3f}.\nFunction RMSE compares predictions with known noiseless truth on a fixed grid; it is not a held-out noisy-target benchmark.")
    finish(fig, "11_polynomial_predictions.png", "Polynomial fits at a weak, CV-selected, and strong native Ridge alpha.",
           "Validate the penalty; show excessive complexity and excessive shrinkage using computed predictions.")


def plot_scaling_effect(data):
    fig, axes = figure(
        "12 / Equal coefficient penalties are sensitive to feature units",
        "Two equally informative latent predictors; observed units multiply feature B by 100,000.",
        ncols=2,
    )
    x = np.arange(2)
    for j, label in enumerate(["Without scaling", "Training-fitted StandardScaler"]):
        bars = axes[0].bar(x + (j - 0.5) * 0.32, data["unit_effects"][j], width=0.32,
                          label=label, color=[COLORS["lasso"], COLORS["ridge"]][j])
        axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[0].axhline(2, color="#172B4D", linestyle=":", label="True effect per latent SD")
    axes[0].set_xticks(x, ["A: units × 1", "B: units × 100,000"])
    axes[0].set_ylabel("Raw coefficient × known feature scale")
    axes[0].legend(fontsize=8, loc="lower left")
    for j, label in enumerate(["Unscaled", "Standardized"]):
        axes[1].scatter(data["y_eval"], data["predictions"][j], alpha=0.35, s=13,
                        color=[COLORS["lasso"], COLORS["ridge"]][j],
                        label=f"{label}: RMSE {data['rmse'][j]:.3f}")
    limits = [min(data["y_eval"].min(), data["predictions"].min()), max(data["y_eval"].max(), data["predictions"].max())]
    axes[1].plot(limits, limits, "--", color="#64748B", linewidth=1)
    axes[1].set(xlabel="Observed reference target", ylabel="Prediction")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)
    raw = data["raw_coefficients"]
    footer(fig, f"Raw coefficients: unscaled ({raw[0, 0]:.3g}, {raw[0, 1]:.3g}); standardized pipeline ({raw[1, 0]:.3g}, {raw[1, 1]:.3g}).\nFixed normalized alpha = {data['alpha']}; scaling changes the penalty geometry, but does not guarantee lower RMSE at an untuned alpha.")
    finish(fig, "12_feature_scaling.png", "Effective coefficients and independent-reference predictions before and after scaling.",
           "Units change the effective penalty; scaling is a modeling choice, not a guarantee of improved error.")


def create_summary_dashboard(paths, bias, grouping):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.09, right=0.95, hspace=0.63, wspace=0.32)
    fig.suptitle("13 / Regularization: four mechanisms to remember", x=0.07, ha="left", y=0.97, fontsize=20, weight="bold")
    fig.text(0.07, 0.90, "All curves below come from this run's synthetic experiments.", fontsize=11)
    for j in range(8):
        axes[0, 0].semilogx(paths["alphas"], paths["ridge"][:, j], linewidth=1.5)
        axes[0, 1].semilogx(paths["alphas"], paths["lasso"][:, j], linewidth=1.5)
    axes[0, 0].set_title("Ridge / smooth shrinkage")
    axes[0, 1].set_title("Lasso / exact zeros, conditional selection")
    for ax in axes[0]:
        ax.axhline(0, c="#94A3B8", linewidth=0.7)
        ax.set(xlabel="Normalized alpha", ylabel="Coefficient")
    for j in range(3):
        axes[1, 0].plot(grouping["ratios"], grouping["mean"][:, j], marker="o", label=f"Group x{j + 1}")
    axes[1, 0].set(xlabel="L1 ratio: Ridge 0 → Lasso 1", ylabel="Mean coefficient", title="ElasticNet / sharing weight within a group")
    axes[1, 0].legend(fontsize=8)
    for key, label, color in [("bias2", "Bias²", COLORS["bias"]), ("variance", "Variance", COLORS["ridge"]), ("risk", "Risk incl. noise", "#172B4D")]:
        axes[1, 1].loglog(bias["alphas"], bias[key], label=label, color=color)
    axes[1, 1].set(xlabel="Normalized alpha", ylabel="Error component", title="Bias–variance / measured trade-off")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(True)
    footer(fig, "Synthetic, finite-sample evidence. No method is universally best. Scale inside training folds and validate the penalty.")
    finish(fig, "13_regularization_summary.png", "Four compact views of shrinkage, sparsity, grouping, and prediction risk.",
           "Regularization changes the objective; choose it using the modeling goal and valid evaluation.")


def experiment_record(paths, geometry, bias, multi, stability, grouping, predictions, scaling):
    def entry(hypothesis, configuration, result, interpretation, limitation):
        return {
            "hypothesis": hypothesis, "configuration": configuration, "result": result,
            "interpretation_candidate_author_review_required": interpretation, "limitation": limitation,
        }
    best = int(np.argmin(bias["risk"]))
    return {
        "status": "Executed by generator; interpretations require author review.",
        "seed": RANDOM_STATE,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "objective": "SSE/(2n) + alpha*rho*L1 + alpha*(1-rho)*L2_squared/2; intercept unpenalized",
        "geometry": entry(
            "An L1 constraint can meet the loss on an axis.",
            {"seed": RANDOM_STATE + 1, "rows": 100, "correlation": 0.35, "true_beta": [2.4, 0.4], "L1_budget": 1, "L2_radius": 1},
            {"ols": geometry["ols"], "ridge": geometry["ridge"], "lasso": geometry["lasso"],
             "ridge_alpha": geometry["ridge_alpha"], "lasso_alpha": geometry["lasso_alpha"]},
            "Compare the computed tangency points; this configuration reaches an L1 corner.",
            "A constructed two-dimensional problem, not every possible design."),
        "paths": entry(
            "L1 has exact zeros while Ridge shrinks continuously.",
            {"seed": RANDOM_STATE, "rows": 240, "features": 8, "direct_beta": [3, -2, 1.5, 0, 0, 0, 0, 0],
             "proxy_noise_sd": 0.12, "target_noise_sd": 1.5, "alphas": paths["alphas"]},
            {"ridge": paths["ridge"], "lasso": paths["lasso"], "alpha_max_lasso": paths["alpha_max_lasso"]},
            "Inspect paths without assuming coordinate-wise monotonicity or permanent elimination.",
            "One fixed training sample; penalty paths are not validation curves."),
        "soft_threshold": entry(
            "The L1 proximal operator has an expanding zero interval.",
            {"lambda_start": 0, "lambda_end": 3.5, "frames": N_FRAMES, "fps": GIF_FPS},
            {"final_zero_interval": [-3.5, 3.5]},
            "The formula maps the whole interval exactly to zero.",
            "A direct coefficient solution only for an orthonormal design."),
        "bias_variance": entry(
            "Some shrinkage can reduce prediction variance enough to offset bias.",
            {"seed": RANDOM_STATE + 2, "repeats": bias["repeats"], "n_train": 35, "degree": 12,
             "noise_sd": 0.6, "truth": "sin(pi*x)+0.3*x", "x_distribution": "Uniform(-1,1)",
             "reference_grid": "201 equally spaced points on [-1,1]", "alphas": bias["alphas"]},
            {key: bias[key] for key in ["bias2", "variance", "risk", "signal_mse", "noise_variance"]} | {
                "minimum_grid_alpha": bias["alphas"][best],
                "minimum_estimated_risk": bias["risk"][best],
                "decomposition_max_error": np.max(np.abs(bias["risk"] - bias["bias2"] - bias["variance"] - bias["noise_variance"]))},
            "Use empirical risk and its components together; the minimum is a simulation diagnostic.",
            "Finite Monte Carlo samples and a bounded reference grid; no confidence interval."),
        "multicollinearity": entry(
            "Redundant predictors can destabilize coefficients more than predictions.",
            {"seed": RANDOM_STATE + 3, "repeats": multi["repeats"], "n_train": 80, "n_eval": 1500,
             "sigmas": multi["sigmas"], "true_beta": [1, 1], "noise_sd": 1, "alpha": multi["alpha"]},
            {"coefficient_sd": multi["coefficients"].std(axis=1), "mean_rmse": multi["rmse"].mean(axis=1),
             "sd_rmse": multi["rmse"].std(axis=1), "mean_correlation": multi["correlations"].mean(axis=1)},
            "Compare raw-unit coefficient SD with separately measured prediction RMSE.",
            "Fixed illustrative alpha; each setting has its own fixed noisy reference sample."),
        "lasso_stability": entry(
            "Sparse proxy selection can vary under bootstrap resampling.",
            {"seed": RANDOM_STATE + 4, "repeats": stability["repeats"], "rows": 160,
             "proxy_noise_sd": 0.06, "target": "2*latent + Normal(0,0.8^2)", "alpha": stability["alpha"]},
            {key: stability[key] for key in ["frequency", "categories", "coefficients", "correlation"]},
            "Selection frequencies quantify this sample's conditional instability.",
            "Bootstrap of one sample at a fixed alpha, not population selection probabilities or causal evidence."),
        "grouping": entry(
            "An L2 component may retain correlated groups more consistently.",
            {"seed": RANDOM_STATE + 5, "repeats": grouping["repeats"], "rows": 180, "features": 8,
             "proxy_noise_sd": 0.08, "target": "3*latent-2*x4+Normal(0,1)", "alpha": grouping["alpha"],
             "ratios": grouping["ratios"]},
            {key: grouping[key] for key in ["mean", "frequency", "group_all_frequency"]},
            "Inspect group-level retention alongside selection of independent noise features.",
            "Fixed alpha; group retention is not support recovery, and Ridge is generally dense."),
        "landscapes": entry(
            "Adding the penalty moves the optimum toward zero.",
            {"dataset": "Same as geometry", "HTML_alphas": np.geomspace(0.01, 10, 20),
             "GIF_alphas": np.r_[0, np.geomspace(0.01, 40, N_FRAMES - 1)], "frames": N_FRAMES},
            {"final_ridge_coef": Ridge(alpha=len(geometry["X"]) * 40, solver="svd").fit(geometry["X"], geometry["y"]).coef_},
            "Compare the moving optimum with the OLS reference and the origin.",
            "Two coefficients and centered intercept; finite grids render a continuous objective."),
        "polynomial_predictions": entry(
            "A validated intermediate penalty can improve over weak and excessive penalties.",
            {"seed": RANDOM_STATE + 6, "cv_seed": RANDOM_STATE, "rows": 32, "degree": 12, "noise_sd": 0.35,
             "truth": "sin(pi*x)+0.3*x", "CV": "5 shuffled folds; mean RMSE; pipeline includes polynomial transform and scaler",
             "native_alpha_grid": np.logspace(-6, 3, 22)},
            {"native_alphas": predictions["native_alphas"], "function_rmse": predictions["function_rmse"], "cv_rmse": predictions["cv_rmse"]},
            "Compare CV selection with known-function error; these are distinct quantities.",
            "One small sample; no noisy held-out test, and the CV score is used for selection."),
        "scaling": entry(
            "Unequal feature units create unequal effective penalties.",
            {"seed": RANDOM_STATE + 7, "n_train": 180, "n_eval": 400, "scales": scaling["scales"],
             "latent_effects": [2, 2], "noise_sd": 0.4, "alpha": scaling["alpha"]},
            {key: scaling[key] for key in ["raw_coefficients", "unit_effects", "rmse"]},
            "Scaling changes shrinkage symmetry; it need not improve RMSE at a fixed untuned alpha.",
            "Equal normalized alpha is intentional for the unit demonstration, not model selection."),
    }


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Use Agg and save without opening plot windows.")
    parser.add_argument("--no-save", action="store_true", help="Compute and display without writing outputs.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=N_REPEATS, help="Resamples for bias/variance, collinearity and Lasso.")
    parser.add_argument("--frames", type=int, default=N_FRAMES, help="GIF frame count, from 30 to 80.")
    parser.add_argument("--matplotlib-3d", action="store_true", help="Exercise the 3D PNG fallback instead of Plotly HTML.")
    return parser.parse_args()


def main():
    global plt, OUTPUT_DIR, SHOW_PLOTS, SAVE_PLOTS, N_REPEATS, N_FRAMES
    args = parse_arguments()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if not 2 <= args.repeats <= 500:
        raise ValueError("--repeats must be between 2 and 500.")
    if not 30 <= args.frames <= 80:
        raise ValueError("--frames must be between 30 and 80.")
    OUTPUT_DIR = args.output_dir.resolve()
    SHOW_PLOTS = SHOW_PLOTS and not args.no_show
    SAVE_PLOTS = SAVE_PLOTS and not args.no_save
    N_REPEATS, N_FRAMES = args.repeats, args.frames
    if not SHOW_PLOTS:
        matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    plt = pyplot
    setup_style()
    if SAVE_PLOTS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        print("[1/13] Computing L1/L2 constraint geometry...", flush=True)
        geometry = math.geometry_problem(RANDOM_STATE)
        plot_constraint_geometry(geometry)
        print("[2/13] Computing Ridge regularization path...", flush=True)
        paths = math.regularization_paths(RANDOM_STATE)
        plot_coefficient_path(paths, "ridge", 2)
        print("[3/13] Rendering Lasso path and zero threshold...", flush=True)
        plot_coefficient_path(paths, "lasso", 3)
        print("[4/13] Animating soft thresholding...", flush=True)
        animate_soft_thresholding()
        print("[5/13] Simulating prediction bias and variance...", flush=True)
        bias = math.simulate_bias_variance(RANDOM_STATE, N_REPEATS)
        plot_bias_variance(bias)
        print("[6/13] Measuring multicollinearity and prediction error...", flush=True)
        multi = math.simulate_multicollinearity(RANDOM_STATE, N_REPEATS)
        plot_multicollinearity(multi)
        print("[7/13] Bootstrapping Lasso feature selection...", flush=True)
        stability = math.analyze_lasso_stability(RANDOM_STATE, N_REPEATS)
        plot_lasso_stability(stability)
        print("[8/13] Comparing ElasticNet grouping across L1 ratios...", flush=True)
        grouping = math.elasticnet_grouping(RANDOM_STATE, min(N_REPEATS, 60))
        plot_elasticnet_grouping(grouping)
        print("[9/13] Building interactive offline 3D surfaces...", flush=True)
        plot_3d_loss_surface(geometry, args.matplotlib_3d)
        print("[10/13] Animating the Ridge landscape...", flush=True)
        animate_regularized_surface(geometry)
        print("[11/13] Selecting a polynomial Ridge penalty with CV...", flush=True)
        predictions = math.polynomial_predictions(RANDOM_STATE)
        plot_predictions(predictions)
        print("[12/13] Comparing feature units and scaling...", flush=True)
        scaling = math.demonstrate_scaling_effect(RANDOM_STATE)
        plot_scaling_effect(scaling)
        print("[13/13] Assembling the summary...", flush=True)
        create_summary_dashboard(paths, bias, grouping)
    record = experiment_record(paths, geometry, bias, multi, stability, grouping, predictions, scaling)
    if SAVE_PLOTS:
        for filename, payload in [("artifacts.json", ARTIFACTS), ("experiment_results.json", record)]:
            (OUTPUT_DIR / filename).write_text(json.dumps(payload, indent=2, default=json_default, allow_nan=False) + "\n", encoding="utf-8")
        missing = [item["filename"] for item in ARTIFACTS if not (OUTPUT_DIR / item["filename"]).is_file()]
        if missing:
            raise RuntimeError(f"Missing generated assets: {missing}")
    print("\nRegularization Visual Lab complete.")
    print("Generated assets:" if SAVE_PLOTS else "Computed views (saving disabled):")
    for item in ARTIFACTS:
        print(f"- {item['filename']}: {item['main_takeaway']}")
    if SAVE_PLOTS:
        print(f"- artifacts.json\n- experiment_results.json\nOutput directory: {OUTPUT_DIR}")
    chosen = predictions["native_alphas"][1]
    print(f"\nCV-selected polynomial Ridge native alpha: {chosen:.6g}")
    print(f"Same final-fit normalized alpha: {chosen / len(predictions['x']):.6g}")
    print(f"Lasso stability experiment fixed alpha: {stability['alpha']}; selection frequencies: {stability['frequency']}")
    print(f"Grouping experiment fixed alpha: {grouping['alpha']}; joint group retention: {grouping['group_all_frequency']}")
    print("No Lasso/ElasticNet alpha was selected by CV in this visual lab.")
    print("Experiment interpretations require author review. No README was edited.")
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
