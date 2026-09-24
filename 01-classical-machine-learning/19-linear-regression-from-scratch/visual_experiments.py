"""Generate the Day 19 visual learning sequence locally, without network access."""

import argparse
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from from_scratch import fit_ols, fit_gradient_descent
from visual_math import (
    comparison_data, design, gradient_trace, linear_data, loss_surface, scaling_data,
)

BLUE, ORANGE, GREEN, RED = "#2563a6", "#d97706", "#14806a", "#b42338"
START = np.array([-2.5, -0.8])
PREVIEWS = [
    "04_gradient_descent_path.gif", "03_loss_contour.png",
    "04b_regression_learning.gif", "06_scaling_geometry.png", "01_residuals.png",
]


def figure(title, subtitle="", figsize=(9, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold", x=0.08, ha="left")
    fig.text(0.08, 0.895, subtitle, fontsize=10, color="#475569")
    fig.subplots_adjust(left=0.1, right=0.94, bottom=0.15, top=0.80)
    return fig, ax


def save_png(fig, output, name):
    fig.savefig(output / name, dpi=135, facecolor="white")
    plt.close(fig)
    return name


def save_gif(fig, update, frames, output, name):
    animation = FuncAnimation(fig, update, frames=frames, interval=125, blit=False)
    animation.save(output / name, writer=PillowWriter(fps=8), dpi=95)
    plt.close(fig)
    return name


def frame_indices(length, maximum=48):
    indices = np.unique((np.linspace(0, 1, min(length, maximum)) ** 2 * (length - 1)).astype(int))
    return list(indices) + [length - 1] * 16


def scatter_fit(ax, X, y, beta, residuals=False):
    x = X[:, 0]
    prediction = beta[0] + beta[1] * x
    ax.scatter(x, y, s=26, color=BLUE, zorder=3, label="Synthetic observations")
    ax.plot(x, prediction, color=ORANGE, lw=2.3, label="Candidate line")
    if residuals:
        ax.vlines(x, prediction, y, color=GREEN, alpha=0.65, lw=1.2, label="Residuals")
    ax.set(xlabel="Feature x", ylabel="Target y")
    ax.grid(alpha=0.15)


def base_surface():
    X, y = linear_data()
    A = design(X)
    fit = fit_ols(X, y)
    grid = loss_surface(A, y, np.linspace(-4, 5, 100), np.linspace(-1.5, 3.8, 100))
    return X, y, A, fit, grid


def contours(ax, grid, optimum):
    B0, B1, losses = grid
    levels = np.geomspace(0.08, float(losses.max() - losses.min()), 13) + losses.min()
    cs = ax.contour(B0, B1, losses, levels=levels, colors="#507497", linewidths=1.0)
    ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt="%.1f")
    ax.scatter(*optimum, marker="*", s=170, c=GREEN, edgecolor="white", zorder=6, label="OLS minimum")
    ax.set(xlabel="Intercept b0", ylabel="Slope b1")
    ax.grid(alpha=0.12)


def residuals(output):
    X, y = linear_data()
    fit = fit_ols(X, y)
    errors = y - fit.predict(X)
    fig, ax = figure("Residuals: OLS minimizes their squared lengths", "Each vertical segment is e = observed y - predicted y; all data are synthetic.")
    scatter_fit(ax, X, y, fit.beta, residuals=True)
    for i in (int(np.argmin(errors)), 21, 32):
        ax.annotate(f"e = {errors[i]:+.2f}", (X[i, 0], y[i]),
                    xytext=(10, 15 if errors[i] > 0 else -24), textcoords="offset points",
                    fontsize=9, arrowprops={"arrowstyle": "-", "color": GREEN})
    ax.legend(loc="upper left", fontsize=9)
    fig.text(0.1, 0.035, f"SSE = {errors @ errors:.3f}     MSE = {np.mean(errors ** 2):.3f}     n = {len(y)}")
    return [save_png(fig, output, "01_residuals.png")], {"sse": float(errors @ errors), "mse": float(np.mean(errors ** 2))}


def squared_error(output):
    X, y = linear_data()
    optimum = fit_ols(X, y).beta
    candidates = [START, np.array([0.5, 1.3]), optimum]
    parameters = np.vstack([np.linspace(candidates[0], candidates[1], 18),
                            np.linspace(candidates[1], candidates[2], 18)])
    fig, ax = figure("Why does OLS prefer one line?", "Candidate coefficients change; the observations stay fixed.")
    low = min(y.min(), *(np.min(b[0] + b[1] * X[:, 0]) for b in candidates)) - 1
    high = y.max() + 1
    def update(i):
        ax.clear()
        beta = parameters[i]
        scatter_fit(ax, X, y, beta, residuals=True)
        error = y - design(X) @ beta
        ax.set(ylim=(low, high), title=f"b0={beta[0]:.2f}, b1={beta[1]:.2f}    SSE={error @ error:.2f}    MSE={np.mean(error ** 2):.3f}")
        ax.legend(loc="upper left", fontsize=8)
    fig.text(0.1, 0.035, "Interpolation between candidates, not an optimizer. Final candidate is the OLS fit.", fontsize=9)
    result = [{"beta": b.tolist(), "sse": float(np.sum((y - design(X) @ b) ** 2)),
               "mse": float(np.mean((y - design(X) @ b) ** 2))} for b in candidates]
    return [save_gif(fig, update, frame_indices(len(parameters)), output, "02_sse_comparison.gif")], {"candidates": result}


def loss_surface_view(output):
    X, y, A, fit, grid = base_surface()
    minimum = float(np.mean((fit.predict(X) - y) ** 2))
    fig, ax = figure("Each line becomes one point on a convex loss surface", "Contours label MSE; the star is the direct OLS solution, not a grid approximation.")
    contours(ax, grid, fit.beta)
    ax.legend(loc="upper left")
    files = [save_png(fig, output, "03_loss_contour.png")]
    try:
        import plotly.graph_objects as go
    except ImportError:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(projection="3d")
        ax.plot_surface(*grid, cmap="Blues", alpha=0.8)
        ax.scatter(*fit.beta, minimum, color=RED, s=60)
        ax.set(xlabel="Intercept b0", ylabel="Slope b1", zlabel="MSE", title="Convex MSE surface; red point = OLS")
        files.append(save_png(fig, output, "03_loss_surface_3d.png"))
        backend = "matplotlib fallback"
    else:
        surface = go.Figure(go.Surface(
            x=grid[0], y=grid[1], z=grid[2], colorscale="Blues", opacity=0.9,
            colorbar={"title": "MSE"},
            hovertemplate="Intercept=%{x:.3f}<br>Slope=%{y:.3f}<br>MSE=%{z:.3f}<extra></extra>",
        ))
        surface.add_trace(go.Scatter3d(x=[fit.beta[0]], y=[fit.beta[1]], z=[minimum],
                         mode="markers", marker={"size": 7, "color": RED}, name="OLS minimum",
                         hovertemplate="OLS<br>Intercept=%{x:.4f}<br>Slope=%{y:.4f}<br>MSE=%{z:.4f}<extra></extra>"))
        surface.update_layout(title="MSE over line parameters | drag to rotate, scroll to zoom",
                              scene={"xaxis_title": "Intercept b0", "yaxis_title": "Slope b1", "zaxis_title": "MSE"},
                              template="plotly_white")
        surface.write_html(output / "03_loss_surface_3d.html", include_plotlyjs=True, full_html=True)
        files.append("03_loss_surface_3d.html")
        backend = "Plotly offline HTML"
    return files, {"ols_beta": fit.beta.tolist(), "minimum_mse": minimum, "backend": backend}


def gradient_descent_view(output, data_space=False):
    X, y, A, fit, grid = base_surface()
    trace = gradient_trace(A, y, START, rate=0.08, steps=180)
    np.testing.assert_allclose(A @ trace.parameters[-1], fit.predict(X), atol=1e-6, rtol=0)
    title = "The line learns as its coefficients move" if data_space else "Gradient descent walks toward the OLS minimum"
    fig, ax = figure(title, "MSE gradient, step = 0.08; same data, initialization and updates in both animations.")
    if data_space:
        scatter_fit(ax, X, y, START)
        ax.lines[0].remove()
        ax.plot(X[:, 0], fit.predict(X), "--", color=GREEN, lw=2, label="OLS line")
        line, = ax.plot(X[:, 0], A @ START, color=ORANGE, lw=2.5, label="Current GD line")
        ax.set(ylim=(min((A @ START).min(), y.min()) - 1, y.max() + 1))
        ax.legend(loc="upper left", fontsize=9)
    else:
        contours(ax, grid, fit.beta)
        ax.scatter(*START, marker="s", color=RED, s=55, label="Start", zorder=7)
        line, = ax.plot([], [], color=ORANGE, lw=2, label="GD path")
        current, = ax.plot([], [], "o", color=RED, markersize=7, label="Current iterate")
        ax.legend(loc="upper left", fontsize=8)
    status = ax.text(0.02, 0.03, "", transform=ax.transAxes, bbox={"facecolor": "white", "alpha": 0.9})
    def update(i):
        if data_space:
            line.set_ydata(A @ trace.parameters[i])
        else:
            line.set_data(trace.parameters[:i + 1, 0], trace.parameters[:i + 1, 1])
            current.set_data([trace.parameters[i, 0]], [trace.parameters[i, 1]])
        status.set_text(f"Update {i}    MSE = {trace.losses[i]:.5f}")
    fig.text(0.1, 0.035, "Frames sample actual iterations; the final frame pauses for two seconds.", fontsize=9)
    name = "04b_regression_learning.gif" if data_space else "04_gradient_descent_path.gif"
    return [save_gif(fig, update, frame_indices(len(trace.losses)), output, name)], {
        "start_mse": float(trace.losses[0]), "final_mse": float(trace.losses[-1]),
        "updates": len(trace.losses) - 1, "status": trace.status,
        "max_prediction_gap": float(np.max(np.abs(A @ trace.parameters[-1] - fit.predict(X)))),
    }


def learning_rates(output):
    X, y = linear_data()
    A = design(X)
    boundary = 2 / np.linalg.eigvalsh(2 * A.T @ A / len(y))[-1]
    rates = [0.001, 0.01, 0.05, 0.1, float(0.95 * boundary), 0.5]
    fig, ax = figure("A convex objective still needs a suitable learning rate", "Finite MSE values only; a run stops before loss exceeds 1e10 or arithmetic overflows.", figsize=(10, 6))
    results = []
    for rate in rates:
        trace = gradient_trace(A, y, START, rate, steps=240)
        line, = ax.semilogy(trace.losses, label=f"{rate:.4g}: {trace.status}")
        if trace.status.startswith("diverged"):
            ax.scatter(len(trace.losses) - 1, trace.losses[-1], marker="x", color=line.get_color(), s=80)
        results.append({"rate": rate, "status": trace.status, "last_safe_mse": float(trace.losses[-1]),
                        "stop_iteration": trace.stop_iteration})
    optimum_loss = np.mean((fit_ols(X, y).predict(X) - y) ** 2)
    ax.axhline(optimum_loss, color=GREEN, ls="--", label=f"OLS MSE = {optimum_loss:.3f}")
    ax.set(xlabel="Gradient updates", ylabel="MSE (log scale)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)
    fig.text(0.1, 0.035, f"Exact-arithmetic stability interval: 0 < step < {boundary:.4f}. Parameter oscillation can occur while MSE falls.", fontsize=9)
    return [save_png(fig, output, "05_learning_rates.png")], {"stability_boundary": float(boundary), "runs": results}


def scaling(output):
    raw, scaled, y, scaler = scaling_data()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.72, bottom=0.24, wspace=0.38)
    fig.suptitle("Feature units change the optimizer's geometry", x=0.065, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.065, 0.87, "Center X and y to profile out the intercept. Both runs start at zero; step = 1.8/L in each coordinate system.", fontsize=10)
    results = {}
    for ax, name, A, color in zip(axes[:2], ("Unscaled", "Standardized"), (raw, scaled), (RED, GREEN)):
        H = 2 * A.T @ A / len(y)
        L = np.linalg.eigvalsh(H)[-1]
        trace = gradient_trace(A, y, [0, 0], 1.8 / L, steps=240)
        optimum = np.linalg.lstsq(A, y, rcond=None)[0]
        # Equal numerical spans expose curvature differences, instead of silently
        # stretching each raw coefficient axis until a thin valley looks circular.
        width = max(np.max(np.abs(optimum)) * 1.3, 2.)
        minimum = np.mean((A @ optimum - y) ** 2)
        # Analytic ellipses resolve the raw valley, which is narrower than the grid.
        eigenvalues, vectors = np.linalg.eigh(A.T @ A / len(y))
        angle = np.linspace(0, 2 * np.pi, 240)
        circle = np.vstack((np.cos(angle), np.sin(angle)))
        for excess in (0.25, 1., 4.):
            ellipse = optimum[:, None] + vectors @ (np.sqrt(excess / eigenvalues)[:, None] * circle)
            ax.plot(*ellipse, color=BLUE, alpha=0.7, lw=1)
        ax.plot(*trace.parameters.T, color=color, lw=1.4, alpha=0.85)
        ax.scatter(*trace.parameters[0], color=ORANGE, marker="s", s=35)
        ax.scatter(*trace.parameters[-1], color=color, s=30, zorder=5)
        ax.scatter(*optimum, color=GREEN, marker="*", s=120, zorder=6)
        ax.set(xlim=(optimum[0] - width, optimum[0] + width),
               ylim=(optimum[1] - width, optimum[1] + width),
               xlabel="Slope w1" if name == "Unscaled" else "Standardized slope w1",
               ylabel="Slope w2" if name == "Unscaled" else "Standardized slope w2",
               title=f"{name}\ncond(H) = {np.linalg.cond(H):.2e}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
        if name == "Unscaled":
            inset = ax.inset_axes([0.13, 0.12, 0.78, 0.28])
            inset.plot(*trace.parameters[:31].T, color=RED, lw=1, marker=".", ms=2)
            inset.set_title("First 30 updates (zoom)", fontsize=8)
            inset.tick_params(labelsize=7)
            inset.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            inset.xaxis.get_offset_text().set_fontsize(7)
            inset.yaxis.get_offset_text().set_fontsize(7)
        axes[2].semilogy(trace.losses, color=color, label=name)
        gap = float(trace.losses[-1] - minimum)
        results[name] = {"rate": float(1.8 / L), "condition_hessian": float(np.linalg.cond(H)),
                         "final_mse": float(trace.losses[-1]), "gap_to_ols": gap, "status": trace.status,
                         "updates": len(trace.losses) - 1}
        if name == "Standardized":
            raw_beta = optimum / scaler.scale_
            np.testing.assert_allclose(raw @ raw_beta, scaled @ optimum, atol=1e-10)
    axes[2].axhline(minimum, color=BLUE, ls="--", label="OLS minimum")
    axes[2].set(xlabel="Gradient updates", ylabel="MSE (log scale)", title="Same objective units\n240-update budget")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.15)
    fig.text(0.065, 0.105, "Contours: MSE above optimum = 0.25, 1, 4. Equal axis scales: the raw ellipse is extremely thin.", fontsize=10)
    fig.text(0.065, 0.06, "Square = start; dot = last iterate; star = OLS. Inset magnifies raw steps; its unequal axes do not represent curvature.", fontsize=10)
    return [save_png(fig, output, "06_scaling_geometry.png")], results


def outliers(output):
    X, y = linear_data()
    contaminated_X = np.vstack((X, [[3.7], [4.0]]))
    contaminated_y = np.r_[y, -13., -16.]
    clean, changed = fit_ols(X, y), fit_ols(contaminated_X, contaminated_y)
    fig, ax = figure("Two extreme residuals pull the least-squares line", "Compare both fits on the SAME original observations; this is a sensitivity example.")
    fig.subplots_adjust(bottom=0.22)
    ax.scatter(X[:, 0], y, color=BLUE, s=28, label="Original observations")
    ax.scatter(contaminated_X[-2:, 0], contaminated_y[-2:], color=RED, marker="X", s=90, label="Added outliers")
    ax.plot(X[:, 0], clean.predict(X), color=GREEN, lw=2, label="Original OLS")
    ax.plot(X[:, 0], changed.predict(X), color=ORANGE, lw=2, label="OLS with outliers")
    ax.set(xlabel="Feature x", ylabel="Target y")
    ax.legend(fontsize=9, loc="lower left")
    metrics = {"clean_beta": clean.beta.tolist(), "contaminated_beta": changed.beta.tolist(),
               "clean_fit_mse_on_original": mean_squared_error(y, clean.predict(X)),
               "contaminated_fit_mse_on_original": mean_squared_error(y, changed.predict(X))}
    fig.text(0.1, 0.07, f"Original: b0={clean.intercept:.3f}, b1={clean.coef[0]:.3f}, original-row MSE={metrics['clean_fit_mse_on_original']:.3f}", fontsize=10)
    fig.text(0.1, 0.025, f"With outliers: b0={changed.intercept:.3f}, b1={changed.coef[0]:.3f}, original-row MSE={metrics['contaminated_fit_mse_on_original']:.3f}", fontsize=10)
    return [save_png(fig, output, "07_outliers.png")], metrics


def multicollinearity(output):
    rng = np.random.default_rng(1909)
    x1 = rng.normal(size=100)
    X = np.column_stack((x1, x1 + rng.normal(0, 0.005, 100)))
    test_x = rng.normal(size=300)
    test = np.column_stack((test_x, test_x + rng.normal(0, 0.005, 300)))
    truth, test_truth = 1 + X @ [2., 2.], 1 + test @ [2., 2.]
    parameters, predictions = [], []
    for _ in range(100):
        fit = fit_ols(X, truth + rng.normal(0, 0.15, len(X)))
        parameters.append(fit.beta)
        predictions.append(fit.predict(test))
    parameters, predictions = np.array(parameters), np.array(predictions)
    rmses = np.sqrt(np.mean((predictions - test_truth) ** 2, axis=1))
    spread = parameters[:, 1:].std(axis=0)
    summed_spread = parameters[:, 1:].sum(axis=1).std()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(top=0.76, bottom=0.20, wspace=0.3)
    fig.suptitle("Stable predictions can hide unstable coefficients", fontsize=17, fontweight="bold")
    fig.text(0.125, 0.865, "100 fits; fixed nearly duplicate features; only target noise is regenerated (std = 0.15).", fontsize=10)
    axes[0].scatter(parameters[:, 1], parameters[:, 2], s=24, alpha=0.7, color=BLUE)
    limits = [parameters[:, 1].min(), parameters[:, 1].max()]
    axes[0].plot(limits, 4 - np.array(limits), "--", color=ORANGE, label="b1 + b2 = 4")
    axes[0].set(xlabel="Estimated b1", ylabel="Estimated b2", title="Individual coefficients trade off")
    axes[0].legend()
    axes[1].hist(rmses, bins=15, color=GREEN, alpha=0.85)
    axes[1].set(xlabel="RMSE against noiseless mean on 300 new rows", ylabel="Fit count", title="New rows preserve the same feature relationship")
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.text(0.125, 0.07, f"Coefficient std: {spread[0]:.3f}, {spread[1]:.3f}; sum std: {summed_spread:.3f}. Mean RMSE: {rmses.mean():.4f}.", fontsize=10)
    fig.text(0.125, 0.025, "No causal interpretation. Stability need not survive a change in the relationship between features.", fontsize=10)
    return [save_png(fig, output, "08_multicollinearity.png")], {
        "coef_std": spread.tolist(), "sum_std": float(summed_spread), "mean_rmse_to_true_mean": float(rmses.mean()),
        "feature_correlation": float(np.corrcoef(X.T)[0, 1]),
    }


def polynomial(output):
    rng = np.random.default_rng(1910)
    x = rng.uniform(-2, 2, 160)
    y = 3 * x ** 2 + rng.normal(0, 0.65, len(x))
    train, test = train_test_split(np.arange(len(x)), test_size=0.25, random_state=19)
    features = np.column_stack((x, x ** 2))
    line = fit_ols(features[train, :1], y[train])
    curve = fit_ols(features[train], y[train])
    grid = np.linspace(-2, 2, 200)
    fig, ax = figure("A curved prediction can still be linear in its coefficients", "Polynomial features: y-hat = b0 + b1*x + b2*x². Both models use ordinary least squares.")
    ax.scatter(x[train], y[train], s=20, color=BLUE, alpha=0.6, label="Training observations")
    ax.scatter(x[test], y[test], s=25, color=RED, marker="x", label="Held-out observations")
    ax.plot(grid, line.predict(grid[:, None]), color=ORANGE, lw=2, label="Features [x]")
    ax.plot(grid, curve.predict(np.column_stack((grid, grid ** 2))), color=GREEN, lw=2, label="Features [x, x²]")
    ax.set(xlabel="x", ylabel="y")
    ax.legend(fontsize=9)
    metrics = {"linear_test_mse": mean_squared_error(y[test], line.predict(features[test, :1])),
               "quadratic_test_mse": mean_squared_error(y[test], curve.predict(features[test])),
               "quadratic_beta": curve.beta.tolist()}
    fig.text(0.1, 0.035, f"Held-out MSE: raw x = {metrics['linear_test_mse']:.3f}; [x, x²] = {metrics['quadratic_test_mse']:.3f}. Fixed degree; no test-set tuning.", fontsize=10)
    return [save_png(fig, output, "09_polynomial_features.png")], metrics


def implementation_comparison(output):
    X, y = comparison_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=19)
    scaler = StandardScaler().fit(Xtr)
    ols = fit_ols(Xtr, ytr)
    gd = fit_gradient_descent(scaler.transform(Xtr), ytr)
    if not gd.converged:
        raise RuntimeError("Comparison GD did not converge.")
    reference = LinearRegression().fit(Xtr, ytr)
    coef = gd.coef / scaler.scale_
    betas = [ols.beta, np.r_[gd.intercept - scaler.mean_ @ coef, coef],
             np.r_[reference.intercept_, reference.coef_]]
    predictions = [ols.predict(Xte), gd.predict(scaler.transform(Xte)), reference.predict(Xte)]
    checks = [bool(np.allclose(p, predictions[-1], atol=1e-6, rtol=0)) for p in predictions[:2]]
    if not all(checks) or not np.allclose(betas[0], betas[1], atol=1e-6, rtol=0):
        raise AssertionError("Solver agreement failed at absolute tolerance 1e-6.")
    rows = []
    for name, beta, pred in zip(("Custom OLS", "Custom GD", "scikit-learn"), betas, predictions):
        rows.append([name] + [f"{b:.5f}" for b in beta] +
                    [f"{mean_squared_error(yte, pred):.6f}", f"{r2_score(yte, pred):.6f}"])
    fig, ax = figure("Three implementations, one least-squares objective", "All coefficients are in original feature units; scaling was fitted only on training rows.", figsize=(12, 5))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Method", "Intercept", "w1", "w2", "w3", "Test MSE", "Test R²"],
                     cellLoc="center", loc="center", colWidths=[0.19] + [0.135] * 6)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.4)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#dbeafe" if r == 0 else "#f1f5f9")
    gaps = [float(np.max(np.abs(p - predictions[-1]))) for p in predictions[:2]]
    fig.text(0.1, 0.1, f"Max test prediction gaps vs sklearn: OLS {gaps[0]:.2e}; GD {gaps[1]:.2e}. Both pass atol=1e-6, rtol=0.", fontsize=10)
    fig.text(0.1, 0.035, "Agreement is a numerical correctness check on synthetic data, not a performance benchmark.", fontsize=10)
    return [save_png(fig, output, "10_implementation_comparison.png")], {
        "parameters": [b.tolist() for b in betas], "test_mse": [mean_squared_error(yte, p) for p in predictions],
        "test_r2": [r2_score(yte, p) for p in predictions], "max_prediction_gaps": gaps, "allclose": checks,
    }


# Metadata is saved beside outputs so every executed experiment carries its caveats.
EXPERIMENTS = {
    "residuals": (residuals, "OLS minimizes squared vertical residuals.",
                  "Seed 19; 45 x values on [-2,4]; y=1.5+1.8*x+N(0,0.9²).",
                  "Residual lengths reveal the errors being squared.", "Training fit only; no inference or generalization claim."),
    "squared-error": (squared_error, "Changing coefficients changes SSE and MSE.",
                      "Same seed-19 data; interpolate START -> [0.5,1.3] -> OLS in 36 frames.",
                      "OLS has lower loss than the displayed alternatives.", "Selected alternatives do not constitute a search proof."),
    "loss-surface": (loss_surface_view, "Line estimation defines a convex quadratic surface.",
                     "Seed-19 data; 100x100 grid, intercept [-4,5], slope [-1.5,3.8].",
                     "The direct solution lies at the bottom of the displayed bowl.", "This full-rank two-parameter example does not show singular flat directions."),
    "gradient-descent": (gradient_descent_view, "Stable MSE-gradient updates approach OLS.",
                         "Seed-19 data; start [-2.5,-0.8]; rate .08; 180 updates; gradient tolerance 1e-9.",
                         "The parameter path approaches the direct minimum.", "Frames subsample updates; no convergence speed benchmark."),
    "regression-learning": (lambda out: gradient_descent_view(out, True), "Parameter updates move the fitted line.",
                            "Identical trace to gradient-descent; same seed, rate, start and budget.",
                            "Parameter-space progress corresponds to an improving line.", "One synthetic linear specification."),
    "learning-rates": (learning_rates, "A convex loss can converge slowly or diverge with different rates.",
                       "Seed-19 data; rates .001,.01,.05,.1,.95*(2/L),.5; 240 updates; cutoff 1e10.",
                       "Observed statuses distinguish convergence, unfinished runs and cutoff divergence.",
                       "A loss cutoff is a safety rule; finite budget exhaustion alone is not divergence."),
    "scaling": (scaling, "Changing feature units changes conditioning and GD progress.",
                "Seed 1907; n=160; uniform features scaled [1,10000]; y=2+8*x1+.0008*x2+N(0,.2²). Center X,y; step 1.8/L per representation; 240 updates.",
                "Standardization improves progress toward the same optimum in this fixture.",
                "Gradients use different coordinates; compare objective gaps, not gradient magnitudes. Centering profiles out the intercept; no held-out evaluation."),
    "outliers": (outliers, "Large residuals can change the OLS line substantially.",
                 "Seed-19 data plus (3.7,-13) and (4,-16); metrics on original rows.",
                 "The contaminated fit sacrifices fit on original rows.", "Chosen outliers; metrics are sensitivity diagnostics, not test errors."),
    "multicollinearity": (multicollinearity, "Individual coefficients can vary while in-distribution predictions remain stable.",
                          "Seed 1909; 100 fixed training rows; x2=x1+N(0,.005²); y mean=1+2*x1+2*x2; 100 independent target-noise draws std .15; 300 new rows.",
                          "Coefficient trade-offs can preserve the combined relationship.",
                          "New rows preserve feature dependence; RMSE is against the noiseless mean, not noisy future targets. No causal claim."),
    "polynomial": (polynomial, "A transformed-feature linear model can fit a quadratic mean.",
                   "Seed 1910; n=160; x uniform [-2,2]; y=3*x²+N(0,.65²); 120/40 split seed 19; degree fixed at 2.",
                   "The specified quadratic features reduce held-out error in this example.",
                   "The data generator favors degree 2; this does not establish superiority for arbitrary real data."),
    "comparison": (implementation_comparison, "Direct OLS and converged GD agree with sklearn.",
                   "Seed 1911; n=320,p=3; scales [1,10,.2]; beta [2.5,3,-.3,4]; noise std .5; 240/80 split seed 19; GD half-MSE rate .1 tol 1e-8.",
                   "Agreement supports correctness on this full-rank fixture.",
                   "No timing or production suitability claim; half-MSE uses half the MSE gradient."),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["all", *EXPERIMENTS], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "assets" / "day19")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    records = {}
    for name in names:
        render, hypothesis, configuration, interpretation, limitation = EXPERIMENTS[name]
        print(f"Running {name}...", flush=True)
        files, result = render(args.output_dir)
        for filename in files:
            path = args.output_dir / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty output: {path}")
            print(f"  {filename}: {path.stat().st_size / 1024:.1f} KiB", flush=True)
        records[name] = {"hypothesis": hypothesis, "configuration": configuration,
                         "result": result, "interpretation_candidate_author_review_required": interpretation,
                         "limitation": limitation, "files": files}
    report = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "matplotlib": matplotlib.__version__, "sklearn": sklearn.__version__},
              "experiments": records}
    report_name = f"results_{args.experiment}.json"
    (args.output_dir / report_name).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Recorded actual results and review caveats in {report_name}")
    print("Recommended public previews (educational priority; generated by --experiment all):")
    for filename in PREVIEWS:
        print(f"  {filename}")
    print("Open 03_loss_surface_3d.html locally for rotation, zoom and hover; keep the large HTML regenerable.")


if __name__ == "__main__":
    main()

