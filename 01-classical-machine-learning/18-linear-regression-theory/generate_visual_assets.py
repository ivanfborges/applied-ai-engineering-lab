"""Generate bounded, ignored previews and a machine-readable experiment record."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import sklearn

import regression_math as rm
import regression_charts as rc


def generate(output: Path, seed=42, frames=36):
    rm.checked_int(frames, "frames", 12, 80)
    rm.generator(seed)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    assets = []

    def save(fig, name):
        fig.savefig(output/name, dpi=140, bbox_inches="tight", pad_inches=.25)
        plt.close(fig)
        assets.append(name)

    x, y, _ = rm.line_data(seed=seed)
    fitted = rm.fit_ols(x, y)
    a, b, z, beta = rc.loss_axes(x, y)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    aa, bb = np.meshgrid(a, b)
    ax.plot_surface(aa, bb, z, cmap="Blues", alpha=.65, zorder=1)
    ax.scatter(*beta, rm.scores(y, fitted["pred"])["RSS"], color=rc.RED, s=55, zorder=5, label="OLS minimum")
    ax.set(xlabel="Intercept", ylabel="Slope", zlabel="RSS",
           title="OLS loss surface — minimum for this synthetic sample")
    ax.legend(loc="upper left")
    save(fig, "ols_loss_surface.png")

    descent = rm.gradient_descent(x, y, steps=90)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(x, y, s=12, color=rc.BLUE)
    grid = np.linspace(x.min(), x.max(), 100)
    current_line, = axes[0].plot(grid, np.zeros_like(grid), color=rc.ORANGE, lw=2)
    axes[0].plot(grid, beta[0]+beta[1]*grid, "--", color=rc.TEAL, label="OLS")
    axes[0].set(xlim=(-3, 3), ylim=(-10, 14), xlabel="x", ylabel="y")
    axes[0].legend()
    a, b, z, _ = rc.loss_axes(x, y, descent["path"])
    axes[1].contour(a, b, z, levels=16, cmap="Blues")
    axes[1].scatter(*beta, color=rc.TEAL, label="OLS")
    path_line, = axes[1].plot([], [], "o-", color=rc.RED, markersize=3)
    axes[1].set(xlabel="Intercept", ylabel="Slope")
    steps = np.unique(np.linspace(0, len(descent["path"])-1, frames).astype(int))

    def descent_frame(index):
        k = steps[index]
        path = descent["path"]
        current_line.set_ydata(path[k, 0]+path[k, 1]*grid)
        path_line.set_data(path[:k+1, 0], path[:k+1, 1])
        fig.suptitle(f"Gradient descent · step {k} · MSE {descent['losses'][k]:.3f}")
        return current_line, path_line

    FuncAnimation(fig, descent_frame, frames=len(steps), interval=110).save(
        output/"gradient_descent_ols.gif", writer=PillowWriter(fps=9), dpi=100)
    plt.close(fig)
    assets.append("gradient_descent_ols.gif")

    cx, cy, linear, quad = rm.curvature(seed=seed)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey="row")
    for col, fit in enumerate((linear, quad)):
        axes[0, col].scatter(cx, cy, s=12, color=rc.BLUE)
        axes[0, col].plot(cx, fit["pred"], color=rc.TEAL)
        axes[0, col].set(title=("x only", "x and x²")[col], ylabel="Target")
        axes[1, col].scatter(cx, fit["residual"], s=12, color=rc.RED)
        axes[1, col].axhline(0, color="gray", ls="--")
        axes[1, col].set(xlabel="x", ylabel="Residual")
    fig.suptitle("Missing curvature despite OLS training orthogonality")
    fig.tight_layout()
    save(fig, "residual_misspecification.png")

    vx, constant, changing, sigma = rm.variance_data(seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, target, title in zip(axes, (constant, changing),
                                  ("Constant variance", "Increasing variance")):
        fit = rm.fit_ols(vx, target)
        ax.scatter(fit["pred"], fit["residual"], s=12, color=rc.BLUE)
        ax.axhline(0, color="gray", ls="--")
        ax.set(xlabel="Predicted", ylabel="Residual", title=title)
    fig.suptitle("Matched average noise variance; shared standardized noise draws")
    fig.tight_layout()
    save(fig, "heteroskedasticity.png")

    independent = rm.stability(rho=0., repeats=60, seed=seed)
    correlated = rm.stability(rho=.99, repeats=60, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, r, title in zip(axes, (independent, correlated), ("Independent", "Correlation 0.99")):
        ax.plot(r["beta"][:, 0, 1], label="beta1", color=rc.BLUE)
        ax.plot(r["beta"][:, 0, 2], label="beta2", color=rc.RED)
        ax.set(xlabel="Independent training sample", ylabel="Coefficient", title=title)
        ax.legend()
    fig.suptitle("Coefficient stability across 60 synthetic training samples")
    fig.tight_layout()
    save(fig, "multicollinearity_stability.png")

    X, yy, _ = rm.correlated_data(rho=.99, seed=seed)
    alphas = np.logspace(-4, 3, 45)
    coef = np.array([rm.fitted_model(X, yy, alpha)[1] for alpha in alphas])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j in (1, 2):
        ax.semilogx(alphas, coef[:, j], label=f"beta{j}")
    ax.set(xlabel="Ridge alpha", ylabel="Slope in original units",
           title="Ridge paths · standardized predictors · intercept unpenalized")
    ax.legend()
    save(fig, "ridge_coefficient_paths.png")

    fit = rm.projection()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    q = np.linalg.qr(fit["design"])[0]
    u, v = np.meshgrid(np.linspace(-4, 4, 8), np.linspace(-4, 4, 8))
    p = u[..., None]*q[:, 0]+v[..., None]*q[:, 1]
    ax.plot_surface(p[..., 0], p[..., 1], p[..., 2], alpha=.15, color=rc.BLUE)
    for start, end, label, color in ((np.zeros(3), fit["y"], "y", rc.BLUE),
        (np.zeros(3), fit["pred"], "y_hat", rc.TEAL),
        (fit["pred"], fit["y"], "e", rc.RED)):
        ax.plot(*np.array([start, end]).T, color=color, label=label, lw=3, marker="o")
    unit = fit["residual"]/np.linalg.norm(fit["residual"])*.4
    square = np.array([fit["pred"]+q[:, 0]*.4, fit["pred"]+q[:, 0]*.4+unit, fit["pred"]+unit])
    ax.plot(*square.T, color=rc.ORANGE, lw=3, zorder=10)
    camera = -np.cross(fit["pred"], fit["residual"])
    ax.view_init(elev=np.degrees(np.arcsin(camera[2]/np.linalg.norm(camera))),
                 azim=np.degrees(np.arctan2(camera[1], camera[0])))
    ax.text2D(.93, .45, "Observation 3", rotation=90, transform=ax.transAxes)
    ax.set(xlabel="Observation 1", ylabel="Observation 2",
           title="OLS projection in response space")
    ax.legend()
    save(fig, "geometric_projection.png")

    X, yy, _ = rm.correlated_data(rho=0., seed=seed)
    plane_fit = rm.fit_ols(X, yy)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    u, v = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
    p = plane_fit["beta"]
    ax.plot_surface(u, v, p[0]+p[1]*u+p[2]*v, alpha=.35, color=rc.TEAL)
    ax.scatter(X[:, 0], X[:, 1], yy, color=rc.BLUE, s=12)
    ax.set(xlabel="x1", ylabel="x2", title="Multiple regression: fitted plane")
    ax.text2D(.93, .5, "Target", transform=ax.transAxes)
    save(fig, "multiple_regression_plane.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    base = rm.outlier_case(seed=seed)
    grid = np.linspace(-3, 10, 150)
    ax.scatter(base["x"], base["y"], s=14, color=rc.BLUE)
    bp = base["before"]["beta"]
    ax.plot(grid, bp[0]+bp[1]*grid, "--", color=rc.TEAL, label="Before added point")
    moving, = ax.plot([], [], "o", color=rc.RED, markersize=9)
    pulled, = ax.plot([], [], color=rc.ORANGE, label="Refitted line")
    ax.set(xlim=(-3, 10), ylim=(-6, 50), xlabel="x", ylabel="y")
    ax.legend()

    def outlier_frame(i):
        offset = 30*i/(frames-1)
        r = rm.outlier_case(offset=offset, seed=seed)
        b = r["after"]["beta"]
        moving.set_data([8], [r["ya"][-1]])
        pulled.set_data(grid, b[0]+b[1]*grid)
        ax.set_title(f"Added point offset {offset:.1f} · fitted slope {b[1]:.2f}")
        return moving, pulled

    FuncAnimation(fig, outlier_frame, frames=frames).save(
        output/"outlier_influence.gif", writer=PillowWriter(fps=9), dpi=100)
    plt.close(fig)
    assets.append("outlier_influence.gif")

    gx, gy, _ = rm.line_data(n=100, noise=1.5, seed=seed)
    order = rm.generator(seed+2).permutation(len(gx))
    gx, gy = gx[order], gy[order]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    points = ax.scatter([], [], s=16, color=rc.BLUE)
    fitted_line, = ax.plot([], [], color=rc.TEAL, label="OLS")
    grid = np.linspace(-3, 3, 100)
    ax.plot(grid, 2+1.5*grid, "--", color=rc.ORANGE, label="Generating mean")
    ax.set(xlim=(-3, 3), ylim=(-7, 11), xlabel="x", ylabel="y")
    ax.legend()
    sizes = np.linspace(5, len(gx), frames).astype(int)

    def growth_frame(i):
        count = sizes[i]
        b = rm.fit_ols(gx[:count], gy[:count])["beta"]
        points.set_offsets(np.column_stack((gx[:count], gy[:count])))
        fitted_line.set_data(grid, b[0]+b[1]*grid)
        ax.set_title(f"Sample growth · n={count} · slope={b[1]:.3f}")
        return points, fitted_line

    FuncAnimation(fig, growth_frame, frames=frames).save(
        output/"regression_sample_growth.gif", writer=PillowWriter(fps=9), dpi=100)
    plt.close(fig)
    assets.append("regression_sample_growth.gif")

    figures = {
        "ols_loss_surface.html": rc.loss_figures(x, y)[0],
        "gradient_descent_ols.html": rc.descent_animation(x, y, descent),
        "geometric_projection.html": rc.projection_chart(fit),
        "multiple_regression_plane.html": rc.plane(X, yy, plane_fit["beta"]),
    }
    for name, figure in figures.items():
        figure.write_html(output/name, include_plotlyjs=True, auto_play=False)
        assets.append(name)

    def record(hypothesis, config, result, interpretation, limitation):
        return dict(hypothesis=hypothesis, configuration=config, result=result,
                    interpretation_candidate=interpretation, limitation=limitation,
                    author_review_required=True)

    report = dict(seed=seed, frames=frames,
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   sklearn=sklearn.__version__, matplotlib=matplotlib.__version__),
                  assets=assets, experiments={
        "optimization": record("Stable gradient descent approaches OLS.",
            dict(n=80, noise=1, start=[6, -2], rate_ratio=.6, steps=90),
            dict(initial_mse=descent["losses"][0], final_mse=descent["losses"][-1],
                 ols_mse=rm.scores(y, fitted["pred"])["MSE"]),
            "Compare the final MSE with the direct solver and inspect the path.",
            "One scaled, one-predictor quadratic; convergence speed is design-dependent."),
        "curvature": record("Adding the generating quadratic term removes systematic lack of fit.",
            dict(n=100, noise=1, mean="3*x + 2*x^2"),
            dict(linear_rmse=rm.scores(cy, linear["pred"])["RMSE"],
                 quadratic_rmse=rm.scores(cy, quad["pred"])["RMSE"]),
            "Inspect training residual structure, not only the reduction in training loss.",
            "Training-only diagnostics; the correct feature was known in advance."),
        "heteroskedasticity": record("Conditional residual spread changes with x.",
            dict(n=150, sigma=".3 + .6*x", shared_standardized_noise=True),
            dict(min_sigma=float(sigma.min()), max_sigma=float(sigma.max())),
            "Inspect the funnel against the matched-average-variance comparison.",
            "No standard-error coverage experiment was run."),
        "collinearity_ridge": record("Correlated designs can destabilize slopes; Ridge changes prediction bias and variance.",
            dict(n=80, rho=.99, noise=1, repeats=60, alpha=10, test_n=250),
            dict(independent_beta_variance=independent["beta"][:, 0, 1:].var(axis=0, ddof=1).tolist(),
                 correlated_beta_variance=correlated["beta"][:, 0, 1:].var(axis=0, ddof=1).tolist(),
                 prediction_bias2=correlated["bias2"].tolist(),
                 prediction_variance=correlated["variance"].tolist(),
                 mean_error=correlated["mean_error"].tolist()),
            "Compare coefficient variability with prediction variability on matched test inputs.",
            "Finite ensemble on a fixed test design; these are not population bias estimates."),
        "outlier": record("An offset high-leverage point can change the fitted slope.",
            dict(n=50, point_x=8, final_offset=30, noise=.7),
            dict(before_slope=float(base["before"]["beta"][1]),
                 final_slope=float(rm.outlier_case(offset=30, seed=seed)["after"]["beta"][1])),
            "Inspect the moving line and distinguish leverage from the final residual.",
            "One synthetic contamination path, not a robustness benchmark."),
        "sample_growth": record("Estimates may settle as random observations accumulate.",
            dict(n=100, noise=1.5, order_seed=seed+2, first_n=5),
            dict(final_beta=rm.fit_ols(gx, gy)["beta"].tolist()),
            "Inspect fluctuations; stability need not improve at every added observation.",
            "One sequence does not estimate sampling variance or prove convergence."),
        "geometry": record("The projection residual is orthogonal to the design.",
            dict(target=fit["y"].tolist(), predictor=[-1, 0, 1]),
            dict(max_orthogonality_error=float(np.max(abs(fit["design"].T@fit["residual"])))),
            "Compare the right angle with the computed inner products.",
            "Three observations chosen for geometry, not a realistic statistical sample."),
        "plane": record("A two-feature OLS fit defines a plane.",
            dict(n=100, rho=0, noise=1, true_beta=[2, 3, -2]),
            dict(beta=plane_fit["beta"].tolist(), training_rmse=rm.scores(yy, plane_fit["pred"])["RMSE"]),
            "Inspect target-direction discrepancies around the fitted plane.",
            "Training fit to an intentionally additive synthetic process."),
    })
    (output/"experiment_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                                                 encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "assets": assets,
                      "report": "experiment_report.json"}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent/"outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=36)
    args = parser.parse_args()
    generate(args.output_dir, args.seed, args.frames)