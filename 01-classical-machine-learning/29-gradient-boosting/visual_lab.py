"""Generate Day 29 PNGs, GIFs, and offline HTML explaining sequential correction."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from PIL import Image
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from visual_core import (
    fit_history, fit_library_path, logistic_loss, logistic_negative_gradient,
    make_regression_data, numerical_negative_gradient, path_summary, save_history,
    squared_loss,
)


MODEL = "#007f86"
CURRENT = "#335c9b"
CORRECTION = "#db751d"
VALIDATION = "#87479f"
POINT = "#667085"


def decorate(ax, title, ylabel="y"):
    ax.set(title=title, xlabel="x", ylabel=ylabel)
    ax.grid(alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)


def regression_background(ax, data, truth=True):
    ax.scatter(data.x_train[:, 0], data.y_train, s=13, alpha=0.45, color=POINT, label="Training observations")
    if truth:
        ax.plot(data.x_grid[:, 0], data.truth_grid, "--", color="#222222", lw=1.7, label="Known true function")


def residual_segments(ax, data, prediction):
    indices = np.argsort(data.x_train[:, 0])[::12]
    ax.vlines(
        data.x_train[indices, 0], prediction[indices], data.y_train[indices],
        color=CORRECTION, alpha=0.8, lw=1.3, label="Residual: observed - current",
    )


def save_figure(fig, path, dpi):
    """Save an opaque figure so labels stay legible on light or dark backgrounds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def initial_model(data, history):
    fig, ax = plt.subplots(figsize=(10, 5))
    regression_background(ax, data)
    mean = history.predictions_train[0, 0]
    ax.axhline(mean, color=CURRENT, lw=2.5, label=f"Initial mean = {mean:.3f}")
    residual_segments(ax, data, history.predictions_train[0])
    decorate(ax, r"01 | Start with a constant: $F_0(x)=\bar y_{\mathrm{train}}$")
    ax.legend(loc="upper left", fontsize=9)
    return fig


def residual_learning(data, history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    regression_background(axes[0], data, truth=False)
    axes[0].plot(data.x_grid[:, 0], history.predictions_grid[0], color=CURRENT, lw=2.5, label="Current prediction F0")
    residual_segments(axes[0], data, history.predictions_train[0])
    decorate(axes[0], "02 | Current predictions and observed errors")
    axes[1].scatter(data.x_train[:, 0], history.residuals_train[0], s=14, alpha=0.5, color=CORRECTION, label="Correction targets: y - F0")
    axes[1].plot(data.x_grid[:, 0], history.corrections_grid[0], color=MODEL, lw=2.4, label="Depth-2 weak learner h1")
    axes[1].axhline(0, color=POINT, lw=0.8)
    decorate(axes[1], r"Fit $x\ \rightarrow\ r_1$, then use it as a correction", ylabel="Residual / raw correction")
    for ax in axes:
        ax.legend(fontsize=8)
    return fig


def one_boosting_step(data, history, stage=5):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    previous = history.predictions_grid[stage - 1]
    correction = history.corrections_grid[stage - 1]
    updated = history.predictions_grid[stage]
    x = data.x_grid[:, 0]
    for ax in (axes[0], axes[2]):
        regression_background(ax, data)
    axes[0].plot(x, previous, color=CURRENT, lw=2.5, label=f"F{stage - 1}")
    decorate(axes[0], f"03 A | Current function (stage {stage - 1})")
    axes[1].scatter(data.x_train[:, 0], history.residuals_train[stage - 1], s=12, alpha=0.45, color=POINT, label="Current residual targets")
    axes[1].plot(x, correction, "--", color=CORRECTION, label="Raw learner h")
    axes[1].plot(x, history.learning_rate * correction, color=MODEL, lw=2.5, label=f"Applied correction: {history.learning_rate} x h")
    axes[1].axhline(0, color=POINT, lw=0.8)
    decorate(axes[1], "B | Learn and shrink a correction", "Correction")
    axes[2].plot(x, previous, "--", color=CURRENT, alpha=0.8, label="Previous function")
    axes[2].plot(x, updated, color=MODEL, lw=2.5, label=f"Updated F{stage}")
    decorate(axes[2], r"C | $F_m=F_{m-1}+\eta h_m$")
    limits = (min(data.y_train.min(), data.truth_grid.min()) - 0.4, max(data.y_train.max(), data.truth_grid.max()) + 0.4)
    axes[0].set_ylim(limits)
    axes[2].set_ylim(limits)
    index = int(np.argmin(np.abs(x - 1.0)))
    annotation = (
        f"At x = {x[index]:.2f}: {previous[index]:.3f} + "
        f"{history.learning_rate:.1f} x ({correction[index]:+.3f}) = {updated[index]:.3f}"
    )
    fig.suptitle(annotation, fontsize=11)
    for ax in axes:
        ax.legend(fontsize=8)
    return fig


def sequential_boosting(data, history, path, dpi):
    """Animate actual ensemble states at bounded, densely spaced early stages."""
    stages = [0, 1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100]
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    regression_background(ax, data)
    line, = ax.plot(data.x_grid[:, 0], history.predictions_grid[0], color=MODEL, lw=2.5, label="Current boosted prediction")
    ax.set_ylim(data.y_train.min() - 0.5, data.y_train.max() + 0.5)
    decorate(ax, "04 | Sequential correction")
    ax.legend(fontsize=8, loc="upper left")
    status = ax.text(0.02, 0.04, "", transform=ax.transAxes, fontsize=9,
                     bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})

    def update(stage):
        line.set_ydata(history.predictions_grid[stage])
        ax.set_title(f"04 | Stage {stage}/100: add a weak correction (eta = {history.learning_rate})")
        status.set_text(f"Train RMSE {history.train_rmse[stage]:.3f} | validation RMSE {history.validation_rmse[stage]:.3f}")
        return line, status

    animation = FuncAnimation(fig, update, frames=stages, interval=550, blit=False)
    animation.save(path, writer=PillowWriter(fps=2), dpi=dpi)
    plt.close(fig)
    return stages


def residual_evolution(data, history, path, dpi):
    stages = [0, 1, 5, 10, 25, 50, 100]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    train = axes[0].scatter(data.x_train[:, 0], history.residuals_train[0], color=MODEL, s=15, alpha=0.65)
    validation = axes[1].scatter(data.x_validation[:, 0], history.residuals_validation[0], color=VALIDATION, s=20, alpha=0.7)
    extent = 1.08 * max(np.abs(history.residuals_train).max(), np.abs(history.residuals_validation).max())
    for ax, label in zip(axes, ("Training residuals", "Validation residuals")):
        ax.axhline(0, color="#222222", lw=1)
        ax.set_ylim(-extent, extent)
        decorate(ax, label, "Observed target - current prediction")

    def update(stage):
        train.set_offsets(np.column_stack([data.x_train[:, 0], history.residuals_train[stage]]))
        validation.set_offsets(np.column_stack([data.x_validation[:, 0], history.residuals_validation[stage]]))
        fig.suptitle(
            f"05 | Residuals after stage {stage}: train RMSE {history.train_rmse[stage]:.3f}, "
            f"validation {history.validation_rmse[stage]:.3f}\nNoise remains; held-out residuals need not converge to zero.",
            fontsize=10,
        )
        return train, validation

    animation = FuncAnimation(fig, update, frames=stages, interval=700, blit=False)
    animation.save(path, writer=PillowWriter(fps=1.5), dpi=dpi)
    plt.close(fig)
    return stages


def additive_prediction(data, history):
    index = int(np.argmin(np.abs(data.x_grid[:, 0] - 1.0)))
    cumulative = history.predictions_grid[:, index]
    contributions = history.learning_rate * history.corrections_grid[:, index]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained", sharex=True)
    stages = np.arange(len(contributions) + 1)
    axes[0].plot(stages, cumulative, color=MODEL, lw=2.5)
    axes[0].scatter([0, stages[-1]], cumulative[[0, -1]], color=CURRENT, zorder=4)
    axes[0].axhline(data.truth_grid[index], color=POINT, ls="--", label="Noiseless target at x*")
    axes[0].set(title=f"06 | Prediction at x* = {data.x_grid[index, 0]:.3f}: {cumulative[0]:.3f} -> {cumulative[-1]:.3f}",
                ylabel="Cumulative prediction")
    axes[0].legend(fontsize=9)
    axes[1].bar(stages[1:], contributions, color=np.where(contributions >= 0, MODEL, CORRECTION), width=0.8)
    axes[1].axhline(0, color=POINT, lw=1)
    axes[1].set(xlabel="Boosting stage", ylabel="Applied contribution", title=r"Signed additions $\eta h_m(x^*)$; no averaging")
    for ax in axes:
        ax.grid(alpha=0.15)
    return fig


def learning_rate_comparison(data, paths):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    colors = (CORRECTION, VALIDATION, MODEL, CURRENT)
    for path, color in zip(paths, colors):
        rate = path["learning_rate"]
        axes[0].plot(path["validation_rmse"], color=color, label=f"eta={rate}")
        # Compare the same number of trees, not a rate-dependent optimum.
        prediction = list(path["model"].staged_predict(data.x_grid))[19]
        axes[1].plot(data.x_grid[:, 0], prediction, color=color, lw=1.8, label=f"eta={rate}")
    axes[0].set(title="07 | Measured validation error by stage", xlabel="Boosting stage (0 = training mean)", ylabel="Validation RMSE")
    axes[1].plot(data.x_grid[:, 0], data.truth_grid, "k--", lw=1.5, label="Known truth")
    decorate(axes[1], "Same budget: predictions after 20 trees")
    fig.suptitle("Shrinkage changes later residual targets; no rate is universally best.", fontsize=11)
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8)
    return fig


def tree_complexity(data, paths):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharex=True, sharey=True)
    for ax, path in zip(axes.flat, paths):
        regression_background(ax, data)
        prediction = path["model"].predict(data.x_grid)
        ax.plot(data.x_grid[:, 0], prediction, color=MODEL, lw=2, label="Boosted prediction")
        decorate(ax, f'Depth {path["max_depth"]}: train {path["train_rmse"][-1]:.3f} | val {path["validation_rmse"][-1]:.3f}')
    fig.suptitle("08 | Final functions: rate 0.1, 100 trees, minimum leaf size 3\nGreater depth permits more complex corrections; compare measured errors.", fontsize=11)
    axes[0, 0].legend(fontsize=8)
    return fig


def train_validation_loss(path):
    fig, ax = plt.subplots(figsize=(10, 5))
    best = path["best_stage"]
    ax.plot(path["train_rmse"], color=MODEL, lw=2, label="Training RMSE")
    ax.plot(path["validation_rmse"], color=VALIDATION, lw=2, label="Validation RMSE")
    ax.axvline(best, color=CORRECTION, ls="--", label=f"Best validation stage = {best}")
    ax.scatter([best], [path["validation_rmse"][best]], color=CORRECTION, zorder=4)
    ax.set(
        xlabel="Boosting stage (0 = mean)", ylabel="RMSE",
        title=f'09 | Actual train/validation path: depth {path["max_depth"]}, eta={path["learning_rate"]}',
    )
    ax.text(0.97, 0.96, f'At stage {best}: train {path["train_rmse"][best]:.3f}, val {path["validation_rmse"][best]:.3f}\n'
            f'Final: train {path["train_rmse"][-1]:.3f}, val {path["validation_rmse"][-1]:.3f}',
            ha="right", va="top", transform=ax.transAxes, fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#dddddd"})
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", fontsize=9)
    inset = ax.inset_axes([0.36, 0.35, 0.6, 0.4])
    stages = np.arange(len(path["train_rmse"]))
    inset.plot(stages[20:], path["train_rmse"][20:], color=MODEL)
    inset.plot(stages[20:], path["validation_rmse"][20:], color=VALIDATION)
    inset.axvline(best, color=CORRECTION, ls="--", lw=1)
    inset.set_title("After stage 20: same measurements", fontsize=9)
    inset.set_xlabel("Stage", fontsize=8)
    inset.set_ylabel("RMSE", fontsize=8)
    inset.tick_params(labelsize=8)
    inset.grid(alpha=0.15)
    return fig


def functional_gradient_descent():
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), layout="constrained")
    rows = (
        ("10 | Parameter gradient descent", [r"$\theta_0$", r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"],
         r"$\theta_m=\theta_{m-1}-\eta\nabla_\theta R(\theta_{m-1})$", "Move parameters within a chosen model."),
        ("Functional gradient boosting", [r"$F_0(x)$", r"$F_1(x)$", r"$F_2(x)$", r"$F_3(x)$"],
         r"$F_m(x)=F_{m-1}(x)+\eta h_m(x)$", "Fit a weak learner to the negative prediction gradient, then add it."),
    )
    for ax, (title, labels, formula, description) in zip(axes, rows):
        ax.set(xlim=(-0.3, 3.5), ylim=(0, 1.6), title=title)
        ax.axis("off")
        for i, label in enumerate(labels):
            ax.text(i, 1.03, label, ha="center", fontsize=16,
                    bbox={"boxstyle": "round,pad=0.45", "facecolor": "#edf5f6", "edgecolor": MODEL})
            if i < 3:
                ax.annotate("", (i + 0.68, 1.07), (i + 0.28, 1.07),
                            arrowprops={"arrowstyle": "->", "color": CORRECTION, "lw": 2})
        ax.text(1.5, 0.48, formula, ha="center", fontsize=15)
        ax.text(1.5, 0.1, description, ha="center", fontsize=10)
    return fig


def negative_gradient_landscape(data, history):
    index = int(np.argmin(np.abs(data.x_train[:, 0] - 1.0)))
    target = float(data.y_train[index])
    current = float(history.predictions_train[0, index])
    ideal = target - current
    fitted = float(history.corrections_train[0, index])
    eta = history.learning_rate
    ideal_update, actual_update = current + eta * ideal, current + eta * fitted
    domain = np.linspace(min(target, current) - 1, max(target, current) + 1, 200)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(domain, squared_loss(target, domain), color=CURRENT, lw=2, label=r"$L=\frac{1}{2}(y-F)^2$")
    ax.scatter([current, target, ideal_update, actual_update], squared_loss(target, np.array([current, target, ideal_update, actual_update])),
               c=[CURRENT, POINT, CORRECTION, MODEL], s=65, zorder=4)
    tangent_domain = np.linspace(current - 0.5, current + 0.5, 50)
    ax.plot(tangent_domain, squared_loss(target, current) + (current - target) * (tangent_domain - current),
            "--", color=POINT, label=f"Slope dL/dF = {current - target:+.3f}")
    ax.annotate("Ideal pointwise negative-gradient step", (ideal_update, squared_loss(target, ideal_update)),
                (0.04, 0.66), textcoords="axes fraction", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": CORRECTION}, color=CORRECTION)
    ax.annotate("Actual tree update (region approximation)", (actual_update, squared_loss(target, actual_update)),
                (0.04, 0.51), textcoords="axes fraction", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": MODEL}, color=MODEL)
    ax.text(0.04, 0.94, f"Current F={current:.3f}; target y={target:.3f}; eta={eta}\n"
            f"Pointwise y-F={ideal:+.3f}; fitted leaf h={fitted:+.3f}",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set(title="11 | Descent at one observation: exact gradient versus tree approximation",
           xlabel="Raw prediction F", ylabel="Half squared error")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.15)
    return fig


def pseudo_residuals(data, history):
    indices = np.argsort(data.x_train[:, 0])[::10]
    y = data.y_train[indices]
    prediction = history.predictions_train[0, indices]
    residual = y - prediction
    numeric = numerical_negative_gradient(squared_loss, y, prediction)
    labels = np.array([0, 1, 0, 1, 1, 0])
    logits = np.array([-2.0, -1.5, 0.4, 0.1, 2.0, 1.2])
    signals = logistic_negative_gradient(labels, logits)
    logistic_numeric = numerical_negative_gradient(logistic_loss, labels, logits)
    probability = labels - signals
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    axes[0].scatter(residual, numeric, color=MODEL, s=30, label="Central finite differences")
    bound = 1.1 * max(np.abs(residual).max(), 1)
    axes[0].plot([-bound, bound], [-bound, bound], "--", color=POINT, label="Identity: y - F")
    axes[0].set(title="12 | Squared error: gradient = ordinary residual", xlabel="Ordinary residual y - F", ylabel="Numerical negative gradient")
    axes[0].text(0.03, 0.95, f"Max absolute difference: {np.max(np.abs(numeric - residual)):.2e}",
                 transform=axes[0].transAxes, va="top", fontsize=9)
    positions = np.arange(len(labels))
    axes[1].bar(positions, signals, color=np.where(signals >= 0, MODEL, CORRECTION), label="Analytic y - sigmoid(F)")
    axes[1].scatter(positions, logistic_numeric, color="#222222", marker="x", s=40, label="Numerical negative gradient")
    axes[1].set_xticks(positions, [f"y={label}\np={p:.2f}" for label, p in zip(labels, probability)])
    axes[1].axhline(0, color=POINT, lw=1)
    axes[1].set(title="Logistic loss: correction targets for the logit", xlabel="Code-defined binary observations", ylabel="y - p (not a probability update)")
    for ax in axes:
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.15)
    return fig, {
        "squared_max_absolute_gradient_error": float(np.max(np.abs(numeric - residual))),
        "logistic_max_absolute_gradient_error": float(np.max(np.abs(logistic_numeric - signals))),
        "classification_labels": labels.tolist(), "classification_logits": logits.tolist(),
        "classification_probabilities": probability.tolist(),
        "classification_negative_gradients": signals.tolist(),
    }


def boosting_vs_bagging(data, history):
    forest = RandomForestRegressor(
        n_estimators=30, max_depth=4, min_samples_leaf=3,
        bootstrap=True, random_state=42, n_jobs=1,
    ).fit(data.x_train, data.y_train)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), layout="constrained")
    for ax in axes[0]:
        ax.axis("off")
        ax.set(xlim=(0, 1), ylim=(0, 1))
    for i in range(3):
        y = 0.78 - 0.3 * i
        axes[0, 0].text(0.05, y, f"Tree {i + 1}: fits y\nbootstrap; one feature available", fontsize=9,
                        bbox={"facecolor": "#edf5f6", "edgecolor": MODEL})
        axes[0, 0].annotate("", (0.82, 0.47), (0.57, y), arrowprops={"arrowstyle": "->", "color": POINT})
    axes[0, 0].text(0.84, 0.47, "Average", ha="center", fontsize=11)
    axes[0, 0].set_title("13 | Random Forest: members fit without earlier predictions")
    for i, label in enumerate(("F0", "h1 from y-F0", "F1", "h2 from y-F1", "F2")):
        y = 0.93 - 0.2 * i
        axes[0, 1].text(0.5, y, label, ha="center", fontsize=11,
                        bbox={"facecolor": "#edf5f6", "edgecolor": MODEL})
        if i < 4:
            axes[0, 1].annotate("", (0.5, y - 0.15), (0.5, y - 0.035),
                                arrowprops={"arrowstyle": "->", "color": CORRECTION})
    axes[0, 1].set_title("Gradient Boosting: later targets depend on earlier additions")
    for tree in forest.estimators_[:3]:
        axes[1, 0].plot(data.x_grid[:, 0], tree.predict(data.x_grid), color=POINT, alpha=0.5, lw=1)
    axes[1, 0].plot(data.x_grid[:, 0], forest.predict(data.x_grid), color=MODEL, lw=2.5, label="Average of all 30 tree predictions")
    decorate(axes[1, 0], "Tree predictions live on the original target scale")
    for i, correction in enumerate(history.corrections_grid[:3], start=1):
        axes[1, 1].plot(data.x_grid[:, 0], correction, lw=1.7, label=f"Raw correction h{i}")
    axes[1, 1].axhline(0, color=POINT, lw=0.8)
    decorate(axes[1, 1], "First three learners predict signed corrections", "Correction scale")
    for ax in axes[1]:
        ax.legend(fontsize=8)
    member_predictions = np.stack([tree.predict(data.x_grid) for tree in forest.estimators_])
    record = {
        "hypothesis": "A forest prediction averages full target predictions; boosting adds signed corrections.",
        "configuration": {"n_estimators": 30, "max_depth": 4, "min_samples_leaf": 3,
                          "bootstrap": True, "max_features": 1.0, "seed": 42,
                          "data": "same univariate training rows as the boosting loop"},
        "result": {"max_absolute_mean_identity_error": float(np.max(np.abs(
            forest.predict(data.x_grid) - member_predictions.mean(axis=0)
        )))},
        "interpretation_candidate": "Inspect the member target scale and booster correction scale as different roles.",
        "limitation": "One input feature means feature subsampling cannot create feature diversity; no performance contest.",
        "review_status": "pending author review",
    }
    return fig, record


def decision_stump(data, history):
    residual = history.residuals_train[0]
    stump = DecisionTreeRegressor(max_depth=1, min_samples_leaf=3, random_state=42).fit(data.x_train, residual)
    threshold = float(stump.tree_.threshold[0])
    if stump.tree_.node_count == 1:
        raise RuntimeError("The synthetic teaching stump did not split; inspect the data.")
    left = data.x_train[:, 0] <= threshold
    left_value, right_value = float(residual[left].mean()), float(residual[~left].mean())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(data.x_train[:, 0], residual, color=POINT, s=15, alpha=0.5, label="Residual targets y - F0")
    ax.plot(data.x_grid[:, 0], stump.predict(data.x_grid), color=MODEL, lw=2.5, label="Fitted depth-1 correction")
    ax.axvline(threshold, color=CORRECTION, ls="--", label=f"Threshold = {threshold:.3f}")
    ax.axhline(0, color=POINT, lw=0.8)
    ax.text(0.03, 0.96, f"x <= {threshold:.3f}: correction {left_value:+.3f} (n={left.sum()})",
            transform=ax.transAxes, va="top", fontsize=10)
    ax.text(0.97, 0.04, f"x > {threshold:.3f}: correction {right_value:+.3f} (n={(~left).sum()})",
            transform=ax.transAxes, ha="right", fontsize=10)
    decorate(ax, "15 | One stump approximates the gradient with two leaf means", "Residual / raw correction")
    ax.legend(loc="lower left", fontsize=9)
    return fig, {"threshold": threshold, "left_correction": left_value, "right_correction": right_value}


def surface_explorer():
    """Generate a bounded 2D experiment and an offline stage slider."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rng = np.random.default_rng(42)
    x = rng.uniform(-2.5, 2.5, size=(600, 2))
    truth = 1.5 * np.sin(x[:, 0]) + 0.4 * x[:, 1] ** 2 + 0.6 * x[:, 0] * x[:, 1]
    y = truth + rng.normal(0, 0.3, len(x))
    from sklearn.model_selection import train_test_split
    train, validation = train_test_split(np.arange(len(x)), test_size=0.25, random_state=42)
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3,
        min_samples_leaf=4, loss="squared_error", subsample=1.0, random_state=42,
    ).fit(x[train], y[train])
    axis = np.linspace(-2.5, 2.5, 32)
    x1, x2 = np.meshgrid(axis, axis)
    grid = np.column_stack([x1.ravel(), x2.ravel()])
    true_surface = 1.5 * np.sin(x1) + 0.4 * x2 ** 2 + 0.6 * x1 * x2
    stages = [0, 1, 5, 10, 25, 50, 100]
    surfaces = [np.full_like(true_surface, y[train].mean())]
    for stage, prediction in enumerate(model.staged_predict(grid), start=1):
        if stage in stages:
            surfaces.append(prediction.reshape(x1.shape))
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]],
                        subplot_titles=("Known noiseless function", "Piecewise boosted prediction"))
    z_range = [float(min(true_surface.min(), min(s.min() for s in surfaces))),
               float(max(true_surface.max(), max(s.max() for s in surfaces)))]
    for column, surface in ((1, true_surface), (2, surfaces[0])):
        fig.add_trace(go.Surface(x=x1, y=x2, z=surface, colorscale="Viridis",
                                 cmin=z_range[0], cmax=z_range[1], showscale=column == 2,
                                 name="True function" if column == 1 else "Boosted prediction"), row=1, col=column)
    fig.frames = [
        go.Frame(name=str(stage), data=[go.Surface(z=surface)], traces=[1],
                 layout=go.Layout(title_text=f"14 | Two-feature boosting surface — stage {stage}"))
        for stage, surface in zip(stages, surfaces)
    ]
    scene = {"xaxis_title": "x1", "yaxis_title": "x2", "zaxis_title": "y",
             "zaxis": {"range": z_range}, "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 1.1}}}
    fig.update_layout(
        title="14 | Two-feature boosting surface — stage 0",
        template="plotly_white", height=650, margin={"l": 20, "r": 20, "b": 90},
        scene=scene, scene2=scene,
        sliders=[{"currentvalue": {"prefix": "Stage: "}, "steps": [
            {"label": str(stage), "method": "animate",
             "args": [[str(stage)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0}}]} for stage in stages
        ]}],
    )
    val_prediction = model.predict(x[validation])
    return fig, {
        "hypothesis": "Additive trees can approximate a nonlinear surface with feature interactions.",
        "configuration": {"n_samples": 600, "train": 450, "validation": 150, "seed": 42,
                          "domain": "Uniform([-2.5,2.5]^2)", "noise_sd": 0.3,
                          "true_function": "1.5*sin(x1) + 0.4*x2**2 + 0.6*x1*x2",
                          "n_estimators": 100, "learning_rate": 0.1, "max_depth": 3,
                          "min_samples_leaf": 4, "grid_shape": [32, 32], "stages": stages},
        "result": {"validation_rmse": float(np.sqrt(np.mean((y[validation] - val_prediction) ** 2)))},
        "interpretation_candidate": "Inspect the evolving piecewise surface against known truth, especially near domain boundaries.",
        "limitation": "One synthetic surface and split; no extrapolation or model comparison evidence.",
        "review_status": "pending author review",
    }


def stage_explorer(data, history):
    """Use frames so a slider changes the prediction, residuals, and latest correction together."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Current function", "Current train and validation residuals", "Latest applied correction (eta x h)"))
    x = data.x_grid[:, 0]
    fig.add_trace(go.Scatter(x=data.x_train[:, 0], y=data.y_train, mode="markers",
                            marker={"size": 4, "color": POINT}, name="Train observations"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=data.truth_grid, line={"dash": "dash", "color": "#222222"}, name="Known truth"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=history.predictions_grid[0], line={"color": MODEL}, name="Current prediction"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.x_train[:, 0], y=history.residuals_train[0], mode="markers",
                            marker={"size": 4, "color": MODEL}, name="Train residual"), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.x_validation[:, 0], y=history.residuals_validation[0], mode="markers",
                            marker={"size": 5, "color": VALIDATION}, name="Validation residual"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=np.zeros(len(x)), line={"color": CORRECTION}, name="Latest applied correction"), row=3, col=1)
    frames = []
    for stage in range(len(history.predictions_grid)):
        latest = np.zeros(len(x)) if stage == 0 else history.learning_rate * history.corrections_grid[stage - 1]
        frames.append(go.Frame(
            name=str(stage),
            data=[go.Scatter(y=history.predictions_grid[stage]),
                  go.Scatter(y=history.residuals_train[stage]),
                  go.Scatter(y=history.residuals_validation[stage]),
                  go.Scatter(y=latest)],
            traces=[2, 3, 4, 5],
            layout=go.Layout(title_text=f"16 | Stage {stage}: train RMSE {history.train_rmse[stage]:.3f}, validation {history.validation_rmse[stage]:.3f}"),
        ))
    fig.frames = frames
    fig.update_layout(
        title=f"16 | Stage 0: train RMSE {history.train_rmse[0]:.3f}, validation {history.validation_rmse[0]:.3f}",
        template="plotly_white", height=850, margin={"b": 100},
        sliders=[{"currentvalue": {"prefix": "Boosting stage: "},
                  "steps": [{"label": str(stage), "method": "animate",
                             "args": [[str(stage)], {"mode": "immediate", "frame": {"duration": 0, "redraw": False},
                                                    "transition": {"duration": 0}}]}
                            for stage in range(len(frames))]}],
    )
    fig.update_xaxes(range=[-3, 3], title_text="x", row=3, col=1)
    fig.update_yaxes(range=[float(data.y_train.min() - 0.5), float(data.y_train.max() + 0.5)], title_text="y", row=1, col=1)
    extent = 1.08 * max(np.abs(history.residuals_train).max(), np.abs(history.residuals_validation).max())
    fig.update_yaxes(range=[-extent, extent], zeroline=True, title_text="y - F", row=2, col=1)
    correction_extent = 1.08 * np.abs(history.learning_rate * history.corrections_grid).max()
    fig.update_yaxes(range=[-correction_extent, correction_extent], zeroline=True, title_text="eta x h", row=3, col=1)
    return fig


def write_offline_html(fig, path):
    """Embed JavaScript; no server, CDN, or static-image export dependency is needed."""
    fig.write_html(path, include_plotlyjs=True, full_html=True, auto_open=False,
                   config={"responsive": True, "displaylogo": False})


def experiment_record(data, history, rates, depths, flexible, gradient_record, stump_record):
    def experiment(hypothesis, configuration, result, interpretation, limitation):
        return {"hypothesis": hypothesis, "configuration": configuration, "result": result,
                "interpretation_candidate": interpretation, "limitation": limitation,
                "review_status": "pending author review"}

    return {
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "library_common": {"loss": "squared_error", "subsample": 1.0, "seed": 42,
                           "n_iter_no_change": None, "criterion": "friedman_mse", "init": "training mean"},
        "dataset": {"source": "synthetic, code-defined", "n_samples": 300,
                    "function": "2*sin(x) + 0.4*x**2", "noise_sd": 0.35, "seed": 42,
                    "train": len(data.y_train), "validation": len(data.y_validation),
                    "reserved_test": len(data.reserved_indices), "reserved_test_scored": False,
                    "split": "20% reserved test, then 25% of development for validation; both random_state=42"},
        "experiments": {
            "stage_history": experiment(
                "Residual-mean tree corrections reduce full-data training squared error.",
                {"n_estimators": 100, "learning_rate": history.learning_rate, "max_depth": 2, "min_samples_leaf": 3},
                {"best_stage": history.best_stage,
                 "train_rmse_at_best": float(history.train_rmse[history.best_stage]),
                 "validation_rmse_at_best": float(history.validation_rmse[history.best_stage]),
                 "train_rmse_at_final": float(history.train_rmse[-1]),
                 "validation_rmse_at_final": float(history.validation_rmse[-1])},
                "Review residual structure and the difference between training and validation trajectories.",
                "Educational loop uses sklearn weak trees and a fixed seed per tree; not general library parity."),
            "learning_rate": experiment(
                "Smaller learning rates can require more stages at fixed capacity.",
                {"rates": [p["learning_rate"] for p in rates], "n_estimators": 300, "max_depth": 2, "min_samples_leaf": 3},
                [path_summary(p) for p in rates],
                "Compare equal-stage predictions and measured minima; a boundary minimum may warrant a larger budget.",
                "One split and a fixed budget; validation minima are selected, not unbiased test results."),
            "tree_depth": experiment(
                "Greater depth changes correction capacity and the train/validation gap.",
                {"depths": [p["max_depth"] for p in depths], "n_estimators": 100, "learning_rate": 0.1, "min_samples_leaf": 3},
                [path_summary(p) for p in depths],
                "Inspect each fitted function together with its measured error; do not rank depths universally.",
                "One univariate synthetic function cannot demonstrate multi-feature interaction capacity."),
            "flexible_path": experiment(
                "Training error may keep falling after validation reaches its minimum.",
                {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 500, "min_samples_leaf": 1},
                path_summary(flexible),
                "A lower final training error and higher final validation error would be consistent with fitting training-specific patterns.",
                "Scanning a full path is not computational early stopping; no noise or data adjustment was made."),
            "gradient_checks": experiment(
                "Finite differences coincide with analytic loss gradients to numerical tolerance.",
                {"central_difference_step": 1e-5, "squared_observations": 18, "logistic_observations": 6},
                gradient_record, "Inspect agreement and note that logistic signals update raw logits.",
                "Code-defined observations; no classifier training or calibration experiment."),
            "stump": experiment(
                "A single tree split approximates residuals with region means.",
                {"max_depth": 1, "min_samples_leaf": 3, "target": "y_train - mean(y_train)"},
                stump_record, "Review the threshold and fitted leaf means on the residual scale.",
                "One stage on one synthetic dataset; no generalization evidence from the split itself."),
        },
        "interpretations": "All empirical interpretation candidates require author review.",
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Generate everything without opening figure windows.")
    parser.add_argument("--skip-gifs", action="store_true", help="Skip the two GIFs for a faster static run.")
    parser.add_argument("--skip-html", action="store_true", help="Skip optional Plotly output and its 2D experiment.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--dpi", type=int, default=110, help="PNG/GIF resolution; integer from 50 to 200.")
    args = parser.parse_args(argv)
    if not 50 <= args.dpi <= 200:
        parser.error("--dpi must be between 50 and 200.")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.headless:
        plt.switch_backend("Agg")
    output = args.output_dir.resolve()
    figures, gifs = output / "figures", output / "gifs"
    figures.mkdir(parents=True, exist_ok=True)
    gifs.mkdir(parents=True, exist_ok=True)
    print("Fitting deterministic synthetic models...", flush=True)
    data = make_regression_data()
    history = fit_history(data)
    rates = [fit_library_path(data, learning_rate=rate) for rate in (1.0, 0.3, 0.1, 0.03)]
    depths = [fit_library_path(data, max_depth=depth, n_estimators=100) for depth in (1, 2, 4, 8)]
    flexible = fit_library_path(data, max_depth=4, n_estimators=500, min_samples_leaf=1)
    generated = []

    def save(fig, name):
        save_figure(fig, figures / name, args.dpi)
        generated.append(f"figures/{name}")
        print(f"Created {name}", flush=True)

    style = {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
             "figure.facecolor": "white", "axes.facecolor": "white", "text.color": "#202939",
             "axes.labelcolor": "#202939", "xtick.color": "#202939", "ytick.color": "#202939",
             "savefig.facecolor": "white"}
    with plt.rc_context(style):
        save(initial_model(data, history), "01_initial_model.png")
        save(residual_learning(data, history), "02_residual_learning.png")
        save(one_boosting_step(data, history), "03_one_boosting_step.png")
        if not args.skip_gifs:
            sequential_stages = sequential_boosting(data, history, gifs / "04_sequential_boosting.gif", args.dpi)
            residual_stages = residual_evolution(data, history, gifs / "05_residual_evolution.gif", args.dpi)
            generated.extend(["gifs/04_sequential_boosting.gif", "gifs/05_residual_evolution.gif"])
            print("Created both stage GIFs.", flush=True)
        save(additive_prediction(data, history), "06_additive_prediction.png")
        save(learning_rate_comparison(data, rates), "07_learning_rate.png")
        save(tree_complexity(data, depths), "08_tree_complexity.png")
        save(train_validation_loss(flexible), "09_train_validation_loss.png")
        save(functional_gradient_descent(), "10_functional_gradient_descent.png")
        save(negative_gradient_landscape(data, history), "11_negative_gradient.png")
        fig, gradient_record = pseudo_residuals(data, history)
        save(fig, "12_pseudo_residuals.png")
        fig, forest_record = boosting_vs_bagging(data, history)
        save(fig, "13_boosting_vs_bagging.png")
        fig, stump_record = decision_stump(data, history)
        save(fig, "15_decision_stump.png")
    record = experiment_record(data, history, rates, depths, flexible, gradient_record, stump_record)
    record["experiments"]["forest_mechanics"] = forest_record
    if not args.skip_html:
        fig, surface_record = surface_explorer()
        write_offline_html(fig, figures / "14_boosting_surface_3d.html")
        write_offline_html(stage_explorer(data, history), figures / "16_stage_explorer.html")
        generated.extend(["figures/14_boosting_surface_3d.html", "figures/16_stage_explorer.html"])
        record["experiments"]["surface"] = surface_record
        print("Created both offline HTML explorers.", flush=True)
    if not args.skip_gifs:
        record["animation_stages"] = {"sequential": sequential_stages, "residual": residual_stages}
    save_history(output / "stage_history.npz", data, history)
    record["artifacts"] = [{"path": name, "bytes": (output / name).stat().st_size,
                            "classification": "regenerable artifact"} for name in generated]
    record_path = output / "visual_experiments.json"
    record_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    best = history.best_stage
    print(f"Dataset: 300 synthetic rows | train {len(data.y_train)} | validation {len(data.y_validation)} | reserved test 60 (unscored)")
    print(f"Educational loop: eta={history.learning_rate}, trees={len(history.trees)}, depth=2, min leaf=3")
    print(f"Best validation stage: {best} | train RMSE {history.train_rmse[best]:.6f} | validation RMSE {history.validation_rmse[best]:.6f}")
    for label, paths in (("Learning-rate", rates), ("Depth", depths), ("Flexible", [flexible])):
        for path in paths:
            summary = path_summary(path)
            print(f'{label}: eta={path["learning_rate"]}, depth={path["max_depth"]}, best stage={summary["best_stage"]}, '
                  f'validation RMSE={summary["validation_rmse_at_best"]:.6f}')
    print(f"Created {len(generated)} visual assets plus stage_history.npz and visual_experiments.json under {output}")
    print("All interpretation candidates are pending author review.")
    if not args.headless:
        from matplotlib import get_backend
        if get_backend().lower() in ("agg", "pdf", "ps", "svg", "pgf", "template", "cairo"):
            print("Noninteractive Matplotlib backend: open the saved images or HTML locally.")
        else:
            for name in ("03_one_boosting_step.png", "07_learning_rate.png", "09_train_validation_loss.png"):
                fig, ax = plt.subplots(figsize=(12, 5))
                with Image.open(figures / name) as image:
                    ax.imshow(np.asarray(image))
                ax.axis("off")
                fig.tight_layout()
            plt.show()
            plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
