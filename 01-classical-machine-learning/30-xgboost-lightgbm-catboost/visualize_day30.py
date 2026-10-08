"""Generate thirteen Day 30 learning views locally; README files are never edited.

Run from this directory: python visualize_day30.py [--all | --only split_gain]
Core requirements come from the repository pyproject.toml. Optional model views
use the boosting extra; Plotly is needed only for the offline 3D explorer.
Generated visuals and experiment_records.json remain ignored until author review.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification, make_moons
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from from_scratch import finite_vector, leaf_weight, logistic_derivatives, sigmoid, split_gain

RANDOM_STATE = 42
DPI = 120
STEP_MS = 1100
OUTPUT_DIR = Path(__file__).resolve().parent / "visuals"
OPTIONAL_AVAILABLE = {name: importlib.util.find_spec(name) is not None
                      for name in ("xgboost", "lightgbm", "catboost", "plotly")}
BLUE, TEAL, ORANGE, RED, GRAY = "#3565a3", "#087f8c", "#d97718", "#c43d4e", "#8793a3"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "figure.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})


class SkipVisualization(RuntimeError):
    """An optional dependency is unavailable; other views can still run."""


def require_optional(*names):
    missing = [name for name in names if not OPTIONAL_AVAILABLE[name]]
    if missing:
        extra = "dev,boosting" if any(name != "plotly" for name in missing) else "dev"
        raise SkipVisualization("Missing packages: " + ", ".join(missing)
                                + f'. From the repository root install: python -m pip install -e ".[{extra}]"')


def model_classes():
    require_optional("xgboost", "lightgbm", "catboost")
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
    except (ImportError, OSError) as error:
        raise SkipVisualization(f"Optional model library could not load: {error}") from error
    return XGBClassifier, LGBMClassifier, CatBoostClassifier


def make_models(rounds=60, depth=3):
    xgb, lgb, cat = model_classes()
    return {
        "XGBoost": xgb(n_estimators=rounds, max_depth=depth, learning_rate=0.08,
            objective="binary:logistic", tree_method="hist", grow_policy="depthwise",
            reg_lambda=1.0, n_jobs=1, random_state=RANDOM_STATE),
        "LightGBM": lgb(n_estimators=rounds, max_depth=depth, num_leaves=2**depth,
            learning_rate=0.08, objective="binary", reg_lambda=1.0, min_child_samples=10,
            deterministic=True, force_col_wise=True, n_jobs=1, verbosity=-1,
            random_state=RANDOM_STATE),
        "CatBoost": cat(iterations=rounds, depth=depth, learning_rate=0.08,
            loss_function="Logloss", l2_leaf_reg=1.0, boosting_type="Ordered",
            grow_policy="SymmetricTree", thread_count=1, random_seed=RANDOM_STATE,
            allow_writing_files=False, verbose=False),
    }


def documented_model_parameters(model):
    params = model.get_params()
    # XGBoost exposes NaN as its missing-value sentinel, not as a measured value.
    if isinstance(params.get("missing"), (float, np.floating)) and np.isnan(params["missing"]):
        params["missing"] = "NaN (library missing-value sentinel)"
    return params


def evidence(kind, hypothesis, configuration, result, interpretation, limitation):
    return {"evidence_type": kind, "hypothesis": hypothesis,
            "configuration": configuration, "result": result,
            "interpretation_candidate": interpretation, "limitations": limitation,
            "review_status": "pending author review"}


def heading(fig, title, message):
    fig.suptitle(title, fontsize=17, fontweight="bold")
    if hasattr(fig, "_day30_caption"):
        fig._day30_caption.set_text(message)
    else:
        fig._day30_caption = fig.text(0.5, 0.025, message, ha="center", va="bottom", fontsize=10, color="#384454")
    fig.subplots_adjust(top=0.81, bottom=0.20, wspace=0.30)


def axes_labels(ax, title, xlabel, ylabel):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.15)


def save_png(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=DPI)
    finally:
        plt.close(fig)


def save_gif(fig, draw, count, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        animation = FuncAnimation(fig, draw, frames=count, interval=STEP_MS,
                                  repeat=True, blit=False, cache_frame_data=False)
        animation.save(path, writer=PillowWriter(fps=1000 / STEP_MS), dpi=110)
    finally:
        plt.close(fig)


def ordered_statistics(categories, targets, permutation, prior=0.5, strength=2.0):
    """Smoothed permutation-prefix statistics; outputs align with original rows."""
    labels = finite_vector(targets, "targets")
    cats = np.asarray(categories)
    order = np.asarray(permutation)
    if (cats.ndim != 1 or len(cats) != len(labels) or order.ndim != 1
            or order.dtype.kind not in "iu" or not np.array_equal(np.sort(order), np.arange(len(labels)))):
        raise ValueError("categories and targets must align; permutation must contain each row once")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(prior) or not 0 <= prior <= 1:
        raise ValueError("targets must be binary and prior must lie in [0,1]")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be finite and positive")
    sums, counts, stats = {}, {}, np.empty(len(labels))
    details = []
    for row in order:
        category = cats[row]
        total, count = sums.get(category, 0.0), counts.get(category, 0)
        stats[row] = (total + strength * prior) / (count + strength)
        details.append({"row": int(row), "previous_sum": float(total), "previous_count": count,
                        "statistic": float(stats[row])})
        # Only after encoding the current row may its label enter future statistics.
        sums[category], counts[category] = total + labels[row], count + 1
    return stats, details


def goss_sample(gradients, top_rate=0.2, other_rate=0.2):
    """Conceptual GOSS sample; both fractions refer to the full row count."""
    g = finite_vector(gradients, "gradients")
    if not (0 < top_rate < 1 and 0 < other_rate < 1 and top_rate + other_rate <= 1):
        raise ValueError("rates must be positive and sum to at most one")
    high_count, sampled_count = int(len(g) * top_rate), int(len(g) * other_rate)
    if high_count < 1 or sampled_count < 1:
        raise ValueError("too few rows for the requested sampling fractions")
    order = np.argsort(-np.abs(g), kind="stable")
    high, remaining = order[:high_count], order[high_count:]
    low = np.random.default_rng(RANDOM_STATE).choice(remaining, sampled_count, replace=False)
    groups, weights = np.zeros(len(g), dtype=int), np.zeros(len(g))
    groups[high], groups[low] = 2, 1
    weights[high], weights[low] = 1.0, len(remaining) / sampled_count
    return groups, weights


def histogram_statistics(x, gradients, hessians, edges):
    values = finite_vector(x, "x")
    g, h, boundaries = (finite_vector(v, n) for v, n in
                         ((gradients, "gradients"), (hessians, "hessians"), (edges, "edges")))
    if (len(values) != len(g) or len(g) != len(h) or len(boundaries) < 2
            or np.any(np.diff(boundaries) <= 0) or np.any(h < 0)
            or values.min() < boundaries[0] or values.max() > boundaries[-1]):
        raise ValueError("aligned arrays, nonnegative Hessians and increasing covering edges required")
    bins = np.searchsorted(boundaries[1:-1], values, side="right")
    size = len(boundaries) - 1
    return bins, np.bincount(bins, weights=g, minlength=size), np.bincount(bins, weights=h, minlength=size)


def visualize_boosting_corrections(path):
    rng = np.random.default_rng(RANDOM_STATE)
    x = np.linspace(-3, 3, 100)
    y = np.sin(x) + rng.normal(0, 0.15, len(x))
    grid = np.linspace(-3, 3, 400)
    rate, baseline = 0.5, float(y.mean())
    train_history, grid_history, corrections, residuals = [np.full(len(x), baseline)], [np.full(len(grid), baseline)], [], []
    for _ in range(5):
        residual = y - train_history[-1]
        tree = DecisionTreeRegressor(max_depth=2, random_state=RANDOM_STATE).fit(x[:, None], residual)
        residuals.append(residual)
        corrections.append(tree.predict(grid[:, None]))
        train_history.append(train_history[-1] + rate * tree.predict(x[:, None]))
        grid_history.append(grid_history[-1] + rate * corrections[-1])
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    def draw(step):
        for ax in axes:
            ax.clear()
        prior = max(0, step - 1)
        axes[0].scatter(x, y, s=13, alpha=0.5, color=GRAY, label="Training observations")
        axes[0].plot(grid, grid_history[prior], color=BLUE, lw=2.4, label="Current model")
        axes[0].axhline(baseline, color=GRAY, ls="--", label="Initial constant")
        axes[0].vlines(x[::8], train_history[prior][::8], y[::8], color=ORANGE, alpha=0.7, label="Residuals")
        if step:
            axes[1].scatter(x, residuals[step - 1], s=13, alpha=0.6, color=GRAY, label="Targets: y - current")
            axes[1].plot(grid, corrections[step - 1], color=ORANGE, label="New depth-2 tree")
            axes[1].plot(grid, rate * corrections[step - 1], color=TEAL, ls="--", label="Applied correction: eta * tree")
        else:
            axes[1].text(0.5, 0.5, "Start with the training mean.\nNext: fit a tree to residuals.", ha="center", transform=axes[1].transAxes)
        axes[2].scatter(x, y, s=13, alpha=0.4, color=GRAY)
        axes[2].plot(grid, grid_history[prior], color=BLUE, ls="--", label="Before update")
        axes[2].plot(grid, grid_history[step], color=TEAL, lw=2.5, label="Updated ensemble")
        for ax, title in zip(axes, ("Current model", "+ New correction", "= Updated model")):
            axes_labels(ax, title, "Feature x", "Target / score" if ax is not axes[1] else "Residual / correction")
            ax.set_ylim(-1.6, 1.6)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, loc="upper left")
        heading(fig, f"01 | Sequential correction: stage {step}/5",
                r"$F_t=F_{t-1}+\eta f_t$, eta=0.5 | Synthetic experiment: each new tree fits residuals, not the entire target.")
    save_gif(fig, draw, 6, path)
    return evidence("synthetic experiment", "Residual-fitted shallow trees can improve an additive model.",
        {"seed": RANDOM_STATE, "generator": "sin(x) + Normal(0,0.15), 100 evenly spaced x in [-3,3]",
         "depth": 2, "learning_rate": rate, "stages": 5},
        {"train_mse_by_stage": [float(np.mean((y - p)**2)) for p in train_history]},
        "Inspect how each correction changes the current fit.", "In-sample one-dimensional example; no generalization claim.")


def visualize_gradient_hessian(path):
    z = np.linspace(-2.5, 4.5, 400)
    z0 = float(np.log(0.7 / 0.3))
    loss = np.logaddexp(0, -z)
    l0 = float(np.logaddexp(0, -z0))
    g, h = -0.3, 0.21
    tangent = l0 + g * (z - z0)
    quadratic = tangent + 0.5 * h * (z - z0)**2
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(z, loss, color=BLUE, lw=3, label="Exact logistic loss, y=1")
    ax.plot(z, tangent, color=ORANGE, ls="--", label="First order: tangent")
    ax.plot(z, quadratic, color=TEAL, ls="-.", label="Second order: local quadratic")
    ax.scatter([z0], [l0], color=RED, s=65, zorder=5)
    ax.annotate(f"Current point: z={z0:.3f}, p=0.70\ng=p-y=-0.30; h=p(1-p)=0.21",
                (z0, l0), xytext=(-2.15, 1.95), arrowprops={"arrowstyle": "->", "color": RED},
                bbox={"facecolor": "white", "edgecolor": GRAY})
    ax.annotate("Negative gradient points toward larger z", (z0 + 1.3, 0.11), (z0 - 0.1, 0.85),
                arrowprops={"arrowstyle": "->", "color": ORANGE})
    ax.set_ylim(-0.55, 3.1)
    axes_labels(ax, "Slope suggests direction; curvature scales a local Newton step", "Model logit z", "Loss for one observation")
    ax.legend(fontsize=10)
    heading(fig, "02 | Gradient and Hessian geometry",
            "Mathematical identity | Hessian measures local curvature, not statistical certainty. The quadratic is local.")
    save_png(fig, path)
    return evidence("mathematical identity", "First and second derivatives describe local logistic-loss geometry.",
        {"target": 1, "probability": 0.7}, {"logit": z0, "gradient": g, "hessian": h},
        "Compare the tangent and quadratic near the current point.", "Curvature is not a confidence interval; distant quadratic values need not match the loss.")


def animate_split_gain(path):
    x = np.arange(1.0, 11.0)
    y = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1])
    g, h = logistic_derivatives(y, np.zeros(len(y)))
    thresholds = (x[:-1] + x[1:]) / 2
    penalty, gamma = 1.0, 0.1
    gains = []
    for t in thresholds:
        left = x <= t
        gains.append(split_gain(g[left].sum(), h[left].sum(), g[~left].sum(), h[~left].sum(), penalty, gamma))
    best = int(np.argmax(gains))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    def draw(step):
        for ax in axes:
            ax.clear()
        done = step == len(thresholds)
        index = best if done else step
        revealed = len(thresholds) if done else index + 1
        t = thresholds[index]
        left = x <= t
        axes[0].scatter(x[left], y[left], color=BLUE, s=70, label="Left child")
        axes[0].scatter(x[~left], y[~left], color=ORANGE, s=70, label="Right child")
        axes[0].axvline(t, color=RED, lw=2)
        axes[0].text(0.02, 0.94, f"G_L={g[left].sum():.2f}; H_L={h[left].sum():.2f}\n"
                     f"G_R={g[~left].sum():.2f}; H_R={h[~left].sum():.2f}\n"
                     f"G_parent={g.sum():.2f}; H_parent={h.sum():.2f}", transform=axes[0].transAxes,
                     va="top", bbox={"facecolor": "white", "edgecolor": GRAY})
        axes[0].set_ylim(-0.2, 1.8)
        axes[0].legend(loc="lower right", fontsize=9)
        axes_labels(axes[0], f"Threshold {t:g}; current gain={gains[index]:.3f}", "Numerical feature", "Binary target")
        axes[1].plot(thresholds[:revealed], gains[:revealed], "o-", color=TEAL)
        axes[1].scatter([t], [gains[index]], color=ORANGE, s=80, zorder=5)
        axes[1].axhline(0, color=GRAY, ls="--", label="Positive gain needed")
        if done or index >= best:
            axes[1].scatter([thresholds[best]], [gains[best]], color=RED, marker="*", s=180, label="Best gain")
        if step == len(thresholds):
            axes[0].axvline(thresholds[best], color=TEAL, ls="--", lw=3)
            axes[1].set_title(f"Search complete: choose threshold {thresholds[best]:g}")
        else:
            axes[1].set_title("Gain curve grows as candidates are evaluated")
        axes[1].set_xlim(1, 10)
        axes[1].set_ylim(min(gains) - 0.2, max(gains) + 0.25)
        axes[1].set(xlabel="Candidate threshold", ylabel="Approximate regularized split gain")
        axes[1].legend(fontsize=9)
        heading(fig, "03 | XGBoost-style split search: computed synthetic gains",
                r"$\mathrm{Gain}=\frac{1}{2}[G_L^2/(H_L+\lambda)+G_R^2/(H_R+\lambda)-G^2/(H+\lambda)]-\gamma$"
                "\nlambda=1, gamma=0.1, initial p=0.5 | Approximate training objective, not held-out improvement.")
    save_gif(fig, draw, len(thresholds) + 1, path)
    return evidence("algorithmic mechanism", "Gradient/Hessian sums determine regularized split gains.",
        {"x": x.tolist(), "y": y.tolist(), "initial_logit": 0, "lambda": penalty, "gamma": gamma},
        {"thresholds": thresholds.tolist(), "gains": gains, "best_threshold": float(thresholds[best])},
        "Inspect why the largest positive gain is selected.", "Single feature and local second-order approximation, not a full XGBoost implementation.")


def visualize_regularization(path):
    penalties = np.linspace(0, 12, 150)
    G, H = -2.0, 1.0
    weights = -G / (H + penalties)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].plot(penalties, abs(weights), color=TEAL, lw=3)
    for penalty in (0, 1, 4, 10):
        weight = leaf_weight(G, H, penalty)
        axes[0].scatter([penalty], [abs(weight)], color=ORANGE)
        axes[0].annotate(f"lambda={penalty}: |w|={abs(weight):.2f}", (penalty, abs(weight)),
                         xytext=(4, 9), textcoords="offset points", fontsize=9)
    axes_labels(axes[0], r"$w^*=-G/(H+\lambda)$, G=-2, H=1", "L2 penalty lambda", "Absolute leaf score update")
    gamma = np.linspace(0, 2.5, 100)
    gains = np.array([split_gain(2, 1, -2, 1, 1, value) for value in gamma])
    axes[1].plot(gamma, gains, color=BLUE, lw=2.5)
    axes[1].axhline(0, color=RED, ls="--")
    axes[1].axvspan(0, 2, color=TEAL, alpha=0.12, label="Positive gain: candidate accepted")
    axes[1].axvspan(2, 2.5, color=RED, alpha=0.12, label="Gain <= 0: split rejected")
    axes_labels(axes[1], "Gamma penalizes the additional leaf", "Leaf-complexity penalty gamma", "Gain = 2 - gamma")
    axes[1].legend(fontsize=8)
    heading(fig, "04 | Regularization controls the size and existence of corrections",
            "Mathematical identity | Larger lambda shrinks leaf scores; larger gamma can reject a split. No generalization claim.")
    save_png(fig, path)
    return evidence("mathematical identity", "L2 shrinkage and leaf penalties control different decisions.",
        {"G": G, "H": H}, {"weights": {str(v): leaf_weight(G, H, v) for v in (0, 1, 4, 10)}, "gamma_zero_gain": 2},
        "Distinguish smaller leaf updates from suppressing extra leaves.", "Fixed aggregate statistics; no empirical regularization benchmark.")


def tree_position(path):
    return 0.5 + sum((-1 if bit == "L" else 1) / 2**(i + 2) for i, bit in enumerate(path)), -len(path)


def draw_tree(ax, splits, max_depth, gains=None, highlight=None, conditions=None):
    nodes = {""}
    for path in splits:
        nodes.update((path, path + "L", path + "R"))
    for path in sorted(nodes, key=lambda value: (len(value), value)):
        x, y = tree_position(path)
        if path:
            px, py = tree_position(path[:-1])
            ax.plot([px, x], [py, y], color=GRAY, lw=1.6, zorder=1)
        color = ORANGE if path == highlight else (BLUE if path in splits else TEAL)
        ax.scatter([x], [y], s=420 if len(path) < 4 else 240, color=color, zorder=3, edgecolor="white")
        label = path or "root"
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=8 if len(path) <= 3 else 6, zorder=4)
        if gains and path in gains:
            label_x = x
            alignment = "center"
            if len(path) >= 4:
                label_x += -0.025 if path.endswith("L") else 0.025
                alignment = "right" if path.endswith("L") else "left"
            ax.text(label_x, y - 0.24, f"gain {gains[path]:.1f}", ha=alignment, fontsize=9, color=color)
    if conditions:
        for depth, condition in conditions.items():
            ax.text(0.5, -depth + 0.39, condition, ha="center", fontsize=9,
                    bbox={"facecolor": "#eef3f8", "edgecolor": "none"})
    ax.set(xlim=(-0.07, 1.07), ylim=(-max_depth - 0.55, 0.7), xlabel="Branch position (illustrative)", ylabel="Tree depth")
    ax.set_yticks(-np.arange(max_depth + 1), labels=np.arange(max_depth + 1))
    ax.set_xticks([0, 1], labels=["Left", "Right"])
    ax.grid(axis="y", alpha=0.12)


def animate_tree_growth(path):
    level_splits = [[], [""], ["", "L", "R"], ["", "L", "R", "LL", "LR", "RL", "RR"]]
    leaf_splits = [[], [""], ["", "R"], ["", "R", "RR"], ["", "R", "RR", "RRL"]]
    gains = [{"": 5.0}, {"L": 1.3, "R": 4.8}, {"L": 1.3, "RL": 0.7, "RR": 3.9},
             {"L": 1.3, "RL": 0.7, "RRL": 2.6, "RRR": 0.4},
             {"L": 1.3, "RL": 0.7, "RRLL": 0.2, "RRLR": 0.3, "RRR": 0.4}]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    def draw(step):
        for ax in axes:
            ax.clear()
        draw_tree(axes[0], level_splits[min(step, 3)], 4)
        selected = max(gains[step], key=gains[step].get)
        draw_tree(axes[1], leaf_splits[step], 4, gains=gains[step], highlight=selected)
        axes[0].set_title("Level-wise: expand the next depth\nBalanced structure in this diagram")
        axes[1].set_title(f"Leaf-wise: next best leaf = {selected or 'root'}\nIllustrative gain values; orange = next candidate")
        heading(fig, f"05 | Two growth policies: step {step}/4",
                "Conceptual illustration; different leaf counts, not equal budgets. XGBoost also offers loss-guided growth.\n"
                "Best-first focuses capacity on larger loss reductions; deep local branches can fit noise. Not an observed benchmark.")
    save_gif(fig, draw, 5, path)
    return evidence("illustrative diagram", "Growth policy changes where a tree adds capacity.",
        {"gain_values": "manually chosen", "level_splits": level_splits, "leaf_splits": leaf_splits},
        {"illustrative_gain_by_step": gains}, "Compare expansion by depth with expansion by gain.",
        "Policies only: different leaf budgets and invented educational gains; no measured speed or loss comparison.")


def animate_histogram_binning(path):
    rng = np.random.default_rng(RANDOM_STATE)
    x = np.sort(rng.uniform(1, 5, 60))
    y = (x + rng.normal(0, 0.5, len(x)) > 3).astype(int)
    g, h = logistic_derivatives(y, np.zeros(len(x)))
    edges = np.linspace(1, 5, 7)
    bins, gb, hb = histogram_statistics(x, g, h, edges)
    gains = [split_gain(gb[:i].sum(), hb[:i].sum(), gb[i:].sum(), hb[i:].sum()) for i in range(1, 6)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    palette = plt.get_cmap("viridis", 6)
    stages = ["Raw unique values", "Quantize into six bins", "Aggregate G and H per bin", "Search five bin boundaries"]
    def draw(step):
        for ax in axes:
            ax.clear()
        colors = GRAY if step == 0 else palette(bins)
        axes[0].scatter(x, g, c=colors, s=30)
        if step == 0:
            thresholds = (x[:-1] + x[1:]) / 2
            axes[0].vlines(thresholds, -0.68, 0.68, color=GRAY, alpha=0.18)
        else:
            for i in range(6):
                axes[0].axvspan(edges[i], edges[i + 1], color=palette(i), alpha=0.14)
            axes[0].vlines(edges[1:-1], -0.68, 0.68, color=ORANGE, alpha=0.7)
        axes_labels(axes[0], stages[step], "Continuous feature value", "Observation gradient g = 0.5 - y")
        axes[0].set_ylim(-0.8, 0.9)
        if step < 2:
            axes[1].hist(x, bins=edges, color=TEAL, edgecolor="white")
            axes_labels(axes[1], "Many values share each bin", "Bin boundaries", "Observation count")
            axes[1].text(0.05, 0.92, "60 unique values: 59 exact-style candidates\n6 bins: 5 histogram boundaries",
                         va="top", transform=axes[1].transAxes, fontsize=10)
        elif step == 2:
            centers = (edges[:-1] + edges[1:]) / 2
            axes[1].bar(centers - 0.12, gb, width=0.23, color=BLUE, label="G = sum of gradients")
            axes[1].bar(centers + 0.12, hb, width=0.23, color=ORANGE, label="H = sum of Hessians")
            for center, gradient, curvature in zip(centers, gb, hb):
                axes[1].text(center, max(gradient, curvature) + 0.25, f"G={gradient:.1f}\nH={curvature:.2f}",
                             ha="center", fontsize=8)
            axes_labels(axes[1], "Bin statistics replace individual scans", "Bin center", "Aggregated derivative")
            axes[1].set_ylim(gb.min() - 1, max(hb.max(), gb.max()) + 3)
            axes[1].legend(loc="lower left", fontsize=8)
        else:
            axes[1].plot(edges[1:-1], gains, "o-", color=TEAL, lw=2.5)
            axes_labels(axes[1], "Candidate gains from cumulative bin sums", "Bin boundary", "Computed split gain, lambda=1")
        heading(fig, f"06 | Histogram split search: {step + 1}/4",
                "Algorithmic mechanism | Less threshold resolution, fewer candidates; runtime benefit is not measured here.\n"
                "Histogram training is central to LightGBM; modern XGBoost also supports histogram trees.")
    save_gif(fig, draw, 4, path)
    return evidence("algorithmic mechanism", "Quantization reduces the candidate set while preserving derivative totals.",
        {"seed": RANDOM_STATE, "n": 60, "edges": edges.tolist(), "initial_probability": 0.5},
        {"raw_candidates": 59, "bin_candidates": 5, "bin_gradients": gb.tolist(), "bin_hessians": hb.tolist(), "gains": gains},
        "Inspect the loss of threshold resolution and conservation of G/H.", "Equal-width bins illustrate the mechanism, not either library's full binning implementation.")


def visualize_goss(path):
    rng = np.random.default_rng(RANDOM_STATE)
    y = rng.integers(0, 2, 100)
    z = rng.normal(0, 1.8, 100)
    g, _ = logistic_derivatives(y, z)
    groups, weights = goss_sample(g)
    order = np.argsort(-abs(g), kind="stable")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    styles = [(2, RED, "Large-gradient rows: retained"), (1, TEAL, "Small-gradient rows: sampled"),
              (0, GRAY, "Small-gradient rows: not selected")]
    for group, color, label in styles:
        mask = groups[order] == group
        ax.scatter(np.arange(len(g))[mask], abs(g[order])[mask], color=color, s=40, label=label)
    ax.axvline(19.5, color=RED, ls="--")
    ax.text(0.39, 0.91, "20 high-gradient rows: weight 1\n20 sampled low-gradient rows: weight 4\n"
            "60 low-gradient rows: omitted\nReweight both gradient and Hessian contributions.",
            transform=ax.transAxes, va="top", bbox={"facecolor": "white", "edgecolor": GRAY})
    axes_labels(ax, "Large |g| means a larger logistic correction signal", "Observation rank by decreasing |gradient|", "Gradient magnitude |p - y|")
    ax.legend(loc="lower left", fontsize=9)
    heading(fig, "07 | Gradient-based One-Side Sampling (GOSS)",
            "Algorithmic mechanism | A computational sampling strategy; the fundamental boosting objective is unchanged.\n"
            "Large gradients are retained; sampled small gradients are reweighted. This is not LightGBM's default sampling mode.")
    save_png(fig, path)
    return evidence("algorithmic mechanism", "GOSS allocates sampling to large gradients while reweighting sampled smaller gradients.",
        {"seed": RANDOM_STATE, "n": 100, "top_rate": 0.2, "other_rate_of_all_rows": 0.2},
        {"retained": int((groups == 2).sum()), "sampled_low": int((groups == 1).sum()),
         "omitted": int((groups == 0).sum()), "small_weight": float(weights[groups == 1][0])},
        "Notice that low-gradient rows are not discarded indiscriminately.", "Conceptual fixed-fraction sampler; no estimation-error or speed benchmark.")


def visualize_efb(path):
    rng = np.random.default_rng(RANDOM_STATE)
    country, device = rng.integers(0, 4, 12), rng.integers(0, 2, 12)
    original = np.column_stack((np.eye(4, dtype=int)[country], np.eye(2, dtype=int)[device]))
    bundled = np.column_stack((country + 1, device + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(6), ["country_BR", "country_US", "country_DE", "country_FR", "device_mobile", "device_desktop"], rotation=35, ha="right", fontsize=9)
    axes[1].imshow(bundled, cmap="viridis", vmin=0, vmax=4, aspect="auto")
    axes[1].set_xticks([0, 1], ["country_bundle", "device_bundle"])
    for ax, matrix in zip(axes, (original, bundled)):
        for row in range(len(matrix)):
            for col in range(matrix.shape[1]):
                light_cell = matrix[row, col] == 0 or (ax is axes[1] and matrix[row, col] >= 3)
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=9,
                        color="black" if light_cell else "white")
        ax.set_yticks(range(12), labels=range(1, 13))
        ax.set_ylabel("Synthetic row")
        ax.set_xlabel("Feature column")
    axes[0].set_title("Before: mutually exclusive sparse columns")
    axes[1].set_title("After: distinct bin codes share a bundle")
    heading(fig, "08 | Exclusive Feature Bundling: a simplified demonstration",
            "Conceptual illustration | Six sparse columns become two bundles; original indicators are recoverable.\n"
            "Country codes 1..4 and device codes 1..2 are bin offsets, not ordinal feature values. Full EFB also manages conflicts.")
    fig.subplots_adjust(bottom=0.31)
    save_png(fig, path)
    return evidence("illustrative diagram", "Mutually exclusive active bins can share storage without collisions.",
        {"seed": RANDOM_STATE, "rows": 12, "features": 6, "conflicts": 0},
        {"original": original.tolist(), "bundled": bundled.tolist()},
        "Decode each bundled code back to the active indicator.", "Not generic integer encoding or a reproduction of LightGBM's conflict-aware bundling algorithm.")


def animate_catboost_ordered_statistics(path):
    cities = np.array(["Rio", "SP", "Rio", "BH", "Rio", "SP"])
    y = np.array([1, 0, 0, 1, 1, 1])
    order = np.random.default_rng(RANDOM_STATE).permutation(len(y))
    stats, details = ordered_statistics(cities, y, order, prior=0.5, strength=2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8))
    def draw(step):
        for ax in axes:
            ax.clear()
            ax.axis("off")
        if step == 0:
            rows = [[str(i + 1), city, str(target), "SELF INCLUDED" if i == 0 else "same category" if city == "Rio" else "other category"]
                    for i, (city, target) in enumerate(zip(cities, y))]
            table = axes[0].table(cellText=rows, colLabels=["Row", "City", "Target", "Contribution"], loc="center", colWidths=[0.12, 0.15, 0.15, 0.55])
            for i in range(len(y)):
                color = "#fbe2e5" if i == 0 else "#e2f2f0" if cities[i] == "Rio" else "#edf0f4"
                for column in range(4):
                    table[(i + 1, column)].set_facecolor(color)
            axes[0].set_title("Naive full-training encoding of row 1")
            axes[1].text(0.02, 0.82, "Rio target mean = (1 + 0 + 1) / 3 = 0.667\n\n"
                "TARGET LEAKAGE: row 1 supplies its own label.\n\n"
                "A singleton category would reveal its label directly.\n"
                "This self-inclusion is separate from leaking test labels.", va="top", fontsize=12)
        else:
            index = min(step - 1, len(y) - 1)
            row = order[index]
            previous = set(order[:index])
            rows, colors = [], []
            for pos, item in enumerate(order):
                if item == row:
                    status, color = "CURRENT: excluded", "#ffe7ca"
                elif item in previous and cities[item] == cities[row]:
                    status, color = "ALLOWED contributor", "#d6eeea"
                elif item in previous:
                    status, color = "Earlier, other city", "#e8edf4"
                else:
                    status, color = "NOT YET ALLOWED", "#f0f0f0"
                rows.append([str(pos + 1), str(item + 1), cities[item], str(y[item]), status])
                colors.append(color)
            table = axes[0].table(cellText=rows, colLabels=["Position", "Row", "City", "y", "Information boundary"],
                                  loc="center", colWidths=[0.13, 0.09, 0.12, 0.08, 0.57])
            for i, color in enumerate(colors):
                for col in range(5):
                    table[(i + 1, col)].set_facecolor(color)
            detail = details[index]
            axes[0].set_title(f"Seeded permutation: encode original row {row + 1}")
            formula = (f"Current city: {cities[row]} | fixed prior=0.5, a=2\n\n"
                "TS = (previous category sum + a * prior)\n"
                "       / (previous category count + a)\n\n"
                f"TS = ({detail['previous_sum']:g} + 2 * 0.5) / ({detail['previous_count']} + 2)\n"
                f"    = {detail['statistic']:.3f}\n\n"
                "Encode first; only then add this row's target.\n"
                "Previous targets from other cities do not contribute.")
            axes[1].text(0.02, 0.88, formula, va="top", fontsize=11)
            encoded = [f"row {int(item) + 1}: {stats[item]:.3f}" for item in order[:index + 1]]
            summary = "\n".join(", ".join(encoded[i:i + 3]) for i in range(0, len(encoded), 3))
            axes[1].text(0.02, 0.14, "Already encoded in this prefix:\n" + summary, fontsize=9)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        heading(fig, f"09 | CatBoost-style ordered target statistics: step {step}/6",
                "Algorithmic mechanism | Prior is fixed independently of labels; current and future rows never supply this statistic.\n"
                "Earlier means earlier in a random permutation, not historical time. Ordered statistics are distinct from ordered boosting.")
        if step == 0:
            fig.texts[-1].set_text("Algorithmic mechanism | Full-training target means include the current row's target.\n"
                                 "Next: exclude self and later rows using a seeded permutation and a fixed prior.")
    save_gif(fig, draw, 7, path)
    return evidence("algorithmic mechanism", "Prefix statistics exclude the current target and later permutation targets.",
        {"seed": RANDOM_STATE, "cities": cities.tolist(), "targets": y.tolist(), "permutation_zero_based": order.tolist(), "prior": 0.5, "strength": 2},
        {"naive_Rio_mean": float(y[cities == "Rio"].mean()), "ordered_statistics_original_row_order": stats.tolist(), "prefix_details": details},
        "Track exactly which labels are allowed when each row is encoded.", "One permutation and one statistic; not CatBoost's full categorical processing or ordered boosting.")


def visualize_symmetric_tree(path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    full = ["", "L", "R", "LL", "LR", "RL", "RR"]
    conditions = {0: "Same split: age > 40", 1: "Same split: income > 5000", 2: "Same split: tenure > 24"}
    draw_tree(axes[0], full, 3, conditions=conditions)
    axes[0].set_title("Depth-3 symmetric / oblivious tree\n2^3 = 8 leaf positions")
    asymmetric = ["", "L", "R", "RL"]
    draw_tree(axes[1], asymmetric, 3)
    axes[1].set_title("Conventional asymmetric tree\nDifferent branches may ask different questions")
    for node, label in {"": "age > 40", "L": "income > 5000", "R": "tenure > 24", "RL": "income > 7000"}.items():
        x, y = tree_position(node)
        axes[1].text(x, y + 0.26, label, ha="center", fontsize=9)
    heading(fig, "10 | CatBoost's symmetric tree structure",
            "Conceptual illustration | One condition per depth permits a compact bit-indexed path and restricts partition flexibility.\n"
            "Predictable traversal can support efficient inference; no latency is measured. CatBoost offers other growth policies too.")
    save_png(fig, path)
    return evidence("illustrative diagram", "Repeating the same split at each depth restricts possible partitions.",
        {"depth": 3, "conditions": conditions}, {"leaf_positions": 8},
        "Compare global per-depth questions with branch-specific questions.", "Illustrative rules, not fitted CatBoost trees; empty leaves are possible and no inference speed is measured.")


def create_3d_decision_surface(path):
    require_optional("plotly")
    try:
        import plotly.graph_objects as go
    except (ImportError, OSError) as error:
        raise SkipVisualization(f"Plotly could not load: {error}") from error
    x, y = make_moons(n_samples=300, noise=0.23, random_state=RANDOM_STATE)
    models = make_models(rounds=50, depth=2)
    bounds = [(float(x[:, i].min() - 0.3), float(x[:, i].max() + 0.3)) for i in range(2)]
    edges_x, edges_y = (np.linspace(low, high, 41) for low, high in bounds)
    centers_x, centers_y = (0.5 * (edges[:-1] + edges[1:]) for edges in (edges_x, edges_y))
    gx, gy = np.meshgrid(centers_x, centers_y)
    grid = np.column_stack((gx.ravel(), gy.ravel()))
    # Duplicated cell boundaries show flat tiles and vertical changes, not smooth interpolation.
    display_x, display_y = (np.repeat(edges, 2)[1:-1] for edges in (edges_x, edges_y))
    fig = go.Figure()
    configurations, result = {}, {}
    for index, (name, model) in enumerate(models.items()):
        model.fit(pd.DataFrame(x, columns=["feature_1", "feature_2"]), y)
        probabilities = model.predict_proba(pd.DataFrame(grid, columns=["feature_1", "feature_2"]))[:, 1].reshape(gx.shape)
        tiled = np.repeat(np.repeat(probabilities, 2, axis=0), 2, axis=1)
        fig.add_trace(go.Surface(x=display_x, y=display_y, z=tiled, visible=index == 0,
            colorscale="Viridis", cmin=0, cmax=1, opacity=0.88, name=name,
            colorbar={"title": "P(y=1)"}, hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>P(y=1)=%{z:.3f}<extra>" + name + "</extra>"))
        configurations[name] = documented_model_parameters(model)
        result[name] = {"probability_range": [float(probabilities.min()), float(probabilities.max())],
                        "grid_unique_probabilities": int(np.unique(probabilities).size)}
    fig.add_trace(go.Scatter3d(x=x[:, 0], y=x[:, 1], z=np.zeros(len(y)), mode="markers",
        name="Training labels on base plane", marker={"size": 3, "color": y, "colorscale": [[0, BLUE], [1, ORANGE]],
        "cmin": 0, "cmax": 1, "opacity": 0.8}, customdata=y,
        hovertemplate="Training row<br>x=%{x:.2f}<br>y=%{y:.2f}<br>Label=%{customdata}<extra></extra>"))
    buttons = []
    for index, name in enumerate(models):
        buttons.append({"label": name, "method": "update",
            "args": [{"visible": [i == index for i in range(3)] + [True]},
                     {"title.text": "11 | " + name + " piecewise-constant probability surface<br>"
                      "<sup>Synthetic experiment; rotate and zoom. Fixed shallow configurations, no performance ranking.</sup>"}]})
    fig.update_layout(title={"text": "11 | XGBoost piecewise-constant probability surface<br>"
        "<sup>Synthetic experiment; rotate and zoom. Fixed shallow configurations, no performance ranking.</sup>", "x": 0.04},
        scene={"xaxis_title": "Feature 1", "yaxis_title": "Feature 2", "zaxis_title": "Predicted P(y=1)",
               "zaxis": {"range": [0, 1]}, "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.1}}},
        updatemenus=[{"buttons": buttons, "x": 0.04, "y": 1.0, "xanchor": "left"}],
        annotations=[{"text": "Flat grid tiles approximate leaf regions; edges are sampled, not exact tree boundaries.<br>"
            "Training labels are on the base plane. Interpretations pending author review.",
            "x": 0.5, "y": -0.05, "xref": "paper", "yref": "paper", "showarrow": False}],
        margin={"t": 130, "b": 80}, height=760)
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={"displaylogo": False, "responsive": True}, div_id="day30-decision-surface")
    return evidence("synthetic experiment", "Shallow boosted trees form probability regions that can be inspected spatially.",
        {"seed": RANDOM_STATE, "generator": "make_moons, n=300, noise=0.23", "parameters": configurations,
         "grid_cells": [40, 40], "split": "training-only illustration, no evaluation metrics"}, result,
        "Rotate and compare the fitted partitions, without interpreting a library as superior.",
        "Coarse sampling approximates boundaries. Depth limits do not equalize tree shape or training; no performance or runtime evaluation.")


def visualize_capacity_overfitting(path):
    x, y = make_classification(n_samples=1000, n_features=12, n_informative=4,
        n_redundant=2, flip_y=0.22, class_sep=0.75, random_state=RANDOM_STATE)
    train, validation = train_test_split(np.arange(len(y)), test_size=0.55, stratify=y, random_state=RANDOM_STATE)
    capacity = [4, 8, 16, 32, 64, 128]
    train_loss, validation_loss, observed_leaves = [], [], []
    if OPTIONAL_AVAILABLE["lightgbm"]:
        try:
            from lightgbm import LGBMClassifier
        except (ImportError, OSError) as error:
            raise SkipVisualization(f"LightGBM could not load: {error}") from error
        name = "LightGBM"
        base = dict(n_estimators=120, learning_rate=0.1, max_depth=-1, min_child_samples=2,
                    min_child_weight=0.001, reg_lambda=0.0, n_jobs=1, verbosity=-1,
                    deterministic=True, force_col_wise=True, random_state=RANDOM_STATE)
    else:
        name = "scikit-learn HistGradientBoostingClassifier (fallback)"
        base = dict(max_iter=120, learning_rate=0.1, max_depth=None, min_samples_leaf=2,
                    l2_regularization=0.0, early_stopping=False, random_state=RANDOM_STATE)
    for leaves in capacity:
        model = (LGBMClassifier(num_leaves=leaves, **base) if name == "LightGBM"
                 else HistGradientBoostingClassifier(max_leaf_nodes=leaves, **base))
        model.fit(x[train], y[train])
        train_loss.append(float(log_loss(y[train], model.predict_proba(x[train])[:, 1])))
        validation_loss.append(float(log_loss(y[validation], model.predict_proba(x[validation])[:, 1])))
        if name == "LightGBM":
            observed_leaves.append(max(tree["num_leaves"] for tree in model.booster_.dump_model()["tree_info"]))
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(capacity, train_loss, "o-", color=BLUE, lw=2.5, label="Training log loss")
    ax.plot(capacity, validation_loss, "o-", color=ORANGE, lw=2.5, label="Validation log loss")
    for leaves, value in zip(capacity, validation_loss):
        ax.annotate(f"{value:.3f}", (leaves, value), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(capacity, labels=capacity)
    axes_labels(ax, name + " | All points are computed; no assumed curve shape",
                "Maximum leaves per tree (capacity limit)", "Binary log loss (lower is better)")
    ax.legend()
    heading(fig, "12 | Capacity and generalization: synthetic experiment",
            "120 fixed rounds, no early stopping; 450 train / 550 validation rows, noisy labels. No test set or model selection.\n"
            "Best-first growth can spend capacity on local noise; a larger leaf limit need not improve held-out loss. Review this run.")
    save_png(fig, path)
    return evidence("synthetic experiment", "Increasing leaf capacity can reduce training loss without improving validation loss.",
        {"seed": RANDOM_STATE, "generator": "make_classification, n=1000, features=12, informative=4, redundant=2, flip_y=0.22, class_sep=0.75",
         "train_indices": train.tolist(), "validation_indices": validation.tolist(),
         "model": name, "base_parameters": base, "capacity_limits": capacity},
        {"train_log_loss": train_loss, "validation_log_loss": validation_loss, "maximum_observed_leaves": observed_leaves},
        "Inspect the measured train/validation gaps, not a presumed monotonic validation curve.",
        "Single synthetic split; capacity caps can exceed realized leaves. No test evaluation, tuning, or general LightGBM benchmark.")


def visualize_calibration(path):
    x, y = make_classification(n_samples=3000, n_features=8, n_informative=5, n_redundant=1,
        weights=[0.65, 0.35], class_sep=0.85, flip_y=0.08, random_state=RANDOM_STATE)
    train, evaluation = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=RANDOM_STATE)
    models = make_models(rounds=100, depth=3)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1.4, 1.05, 1.25]})
    curves, metrics, params, probabilities = {}, {}, {}, {}
    axes[0].plot([0, 1], [0, 1], "--", color=GRAY, label="Perfect calibration")
    for (name, model), color in zip(models.items(), (BLUE, TEAL, ORANGE)):
        model.fit(pd.DataFrame(x[train], columns=[f"x{i}" for i in range(8)]), y[train])
        p = model.predict_proba(pd.DataFrame(x[evaluation], columns=[f"x{i}" for i in range(8)]))[:, 1]
        probabilities[name] = p
        fraction, average = calibration_curve(y[evaluation], p, n_bins=8, strategy="uniform")
        counts, _ = np.histogram(p, bins=np.linspace(0, 1, 9))
        axes[0].plot(average, fraction, "o-", color=color, label=name)
        axes[1].hist(p, bins=np.linspace(0, 1, 9), histtype="step", lw=2, color=color, label=name)
        metrics[name] = {"roc_auc": float(roc_auc_score(y[evaluation], p)),
                         "log_loss": float(log_loss(y[evaluation], p)), "brier": float(brier_score_loss(y[evaluation], p))}
        curves[name] = {"mean_predicted_probability": average.tolist(), "observed_fraction": fraction.tolist(), "bin_counts": counts.tolist()}
        params[name] = documented_model_parameters(model)
    # A strictly increasing logit stretch isolates ranking from probability distortion.
    reference = probabilities["XGBoost"]
    logits = np.log(reference / (1 - reference))
    distorted = sigmoid(2 * logits)
    distorted_metrics = {"roc_auc": float(roc_auc_score(y[evaluation], distorted)),
                         "log_loss": float(log_loss(y[evaluation], distorted)), "brier": float(brier_score_loss(y[evaluation], distorted))}
    fraction, average = calibration_curve(y[evaluation], distorted, n_bins=8)
    axes[0].plot(average, fraction, ":", color=RED, lw=2, label="XGBoost: logit stretch x2")
    rows = [[name, f"{value['roc_auc']:.3f}", f"{value['log_loss']:.3f}", f"{value['brier']:.3f}"] for name, value in metrics.items()]
    rows.append(["XGB stretch x2", f"{distorted_metrics['roc_auc']:.3f}", f"{distorted_metrics['log_loss']:.3f}", f"{distorted_metrics['brier']:.3f}"])
    axes[2].axis("off")
    table = axes[2].table(cellText=rows, colLabels=["Model", "AUC", "Log loss", "Brier"],
                         colWidths=[0.46, 0.18, 0.20, 0.18], loc="upper center", bbox=[0, 0.43, 1, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    axes[2].text(0, 0.29, "Same order, different probabilities:\n"
        "sigmoid(2 * logit(p)) preserves ranking.\n"
        "Log loss and Brier need not stay fixed.\n\n"
        "Bin averages hide uncertainty.\n"
        "These scores do not isolate calibration\n"
        "from discrimination, or rank libraries.", va="top", fontsize=9)
    axes_labels(axes[0], "Reliability: compare predicted and observed", "Mean predicted probability in bin", "Observed positive fraction")
    axes[0].set(xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=8, loc="upper left")
    axes_labels(axes[1], "Bin support matters", "Predicted probability", "Evaluation observations")
    heading(fig, "13 | Ranking and calibration answer different questions",
            "Single synthetic experiment. Not a general benchmark. Fixed models; 1500 train / 1500 held-out evaluation rows.\n"
            "No post-hoc calibration fitting, threshold tuning, or library selection. All interpretations require author review.")
    fig.subplots_adjust(wspace=0.38)
    save_png(fig, path)
    return evidence("synthetic experiment", "Ranking can remain fixed while probability quality changes under a monotonic distortion.",
        {"seed": RANDOM_STATE, "generator": "make_classification n=3000, features=8, informative=5, redundant=1, weights=[0.65,0.35], class_sep=0.85, flip_y=0.08",
         "train_indices": train.tolist(), "evaluation_indices": evaluation.tolist(), "parameters": params,
         "calibration_bins": 8, "distortion": "sigmoid(2 * logit(XGBoost probability))", "selection": "none; fixed configurations"},
        {"metrics": metrics, "reliability": curves, "distorted_XGBoost_metrics": distorted_metrics},
        "Review unchanged ranking versus changed probability scores in the distortion control; inspect bin support before interpreting reliability curves.",
        "One synthetic split; unequal effective model capacity, no confidence intervals, no fitted calibration or production reliability claim.")


VISUALIZATIONS = {
    "boosting": ("01_boosting_sequential_correction.gif", visualize_boosting_corrections,
        "Each new shallow tree fits residuals of the current ensemble; synthetic in-sample demonstration."),
    "gradient_hessian": ("02_gradient_hessian_geometry.png", visualize_gradient_hessian,
        "The gradient describes local slope; the Hessian describes curvature, not statistical confidence."),
    "split_gain": ("03_xgboost_split_gain.gif", animate_split_gain,
        "Computed gradient/Hessian sums select the largest positive regularized approximate gain."),
    "regularization": ("04_regularization_effect.png", visualize_regularization,
        "L2 shrinks leaf scores; gamma can reject an additional leaf."),
    "tree_growth": ("05_levelwise_vs_leafwise.gif", animate_tree_growth,
        "Illustrative gains show expansion by depth versus best-first expansion. Leaf budgets differ."),
    "histogram": ("06_histogram_split_search.gif", animate_histogram_binning,
        "Bin aggregation reduces candidate thresholds; this is not a runtime benchmark."),
    "goss": ("07_goss_sampling.png", visualize_goss,
        "Large-gradient rows are retained; sampled smaller-gradient contributions are reweighted."),
    "efb": ("08_exclusive_feature_bundling.png", visualize_efb,
        "Simplified exclusive bin codes share storage without inventing ordinal feature semantics."),
    "catboost_ordered": ("09_catboost_ordered_target_statistics.gif", animate_catboost_ordered_statistics,
        "Permutation-prefix statistics exclude the current row and later labels; the prior is fixed."),
    "symmetric_tree": ("10_catboost_symmetric_tree.png", visualize_symmetric_tree,
        "A symmetric tree repeats one split per depth; depth three has eight leaf positions."),
    "decision_surface": ("11_decision_surface_3d.html", create_3d_decision_surface,
        "Rotate the offline surface and switch fixed shallow models to inspect piecewise-constant regions."),
    "capacity": ("12_capacity_overfitting.png", visualize_capacity_overfitting,
        "Measured synthetic training and validation loss show how capacity changes held-out behavior."),
    "calibration": ("13_calibration_comparison.png", visualize_calibration,
        "Held-out synthetic reliability curves and a monotonic distortion distinguish ranking from probability quality."),
}
# Edit these switches for repeated local study; --only and --all override them.
RUN = {key: True for key in VISUALIZATIONS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Generate all thirteen requested views")
    group.add_argument("--only", nargs="+", choices=tuple(VISUALIZATIONS), help="Generate selected views by concept name")
    args = parser.parse_args(argv)
    selected = args.only or [key for key in VISUALIZATIONS if args.all or RUN[key]]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records, status, failures = {}, [], []
    for key in selected:
        filename, function, description = VISUALIZATIONS[key]
        try:
            records[key] = function(OUTPUT_DIR / filename)
            status.append(("OK", key, filename, ""))
        except SkipVisualization as error:
            status.append(("SKIP", key, filename, str(error)))
        except Exception as error:
            # Keep other lessons available, but return failure for unexpected rendering bugs.
            status.append(("ERROR", key, filename, f"{type(error).__name__}: {error}"))
            failures.append(key)
        finally:
            plt.close("all")
    environment = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "matplotlib", "scikit-learn", "Pillow", "xgboost", "lightgbm", "catboost", "plotly"):
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = "not installed"
    # Per-run metadata names avoid overwriting the record of an earlier --all run.
    record_name = "experiment_records.json" if len(selected) == len(VISUALIZATIONS) else "experiment_records_" + "_".join(selected) + ".json"
    record_path = OUTPUT_DIR / record_name
    record_path.write_text(json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": RANDOM_STATE, "environment": environment, "records": records,
        "status": [{"state": state, "visualization": key, "file": filename, "reason": reason}
                   for state, key, filename, reason in status]}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nGenerated visualizations:\n")
    for state, _, filename, reason in status:
        print(f"[{state}] {filename}")
        if reason:
            print(f"Reason: {reason}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Ignored evidence record: {record_path.name}")
    print("All interpretations and public-preview selection remain pending author review.")
    print("\nMarkdown snippet for manual review (paths relative to Day 30 README):")
    print("These assets are ignored. Select and deliberately unignore previews before publishing links.\n")
    print("## Visual intuition\n")
    for state, key, filename, _ in status:
        if state != "OK":
            continue
        title = key.replace("_", " ").title()
        print(f"### {title}\n")
        if filename.endswith(".html"):
            print(f"[Open the offline {title.lower()} explorer](visuals/{filename})\n")
        else:
            print(f"![{title}](visuals/{filename})\n")
        print(VISUALIZATIONS[key][2] + "\n")
    print("Synthetic results and illustrative diagrams are educational demonstrations; interpretations pending author review.\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
