"""Day 23 visual laboratory: synthetic ranking, decisions, and calibration.

Run from the repository root:
    python 01-classical-machine-learning/23-classification-metrics-ii/visualizations.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from from_scratch import threshold_metrics

BLUE = "#2563a6"
ORANGE = "#da7b2f"
GREEN = "#18856a"
RED = "#c44848"
PURPLE = "#8152a2"
INK = "#233144"
GRID = "#e4e9ee"
THRESHOLDS = np.linspace(0.0, 1.0, 101)


def style():
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5.4),
            "figure.dpi": 120,
            "savefig.dpi": 170,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.7,
        }
    )


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_splits(seed=23):
    """One synthetic population; calibration, policy, and test sets never overlap."""
    x, y = make_classification(
        n_samples=6000, n_features=12, n_informative=5, n_redundant=2,
        weights=[0.95, 0.05], class_sep=1.3, flip_y=0.01,
        random_state=seed,
    )
    x_train, x_other, y_train, y_other = train_test_split(
        x, y, test_size=0.5, stratify=y, random_state=seed
    )
    x_cal, x_rest, y_cal, y_rest = train_test_split(
        x_other, y_other, test_size=2 / 3, stratify=y_other,
        random_state=seed + 1,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, test_size=0.5, stratify=y_rest,
        random_state=seed + 2,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x_train, y_train)
    scores = {
        "calibration": model.predict_proba(x_cal)[:, 1],
        "validation": model.predict_proba(x_val)[:, 1],
        "test": model.predict_proba(x_test)[:, 1],
    }
    labels = {"calibration": y_cal, "validation": y_val, "test": y_test}
    if any(np.any((p < 0) | (p > 1)) for p in scores.values()):
        raise AssertionError("Model probabilities must lie in [0, 1]")
    return labels, scores


def score_summary(y, p):
    return {
        "roc_auc": roc_auc_score(y, p),
        "ap": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
    }


def threshold_table(y, p, thresholds=THRESHOLDS):
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if (y.ndim != 1 or p.ndim != 1 or len(y) == 0 or len(y) != len(p)
            or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2):
        raise ValueError("Expected equal-length binary labels containing both classes")
    if (not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or thresholds.ndim != 1 or len(thresholds) == 0
            or not np.isfinite(thresholds).all()
            or np.any((thresholds < 0) | (thresholds > 1))):
        raise ValueError("Scores and thresholds must be finite values in [0, 1]")
    rows = []
    for t in thresholds:
        m = threshold_metrics(y, p, float(t))
        m["threshold"] = float(t)
        m["fpr"] = m["fp"] / (m["fp"] + m["tn"])
        m["specificity"] = 1 - m["fpr"]
        rows.append(m)
    return rows


def choose_policies(rows, min_precision=0.8, cost_fp=1.0, cost_fn=9.0,
                    capacity=100):
    if (not rows or not np.isfinite([min_precision, cost_fp, cost_fn]).all()
            or not 0 < min_precision <= 1 or cost_fp < 0 or cost_fn < 0
            or capacity < 1):
        raise ValueError("Invalid policy constraints or costs")
    f1 = max(rows, key=lambda r: (r["f1"], r["threshold"]))
    feasible = [
        r for r in rows if r["alerts"] > 0 and r["precision"] >= min_precision
    ]
    precision = max(
        feasible, key=lambda r: (r["recall"], r["precision"], r["threshold"])
    ) if feasible else None
    cost = min(
        rows, key=lambda r: (cost_fp * r["fp"] + cost_fn * r["fn"],
                             -r["threshold"])
    )
    capacity_rows = [r for r in rows if r["alerts"] <= capacity]
    capacity_choice = max(
        capacity_rows, key=lambda r: (r["recall"], r["precision"],
                                     r["threshold"])
    )
    assert f1["f1"] == max(r["f1"] for r in rows)
    assert precision is None or precision["precision"] >= min_precision
    return {
        "f1": f1, "precision": precision, "cost": cost,
        "capacity": capacity_choice,
    }


def operating_rates(m):
    return m["fp"] / (m["fp"] + m["tn"]), m["recall"]


def draw_score_distribution(ax, y, p, t, *, title="Score distribution"):
    bins = np.linspace(0, 1, 25)
    ax.hist(p[y == 0], bins=bins, alpha=0.67, color=BLUE, label="Actual 0")
    ax.hist(p[y == 1], bins=bins, alpha=0.78, color=ORANGE, label="Actual 1")
    ax.axvline(t, color=RED, lw=2.2, label=f"Threshold {t:.2f}")
    ax.set(xlim=(0, 1), xlabel="Predicted probability", ylabel="Cases",
           title=title)
    ax.legend(loc="upper center", fontsize=8)


def small_visible_sample(y, p, seed=23):
    rng = np.random.default_rng(seed)
    pos = rng.choice(np.flatnonzero(y == 1), size=20, replace=False)
    neg = rng.choice(np.flatnonzero(y == 0), size=45, replace=False)
    idx = np.r_[pos, neg]
    return y[idx], p[idx]


def plot_threshold_sweep(y, p, out):
    small_y, small_p = small_visible_sample(y, p)
    jitter = np.random.default_rng(23).normal(0, 0.075, len(small_y))
    thresholds = np.linspace(0.95, 0.05, 23)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"TP": GREEN, "FP": RED, "FN": ORANGE, "TN": BLUE}

    def update(i):
        ax.clear()
        t = thresholds[i]
        predicted = small_p >= t
        groups = {
            "TP": (small_y == 1) & predicted,
            "FP": (small_y == 0) & predicted,
            "FN": (small_y == 1) & ~predicted,
            "TN": (small_y == 0) & ~predicted,
        }
        for name, mask in groups.items():
            ax.scatter(small_p[mask], small_y[mask] + jitter[mask],
                       s=52, color=colors[name], alpha=0.84, label=name,
                       edgecolor="white", linewidth=0.5)
        ax.axvline(t, color=INK, lw=2, linestyle="--")
        m = threshold_metrics(small_y, small_p, t)
        fpr, tpr = operating_rates(m)
        ax.text(
            0.99, 0.98,
            f"t={t:.2f}  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}\n"
            f"Precision={m['precision']:.2f}  Recall/TPR={tpr:.2f}  "
            f"F1={m['f1']:.2f}  FPR={fpr:.2f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.94),
        )
        ax.set(xlim=(0, 1), ylim=(-0.25, 1.3), yticks=[0, 1],
               yticklabels=["Actual 0", "Actual 1"],
               xlabel="Predicted probability",
               title="Threshold changes decisions | synthetic 65-case subset")
        ax.legend(loc="upper left", ncol=2, fontsize=8)
        return ax.collections

    update(0)
    animation = FuncAnimation(fig, update, frames=len(thresholds),
                              interval=280, blit=False, repeat=True)
    animation.save(out / "threshold_sweep.gif", writer=PillowWriter(fps=4))
    plt.close(fig)


def animate_operating(y, p, out, kind):
    thresholds = np.linspace(0.95, 0.05, 23)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8))
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)

    def update(i):
        left.clear()
        right.clear()
        t = thresholds[i]
        m = threshold_metrics(y, p, t)
        draw_score_distribution(left, y, p, t)
        if kind == "roc":
            right.plot(fpr, tpr, color=BLUE, lw=2, label="All thresholds")
            right.plot([0, 1], [0, 1], ls="--", color="#8895a3",
                       label="Random ranking")
            x, v = operating_rates(m)
            right.set(xlabel="False positive rate", ylabel="True positive rate",
                      title=f"ROC across thresholds | AUC={auc:.3f}")
            name = "roc_threshold_animation.gif"
        else:
            right.plot(recall, precision, color=PURPLE, lw=2,
                       label="All thresholds")
            right.axhline(np.mean(y), ls="--", color="#8895a3",
                          label=f"Prevalence {np.mean(y):.3f}")
            x, v = m["recall"], m["precision"]
            right.set(xlabel="Recall", ylabel="Precision",
                      title=f"PR across thresholds | AP={ap:.3f}")
            name = "pr_threshold_animation.gif"
        right.scatter([x], [v], s=110, color=RED, zorder=5,
                      label=f"t={t:.2f}: ({x:.2f}, {v:.2f})")
        right.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
        right.legend(loc="lower right" if kind == "roc" else "upper right",
                     fontsize=8)
        fig.suptitle("Synthetic validation population", fontsize=10)
        return []

    update(0)
    animation = FuncAnimation(fig, update, frames=len(thresholds),
                              interval=280, blit=False, repeat=True)
    animation.save(out / ("roc_threshold_animation.gif" if kind == "roc"
                          else "pr_threshold_animation.gif"),
                   writer=PillowWriter(fps=4))
    plt.close(fig)


def plot_roc_pr(y, p, selected, out):
    roc_x, roc_y, _ = roc_curve(y, p)
    pr_y, pr_x, _ = precision_recall_curve(y, p)
    m = threshold_metrics(y, p, selected["threshold"])
    fpr, tpr = operating_rates(m)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.8, 4.5))
    a.plot(roc_x, roc_y, color=BLUE, lw=2.2)
    a.plot([0, 1], [0, 1], "--", color="#9ca8b6", label="Random ranking")
    a.scatter([fpr], [tpr], color=RED, s=85, zorder=4,
              label=f"Selected t={selected['threshold']:.2f}")
    a.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate",
          ylabel="True positive rate",
          title=f"ROC | AUC={roc_auc_score(y, p):.3f}")
    a.legend(fontsize=8)
    b.plot(pr_x, pr_y, color=PURPLE, lw=2.2)
    b.axhline(np.mean(y), ls="--", color="#9ca8b6",
              label=f"Prevalence={np.mean(y):.3f}")
    b.scatter([m["recall"]], [m["precision"]], color=RED, s=85, zorder=4,
              label=f"Same t={selected['threshold']:.2f}")
    b.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
          title=f"Precision-recall | AP={average_precision_score(y, p):.3f}")
    b.legend(fontsize=8)
    fig.suptitle("One validation score set and one decision threshold | synthetic",
                 fontsize=11)
    fig.tight_layout()
    save(fig, out / "roc_vs_pr.png")


def prevalence_experiment(seed=231):
    """Weighted synthetic populations with identical class-conditional scores."""
    rng = np.random.default_rng(seed)
    pos = rng.beta(4, 3, 2500)
    neg = rng.beta(2, 7, 2500)
    y = np.r_[np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
    scores = np.r_[pos, neg]
    operating_threshold = float(np.quantile(neg, 0.95))
    results = []
    for prevalence in (0.5, 0.1, 0.01, 0.001):
        weights = np.where(y == 1, prevalence / len(pos),
                           (1 - prevalence) / len(neg))
        pr_precision, recall, _ = precision_recall_curve(
            y, scores, sample_weight=weights
        )
        roc_fpr, roc_tpr, _ = roc_curve(y, scores, sample_weight=weights)
        positive = scores >= operating_threshold
        tp = weights[(y == 1) & positive].sum()
        fp = weights[(y == 0) & positive].sum()
        results.append(
            dict(prevalence=prevalence, roc_auc=roc_auc_score(
                     y, scores, sample_weight=weights),
                 ap=average_precision_score(y, scores, sample_weight=weights),
                 precision_at_threshold=tp / (tp + fp),
                 precision=pr_precision, recall=recall,
                 roc_fpr=roc_fpr, roc_tpr=roc_tpr,
                 threshold=operating_threshold)
        )
    return results


def plot_prevalence(out):
    results = prevalence_experiment()
    fig, ax = plt.subplots(figsize=(8.7, 5.5))
    colors = [BLUE, GREEN, ORANGE, RED]
    for row, color in zip(results, colors):
        ax.plot(row["recall"], row["precision"], color=color, lw=2,
                label=f"Prevalence {row['prevalence']:.1%} | AP {row['ap']:.3f}")
        ax.axhline(row["prevalence"], color=color, lw=0.9,
                   ls=":", alpha=0.65)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
           title="PR changes with prevalence | fixed conditional score sets")
    ax.legend(loc="lower left", fontsize=8)
    ax.text(0.99, 0.97, "Dotted lines: no-skill baselines\nWeighted synthetic populations",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none"))
    save(fig, out / "prevalence_pr_curves.png")
    return results


def plot_bayes_precision(out, tpr=0.8):
    prevalence = np.geomspace(0.001, 0.5, 300)
    fig, ax = plt.subplots(figsize=(8.3, 5))
    for fpr, color in zip((0.001, 0.005, 0.02, 0.05),
                          (GREEN, BLUE, ORANGE, RED)):
        precision = tpr * prevalence / (
            tpr * prevalence + fpr * (1 - prevalence)
        )
        ax.plot(prevalence * 100, precision * 100, color=color, lw=2,
                label=f"FPR={fpr:.3f}")
    ax.set(xscale="log", xlim=(0.1, 50), ylim=(0, 100),
           xlabel="Positive prevalence (%)", ylabel="Expected precision (%)",
           title=f"Bayes precision relationship | fixed TPR={tpr:.2f}")
    ax.legend(fontsize=8)
    ax.text(0.98, 0.03, "Mathematical illustration; not fitted-model evidence",
            transform=ax.transAxes, ha="right", fontsize=8)
    save(fig, out / "precision_vs_prevalence.png")


def plot_threshold_landscape(rows, choices, out):
    t = np.array([r["threshold"] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, color, label in (
        ("precision", BLUE, "Precision"), ("recall", ORANGE, "Recall"),
        ("f1", GREEN, "F1"), ("specificity", PURPLE, "Specificity"),
    ):
        ax.plot(t, [r[key] for r in rows], color=color, lw=2, label=label)
    ax.axvline(0.5, ls="--", color="#758292", label="0.50")
    ax.axvline(choices["f1"]["threshold"], ls=":", color=INK,
               label=f"Max F1 on grid: {choices['f1']['threshold']:.2f}")
    if choices["precision"] is not None:
        ax.axvline(choices["precision"]["threshold"], ls="-.", color=RED,
                   label=f"Precision ≥ 0.80: {choices['precision']['threshold']:.2f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Decision threshold",
           ylabel="Metric value", title="Validation metrics across thresholds | synthetic")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    save(fig, out / "metrics_vs_threshold.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(t, [r["alerts"] for r in rows], color=BLUE, lw=2.3,
            label="Predicted positives / alerts")
    ax.axvline(0.5, ls="--", color="#758292", label="0.50")
    ax.set(xlim=(0, 1), ylim=(0, rows[0]["alerts"] * 1.02),
           xlabel="Decision threshold", ylabel="Number of alerts",
           title="Validation alert volume | synthetic")
    ax.legend(fontsize=8)
    save(fig, out / "alert_volume_vs_threshold.png")


def calibration_points(y, p, bins=10):
    observed, predicted = calibration_curve(
        y, p, n_bins=bins, strategy="quantile"
    )
    return predicted, observed


def plot_ranking_calibration(y, p, out):
    distorted = p ** 4
    original = score_summary(y, p)
    altered = score_summary(y, distorted)
    if np.any(np.argsort(p) != np.argsort(distorted)):
        raise AssertionError("Strictly monotonic transform changed score order")
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.4))
    a, b, c, d = axes.flat
    bins = np.linspace(0, 1, 24)
    a.hist(p, bins=bins, color=BLUE, alpha=0.8)
    a.set(xlabel="Predicted probability", ylabel="Cases",
          title="Original score distribution")
    b.hist(distorted, bins=bins, color=ORANGE, alpha=0.8)
    b.set(xlabel="Transformed probability (p⁴)", ylabel="Cases",
          title="Same cases, changed magnitudes")
    for scores, color, name in (
        (p, BLUE, "Original"), (distorted, ORANGE, "p⁴"),
    ):
        x, yy, _ = roc_curve(y, scores)
        c.plot(x, yy, color=color, lw=2,
               label=f"{name}: AUC={roc_auc_score(y, scores):.3f}")
        predicted, observed = calibration_points(y, scores)
        d.plot(predicted, observed, "o-", color=color,
               label=f"{name}: Brier={brier_score_loss(y, scores):.3f}")
    c.plot([0, 1], [0, 1], "--", color="#9ca8b6")
    c.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate",
          ylabel="True positive rate", title="Ranking curve")
    c.legend(fontsize=8)
    d.plot([0, 1], [0, 1], "--", color="#9ca8b6",
           label="Perfect calibration")
    d.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
          ylabel="Observed event rate", title="Reliability | quantile bins")
    d.legend(fontsize=8)
    fig.suptitle("Ranking can stay fixed while probability meaning changes | synthetic",
                 fontsize=12)
    fig.tight_layout()
    save(fig, out / "ranking_vs_calibration.png")
    return original, altered


def animate_calibration(y, p, out):
    gammas = np.r_[np.linspace(0.5, 1, 5), np.linspace(1.3, 6, 12)]
    auc_original = roc_auc_score(y, p)
    fig, ax = plt.subplots(figsize=(6.7, 5.8))

    def update(i):
        ax.clear()
        gamma = gammas[i]
        changed = p ** gamma
        predicted, observed = calibration_points(y, changed)
        ax.plot([0, 1], [0, 1], "--", color="#9ca8b6",
                label="Perfect calibration")
        ax.plot(predicted, observed, "o-", color=PURPLE, lw=2,
                label=f"p^{gamma:.2f}")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
               ylabel="Observed event rate",
               title="Monotonic probability distortion | synthetic test")
        ax.text(
            0.98, 0.03,
            f"ROC-AUC {roc_auc_score(y, changed):.3f}\n"
            f"Brier {brier_score_loss(y, changed):.3f}\n"
            f"Original AUC {auc_original:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.94),
        )
        ax.legend(loc="upper left", fontsize=8)
        return []

    update(0)
    animation = FuncAnimation(fig, update, frames=len(gammas),
                              interval=350, blit=False, repeat=True)
    animation.save(out / "calibration_distortion.gif",
                   writer=PillowWriter(fps=3))
    plt.close(fig)


def plot_reliability(y, p, out):
    distorted = p ** 4
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.4, 7), height_ratios=(2, 1), sharex=True
    )
    top.plot([0, 1], [0, 1], "--", color="#8c99a7",
             label="Perfect calibration")
    bins = np.linspace(0, 1, 11)
    for scores, color, name in ((p, BLUE, "Original"), (distorted, ORANGE, "p⁴")):
        x, yy = calibration_points(y, scores)
        top.plot(x, yy, "o-", color=color, lw=2, label=name)
        bottom.hist(scores, bins=bins, alpha=0.55, color=color, label=name)
    top.set(ylabel="Observed event rate", ylim=(0, 1),
            title="Reliability by equal-count bins | synthetic test")
    top.legend(fontsize=8)
    bottom.set(xlabel="Mean predicted probability / probability bin",
               ylabel="Cases", xlim=(0, 1), yscale="symlog")
    bottom.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out / "reliability_diagram.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    ax.hist(p, bins=bins, alpha=0.6, color=BLUE, label="Original")
    ax.hist(distorted, bins=bins, alpha=0.55, color=ORANGE, label="p⁴")
    ax.set(xlim=(0, 1), yscale="symlog",
           xlabel="Predicted probability", ylabel="Cases (symlog scale)",
           title="Probability mass shifts under p⁴ | synthetic test")
    ax.legend()
    save(fig, out / "probability_distribution.png")


def fit_calibrators(y_cal, p_cal, p_test):
    """Fit mappings on calibration data; apply only to held-out test scores."""
    clipped_cal = np.clip(p_cal, 1e-8, 1 - 1e-8)
    clipped_test = np.clip(p_test, 1e-8, 1 - 1e-8)
    logit_cal = np.log(clipped_cal / (1 - clipped_cal)).reshape(-1, 1)
    logit_test = np.log(clipped_test / (1 - clipped_test)).reshape(-1, 1)
    sigmoid = LogisticRegression(max_iter=1000).fit(logit_cal, y_cal)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
    alternatives = {
        "Original": p_test,
        "Sigmoid": sigmoid.predict_proba(logit_test)[:, 1],
        "Isotonic": isotonic.predict(p_test),
    }
    if any(np.any((p < 0) | (p > 1)) for p in alternatives.values()):
        raise AssertionError("Calibration probabilities must lie in [0, 1]")
    return alternatives


def plot_calibration_methods(y, alternatives, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.9))
    left.plot([0, 1], [0, 1], "--", color="#9ca8b6",
              label="Perfect calibration")
    colors = {"Original": BLUE, "Sigmoid": GREEN, "Isotonic": ORANGE}
    for name, p in alternatives.items():
        x, observed = calibration_points(y, p)
        left.plot(x, observed, "o-", color=colors[name], lw=2, label=name)
    left.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
             ylabel="Observed event rate",
             title="Held-out test reliability | quantile bins")
    left.legend(fontsize=8)
    names = list(alternatives)
    values = [brier_score_loss(y, alternatives[name]) for name in names]
    bars = right.bar(names, values, color=[colors[name] for name in names])
    for bar, value in zip(bars, values):
        right.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + max(values) * 0.012,
                   f"{value:.4f}", ha="center", fontsize=9)
    right.set(ylabel="Brier score (lower is better)",
              ylim=(0, max(values) * 1.16),
              title="Probability error on the same test set")
    fig.suptitle("Calibration maps fitted on a separate calibration split | synthetic",
                 fontsize=11)
    fig.tight_layout()
    save(fig, out / "calibration_methods.png")


def plot_cost(rows, choice, out, cost_fp, cost_fn):
    t = np.array([r["threshold"] for r in rows])
    cost = np.array([cost_fp * r["fp"] + cost_fn * r["fn"] for r in rows])
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.plot(t, cost, color=PURPLE, lw=2.2,
            label=f"Cost = {cost_fp:g}·FP + {cost_fn:g}·FN")
    t_best = choice["threshold"]
    best_cost = cost_fp * choice["fp"] + cost_fn * choice["fn"]
    ax.scatter([t_best], [best_cost], color=GREEN, s=95, zorder=4,
               label=f"Grid minimum t={t_best:.2f}, cost={best_cost:g}")
    at_half = next(r for r in rows if np.isclose(r["threshold"], 0.5))
    half_cost = cost_fp * at_half["fp"] + cost_fn * at_half["fn"]
    ax.scatter([0.5], [half_cost], color=RED, s=75, zorder=4,
               label=f"t=0.50, cost={half_cost:g}")
    ax.set(xlim=(0, 1), xlabel="Decision threshold",
           ylabel="Total illustrative classification cost",
           title="Empirical validation cost across threshold grid | synthetic")
    ax.legend(fontsize=8)
    save(fig, out / "cost_vs_threshold.png")


def plot_decision_cost(out):
    p = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, (cost_fp, cost_fn) in zip(axes, ((1, 1), (1, 9))):
        positive = (1 - p) * cost_fp
        negative = p * cost_fn
        threshold = cost_fp / (cost_fp + cost_fn)
        ax.plot(p, positive, color=RED, lw=2,
                label="Predict positive: (1-p)×C_FP")
        ax.plot(p, negative, color=BLUE, lw=2,
                label="Predict negative: p×C_FN")
        ax.scatter([threshold], [(1 - threshold) * cost_fp],
                   color=INK, zorder=5)
        ax.axvline(threshold, ls=":", color=INK,
                   label=f"Intersection t={threshold:.2f}")
        ax.set(xlim=(0, 1), xlabel="Calibrated probability p",
               ylabel="Expected cost per case",
               title=f"C_FP={cost_fp}, C_FN={cost_fn}")
        ax.legend(fontsize=7)
    fig.suptitle("Decision-theory cutoff | constant costs, calibrated p",
                 fontsize=11)
    fig.tight_layout()
    save(fig, out / "decision_cost_threshold.png")


def plot_capacity(rows, choices, out, capacity=100):
    t = [r["threshold"] for r in rows]
    alerts = [r["alerts"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.7, 5))
    ax.plot(t, alerts, color=BLUE, lw=2.2, label="Validation alerts")
    ax.axhline(capacity, color=RED, ls="--",
               label=f"Illustrative capacity: {capacity}")
    for key, color, label in (
        ("capacity", GREEN, "Capacity-constrained"),
        ("f1", PURPLE, "Max F1"),
        ("precision", ORANGE, "Precision ≥ 0.80"),
    ):
        row = choices[key]
        if row is not None:
            ax.scatter([row["threshold"]], [row["alerts"]], color=color,
                       s=85, zorder=5,
                       label=f"{label}: t={row['threshold']:.2f}, "
                             f"{row['alerts']} alerts")
    ax.set(xlim=(0, 1), xlabel="Decision threshold",
           ylabel="Predicted positives / reviews",
           title="Policy choice under finite review capacity | synthetic")
    ax.legend(fontsize=8)
    save(fig, out / "capacity_constraint.png")


def write_cost_surface(rows, out, cost_fp=1.0):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Plotly unavailable; skipping threshold_cost_surface.html")
        return False
    ratios = np.linspace(1, 50, 50)
    thresholds = np.array([r["threshold"] for r in rows])
    fp = np.array([r["fp"] for r in rows])
    fn = np.array([r["fn"] for r in rows])
    n = rows[0]["tp"] + rows[0]["fp"] + rows[0]["tn"] + rows[0]["fn"]
    cost_per_1000 = (cost_fp * fp[None, :] +
                     cost_fp * ratios[:, None] * fn[None, :]) / n * 1000
    minima = np.argmin(cost_per_1000, axis=1)
    fig = go.Figure()
    fig.add_surface(
        x=thresholds, y=ratios, z=cost_per_1000,
        colorscale="Viridis", colorbar=dict(title="Cost / 1,000 cases"),
        hovertemplate="Threshold %{x:.2f}<br>C_FN/C_FP %{y:.1f}"
                      "<br>Cost per 1,000 %{z:.1f}<extra></extra>",
        name="Cost surface",
    )
    fig.add_scatter3d(
        x=thresholds[minima], y=ratios,
        z=cost_per_1000[np.arange(len(ratios)), minima] + 0.3,
        mode="lines", line=dict(color=RED, width=8),
        name="Grid minimum by cost ratio",
    )
    fig.update_layout(
        title=f"Validation cost surface | synthetic, C_FP fixed at {cost_fp:g}",
        scene=dict(xaxis_title="Threshold",
                   yaxis_title="C_FN / C_FP",
                   zaxis_title="Cost per 1,000 validation cases"),
        margin=dict(l=0, r=0, b=0, t=60),
    )
    fig.write_html(out / "threshold_cost_surface.html",
                   include_plotlyjs=True, full_html=True)
    return True


def write_interactive_explorer(y, p, out):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly unavailable; skipping interactive_threshold_explorer.html")
        return False
    thresholds = np.linspace(0.95, 0.05, 19)
    roc_x, roc_y, _ = roc_curve(y, p)
    pr_precision, pr_recall, _ = precision_recall_curve(y, p)
    bins = np.linspace(0, 1, 25)
    hist_max = max(np.histogram(p[y == 0], bins=bins)[0].max(),
                   np.histogram(p[y == 1], bins=bins)[0].max()) * 1.08

    def count_grid(m):
        return [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]

    def title(m, t):
        fpr, tpr = operating_rates(m)
        return (
            f"Synthetic validation | t={t:.2f} | TP={m['tp']} FP={m['fp']} "
            f"TN={m['tn']} FN={m['fn']} | P={m['precision']:.2f} "
            f"R/TPR={tpr:.2f} F1={m['f1']:.2f} FPR={fpr:.2f}"
        )

    first = threshold_metrics(y, p, thresholds[0])
    first_fpr, first_tpr = operating_rates(first)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Scores and threshold", "Confusion matrix",
                        "ROC operating point", "PR operating point"),
        specs=[[{}, {}], [{}, {}]],
        horizontal_spacing=0.13, vertical_spacing=0.14,
    )
    fig.add_trace(go.Histogram(x=p[y == 0], xbins=dict(start=0, end=1, size=1/24),
                               marker_color=BLUE, opacity=0.65, name="Actual 0"),
                  row=1, col=1)
    fig.add_trace(go.Histogram(x=p[y == 1], xbins=dict(start=0, end=1, size=1/24),
                               marker_color=ORANGE, opacity=0.7, name="Actual 1"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[thresholds[0]] * 2, y=[0, hist_max],
                             mode="lines", line=dict(color=RED, width=3),
                             name="Threshold"), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=count_grid(first), x=["Predict 0", "Predict 1"],
        y=["Actual 0", "Actual 1"], text=count_grid(first),
        texttemplate="%{text}", colorscale="Blues", showscale=False,
        hovertemplate="%{y}, %{x}: %{z} cases<extra></extra>",
        name="Counts"), row=1, col=2)
    fig.add_trace(go.Scatter(x=roc_x, y=roc_y, mode="lines",
                             line=dict(color=BLUE, width=2), name="ROC curve"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=[first_fpr], y=[first_tpr], mode="markers",
                             marker=dict(color=RED, size=13), name="ROC point"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=pr_recall, y=pr_precision, mode="lines",
                             line=dict(color=PURPLE, width=2), name="PR curve"),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=[first["recall"]], y=[first["precision"]],
                             mode="markers", marker=dict(color=RED, size=13),
                             name="PR point"), row=2, col=2)
    frames = []
    for t in thresholds:
        m = threshold_metrics(y, p, t)
        fpr, tpr = operating_rates(m)
        frames.append(go.Frame(
            name=f"{t:.2f}",
            data=[
                go.Scatter(x=[t, t], y=[0, hist_max]),
                go.Heatmap(z=count_grid(m), text=count_grid(m)),
                go.Scatter(x=[fpr], y=[tpr]),
                go.Scatter(x=[m["recall"]], y=[m["precision"]]),
            ],
            traces=[2, 3, 5, 7],
            layout=go.Layout(title=title(m, t)),
        ))
    fig.frames = frames
    steps = [dict(method="animate", label=f"{t:.2f}",
                  args=[[f"{t:.2f}"], {"mode": "immediate",
                                       "frame": {"duration": 0,
                                                 "redraw": True},
                                       "transition": {"duration": 0}}])
             for t in thresholds]
    fig.update_layout(
        title=title(first, thresholds[0]), barmode="overlay",
        width=1150, height=800, template="plotly_white",
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix="Threshold "))],
        updatemenus=[dict(type="buttons", showactive=False,
                          buttons=[dict(label="Play", method="animate",
                                        args=[None, {"frame": {"duration": 350,
                                                               "redraw": True},
                                                     "fromcurrent": True}])],
                          x=0, y=-0.08)],
        legend=dict(orientation="h", y=-0.16),
    )
    for row, col in ((2, 1), (2, 2)):
        fig.update_xaxes(range=[0, 1], row=row, col=col)
        fig.update_yaxes(range=[0, 1], row=row, col=col)
    fig.update_xaxes(title_text="Predicted probability", range=[0, 1],
                     row=1, col=1)
    fig.update_yaxes(title_text="Cases", row=1, col=1)
    fig.update_xaxes(title_text="False positive rate", row=2, col=1)
    fig.update_yaxes(title_text="True positive rate", row=2, col=1)
    fig.update_xaxes(title_text="Recall", row=2, col=2)
    fig.update_yaxes(title_text="Precision", row=2, col=2)
    fig.write_html(out / "interactive_threshold_explorer.html",
                   include_plotlyjs=True, full_html=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--cost-fp", type=float, default=1.0)
    parser.add_argument("--cost-fn", type=float, default=9.0)
    parser.add_argument("--capacity", type=int, default=100)
    args = parser.parse_args()
    if not np.isfinite(args.cost_fp) or not np.isfinite(args.cost_fn):
        parser.error("Costs must be finite")
    style()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    labels, scores = make_splits()
    y_val, p_val = labels["validation"], scores["validation"]
    y_test, p_test = labels["test"], scores["test"]
    rows = threshold_table(y_val, p_val)
    choices = choose_policies(
        rows, min_precision=0.8, cost_fp=args.cost_fp,
        cost_fn=args.cost_fn, capacity=args.capacity,
    )
    validation_summary = score_summary(y_val, p_val)
    if not 0 <= validation_summary["roc_auc"] <= 1:
        raise AssertionError("ROC-AUC must lie in [0, 1]")
    plot_threshold_sweep(y_val, p_val, out)
    animate_operating(y_val, p_val, out, "roc")
    animate_operating(y_val, p_val, out, "pr")
    plot_roc_pr(y_val, p_val, choices["f1"], out)
    prevalence = plot_prevalence(out)
    plot_bayes_precision(out)
    plot_threshold_landscape(rows, choices, out)
    original, distorted = plot_ranking_calibration(y_test, p_test, out)
    animate_calibration(y_test, p_test, out)
    plot_reliability(y_test, p_test, out)
    alternatives = fit_calibrators(
        labels["calibration"], scores["calibration"], p_test
    )
    plot_calibration_methods(y_test, alternatives, out)
    plot_cost(rows, choices["cost"], out, args.cost_fp, args.cost_fn)
    plot_decision_cost(out)
    plot_capacity(rows, choices, out, args.capacity)
    write_cost_surface(rows, out, args.cost_fp)
    write_interactive_explorer(y_val, p_val, out)

    print(f"Synthetic splits: train=3000, calibration=1000, validation=1000, test=1000")
    print(f"Validation positives={int(y_val.sum())}, prevalence={np.mean(y_val):.4f}")
    print("Validation:", ", ".join(
        f"{name}={value:.4f}" for name, value in validation_summary.items()
    ))
    for name, row in choices.items():
        print(f"{name} policy: " + (
            f"threshold={row['threshold']:.2f}, precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, alerts={row['alerts']}"
            if row is not None else "no feasible threshold on grid"
        ))
    print("Test original:", original)
    print("Test p^4:", distorted)
    for name, p in alternatives.items():
        print(f"Test calibration method {name}: {score_summary(y_test, p)}")
    for row in prevalence:
        print(
            f"Weighted prevalence {row['prevalence']:.1%}: "
            f"ROC-AUC={row['roc_auc']:.4f}, AP={row['ap']:.4f}, "
            f"precision at fixed threshold={row['precision_at_threshold']:.4f}"
        )
    print(f"Generated outputs in {out}")


if __name__ == "__main__":
    main()
