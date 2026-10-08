"""Day 33: deterministic visual intuition for imbalanced classification.

Run from the repository root. All resampling uses training rows, all policy
selection uses validation rows, and final demonstrations use untouched test rows.
Generated artifacts are relative to this script, not the current directory.
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
NEGATIVE, POSITIVE, SYNTHETIC = "#2878A5", "#D87522", "#238B68"
OUTPUT_DIR = Path(__file__).resolve().parent / "assets"
COST_SCENARIOS = ((1.0, 20.0), (10.0, 2.0))


def optional_dependencies():
    """Give an actionable message instead of silently omitting requested views."""
    modules = {}
    for name in ("imblearn.over_sampling", "imblearn.under_sampling", "plotly.graph_objects", "PIL"):
        try:
            modules[name] = importlib.import_module(name)
        except ImportError as error:
            raise RuntimeError(
                f"Visual dependency {name!r} is missing or incompatible. "
                'From the repository root run: python -m pip install -e ".[imbalance]"'
            ) from error
    return modules


def validate_scores(y, probabilities):
    y = np.asarray(y)
    p = np.asarray(probabilities, dtype=float)
    if (
        y.ndim != 1 or y.size == 0 or not np.isin(y, [0, 1]).all()
        or p.shape != y.shape or not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError("scores need matching nonempty binary labels and probabilities in [0, 1]")
    return y.astype(int), p


def threshold_metrics(y, probabilities, thresholds):
    """Compute counts and metrics for fixed scores, including tied endpoints."""
    y, p = validate_scores(y, probabilities)
    thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))
    if thresholds.ndim != 1 or not thresholds.size or not np.isfinite(thresholds).all() or (thresholds < 0).any():
        raise ValueError("thresholds must be a nonempty finite nonnegative vector")
    prediction = p[None, :] >= thresholds[:, None]
    tp = np.sum(prediction & (y == 1), axis=1)
    fp = np.sum(prediction & (y == 0), axis=1)
    fn = np.sum(~prediction & (y == 1), axis=1)
    tn = np.sum(~prediction & (y == 0), axis=1)

    def divide(top, bottom):
        return np.divide(top, bottom, out=np.zeros(len(thresholds), dtype=float), where=bottom != 0)

    return {
        "threshold": thresholds, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
        "f2": divide(5 * tp, 5 * tp + 4 * fn + fp),
        "accuracy": (tp + tn) / len(y), "alert_rate": (tp + fp) / len(y),
        "predicted_positives": tp + fp,
    }


def empirical_cost(curves, false_positive_cost, false_negative_cost):
    costs = np.asarray([false_positive_cost, false_negative_cost], dtype=float)
    if costs.shape != (2,) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("error costs must be finite positive scalars")
    return curves["fp"] * costs[0] + curves["fn"] * costs[1]


def validation_policies(y_validation, p_validation):
    """Enumerate distinct validation decisions; ties prefer fewer alerts."""
    y, p = validate_scores(y_validation, p_validation)
    thresholds = np.append(np.unique(p), np.nextafter(p.max(), np.inf))
    curves = threshold_metrics(y, p, thresholds)

    def choose(values, minimize):
        optimum = np.min(values) if minimize else np.max(values)
        index = np.flatnonzero(values == optimum)[-1]
        return float(thresholds[index])

    return {
        "f1": choose(curves["f1"], False),
        "cost": [
            {"c_fp": c_fp, "c_fn": c_fn, "threshold": choose(empirical_cost(curves, c_fp, c_fn), True)}
            for c_fp, c_fn in COST_SCENARIOS
        ],
    }


def precision_from_prevalence(prevalence, tpr=0.90, fpr=0.05):
    prevalence = np.asarray(prevalence, dtype=float)
    rates = np.asarray([tpr, fpr], dtype=float)
    if (
        not np.isfinite(prevalence).all() or ((prevalence <= 0) | (prevalence >= 1)).any()
        or rates.shape != (2,) or not np.isfinite(rates).all()
        or ((rates < 0) | (rates > 1)).any()
    ):
        raise ValueError("prevalence must be in (0, 1), and TPR/FPR in [0, 1]")
    denominator = tpr * prevalence + fpr * (1 - prevalence)
    if (denominator == 0).any():
        raise ValueError("precision is undefined when no positives are predicted")
    return tpr * prevalence / denominator


def interpolate(anchor, neighbor, fraction):
    anchor, neighbor = np.asarray(anchor, dtype=float), np.asarray(neighbor, dtype=float)
    fraction = np.asarray(fraction, dtype=float)
    if (
        anchor.shape != (2,) or neighbor.shape != (2,)
        or not np.isfinite(anchor).all() or not np.isfinite(neighbor).all()
        or fraction.ndim != 0 or not np.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("interpolation requires finite 2D endpoints and a fraction in [0, 1]")
    return anchor + fraction * (neighbor - anchor)


def make_dataset(n_samples=3000, prevalence=0.05, n_features=2, class_sep=1.2, n_clusters_per_class=1):
    if (
        isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 100
        or not np.isfinite(prevalence) or not 0 < prevalence < 0.5
        or n_features not in (2, 10)
        or not np.isfinite(class_sep) or class_sep <= 0
        or n_clusters_per_class not in (1, 2)
    ):
        raise ValueError("use at least 100 rows, prevalence in (0, 0.5), and 2 or 10 features")
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=2 if n_features == 2 else 5,
        n_redundant=0 if n_features == 2 else 2,
        n_clusters_per_class=n_clusters_per_class, weights=[1 - prevalence, prevalence],
        class_sep=class_sep, flip_y=0, random_state=SEED,
    )
    if np.min(np.bincount(y, minlength=2)) < 10:
        raise ValueError("increase rows or prevalence to obtain enough positives for three splits")
    train, rest = train_test_split(np.arange(len(y)), test_size=0.4, stratify=y, random_state=SEED)
    validation, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=SEED)
    for indices in (train, validation, test):
        if len(np.unique(y[indices])) != 2:
            raise ValueError("each split must contain both classes; increase positive counts")
    return {"X": X, "y": y, "train": train, "validation": validation, "test": test,
            "configuration": {"n_samples": n_samples, "prevalence": prevalence, "n_features": n_features,
                              "n_clusters_per_class": n_clusters_per_class, "class_sep": class_sep, "flip_y": 0, "seed": SEED}}


def fit_models(data):
    """Both models fit identical original training rows, with train-only scaling."""
    X, y, rows = data["X"], data["y"], data["train"]
    models = {}
    for label, class_weight in (("Ordinary", None), ("Balanced weights", "balanced")):
        models[label] = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1, solver="lbfgs", max_iter=1000, class_weight=class_weight, random_state=SEED),
        ).fit(X[rows], y[rows])
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y[rows])
    return models, {str(label): float(weight) for label, weight in enumerate(weights)}


def resample_training(X_train, y_train, dependencies):
    """Return training-only display data; never accept validation or test rows."""
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    if X_train.ndim != 2 or X_train.shape[1] != 2 or len(X_train) != len(y_train):
        raise ValueError("resampling panels require matching two-dimensional training data")
    scaler = StandardScaler().fit(X_train)
    scaled = scaler.transform(X_train)
    over = dependencies["imblearn.over_sampling"]
    under = dependencies["imblearn.under_sampling"]
    samplers = {
        "Random oversampling": over.RandomOverSampler(random_state=SEED),
        "Random undersampling": under.RandomUnderSampler(random_state=SEED),
        "SMOTE": over.SMOTE(random_state=SEED, k_neighbors=5),
    }
    panels = {"Original training": {"X": X_train.copy(), "y": y_train.copy()}}
    # Sampling before splitting would let duplicates/synthetic relatives leak.
    # Here the scaler and every sampler see only original training observations.
    for name, sampler in samplers.items():
        sampled_X, sampled_y = sampler.fit_resample(scaled, y_train)
        panel = {"X": scaler.inverse_transform(sampled_X), "y": sampled_y}
        if name == "Random oversampling":
            panel["multiplicity"] = np.bincount(sampler.sample_indices_, minlength=len(y_train))
        panels[name] = panel
    return panels


def finish(fig, path, subtitle=None):
    if subtitle:
        fig.text(0.5, 0.015, subtitle, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0, 0.06 if subtitle else 0, 1, 0.96))
    fig.savefig(path, dpi=125, facecolor="white")
    plt.close(fig)


def scatter_classes(ax, X, y, size=12, alpha=0.5):
    for label, color, marker in ((0, NEGATIVE, "."), (1, POSITIVE, "o")):
        rows = y == label
        ax.scatter(X[rows, 0], X[rows, 1], s=size, c=color, marker=marker,
                   alpha=alpha, label=f"Class {label} (n={rows.sum()})")
    ax.set(xlabel="Feature 1 (raw units)", ylabel="Feature 2 (raw units)")
    ax.grid(alpha=0.15)


def plot_class_distribution(data, path):
    X, y = data["X"], data["y"]
    counts = np.bincount(y)
    fig, (ax, bars) = plt.subplots(1, 2, figsize=(11, 4.7), gridspec_kw={"width_ratios": [2, 1]})
    scatter_classes(ax, X, y)
    ax.legend()
    ax.set_title("Synthetic population geometry")
    bars.bar(["Class 0", "Class 1"], counts, color=[NEGATIVE, POSITIVE])
    for i, count in enumerate(counts):
        bars.text(i, count + len(y) * 0.02, f"{count:,}\n{count / len(y):.1%}", ha="center")
    bars.set(ylabel="Observation count", ylim=(0, counts.max() * 1.22), title="Empirical probability mass")
    fig.suptitle(f"Imbalanced dataset: {1-y.mean():.0%} majority / {y.mean():.0%} minority")
    finish(fig, path, "Synthetic descriptive view of all rows; fitting and policy selection use separate splits.")


def plot_accuracy_trap(y_test, probabilities, path):
    measures = ("accuracy", "precision", "recall", "f1")
    majority = threshold_metrics(y_test, np.zeros(len(y_test)), [0.5])
    trained = threshold_metrics(y_test, probabilities, [0.5])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(measures))
    for offset, label, result, color in ((-0.18, "Always negative", majority, NEGATIVE),
                                          (0.18, "Trained logistic regression", trained, POSITIVE)):
        values = [result[m][0] for m in measures]
        bars = ax.bar(x + offset, values, 0.36, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set(xticks=x, xticklabels=["Accuracy", "Precision", "Recall", "F1"],
           ylim=(0, 1.19), ylabel="Test metric", title="High accuracy does not guarantee useful rare-event decisions")
    ax.legend(loc="upper right")
    finish(fig, path, "Untouched test set; threshold 0.50. Precision is defined as 0 when no alerts are generated.")


def plot_confusion_thresholds(y_test, probabilities, path):
    thresholds = [0.80, 0.50, 0.25, 0.10]
    curves = threshold_metrics(y_test, probabilities, thresholds)
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    matrices = [np.array([[curves["tn"][i], curves["fp"][i]],
                         [curves["fn"][i], curves["tp"][i]]]) for i in range(4)]
    # Shared normalization keeps color comparable across thresholds.
    maximum = max(matrix.max() for matrix in matrices)
    for ax, threshold, matrix in zip(axes, thresholds, matrices):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for row in range(2):
            for col in range(2):
                name = [["TN", "FP"], ["FN", "TP"]][row][col]
                ax.text(col, row, f"{name}\n{matrix[row, col]}", ha="center", va="center",
                        color="white" if matrix[row, col] > maximum / 2 else "#132B3B", fontsize=13)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"], yticklabels=["0", "1"],
               xlabel="Predicted class", ylabel="Actual class", title=f"Threshold {threshold:.2f}")
    fig.suptitle("One fitted model, four decision policies: test confusion counts")
    finish(fig, path, "Lower threshold: TP and FP cannot decrease; FN and TN cannot increase on fixed scores.")


def animate_threshold(y_test, probabilities, path, frames=36):
    if not isinstance(frames, int) or frames < 2:
        raise ValueError("animation needs at least two frames")
    thresholds = np.linspace(0.90, 0.05, frames)
    curves = threshold_metrics(y_test, probabilities, thresholds)
    rng = np.random.default_rng(SEED)
    fig, (ax, info) = plt.subplots(1, 2, figsize=(9, 4.3), gridspec_kw={"width_ratios": [3, 1.7]})
    for label, color in ((0, NEGATIVE), (1, POSITIVE)):
        rows = y_test == label
        ax.scatter(probabilities[rows], label + rng.uniform(-0.1, 0.1, rows.sum()),
                   s=14, alpha=0.55, c=color, label=f"Actual class {label}")
    ax.set(xlim=(0, 1), ylim=(-0.3, 1.3), yticks=[0, 1],
           yticklabels=["Negative", "Positive"], xlabel="Predicted positive probability",
           ylabel="Actual test label", title="Fixed scores: the decision line moves")
    ax.legend(loc="center right", fontsize=9)
    line = ax.axvline(thresholds[0], color="#333333", lw=2)
    ax.text(0.02, 0.98, "Predict 1 on or right of the line", transform=ax.transAxes,
            fontsize=9, va="top", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    info.axis("off")
    text_box = info.text(0.02, 0.92, "", va="top", fontsize=12, family="monospace")
    fig.suptitle("Thresholding changes actions, not model training")
    fig.text(0.5, 0.02, "Test demonstration only. Recall and FP are monotone; precision need not be.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))

    def update(i):
        line.set_xdata([thresholds[i], thresholds[i]])
        text_box.set_text(
            f"Threshold  {thresholds[i]:.3f}\n\n"
            f"Precision  {curves['precision'][i]:.3f}\n"
            f"Recall     {curves['recall'][i]:.3f}\n"
            f"F1         {curves['f1'][i]:.3f}\n"
            f"Alerts     {curves['predicted_positives'][i]}\n\n"
            f"TP {curves['tp'][i]:3d}   FP {curves['fp'][i]:3d}\n"
            f"FN {curves['fn'][i]:3d}   TN {curves['tn'][i]:3d}"
        )
        return line, text_box

    animation = FuncAnimation(fig, update, frames=frames, interval=125, blit=False)
    animation.save(path, writer=PillowWriter(fps=8), dpi=90)
    plt.close(fig)


def plot_threshold_tradeoff(y_validation, probabilities, selected_f1, path):
    thresholds = np.unique(np.append(np.linspace(0, 1, 201), selected_f1))
    curves = threshold_metrics(y_validation, probabilities, thresholds)
    fig, (ax, volume) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    for metric, color in (("precision", NEGATIVE), ("recall", POSITIVE), ("f1", SYNTHETIC)):
        ax.plot(thresholds, curves[metric], color=color, label=metric.capitalize())
    # F-beta is shown as an optional contrast, not as a business-cost proxy.
    ax.plot(thresholds, curves["f2"], "--", color="#7856A0", label="F2 (recall emphasis)", alpha=0.8)
    volume.plot(thresholds, 100 * curves["alert_rate"], color="#333333")
    for panel in (ax, volume):
        panel.axvline(0.5, color="gray", ls=":", label="Default 0.50" if panel is ax else None)
        panel.axvline(selected_f1, color=SYNTHETIC, ls="--",
                      label=f"Validation max F1: {selected_f1:.3f}" if panel is ax else None)
        panel.grid(alpha=0.2)
    ax.set(ylabel="Validation metric", ylim=(-0.03, 1.07),
           title="Precision, recall, F1 and review volume: one score distribution")
    ax.legend(ncol=2, fontsize=9, loc="best")
    volume.set(xlabel="Decision threshold", ylabel="Alerts (%)", xlim=(0, 1))
    finish(fig, path, "Metric-optimal threshold need not be business-optimal. Selection uses validation only.")


def plot_decision_boundary(ax, X_train, y_train, model, limits):
    x = np.linspace(*limits[0], 160)
    y = np.linspace(*limits[1], 160)
    xx, yy = np.meshgrid(x, y)
    probabilities = model.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, probabilities, levels=[0, 0.5, 1],
                colors=[NEGATIVE, POSITIVE], alpha=0.10)
    boundary = ax.contour(xx, yy, probabilities, levels=[0.5], colors="#333333", linewidths=2)
    ax.clabel(boundary, fmt={0.5: "p = 0.50"}, fontsize=10)
    scatter_classes(ax, X_train, y_train, size=14)
    ax.set(xlim=limits[0], ylim=limits[1])
    ax.legend(fontsize=9, loc="best")


def plot_class_weight_effect(data, models, weights, path):
    X, y = data["X"][data["train"]], data["y"][data["train"]]
    limits = [(X[:, i].min() - 0.5, X[:, i].max() + 0.5) for i in range(2)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, (label, model) in zip(axes, models.items()):
        plot_decision_boundary(ax, X, y, model, limits)
        ax.set_title(label)
    fig.suptitle(f"Identical training rows; weights change the optimization objective (w0={weights['0']:.3f}, w1={weights['1']:.3f})")
    finish(fig, path, "Boundary shown at each model's 0.50 score. No dataset balancing or resampling is performed here.")


def plot_resampling_comparison(X_train, y_train, panels, path):
    limits = [(X_train[:, i].min() - 0.5, X_train[:, i].max() + 0.5) for i in range(2)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (name, panel) in zip(axes.flat, panels.items()):
        X, y = panel["X"], panel["y"]
        if "multiplicity" in panel:
            # Identical duplicates occupy the same coordinates; marker area reveals multiplicity.
            scatter_classes(ax, X_train, y_train, size=7, alpha=0.3)
            rows = y_train == 1
            ax.scatter(X_train[rows, 0], X_train[rows, 1],
                       s=8 * panel["multiplicity"][rows], c=POSITIVE, alpha=0.35,
                       edgecolors="#9E471A", linewidths=0.5, label="Area reflects duplicate count")
            note = "Copies share coordinates; marker area shows exposure"
        elif name == "SMOTE":
            scatter_classes(ax, X_train, y_train, size=10, alpha=0.25)
            new = X[len(X_train):]
            ax.scatter(new[:, 0], new[:, 1], s=5, color=SYNTHETIC, alpha=0.28,
                       label=f"New synthetic points (n={len(new)})")
            note = "Training-fitted scaled distances; numerical interpolation"
        else:
            scatter_classes(ax, X, y, size=12, alpha=0.45)
            note = "All original training rows" if name.startswith("Original") else "Majority observations discarded"
        counts = np.bincount(y, minlength=2)
        ax.set(title=f"{name}: class counts {counts[0]} / {counts[1]}", xlim=limits[0], ylim=limits[1])
        ax.legend(fontsize=8, loc="upper right")
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    fig.suptitle("Resampling changes training exposure; validation and test stay untouched")
    finish(fig, path, "All panels share axis limits. The sampler views show training data, not new independent evidence.")


def smote_pair(X_minority, distance_coordinates=None):
    X = np.asarray(X_minority, dtype=float)
    distances_X = X if distance_coordinates is None else np.asarray(distance_coordinates, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or len(X) < 2 or distances_X.shape != X.shape or not np.isfinite(X).all() or not np.isfinite(distances_X).all():
        raise ValueError("SMOTE geometry needs at least two finite 2D minority rows")
    anchor = int(np.argmin(np.sum((distances_X - distances_X.mean(axis=0)) ** 2, axis=1)))
    distances = np.sum((distances_X - distances_X[anchor]) ** 2, axis=1)
    distances[anchor] = np.inf
    return anchor, int(np.argmin(distances))


def animate_smote(X_minority, path, frames=32, distance_coordinates=None):
    if not isinstance(frames, int) or frames < 2:
        raise ValueError("animation needs at least two frames")
    i, j = smote_pair(X_minority, distance_coordinates)
    anchor, neighbor = X_minority[i], X_minority[j]
    delta = neighbor - anchor
    padding = max(float(np.linalg.norm(delta)) * 0.6, 0.1)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(X_minority[:, 0], X_minority[:, 1], color=POSITIVE, alpha=0.35,
               s=30, label="Original training minority")
    ax.plot([anchor[0], neighbor[0]], [anchor[1], neighbor[1]], "--", color="#444444")
    ax.scatter(*anchor, s=95, c=POSITIVE, edgecolor="black", label="$x_i$: anchor")
    ax.scatter(*neighbor, s=95, c=NEGATIVE, edgecolor="black", label="$x_j$: nearest minority neighbor")
    ax.annotate("$x_i$", anchor, xytext=(-20, -20), textcoords="offset points")
    ax.annotate("$x_j$", neighbor, xytext=(8, 8), textcoords="offset points")
    moving = ax.scatter(*anchor, s=100, c=SYNTHETIC, marker="D", label="$x_{new}$")
    moving_label = ax.annotate("$x_{new}$", anchor, xytext=(8, 5), textcoords="offset points")
    current = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=12,
                      bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    ax.set(xlim=(min(anchor[0], neighbor[0]) - padding, max(anchor[0], neighbor[0]) + padding),
           ylim=(min(anchor[1], neighbor[1]) - padding, max(anchor[1], neighbor[1]) + padding),
           xlabel="Feature 1 (raw units)", ylabel="Feature 2 (raw units)",
           title=r"SMOTE geometry: $x_{new} = x_i + \lambda(x_j - x_i)$")
    ax.legend(fontsize=9, loc="lower right")
    fig.text(0.5, 0.02, "Zoomed training neighborhood. One scalar moves all coordinates along the same segment.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    def update(frame):
        fraction = frame / (frames - 1)
        point = interpolate(anchor, neighbor, fraction)
        moving.set_offsets(point[None, :])
        moving_label.xy = point
        current.set_text(f"Interpolation fraction = {fraction:.2f}\nEndpoints shown for intuition")
        return moving, moving_label, current

    animation = FuncAnimation(fig, update, frames=frames, interval=125, blit=False)
    animation.save(path, writer=PillowWriter(fps=8), dpi=95)
    plt.close(fig)


def smote_failure_fixture(dependencies):
    """Construct a training-only geometric diagnostic, not a fitted-model dataset."""
    rng = np.random.default_rng(SEED)
    majority = rng.normal([0, 0], [0.13, 0.32], size=(250, 2))
    minority = np.vstack([rng.normal([-0.8, 0], [0.045, 0.08], (3, 2)),
                          rng.normal([0.8, 0], [0.045, 0.08], (3, 2))])
    X = np.vstack([majority, minority])
    y = np.concatenate([np.zeros(len(majority), dtype=int), np.ones(len(minority), dtype=int)])
    # k=5 deliberately spans both three-point minority islands.
    # These rows are a constructed training fixture; there is no held-out scoring.
    sampled_X, sampled_y = dependencies["imblearn.over_sampling"].SMOTE(
        random_state=SEED, k_neighbors=5
    ).fit_resample(X, y)
    generated = sampled_X[len(X):]
    in_region = (np.abs(generated[:, 0]) < 0.25) & (np.abs(generated[:, 1]) < 0.65)
    return X, y, generated, in_region


def plot_smote_failure(path, dependencies):
    from matplotlib.patches import Rectangle
    X, y, generated, in_region = smote_failure_fixture(dependencies)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax in axes:
        scatter_classes(ax, X, y, size=35, alpha=0.6)
        ax.add_patch(Rectangle((-0.25, -0.65), 0.5, 1.3, fill=False,
                               hatch="//", edgecolor="#9C3E35", linewidth=1.2))
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.2, 1.2))
    axes[0].set_title("Six minority rows in two disconnected islands")
    axes[1].scatter(generated[:, 0], generated[:, 1], s=13, color=SYNTHETIC, alpha=0.6,
                    label="SMOTE-generated training points")
    axes[1].scatter(generated[in_region, 0], generated[in_region, 1], s=24,
                    facecolor="none", edgecolor="#9C3E35", label="Inside diagnostic majority region")
    axes[1].set_title(f"k=5 bridges islands: {in_region.sum()} / {len(generated)} in marked region")
    axes[1].annotate("Potentially questionable\nsynthetic region",
                     xy=(0.0, 0.12), xytext=(-0.92, 0.85),
                     arrowprops={"arrowstyle": "->", "color": "#9C3E35"}, fontsize=10)
    for ax in axes:
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("SMOTE's local geometry assumption can fail")
    finish(fig, path, "Constructed training-only fixture. The rectangle is an illustrative diagnostic, not proof of invalid labels or a general failure rate.")
    return {"generated_count": int(len(generated)), "inside_diagnostic_region": int(in_region.sum()),
            "fixture": "250 central negatives; 3 positives per island; SMOTE k=5; seed 42; raw coordinates",
            "region": "|feature1| < 0.25 and |feature2| < 0.65"}


def plot_roc_vs_pr(y_test, probabilities, path):
    from sklearn.metrics import precision_recall_curve, roc_curve
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    ap = average_precision_score(y_test, probabilities)
    prevalence = float(np.mean(y_test))
    fig, (roc, pr) = plt.subplots(1, 2, figsize=(11, 4.8))
    roc.plot(fpr, tpr, color=NEGATIVE, lw=2, label=f"ROC-AUC = {auc:.3f}")
    roc.plot([0, 1], [0, 1], "--", c="gray", label="No-skill diagonal")
    roc.set(xlabel="False-positive rate", ylabel="True-positive rate",
            xlim=(0, 1), ylim=(0, 1.03), title="Ranking across all operating points")
    pr.step(recall, precision, where="post", color=POSITIVE, label=f"Average precision = {ap:.3f}")
    pr.axhline(prevalence, ls="--", c="gray", label=f"No-skill precision = prevalence ({prevalence:.1%})")
    pr.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.03),
           title="Precision exposes the rare-positive alert burden")
    fixed = threshold_metrics(y_test, probabilities, [0.5])
    fixed_fpr = fixed["fp"][0] / (fixed["fp"][0] + fixed["tn"][0])
    roc.scatter(fixed_fpr, fixed["recall"][0], c=SYNTHETIC, zorder=5, label="Threshold 0.50")
    pr.scatter(fixed["recall"][0], fixed["precision"][0], c=SYNTHETIC, zorder=5, label="Threshold 0.50")
    for ax in (roc, pr):
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle(f"Same test scores: ROC vs precision-recall at {prevalence:.1%} positive prevalence")
    finish(fig, path, "Synthetic 99/1 target mixture; AP is recall-weighted average precision, not trapezoidal PR area.")


def plot_precision_prevalence(path):
    prevalence = np.geomspace(0.001, 0.5, 200)
    precision = precision_from_prevalence(prevalence)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(100 * prevalence, precision, color=NEGATIVE, lw=2)
    for value in (0.001, 0.01, 0.05, 0.5):
        ppv = float(precision_from_prevalence(value))
        ax.scatter(100 * value, ppv, c=POSITIVE, zorder=5)
        ax.annotate(f"{value:.1%} prevalence\n{ppv:.1%} precision",
                    (100 * value, ppv), xytext=(8, 8 if value < 0.5 else -35),
                    textcoords="offset points", fontsize=9)
    ax.set(xscale="log", xlabel="Positive prevalence (%) — log scale", ylabel="Expected precision (PPV)",
           ylim=(0, 1.08), xlim=(0.075, 65),
           title="Precision depends on prevalence, even at fixed TPR and FPR")
    ax.grid(alpha=0.25)
    ax.text(0.03, 0.93, "Illustrative assumptions: TPR = 0.90, FPR = 0.05\n"
            "PPV = TPR × prevalence / [TPR × prevalence + FPR × (1 − prevalence)]",
            transform=ax.transAxes, va="top", fontsize=10)
    finish(fig, path, "Theoretical Bayes calculation, not an experiment. Class-conditional error rates are assumed unchanged.")


def plot_business_cost(y_validation, probabilities, policies, path):
    thresholds = np.unique(np.concatenate([np.linspace(0, 1, 201),
                                          [scenario["threshold"] for scenario in policies["cost"]]]))
    curves = threshold_metrics(y_validation, probabilities, thresholds)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, scenario in zip(axes, policies["cost"]):
        c_fp, c_fn, selected = scenario["c_fp"], scenario["c_fn"], scenario["threshold"]
        ax.plot(thresholds, empirical_cost(curves, c_fp, c_fn), color=NEGATIVE,
                label="Validation error cost")
        ax.axvline(0.5, color="gray", ls=":", label="Default 0.50")
        ax.axvline(selected, color=POSITIVE, ls="--", label=f"Validation minimum: {selected:.3f}")
        ax.set(xlabel="Decision threshold", ylabel="Total empirical cost (illustrative units)",
               title=f"C_FP = {c_fp:g}, C_FN = {c_fn:g}")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle("Same model, different error costs: different preferred policies")
    finish(fig, path, "Cost = FP × C_FP + FN × C_FN on validation only. These are pedagogical assumptions, not real-world estimates.")


def plot_3d_cost_surface(y_validation, probabilities, path, go):
    # C_FP=1 fixes the units: changing the ratio then has a clear cost meaning.
    thresholds = np.unique(np.append(np.linspace(0, 1, 101), np.nextafter(max(probabilities), np.inf)))
    ratios = np.geomspace(0.2, 50, 35)
    curves = threshold_metrics(y_validation, probabilities, thresholds)
    total_cost = curves["fp"][None, :] + ratios[:, None] * curves["fn"][None, :]
    fig = go.Figure(go.Surface(
        x=thresholds, y=ratios, z=total_cost, colorscale="Viridis",
        colorbar={"title": "Total cost"},
        hovertemplate="Threshold: %{x:.3f}<br>C_FN/C_FP: %{y:.2f}<br>Validation cost: %{z:.1f}<extra></extra>",
    ))
    marker_x, marker_y, marker_z = [], [], []
    for ratio in (0.2, 1, 2, 10, 20, 50):
        # Mark exact validation minima, not just minima on the display grid.
        candidates = np.append(np.unique(probabilities), np.nextafter(max(probabilities), np.inf))
        exact = threshold_metrics(y_validation, probabilities, candidates)
        costs = empirical_cost(exact, 1, ratio)
        index = np.flatnonzero(costs == costs.min())[-1]
        marker_x.append(float(candidates[index]))
        marker_y.append(ratio)
        marker_z.append(float(costs[index]))
    fig.add_trace(go.Scatter3d(x=marker_x, y=marker_y, z=marker_z, mode="markers+lines",
                              name="Exact validation minima", marker={"color": "#D87522", "size": 5},
                              hovertemplate="Optimal threshold: %{x:.3f}<br>Ratio: %{y:.2f}<br>Cost: %{z:.1f}<extra></extra>"))
    fig.update_layout(
        title={"text": "No universal threshold: empirical cost depends on error-cost ratio<br>"
                       "<sup>Validation only; C_FP=1, C_FN=ratio. Illustrative units; rotate, zoom and hover.</sup>", "x": 0.5},
        scene={"xaxis_title": "Decision threshold", "yaxis_title": "C_FN / C_FP (log scale)",
               "yaxis": {"type": "log"}, "zaxis_title": "Total validation cost"},
        margin={"l": 15, "r": 15, "t": 90, "b": 30}, height=700,
    )
    # Embed Plotly itself so the artifact works without a network connection.
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={"displaylogo": False, "responsive": True})
    return {"threshold_grid": len(thresholds), "cost_ratio_grid": len(ratios),
            "normalization": "C_FP=1; C_FN=ratio", "minimum_thresholds": marker_x, "ratios": marker_y}


def plot_ranking_vs_decision(y_test, probabilities, path):
    thresholds = [0.8, 0.5, 0.25, 0.1]
    curves = threshold_metrics(y_test, probabilities, thresholds)
    auc, ap = roc_auc_score(y_test, probabilities), average_precision_score(y_test, probabilities)
    fig, (flow, table_ax) = plt.subplots(2, 1, figsize=(11, 5.8), gridspec_kw={"height_ratios": [1, 2]})
    flow.axis("off")
    labels = ["Fitted model", "Fixed score / probability", "Threshold policy", "Binary action"]
    positions = [0.1, 0.37, 0.64, 0.89]
    for pos, label in zip(positions, labels):
        flow.text(pos, 0.6, label, ha="center", va="center", fontsize=11,
                  bbox={"boxstyle": "round,pad=0.5", "facecolor": "#EAF1F4", "edgecolor": NEGATIVE})
    for left, right in zip(positions[:-1], positions[1:]):
        flow.annotate("", xy=(right - 0.09, 0.6), xytext=(left + 0.09, 0.6),
                      arrowprops={"arrowstyle": "->", "color": "#555555"})
    flow.text(0.37, 0.1, "Ranking: ROC-AUC and AP     |     Probability quality: calibration",
              ha="center", fontsize=10)
    table_ax.axis("off")
    rows = []
    for i, threshold in enumerate(thresholds):
        rows.append([f"{threshold:.2f}", f"{auc:.4f}", f"{ap:.4f}",
                     f"{curves['precision'][i]:.3f}", f"{curves['recall'][i]:.3f}", f"{curves['f1'][i]:.3f}",
                     f"{curves['tp'][i]} / {curves['fp'][i]}", f"{curves['tn'][i]} / {curves['fn'][i]}"])
    table = table_ax.table(cellText=rows,
                           colLabels=["Threshold", "ROC-AUC", "AP", "Precision", "Recall", "F1", "TP / FP", "TN / FN"],
                           loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.1)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#DCE8EE")
        elif c in (1, 2):
            cell.set_facecolor("#E8F1EA")
    fig.suptitle("Changing only the threshold preserves ranking and probability scores")
    finish(fig, path, "Same fitted classifier and test probabilities in every row. Green columns remain constant; decisions change.")


def calibration_summary(y_test, probabilities, n_bins=6):
    y, p = validate_scores(y_test, probabilities)
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("calibration needs at least two bins")
    observed, mean_score = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    # Equal-score groups may collapse bins, which sklearn handles explicitly.
    return mean_score, observed, float(brier_score_loss(y, p))


def plot_calibration(y_test, scores, path):
    fig, (ax, density) = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [2, 1]})
    scores_report = {}
    for (label, probabilities), color in zip(scores.items(), (NEGATIVE, POSITIVE)):
        mean_score, observed, brier = calibration_summary(y_test, probabilities)
        ax.plot(mean_score, observed, "o-", color=color, label=f"{label}: Brier = {brier:.4f}")
        density.hist(probabilities, bins=np.linspace(0, 1, 21), histtype="step", color=color, label=label)
        scores_report[label] = {"brier": brier, "mean_score_bins": mean_score.tolist(),
                                "observed_positive_fraction_bins": observed.tolist()}
    ax.plot([0, 1], [0, 1], "--", c="gray", label="Perfect calibration reference")
    ax.set(xlabel="Mean predicted probability within bin", ylabel="Observed positive fraction",
           xlim=(0, 1), ylim=(0, 1), title="Ranking quality and calibration are different properties")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    density.set(xlabel="Predicted positive probability", ylabel="Test count")
    density.legend(fontsize=9)
    finish(fig, path, f"Untouched test set: {sum(y_test)} positives. Six quantile bins per model; no calibrator fitted. Brier is not calibration-only.")
    return scores_report


def metrics_at(y, probabilities, threshold):
    curves = threshold_metrics(y, probabilities, [threshold])
    return {name: values[0].item() for name, values in curves.items()}


def score_summary(y, probabilities):
    y, probabilities = validate_scores(y, probabilities)
    if len(np.unique(y)) != 2:
        raise ValueError("ranking summaries require both classes")
    return {"roc_auc": float(roc_auc_score(y, probabilities)),
            "average_precision": float(average_precision_score(y, probabilities)),
            "positive_prevalence": float(y.mean()),
            "brier": float(brier_score_loss(y, probabilities))}


def print_summary(data, weights, policies, results, rare_results, calibration, seconds):
    print("\nDataset: synthetic 2D; class counts [negative, positive]")
    for label in ("train", "validation", "test"):
        y = data["y"][data[label]]
        print(f"  {label:<12} counts={np.bincount(y).tolist()}, prevalence={y.mean():.3%}")
    print("Balanced class weights (original training only):", weights)
    print("\nBaseline model, untouched test ranking")
    print(f"  ROC-AUC: {results['scores']['roc_auc']:.4f}")
    print(f"  Average precision (AP): {results['scores']['average_precision']:.4f}")
    for label, result in results["policies"].items():
        print(f"\n{label} (test evaluation, selection uses validation)")
        print(f"  Threshold={result['threshold']:.6f}, precision={result['precision']:.4f}, "
              f"recall={result['recall']:.4f}, F1={result['f1']:.4f}, alerts={result['alert_rate']:.2%}")
        print(f"  TP={result['tp']} FP={result['fp']} FN={result['fn']} TN={result['tn']}")
    print("\nIllustrative cost policies (not real-world cost estimates)")
    for scenario in results["cost_policies"]:
        print(f"  C_FP={scenario['c_fp']:g}, C_FN={scenario['c_fn']:g}: "
              f"validation t={scenario['threshold']:.6f}, validation cost={scenario['validation_cost']:.1f}, "
              f"test cost={scenario['test_cost']:.1f}")
    print("\nCalibration: binary test Brier loss (not a calibration-only score)")
    for label, value in calibration.items():
        print(f"  {label}: {value['brier']:.6f}")
    print(f"\n99/1 synthetic test: ROC-AUC={rare_results['roc_auc']:.4f}, "
          f"AP={rare_results['average_precision']:.4f}, prevalence={rare_results['positive_prevalence']:.2%}")
    print(f"\nGeneration completed in {seconds:.1f}s. Interpretation candidates remain pending author review.")


def generate_lab(output_dir=OUTPUT_DIR, frames=36):
    if not isinstance(frames, int) or frames < 2 or frames > 120:
        raise ValueError("frames must be an integer from 2 to 120")
    started = time.perf_counter()
    dependencies = optional_dependencies()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11,
                         "figure.titlesize": 14, "axes.spines.top": False, "axes.spines.right": False})

    data = make_dataset()
    rare = make_dataset(n_samples=10000, prevalence=0.01, n_features=10, class_sep=0.8, n_clusters_per_class=2)
    models, weights = fit_models(data)
    rare_models, _ = fit_models(rare)
    X, y = data["X"], data["y"]
    training, validation, test = data["train"], data["validation"], data["test"]
    unchanged = {label: (X[data[label]].copy(), y[data[label]].copy()) for label in ("validation", "test")}
    panels = resample_training(X[training], y[training], dependencies)

    baseline = models["Ordinary"]
    validation_p = baseline.predict_proba(X[validation])[:, 1]
    # Policies are frozen here, before any final test scores are obtained.
    policies = validation_policies(y[validation], validation_p)
    test_scores = {label: model.predict_proba(X[test])[:, 1] for label, model in models.items()}
    test_p = test_scores["Ordinary"]
    rare_y = rare["y"][rare["test"]]
    rare_p = rare_models["Ordinary"].predict_proba(rare["X"][rare["test"]])[:, 1]

    def artifact(name, renderer, *args, **kwargs):
        destination = output_dir / name
        print(f"Generating {name}", flush=True)
        return renderer(*args, path=destination, **kwargs)

    artifact("01_class_imbalance.png", plot_class_distribution, data)
    artifact("02_accuracy_trap.png", plot_accuracy_trap, y[test], test_p)
    artifact("03_confusion_matrix_thresholds.png", plot_confusion_thresholds, y[test], test_p)
    artifact("04_threshold_animation.gif", animate_threshold, y[test], test_p, frames=frames)
    artifact("05_threshold_metrics.png", plot_threshold_tradeoff, y[validation], validation_p, policies["f1"])
    artifact("06_class_weights_boundary.png", plot_class_weight_effect, data, models, weights)
    artifact("07_resampling_comparison.png", plot_resampling_comparison, X[training], y[training], panels)
    minority = X[training][y[training] == 1]
    scaled_minority = baseline.named_steps["standardscaler"].transform(minority)
    artifact("08_smote_geometry.gif", animate_smote, minority, frames=frames, distance_coordinates=scaled_minority)
    failure = artifact("09_smote_failure_mode.png", plot_smote_failure, dependencies=dependencies)
    artifact("10_roc_vs_pr.png", plot_roc_vs_pr, rare_y, rare_p)
    artifact("11_precision_vs_prevalence.png", plot_precision_prevalence)
    artifact("12_business_cost_threshold.png", plot_business_cost, y[validation], validation_p, policies)
    surface = artifact("13_cost_surface.html", plot_3d_cost_surface, y[validation], validation_p,
                       go=dependencies["plotly.graph_objects"])
    artifact("14_ranking_vs_decision.png", plot_ranking_vs_decision, y[test], test_p)
    calibration = artifact("15_calibration.png", plot_calibration, y[test], test_scores)
    for label, (old_X, old_y) in unchanged.items():
        if not np.array_equal(X[data[label]], old_X) or not np.array_equal(y[data[label]], old_y):
            raise RuntimeError(f"{label} data changed during generation")

    test_policies = {"Threshold 0.50": metrics_at(y[test], test_p, 0.5),
                     "Validation max-F1 threshold": metrics_at(y[test], test_p, policies["f1"])}
    cost_policies = []
    for scenario in policies["cost"]:
        threshold = scenario["threshold"]
        c_fp, c_fn = scenario["c_fp"], scenario["c_fn"]
        val_count = threshold_metrics(y[validation], validation_p, [threshold])
        test_count = threshold_metrics(y[test], test_p, [threshold])
        cost_policies.append({
            **scenario, "validation_cost": float(empirical_cost(val_count, c_fp, c_fn)[0]),
            "test_cost": float(empirical_cost(test_count, c_fp, c_fn)[0]),
            "test_metrics": metrics_at(y[test], test_p, threshold),
        })
    results = {"scores": score_summary(y[test], test_p), "policies": test_policies,
               "cost_policies": cost_policies,
               "always_negative": metrics_at(y[test], np.zeros(len(test)), 0.5)}
    rare_results = {**score_summary(rare_y, rare_p), "threshold_0.5": metrics_at(rare_y, rare_p, 0.5)}
    elapsed = time.perf_counter() - started
    artifacts = [file for file in sorted(output_dir.iterdir())
                 if file.name[:2].isdigit() and file.suffix in {".png", ".gif", ".html"}]
    report = {
        "configuration": {
            "primary": data["configuration"], "rare": rare["configuration"],
            "split": "60/20/20 stratified; both split seeds 42; no cross-validation",
            "model": "StandardScaler then LogisticRegression(C=1, solver=lbfgs, max_iter=1000)",
            "weighting": weights, "resampling": "training only; train-fitted scaled distances; 1:1 target; seed 42; SMOTE k=5",
            "threshold_selection": "all distinct validation decisions; >= rule; highest threshold on ties",
            "cost_scenarios": [{"c_fp": a, "c_fn": b} for a, b in COST_SCENARIOS],
            "calibration": "6 quantile bins per model; test only; no calibrator fitted",
            "animation": {"frames": frames, "fps": 8, "threshold_range": [0.90, 0.05]},
            "versions": {name: version(name) for name in ("numpy", "scikit-learn", "imbalanced-learn", "matplotlib", "Pillow", "plotly")},
            "python": platform.python_version(),
        },
        "split_counts": {label: np.bincount(y[data[label]], minlength=2).tolist() for label in ("train", "validation", "test")},
        "resampled_training_counts": {label: np.bincount(panel["y"], minlength=2).tolist() for label, panel in panels.items()},
        "hypotheses": [
            "Majority accuracy can obscure positive recall.",
            "Policy changes preserve ranking while changing confusion counts and workload.",
            "Changing illustrative error costs can change validation-selected policies.",
            "Weighted and ordinary probabilities may differ in calibration; no better model is assumed.",
            "A deliberately cross-island SMOTE neighborhood can generate points in a central majority region.",
        ],
        "primary_test": results, "rare_test": rare_results,
        "calibration_test": calibration, "smote_diagnostic": failure, "cost_surface": surface,
        "interpretation_candidates": [
            "Use recorded test precision/recall and workload to explain the policy trade-off.",
            "Compare the two validation-selected cost policies without assuming real business applicability.",
            "A difference in Brier loss does not isolate calibration; inspect bins and positive counts.",
            "SMOTE diagnostic illustrates a constructed assumption failure, not a general failure rate.",
        ],
        "interpretation_status": "Pending author review",
        "limitations": [
            "One synthetic seed, one split per dataset; not a benchmark or production validation.",
            "Only 30 primary and 20 rare-dataset test positives; calibration and recall estimates are uncertain.",
            "No repeated-seed uncertainty, capacity constraints, temporal drift or calibration fitting.",
            "Resampling panels demonstrate exposure/geometry and do not benchmark resampled models.",
            "Cost surface uses C_FP=1 and illustrative C_FN ratios; display grid can miss exact minima.",
            "Theoretical prevalence plot assumes fixed class-conditional error rates.",
            "Constructed failure fixture forces cross-island neighbors with k=5.",
        ],
        "initial_rare_design_check": {
            "configuration": {"n_samples": 10000, "prevalence": 0.01, "n_features": 10,
                              "n_clusters_per_class": 1, "class_sep": 1.2, "flip_y": 0, "seed": 42},
            "executed_test_roc_auc": 1.0, "executed_test_average_precision": 1.0,
            "reason_for_geometry_revision": "Perfect ranking did not illustrate the intended ROC/PR trade-off; final view uses explicit overlap and two clusters. This is educational generator design, not a benchmark comparison.",
            "interpretation_status": "Pending author review",
        },
        "generation_seconds": elapsed,
        "artifacts": [{"name": file.name, "bytes": file.stat().st_size} for file in artifacts],
    }
    report_path = output_dir / "visual_run.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print_summary(data, weights, policies, results, rare_results, calibration, elapsed)
    print("Artifacts:", output_dir.resolve())
    print("Experiment record:", report_path.resolve())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="default: assets beside the script")
    parser.add_argument("--frames", type=int, default=36, help="2–120 frames per animation; default 36")
    args = parser.parse_args()
    try:
        generate_lab(args.output_dir, args.frames)
    except (RuntimeError, ValueError, ImportError, OSError) as error:
        parser.exit(1, f"Visual lab failed: {error}\n")


if __name__ == "__main__":
    main()
