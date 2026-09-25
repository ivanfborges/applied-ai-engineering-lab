"""Generate the Day 22 Classification Metrics Visual Lab from synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from from_scratch import classification_metrics


SEED = 22
BLUE = "#2a6287"
ORANGE = "#cc693c"
GREEN = "#247a5a"
PURPLE = "#765a9a"
GRAY = "#53606b"
COLORS = {"accuracy": BLUE, "precision": ORANGE, "recall": GREEN, "f1": PURPLE}
OUTPUT_NAMES = (
    "confusion_matrix_foundation.png",
    "accuracy_imbalance.png",
    "threshold_tradeoff.gif",
    "metrics_vs_threshold.png",
    "precision_recall_denominators.png",
    "f1_harmonic_mean.png",
    "f1_surface_3d.html",
    "prevalence_metric_sensitivity.png",
    "error_tradeoff_operational.png",
)


def harmonic_mean(precision, recall):
    """Elementwise F1, with zero when both inputs are zero."""
    p, r = np.broadcast_arrays(
        np.asarray(precision, dtype=float), np.asarray(recall, dtype=float)
    )
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("precision and recall must be finite")
    if np.any((p < 0) | (p > 1) | (r < 0) | (r > 1)):
        raise ValueError("precision and recall must be in [0, 1]")
    result = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(result) if result.ndim == 0 else result


def metrics_at_threshold(y_true, probabilities, threshold):
    """Apply one threshold and return counts, rates, and alert volume."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be a finite one-dimensional array")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    predicted = (probabilities >= threshold).astype(int)
    result = classification_metrics(y_true, predicted)
    result["predicted_positives"] = result["tp"] + result["fp"]
    return result


def threshold_table(y_true, probabilities, thresholds):
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or len(thresholds) == 0:
        raise ValueError("thresholds must be a nonempty one-dimensional array")
    return [dict(threshold=float(t), **metrics_at_threshold(y_true, probabilities, t))
            for t in thresholds]


def rates_at_prevalence(prevalence, tpr, fpr):
    """Expected metrics if conditional TPR and FPR stay fixed."""
    p = np.asarray(prevalence, dtype=float)
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError("prevalence must be strictly between 0 and 1")
    if not (np.isfinite(tpr) and np.isfinite(fpr)
            and 0 <= tpr <= 1 and 0 <= fpr <= 1):
        raise ValueError("TPR and FPR must be finite rates in [0, 1]")
    tp_share = tpr * p
    fp_share = fpr * (1 - p)
    precision = np.divide(
        tp_share, tp_share + fp_share,
        out=np.zeros_like(p), where=(tp_share + fp_share) > 0,
    )
    return {
        "accuracy": tp_share + (1 - fpr) * (1 - p),
        "precision": precision,
        "recall": np.full_like(p, tpr),
        "f1": harmonic_mean(precision, tpr),
    }


def make_lab_data():
    """Reserve validation for threshold exploration and test for final checks."""
    X, y = make_classification(
        n_samples=2400, n_features=10, n_informative=6, n_redundant=2,
        weights=[0.85, 0.15], flip_y=0.02, class_sep=1.1, random_state=SEED,
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED
    )
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
    )
    model.fit(X_train, y_train)
    return {
        "train_n": len(y_train),
        "val_y": y_val,
        "val_p": model.predict_proba(X_val)[:, 1],
        "test_y": y_test,
        "test_p": model.predict_proba(X_test)[:, 1],
    }


def save_figure(fig, output_dir, filename):
    fig.savefig(output_dir / filename, dpi=130, bbox_inches="tight")
    plt.close(fig)


def draw_matrix(ax, counts, *, compact=False):
    """True labels are rows; predicted labels are columns."""
    ax.imshow([[0, 1], [1, 0]], cmap=ListedColormap(["#e5f4ea", "#fce9e3"]))
    cells = [
        (0, 0, "TN", "Actual negative\nPredicted negative", counts["tn"], True),
        (0, 1, "FP", "Actual negative\nPredicted positive", counts["fp"], False),
        (1, 0, "FN", "Actual positive\nPredicted negative", counts["fn"], False),
        (1, 1, "TP", "Actual positive\nPredicted positive", counts["tp"], True),
    ]
    for row, col, abbreviation, meaning, count, correct in cells:
        if compact:
            label = f"{abbreviation}\n{count}"
            fontsize = 15
        else:
            label = (
                f"{abbreviation}  |  {count}\n{meaning}\n"
                f"{'Correct' if correct else 'Error'}"
            )
            fontsize = 11
        ax.text(col, row, label, ha="center", va="center", fontsize=fontsize,
                color="#17334b", linespacing=1.35)
    ax.set_xticks([0, 1], ["Negative (0)", "Positive (1)"])
    ax.set_yticks([0, 1], ["Negative (0)", "Positive (1)"])
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Predicted class", labelpad=10)
    ax.set_ylabel("Actual class")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((-0.5, -0.5), 2, 2, fill=False, ec="white", lw=3))


def plot_confusion_matrix_foundation(counts, output_dir):
    fig, (matrix_ax, logic_ax) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.05, 1]}
    )
    draw_matrix(matrix_ax, counts)
    logic_ax.axis("off")
    n = sum(counts[key] for key in ("tn", "fp", "fn", "tp"))
    blocks = [
        ("Correct decisions", f"TP + TN = {counts['tp'] + counts['tn']}",
         f"Accuracy = {counts['tp'] + counts['tn']} / {n} = {counts['accuracy']:.3f}"),
        ("Predicted positives", f"TP + FP = {counts['tp'] + counts['fp']}",
         f"Precision = TP / (TP + FP) = {counts['precision']:.3f}"),
        ("Actual positives", f"TP + FN = {counts['tp'] + counts['fn']}",
         f"Recall = TP / (TP + FN) = {counts['recall']:.3f}"),
    ]
    for i, (heading, denominator, metric) in enumerate(blocks):
        y = 0.87 - i * 0.30
        logic_ax.text(0.05, y, heading, fontsize=13, weight="bold",
                      color=BLUE, transform=logic_ax.transAxes)
        logic_ax.text(0.05, y - 0.075, denominator, fontsize=12,
                      transform=logic_ax.transAxes)
        logic_ax.annotate(
            "", xy=(0.08, y - 0.16), xytext=(0.08, y - 0.09),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": GRAY},
        )
        logic_ax.text(0.14, y - 0.18, metric, fontsize=10.5,
                      color="#17334b", transform=logic_ax.transAxes)
    fig.suptitle("The confusion matrix is the source of the metrics", fontsize=16)
    fig.text(0.02, 0.02, "Synthetic validation data; positive class = 1; threshold = 0.50.",
             color=GRAY, fontsize=9)
    save_figure(fig, output_dir, "confusion_matrix_foundation.png")


def imbalance_case():
    """Construct transparent synthetic predictions for a rare-event example."""
    y = np.r_[np.zeros(990, dtype=int), np.ones(10, dtype=int)]
    baseline = np.zeros_like(y)
    detector = np.zeros_like(y)
    detector[:12] = 1
    detector[990:997] = 1
    return (
        classification_metrics(y, baseline),
        classification_metrics(y, detector),
    )


def plot_accuracy_imbalance(output_dir):
    baseline, detector = imbalance_case()
    fig = plt.figure(figsize=(13, 5.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
    ax_a, ax_b, ax_rates = [fig.add_subplot(grid[0, i]) for i in range(3)]
    draw_matrix(ax_a, baseline, compact=True)
    draw_matrix(ax_b, detector, compact=True)
    ax_a.set_title("A: always negative", pad=20)
    ax_b.set_title("B: illustrative detector", pad=20)
    names = ["accuracy", "precision", "recall", "f1"]
    positions = np.arange(4)
    ax_rates.barh(positions - 0.18, [baseline[k] for k in names], height=0.34,
                  color=GRAY, label="A")
    ax_rates.barh(positions + 0.18, [detector[k] for k in names], height=0.34,
                  color=GREEN, label="B")
    ax_rates.set_yticks(positions, ["Accuracy", "Precision", "Recall", "F1"])
    ax_rates.set_xlim(0, 1.05)
    ax_rates.invert_yaxis()
    ax_rates.set_xlabel("Metric value")
    ax_rates.legend(frameon=False)
    ax_rates.set_title("Same 1,000 labels")
    fig.suptitle("High accuracy can coexist with zero positive detection",
                 fontsize=16)
    fig.text(
        0.02, 0.01,
        "Synthetic construction: 990 negatives, 10 positives. B flags 12 negatives "
        "and 7 positives; metrics are calculated from predictions.",
        fontsize=9, color=GRAY,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_figure(fig, output_dir, "accuracy_imbalance.png")
    return baseline, detector


def create_threshold_animation(y_true, probabilities, output_dir):
    thresholds = np.r_[np.linspace(0.05, 0.95, 23), np.repeat(0.95, 3)]
    rng = np.random.default_rng(SEED)
    jitter = rng.uniform(-0.11, 0.11, len(y_true))
    fig = plt.figure(figsize=(10, 6))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], hspace=0.35, top=0.80)
    scores_ax = fig.add_subplot(grid[:, 0])
    matrix_ax = fig.add_subplot(grid[0, 1])
    metrics_ax = fig.add_subplot(grid[1, 1])

    def draw(threshold):
        counts = metrics_at_threshold(y_true, probabilities, float(threshold))
        scores_ax.clear()
        scores_ax.axvspan(threshold, 1, color="#e8f5ec", alpha=0.85)
        scores_ax.scatter(
            probabilities[y_true == 0], jitter[y_true == 0],
            s=20, c=BLUE, alpha=0.65, label="Actual negative",
        )
        scores_ax.scatter(
            probabilities[y_true == 1], 1 + jitter[y_true == 1],
            s=26, c=ORANGE, alpha=0.8, label="Actual positive",
        )
        scores_ax.axvline(threshold, color="#1c2630", lw=2)
        scores_ax.set(xlim=(0, 1), ylim=(-0.3, 1.3),
                      xlabel="Predicted probability of class 1",
                      title="Right of the line = predicted positive")
        scores_ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
        scores_ax.legend(loc="upper center", frameon=False, ncol=2, fontsize=8)
        matrix_ax.clear()
        draw_matrix(matrix_ax, counts, compact=True)
        metrics_ax.clear()
        metrics_ax.axis("off")
        labels = [
            f"Threshold              {threshold:.2f}",
            f"Predicted positives   {counts['predicted_positives']}",
            f"Accuracy               {counts['accuracy']:.3f}",
            f"Precision              {counts['precision']:.3f}",
            f"Recall                 {counts['recall']:.3f}",
            f"F1                     {counts['f1']:.3f}",
            f"FP / FN                {counts['fp']} / {counts['fn']}",
        ]
        metrics_ax.text(0.03, 0.95, "\n".join(labels), va="top", family="monospace",
                        fontsize=10.5, linespacing=1.55)
        fig.suptitle("One fitted model, changing decision threshold", fontsize=15)
        return []

    animation = FuncAnimation(
        fig, draw, frames=thresholds, interval=350, repeat=True, blit=False
    )
    animation.save(output_dir / "threshold_tradeoff.gif", writer=PillowWriter(fps=3),
                   dpi=90)
    plt.close(fig)


def select_thresholds(rows):
    """Select on validation rows only; break metric ties toward lower t."""
    if not rows:
        raise ValueError("at least one validation threshold is required")
    best_f1 = max(rows, key=lambda row: (row["f1"], -row["threshold"]))
    recall_90 = [row for row in rows if row["recall"] >= 0.90]
    precision_90 = [row for row in rows if row["precision"] >= 0.90]
    return {
        "best_f1": best_f1,
        "recall_90": max(
            recall_90, key=lambda row: (row["precision"], -row["threshold"])
        ) if recall_90 else None,
        "precision_90": max(
            precision_90, key=lambda row: (row["recall"], -row["threshold"])
        ) if precision_90 else None,
    }


def plot_metrics_vs_threshold(rows, selected, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = [row["threshold"] for row in rows]
    for name in COLORS:
        ax.plot(x, [row[name] for row in rows], lw=2, color=COLORS[name],
                label=name.capitalize())
    ax.axvline(0.5, color=GRAY, linestyle="--", label="Default 0.50")
    best = selected["best_f1"]
    ax.scatter([best["threshold"]], [best["f1"]], c=PURPLE, s=70, zorder=5)
    ax.annotate(f"Best validation F1\nat t={best['threshold']:.2f}",
                (best["threshold"], best["f1"]), xytext=(12, -34),
                textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": PURPLE})
    ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Classification threshold",
           ylabel="Metric value",
           title="Validation metrics from one fixed probability model")
    ax.legend(ncol=2, frameon=False)
    ax.grid(alpha=0.2)
    fig.text(0.02, 0.01,
             "Synthetic validation data. The marked threshold maximizes F1 on "
             "this grid only; it is not a universal policy.",
             color=GRAY, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, output_dir, "metrics_vs_threshold.png")


def plot_precision_recall_denominators(counts, output_dir):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    specs = [
        ("Precision: among predicted positives", "tp", "fp", "TP", "FP",
         counts["precision"], ORANGE),
        ("Recall: among actual positives", "tp", "fn", "TP", "FN",
         counts["recall"], GREEN),
    ]
    max_total = max(counts["tp"] + counts["fp"], counts["tp"] + counts["fn"])
    for ax, (title, good, other, good_label, other_label, rate, color) in zip(
        axes, specs
    ):
        ax.barh(0, counts[good], color=color, height=0.6, label=good_label)
        ax.barh(0, counts[other], left=counts[good], color="#d7dfe4",
                height=0.6, label=other_label)
        ax.set_xlim(0, max_total * 1.12)
        ax.set_ylim(-0.7, 0.7)
        ax.set_yticks([])
        ax.set_title(title, loc="left")
        ax.text(0, -0.48,
                f"{good_label}={counts[good]}  |  {other_label}={counts[other]}"
                f"  |  {good_label}/({good_label}+{other_label}) = {rate:.3f}",
                fontsize=10)
        ax.legend(frameon=False, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Number of validation examples in the denominator")
    fig.suptitle("Same TP count; different populations answer different questions",
                 fontsize=15)
    fig.tight_layout()
    save_figure(fig, output_dir, "precision_recall_denominators.png")


def plot_f1_harmonic_mean(output_dir):
    pairs = np.array([(1, .1), (.9, .9), (.5, .5), (.95, .4), (.4, .95)])
    arithmetic = pairs.mean(axis=1)
    f1 = harmonic_mean(pairs[:, 0], pairs[:, 1])
    fig, (bars_ax, heat_ax) = plt.subplots(
        1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1, 1.05]}
    )
    positions = np.arange(len(pairs))
    bars_ax.barh(positions - 0.18, arithmetic, height=0.34, color=GRAY,
                 label="Arithmetic mean")
    bars_ax.barh(positions + 0.18, f1, height=0.34, color=PURPLE,
                 label="Harmonic mean / F1")
    bars_ax.set_yticks(positions, [f"P={p:.2f}, R={r:.2f}" for p, r in pairs])
    bars_ax.invert_yaxis()
    bars_ax.set_xlim(0, 1)
    bars_ax.set_xlabel("Mean value")
    bars_ax.set_title("Uneven pairs receive lower F1")
    bars_ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2)
    grid = np.linspace(0, 1, 151)
    p, r = np.meshgrid(grid, grid)
    field = harmonic_mean(p, r)
    image = heat_ax.imshow(field, origin="lower", extent=[0, 1, 0, 1],
                           vmin=0, vmax=1, cmap="viridis", aspect="equal")
    heat_ax.plot(pairs[:, 0], pairs[:, 1], "o", color="white",
                 mec="#1d2730", ms=7)
    heat_ax.set(xlabel="Precision", ylabel="Recall", title="F1 across all pairs")
    fig.colorbar(image, ax=heat_ax, label="F1", shrink=0.85)
    fig.suptitle("F1 balances precision and recall; TN is absent from its formula", fontsize=15)
    fig.tight_layout()
    save_figure(fig, output_dir, "f1_harmonic_mean.png")


def create_f1_surface_3d(output_dir):
    grid = np.linspace(0, 1, 65)
    p, r = np.meshgrid(grid, grid)
    z = harmonic_mean(p, r)
    fig = go.Figure(
        data=[go.Surface(
            x=p, y=r, z=z, colorscale="Viridis", cmin=0, cmax=1,
            colorbar={"title": "F1"},
            hovertemplate="Precision %{x:.2f}<br>Recall %{y:.2f}<br>F1 %{z:.2f}<extra></extra>",
        )]
    )
    fig.update_layout(
        title="F1 = 2PR / (P + R) (zero at P = R = 0)",
        scene={"xaxis_title": "Precision", "yaxis_title": "Recall",
               "zaxis_title": "F1", "zaxis_range": [0, 1]},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    fig.write_html(output_dir / "f1_surface_3d.html",
                   include_plotlyjs=True, full_html=True)


def plot_prevalence_sensitivity(output_dir):
    prevalence = np.linspace(0.01, 0.50, 100)
    tpr, fpr = 0.80, 0.05
    rates = rates_at_prevalence(prevalence, tpr, fpr)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in COLORS:
        ax.plot(prevalence * 100, rates[name], lw=3 if name == "precision" else 1.8,
                color=COLORS[name], label=name.capitalize())
    ax.set(xlim=(1, 50), ylim=(0, 1), xlabel="Positive-class prevalence (%)",
           ylabel="Expected metric value",
           title="Precision changes when prevalence changes")
    ax.legend(frameon=False, ncol=4)
    ax.grid(alpha=0.2)
    fig.text(0.02, 0.01,
             "Controlled expectation: TPR=0.80 and FPR=0.05 fixed. "
             "Real conditional rates may shift too.",
             fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, output_dir, "prevalence_metric_sensitivity.png")
    return rates


def plot_operational_tradeoff(rows, output_dir, capacity=100):
    x = np.array([row["threshold"] for row in rows])
    positives = np.array([row["predicted_positives"] for row in rows])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, [row["fp"] for row in rows], color=ORANGE, lw=2, label="False positives")
    ax.plot(x, [row["fn"] for row in rows], color=PURPLE, lw=2, label="False negatives")
    ax.plot(x, positives, color=BLUE, lw=2, label="Predicted positives / alerts")
    ax.axhline(capacity, color=GRAY, linestyle="--",
               label=f"Illustrative review capacity: {capacity} alerts")
    ax.fill_between(x, 0, positives.max() * 1.05, where=positives > capacity,
                    color=ORANGE, alpha=0.07, label="Capacity exceeded")
    ax.set(xlim=(0, 1), ylim=(0, positives.max() * 1.05),
           xlabel="Classification threshold", ylabel="Validation example count",
           title="Threshold changes misses and review workload")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.grid(alpha=0.2)
    fig.text(0.02, 0.01,
             "Synthetic validation set. Capacity is illustrative, not a measured "
             "staffing limit.",
             fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, output_dir, "error_tradeoff_operational.png")


def verify_library_parity(y_true, probabilities):
    counts = metrics_at_threshold(y_true, probabilities, 0.5)
    predicted = (probabilities >= 0.5).astype(int)
    assert counts["tn"] + counts["fp"] + counts["fn"] + counts["tp"] == len(y_true)
    comparisons = {
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
    }
    if not all(np.isclose(counts[key], value) for key, value in comparisons.items()):
        raise AssertionError("Manual metrics differ from scikit-learn")


def print_counts(name, counts):
    print(
        f"{name}: TP={counts['tp']} TN={counts['tn']} FP={counts['fp']} "
        f"FN={counts['fn']} alerts={counts['predicted_positives']} "
        f"accuracy={counts['accuracy']:.4f} precision={counts['precision']:.4f} "
        f"recall={counts['recall']:.4f} F1={counts['f1']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "assets",
        help="Directory for generated assets (default: topic assets/).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = make_lab_data()
    y_val, p_val = data["val_y"], data["val_p"]
    verify_library_parity(y_val, p_val)
    fixed = metrics_at_threshold(y_val, p_val, 0.5)
    rows = threshold_table(y_val, p_val, np.linspace(0.01, 0.99, 99))
    selected = select_thresholds(rows)

    plot_confusion_matrix_foundation(fixed, args.output_dir)
    imbalance_a, imbalance_b = plot_accuracy_imbalance(args.output_dir)
    create_threshold_animation(y_val, p_val, args.output_dir)
    plot_metrics_vs_threshold(rows, selected, args.output_dir)
    plot_precision_recall_denominators(fixed, args.output_dir)
    plot_f1_harmonic_mean(args.output_dir)
    create_f1_surface_3d(args.output_dir)
    plot_prevalence_sensitivity(args.output_dir)
    plot_operational_tradeoff(rows, args.output_dir)

    print("Classification Metrics Visual Lab")
    print("Synthetic data: seed=22, 2,400 rows, 10 features, 85/15 class weights.")
    print(f"Train={data['train_n']}; validation={len(y_val)} "
          f"({int(y_val.sum())} positives); test={len(data['test_y'])} "
          f"({int(data['test_y'].sum())} positives).")
    print_counts("Validation t=0.50", fixed)
    print(
        f"Imbalance construction (990 negatives, 10 positives): "
        f"always-negative accuracy={imbalance_a['accuracy']:.4f}, "
        f"recall={imbalance_a['recall']:.4f}; detector accuracy="
        f"{imbalance_b['accuracy']:.4f}, recall={imbalance_b['recall']:.4f}."
    )
    best = selected["best_f1"]
    print(f"Best F1 on validation grid: t={best['threshold']:.2f}, F1={best['f1']:.4f}")
    for key, label in (
        ("recall_90", "Recall >= 0.90"),
        ("precision_90", "Precision >= 0.90"),
    ):
        option = selected[key]
        print(f"{label}: " + (
            f"t={option['threshold']:.2f}, precision={option['precision']:.4f}, "
            f"recall={option['recall']:.4f}" if option else "no grid threshold satisfied"
        ))
    print_counts(
        "Test t=0.50", metrics_at_threshold(data["test_y"], data["test_p"], 0.5)
    )
    print_counts(
        "Test selected validation-F1 threshold",
        metrics_at_threshold(data["test_y"], data["test_p"], best["threshold"]),
    )
    print("Generated: " + ", ".join(OUTPUT_NAMES))
    print("Interpretations require author review; this is one synthetic split.")


if __name__ == "__main__":
    main()
