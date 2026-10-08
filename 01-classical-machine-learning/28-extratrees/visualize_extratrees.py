"""Day 28 visual lab: random split proposals, diversity, and averaging.

Run from the repository root:
    python 01-classical-machine-learning/28-extratrees/visualize_extratrees.py

All data are synthetic. Outputs are regenerable, ignored local artifacts.
The 1D best-so-far animation is a teaching pool, not a literal ExtraTree trace:
an actual ExtraTree proposes one threshold per candidate feature at a node.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np
import sklearn
from sklearn.datasets import make_classification, make_moons
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


SEED = 42
COLORS = ["#2574ab", "#db6b35"]
CLASS_MAP = ListedColormap(["#c7dfef", "#f5d6c6"])
FAMILIES = {"RF": RandomForestClassifier, "ET": ExtraTreesClassifier}
COMMON = dict(criterion="gini", max_depth=5, min_samples_leaf=3,
              max_features=1, n_jobs=1, random_state=SEED)
TABULAR_CONFIG = dict(n_samples=1000, n_features=10, n_informative=4,
                      n_redundant=2, n_repeated=0, n_classes=2,
                      n_clusters_per_class=2, weights=None, class_sep=1.0,
                      flip_y=0.05, shuffle=True, random_state=SEED)
TREE_COUNTS = [1, 2, 5, 10, 20, 50, 100, 200, 300]


def binary_labels(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1 or y.size == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected nonempty 1D binary labels (0/1).")
    return y.astype(float)


def gini_impurity(y) -> float:
    p = binary_labels(y).mean()
    return float(2 * p * (1 - p))


def split_gain(x, y, threshold: float) -> float:
    x = np.asarray(x, dtype=float)
    y = binary_labels(y)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(threshold):
        raise ValueError("x and y must be matching finite 1D arrays; threshold must be finite.")
    left = x <= threshold
    if left.all() or not left.any():
        return float("nan")  # Empty-child proposals are invalid, not zero-gain splits.
    weight = left.mean()
    return gini_impurity(y) - (
        weight * gini_impurity(y[left]) + (1 - weight) * gini_impurity(y[~left])
    )


def threshold_gains(x, y, thresholds) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not len(thresholds):
        raise ValueError("Thresholds must be a nonempty 1D array.")
    return np.array([split_gain(x, y, t) for t in thresholds])


def ensemble_variance(n_trees, rho, sigma_squared=1.0):
    """Equal member variance and equal pairwise correlation; scalar predictions."""
    n_trees, rho = np.broadcast_arrays(
        np.asarray(n_trees, dtype=float), np.asarray(rho, dtype=float)
    )
    if (not np.isfinite(n_trees).all() or np.any(n_trees < 1)
            or np.any(n_trees != np.floor(n_trees))
            or not np.isfinite(rho).all() or np.any((rho < 0) | (rho > 1))
            or not np.isfinite(sigma_squared) or sigma_squared < 0):
        raise ValueError("Use integer M >= 1, rho in [0, 1], and finite sigma_squared >= 0.")
    return sigma_squared * (rho + (1 - rho) / n_trees)


def tree_diversity(predictions) -> dict:
    """Hard-label disagreement and Pearson correlation across evaluation rows.

    Constant tree predictions have undefined Pearson correlations. Keep NaNs
    instead of imputing similarity, and count the defined off-diagonal pairs.
    These correlations are not correlations over repeated training datasets.
    """
    predictions = np.asarray(predictions, dtype=float)
    if (predictions.ndim != 2 or predictions.shape[0] < 2
            or predictions.shape[1] < 2 or not np.isin(predictions, [0, 1]).all()):
        raise ValueError("Expected a binary matrix with at least two trees and two rows.")
    counts = predictions.sum(axis=1)
    disagreement = (counts[:, None] + counts[None, :]
                    - 2 * predictions @ predictions.T) / predictions.shape[1]
    centered = predictions - predictions.mean(axis=1, keepdims=True)
    lengths = np.linalg.norm(centered, axis=1)
    denominator = lengths[:, None] * lengths[None, :]
    correlation = np.full_like(denominator, np.nan)
    np.divide(centered @ centered.T, denominator, out=correlation, where=denominator > 0)
    correlation = np.clip(correlation, -1, 1)
    upper = np.triu_indices(len(predictions), k=1)
    defined = correlation[upper]
    defined = defined[np.isfinite(defined)]
    return dict(disagreement_matrix=disagreement, correlation_matrix=correlation,
                mean_disagreement=float(disagreement[upper].mean()),
                mean_correlation=float(defined.mean()) if len(defined) else None,
                defined_correlation_pairs=len(defined), total_pairs=len(upper[0]))


def cumulative_probabilities(probabilities, counts) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    counts = np.asarray(counts)
    if (probabilities.ndim != 2 or min(probabilities.shape) == 0
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or counts.ndim != 1 or not len(counts)
            or counts.dtype.kind not in "iu" or np.any(counts < 1)
            or np.any(counts > len(probabilities))):
        raise ValueError("Use a finite probability matrix and integer prefix counts within its tree count.")
    return probabilities.cumsum(axis=0)[counts - 1] / counts[:, None]


def make_datasets() -> dict:
    rng = np.random.default_rng(SEED)
    x = np.sort(rng.uniform(-2.5, 2.5, size=60))
    y = (x > 0.15).astype(int)
    y[rng.choice(len(y), size=5, replace=False)] ^= 1
    moon_X, moon_y = make_moons(n_samples=400, noise=0.23, random_state=SEED)
    tab_X, tab_y = make_classification(**TABULAR_CONFIG)
    splits = {}
    for name, X, labels in [("moons", moon_X, moon_y), ("tabular", tab_X, tab_y)]:
        splits[name] = train_test_split(X, labels, test_size=0.30,
                                      stratify=labels, random_state=SEED)
    return dict(split=(x, y), **splits)


def make_model(family, bootstrap=False, **overrides):
    config = dict(n_estimators=80) | COMMON | overrides
    return FAMILIES[family](bootstrap=bootstrap, **config)


def evaluate(model, X, y) -> dict:
    probability = model.predict_proba(X)[:, 1]
    predictions = np.stack([tree.predict(X) for tree in model.estimators_])
    diversity = tree_diversity(predictions)
    return dict(roc_auc=float(roc_auc_score(y, probability)),
                accuracy=float(accuracy_score(y, model.predict(X))),
                log_loss=float(log_loss(y, probability, labels=[0, 1])),
                mean_tree_accuracy=float(np.mean(predictions == y)),
                disagreement=diversity["mean_disagreement"],
                correlation=diversity["mean_correlation"],
                defined_correlation_pairs=diversity["defined_correlation_pairs"],
                total_pairs=diversity["total_pairs"])


def save_figure(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path.name


def print_view(number, explanation):
    print(f"\n[{number}] {explanation}", flush=True)


def plot_split_search(data, output, dpi):
    x, y = data
    unique = np.unique(x)
    greedy = (unique[:-1] + unique[1:]) / 2
    proposals = np.random.default_rng(SEED + 1).uniform(x.min(), x.max(), size=8)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, layout="constrained")
    records = {}
    for col, (label, candidates) in enumerate([
            ("Greedy search: all observed-value midpoints", greedy),
            ("Teaching pool: 8 uniform random proposals", proposals)]):
        gains = threshold_gains(x, y, candidates)
        best = int(np.nanargmax(gains))
        axes[0, col].scatter(x, y, c=y, cmap=ListedColormap(COLORS), s=26, zorder=3)
        axes[0, col].vlines(candidates, -0.15, 1.15, color="0.7", lw=0.8, alpha=0.7)
        axes[0, col].axvline(candidates[best], color="#28743c", lw=2, label="Selected candidate")
        axes[0, col].set(title=label, ylabel="Class", yticks=[0, 1], ylim=(-0.2, 1.2))
        axes[0, col].legend(fontsize=8)
        axes[1, col].scatter(candidates, gains, color=COLORS[col], s=24)
        axes[1, col].scatter(candidates[best], gains[best], marker="*", s=180, color="#28743c")
        axes[1, col].set(xlabel="Threshold position", ylabel="Gini reduction", ylim=(-0.02, 0.55))
        records[label] = dict(candidate_count=len(candidates),
                              selected_threshold=float(candidates[best]),
                              selected_gain=float(gains[best]))
    fig.suptitle("Search many candidates vs evaluate a small random proposal pool\n"
                 "Actual ExtraTrees: one proposal per candidate feature at each node", fontsize=12)
    print_view("01", f"Greedy gain={records[next(iter(records))]['selected_gain']:.4f}; "
               f"random-pool gain={records[list(records)[1]]['selected_gain']:.4f}. "
               "Both select by impurity; the 1D pool is an explicit teaching simplification.")
    return save_figure(fig, output / "01_split_search_vs_random.png", dpi), records


def animate_random_thresholds(data, output, dpi):
    x, y = data
    candidates = np.random.default_rng(SEED + 2).uniform(x.min(), x.max(), size=24)
    gains = threshold_gains(x, y, candidates)
    best_indices = np.array([np.argmax(gains[:i + 1]) for i in range(len(gains))])
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), layout="constrained")

    def frame(i):
        for ax in axes:
            ax.clear()
        best = best_indices[i]
        axes[0].scatter(x, y, c=y, cmap=ListedColormap(COLORS), s=24)
        axes[0].axvline(candidates[i], color="#a43ba6", lw=2, label="Current random proposal")
        axes[0].axvline(candidates[best], color="#28743c", lw=2, ls="--", label="Best in teaching pool")
        axes[0].set(xlim=(x.min() - 0.15, x.max() + 0.15),
                    ylim=(-0.2, 1.2), yticks=[0, 1], ylabel="Class")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[1].scatter(candidates[:i + 1], gains[:i + 1], color="0.55", s=25)
        axes[1].scatter(candidates[i], gains[i], color="#a43ba6", s=65)
        axes[1].scatter(candidates[best], gains[best], color="#28743c", marker="*", s=150)
        axes[1].set(xlabel="Threshold position", ylabel="Gini reduction",
                    xlim=(x.min() - 0.15, x.max() + 0.15), ylim=(-0.02, 0.55))
        previous = "first proposal" if i == 0 else (
            "better than previous" if gains[i] > gains[i - 1] else "no better than previous")
        update = "updated" if best == i else "kept"
        fig.suptitle(f"Proposal {i + 1}/24: t={candidates[i]:+.2f}, weighted Gini="
                     f"{gini_impurity(y) - gains[i]:.3f}, gain={gains[i]:.3f}\n"
                     f"{previous}; best {update}: t={candidates[best]:+.2f}, gain={gains[best]:.3f}\n"
                     "Teaching pool, not literal 1D ExtraTree fitting", fontsize=10)
        return []

    animation = FuncAnimation(fig, frame, frames=len(candidates), interval=350, blit=False)
    path = output / "02_random_threshold_search.gif"
    animation.save(path, writer=PillowWriter(fps=3), dpi=min(dpi, 110))
    plt.close(fig)
    print_view("02", "24 proposals on unchanged data; show current vs previous and best-so-far. "
               "Accumulating proposals explains selection, not the library's per-node draw count.")
    return path.name, dict(candidate_count=24, best_gain=float(gains.max()))


def evaluation_grid(X, resolution=110):
    low, high = X.min(axis=0) - 0.5, X.max(axis=0) + 0.5
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], resolution),
                         np.linspace(low[1], high[1], resolution))
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def boundary_panel(ax, probability, grid, split, title, legend=False):
    xx, yy, _ = grid
    X_train, X_test, y_train, y_test = split
    ax.contourf(xx, yy, (probability.reshape(xx.shape) > 0.5).astype(int),
                levels=[-0.5, 0.5, 1.5], cmap=CLASS_MAP)
    if probability.min() < 0.5 < probability.max():
        ax.contour(xx, yy, probability.reshape(xx.shape), levels=[0.5], colors="0.3", linewidths=0.8)
    for cls, color in enumerate(COLORS):
        ax.scatter(*X_train[y_train == cls].T, c=color, s=12, alpha=0.65,
                   label=f"Train class {cls}")
        ax.scatter(*X_test[y_test == cls].T, c=color, s=26, marker="^",
                   edgecolors="black", linewidths=0.4, label=f"Validation class {cls}")
    ax.set(title=title, xlabel="Synthetic feature 1", ylabel="Synthetic feature 2",
           xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()))
    if legend:
        ax.legend(fontsize=7, loc="upper right")


def plot_decision_boundaries(split, output, dpi):
    X_train, X_test, y_train, y_test = split
    grid = evaluation_grid(np.vstack([X_train, X_test]))
    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=3, max_features=None, random_state=SEED),
        "RF": make_model("RF", bootstrap=False),
        "ET": make_model("ET", bootstrap=False),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X_train, y_train)
        config = "1 tree, all features" if name == "Decision Tree" else "80 trees, max_features=1, bootstrap=False"
        boundary_panel(ax, model.predict_proba(grid[2])[:, 1], grid, split,
                       f"{name}\n{config}", legend=name == "Decision Tree")
    fig.suptitle("Same synthetic moons and split: partition geometry, not a universal ranking\n"
                 "All models: max_depth=5, min_samples_leaf=3, seed=42", fontsize=12)
    paths = [save_figure(fig, output / "03_decision_boundaries.png", dpi)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for i, ax in enumerate(axes.flat):
        tree = models["ET"].estimators_[i]
        boundary_panel(ax, tree.predict_proba(grid[2])[:, 1], grid, split,
                       f"ExtraTree {i + 1}", legend=i == 0)
    fig.suptitle("Six individual ExtraTrees: identical training rows, different random split draws", fontsize=12)
    paths.append(save_figure(fig, output / "04_individual_extratrees.png", dpi))
    print_view("03-04", "Compare the final partitions and six component trees. "
               "RF/ET use bootstrap=False here, isolating threshold strategy at matched controls.")
    return paths, models, grid


def plot_tree_correlation(models, split, output, dpi):
    _, X_test, _, y_test = split
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    records = {}
    for col, family in enumerate(FAMILIES):
        model = models[family]
        predictions = np.stack([tree.predict(X_test) for tree in model.estimators_])
        diversity = tree_diversity(predictions)
        records[family] = evaluate(model, X_test, y_test)
        for row, (key, label, limits, cmap) in enumerate([
                ("disagreement_matrix", "Hard-label disagreement", (0, 1), "magma"),
                ("correlation_matrix", "Hard-label Pearson correlation", (-1, 1), "coolwarm")]):
            ax = axes[row, col]
            artist = ax.imshow(diversity[key], vmin=limits[0], vmax=limits[1], cmap=cmap)
            ax.set(title=f"{family}: {label}", xlabel="Tree index", ylabel="Tree index")
            fig.colorbar(artist, ax=ax, shrink=0.8)
    fig.suptitle("80 trees; same 120 validation rows; bootstrap=False in both families\n"
                 "Correlation across rows is not repeated-training variance; undefined cells are blank", fontsize=11)
    print_view("05", "Heatmaps include all tree pairs; numerical means exclude diagonals. "
               "Compare disagreement alongside tree accuracy; neither statistic alone proves reduced error variance.")
    return save_figure(fig, output / "05_tree_correlation_heatmaps.png", dpi), records


def plot_variance_surface(output, dpi, interactive=True):
    M, rho = np.meshgrid(np.unique(np.rint(np.geomspace(1, 500, 80)).astype(int)),
                         np.linspace(0, 1, 60))
    variance = ensemble_variance(M, rho)
    fig = plt.figure(figsize=(12, 5), layout="constrained")
    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(M, rho, variance, cmap="viridis", alpha=0.9)
    ax.set(xlabel="Number of trees M", ylabel="Correlation rho")
    # A 2D axis label avoids clipped 3D z-labels in static exports.
    ax.text2D(0.02, 0.55, "Ensemble variance", transform=ax.transAxes,
              rotation=90, ha="left", va="center")
    ax.view_init(elev=24, azim=-125)
    curves = fig.add_subplot(122)
    for value in [0, 0.1, 0.5, 0.9]:
        count = np.arange(1, 501)
        curves.plot(count, ensemble_variance(count, value), label=f"rho={value}")
        curves.axhline(value, ls=":", alpha=0.35)
    curves.set(xlabel="Number of trees M", ylabel="Ensemble variance", xscale="log",
               title="Dotted lines: limiting variance floors")
    curves.legend()
    fig.suptitle("Analytical model: sigma squared = 1; equal member variance and pairwise correlation\n"
                 "Var(mean) = rho + (1-rho)/M | This is not fitted experimental variance", fontsize=11)
    paths = [save_figure(fig, output / "06_ensemble_variance_surface.png", dpi)]
    if interactive:
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly unavailable: static surface saved; interactive HTML skipped.")
        else:
            figure = go.Figure(go.Surface(x=M, y=rho, z=variance, colorscale="Viridis",
                                         colorbar=dict(title="Variance")))
            figure.update_layout(
                title="Analytical variance, not measured: equal variance=1 and equal pairwise rho",
                scene=dict(xaxis_title="Number of trees M", yaxis_title="Correlation rho",
                           zaxis_title="Ensemble variance"), width=1000, height=700)
            path = output / "06_ensemble_variance_surface.html"
            figure.write_html(str(path), include_plotlyjs=True, full_html=True)
            paths.append(path.name)
    print_view("06", "Analytical assumptions, not synthetic measurements: increasing M approaches rho. "
               "Do not insert cross-row heatmap correlations as estimates of this variance model.")
    return paths

def experiment_n_estimators(split, output, dpi):
    X_train, X_test, y_train, y_test = split
    seeds = [SEED + i for i in range(5)]
    records, base_models = {}, {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for family, color in zip(FAMILIES, COLORS):
        runs, prefix_predictions = [], []
        for seed in seeds:
            model = make_model(family, n_estimators=max(TREE_COUNTS), random_state=seed)
            model.fit(X_train, y_train)
            per_tree = np.stack([tree.predict_proba(X_test)[:, 1] for tree in model.estimators_])
            prefixes = cumulative_probabilities(per_tree, TREE_COUNTS)
            prefix_predictions.append(prefixes)
            runs.append([dict(n_estimators=n, roc_auc=float(roc_auc_score(y_test, probability)))
                         for n, probability in zip(TREE_COUNTS, prefixes)])
            if seed == SEED:
                base_models[family] = model
        auc = np.array([[row["roc_auc"] for row in run] for run in runs])
        # Same train/validation rows: variation here is algorithmic seed variation only.
        variability = np.std(np.stack(prefix_predictions), axis=0, ddof=1).mean(axis=1)
        for run in auc:
            axes[0].plot(TREE_COUNTS, run, color=color, alpha=0.18, lw=1)
        axes[0].plot(TREE_COUNTS, auc.mean(axis=0), "o-", color=color, label=f"{family}: mean")
        axes[1].plot(TREE_COUNTS, variability, "o-", color=color, label=family)
        records[family] = dict(seeds=seeds, per_seed_auc=runs,
                               mean_auc=auc.mean(axis=0).tolist(),
                               mean_prediction_sd=variability.tolist())
    axes[0].set(xlabel="Number of trees (nested prefixes)", ylabel="Validation ROC-AUC",
                xscale="log", title="Faint lines: each seed; bold: five-seed mean")
    axes[1].set(xlabel="Number of trees (nested prefixes)", ylabel="Mean SD of P(class=1) across seeds",
                xscale="log", title="Variation at fixed validation inputs")
    for ax in axes:
        ax.legend()
    fig.suptitle("Adding trees: actual measurements, no smoothing or monotonicity imposed\n"
                 "Same synthetic moons split; bootstrap=False; depth=5; max_features=1", fontsize=11)
    print_view("07", "Nested prefixes of 300-tree fits across five seeds. "
               "Prediction SD studies algorithmic randomness with training data fixed, not sampling variance.")
    for family, record in records.items():
        print(f"  {family}: mean AUC {record['mean_auc'][0]:.4f} -> {record['mean_auc'][-1]:.4f}; "
              f"mean probability SD {record['mean_prediction_sd'][0]:.4f} -> "
              f"{record['mean_prediction_sd'][-1]:.4f} (1 -> 300 trees).")
    return save_figure(fig, output / "07_n_estimators_stability.png", dpi), records, base_models


def experiment_max_features(split, output, dpi):
    X_train, X_test, y_train, y_test = split
    settings = [(1, "1"), (2, "2"), ("sqrt", "sqrt=3"), (0.5, "0.5=5"), (None, "None=10")]
    records = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for family_index, (family, color, marker) in enumerate([
            ("RF", COLORS[0], "o"), ("ET", COLORS[1], "^")]):
        for index, (value, label) in enumerate(settings):
            model = make_model(family, max_features=value, max_depth=6)
            model.fit(X_train, y_train)
            result = dict(family=family, max_features=value, label=label,
                          **evaluate(model, X_test, y_test))
            records.append(result)
            for ax, metric in zip(axes[family_index], ["roc_auc", "mean_tree_accuracy"]):
                ax.scatter(result["disagreement"], result[metric], color=color, marker=marker, s=55)
                offset = (7, 12 if index % 2 == 0 else -18)
                ax.annotate(label, (result["disagreement"], result[metric]), xytext=offset,
                            textcoords="offset points", fontsize=9, color=color,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
    for row, family in enumerate(FAMILIES):
        axes[row, 0].set(ylabel="Validation ROC-AUC", title=f"{family}: ensemble ranking quality")
        axes[row, 1].set(ylabel="Mean individual-tree accuracy", title=f"{family}: member strength")
        for column, metric in enumerate(["roc_auc", "mean_tree_accuracy"]):
            ax = axes[row, column]
            values = [record[metric] for record in records]
            padding = max((max(values) - min(values)) * 0.2, 0.02)
            ax.set(xlabel="Mean pairwise hard-label disagreement",
                   xlim=(-0.035, 0.46), ylim=(min(values) - padding, max(values) + padding))
    fig.suptitle("max_features on 10-feature synthetic data: one seed, untuned configurations\n"
                 "80 trees; bootstrap=False; depth=6; leaf minimum=3 | None means all 10 features", fontsize=11)
    print_view("08", "Each point is a measured feature-count setting, not a tuned optimum. "
               "The second panel checks individual strength. Cross-row correlations are also recorded.")
    print("  family max_features    AUC    tree_acc  disagree  correlation")
    for row in records:
        correlation = "undefined" if row["correlation"] is None else f"{row['correlation']:.4f}"
        print(f"  {row['family']:<6} {row['label']:<12} {row['roc_auc']:.4f} "
              f"{row['mean_tree_accuracy']:.4f}   {row['disagreement']:.4f}    {correlation}")
    return save_figure(fig, output / "08_diversity_vs_performance.png", dpi), records


def experiment_bootstrap(split, output, dpi):
    X_train, X_test, y_train, y_test = split
    records = []
    for family in FAMILIES:
        for bootstrap in [False, True]:
            model = make_model(family, bootstrap=bootstrap, max_features="sqrt", max_depth=6)
            start = perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = perf_counter() - start
            records.append(dict(family=family, bootstrap=bootstrap, fit_seconds=fit_seconds,
                                **evaluate(model, X_test, y_test)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    labels = [f"{row['family']} | bootstrap={row['bootstrap']}" for row in records]
    for ax, metric, title in zip(axes.flat,
            ["roc_auc", "log_loss", "fit_seconds", "disagreement"],
            ["ROC-AUC (higher better)", "Log loss (lower better)",
             "Fit time: one run, seconds", "Disagreement (no preferred direction)"]):
        bars = ax.barh(labels, [row[metric] for row in records],
                       color=[COLORS[0], "#81acd0", COLORS[1], "#e7a384"])
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set(title=title, xlabel=metric.replace("_", " "))
        ax.set_xlim(0, max(row[metric] for row in records) * 1.23)
        ax.invert_yaxis()
    fig.suptitle("Independent design choices: greedy vs random thresholds; full rows vs bootstrap\n"
                 "Same tabular split, 80 trees, sqrt features, depth=6, minimum leaf=3, seed=42", fontsize=11)
    print_view("09", "Four controls include RF without bootstrap. Compare threshold strategies at "
               "fixed row sampling, or row sampling within a family. Timing is a single local observation.")
    for row in records:
        print(f"  {row['family']} bootstrap={row['bootstrap']}: AUC={row['roc_auc']:.4f}; "
              f"log loss={row['log_loss']:.4f}; fit={row['fit_seconds']:.4f}s; "
              f"disagreement={row['disagreement']:.4f}.")
    return save_figure(fig, output / "09_sources_of_randomness.png", dpi), records


def plot_algorithm_comparison(output, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(13, 6), layout="constrained")
    descriptions = [
        ("Decision Tree", [0, 1, 2, 3], False, "All features\nGreedy thresholds"),
        ("Random Forest", [0, 2], False, "Feature subset per node\nGreedy thresholds"),
        ("ExtraTrees", [0, 2], True, "Feature subset per node\nRandom threshold per feature"),
    ]
    for ax, (name, selected, random_thresholds, subtitle) in zip(axes, descriptions):
        ax.set(xlim=(-0.5, 3.5), ylim=(-1.3, 4.1))
        ax.axis("off")
        ax.set_title(name, fontsize=15, fontweight="bold")
        for feature in range(4):
            active = feature in selected
            ax.text(feature, 3.2, f"x{feature + 1}", ha="center", va="center",
                    bbox=dict(boxstyle="round", facecolor="#d9e9df" if active else "#eeeeee",
                              edgecolor="#28743c" if active else "0.8"), color="black" if active else "0.6")
            if not active:
                continue
            ax.annotate("", xy=(feature, 2.45), xytext=(feature, 2.95),
                        arrowprops=dict(arrowstyle="->", color="0.4"))
            thresholds = [0.5] if random_thresholds else [0.2, 0.4, 0.6, 0.8]
            for t in thresholds:
                ax.plot([feature - 0.25, feature + 0.25], [1.8 + t * 0.6] * 2,
                        color="#db6b35" if random_thresholds else "#2574ab", lw=2)
            ax.annotate("", xy=(1.5, 0.85), xytext=(feature, 1.7),
                        arrowprops=dict(arrowstyle="->", color="0.5"))
        ax.text(1.5, 0.55, "Best impurity reduction", ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3c4", edgecolor="#b09533"))
        ax.text(1.5, -0.05, subtitle, ha="center", va="top", fontsize=10)
        rows = "One training dataset" if name == "Decision Tree" else (
            "Rows: bootstrap by default" if name == "Random Forest" else "Rows: full data by default")
        ax.text(1.5, -0.65, rows, ha="center", fontsize=9, color="0.35")
        # Schematic row IDs show duplication/omission, not sampled experimental results.
        row_ids = [1, 1, 3, 4, 4, 6] if name == "Random Forest" else [1, 2, 3, 4, 5, 6]
        row_colors = ["#bdd8e8", "#f0d0b4", "#cedeba", "#dbc9ea", "#e5d3a9", "#bddeda"]
        for i, row_id in enumerate(row_ids):
            ax.text(0.42 + i * 0.43, -1.03, str(row_id), ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor=row_colors[row_id - 1], edgecolor="0.7"))
    fig.suptitle("Node mechanics: feature candidates -> threshold proposals -> supervised selection\n"
                 "Bootstrap is configurable in both ensembles; threshold strategy is the defining distinction",
                 fontsize=12)
    print_view("10", "Conceptual schematic, not measured evidence: ET still selects using labels. "
               "Feature subsets are sampled at each node; bootstrap is a separate per-tree choice.")
    return save_figure(fig, output / "10_tree_ensemble_mechanics.png", dpi)


def animate_ensemble_growth(models, grid, split, output, dpi):
    stages = [1, 5, 10, 25, 50, 100, 200]
    probabilities = {
        family: cumulative_probabilities(
            np.stack([tree.predict_proba(grid[2])[:, 1]
                      for tree in model.estimators_[:max(stages)]]), stages)
        for family, model in models.items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    def frame(i):
        for ax, family in zip(axes, FAMILIES):
            ax.clear()
            boundary_panel(ax, probabilities[family][i], grid, split,
                           f"{family}: {stages[i]} trees", legend=False)
        fig.suptitle("Nested ensemble growth on the same synthetic moons\n"
                     "Bootstrap=False, depth=5, max_features=1 | No guarantee of monotone improvement",
                     fontsize=11)
        return []

    animation = FuncAnimation(fig, frame, frames=len(stages), interval=1000, blit=False)
    path = output / "11_ensemble_growth.gif"
    animation.save(path, writer=PillowWriter(fps=1), dpi=min(dpi, 100))
    plt.close(fig)
    print_view("11", "Seven actual prefixes of seed-42 ensembles; geometry changes as probabilities "
               "are averaged. Local boundary changes need not shrink monotonically.")
    changes = {family: np.abs(np.diff(values, axis=0)).mean(axis=1).tolist()
               for family, values in probabilities.items()}
    return path.name, dict(tree_counts=stages, mean_absolute_grid_probability_change=changes)


def experiment_record(hypothesis, configuration, result, candidate, limitation):
    return dict(hypothesis=hypothesis, configuration=configuration, result=result,
                interpretation_candidate=candidate, interpretation_status="Pending author review",
                limitation=limitation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--skip-html", action="store_true", help="Generate the static variance surface only.")
    parser.add_argument("--skip-animations", action="store_true", help="Generate PNGs and optional HTML only.")
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution; GIFs use at most 110 DPI.")
    args = parser.parse_args()
    if not 80 <= args.dpi <= 250:
        parser.error("--dpi must be between 80 and 250.")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    started = perf_counter()
    data = make_datasets()
    artifacts, experiments = [], {}

    path, result = plot_split_search(data["split"], output, args.dpi)
    artifacts.append(path)
    experiments["01"] = experiment_record(
        "Greedy search can find a stronger threshold than a restricted proposal pool.",
        dict(seed=SEED, proposal_seed=SEED + 1, rows=60, flipped_labels=5, random_proposals=8),
        result, "Compare the observed candidate gains; random proposals still undergo impurity selection.",
        "A one-feature multi-proposal pool is a teaching simplification, not literal ET construction.")
    if not args.skip_animations:
        path, result = animate_random_thresholds(data["split"], output, args.dpi)
        artifacts.append(path)
        experiments["02"] = experiment_record(
            "A small sequence of random proposals need not inspect all greedy thresholds.",
            dict(proposal_seed=SEED + 2, frames=24, fps=3), result,
            "Watch candidate quality vary while best-so-far improves only on a better proposal.",
            "Best-so-far accumulation is a teaching pool, not repeated search within a 1D ExtraTree node.")

    paths, models, grid = plot_decision_boundaries(data["moons"], output, args.dpi)
    artifacts.extend(paths)
    experiments["03-04"] = experiment_record(
        "Different split proposals can create different partitions on identical rows.",
        dict(moons_rows=400, noise=0.23, test_size=0.30, seed=SEED, settings=COMMON,
             ensemble_trees=80, bootstrap=False, individual_trees_shown=6),
        {family: evaluate(models[family], data["moons"][1], data["moons"][3]) for family in FAMILIES},
        "Inspect component partition differences alongside final aggregation.",
        "A two-feature grid depicts geometry; it is not an estimate of bias or repeated-training variance.")
    path, result = plot_tree_correlation(models, data["moons"], output, args.dpi)
    artifacts.append(path)
    experiments["05"] = experiment_record(
        "Random thresholds may change member strength and cross-row diversity at fixed bootstrap.",
        experiments["03-04"]["configuration"], result,
        "Compare actual disagreement and tree accuracy without assuming ET must be more diverse.",
        "Prediction correlation across validation rows is not training-set variance or error correlation.")
    artifacts.extend(plot_variance_surface(output, args.dpi, not args.skip_html))
    experiments["06"] = experiment_record(
        "At fixed member variance, nonzero correlation yields a limiting variance floor.",
        dict(sigma_squared=1, M_range=[1, 500], rho_range=[0, 1]),
        "Analytical surface; no fitted variance estimate.",
        "Under the stated equal-variance/equal-correlation assumptions, the limit is rho.",
        "An illustrative mathematical model; heatmap correlations cannot be substituted as empirical rho.")

    path, result, grown_models = experiment_n_estimators(data["moons"], output, args.dpi)
    artifacts.append(path)
    experiments["07"] = experiment_record(
        "Averaging more trees can reduce variation caused by algorithmic seeds.",
        dict(settings=COMMON, bootstrap=False, seeds=list(range(SEED, SEED + 5)),
             tree_counts=TREE_COUNTS, nested_prefixes=True, data="Same fixed moons split"),
        result, "Inspect the measured AUC paths and prediction SD; monotonic gains are not guaranteed.",
        "Five seeds on one training set estimate algorithmic variation only; no population confidence interval.")
    path, result = experiment_max_features(data["tabular"], output, args.dpi)
    artifacts.append(path)
    experiments["08"] = experiment_record(
        "Changing feature access affects tree strength and diversity; the best setting is task-dependent.",
        dict(data=TABULAR_CONFIG, test_size=0.30, depth=6, min_samples_leaf=3,
             n_estimators=80, bootstrap=False, max_features=[1, 2, "sqrt", 0.5, None], seed=SEED),
        result, "Review both scatter panels; no tuned sweet spot is claimed.",
        "One synthetic dataset and seed, untuned settings; correlation is computed across validation rows.")
    path, result = experiment_bootstrap(data["tabular"], output, args.dpi)
    artifacts.append(path)
    experiments["09"] = experiment_record(
        "Row sampling and threshold strategy are separate factors.",
        dict(data=TABULAR_CONFIG, test_size=0.30, depth=6, min_samples_leaf=3,
             n_estimators=80, max_features="sqrt", seed=SEED, n_jobs=1,
             fit_order=["RF False", "RF True", "ET False", "ET True"]),
        result, "Compare within a row-sampling setting or a model family; inspect task metrics.",
        "Same seed does not pair all RNG draws; timings are single sequential local measurements.")
    artifacts.append(plot_algorithm_comparison(output, args.dpi))
    experiments["10"] = experiment_record(
        "Threshold strategy distinguishes RF and ET independently of bootstrap defaults.",
        dict(schematic_features=4, illustrated_feature_subset=[0, 2]),
        "Conceptual node diagram with illustrative row IDs; no fitted model or sampled bootstrap.",
        "Inspect greedy candidate pools vs one random proposal per selected feature.",
        "Schematic defaults illustrate one configuration; row IDs expose duplicates and omissions.")
    if not args.skip_animations:
        path, result = animate_ensemble_growth(grown_models, grid, data["moons"], output, args.dpi)
        artifacts.append(path)
        experiments["11"] = experiment_record(
            "Averaging more randomized trees can change and stabilize partition geometry.",
            dict(settings=COMMON, bootstrap=False, data="Same moons split", seed=SEED,
                 grid_resolution=110, frames=7, fps=1),
            result, "Inspect actual probability changes between nested prefixes; no monotonic trend is imposed.",
            "One model seed and a plotting grid; this is not repeated-training prediction variance.")

    summary = dict(
        data_source="Synthetic make_moons, make_classification, and a code-defined 1D threshold rule.",
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         sklearn=sklearn.__version__, matplotlib=matplotlib.__version__,
                         platform=platform.platform()),
        experiments=experiments, artifacts=artifacts,
        output_policy="Regenerable artifacts; ignored by default. No public previews selected.",
        interpretation_status="All empirical interpretations require author review.",
        elapsed_seconds=perf_counter() - started,
    )
    path = output / "visual_experiments.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nDAY 28 - EXTRATREES VISUAL LAB | Current synthetic experiments only")
    for family, row in experiments["05"]["result"].items():
        correlation = "undefined" if row["correlation"] is None else f"{row['correlation']:.4f}"
        print(f"{family} (moons, no bootstrap): AUC={row['roc_auc']:.4f}, "
              f"tree accuracy={row['mean_tree_accuracy']:.4f}, disagreement={row['disagreement']:.4f}, "
              f"correlation={correlation}")
    print(f"Generated {len(artifacts)} visual artifacts in {output}:")
    for artifact in artifacts:
        print(f"  {artifact}")
    print(f"Configurations, actual results, review candidates and limitations: {path.name}")
    print(f"Elapsed: {summary['elapsed_seconds']:.1f}s; all interpretations pending author review.")


if __name__ == "__main__":
    main()
