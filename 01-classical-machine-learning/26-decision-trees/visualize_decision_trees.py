"""Decision Tree Visual Lab: impurity, geometry, regularization, and ensembles.

Run from any working directory; outputs default to this script's outputs folder.
    python visualize_decision_trees.py --no-show
    python visualize_decision_trees.py --view candidate-splits recursive-splits
    python visualize_decision_trees.py --include-3d
    python visualize_decision_trees.py --list

Interactive Matplotlib backends show each figure by default. Use --no-show for
batch generation. All data is synthetic. Validation is exploratory; repeated
inspection here is not an independent estimate of final model performance.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

import decision_tree_visual_math as vm

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs"
COLORS = ("#2563eb", "#d97706")
PALE = ListedColormap(("#dbeafe", "#ffedd5"))
PROBABILITY = LinearSegmentedColormap.from_list("class_probability", COLORS)
HIGHLIGHT = "#059669"
STYLE = {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10}


def ensure_output_directory(output_dir=DEFAULT_OUTPUT):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig, output_dir, filename, show):
    path = ensure_output_directory(output_dir) / filename
    try:
        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        fig.savefig(path, dpi=130, bbox_inches="tight")
        if show:
            plt.show()
    finally:
        plt.close(fig)
    return path


def _save_animation(fig, draw, frames, output_dir, filename, show):
    path = ensure_output_directory(output_dir) / filename
    try:
        movie = animation.FuncAnimation(
            fig, draw, frames=frames, interval=1200, repeat=True, cache_frame_data=False
        )
        movie.save(path, writer=animation.PillowWriter(fps=1), dpi=95)
        if show:
            plt.show()
    finally:
        plt.close(fig)
    return path


def _footer(fig, text):
    fig.text(0.5, 0.025, text, ha="center", va="center", fontsize=8)


def _experiment(hypothesis, configuration, result, candidate, limitation):
    return {
        "hypothesis": hypothesis, "configuration": configuration, "result": result,
        "interpretation_candidate_for_author_review": candidate,
        "limitation": limitation, "review_status": "pending author review",
    }


def _scatter(ax, data, validation=True):
    for label, color in enumerate(COLORS):
        rows = data["X_train"][data["y_train"] == label]
        ax.scatter(rows[:, 0], rows[:, 1], c=color, s=16, edgecolors="white", linewidths=0.3)
        if validation:
            rows = data["X_val"][data["y_val"] == label]
            ax.scatter(rows[:, 0], rows[:, 1], c=color, marker="x", s=15, linewidths=0.7)


def _legend(ax, validation=True):
    handles = [
        Line2D([], [], marker="o", linestyle="", color=color, label=f"Class {label}")
        for label, color in enumerate(COLORS)
    ]
    handles.append(Line2D([], [], marker="o", linestyle="", color="#555", label="Training"))
    if validation:
        handles.append(Line2D([], [], marker="x", linestyle="", color="#555", label="Validation"))
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)


def _axes(ax, bounds):
    ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="X1", ylabel="X2")
    ax.set_aspect("equal", adjustable="box")


def _draw_splits(ax, model, bounds):
    for node in vm.node_geometry(model, bounds):
        if not node["leaf"]:
            segment = np.asarray(node["segment"])
            ax.plot(segment[:, 0], segment[:, 1], color="#374151", lw=0.65, alpha=0.7)


def _draw_boundary(ax, model, data, splits=True, legend=False):
    xx, yy, points = vm.make_mesh(data["bounds"])
    prediction = model.predict(points).reshape(xx.shape)
    ax.pcolormesh(xx, yy, prediction, cmap=PALE, vmin=0, vmax=1, shading="nearest", rasterized=True)
    if splits and hasattr(model, "tree_"):
        _draw_splits(ax, model, data["bounds"])
    if not hasattr(model, "tree_"):
        ax.contour(xx, yy, model.predict_proba(points)[:, 1].reshape(xx.shape),
                   levels=[0.5], colors="#374151", linewidths=1)
    _scatter(ax, data)
    _axes(ax, data["bounds"])
    if legend:
        _legend(ax)


def _draw_probability(ax, model, data, legend=False):
    xx, yy, points = vm.make_mesh(data["bounds"])
    class_one = int(np.flatnonzero(model.classes_ == 1)[0])
    probabilities = model.predict_proba(points)[:, class_one].reshape(xx.shape)
    artist = ax.pcolormesh(xx, yy, probabilities, cmap=PROBABILITY, vmin=0, vmax=1,
                          shading="nearest", alpha=0.85, rasterized=True)
    _scatter(ax, data)
    _axes(ax, data["bounds"])
    if legend:
        _legend(ax)
    return artist


def _tree_caption(model, data):
    scores = vm.model_scores(model, data)
    return (f"Depth {scores['depth']} | Leaves {scores['leaves']}\n"
            f"Training {scores['training_accuracy']:.3f} | Validation {scores['validation_accuracy']:.3f}")


def plot_gini(data, output_dir, show):
    p = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(p, vm.gini_binary(p), color=COLORS[0], lw=2.5)
    ax.scatter([0, 0.5, 1], [0, 0.5, 0], color=HIGHLIGHT, zorder=3)
    ax.annotate("Pure node\nGini = 0", (0, 0), xytext=(0.13, 0.11),
                arrowprops={"arrowstyle": "->"}, ha="center")
    ax.annotate("Pure node\nGini = 0", (1, 0), xytext=(0.87, 0.11),
                arrowprops={"arrowstyle": "->"}, ha="center")
    ax.annotate("Equal class mixture\nMaximum Gini = 0.5", (0.5, 0.5),
                xytext=(0.5, 0.36), ha="center", arrowprops={"arrowstyle": "->"})
    ax.set(xlabel="p = P(Y = 1) in the node", ylabel="Gini impurity",
           xlim=(-0.03, 1.03), ylim=(-0.02, 0.56), xticks=[0, 0.5, 1])
    fig.suptitle("Gini Impurity: How Mixed Are the Classes?", fontweight="bold")
    _footer(fig, "Gini(p) = 1 - p² - (1-p)². A pure node has no class mixture.")
    return [_save(fig, output_dir, "01_gini_impurity.png", show)], None


def plot_entropy(data, output_dir, show):
    p = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(p, vm.entropy_binary(p), color=COLORS[1], lw=2.5)
    ax.scatter([0, 0.5, 1], [0, 1, 0], color=HIGHLIGHT, zorder=3)
    for point, text_x in [(0, 0.13), (1, 0.87)]:
        ax.annotate("Pure node\nH = 0 bits", (point, 0), xytext=(text_x, 0.2),
                    ha="center", arrowprops={"arrowstyle": "->"})
    ax.annotate("Equal class mixture\nH = 1 bit", (0.5, 1), xytext=(0.5, 0.72),
                ha="center", arrowprops={"arrowstyle": "->"})
    ax.set(xlabel="p = P(Y = 1) in the node", ylabel="Entropy (bits)",
           xlim=(-0.03, 1.03), ylim=(-0.04, 1.12), xticks=[0, 0.5, 1])
    fig.suptitle("Entropy: Class Uncertainty in Bits", fontweight="bold")
    _footer(fig, "H(p) = -p log₂(p) - (1-p) log₂(1-p); zero-probability terms contribute zero.")
    return [_save(fig, output_dir, "02_entropy.png", show)], None


def compare_impurity_metrics(data, output_dir, show):
    p = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(p, vm.gini_binary(p), label="Gini impurity (raw scale)", color=COLORS[0], lw=2)
    ax.plot(p, vm.entropy_binary(p), label="Entropy (bits, raw scale)", color=COLORS[1], lw=2)
    ax.axvline(0.5, color="#666", linestyle=":", lw=1)
    ax.set(xlabel="p = P(Y = 1) in the node", ylabel="Criterion value (different scales)",
           xlim=(0, 1), ylim=(0, 1.1), xticks=[0, 0.5, 1])
    ax.legend()
    ax.text(0.5, 0.08, "Both peak at balanced classes and fall toward pure nodes.\n"
            "Similar shapes can yield similar split rankings; rankings need not agree.",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    fig.suptitle("Gini and Entropy: Similar Shape, Different Objectives", fontweight="bold")
    _footer(fig, "Neither curve is normalized. Split choice also depends on child sizes and class mixtures.")
    return [_save(fig, output_dir, "03_gini_vs_entropy.png", show)], None


def demonstrate_candidate_splits(data, output_dir, show):
    x, y = vm.toy_data()
    rows = vm.evaluate_thresholds(x, y)
    best = int(np.argmax([row["gain"] for row in rows]))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 1.8]})
    ax = axes[0]
    for label, color in enumerate(COLORS):
        ax.scatter(x[y == label], y[y == label], color=color, marker="o" if label == 0 else "^",
                   s=85, label=f"Class {label}")
    for index, row in enumerate(rows):
        ax.axvline(row["threshold"], color=HIGHLIGHT if index == best else "#aaa",
                   lw=2.5 if index == best else 0.8, linestyle="--")
    ax.set(xlabel="One numeric feature: x", ylabel="Observed class",
           yticks=[0, 1], ylim=(-0.4, 1.5), xlim=(0.5, 8.5))
    ax.legend(loc="upper left", fontsize=8)
    ax.text(rows[best]["threshold"], 1.28, "Best split", color=HIGHLIGHT, ha="center")
    axes[1].axis("off")
    table = axes[1].table(
        cellText=[[f"{r['threshold']:.1f}", r["n_left"], r["n_right"],
                   f"{r['left_impurity']:.3f}", f"{r['right_impurity']:.3f}",
                   f"{r['weighted_impurity']:.3f}", f"{r['gain']:.3f}"] for r in rows],
        colLabels=["Threshold", "N left", "N right", "Gini left", "Gini right", "Weighted Gini", "Gini gain"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    for column in range(7):
        table[best + 1, column].set_facecolor("#d1fae5")
    fig.suptitle(f"Candidate Splits: Parent Gini = {rows[0]['parent_impurity']:.4f}", fontweight="bold")
    _footer(fig, "Weighted child Gini = (N left × Gini left + N right × Gini right) / N parent.")
    result = {"criterion": "gini", "candidates": rows, "best_threshold": rows[best]["threshold"]}
    return [_save(fig, output_dir, "04_candidate_splits.png", show)], _experiment(
        "The largest weighted Gini reduction selects the greedy threshold.",
        {"dataset": "eight code-defined synthetic rows", "x": x.tolist(), "y": y.tolist()},
        result, "Inspect why the highlighted row wins; this is a training split calculation.",
        "A one-feature toy calculation does not measure generalization.",
    )


def visualize_information_gain(data, output_dir, show):
    x, y = vm.toy_data()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    measured = {}
    for ax, criterion, title, ylabel in [
        (axes[0], "gini", "Gini Impurity Reduction", "Gini gain"),
        (axes[1], "entropy", "Entropy Reduction", "Information gain (bits)"),
    ]:
        rows = vm.evaluate_thresholds(x, y, criterion)
        thresholds = [r["threshold"] for r in rows]
        gains = [r["gain"] for r in rows]
        model = DecisionTreeClassifier(criterion=criterion, max_depth=1, random_state=vm.SEED).fit(x[:, None], y)
        chosen = float(model.tree_.threshold[0])
        index = int(np.argmin(np.abs(np.asarray(thresholds) - chosen)))
        ax.plot(thresholds, gains, "o-", color=COLORS[0] if criterion == "gini" else COLORS[1])
        ax.scatter([chosen], [gains[index]], color=HIGHLIGHT, s=90, zorder=3, label="Tree's selected split")
        ax.axvline(chosen, color=HIGHLIGHT, linestyle=":", lw=1)
        ax.set(title=title, xlabel="Candidate threshold", ylabel=ylabel)
        ax.legend(fontsize=8)
        measured[criterion] = {"candidates": rows, "selected_threshold": chosen}
    fig.suptitle("Greedy Split Search: Choose the Largest Reduction", fontweight="bold")
    _footer(fig, "Entropy reduction is information gain; Gini reduction is a different impurity gain.")
    return [_save(fig, output_dir, "05_information_gain_by_threshold.png", show)], _experiment(
        "Each criterion's chosen stump should maximize its calculated weighted gain.",
        {"dataset": "same eight synthetic rows as candidate-splits", "max_depth": 1},
        measured, "Compare the selected thresholds on this particular toy sample.",
        "Similar choices here do not establish equivalence between Gini and entropy.",
    )


def compare_tree_depths(data, output_dir, show):
    depths = [1, 2, 3, 5, None]
    measured = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for index, depth in enumerate(depths):
        ax = axes.flat[index]
        model = vm.fit_tree(data, max_depth=depth)
        measured.append({"max_depth": depth, **vm.model_scores(model, data)})
        _draw_boundary(ax, model, data, legend=index == 0)
        ax.set_title(("Unrestricted" if depth is None else f"Maximum depth = {depth}")
                     + "\n" + _tree_caption(model, data), fontsize=9)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.05, 0.7, "Each internal rule splits one axis.\n"
        "Lines stop at the parent region edges.\n\n"
        "Circles: Training  |  Crosses: Validation\n"
        "Background: predicted class\n\n"
        "These are separately fitted depth caps.", fontsize=11)
    fig.suptitle("Recursive Rules Partition Two Features into Rectangles", fontweight="bold")
    _footer(fig, "Synthetic noisy moons; fixed seed and split. More regions do not guarantee better validation accuracy.")
    return [_save(fig, output_dir, "06_tree_depth_decision_boundaries.png", show)], _experiment(
        "Increasing depth allows finer axis-aligned partitions.",
        data["configuration"], measured,
        "Compare actual geometry and scores at fixed depth caps.",
        "One exploratory Validation split, not a final test estimate.",
    )


def animate_tree_growth(data, output_dir, show):
    depths = [1, 2, 3, 4, 5, None]
    models = [vm.fit_tree(data, max_depth=depth) for depth in depths]
    scores = [{"max_depth": depth, **vm.model_scores(model, data)} for depth, model in zip(depths, models)]
    fig, ax = plt.subplots(figsize=(7, 5.4))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.79, bottom=0.16)
    _footer(fig, "Separate fits with increasing depth caps; fixed Training/Validation split.")

    def draw(frame):
        ax.clear()
        model = models[frame]
        _draw_boundary(ax, model, data, legend=True)
        cap = "Unrestricted" if depths[frame] is None else f"Depth cap {depths[frame]}"
        ax.set_title(f"{cap}\n{_tree_caption(model, data)}", fontsize=10)
        fig.suptitle("Tree Growth: Increasing the Depth Cap", fontweight="bold")

    path = _save_animation(fig, draw, len(models), output_dir, "07_tree_growth.gif", show)
    return [path], _experiment(
        "Increasing the depth cap permits more specific partitions.",
        data["configuration"], scores,
        "Compare fragmentation and the measured Training/Validation scores frame by frame.",
        "Six separately refitted models, not a time trace of one model's fitting algorithm.",
    )


def _plot_tree_counts(model, X, y, ax):
    artists = plot_tree(model, feature_names=["X1", "X2"], class_names=["Class 0", "Class 1"],
                        node_ids=True, filled=True, rounded=True, precision=2, fontsize=8, ax=ax)
    membership = model.decision_path(X)
    for artist in artists:
        text = artist.get_text()
        match = re.search(r"node #(\d+)", text)
        if match:
            node_id = int(match.group(1))
            mask = membership[:, node_id].toarray().ravel().astype(bool)
            counts = [int(np.sum(y[mask] == label)) for label in model.classes_]
            artist.set_text(re.sub(r"value = \[.*?\]", f"class counts = {counts}", text, flags=re.S))
            dominant = int(np.argmax(counts))
            purity = abs(counts[1] - counts[0]) / sum(counts)
            color = np.asarray(to_rgb(COLORS[dominant]))
            artist.get_bbox_patch().set_facecolor((1 - 0.7 * purity) + 0.7 * purity * color)


def visualize_tree_structure(data, output_dir, show):
    model = vm.fit_tree(data, max_depth=3)
    fig, ax = plt.subplots(figsize=(15, 7))
    _plot_tree_counts(model, data["X_train"], data["y_train"], ax)
    fig.suptitle("From Geometric Partitions to If/Else Rules: Depth 3", fontweight="bold")
    _footer(fig, "X1 and X2 are synthetic feature coordinates. Counts are unweighted Training rows in each node.")
    return [_save(fig, output_dir, "08_tree_structure.png", show)], None


def animate_recursive_partitioning(data, output_dir, show):
    model = vm.fit_tree(data, max_depth=3)
    splits = [node for node in vm.node_geometry(model, data["bounds"]) if not node["leaf"]]
    fig, ax = plt.subplots(figsize=(7, 5.4))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.84, bottom=0.16)
    _footer(fig, "One fitted tree; splits revealed breadth first. Each segment stays inside its parent region.")

    def draw(frame):
        ax.clear()
        active = {node["node"] for node in splits[:frame]}
        for region in vm.frontier_regions(model, data["bounds"], active):
            xmin, xmax, ymin, ymax = region["bounds"]
            ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   facecolor=PROBABILITY(region["probability"]), alpha=0.3))
        _scatter(ax, data, validation=False)
        for index, node in enumerate(splits[:frame]):
            segment = np.asarray(node["segment"])
            newest = index == frame - 1
            ax.plot(segment[:, 0], segment[:, 1], color=HIGHLIGHT if newest else "#444",
                    lw=3 if newest else 1)
        _axes(ax, data["bounds"])
        _legend(ax, validation=False)
        subtitle = "No splits: all rows share the root" if frame == 0 else (
            f"Node {splits[frame - 1]['node']}: X{splits[frame - 1]['feature'] + 1} "
            f"<= {splits[frame - 1]['threshold']:.2f}"
        )
        ax.set_title(subtitle)
        fig.suptitle("Recursive Splits Create Axis-Aligned Regions", fontweight="bold")

    path = _save_animation(fig, draw, len(splits) + 1, output_dir, "09_recursive_partitioning.gif", show)
    return [path], None


def analyze_depth_overfitting(data, output_dir, show):
    rows = vm.depth_sweep(data)
    depths = [row["max_depth"] for row in rows]
    training = [row["training_accuracy"] for row in rows]
    validation = [row["validation_accuracy"] for row in rows]
    best = int(np.argmax(validation))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(depths, training, "o-", color=COLORS[0], label="Training")
    ax.plot(depths, validation, "s-", color=COLORS[1], label="Validation")
    ax.scatter(depths[best], validation[best], s=110, facecolors="none", edgecolors=HIGHLIGHT, lw=2)
    ax.annotate(f"Highest observed Validation\nDepth cap {depths[best]}: {validation[best]:.3f}",
                (depths[best], validation[best]), xytext=(0.45, 0.15), textcoords="axes fraction",
                arrowprops={"arrowstyle": "->"}, fontsize=9)
    ax.set(xlabel="Maximum depth", ylabel="Accuracy", xticks=[1, 5, 10, 15, 20], ylim=(0.5, 1.02))
    ax.legend(loc="lower right")
    fig.suptitle("Depth and Generalization: Observed Training and Validation Scores", fontweight="bold")
    _footer(fig, "One noisy synthetic split. A score gap is a diagnostic; this is not a repeated-sampling bias/variance estimate.")
    return [_save(fig, output_dir, "10_depth_vs_generalization.png", show)], _experiment(
        "Extra depth may fit Training observations without improving Validation accuracy.",
        data["configuration"], rows,
        "Inspect the observed score gap and plateaus; review whether added depth is useful here.",
        "Twenty configurations share one Validation sample; its maximum is exploratory, not a final test estimate.",
    )


def analyze_depth_leaves(data, output_dir, show):
    rows = vm.depth_sweep(data)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r["max_depth"] for r in rows], [r["leaves"] for r in rows], "o-", color=HIGHLIGHT)
    ax.set(xlabel="Maximum depth", ylabel="Number of leaves", xticks=[1, 5, 10, 15, 20])
    fig.suptitle("Depth Controls the Number of Terminal Regions", fontweight="bold")
    _footer(fig, "A depth cap is an upper bound; the fitted tree may stop before reaching it.")
    return [_save(fig, output_dir, "11_depth_vs_leaves.png", show)], _experiment(
        "Increasing the depth cap permits more terminal regions.",
        data["configuration"], rows,
        "Inspect where the fitted leaf count grows or stops growing.",
        "Leaf count measures model size, not predictive quality.",
    )


def analyze_min_samples_leaf(data, output_dir, show):
    rows = vm.leaf_sweep(data)
    sizes = [r["min_samples_leaf"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(sizes, [r["training_accuracy"] for r in rows], "o-", color=COLORS[0], label="Training")
    axes[0].plot(sizes, [r["validation_accuracy"] for r in rows], "s-", color=COLORS[1], label="Validation")
    axes[0].set(ylabel="Accuracy", ylim=(0.5, 1.02))
    axes[0].legend()
    axes[1].plot(sizes, [r["leaves"] for r in rows], "o-", color=HIGHLIGHT)
    axes[1].set_ylabel("Number of leaves")
    for ax in axes:
        ax.set_xlabel("Minimum Training rows per leaf")
        ax.set_xticks(sizes)
    fig.suptitle("Minimum Leaf Size Restricts Local Specificity", fontweight="bold")
    _footer(fig, "All other growth settings are held fixed. Validation accuracy need not improve as leaves get larger.")
    paths = [_save(fig, output_dir, "12_min_samples_leaf.png", show)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for index, size in enumerate((1, 5, 20)):
        model = vm.fit_tree(data, min_samples_leaf=size)
        _draw_boundary(axes[index], model, data, legend=index == 0)
        axes[index].set_title(f"Minimum leaf size = {size}\n{_tree_caption(model, data)}", fontsize=9)
    fig.suptitle("Leaf-Size Constraints Change the Partition", fontweight="bold")
    _footer(fig, "Background: predicted class. Boundaries and scores are computed from the fitted models.")
    paths.append(_save(fig, output_dir, "12b_leaf_size_decision_boundaries.png", show))
    return paths, _experiment(
        "Increasing the leaf-size floor restricts small Training-specific regions.",
        data["configuration"], rows,
        "Compare leaf counts, geometry, and observed scores; choose no universal winner from this picture.",
        "One Validation split and no calibration analysis.",
    )


def analyze_pruning(data, output_dir, show):
    rows = vm.pruning_sweep(data)
    alphas = [r["ccp_alpha"] for r in rows]
    best = max(rows, key=lambda r: (r["validation_accuracy"], r["ccp_alpha"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(alphas, [r["leaves"] for r in rows], "o-", color=HIGHLIGHT)
    axes[0].set_ylabel("Number of leaves")
    axes[1].plot(alphas, [r["training_accuracy"] for r in rows], "o-", label="Training", color=COLORS[0])
    axes[1].plot(alphas, [r["validation_accuracy"] for r in rows], "s-", label="Validation", color=COLORS[1])
    axes[1].scatter(best["ccp_alpha"], best["validation_accuracy"], s=100,
                    facecolors="none", edgecolors=HIGHLIGHT, lw=2, label="Highest observed Validation")
    axes[1].set(ylabel="Accuracy", ylim=(0.4, 1.02))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_xlim(left=0)
        ax.set_xlabel("ccp_alpha (symlog; linear near zero)")
    fig.suptitle("Cost-Complexity Pruning Trades Fit for Fewer Leaves", fontweight="bold")
    _footer(fig, f"Training-only pruning path, at most 24 candidates. Highest observed Validation: alpha={best['ccp_alpha']:.5g}.")
    return [_save(fig, output_dir, "13_cost_complexity_pruning.png", show)], _experiment(
        "A larger leaf-count penalty produces a smaller subtree; Validation behavior is data-dependent.",
        {**data["configuration"], "max_path_candidates": 24, "path_data": "Training only"},
        {"candidates": rows, "best_exploratory_validation": best},
        "Review how much Training fit is lost as branches are removed and whether Validation improves here.",
        "Only a subset of the path is inspected; the best Validation point is not an independent test score.",
    )


def animate_pruning(data, output_dir, show):
    rows = vm.pruning_sweep(data, max_candidates=8)
    models = [vm.fit_tree(data, ccp_alpha=r["ccp_alpha"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5.4))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.79, bottom=0.16)
    _footer(fig, "Increasing ccp_alpha on the Training-only pruning path; Validation is measured, not assumed.")

    def draw(frame):
        ax.clear()
        _draw_boundary(ax, models[frame], data, legend=True)
        ax.set_title(f"ccp_alpha = {rows[frame]['ccp_alpha']:.5g}\n"
                     + _tree_caption(models[frame], data), fontsize=10)
        fig.suptitle("Pruning: Removing Branches as the Penalty Increases", fontweight="bold")

    path = _save_animation(fig, draw, len(rows), output_dir, "14_pruning.gif", show)
    return [path], _experiment(
        "Increasing the pruning penalty should simplify the fitted partition.",
        {**data["configuration"], "max_path_candidates": 8}, rows,
        "Inspect which regions disappear and how the measured Validation score changes.",
        "Eight or fewer path values; this animation is explanatory rather than an exhaustive parameter search.",
    )


def visualize_probability_regions(data, output_dir, show):
    model = vm.fit_tree(data, max_depth=4, min_samples_leaf=5)
    fig, ax = plt.subplots(figsize=(8, 5.6))
    artist = _draw_probability(ax, model, data, legend=True)
    _draw_splits(ax, model, data["bounds"])
    fig.colorbar(artist, ax=ax, label="P(Class 1 | fitted leaf)", ticks=[0, 0.25, 0.5, 0.75, 1])
    ax.set_title(_tree_caption(model, data), fontsize=10)
    fig.suptitle("Leaf Probabilities Are Constant Inside Each Rectangle", fontweight="bold")
    _footer(fig, "Training class fraction in the leaf, not evidence of calibrated probability quality.")
    return [_save(fig, output_dir, "15_probability_regions.png", show)], _experiment(
        "Each fitted leaf assigns one empirical class fraction throughout its rectangle.",
        {**data["configuration"], "max_depth": 4, "min_samples_leaf": 5},
        vm.model_scores(model, data),
        "Connect constant color regions to empirical leaf counts.",
        "The displayed Training/Validation scores do not evaluate probability calibration.",
    )


def visualize_leaf_probability_effect(data, output_dir, show):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    result = []
    _, _, points = vm.make_mesh(data["bounds"])
    for index, size in enumerate((1, 20)):
        model = vm.fit_tree(data, min_samples_leaf=size)
        artist = _draw_probability(axes[index], model, data, legend=index == 0)
        probabilities = model.predict_proba(points)[:, 1]
        extreme = float(np.mean(np.isclose(probabilities, 0) | np.isclose(probabilities, 1)))
        leaf_ids = np.flatnonzero(model.tree_.children_left < 0)
        smallest_leaf = int(model.tree_.n_node_samples[leaf_ids].min())
        axes[index].set_title(f"Minimum leaf size = {size}\n"
            f"Leaves {model.get_n_leaves()} | Smallest fitted leaf {smallest_leaf}\n"
            f"Mesh fraction at probability 0 or 1: {extreme:.1%}", fontsize=9)
        fig.colorbar(artist, ax=axes[index], label="P(Class 1 | fitted leaf)", ticks=[0, 0.5, 1])
        result.append({"min_samples_leaf": size, "smallest_fitted_leaf": smallest_leaf,
                       "extreme_probability_mesh_fraction": extreme, **vm.model_scores(model, data)})
    fig.suptitle("Small Leaves Can Produce Extreme Empirical Probabilities", fontweight="bold")
    _footer(fig, "Mesh fractions describe plotted area, not population prevalence. Calibration must be evaluated separately.")
    return [_save(fig, output_dir, "16_leaf_size_probability_effect.png", show)], _experiment(
        "Smaller leaves may create more regions with extreme empirical class fractions.",
        {**data["configuration"], "mesh_resolution": 160}, result,
        "Compare the measured extreme-probability area; review the sampling uncertainty implied by leaf sizes.",
        "Neither visual extremity nor area fraction establishes miscalibration.",
    )


def demonstrate_feature_interaction(data, output_dir, show):
    X, y = vm.interaction_dataset()
    model = DecisionTreeClassifier(max_depth=2, random_state=vm.SEED).fit(X, y)
    illustration = {"X_train": X, "y_train": y, "X_val": np.empty((0, 2)),
                    "y_val": np.empty(0, dtype=int), "bounds": (-1.1, 1.1, -1.1, 1.1)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1, 1.3]})
    _draw_boundary(axes[0], model, illustration)
    _legend(axes[0], validation=False)
    axes[0].set_title("Clean rule: Class 1 if X1 > 0 and X2 > -0.2")
    _plot_tree_counts(model, X, y, axes[1])
    axes[1].set_title("A child rule makes X2 conditionally relevant")
    fig.suptitle("Interactions Through Hierarchical Conditions", fontweight="bold")
    _footer(fig, "Noise-free synthetic grid; fitted thresholds lie between grid values. No held-out performance claim.")
    return [_save(fig, output_dir, "17_feature_interaction.png", show)], None


def demonstrate_tree_instability(data, output_dir, show):
    models = vm.bootstrap_trees(data)
    _, _, points = vm.make_mesh(data["bounds"], resolution=100)
    predictions = np.array([model.predict(points) for model in models])
    disagreements = [float(np.mean(predictions[i] != predictions[j]))
                     for i in range(len(models)) for j in range(i + 1, len(models))]
    average_disagreement = float(np.mean(disagreements))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    result = []
    for index, (ax, model) in enumerate(zip(axes.flat, models)):
        _draw_boundary(ax, model, data, legend=index == 0)
        feature = int(model.tree_.feature[0])
        threshold = float(model.tree_.threshold[0])
        root = "Root is a leaf" if feature < 0 else f"Root: X{feature + 1} <= {threshold:.2f}"
        score = float(model.score(data["X_val"], data["y_val"]))
        ax.set_title(f"Bootstrap {index + 1}: {root}\n"
                     f"Leaves {model.get_n_leaves()} | Validation {score:.3f}", fontsize=9)
        result.append({"bootstrap": index + 1, "root_feature": feature,
                       "root_threshold": threshold, **vm.model_scores(model, data)})
    fig.suptitle("Same Hyperparameters, Different Training Resamples", fontweight="bold")
    _footer(fig, f"Bootstraps replace many rows, not tiny perturbations. Mean pairwise prediction disagreement on the fixed mesh: {average_disagreement:.1%}.")
    return [_save(fig, output_dir, "18_tree_instability.png", show)], _experiment(
        "Training resampling can change fitted rules and predictions at fixed hyperparameters.",
        {**data["configuration"], "bootstrap_count": 6, "max_depth": 5,
         "bootstrap_size": len(data["y_train"]), "mesh_resolution": 100,
         "training_score_reference": "original base Training rows, not the resampled rows"},
        {"trees": result, "mean_pairwise_mesh_disagreement": average_disagreement},
        "Inspect actual differences across the six resamples without assuming that their root rules must differ.",
        "Bootstrap resamples can replace a substantial fraction of rows; mesh disagreement is not a formal variance estimate.",
    )


def compare_tree_logistic(data, output_dir, show):
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=vm.SEED))
    logistic.fit(data["X_train"], data["y_train"])
    tree = vm.fit_tree(data, max_depth=5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    result = {}
    for index, (name, model) in enumerate([("Logistic Regression", logistic), ("Decision Tree (depth cap 5)", tree)]):
        _draw_boundary(axes[index], model, data, legend=index == 0)
        training = float(model.score(data["X_train"], data["y_train"]))
        validation = float(model.score(data["X_val"], data["y_val"]))
        axes[index].set_title(f"{name}\nTraining {training:.3f} | Validation {validation:.3f}", fontsize=10)
        result[name] = {"training_accuracy": training, "validation_accuracy": validation}
    fig.suptitle("Global Linear Boundary vs Conditional Axis-Aligned Rules", fontweight="bold")
    _footer(fig, "Logistic Regression uses two features without nonlinear expansions; its scaler is fitted on Training only.")
    return [_save(fig, output_dir, "19_tree_vs_logistic_regression.png", show)], _experiment(
        "A two-feature linear logit has a different boundary family from a tree on nonlinear moons.",
        {**data["configuration"], "tree_max_depth": 5, "logistic_scaling": "Training-fitted StandardScaler"},
        result, "Compare boundary shapes alongside the actual scores, rather than treating one family as always superior.",
        "One dataset and fixed model settings; not a broadly tuned model benchmark.",
    )


def compare_tree_random_forest(data, output_dir, show):
    tree = vm.fit_tree(data)
    forest = RandomForestClassifier(n_estimators=64, random_state=vm.SEED, n_jobs=1)
    forest.fit(data["X_train"], data["y_train"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    result = {}
    for index, (name, model) in enumerate([("Single unrestricted tree", tree), ("Random Forest (64 trees)", forest)]):
        artist = _draw_probability(axes[index], model, data, legend=index == 0)
        xx, yy, points = vm.make_mesh(data["bounds"])
        probability = model.predict_proba(points)[:, 1].reshape(xx.shape)
        if probability.min() < 0.5 < probability.max():
            axes[index].contour(xx, yy, probability, levels=[0.5], colors="#374151", linewidths=0.8)
        training = float(model.score(data["X_train"], data["y_train"]))
        validation = float(model.score(data["X_val"], data["y_val"]))
        axes[index].set_title(f"{name}\nTraining {training:.3f} | Validation {validation:.3f}", fontsize=10)
        fig.colorbar(artist, ax=axes[index], label="Predicted P(Class 1)", ticks=[0, 0.5, 1])
        result[name] = {"training_accuracy": training, "validation_accuracy": validation}
    fig.suptitle("Single-Tree Fractions vs Averaged Forest Probabilities", fontweight="bold")
    _footer(fig, "The forest averages leaf probabilities from bootstrap trees; the line marks probability 0.5. Stability is not measured here.")
    return [_save(fig, output_dir, "20_tree_vs_random_forest.png", show)], _experiment(
        "Averaging many trees changes the probability surface relative to one tree.",
        {**data["configuration"], "forest_n_estimators": 64, "forest_n_jobs": 1},
        result, "Inspect the averaged surface and computed scores; any performance interpretation requires review.",
        "One split, no hyperparameter search, and no repeated forest stability or calibration evaluation.",
    )


def create_optional_3d_visualization(data, output_dir, show):
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise RuntimeError("The optional probability-3d view requires plotly; install the shared project dependencies.") from error
    model = vm.fit_tree(data, max_depth=4, min_samples_leaf=5)
    fig = go.Figure()
    leaves = [node for node in vm.node_geometry(model, data["bounds"]) if node["leaf"]]
    for index, leaf in enumerate(leaves):
        xmin, xmax, ymin, ymax = leaf["bounds"]
        probability = vm.node_probability(model, leaf["node"])
        # Each leaf is a genuinely flat rectangle, not an interpolated mesh ramp.
        fig.add_trace(go.Mesh3d(
            x=[xmin, xmax, xmax, xmin], y=[ymin, ymin, ymax, ymax], z=[probability] * 4,
            i=[0, 0], j=[1, 2], k=[2, 3], intensity=[probability] * 4,
            colorscale=[[0, COLORS[0]], [1, COLORS[1]]], cmin=0, cmax=1,
            showscale=index == 0, colorbar={"title": "P(Class 1)"}, opacity=0.85,
            name=f"Leaf {leaf['node']}", showlegend=False,
            hovertemplate=f"Leaf {leaf['node']}<br>P(Class 1)={probability:.3f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter3d(
            x=[xmin, xmax, xmax, xmin, xmin], y=[ymin, ymin, ymax, ymax, ymin],
            z=[probability] * 5, mode="lines", line={"color": "#444", "width": 2},
            showlegend=False, hoverinfo="skip",
        ))
    for label, color in enumerate(COLORS):
        rows = data["X_train"][data["y_train"] == label]
        fig.add_trace(go.Scatter3d(
            x=rows[:, 0], y=rows[:, 1], z=model.predict_proba(rows)[:, 1], mode="markers",
            marker={"size": 2, "color": color}, name=f"Class {label} Training",
        ))
    fig.update_layout(
        title="Rotate the Flat Probability Plateaus of Fitted Leaves",
        scene={"xaxis_title": "X1", "yaxis_title": "X2",
               "zaxis_title": "P(Class 1 | fitted leaf)",
               "zaxis": {"range": [0, 1]}, "aspectratio": {"x": 1, "y": 1, "z": 0.6}},
        annotations=[{"text": "Height is a fitted class fraction, not a third feature or a calibration result.",
                      "xref": "paper", "yref": "paper", "x": 0.5, "y": 0, "showarrow": False}],
        margin={"l": 0, "r": 0, "t": 65, "b": 55},
    )
    path = ensure_output_directory(output_dir) / "interactive_decision_tree_lab.html"
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return [path], None


VIEWS = {
    "gini": plot_gini,
    "entropy": plot_entropy,
    "gini-vs-entropy": compare_impurity_metrics,
    "candidate-splits": demonstrate_candidate_splits,
    "information-gain": visualize_information_gain,
    "depth-boundaries": compare_tree_depths,
    "tree-growth": animate_tree_growth,
    "tree-structure": visualize_tree_structure,
    "recursive-splits": animate_recursive_partitioning,
    "depth-generalization": analyze_depth_overfitting,
    "depth-leaves": analyze_depth_leaves,
    "min-samples-leaf": analyze_min_samples_leaf,
    "pruning": analyze_pruning,
    "pruning-animation": animate_pruning,
    "probability-regions": visualize_probability_regions,
    "leaf-probabilities": visualize_leaf_probability_effect,
    "interaction": demonstrate_feature_interaction,
    "instability": demonstrate_tree_instability,
    "tree-vs-logistic": compare_tree_logistic,
    "tree-vs-forest": compare_tree_random_forest,
    "probability-3d": create_optional_3d_visualization,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Generate all 20 core views (the default).")
    selection.add_argument("--view", nargs="+", choices=VIEWS, help="Generate selected named views.")
    parser.add_argument("--list", action="store_true", help="List view names and exit without generating files.")
    parser.add_argument("--include-3d", action="store_true", help="Also generate the optional offline Plotly HTML.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--show", dest="show", action="store_true", help="Show each Matplotlib plot after saving.")
    display.add_argument("--no-show", dest="show", action="store_false", help="Save and close figures without showing.")
    parser.set_defaults(show=None)
    args = parser.parse_args(argv)
    if args.list:
        for name in VIEWS:
            print(name)
        return 0
    names = list(dict.fromkeys(args.view)) if args.view else [name for name in VIEWS if name != "probability-3d"]
    if args.include_3d and "probability-3d" not in names:
        names.append("probability-3d")
    noninteractive = {"agg", "pdf", "svg", "ps", "cairo", "template"}
    show = args.show if args.show is not None else plt.get_backend().lower() not in noninteractive
    output_dir = ensure_output_directory(args.output_dir)
    data = vm.create_dataset()
    report = {
        "configuration": {**data["configuration"], "versions": {
            package: version(package) for package in ("numpy", "matplotlib", "scikit-learn", "Pillow")
        }},
        "evidence_boundary": "Synthetic exploratory Training/Validation demonstrations; no independent test set.",
        "experiments": {},
    }
    generated = []
    with plt.rc_context(STYLE):
        for name in names:
            print(f"Generating {name}...", flush=True)
            paths, experiment = VIEWS[name](data, output_dir, show)
            for path in paths:
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Expected generated file is missing or empty: {path}")
                generated.append(path)
                print(f"  Saved {path.name}", flush=True)
            if experiment is not None:
                report["experiments"][name] = experiment
    report["generated_files"] = [path.name for path in generated]
    report_path = output_dir / "experiment_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    generated.append(report_path)
    print("\nDecision Tree Visual Lab\n========================")
    print("Generated:")
    for path in generated:
        print(f"- {path.name}")
    print(f"\nOutputs saved under: {output_dir.resolve()}")
    print("Measured values are in experiment_report.json; interpretations remain pending author review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
