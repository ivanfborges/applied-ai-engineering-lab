"""Command-line visual lab for Random Forest mechanics on seeded synthetic data.

Generated artifacts are local and ignored by Git. Run --all, --static,
--animations, --interactive, or --only NAME from the repository root.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

SEED = 27
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs" / "visualizations"
BLUE = "#2364aa"
ORANGE = "#e07a36"
TEAL = "#178f83"
DARK = "#253346"
PALE = ListedColormap(["#dce9f5", "#fbe4d5"])

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, name, tight=True):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / name
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    return path


def _report(path, question, observation, limitation):
    print("-" * 58)
    print(path.name)
    print(f"Question: {question}")
    print(f"What to observe: {observation}")
    print(f"Important limitation: {limitation}")
    return path


def make_boundary_data():
    """One fixed two-dimensional synthetic data split shared across panels."""
    X, y = make_moons(n_samples=360, noise=0.23, random_state=SEED)
    return train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)


def _mesh(X, resolution=115):
    x0 = np.linspace(X[:, 0].min() - 0.35, X[:, 0].max() + 0.35, resolution)
    x1 = np.linspace(X[:, 1].min() - 0.35, X[:, 1].max() + 0.35, resolution)
    xx, yy = np.meshgrid(x0, x1)
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def _boundary(ax, model, X, y, title, probability=False):
    xx, yy, grid = _mesh(X)
    values = model.predict_proba(grid)[:, 1] if probability else model.predict(grid)
    ax.contourf(xx, yy, values.reshape(xx.shape), levels=np.linspace(0, 1, 9), cmap=PALE, alpha=0.85)
    if probability:
        ax.contour(xx, yy, values.reshape(xx.shape), levels=[0.5], colors=[DARK], linewidths=1)
    for klass, color, marker in ((0, BLUE, "o"), (1, ORANGE, "^")):
        mask = y == klass
        ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker, s=13, alpha=0.65,
                   edgecolors="white", linewidths=0.2, label=f"class {klass}")
    ax.set_title(title)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    return values


def bootstrap_membership(n_rows, n_trees, seed=SEED):
    """Boolean in-bag membership, not bootstrap multiplicity."""
    if n_rows < 2 or n_trees < 1:
        raise ValueError("n_rows must be >= 2 and n_trees must be >= 1")
    rng = np.random.default_rng(seed)
    matrix = np.zeros((n_rows, n_trees), dtype=bool)
    samples = []
    for b in range(n_trees):
        indices = rng.integers(0, n_rows, size=n_rows)
        matrix[indices, b] = True
        samples.append(indices)
    return matrix, samples


def simulate_bootstrap_unique_fraction(sizes=(10, 25, 50, 100, 250, 500), repeats=300, seed=SEED):
    """Return simulated mean and central 90% interval of unique-row fractions."""
    if repeats < 1 or any(n < 2 for n in sizes):
        raise ValueError("repeats must be positive and all sizes must be >= 2")
    rng = np.random.default_rng(seed)
    draws = []
    for n in sizes:
        fractions = np.array([len(np.unique(rng.integers(0, n, size=n))) / n for _ in range(repeats)])
        draws.append(fractions)
    return np.array(sizes), np.array([x.mean() for x in draws]), np.array(
        [np.quantile(x, [0.05, 0.95]) for x in draws]
    )


def pairwise_agreement(predictions):
    """Mean pairwise equality for a trees-by-observations prediction matrix."""
    values = np.asarray(predictions)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("predictions must have at least two trees and one observation")
    pairs = np.triu_indices(values.shape[0], k=1)
    return float(np.mean(values[pairs[0]] == values[pairs[1]]))


def variance_curve(tree_counts, rho):
    """Normalized equal-variance, equal-pairwise-correlation formula."""
    counts = np.asarray(tree_counts)
    if np.any(counts < 1) or not 0 <= rho <= 1:
        raise ValueError("tree counts must be positive and rho must be in [0, 1]")
    return rho + (1 - rho) / counts


def plot_tree_instability():
    X_train, X_test, y_train, y_test = make_boundary_data()
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat[:5]):
        idx = rng.integers(0, len(X_train), size=len(X_train))
        tree = DecisionTreeClassifier(random_state=SEED).fit(X_train[idx], y_train[idx])
        _boundary(ax, tree, X_train, y_train, f"Bootstrap tree {i + 1}")
    forest = RandomForestClassifier(n_estimators=120, max_features=1, min_samples_leaf=2,
                                    random_state=SEED, n_jobs=1).fit(X_train, y_train)
    _boundary(axes.flat[5], forest, X_train, y_train, "Random Forest aggregate")
    axes.flat[5].legend(loc="lower right", fontsize=8)
    fig.suptitle("One dataset, different bootstrap trees, one aggregate", fontsize=15)
    fig.text(0.5, 0.01, "All panels overlay the same original training rows. Boundaries come from fitted models.",
             ha="center", color=DARK)
    path = _save(fig, "01_tree_instability.png")
    return _report(path, "Why can a single tree be unstable?",
                   f"Five trees used different bootstrap indices; the forest held-out accuracy was {accuracy_score(y_test, forest.predict(X_test)):.3f}. Compare boundary shapes, not just this score.",
                   "One synthetic moons dataset and one sequence of bootstrap samples.")


def plot_bootstrap_sampling():
    n = 12
    _, samples = bootstrap_membership(n, 1)
    draws = samples[0]
    counts = np.bincount(draws, minlength=n)
    colors = [ORANGE if c > 1 else TEAL if c == 1 else "#c8d0d9" for c in counts]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [1, 2]})
    ax1.scatter(np.arange(1, n + 1), np.ones(n), c=[colors[i] for i in draws], s=500, marker="s")
    for j, i in enumerate(draws):
        ax1.text(j + 1, 1, str(i + 1), ha="center", va="center", color="white" if counts[i] else DARK)
    ax1.set(xlim=(0.3, n + 0.7), ylim=(0.7, 1.3), xticks=np.arange(1, n + 1), yticks=[],
            xlabel="bootstrap draw position", title="N draws with replacement: repeated labels are allowed")
    ax2.bar(np.arange(1, n + 1), counts, color=colors, edgecolor="white")
    missing = np.flatnonzero(counts == 0) + 1
    ax2.scatter(missing, np.zeros(len(missing)), marker="s", s=120, color="#c8d0d9", zorder=3)
    for index in missing:
        ax2.text(index, 0.15, "OOB", ha="center", fontsize=8, color=DARK)
    ax2.set_ylim(-0.12, max(counts) + 0.35)
    ax2.set(xticks=np.arange(1, n + 1), xlabel="original observation index",
            ylabel="times selected", title="Original rows: gray = OOB, teal = once, orange = repeated")
    ax2.axhline(0, color=DARK, linewidth=0.5)
    unique = int(np.count_nonzero(counts))
    fig.suptitle(f"Bootstrap sample: {n} draws, {unique} unique rows, {n - unique} OOB rows", fontsize=15)
    path = _save(fig, "02_bootstrap_sampling.png")
    return _report(path, "What changes under sampling with replacement?",
                   f"This draw had {unique}/{n} unique selected rows and {n - unique} OOB rows; bar heights show multiplicity.",
                   "The 63.2% result is a large-n expectation, not a fixed proportion for each draw.")


def plot_bootstrap_632():
    sizes, means, bounds = simulate_bootstrap_unique_fraction()
    exact = 1 - (1 - 1 / sizes) ** sizes
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, means, "o-", color=BLUE, label="simulated mean (300 bootstraps per N)")
    ax.fill_between(sizes, bounds[:, 0], bounds[:, 1], color=BLUE, alpha=0.15, label="simulation 5th–95th percentile")
    ax.plot(sizes, exact, "--", color=ORANGE, label="exact expected fraction, finite N")
    ax.axhline(1 - 1 / math.e, color=DARK, linestyle=":", label="limit 1 − 1/e ≈ 0.632")
    ax.set(xlabel="number of original observations N", ylabel="unique selected / N",
           title="Unique-row fraction approaches its bootstrap limit")
    ax.set_ylim(0.45, 0.86)
    ax.legend(loc="upper right")
    path = _save(fig, "03_bootstrap_632.png")
    return _report(path, "Why does the unique fraction approach 63.2%?",
                   f"At N={sizes[-1]}, the simulated mean was {means[-1]:.3f}; the finite-N expectation was {exact[-1]:.3f}.",
                   "Each point averages 300 seeded bootstrap draws; the shaded band is draw-to-draw variation.")


def plot_oob_mechanism():
    n, trees = 10, 8
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] > 0) | (X[:, 1] > 0.55)).astype(int)
    membership, samples = bootstrap_membership(n, trees)
    fitted = [DecisionTreeClassifier(max_depth=2, random_state=SEED + b).fit(X[idx], y[idx])
              for b, idx in enumerate(samples)]
    row = int(np.argmax((~membership).sum(axis=1)))
    eligible = np.flatnonzero(~membership[row])
    votes = [int(fitted[b].predict(X[row:row + 1])[0]) for b in eligible]
    prediction = int(np.mean(votes) >= 0.5)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow((~membership).astype(int), aspect="auto", cmap=ListedColormap(["#cad3dd", "#70c1b3"]), vmin=0, vmax=1)
    for i in range(n):
        for b in range(trees):
            label = f"vote {fitted[b].predict(X[i:i + 1])[0]}" if i == row and not membership[i, b] else (
                "OOB" if not membership[i, b] else "in")
            ax.text(b, i, label, ha="center", va="center", fontsize=8,
                    fontweight="bold" if i == row else "normal")
    ax.add_patch(plt.Rectangle((-0.5, row - 0.5), trees, 1, fill=False, color=ORANGE, linewidth=2.5))
    ax.set(xticks=np.arange(trees), xticklabels=[f"tree {i + 1}" for i in range(trees)],
           yticks=np.arange(n), yticklabels=[f"row {i + 1}" for i in range(n)],
           xlabel="bootstrap-fitted tree", ylabel="original training observation")
    ax.set_title("OOB prediction uses only trees that omitted the highlighted row")
    fig.text(0.5, 0.01, f"Row {row + 1}: eligible trees {list(eligible + 1)} → votes {votes} → OOB class {prediction}",
             ha="center", color=DARK)
    path = _save(fig, "04_oob_mechanism.png")
    return _report(path, "Which trees may vote on a training row's OOB prediction?",
                   f"Row {row + 1} was OOB for {len(eligible)} of {trees} trees; its eligible votes gave class {prediction}.",
                   "A ten-row schematic with shallow fitted trees; real OOB scores aggregate across all covered rows.")


def plot_feature_randomness():
    rng = np.random.default_rng(SEED)
    names = [f"x{i}" for i in range(1, 7)]
    subsets = [rng.choice(names, size=3, replace=False).tolist() for _ in range(3)]
    winners = [rng.choice(items) for items in subsets]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    positions = [(5, 4.9), (2.5, 2.7), (7.5, 2.7)]
    for child in positions[1:]:
        ax.annotate("", xy=child, xytext=positions[0], arrowprops={"arrowstyle": "->", "color": DARK, "lw": 1.5})
    for j, (x, y) in enumerate(positions):
        label = f"{'root' if j == 0 else 'child ' + str(j)}\n candidates: {', '.join(subsets[j])}\n illustrative winner: {winners[j]}"
        ax.text(x, y, label, ha="center", va="center", fontsize=11,
                bbox={"boxstyle": "round,pad=0.55", "facecolor": "#eaf3fa" if j == 0 else "#ecf7f4",
                      "edgecolor": BLUE if j == 0 else TEAL})
    ax.text(5, 0.8, "A new candidate subset is sampled at each node, not once per tree.",
            ha="center", fontsize=12, fontweight="bold", color=DARK)
    ax.set_title("Conceptual illustration: feature randomness at each split", fontsize=15)
    path = _save(fig, "05_feature_randomness.png")
    return _report(path, "Where does Random Forest sample features?",
                   "The three displayed nodes have separately sampled candidate lists; winners are illustrative choices from those lists.",
                   "This is a labeled schematic, not a trace of scikit-learn's hidden candidate subsets.")


def compare_bagging_random_forest():
    X_train, X_test, y_train, y_test = make_boundary_data()
    tree = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)
    bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=80,
                            bootstrap=True, random_state=SEED, n_jobs=1).fit(X_train, y_train)
    forest = RandomForestClassifier(n_estimators=80, max_features=1,
                                    random_state=SEED, n_jobs=1).fit(X_train, y_train)
    bag_preds = np.stack([est.predict(X_test[:, features])
                          for est, features in zip(bag.estimators_, bag.estimators_features_)])
    forest_preds = np.stack([est.predict(X_test) for est in forest.estimators_])
    bag_agree = pairwise_agreement(bag_preds)
    forest_agree = pairwise_agreement(forest_preds)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, model, title in zip(axes, (tree, bag, forest),
                                ("Single tree", f"Bagging\npairwise agreement {bag_agree:.3f}",
                                 f"Random Forest\npairwise agreement {forest_agree:.3f}")):
        _boundary(ax, model, X_train, y_train, title)
    fig.suptitle("Same synthetic data: one tree, row-randomized bagging, row + split-feature randomized forest",
                 fontsize=13)
    fig.text(0.5, 0.01, "Agreement = fraction of held-out labels on which two fitted trees match; higher means more similar.",
             ha="center", color=DARK)
    path = _save(fig, "06_bagging_vs_random_forest.png")
    return _report(path, "How does a forest differ from plain bagging?",
                   f"Mean pairwise held-out prediction agreement was {bag_agree:.3f} for bagging and {forest_agree:.3f} for the forest.",
                   "Agreement is sample- and label-distribution-dependent; this run does not prove a universal ordering.")


def plot_variance_correlation_relationship():
    counts = np.arange(1, 201)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for rho, color in zip((0, 0.1, 0.3, 0.6, 0.9, 1), (BLUE, TEAL, "#8b6cc1", ORANGE, "#b75555", DARK)):
        ax.plot(counts, variance_curve(counts, rho), label=f"ρ = {rho:.1f}", color=color, linewidth=2)
    ax.annotate("Perfect correlation: no variance reduction", xy=(145, 1), xytext=(75, 0.82),
                arrowprops={"arrowstyle": "->", "color": DARK}, color=DARK)
    ax.annotate("Independent trees: 1/B", xy=(80, 1 / 80), xytext=(90, 0.19),
                arrowprops={"arrowstyle": "->", "color": BLUE}, color=BLUE)
    ax.set(xlabel="number of trees B", ylabel="variance of average / variance of one tree",
           xlim=(1, 200), ylim=(0, 1.05), title="Tree correlation limits the benefit of averaging")
    ax.legend(ncol=2)
    fig.text(0.5, 0.01, "Theory: equal tree variance and equal pairwise correlation; normalized individual variance = 1.",
             ha="center", color=DARK)
    path = _save(fig, "07_correlation_variance.png")
    return _report(path, "Why does ensemble diversity matter?",
                   f"At B=200, the theoretical relative variance is {variance_curve(200, 0):.3f} for rho=0 and {variance_curve(200, 0.9):.3f} for rho=0.9.",
                   "The curves are a theoretical equal-correlation model, not measured forest performance.")


def animate_forest_growth():
    X_train, _, y_train, _ = make_boundary_data()
    counts = (1, 5, 10, 25, 50, 100, 120)
    forest = RandomForestClassifier(n_estimators=1, max_features=1, min_samples_leaf=2,
                                    warm_start=True, random_state=SEED, n_jobs=1)
    xx, yy, grid = _mesh(X_train, resolution=90)
    fields = []
    for count in counts:
        forest.n_estimators = count
        forest.fit(X_train, y_train)
        fields.append(forest.predict_proba(grid)[:, 1].reshape(xx.shape))
    fig, ax = plt.subplots(figsize=(6.3, 4.8))

    def update(frame):
        ax.clear()
        ax.contourf(xx, yy, fields[frame], levels=np.linspace(0, 1, 9), cmap=PALE)
        ax.contour(xx, yy, fields[frame], levels=[0.5], colors=[DARK], linewidths=1.3)
        for klass, color, marker in ((0, BLUE, "o"), (1, ORANGE, "^")):
            mask = y_train == klass
            ax.scatter(X_train[mask, 0], X_train[mask, 1], c=color, marker=marker,
                       s=12, alpha=0.6, edgecolors="white", linewidths=0.2)
        ax.set(xlabel="feature 1", ylabel="feature 2",
               title=f"Growing one seeded forest: {counts[frame]} tree{'s' if counts[frame] > 1 else ''}")
        ax.text(0.02, 0.02, "Boundary = mean class probability ≥ 0.5", transform=ax.transAxes,
                fontsize=8, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        return []

    animation = FuncAnimation(fig, update, frames=len(counts), interval=900, blit=False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "08_forest_growth.gif"
    animation.save(path, writer=PillowWriter(fps=1), dpi=90)
    plt.close(fig)
    changed = float(np.mean((fields[0] >= 0.5) != (fields[-1] >= 0.5)))
    return _report(path, "How does the aggregate boundary change as trees are added?",
                   f"Between B=1 and B=120, {changed:.1%} of grid cells changed predicted class in this seeded run.",
                   "Frames use one warm-start forest; they do not imply monotonic improvement in held-out accuracy.")



def experiment_n_estimators():
    X_train, X_test, y_train, y_test = make_boundary_data()
    counts = np.array([1, 2, 5, 10, 20, 50, 100, 200])
    train_scores, test_scores, oob_scores, seconds = [], [], [], []
    for count in counts:
        model = RandomForestClassifier(n_estimators=int(count), max_features=1,
                                       min_samples_leaf=2, oob_score=count >= 20,
                                       random_state=SEED, n_jobs=1)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        seconds.append(time.perf_counter() - start)
        train_scores.append(accuracy_score(y_train, model.predict(X_train)))
        test_scores.append(accuracy_score(y_test, model.predict(X_test)))
        if count >= 20 and np.all(model.oob_decision_function_.sum(axis=1) > 0):
            oob_scores.append(model.oob_score_)
        else:
            oob_scores.append(np.nan)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.7))
    ax1.plot(counts, train_scores, "o-", color=BLUE, label="training")
    ax1.plot(counts, test_scores, "s-", color=ORANGE, label="held-out test")
    ax1.plot(counts, oob_scores, "^-", color=TEAL, label="OOB, full coverage only")
    ax1.set(xscale="log", xlabel="number of trees", ylabel="accuracy", ylim=(0.5, 1.02),
            title="Scores are measured, not guaranteed monotonic")
    ax1.set_xticks(counts, labels=[str(x) for x in counts])
    ax1.legend()
    ax2.plot(counts, seconds, "o-", color=DARK)
    ax2.set(xscale="log", xlabel="number of trees", ylabel="fit time (seconds)",
            title="Measured local fit cost")
    ax2.set_xticks(counts, labels=[str(x) for x in counts])
    fig.suptitle("Tree count: measured scores and CPU fit time on synthetic moons", fontsize=14)
    path = _save(fig, "09_n_estimators.png")
    valid = [(n, s) for n, s in zip(counts, oob_scores) if np.isfinite(s)]
    oob_note = f"OOB ranged {min(s for _, s in valid):.3f} to {max(s for _, s in valid):.3f} over eligible counts." if valid else "No OOB point had full row coverage."
    return _report(path, "How do scores and cost evolve with tree count?",
                   f"Test accuracy was {test_scores[0]:.3f} at B=1 and {test_scores[-1]:.3f} at B=200. {oob_note}",
                   "One train/test split; timing depends on this machine. OOB is omitted below 20 trees or without full coverage.")


def _max_features_data():
    rng = np.random.default_rng(SEED + 1)
    X = rng.normal(size=(500, 6))
    y = ((X[:, 0] + 0.45 * X[:, 1]) > 0.25).astype(int)
    y ^= (rng.random(len(y)) < 0.09).astype(int)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)


def experiment_max_features():
    X_train, X_test, y_train, y_test = _max_features_data()
    settings = [(1, 1, "1"), ("sqrt", 2, "sqrt (=2)"), (3, 3, "3"), (4, 4, "4"), (1.0, 6, "all 6")]
    tree_scores, oob_scores, test_scores, agreements = [], [], [], []
    for value, _, _ in settings:
        model = RandomForestClassifier(n_estimators=100, max_features=value,
                                       oob_score=True, random_state=SEED, n_jobs=1).fit(X_train, y_train)
        tree_predictions = np.stack([tree.predict(X_test) for tree in model.estimators_])
        tree_scores.append(float(np.mean([accuracy_score(y_test, p) for p in tree_predictions])))
        agreements.append(pairwise_agreement(tree_predictions))
        oob_scores.append(float(model.oob_score_))
        test_scores.append(float(accuracy_score(y_test, model.predict(X_test))))
    x = np.array([item[1] for item in settings])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.7))
    ax1.plot(x, tree_scores, "o--", color=BLUE, label="mean individual-tree test accuracy")
    ax1.plot(x, oob_scores, "s-", color=TEAL, label="forest OOB accuracy")
    ax1.plot(x, test_scores, "^-", color=ORANGE, label="forest held-out accuracy")
    ax1.set(xlabel="candidate features per split", ylabel="accuracy",
            title="Tree strength and ensemble scores")
    ax1.set_xticks(x, labels=[item[2] for item in settings])
    ax1.set_xlabel("candidate features per split", labelpad=12)
    ax1.legend(fontsize=8)
    ax2.plot(x, agreements, "o-", color=DARK)
    ax2.set(xlabel="candidate features per split", ylabel="mean pairwise held-out agreement",
            title="Similarity between individual tree predictions")
    ax2.set_xticks(x, labels=[item[2] for item in settings])
    ax2.set_xlabel("candidate features per split", labelpad=12)
    fig.suptitle("Changing max_features on six-feature synthetic data", fontsize=14)
    path = _save(fig, "10_max_features.png")
    return _report(path, "What changes when more features may compete at each split?",
                   f"At 1 versus 6 candidates, mean tree accuracy was {tree_scores[0]:.3f} versus {tree_scores[-1]:.3f}; agreement was {agreements[0]:.3f} versus {agreements[-1]:.3f}.",
                   "This is one seeded generator; agreement and score need not vary monotonically elsewhere.")


def plot_regression_ensemble():
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-3, 3, size=(90, 1))
    truth = lambda x: np.sin(1.5 * x) + 0.25 * x
    y = truth(X[:, 0]) + rng.normal(0, 0.30, size=len(X))
    forest = RandomForestRegressor(n_estimators=80, max_features=1,
                                   min_samples_leaf=2, random_state=SEED, n_jobs=1).fit(X, y)
    grid = np.linspace(-3.2, 3.2, 350).reshape(-1, 1)
    predictions = np.stack([tree.predict(grid) for tree in forest.estimators_])
    mean_prediction = forest.predict(grid)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for i in range(8):
        ax.plot(grid[:, 0], predictions[i], color=BLUE, alpha=0.24, linewidth=1.1,
                label="individual tree" if i == 0 else None)
    ax.plot(grid[:, 0], mean_prediction, color=ORANGE, linewidth=2.7, label="forest average")
    ax.plot(grid[:, 0], truth(grid[:, 0]), "--", color=TEAL, linewidth=1.5, label="known generating function")
    ax.scatter(X[:, 0], y, s=16, color=DARK, alpha=0.55, label="noisy synthetic training points")
    ax.set(xlabel="input x", ylabel="predicted or observed y",
           title="Regression: individual step functions and their average")
    ax.legend(fontsize=8)
    path = _save(fig, "11_regression_ensemble.png")
    mid = len(grid) // 2
    return _report(path, "What does averaging noisy tree predictions look like?",
                   f"At x={grid[mid, 0]:.2f}, the 80 tree predictions had SD {predictions[:, mid].std():.3f}; their mean was {mean_prediction[mid]:.3f}.",
                   "The true function is known only because the data are synthetic; the forest average is still piecewise.")


def plot_probability_surface_3d():
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("12_probability_surface.html skipped: install plotly to generate the optional 3D view.")
        return None
    X_train, _, y_train, _ = make_boundary_data()
    forest = RandomForestClassifier(n_estimators=100, max_features=1, min_samples_leaf=2,
                                    random_state=SEED, n_jobs=1).fit(X_train, y_train)
    xx, yy, grid = _mesh(X_train, resolution=70)
    zz = forest.predict_proba(grid)[:, 1].reshape(xx.shape)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale="RdBu", cmin=0, cmax=1,
                             opacity=0.9, colorbar={"title": "P(class 1)"}))
    fig.add_trace(go.Scatter3d(x=X_train[:, 0], y=X_train[:, 1], z=y_train,
                               mode="markers", name="training labels (z=0/1)",
                               marker={"size": 2, "color": y_train, "colorscale": "RdBu",
                                       "cmin": 0, "cmax": 1, "opacity": 0.55}))
    fig.update_layout(title="Synthetic moons: piecewise Random Forest class probability",
                      scene={"xaxis_title": "feature 1", "yaxis_title": "feature 2",
                             "zaxis_title": "predicted P(class 1)", "zaxis_range": [0, 1]},
                      margin={"l": 0, "r": 0, "t": 55, "b": 0})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "12_probability_surface.html"
    fig.write_html(path, include_plotlyjs=True, full_html=True, auto_open=False)
    return _report(path, "How does a forest's class probability vary over two features?",
                   f"The fitted surface spans probabilities {zz.min():.3f} to {zz.max():.3f} on the displayed grid.",
                   "Leaf class proportions averaged across trees are not automatically calibrated probabilities.")


def _importance_experiment():
    rng = np.random.default_rng(SEED + 2)
    n = 600
    signal_a = rng.uniform(-1, 1, n)
    signal_b = rng.uniform(-1, 1, n)
    X = np.column_stack([signal_a, signal_b, rng.uniform(-1, 1, n),
                         rng.integers(0, 2, n), rng.integers(0, 30, n)])
    names = ("signal_a", "signal_b", "noise_continuous", "noise_binary", "noise_30_values")
    y = ((signal_a > 0.25) | (signal_b < -0.45)).astype(int)
    y ^= (rng.random(n) < 0.10).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    model = RandomForestClassifier(n_estimators=150, max_features="sqrt",
                                   random_state=SEED, n_jobs=1).fit(X_train, y_train)
    perm = permutation_importance(model, X_test, y_test, scoring="accuracy",
                                  n_repeats=10, random_state=SEED, n_jobs=1)
    return names, model.feature_importances_, perm.importances_mean, perm.importances_std, (
        accuracy_score(y_test, model.predict(X_test))
    )


def plot_mdi_importance():
    names, mdi, _, _, score = _importance_experiment()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, mdi, color=[TEAL, TEAL, ORANGE, ORANGE, ORANGE])
    ax.invert_yaxis()
    ax.set(xlabel="normalized training impurity decrease", title="Impurity-based feature importance (MDI)")
    ax.text(0.98, 0.03, "Teal: in generating rule\nOrange: unrelated noise",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#d8e0e8"})
    path = _save(fig, "13_mdi_importance.png")
    return _report(path, "What did the fitted trees use to reduce training impurity?",
                   f"MDI for continuous noise was {mdi[2]:.3f}, binary noise {mdi[3]:.3f}; held-out accuracy was {score:.3f}.",
                   "MDI is based on training splits and may favor many-valued variables; it is not causal importance.")


def compare_feature_importance():
    names, mdi, perm, sd, score = _importance_experiment()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    y_pos = np.arange(len(names))
    ax1.barh(y_pos, mdi, color=BLUE)
    ax1.set(xlabel="normalized impurity decrease", title="MDI: fitted training splits")
    ax2.barh(y_pos, perm, xerr=sd, color=ORANGE, error_kw={"capsize": 3})
    ax2.axvline(0, color=DARK, linewidth=0.8)
    ax2.set(xlabel="held-out accuracy decrease", title="Permutation: score after shuffling")
    ax1.set_yticks(y_pos, labels=names)
    ax1.invert_yaxis()
    fig.suptitle("Two importance questions, same synthetic model and held-out set", fontsize=14)
    fig.text(0.5, 0.01, "Different units; compare rankings and noise behavior. Whiskers = SD across 10 shuffles.",
             ha="center", color=DARK)
    path = _save(fig, "14_mdi_vs_permutation.png")
    mdi_order = [names[i] for i in np.argsort(-mdi)[:2]]
    perm_order = [names[i] for i in np.argsort(-perm)[:2]]
    return _report(path, "How does held-out permutation importance differ from MDI?",
                   f"Top two features were {mdi_order} by MDI and {perm_order} by permutation. Continuous noise: MDI {mdi[2]:.3f}, permutation {perm[2]:+.3f}; accuracy {score:.3f}.",
                   "One dataset and one score; permutation can be unstable and can break feature dependence.")


def plot_correlated_feature_effect():
    rng = np.random.default_rng(SEED + 3)
    n = 700
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.04, size=n)
    noise = rng.normal(size=n)
    y = (x1 > 0.1).astype(int) ^ (rng.random(n) < 0.10).astype(int)
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.30,
                                           stratify=y, random_state=SEED)
    X_both = np.column_stack([x1, x2, noise])
    X_single = np.column_stack([x1, noise])
    model_both = RandomForestClassifier(n_estimators=150, max_features=2,
                                        random_state=SEED, n_jobs=1).fit(X_both[train_idx], y[train_idx])
    model_single = RandomForestClassifier(n_estimators=150, max_features=1,
                                          random_state=SEED, n_jobs=1).fit(X_single[train_idx], y[train_idx])
    both = permutation_importance(model_both, X_both[test_idx], y[test_idx],
                                  scoring="accuracy", n_repeats=12, random_state=SEED, n_jobs=1)
    single = permutation_importance(model_single, X_single[test_idx], y[test_idx],
                                    scoring="accuracy", n_repeats=12, random_state=SEED, n_jobs=1)
    corr = float(np.corrcoef(x1[train_idx], x2[train_idx])[0, 1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    for klass, color, marker in ((0, BLUE, "o"), (1, ORANGE, "^")):
        mask = y[train_idx] == klass
        ax1.scatter(x1[train_idx][mask], x2[train_idx][mask], c=color, marker=marker,
                    s=14, alpha=0.55, label=f"class {klass}")
    ax1.legend(fontsize=8)
    ax1.set(xlabel="x1 (signal)", ylabel="x2 (near-duplicate)",
            title=f"Training correlation = {corr:.3f}")
    positions = np.arange(4)
    values = [single.importances_mean[0], both.importances_mean[0],
              both.importances_mean[1], both.importances_mean[2]]
    errors = [single.importances_std[0], both.importances_std[0],
              both.importances_std[1], both.importances_std[2]]
    ax2.bar(positions, values, yerr=errors, color=[BLUE, TEAL, TEAL, ORANGE],
            error_kw={"capsize": 3})
    ax2.axhline(0, color=DARK, linewidth=0.8)
    ax2.set(xticks=positions, xticklabels=["x1 alone", "x1 with x2", "x2 with x1", "noise"],
            ylabel="held-out accuracy decrease", title="Individual permutation importance")
    ax2.tick_params(axis="x", labelrotation=20)
    fig.suptitle("Correlated substitutes can share predictive information", fontsize=14)
    path = _save(fig, "15_correlated_features.png")
    return _report(path, "What happens to permutation importance with a near-duplicate?",
                   f"Training correlation was {corr:.3f}; x1 importance was {single.importances_mean[0]:.3f} alone and {both.importances_mean[0]:.3f} beside x2.",
                   "The models have different feature sets; the comparison is illustrative, not a causal attribution.")


def plot_tree_vs_forest():
    X_train, X_test, y_train, y_test = make_boundary_data()
    tree = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)
    forest = RandomForestClassifier(n_estimators=120, max_features=1, min_samples_leaf=2,
                                    random_state=SEED, n_jobs=1).fit(X_train, y_train)
    xx, yy, grid = _mesh(X_train)
    probs = [model.predict_proba(grid)[:, 1].reshape(xx.shape) for model in (tree, forest)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, model, zz, title in zip(axes, (tree, forest), probs, ("One deep tree", "120-tree Random Forest")):
        image = ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap="RdBu_r", vmin=0, vmax=1)
        ax.contour(xx, yy, zz, levels=[0.5], colors=[DARK], linewidths=1)
        for klass, color, marker in ((0, BLUE, "o"), (1, ORANGE, "^")):
            mask = y_train == klass
            ax.scatter(X_train[mask, 0], X_train[mask, 1], c=color, marker=marker,
                       edgecolors="white", linewidths=0.3, s=16, alpha=0.8)
        ax.set(xlabel="feature 1", ylabel="feature 2", title=title)
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.13, top=0.83, wspace=0.12)
    colorbar_axis = fig.add_axes([0.89, 0.17, 0.02, 0.60])
    fig.colorbar(image, cax=colorbar_axis, label="predicted P(class 1)")
    fig.suptitle("Same synthetic training data: prediction structure after aggregation", fontsize=14)
    path = _save(fig, "16_tree_vs_forest.png", tight=False)
    disagreement = float(np.mean(tree.predict(X_test) != forest.predict(X_test)))
    return _report(path, "How does aggregation change a deep tree's prediction structure?",
                   f"The two fitted models disagreed on {disagreement:.1%} of held-out rows; compare their probability regions.",
                   "This one split does not prove that a forest is always simpler, smoother, or more accurate.")


STATIC = {
    "01_tree_instability": plot_tree_instability,
    "02_bootstrap_sampling": plot_bootstrap_sampling,
    "03_bootstrap_632": plot_bootstrap_632,
    "04_oob_mechanism": plot_oob_mechanism,
    "05_feature_randomness": plot_feature_randomness,
    "06_bagging_vs_random_forest": compare_bagging_random_forest,
    "07_correlation_variance": plot_variance_correlation_relationship,
    "09_n_estimators": experiment_n_estimators,
    "10_max_features": experiment_max_features,
    "11_regression_ensemble": plot_regression_ensemble,
    "13_mdi_importance": plot_mdi_importance,
    "14_mdi_vs_permutation": compare_feature_importance,
    "15_correlated_features": plot_correlated_feature_effect,
    "16_tree_vs_forest": plot_tree_vs_forest,
}
ANIMATIONS = {"08_forest_growth": animate_forest_growth}
INTERACTIVE = {"12_probability_surface": plot_probability_surface_3d}
ALL = {**STATIC, **ANIMATIONS, **INTERACTIVE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="generate every available view (default)")
    parser.add_argument("--static", action="store_true", help="generate PNG figures")
    parser.add_argument("--animations", action="store_true", help="generate the bounded GIF")
    parser.add_argument("--interactive", action="store_true", help="generate optional self-contained Plotly HTML")
    parser.add_argument("--only", choices=tuple(ALL), help="generate one named view")
    args = parser.parse_args(argv)
    if args.only:
        selected = {args.only: ALL[args.only]}
    elif args.all or not (args.static or args.animations or args.interactive):
        selected = ALL
    else:
        selected = {}
        if args.static:
            selected.update(STATIC)
        if args.animations:
            selected.update(ANIMATIONS)
        if args.interactive:
            selected.update(INTERACTIVE)
    generated = [path for function in selected.values() if (path := function()) is not None]
    print("\nGenerated visualizations:")
    for path in generated:
        print(f"- {path.relative_to(HERE) if path.is_relative_to(HERE) else path}")
    return generated


if __name__ == "__main__":
    main()
