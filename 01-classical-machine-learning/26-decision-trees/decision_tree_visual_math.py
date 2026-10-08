"""Deterministic data and numerical calculations for the Decision Tree visual lab."""

from collections import deque

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

SEED = 26


def _probabilities(p):
    p = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    return p


def gini_binary(p):
    p = _probabilities(p)
    return 2 * p * (1 - p)


def entropy_binary(p):
    p = _probabilities(p)
    result = np.zeros_like(p)
    interior = (p > 0) & (p < 1)
    q = p[interior]
    result[interior] = -q * np.log2(q) - (1 - q) * np.log2(1 - q)
    return result


def class_impurity(y, criterion="gini"):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0 or not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must be a nonempty binary label vector")
    if criterion not in ("gini", "entropy"):
        raise ValueError("criterion must be gini or entropy")
    p = np.mean(y == 1)
    return float(gini_binary(p) if criterion == "gini" else entropy_binary(p))


def toy_data():
    return np.arange(1, 9, dtype=float), np.array([0, 0, 1, 0, 1, 1, 1, 1])


def evaluate_thresholds(x, y, criterion="gini"):
    """Compute size-weighted child impurity for every distinct midpoint."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if x.ndim != 1 or len(x) != len(y) or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite vector with one value per label")
    parent = class_impurity(y, criterion)
    values = np.unique(x)
    if len(values) < 2:
        raise ValueError("at least two distinct feature values are required")
    rows = []
    for lower, upper in zip(values[:-1], values[1:]):
        threshold = lower / 2 + upper / 2
        left = x <= threshold
        n_left, n_right = int(left.sum()), int((~left).sum())
        if not n_left or not n_right:
            continue
        left_impurity = class_impurity(y[left], criterion)
        right_impurity = class_impurity(y[~left], criterion)
        weighted = (n_left * left_impurity + n_right * right_impurity) / len(y)
        rows.append({
            "threshold": float(threshold), "parent_impurity": parent,
            "n_left": n_left, "n_right": n_right,
            "left_impurity": left_impurity, "right_impurity": right_impurity,
            "weighted_impurity": weighted, "gain": float(parent - weighted),
        })
    return rows


def create_dataset(n_samples=450, noise=0.28):
    """One fixed train/validation split for all noisy-moons comparisons."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 100:
        raise ValueError("n_samples must be an integer of at least 100")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    # Plot limits use training features only; validation labels never guide fitting.
    lower = X_train.min(axis=0) - 0.4
    upper = X_train.max(axis=0) + 0.4
    return {
        "X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
        "bounds": (float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])),
        "configuration": {
            "dataset": "synthetic sklearn.make_moons", "n_samples": n_samples,
            "noise": float(noise), "seed": SEED, "validation_fraction": 0.3,
            "n_training": len(y_train), "n_validation": len(y_val),
            "criterion": "gini", "metric": "accuracy",
        },
    }


def fit_tree(data, **parameters):
    model = DecisionTreeClassifier(random_state=SEED, **parameters)
    return model.fit(data["X_train"], data["y_train"])


def model_scores(model, data):
    return {
        "depth": int(model.get_depth()), "leaves": int(model.get_n_leaves()),
        "training_accuracy": float(model.score(data["X_train"], data["y_train"])),
        "validation_accuracy": float(model.score(data["X_val"], data["y_val"])),
    }


def depth_sweep(data):
    rows = []
    for depth in range(1, 21):
        model = fit_tree(data, max_depth=depth)
        rows.append({"max_depth": depth, **model_scores(model, data)})
    return rows


def leaf_sweep(data):
    return [
        {"min_samples_leaf": size, **model_scores(fit_tree(data, min_samples_leaf=size), data)}
        for size in (1, 2, 5, 10, 20, 30)
    ]


def pruning_sweep(data, max_candidates=24):
    """The path uses only training data, including its final root-only tree."""
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 2:
        raise ValueError("max_candidates must be an integer of at least two")
    path = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(
        data["X_train"], data["y_train"]
    )
    alphas = np.unique(path.ccp_alphas)
    if len(alphas) > max_candidates:
        indices = np.unique(np.linspace(0, len(alphas) - 1, max_candidates).astype(int))
        alphas = alphas[indices]
    rows = [
        {"ccp_alpha": float(alpha), **model_scores(fit_tree(data, ccp_alpha=float(alpha)), data)}
        for alpha in alphas
    ]
    return rows


def make_mesh(bounds, resolution=160):
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 2:
        raise ValueError("resolution must be an integer of at least two")
    xmin, xmax, ymin, ymax = bounds
    if not np.all(np.isfinite(bounds)) or xmin >= xmax or ymin >= ymax:
        raise ValueError("bounds must be finite and increasing on both axes")
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, resolution), np.linspace(ymin, ymax, resolution))
    return xx, yy, np.column_stack((xx.ravel(), yy.ravel()))


def node_geometry(model, bounds):
    """Return every fitted node in breadth-first order with its region bounds.

    Each split segment stops at its parent region edges, rather than extending
    across unrelated branches of feature space.
    """
    if model.n_features_in_ != 2:
        raise ValueError("node geometry requires a fitted two-feature tree")
    make_mesh(bounds, resolution=2)
    tree = model.tree_
    queue = deque([(0, tuple(bounds))])
    nodes = []
    while queue:
        node_id, box = queue.popleft()
        feature = int(tree.feature[node_id])
        info = {"node": node_id, "bounds": box, "leaf": feature < 0}
        if feature >= 0:
            if feature not in (0, 1):
                raise ValueError("only two feature axes are supported")
            threshold = float(tree.threshold[node_id])
            xmin, xmax, ymin, ymax = box
            info.update(feature=feature, threshold=threshold)
            if feature == 0:
                left = (xmin, threshold, ymin, ymax)
                right = (threshold, xmax, ymin, ymax)
                info["segment"] = ((threshold, ymin), (threshold, ymax))
            else:
                left = (xmin, xmax, ymin, threshold)
                right = (xmin, xmax, threshold, ymax)
                info["segment"] = ((xmin, threshold), (xmax, threshold))
            queue.append((int(tree.children_left[node_id]), left))
            queue.append((int(tree.children_right[node_id]), right))
        nodes.append(info)
    return nodes


def node_probability(model, node_id):
    values = model.tree_.value[node_id].reshape(-1)
    class_one = int(np.flatnonzero(model.classes_ == 1)[0])
    return float(values[class_one] / values.sum())


def frontier_regions(model, bounds, introduced_splits):
    """Regions after revealing some ancestor-first splits of one fitted tree."""
    tree = model.tree_
    geometries = {node["node"]: node for node in node_geometry(model, bounds)}
    queue = [0]
    regions = []
    while queue:
        node_id = queue.pop()
        node = geometries[node_id]
        if node["leaf"] or node_id not in introduced_splits:
            regions.append({**node, "probability": node_probability(model, node_id)})
        else:
            queue.extend([int(tree.children_right[node_id]), int(tree.children_left[node_id])])
    return regions


def interaction_dataset():
    """Noise-free conditional rule: X2 matters only when X1 > 0."""
    axis = np.linspace(-1, 1, 20)
    xx, yy = np.meshgrid(axis, axis)
    X = np.column_stack((xx.ravel(), yy.ravel()))
    y = ((X[:, 0] > 0) & (X[:, 1] > -0.2)).astype(int)
    return X, y


def bootstrap_trees(data, count=6):
    rng = np.random.default_rng(SEED)
    models = []
    for _ in range(count):
        indices = rng.integers(0, len(data["y_train"]), size=len(data["y_train"]))
        model = DecisionTreeClassifier(max_depth=5, random_state=SEED)
        model.fit(data["X_train"][indices], data["y_train"][indices])
        models.append(model)
    return models
