"""Numerical experiments for the Day 35 visual lab; no rendering or UI state."""

from __future__ import annotations

from itertools import permutations
from numbers import Integral

import numpy as np
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance as library_permutation
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array

from from_scratch import pdp_ice, permutation_importance

FEATURES = ("signal", "signal_proxy", "context", "noise")
SEED = 35
DEFAULT_CONFIG = (900, 0.95, 1.6, 60, 6, 5)
METRICS = {
    "ROC-AUC": "roc_auc",
    "Average precision": "average_precision",
    "Negative log loss": "neg_log_loss",
}


def integer(value, minimum, maximum, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")
    return int(value)


def validate_config(config):
    if len(config) != 6:
        raise ValueError("Configuration must contain size, correlation, interaction, trees, depth, leaf.")
    n, rho, interaction, trees, depth, leaf = config
    integer(n, 300, 2000, "Sample size")
    integer(trees, 20, 120, "Trees")
    integer(depth, 2, 12, "Depth")
    integer(leaf, 1, 20, "Minimum leaf size")
    if not np.isfinite(rho) or not 0 <= rho <= 0.995:
        raise ValueError("Correlation must be finite and in [0, 0.995].")
    if not np.isfinite(interaction) or not 0 <= interaction <= 3:
        raise ValueError("Interaction must be finite and in [0, 3].")
    return tuple(config)


def build_lab(config=DEFAULT_CONFIG):
    n, rho, interaction, trees, depth, leaf = validate_config(config)
    rng = np.random.default_rng(SEED)
    signal, context, proxy_noise, noise = rng.normal(size=(4, n))
    X = np.column_stack([
        signal, rho * signal + np.sqrt(1 - rho**2) * proxy_noise, context, noise
    ])
    probability = expit(1.4 * signal + 0.9 * context + interaction * signal * context)
    # A shared uniform draw makes interaction comparisons paired at generation.
    y = (rng.random(n) < probability).astype(int)
    train_ids, test_ids = train_test_split(
        np.arange(n), test_size=0.25, stratify=y, random_state=SEED
    )
    model = RandomForestClassifier(
        n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
        max_features="sqrt", random_state=SEED, n_jobs=1
    ).fit(X[train_ids], y[train_ids])
    return {
        "config": tuple(config), "model": model, "X": X, "y": y,
        "generator_probability": probability, "train_ids": train_ids, "test_ids": test_ids,
        "X_train": X[train_ids], "y_train": y[train_ids],
        "X_test": X[test_ids], "y_test": y[test_ids],
    }


def positive_probability(model, rows):
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    if classes.ndim != 1 or len(matches) != 1:
        raise ValueError("Expected a single-output classifier containing class 1.")
    result = np.asarray(model.predict_proba(rows)[:, int(matches[0])], dtype=float)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Model outputs must be finite probabilities in [0, 1].")
    return result


def metric_score(y, probability, metric):
    if metric not in METRICS:
        raise ValueError("Unknown evaluation metric.")
    y, probability = np.asarray(y), np.asarray(probability, dtype=float)
    if y.ndim != 1 or probability.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Need aligned binary targets containing both classes.")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must lie in [0, 1].")
    if metric == "ROC-AUC":
        return float(roc_auc_score(y, probability))
    if metric == "Average precision":
        return float(average_precision_score(y, probability))
    return float(-log_loss(y, np.column_stack([1 - probability, probability]), labels=[0, 1]))


def gini(y):
    labels = np.asarray(y)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Gini expects a one-dimensional binary target.")
    if len(labels) == 0:
        return 0.0
    p = float(labels.mean())
    return 2 * p * (1 - p)


def split_statistics(X, y, feature, threshold):
    X = check_array(X, dtype=float)
    y = np.asarray(y)
    feature = integer(feature, 0, X.shape[1] - 1, "Feature")
    if y.shape != (len(X),) or not np.isfinite(threshold):
        raise ValueError("Need aligned targets and a finite threshold.")
    left = X[:, feature] <= threshold
    weighted = (left.mean() * gini(y[left]) + (~left).mean() * gini(y[~left]))
    return {
        "mask": left, "parent": gini(y), "left": gini(y[left]), "right": gini(y[~left]),
        "weighted_children": weighted, "gain": gini(y) - weighted,
        "counts": [np.bincount(y.astype(int), minlength=2),
                   np.bincount(y[left].astype(int), minlength=2),
                   np.bincount(y[~left].astype(int), minlength=2)],
    }


def importance_results(lab, metric="ROC-AUC", repeats=8):
    repeats = integer(repeats, 2, 30, "Repeats")
    own = permutation_importance(
        lab["model"], lab["X_test"], lab["y_test"],
        scoring=METRICS[metric], groups=[(0,), (1,), (2,), (3,), (0, 1)],
        repeats=repeats, seed=SEED
    )
    library = library_permutation(
        lab["model"], lab["X_test"], lab["y_test"], scoring=METRICS[metric],
        n_repeats=repeats, random_state=SEED, n_jobs=1
    )
    return {"scratch": own, "library": library, "mdi": lab["model"].feature_importances_}


def permutation_path(lab, columns=(0,), repeat=0, frames=13):
    """Visual transport only: intermediate interpolated rows are NOT permutations."""
    X = lab["X_test"]
    columns = tuple(integer(c, 0, 3, "Feature") for c in columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Need distinct columns for shuffling.")
    integer(repeat, 0, 29, "Repeat")
    integer(frames, 2, 30, "Frames")
    order = np.random.default_rng(SEED + repeat).permutation(len(X))
    shuffled = X.copy()
    shuffled[:, columns] = X[order][:, columns]
    alphas = np.linspace(0, 1, frames)
    states = [(1 - alpha) * X + alpha * shuffled for alpha in alphas]
    probabilities = [positive_probability(lab["model"], rows) for rows in states]
    return {"states": states, "probabilities": probabilities, "alphas": alphas,
            "order": order, "columns": columns}


def response_curves(lab, feature=0, rows=100, grid_size=25):
    feature = integer(feature, 0, 3, "Feature")
    integer(rows, 1, len(lab["X_test"]), "Reference rows")
    integer(grid_size, 3, 40, "Grid size")
    reference = lab["X_test"][:rows]
    grid = np.linspace(*np.quantile(reference[:, feature], [0.05, 0.95]), grid_size)
    result = pdp_ice(lambda X: positive_probability(lab["model"], X), reference, feature, grid)
    result["reference"] = reference
    return result


def replacement_rows(X, feature, value):
    X = check_array(X, dtype=float)
    feature = integer(feature, 0, X.shape[1] - 1, "Feature")
    if not np.isfinite(value):
        raise ValueError("Replacement value must be finite.")
    result = X.copy()
    result[:, feature] = value
    return result


def proxy_outside_band(rows, rho):
    """Known synthetic 95% conditional band, NOT an empirical density estimator."""
    rows = check_array(rows, dtype=float)
    if rows.shape[1] < 2 or not np.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("Need signal/proxy columns and correlation in [0, 1).")
    return np.abs(rows[:, 1] - rho * rows[:, 0]) > 1.96 * np.sqrt(1 - rho**2)


def surfaces(lab, row=0, grid_size=17, reference_count=60):
    integer(row, 0, len(lab["X_test"]) - 1, "Observation")
    integer(grid_size, 5, 25, "Surface grid size")
    integer(reference_count, 1, len(lab["X_test"]), "Surface reference count")
    reference = lab["X_test"][:reference_count]
    gx = np.linspace(*np.quantile(lab["X_train"][:, 0], [0.05, 0.95]), grid_size)
    gy = np.linspace(*np.quantile(lab["X_train"][:, 2], [0.05, 0.95]), grid_size)
    sx, cy = np.meshgrid(gx, gy)
    queries = np.tile(reference, (grid_size**2, 1))
    queries[:, 0] = np.repeat(sx.ravel(), reference_count)
    queries[:, 2] = np.repeat(cy.ravel(), reference_count)
    predictions = positive_probability(lab["model"], queries)
    pd = predictions.reshape(grid_size**2, reference_count).mean(axis=1).reshape(sx.shape)
    fixed_queries = np.tile(lab["X_test"][row], (grid_size**2, 1))
    fixed_queries[:, 0], fixed_queries[:, 2] = sx.ravel(), cy.ravel()
    fixed = positive_probability(lab["model"], fixed_queries).reshape(sx.shape)
    return {"signal": gx, "context": gy, "pdp": pd, "fixed": fixed,
            "row": lab["X_test"][row].copy(), "reference_count": reference_count}


def shapley_product(x1=2.0, x2=3.0):
    if not np.isfinite([x1, x2]).all():
        raise ValueError("Feature values must be finite.")
    x = np.array([x1, x2], dtype=float)
    coalition = {(): 0.0, (0,): 0.0, (1,): 0.0, (0, 1): float(x.prod())}
    paths, attribution = [], np.zeros(2)
    for order in permutations((0, 1)):
        members, previous, marginal = [], 0.0, np.zeros(2)
        steps = [{"members": (), "value": 0.0, "joining": None, "marginal": 0.0}]
        for feature in order:
            members.append(feature)
            value = coalition[tuple(sorted(members))]
            marginal[feature] = value - previous
            steps.append({"members": tuple(members), "value": value,
                          "joining": feature, "marginal": value - previous})
            previous = value
        attribution += marginal / 2
        paths.append({"order": order, "steps": steps, "contributions": marginal})
    np.testing.assert_allclose(attribution.sum(), x.prod(), rtol=0, atol=1e-12)
    return {"coalitions": coalition, "paths": paths, "values": attribution,
            "baseline": 0.0, "prediction": float(x.prod())}


def forest_shap(lab, background_size=50, sample_seed=35, explained_count=80):
    import shap

    integer(background_size, 10, min(300, len(lab["X_train"])), "Background size")
    integer(sample_seed, 0, 10000, "Background seed")
    integer(explained_count, 1, min(100, len(lab["X_test"])), "Explained rows")
    positions = np.random.default_rng(sample_seed).permutation(len(lab["X_train"]))[:background_size]
    background = lab["X_train"][positions]
    # Prevent the default 100-row cap from silently changing a larger background.
    masker = shap.maskers.Independent(background, max_samples=background_size)
    explainer = shap.TreeExplainer(
        lab["model"], data=masker, feature_perturbation="interventional", model_output="probability"
    )
    np.testing.assert_array_equal(explainer.data, background)
    rows = lab["X_test"][:explained_count]
    explanation = explainer(rows)
    values, bases = np.asarray(explanation.values), np.asarray(explanation.base_values)
    classes = np.asarray(lab["model"].classes_)
    positive = int(np.flatnonzero(classes == 1)[0])
    if values.shape != (len(rows), 4, len(classes)) or bases.shape != (len(rows), len(classes)):
        raise ValueError(f"Unsupported SHAP output shape: {values.shape}, {bases.shape}.")
    values, bases = values[:, :, positive], bases[:, positive]
    predictions = positive_probability(lab["model"], rows)
    error = float(np.max(np.abs(bases + values.sum(axis=1) - predictions)))
    np.testing.assert_allclose(bases + values.sum(axis=1), predictions, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        bases, positive_probability(lab["model"], background).mean(), rtol=0, atol=1e-6
    )
    return {"values": values, "baseline": bases, "predictions": predictions, "rows": rows,
            "background_ids": lab["train_ids"][positions], "error": error,
            "background_size": background_size, "sample_seed": sample_seed,
            "output": "P(y=1)", "convention": "interventional"}


def cardinality_experiment(seed=35):
    """Matched nuisance variables under random labels; split search can overfit."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.normal(size=600), rng.binomial(1, 0.5, 600)])
    y = rng.binomial(1, 0.5, 600)
    tr, te = train_test_split(np.arange(600), test_size=0.4, stratify=y, random_state=seed)
    model = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=1).fit(X[tr], y[tr])
    pi = library_permutation(model, X[te], y[te], scoring="roc_auc", n_repeats=8, random_state=seed)
    # Count admissible split positions before any target-driven selection.
    candidates = [len(np.unique(X[tr, j])) - 1 for j in range(2)]
    return {"mdi": model.feature_importances_, "pi": pi.importances_mean,
            "pi_std": pi.importances_std, "candidates": candidates,
            "auc": metric_score(y[te], positive_probability(model, X[te]), "ROC-AUC"),
            "configuration": {"synthetic": True, "seed": seed, "rows": 600,
                "split": "stratified 60/40", "features": ["continuous_noise", "binary_noise"],
                "labels": "independent random Bernoulli(0.5)", "model": model.get_params(),
                "scoring": "roc_auc", "permutation_repeats": 8}}


def leakage_experiment(lab):
    """Deliberately invalid: a post-outcome field leaks the label in both splits."""
    post_outcome = lab["y"].astype(float)
    X = np.column_stack([lab["X"], post_outcome])
    _, _, _, trees, depth, leaf = lab["config"]
    invalid = RandomForestClassifier(
        n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
        random_state=SEED, n_jobs=1, max_features="sqrt"
    ).fit(X[lab["train_ids"]], lab["y_train"])
    test = X[lab["test_ids"]]
    pi = library_permutation(invalid, test, lab["y_test"], scoring="roc_auc",
                             n_repeats=5, random_state=SEED)
    clean_probability = positive_probability(lab["model"], lab["X_test"])
    invalid_probability = positive_probability(invalid, test)
    return {"clean_auc": metric_score(lab["y_test"], clean_probability, "ROC-AUC"),
            "invalid_auc": metric_score(lab["y_test"], invalid_probability, "ROC-AUC"),
            "clean_mdi": lab["model"].feature_importances_, "invalid_mdi": invalid.feature_importances_,
            "invalid_pi": pi.importances_mean, "invalid_probability": invalid_probability,
            "configuration": {"post_outcome": "equal to y; unavailable at prediction time",
                "invalid_model": invalid.get_params(), "scoring": "roc_auc",
                "permutation_repeats": 5, "feature_names": [*FEATURES, "post_outcome"]}}


def tree_split_rows(lab):
    return lab["X_train"][:180, [0, 2]], lab["y_train"][:180]


def best_stump_threshold(X, y, feature):
    stump = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(X[:, [feature]], y)
    return float(stump.tree_.threshold[0])
