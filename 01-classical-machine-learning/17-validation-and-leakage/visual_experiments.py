"""Numerical experiments for the Visual Validation Lab; no plotting or I/O."""
from numbers import Integral

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, roc_auc_score, roc_curve
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def checked_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return int(seed)


def assert_partition(n, *parts):
    """Reject overlapping, missing, duplicated or out-of-range row indices."""
    if not parts or any(np.asarray(p).ndim != 1 or not len(p) for p in parts):
        raise ValueError("partitions must be nonempty one-dimensional arrays")
    indices = np.concatenate(parts)
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("partition indices must be integers")
    if not np.array_equal(np.sort(indices), np.arange(n)):
        raise ValueError("partitions must cover each row exactly once")


def split_three(n, seed=42):
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 10:
        raise ValueError("n must be an integer >= 10")
    seed = checked_seed(seed)
    dev, test = train_test_split(np.arange(n), test_size=0.2, random_state=seed)
    train, valid = train_test_split(dev, test_size=0.25, random_state=seed)
    assert_partition(n, train, valid, test)
    return train, valid, test


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def classification_data(seed=42, n=600):
    return make_classification(
        n_samples=n, n_features=10, n_informative=5, n_redundant=2,
        weights=[0.8, 0.2], class_sep=0.8, random_state=checked_seed(seed),
    )


def fold_scores(X, y, folds, model, metric="auc"):
    scores = []
    for train, valid in folds:
        if np.intersect1d(train, valid).size:
            raise ValueError("training and validation rows overlap")
        fitted = clone(model).fit(X[train], y[train])
        if metric == "auc":
            if np.unique(y[train]).size != 2 or np.unique(y[valid]).size != 2:
                raise ValueError("AUC requires both classes in every fold")
            score = roc_auc_score(y[valid], fitted.predict_proba(X[valid])[:, 1])
        elif metric == "accuracy":
            score = accuracy_score(y[valid], fitted.predict(X[valid]))
        elif metric == "mse":
            score = mean_squared_error(y[valid], fitted.predict(X[valid]))
        else:
            raise ValueError("metric must be auc, accuracy or mse")
        scores.append(float(score))
    return np.array(scores)


def kfold_experiment(seed=42):
    X, y = classification_data(seed)
    folds = list(KFold(5, shuffle=True, random_state=seed).split(X))
    for train, valid in folds:
        assert_partition(len(y), train, valid)
    return {"X": X, "y": y, "folds": folds,
            "scores": fold_scores(X, y, folds, classifier())}


def stratification_experiment(seed=42):
    rng = np.random.default_rng(checked_seed(seed))
    y = rng.permutation(np.r_[np.zeros(180, dtype=int), np.ones(20, dtype=int)])
    folds = [
        list(KFold(5, shuffle=True, random_state=seed).split(y)),
        list(StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y)),
    ]
    rates = np.array([[y[v].mean() for _, v in method] for method in folds])
    return {"y": y, "rates": rates, "global_rate": float(y.mean())}


def group_experiment(seed=42):
    rng = np.random.default_rng(checked_seed(seed))
    groups = np.repeat(np.arange(100), 20)
    centers = rng.normal(size=(100, 8))
    labels = rng.integers(0, 2, size=100)
    X = centers[groups] + rng.normal(0, 0.06, (len(groups), 8))
    y = labels[groups]
    folds = {
        "Random KFold": list(KFold(5, shuffle=True, random_state=seed).split(X)),
        "GroupKFold": list(GroupKFold(5).split(X, y, groups)),
    }
    scores, overlap, fractions = {}, {}, {}
    for name, method in folds.items():
        scores[name] = fold_scores(X, y, method, KNeighborsClassifier(1), "accuracy")
        overlap[name] = [
            np.intersect1d(groups[t], groups[v]).size for t, v in method
        ]
        fractions[name] = np.array([
            [np.mean(np.isin(np.flatnonzero(groups == g), v)) for _, v in method]
            for g in range(100)
        ])
    assert not any(overlap["GroupKFold"])
    return {"X": X, "y": y, "groups": groups, "folds": folds,
            "scores": scores, "overlap": overlap, "fractions": fractions}


def drift_data(seed=42):
    rng = np.random.default_rng(checked_seed(seed))
    t = np.arange(600)
    x = rng.normal(size=600)
    slope = 1 - 2 * t / (len(t) - 1)
    y = slope * x + rng.normal(0, 0.25, size=len(t))
    return {"t": t, "x": x, "y": y, "slope": slope}


def drift_experiment(seed=42):
    data = drift_data(seed)
    X, y = data["x"][:, None], data["y"]
    dev = np.arange(480)
    test = np.arange(480, 600)
    methods = {
        "Random": list(KFold(4, shuffle=True, random_state=seed).split(dev)),
        "Expanding": list(TimeSeriesSplit(4, test_size=60).split(dev)),
        "Rolling": list(TimeSeriesSplit(4, test_size=60, max_train_size=120).split(dev)),
    }
    scores = {}
    coefficients = {}
    for name, folds in methods.items():
        scores[name] = fold_scores(X, y, folds, LinearRegression(), "mse")
        coefficients[name] = [
            float(LinearRegression().fit(X[t], y[t]).coef_[0]) for t, _ in folds
        ]
        if name != "Random":
            for train, valid in folds:
                assert train.max() < valid.min() < test.min()
    final = {}
    for name, train in (("Expanding", dev), ("Rolling", dev[-120:])):
        fitted = LinearRegression().fit(X[train], y[train])
        final[name] = float(mean_squared_error(y[test], fitted.predict(X[test])))
    return {**data, "folds": methods, "scores": scores, "coefficients": coefficients,
            "test_mse": final}


def preprocessing_experiment(seed=42):
    rng = np.random.default_rng(checked_seed(seed))
    X = rng.normal(size=(500, 2))
    X[-100:, 0] += 4
    y = (X[:, 0] + X[:, 1] + rng.normal(size=500) > 0).astype(int)
    folds = list(TimeSeriesSplit(3, test_size=100).split(X))
    global_scaler = StandardScaler().fit(X)
    local_scaler = StandardScaler().fit(X[folds[0][0]])
    invalid = fold_scores(
        global_scaler.transform(X), y, folds, LogisticRegression(C=0.1, max_iter=1000)
    )
    safe = fold_scores(
        X, y, folds,
        make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
    )
    # A separate noise-only control isolates supervised feature-selection leakage.
    noise = rng.normal(size=(240, 500))
    labels = rng.integers(0, 2, size=240)
    selection_folds = list(StratifiedKFold(4, shuffle=True, random_state=seed).split(noise, labels))
    leaked = SelectKBest(f_classif, k=15).fit_transform(noise, labels)
    selection = {
        "Global selector": fold_scores(leaked, labels, selection_folds, classifier()),
        "Fold-local selector": fold_scores(
            noise, labels, selection_folds,
            make_pipeline(SelectKBest(f_classif, k=15), classifier()),
        ),
    }
    return {"X": X, "y": y, "folds": folds, "global": global_scaler, "local": local_scaler,
            "global_auc": invalid, "local_auc": safe, "selection": selection}


def target_experiment(seed=42):
    X, y = classification_data(seed, n=900)
    rng = np.random.default_rng(checked_seed(seed))
    after_target = y + rng.normal(0, 0.3, size=len(y))
    train, valid = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed
    )
    assert_partition(len(y), train, valid)
    curves, aucs, matrices = {}, {}, {}
    for name, features in (("Available features", X), ("With post-outcome feature", np.c_[X, after_target])):
        model = classifier().fit(features[train], y[train])
        probabilities = model.predict_proba(features[valid])[:, 1]
        curves[name] = roc_curve(y[valid], probabilities)[:2]
        aucs[name] = float(roc_auc_score(y[valid], probabilities))
        matrices[name] = confusion_matrix(y[valid], probabilities >= 0.5)
    return {"y": y, "leaked": after_target, "curves": curves, "aucs": aucs, "matrices": matrices}


def purchase_experiment(seed=42):
    rng = np.random.default_rng(checked_seed(seed))
    monthly = rng.lognormal(mean=4, sigma=0.35, size=12) + np.arange(12) * 3
    return {"monthly": monthly, "asof_march": float(monthly[:3].mean()),
            "full_year": float(monthly.mean())}


def running_winners(validation_scores):
    scores = np.asarray(validation_scores, dtype=float)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("validation_scores must be a finite nonempty vector")
    best, winners = 0, []
    for i in range(len(scores)):
        if scores[i] > scores[best]:
            best = i
        winners.append(best)
    return np.array(winners)


def overfitting_experiment(seed=42):
    rng = np.random.default_rng(checked_seed(seed))
    # Hypothetical null classifiers: independent scores, no fitted signal.
    y_valid = np.tile([0, 1], 40)
    y_test = np.tile([0, 1], 1000)
    valid = rng.normal(size=(200, len(y_valid)))
    test = rng.normal(size=(200, len(y_test)))
    valid_auc = np.array([roc_auc_score(y_valid, score) for score in valid])
    test_auc = np.array([roc_auc_score(y_test, score) for score in test])
    winners = running_winners(valid_auc)
    return {"validation": valid_auc, "test": test_auc, "winners": winners,
            "best_validation": valid_auc[winners], "winner_test": test_auc[winners]}


def nested_experiment(seed=42):
    X, y = classification_data(seed, n=360)
    outer = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y))
    rows, inner_global = [], []
    for train, valid in outer:
        inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X[train], y[train]))
        mapped = [(train[t], train[v]) for t, v in inner]
        for t, v in mapped:
            assert not np.intersect1d(np.r_[t, v], valid).size
            assert_partition(len(train), np.flatnonzero(np.isin(train, t)), np.flatnonzero(np.isin(train, v)))
        search = GridSearchCV(
            classifier(), {"logisticregression__C": [0.01, 0.1, 1, 10]},
            cv=inner, scoring="roc_auc", n_jobs=1, error_score="raise",
        ).fit(X[train], y[train])
        rows.append({
            "C": float(search.best_params_["logisticregression__C"]),
            "inner_best_auc": float(search.best_score_),
            "outer_auc": float(roc_auc_score(y[valid], search.predict_proba(X[valid])[:, 1])),
        })
        inner_global.append(mapped)
    return {"outer": outer, "inner": inner_global, "rows": rows, "n": len(y)}


def holdout_experiment(seed=42):
    X, y = classification_data(seed, n=400)
    holdout, cv_means = [], []
    for repetition in range(16):
        split_seed = (checked_seed(seed) + repetition) % (2**32)
        train, valid = train_test_split(
            np.arange(len(y)), test_size=0.2, stratify=y, random_state=split_seed
        )
        holdout.append(fold_scores(X, y, [(train, valid)], classifier())[0])
        folds = list(StratifiedKFold(5, shuffle=True, random_state=split_seed).split(X, y))
        cv_means.append(fold_scores(X, y, folds, classifier()).mean())
    return {"holdout": np.array(holdout), "cv": np.array(cv_means)}
