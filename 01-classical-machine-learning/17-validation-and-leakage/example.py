"""Deterministic synthetic demonstrations of validation information boundaries."""
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit,
    cross_val_score, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def classification_pipeline():
    """Return fresh preprocessing and classifier instances."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def model_selection_demo():
    """Tune on development folds, then score one frozen test set."""
    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=8, n_redundant=4,
        weights=[0.85, 0.15], random_state=SEED,
    )
    dev, test = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED,
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        classification_pipeline(), {"model__C": [0.01, 0.1, 1.0, 10.0]},
        cv=cv, scoring="roc_auc", n_jobs=1, error_score="raise",
    )
    search.fit(X[dev], y[dev])
    scores = np.array([
        search.cv_results_[f"split{k}_test_score"][search.best_index_] for k in range(5)
    ])
    # GridSearchCV refits the chosen pipeline on all development rows.
    test_auc = roc_auc_score(y[test], search.predict_proba(X[test])[:, 1])
    return {
        "development_rows": len(dev), "test_rows": len(test),
        "development_positive_rate": float(y[dev].mean()),
        "test_positive_rate": float(y[test].mean()),
        "selected_C": search.best_params_["model__C"],
        "selected_cv_auc": scores.tolist(),
        "cv_mean": float(scores.mean()), "cv_sample_sd": float(scores.std(ddof=1)),
        "final_test_auc": float(test_auc),
    }


def feature_leakage_demo():
    """Contrast intentional label leakage with fold-local selection on noise."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(300, 2000))
    y = rng.integers(0, 2, size=300)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    safe = Pipeline([
        ("select", SelectKBest(f_classif, k=20)),
        ("classifier", classification_pipeline()),
    ])
    # Invalid control: held-out labels influence which columns are selected.
    leaked_X = SelectKBest(f_classif, k=20).fit_transform(X, y)
    return {
        "invalid_global_selection_auc": cross_val_score(
            classification_pipeline(), leaked_X, y, cv=cv,
            scoring="roc_auc", error_score="raise",
        ).tolist(),
        "fold_local_selection_auc": cross_val_score(
            safe, X, y, cv=cv, scoring="roc_auc", error_score="raise",
        ).tolist(),
    }


def group_demo():
    """Contrast known-entity interpolation with unseen-entity validation."""
    rng = np.random.default_rng(SEED)
    groups = np.repeat(np.arange(100), 10)
    signatures = rng.normal(size=(100, 8))
    labels = rng.integers(0, 2, size=100)
    X = signatures[groups] + rng.normal(scale=0.01, size=(1000, 8))
    y = labels[groups]
    random_folds = list(KFold(5, shuffle=True, random_state=SEED).split(X))
    group_folds = list(GroupKFold(5).split(X, y, groups))
    result = {}
    for name, folds in (("random", random_folds), ("group", group_folds)):
        result[f"{name}_accuracy"] = cross_val_score(
            KNeighborsClassifier(n_neighbors=1), X, y, cv=folds,
            scoring="accuracy", error_score="raise",
        ).tolist()
        result[f"{name}_overlapping_groups"] = [
            len(np.intersect1d(groups[train], groups[valid])) for train, valid in folds
        ]
    return result


def temporal_folds():
    """Expanding folds over 30 equally spaced development timestamps."""
    # Toy labels arrive two steps later; all must arrive before validation starts.
    return list(TimeSeriesSplit(n_splits=3, test_size=5, gap=2).split(np.arange(30)))


def main():
    print(f"Synthetic data; seed={SEED}; numpy={np.__version__}; sklearn={sklearn.__version__}")
    for name, result in (
        ("Development-only model selection", model_selection_demo()),
        ("Intentional feature-selection leakage", feature_leakage_demo()),
        ("Unseen-customer validation", group_demo()),
    ):
        print(f"\n{name}")
        for key, value in result.items():
            print(f"  {key}: {value}")
    print("\nTemporal development folds (rows 30..34 reserved as future test)")
    for train, valid in temporal_folds():
        print(f"  train={train[0]}..{train[-1]}, validation={valid[0]}..{valid[-1]}")
    print("  Final refit: 0..27; gap: 28..29; test: 30..34 (no model scored)")
    print("Interpretation candidates require author review; these are not benchmarks.")


if __name__ == "__main__":
    main()
