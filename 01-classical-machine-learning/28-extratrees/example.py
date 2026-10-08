"""Compare threshold randomization and bootstrap on shared synthetic data."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


SEED = 42
DATA_CONFIG = dict(
    n_samples=2500,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=None,
    class_sep=1.0,
    flip_y=0.05,
    shuffle=True,
    random_state=SEED,
)
MODEL_CONFIG = dict(
    n_estimators=120,
    criterion="gini",
    max_features="sqrt",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=2,
    n_jobs=1,
    random_state=SEED,
)


def pairwise_disagreement(predictions) -> float:
    """Mean fraction of differing labels over all unordered pairs of trees.

    Input shape is (n_trees, n_rows), with binary labels 0/1. At one row,
    k positive votes produce k * (M - k) disagreeing unordered pairs.
    """
    predictions = np.asarray(predictions)
    if (
        predictions.ndim != 2
        or predictions.shape[0] < 2
        or predictions.shape[1] == 0
        or not np.isin(predictions, [0, 1]).all()
    ):
        raise ValueError("Expected binary predictions with shape (at least 2 trees, at least 1 row).")
    n_trees = predictions.shape[0]
    positive_votes = predictions.astype(float).sum(axis=0)
    n_pairs = n_trees * (n_trees - 1) / 2
    return float(np.mean(positive_votes * (n_trees - positive_votes) / n_pairs))


def build_models(**overrides) -> dict:
    config = MODEL_CONFIG | overrides
    return {
        f"{name}_bootstrap_{str(bootstrap).lower()}": model_class(bootstrap=bootstrap, **config)
        for name, model_class in [("RF", RandomForestClassifier), ("ET", ExtraTreesClassifier)]
        for bootstrap in [False, True]
    }


def run_comparison() -> dict:
    X, y = make_classification(**DATA_CONFIG)
    # One shared stratified split of independent synthetic rows; no learned preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    results = []
    for name, model in build_models().items():
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start
        probabilities = model.predict_proba(X_test)
        predictions = model.classes_[probabilities.argmax(axis=1)]
        tree_predictions = np.stack([tree.predict(X_test) for tree in model.estimators_])
        results.append(dict(
            model=name,
            accuracy=float(accuracy_score(y_test, predictions)),
            roc_auc=float(roc_auc_score(y_test, probabilities[:, 1])),
            log_loss=float(log_loss(y_test, probabilities, labels=[0, 1])),
            mean_tree_accuracy=float(np.mean(tree_predictions == y_test)),
            disagreement=pairwise_disagreement(tree_predictions),
            fit_seconds=fit_seconds,
        ))
    return dict(
        hypothesis="Random thresholds may increase tree disagreement; held-out performance need not improve.",
        data_source="Synthetic sklearn.datasets.make_classification; no external dataset.",
        data_config=DATA_CONFIG,
        model_config=MODEL_CONFIG,
        split=dict(test_size=0.25, stratify=True, random_state=SEED,
                   train_rows=len(y_train), test_rows=len(y_test),
                   train_class_counts=np.bincount(y_train).tolist(),
                   test_class_counts=np.bincount(y_test).tolist()),
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         scikit_learn=sklearn.__version__, platform=platform.platform()),
        results=results,
        interpretation_status="Pending author review; disagreement is not training-set prediction variance or error correlation.",
        limitation="One synthetic dataset, split and model seed; untuned settings and single-run fit timings.",
    )


def main() -> None:
    record = run_comparison()
    print("Synthetic held-out comparison; higher accuracy/AUC, lower log loss are better.")
    print(f"{'model':<24} {'accuracy':>9} {'ROC-AUC':>9} {'log loss':>9} {'tree acc':>9} {'disagree':>9} {'fit (s)':>9}")
    for row in record["results"]:
        print(f"{row['model']:<24} {row['accuracy']:9.4f} {row['roc_auc']:9.4f} "
              f"{row['log_loss']:9.4f} {row['mean_tree_accuracy']:9.4f} "
              f"{row['disagreement']:9.4f} {row['fit_seconds']:9.4f}")
    output = Path(__file__).resolve().parent / "outputs" / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Run record: {output}")
    print(record["interpretation_status"])


if __name__ == "__main__":
    main()
