"""Compare training interventions and validation-selected decision policies."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, fbeta_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from from_scratch import resample_training, select_cost_threshold


SEED = 42
C_FP, C_FN = 1.0, 10.0
STRATEGIES = ("baseline", "weighted", "random_over", "random_under", "smote")
DATA_CONFIG = dict(
    n_samples=6000, n_features=12, n_informative=5, n_redundant=2,
    n_clusters_per_class=2, weights=[0.97, 0.03], class_sep=1.2,
    flip_y=0.01, random_state=SEED,
)


def class_counts(y):
    return np.bincount(y, minlength=2).astype(int).tolist()


def fit_strategy(X_train, y_train, strategy, seed=SEED):
    """Fit scaling on original training rows, then resample and fit the model."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    if (
        X_train.ndim != 2 or 0 in X_train.shape
        or not np.isfinite(X_train).all()
        or y_train.ndim != 1 or len(y_train) != len(X_train)
        or not np.isin(y_train, [0, 1]).all()
        or len(np.unique(y_train)) != 2
    ):
        raise ValueError("training requires finite numerical rows and both binary classes")
    y_train = y_train.astype(int)
    scaler = StandardScaler().fit(X_train)
    X_fit = scaler.transform(X_train)
    y_fit = np.asarray(y_train).copy()
    if strategy in {"random_over", "random_under", "smote"}:
        X_fit, y_fit = resample_training(X_fit, y_fit, strategy, seed=seed)
    model = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=2000, random_state=seed,
        class_weight="balanced" if strategy == "weighted" else None,
    ).fit(X_fit, y_fit)
    return scaler, model, class_counts(y_fit)


def evaluate(y, probabilities, threshold):
    y = np.asarray(y)
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        y.ndim != 1 or not np.isin(y, [0, 1]).all()
        or len(np.unique(y)) != 2
        or probabilities.shape != y.shape
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("evaluation requires both binary classes and matching probabilities in [0, 1]")
    if np.ndim(threshold) != 0 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite nonnegative scalar")
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    cost = C_FP * fp + C_FN * fn
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f2": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
        "alert_rate": float(predicted.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost": float(cost), "cost_per_row": float(cost / len(y)),
        # Ranking and probability metrics use original scores, never labels.
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "brier": float(brier_score_loss(y, probabilities)),
    }


def run_experiment():
    X, y = make_classification(**DATA_CONFIG)
    train, holdout = train_test_split(
        np.arange(len(y)), test_size=0.4, stratify=y, random_state=SEED,
    )
    validation, test = train_test_split(
        holdout, test_size=0.5, stratify=y[holdout], random_state=SEED,
    )
    fitted, selections, fit_counts = {}, {}, {}
    for strategy in STRATEGIES:
        scaler, model, counts = fit_strategy(X[train], y[train], strategy)
        validation_p = model.predict_proba(scaler.transform(X[validation]))[:, 1]
        selections[strategy] = select_cost_threshold(
            y[validation], validation_p, C_FP, C_FN
        )
        fitted[strategy] = (scaler, model)
        fit_counts[strategy] = counts

    # All models and policies are frozen before final test scoring. No refit.
    results = [{
        "strategy": "always_negative", "policy": "fixed",
        **evaluate(y[test], np.zeros(len(test)), 0.5),
    }]
    for strategy, (scaler, model) in fitted.items():
        test_p = model.predict_proba(scaler.transform(X[test]))[:, 1]
        for policy, threshold in (
            ("fixed_0.5", 0.5),
            ("validation_cost", selections[strategy]["threshold"]),
        ):
            results.append({
                "strategy": strategy, "policy": policy,
                **evaluate(y[test], test_p, threshold),
            })
    return {
        "hypothesis": "Training interventions and cost thresholding may change minority recall, alert volume, and error cost differently; balancing need not improve ranking.",
        "configuration": {
            "data": "Entirely synthetic sklearn.make_classification data; no public dataset",
            "generator": DATA_CONFIG,
            "split": "stratified 60/20/20; both split seeds 42",
            "costs": {"false_positive": C_FP, "false_negative": C_FN},
            "model": "LogisticRegression(C=1, solver=lbfgs, max_iter=2000)",
            "preprocessing": "StandardScaler fit before resampling on original training rows only",
            "resampling": "1:1 training balance; seed 42; educational numerical SMOTE k=5",
            "threshold_selection": "all validation score decision sets; score >= threshold; ties choose highest threshold",
            "versions": {
                "python": platform.python_version(), "numpy": np.__version__,
                "scikit-learn": sklearn.__version__,
            },
        },
        "split_counts_0_1": {
            "train": class_counts(y[train]),
            "validation": class_counts(y[validation]), "test": class_counts(y[test]),
        },
        "fit_counts_0_1": fit_counts,
        "validation_selection": selections,
        "test_results": results,
        "interpretation_candidate": "Compare fixed and selected policies within each model, then compare ranking and Brier scores across training strategies. No strategy is assumed superior.",
        "interpretation_status": "Pending author review",
        "limitations": [
            "One synthetic seed and split; no uncertainty interval or production validation.",
            "Fixed C and changed sample/weight totals do not isolate regularization effects.",
            "Artificial constant costs; no capacity constraint or calibration fitting.",
            "Brier score mixes calibration and discrimination; it is not a calibration-only metric.",
            "SMOTE uses numerical interpolation with dense minority distances; no categorical or manifold guarantees.",
            "Test comparisons are descriptive; no test-based model winner or threshold selection.",
        ],
    }


def main():
    report = run_experiment()
    print("Synthetic data: class counts [negative, positive]")
    print("Original splits:", report["split_counts_0_1"])
    print("Training after intervention:", report["fit_counts_0_1"])
    print("Test policies (cost: FP + 10*FN; AP = average precision):")
    print(
        f"{'strategy':<16} {'policy':<16} {'t':>7} {'AP':>7} {'AUC':>7} "
        f"{'prec':>7} {'recall':>7} {'alerts':>7} {'cost':>7} {'Brier':>7}"
    )
    for row in report["test_results"]:
        print(
            f"{row['strategy']:<16} {row['policy']:<16} "
            f"{row['threshold']:7.4f} {row['average_precision']:7.4f} "
            f"{row['roc_auc']:7.4f} {row['precision']:7.4f} "
            f"{row['recall']:7.4f} {row['alert_rate']:7.4f} "
            f"{row['cost']:7.0f} {row['brier']:7.4f}"
        )
    output = Path(__file__).resolve().parent / "outputs" / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("Report:", output)
    print("Interpretation candidates are pending author review.")


if __name__ == "__main__":
    main()
