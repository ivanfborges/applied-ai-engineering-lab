"""Tune C and gamma on synthetic moons with fold-local scaling."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def tune_rbf(X_train, y_train, *, C_values=(0.1, 1.0, 10.0, 100.0),
             gamma_values=(0.01, 0.1, 1.0, 10.0), folds=5):
    """Select on training folds only; the caller reserves the test rows."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf"))])
    search = GridSearchCV(
        pipeline, {"svm__C": C_values, "svm__gamma": gamma_values},
        scoring="accuracy", cv=StratifiedKFold(folds, shuffle=True, random_state=42),
        return_train_score=True, n_jobs=1, error_score="raise")
    return search.fit(X_train, y_train)


def main():
    X, y = make_moons(n_samples=400, noise=0.20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    search = tune_rbf(X_train, y_train)
    selected = search.best_estimator_
    # A prespecified linear baseline supplies context on this nonlinear task.
    linear = Pipeline([("scaler", StandardScaler()),
                       ("svm", SVC(kernel="linear", C=1.0))]).fit(X_train, y_train)
    results = search.cv_results_
    rows = [{
        "C": float(params["svm__C"]), "gamma": float(params["svm__gamma"]),
        "train_accuracy": float(results["mean_train_score"][i]),
        "cv_accuracy": float(results["mean_test_score"][i]),
        "cv_std": float(results["std_test_score"][i]),
    } for i, params in enumerate(results["params"])]
    prediction = selected.predict(X_test)
    svm = selected.named_steps["svm"]
    report = {
        "hypothesis": "Joint C/gamma validation can find useful nonlinear capacity on noisy synthetic moons; training accuracy alone is insufficient for selection.",
        "configuration": {
            "generator": "sklearn.datasets.make_moons (synthetic)", "n_samples": 400,
            "noise": 0.20, "seed": 42, "split_seed": 42,
            "stratified_test_fraction": 0.25,
            "train_rows": len(X_train), "test_rows": len(X_test),
            "C_grid": [0.1, 1.0, 10.0, 100.0], "gamma_grid": [0.01, 0.1, 1.0, 10.0],
            "cv": "5-fold stratified, shuffled, seed 42", "selection_metric": "mean CV accuracy",
            "tie_rule": "first candidate in GridSearchCV order", "svc_tolerance": 0.001,
            "preprocessing": "StandardScaler fitted inside each training fold; refit on 300 training rows",
            "linear_baseline": {"kernel": "linear", "C": 1.0},
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__},
        "candidates": rows,
        "result": {
            "selected_parameters": search.best_params_, "selected_cv_accuracy": float(search.best_score_),
            "selected_train_accuracy": float(selected.score(X_train, y_train)),
            "selected_test_accuracy": float(accuracy_score(y_test, prediction)),
            "linear_test_accuracy": float(linear.score(X_test, y_test)),
            "support_vectors_per_class": svm.n_support_.tolist(),
            "support_fraction": float(len(svm.support_) / len(X_train)),
            "classification_report": classification_report(y_test, prediction, output_dict=True, zero_division=0),
        },
        "interpretation_candidate": "Read CV scores alongside training fit and the fixed linear baseline; any nonlinear advantage applies only to this generated task.",
        "review_status": "pending author review",
        "limitation": "One seed, a finite grid and balanced synthetic classes; unequal tuning budgets; no calibration, latency or general superiority evidence.",
    }
    output = Path(__file__).resolve().parent / "outputs" / "kernel_search.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("C       gamma   train    CV       CV std")
    for row in rows:
        print(f"{row['C']:7g} {row['gamma']:7g} {row['train_accuracy']:.4f}   {row['cv_accuracy']:.4f}   {row['cv_std']:.4f}")
    print(json.dumps(report["result"], indent=2))
    print(f"Ignored run record: {output}")


if __name__ == "__main__":
    main()
