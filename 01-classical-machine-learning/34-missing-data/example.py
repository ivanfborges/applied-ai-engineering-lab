"""Synthetic missing-data strategies and frozen-model robustness checks."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from from_scratch import numeric_matrix

SEED = 42
MASK_SEEDS = {"train": 101, "validation": 102, "test": 103}
EXTRA_DROPOUT = 0.40


def inject_missingness(X, seed):
    """Mask complete synthetic data; x2 stays observed as the MAR driver."""
    values = numeric_matrix(X)
    if values.shape[1] < 4 or np.isnan(values).any():
        raise ValueError("Injection requires complete data with at least four columns.")
    rng = np.random.default_rng(seed)
    result = values.copy()
    probabilities = (
        (0, np.full(len(values), 0.15)),
        (1, np.where(values[:, 2] > 0, 0.35, 0.05)),
        (3, np.where(values[:, 3] > 0.8, 0.50, 0.05)),
    )
    # MNAR probabilities use complete x3 before its values disappear.
    for column, probability in probabilities:
        result[rng.random(len(values)) < probability, column] = np.nan
    return result


def stress_scenarios(X, extra_rate=EXTRA_DROPOUT, seed=104):
    """Keep the same rows; add independent dropout or a full x2 outage."""
    values = numeric_matrix(X)
    if values.shape[1] < 4:
        raise ValueError("Stress scenarios require at least four columns.")
    if (isinstance(extra_rate, (bool, np.bool_)) or
            not np.isfinite(extra_rate) or not 0 <= extra_rate <= 1):
        raise ValueError("extra_rate must be a finite probability in [0, 1].")
    shifted = values.copy()
    rng = np.random.default_rng(seed)
    for column in (0, 1, 3):
        shifted[rng.random(len(values)) < extra_rate, column] = np.nan
    outage = values.copy()
    outage[:, 2] = np.nan
    return {"matched": values.copy(), "extra_dropout": shifted, "x2_outage": outage}


def build_models():
    def forest():
        return RandomForestClassifier(
            n_estimators=120, min_samples_leaf=3, random_state=SEED, n_jobs=1
        )

    return {
        "median": Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("model", forest())
        ]),
        "median_indicator": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", forest()),
        ]),
        "scaled_knn": Pipeline([
            # Scaling fits observed training values and preserves NaNs.
            ("scaler", StandardScaler()),
            ("imputer", KNNImputer(n_neighbors=5)),
            ("model", forest()),
        ]),
        "native_hgb": HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=15, l2_regularization=1.0,
            early_stopping=False, random_state=SEED,
        ),
    }


def fit_and_select(X_train, y_train, X_validation, y_validation):
    """Select fixed configurations on validation AUC; never receive test data."""
    models = build_models()
    validation_auc = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_validation)[:, 1]
        validation_auc[name] = float(roc_auc_score(y_validation, probabilities))
    # Dictionary order provides a deterministic tie rule.
    selected = max(validation_auc, key=validation_auc.get)
    return models, validation_auc, selected


def classification_metrics(y, probabilities):
    labels = np.asarray(y)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or labels.size == 0 or labels.shape != scores.shape:
        raise ValueError("Labels and probabilities must be aligned, nonempty vectors.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Predicted probabilities must be finite and in [0, 1].")
    return {
        "n": int(labels.size),
        "positives": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None,
        "brier": float(brier_score_loss(labels, scores)),
    }


def evaluate(model, X, y):
    values = numeric_matrix(X)
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(values):
        raise ValueError("y must contain one label per row of X.")
    probabilities = model.predict_proba(values)[:, 1]
    missing_rows = np.isnan(values).any(axis=1)
    groups = {}
    for name, mask in (("complete_rows", ~missing_rows), ("any_missing", missing_rows)):
        groups[name] = classification_metrics(labels[mask], probabilities[mask]) if mask.any() else None
    return {"overall": classification_metrics(labels, probabilities), "groups": groups}


def run_experiment():
    X, y = make_classification(
        n_samples=2400, n_features=8, n_informative=5, n_redundant=1,
        weights=[0.5, 0.5], flip_y=0.02, class_sep=1.0, shuffle=False,
        random_state=SEED,
    )
    train_ids, holdout_ids = train_test_split(
        np.arange(len(y)), test_size=0.4, stratify=y, random_state=SEED
    )
    validation_ids, test_ids = train_test_split(
        holdout_ids, test_size=0.5, stratify=y[holdout_ids], random_state=SEED
    )
    partitions = {"train": train_ids, "validation": validation_ids, "test": test_ids}
    masked = {name: inject_missingness(X[ids], MASK_SEEDS[name]) for name, ids in partitions.items()}
    models, validation_auc, selected = fit_and_select(
        masked["train"], y[train_ids], masked["validation"], y[validation_ids]
    )
    # All models freeze before stress scoring; test results cannot change selection.
    scenarios = stress_scenarios(masked["test"])
    results = {
        scenario: {name: evaluate(model, values, y[test_ids]) for name, model in models.items()}
        for scenario, values in scenarios.items()
    }
    train_median = np.nanmedian(masked["train"], axis=0)
    test_filled = np.where(np.isnan(masked["test"]), train_median, masked["test"])
    distribution = {}
    for column in (0, 1, 3):
        distribution[f"x{column}"] = {
            name: {"n": int(len(values)), "mean": float(np.mean(values)), "variance_ddof0": float(np.var(values))}
            for name, values in {
                "complete_synthetic_truth": X[test_ids, column],
                "observed_only": masked["test"][~np.isnan(masked["test"][:, column]), column],
                "training_median_imputed": test_filled[:, column],
            }.items()
        }
    return {
        "hypothesis": "Imputation and indicators may change prediction quality; frozen models may degrade under additional feature loss.",
        "configuration": {
            "data": "Entirely synthetic sklearn.make_classification; no external dataset.",
            "n_samples": 2400, "n_features": 8, "n_informative": 5,
            "n_redundant": 1, "class_sep": 1.0, "weights": [0.5, 0.5],
            "flip_y": 0.02, "shuffle": False,
            "seed": SEED, "mask_seeds": MASK_SEEDS, "stress_seed": 104,
            "extra_dropout": EXTRA_DROPOUT,
            "mechanisms": {"x0": "MCAR p=0.15", "x1": "MAR p=0.35 if observed x2>0, else 0.05", "x3": "MNAR p=0.50 if complete x3>0.8, else 0.05"},
            "stress": "Add independent 40% dropout to x0/x1/x3, or hide all x2; retain matched test rows and labels.",
            "forest": {"n_estimators": 120, "min_samples_leaf": 3, "n_jobs": 1, "random_state": SEED},
            "knn": {"n_neighbors": 5, "weights": "uniform", "scaling": "observed training StandardScaler before imputation"},
            "hgb": {"max_iter": 120, "max_leaf_nodes": 15, "l2_regularization": 1.0, "early_stopping": False, "random_state": SEED},
            "other_parameters": "Library defaults; package versions recorded below.",
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "partitions": {
            name: {"n": int(len(ids)), "positives": int(y[ids].sum()), "missing_rates": np.isnan(masked[name]).mean(axis=0).tolist()}
            for name, ids in partitions.items()
        },
        "scenario_missing_rates": {name: np.isnan(values).mean(axis=0).tolist() for name, values in scenarios.items()},
        "validation_auc": validation_auc,
        "selected_on_validation": selected,
        "selection_rule": "Highest validation ROC-AUC; ties follow median, median_indicator, scaled_knn, native_hgb order.",
        "test_results": results,
        "distribution_diagnostics": distribution,
        "interpretation_candidate": "Compare matched and stressed results within each frozen model; native HGB also changes the classifier and cannot isolate imputation effects.",
        "review_status": "Interpretation candidates pending author review.",
        "limitations": [
            "One synthetic seed and split; no confidence intervals or general robustness guarantee.",
            "Mechanisms are known by construction, not inferred from observed data.",
            "The mixed mask does not estimate separate causal effects of MCAR, MAR and MNAR.",
            "Stress scenarios are illustrative, not measured production outages.",
            "Subgroup metrics are descriptive; single-class AUC and empty groups are null.",
            "No runtime benchmark, statistical inference, or recovery of true missing values is established.",
        ],
    }


def main():
    with threadpool_limits(limits=1):
        report = run_experiment()
    print("Validation ROC-AUC:", report["validation_auc"])
    print("Selected on validation:", report["selected_on_validation"])
    print("Scenario          Model                 ROC-AUC    Brier")
    for scenario, models in report["test_results"].items():
        for name, result in models.items():
            metrics = result["overall"]
            print(f"{scenario:17} {name:21} {metrics['roc_auc']:.4f}     {metrics['brier']:.4f}")
    output = Path(__file__).resolve().parent / "outputs" / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Report:", output)
    print(report["review_status"])


if __name__ == "__main__":
    main()
