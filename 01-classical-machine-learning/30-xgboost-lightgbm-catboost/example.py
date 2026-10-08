"""Compare three boosting workflows on explicitly synthetic tabular data."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

SEED = 42
ROUNDS = 400
PATIENCE = 30
RATE = 0.05


def make_data(dataset: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if dataset == "numeric":
        x, y = make_classification(n_samples=2400, n_features=8, n_informative=5,
                                  n_redundant=1, weights=[0.65, 0.35], flip_y=0.03,
                                  class_sep=1.0, random_state=SEED)
        return pd.DataFrame(x, columns=[f"x{i}" for i in range(8)]), y, []
    if dataset != "mixed":
        raise ValueError("dataset must be numeric or mixed")
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(2400, 4))
    merchant = rng.integers(0, 80, size=len(x))
    channel = rng.integers(0, 4, size=len(x))
    merchant_effect = rng.normal(0.0, 1.0, size=80)
    z = (-0.8 + 1.2 * x[:, 0] - 0.9 * x[:, 1] + 0.7 * x[:, 2] * x[:, 3]
         + merchant_effect[merchant] + 0.6 * (channel == 1))
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-z)))
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(4)])
    frame["merchant"] = [f"merchant_{i}" for i in merchant]
    frame["channel"] = [f"channel_{i}" for i in channel]
    return frame, y, ["merchant", "channel"]


def split_indices(y) -> dict[str, np.ndarray]:
    labels = np.asarray(y)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("y must be a one-dimensional binary target containing both classes")
    try:
        development, test = train_test_split(np.arange(len(labels)), test_size=0.2,
                                             stratify=labels, random_state=SEED)
        train, validation = train_test_split(development, test_size=0.25,
                                             stratify=labels[development], random_state=SEED)
    except ValueError as error:
        raise ValueError("not enough rows per class for stratified train/validation/test splits") from error
    return {"train": train, "validation": validation, "test": test}


def lightgbm_frames(frames: dict[str, pd.DataFrame], categoricals: list[str]) -> dict[str, pd.DataFrame]:
    result = {key: frame.copy() for key, frame in frames.items()}
    for column in categoricals:
        dtype = pd.CategoricalDtype(categories=sorted(frames["train"][column].unique()))
        for frame in result.values():
            # Held-out categories absent from training become missing, never new codes.
            known = frame[column].where(frame[column].isin(dtype.categories))
            frame[column] = known.astype(dtype)
    return result


def metrics(y, probability) -> dict[str, float]:
    return {"log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "roc_auc": float(roc_auc_score(y, probability)),
            "brier": float(brier_score_loss(y, probability))}


def run_experiment(dataset: str) -> dict:
    try:
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier, early_stopping
        from catboost import CatBoostClassifier
    except ImportError as error:
        raise RuntimeError('Install the shared boosting extra: python -m pip install -e ".[dev,boosting]"') from error

    frame, y, categoricals = make_data(dataset)
    indices = split_indices(y)
    frames = {key: frame.iloc[rows].copy() for key, rows in indices.items()}
    labels = {key: y[rows] for key, rows in indices.items()}
    numeric = [column for column in frame if column not in categoricals]
    encoder = ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False), categoricals)], remainder="passthrough")
    xgb_frames = {"train": encoder.fit_transform(frames["train"])}
    for key in ("validation", "test"):
        xgb_frames[key] = encoder.transform(frames[key])
    lgb_frames = lightgbm_frames(frames, categoricals)
    common = dict(n_estimators=ROUNDS, learning_rate=RATE, max_depth=3,
                  reg_lambda=1.0, random_state=SEED, n_jobs=1)
    xgb_params = dict(**common, objective="binary:logistic", tree_method="hist",
                      grow_policy="depthwise", min_child_weight=1.0, reg_alpha=0.0,
                      gamma=0.0, subsample=1.0, colsample_bytree=1.0,
                      eval_metric="logloss", early_stopping_rounds=PATIENCE)
    lgb_params = dict(**common, objective="binary", num_leaves=8, min_child_samples=20,
                      reg_alpha=0.0, subsample=1.0, subsample_freq=0, colsample_bytree=1.0,
                      boosting_type="gbdt", data_sample_strategy="bagging",
                      deterministic=True, force_col_wise=True, verbosity=-1)
    cat_params = dict(iterations=ROUNDS, learning_rate=RATE, depth=3, l2_leaf_reg=1.0,
                      loss_function="Logloss", eval_metric="Logloss", random_seed=SEED,
                      thread_count=1, task_type="CPU", boosting_type="Ordered",
                      grow_policy="SymmetricTree", bootstrap_type="No", random_strength=0.0,
                      leaf_estimation_method="Newton", leaf_estimation_iterations=1,
                      allow_writing_files=False, verbose=False)
    models = {"XGBoost": XGBClassifier(**xgb_params), "LightGBM": LGBMClassifier(**lgb_params),
              "CatBoost": CatBoostClassifier(**cat_params)}
    inputs = {"XGBoost": xgb_frames, "LightGBM": lgb_frames, "CatBoost": frames}
    models["XGBoost"].fit(xgb_frames["train"], labels["train"],
                          eval_set=[(xgb_frames["validation"], labels["validation"])], verbose=False)
    models["LightGBM"].fit(lgb_frames["train"], labels["train"],
                           eval_X=lgb_frames["validation"], eval_y=labels["validation"],
                           eval_metric="binary_logloss", categorical_feature=categoricals or "auto",
                           callbacks=[early_stopping(PATIENCE, first_metric_only=True, verbose=False)])
    models["CatBoost"].fit(frames["train"], labels["train"], cat_features=categoricals,
                          eval_set=(frames["validation"], labels["validation"]),
                          early_stopping_rounds=PATIENCE, use_best_model=True)
    rounds = {"XGBoost": int(models["XGBoost"].best_iteration + 1),
              "LightGBM": int(models["LightGBM"].best_iteration_),
              "CatBoost": int(models["CatBoost"].tree_count_)}
    results = []
    for name, model in models.items():
        results.append({"model": name, "selected_rounds": rounds[name],
                        "train": metrics(labels["train"], model.predict_proba(inputs[name]["train"])[:, 1]),
                        "validation": metrics(labels["validation"], model.predict_proba(inputs[name]["validation"])[:, 1])})
    selected = min(results, key=lambda record: record["validation"]["log_loss"])["model"]
    # Selection is fixed before any test labels are scored; no refit is performed.
    test_probability = models[selected].predict_proba(inputs[selected]["test"])[:, 1]
    baseline = np.full(len(labels["test"]), labels["train"].mean())
    return {
        "hypothesis": "With shared rows and validation selection, limited-capacity boosting "
                      "may improve on a constant probability baseline; category representations "
                      "can change validation behavior without establishing a library ranking.",
        "configuration": {"dataset": dataset, "synthetic": True, "seed": SEED,
            "generator": "make_data() in example.py; no public dataset or downloaded data",
            "split_indices": {key: rows.tolist() for key, rows in indices.items()},
            "split_sizes": {key: len(rows) for key, rows in indices.items()},
            "numeric_columns": numeric, "categorical_columns": categoricals,
            "categorical_cardinality_train": {c: int(frames["train"][c].nunique()) for c in categoricals},
            "representations": {"XGBoost": "train-fitted dense one-hot; unknown categories all-zero",
                                "LightGBM": "train-fitted pandas category codes; unknowns missing",
                                "CatBoost": "raw strings; internal categorical statistics"},
            "feature_counts": {name: int(data["train"].shape[1]) for name, data in inputs.items()},
            "parameters": {"XGBoost": xgb_params, "LightGBM": lgb_params, "CatBoost": cat_params},
            "patience": PATIENCE, "selection": "minimum validation log loss; no refit; selected-model test only"},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in
               ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost")}},
        "results": {"candidates": results, "selected_model": selected,
                    "selected_test": metrics(labels["test"], test_probability),
                    "constant_baseline_test": metrics(labels["test"], baseline)},
        "interpretation_candidate": "Review train/validation gaps, selected iterations, and the "
            "selected-model test result relative to the baseline. Differences confound representation "
            "and training algorithms; they do not isolate categorical handling or establish superiority.",
        "review_status": "pending author review",
        "limitations": ["One seed and IID synthetic split per scenario; no uncertainty interval or business validation.",
            "Depth and leaf limits bound capacity but do not equalize tree shapes, regularization, bins, or initialization.",
            "Validation is reused for early stopping and model choice and is optimistically selected.",
            "No runtime benchmark, high-cardinality scale claim, calibration fit, GPU, or missingness study.",
            "Dense one-hot is deliberate for this small example; large cardinalities need a different memory strategy."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("numeric", "mixed"), default="numeric")
    args = parser.parse_args()
    record = run_experiment(args.dataset)
    print(f"Synthetic {args.dataset} data; validation log loss selects rounds and model.")
    for candidate in record["results"]["candidates"]:
        print(f'{candidate["model"]}: rounds={candidate["selected_rounds"]}, '
              f'train={candidate["train"]}, validation={candidate["validation"]}')
    print(f'Selected: {record["results"]["selected_model"]}')
    print(f'Test: {record["results"]["selected_test"]}')
    print(f'Constant baseline test: {record["results"]["constant_baseline_test"]}')
    output = Path(__file__).resolve().parent / "outputs" / f"{args.dataset}_experiment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Ignored experiment record: {output}")
    print("Interpretation pending author review.")


if __name__ == "__main__":
    main()
