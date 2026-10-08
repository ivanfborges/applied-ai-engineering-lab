"""Inspect boosting stages and select learning rate/stage on synthetic validation data."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


SEED = 42
LEARNING_RATES = (1.0, 0.1, 0.01)
N_ESTIMATORS = 300


def make_data():
    """Code-defined synthetic observations; the noiseless function is not a fit target."""
    x = np.linspace(-3.0, 3.0, 300).reshape(-1, 1)
    rng = np.random.default_rng(SEED)
    y = 2.0 * np.sin(x[:, 0]) + 0.4 * x[:, 0] ** 2 + rng.normal(0.0, 0.35, len(x))
    return x, y


def rmse(y, prediction) -> float:
    return float(np.sqrt(mean_squared_error(y, prediction)))


def staged_rmse(model, x, y, baseline: float) -> list[float]:
    return [rmse(y, np.full(len(y), baseline))] + [
        rmse(y, prediction) for prediction in model.staged_predict(x)
    ]


def run_experiment() -> dict:
    x, y = make_data()
    indices = np.arange(len(x))
    development, test = train_test_split(indices, test_size=0.2, random_state=SEED)
    train, validation = train_test_split(development, test_size=0.25, random_state=SEED)
    baseline = float(y[train].mean())
    records = []
    models = []
    for rate in LEARNING_RATES:
        model = GradientBoostingRegressor(
            loss="squared_error",
            learning_rate=rate,
            n_estimators=N_ESTIMATORS,
            max_depth=2,
            min_samples_leaf=3,
            subsample=1.0,
            random_state=SEED,
            n_iter_no_change=None,
        ).fit(x[train], y[train])
        train_curve = staged_rmse(model, x[train], y[train], baseline)
        validation_curve = staged_rmse(model, x[validation], y[validation], baseline)
        # Include the constant baseline; exact ties choose the earliest stage.
        best_stage = int(np.argmin(validation_curve))
        records.append({
            "learning_rate": rate,
            "train_rmse_by_stage": train_curve,
            "validation_rmse_by_stage": validation_curve,
            "best_stage": best_stage,
            "best_validation_rmse": validation_curve[best_stage],
        })
        models.append(model)

    selected_index = min(
        range(len(records)), key=lambda i: records[i]["best_validation_rmse"]
    )
    selected_record, selected_model = records[selected_index], models[selected_index]
    selected_stage = selected_record["best_stage"]
    # Test targets enter only after all validation-based choices are fixed.
    test_prediction = np.full(len(test), baseline)
    if selected_stage:
        for stage, prediction in enumerate(selected_model.staged_predict(x[test]), start=1):
            if stage == selected_stage:
                test_prediction = prediction
                break
    return {
        "hypothesis": "At a fixed tree budget, smaller learning rates may need more stages; "
                      "training improvement need not imply validation improvement.",
        "configuration": {
            "data": "synthetic: 300 evenly spaced x in [-3,3]; "
                    "y = 2*sin(x) + 0.4*x**2 + Normal(0, 0.35**2)",
            "seed": SEED,
            "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
            "split_indices": {"train": train.tolist(), "validation": validation.tolist(), "test": test.tolist()},
            "learning_rates": list(LEARNING_RATES),
            "n_estimators": N_ESTIMATORS,
            "max_depth": 2,
            "min_samples_leaf": 3,
            "subsample": 1.0,
            "loss": "squared_error",
            "n_iter_no_change": None,
            "selection": "minimum validation RMSE over all learning rates and stages 0..300; "
                         "no refit; retain selected prefix",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "results": {
            "learning_rate_comparison": records,
            "selected_learning_rate": selected_record["learning_rate"],
            "selected_stage": selected_stage,
            "selected_test_rmse": rmse(y[test], test_prediction),
            "constant_baseline_test_rmse": rmse(y[test], np.full(len(test), baseline)),
        },
        "interpretation_candidate": "Review the measured trajectories before deciding whether "
                                    "they support slower correction or fitting noise in this run.",
        "review_status": "pending author review",
        "limitations": [
            "One synthetic function, noise realization, and random split; no benchmark or uncertainty interval.",
            "Validation minima are used for selection and are optimistically biased.",
            "Scanning a fitted path selects a prefix after training; this does not save training time.",
            "No comparison of losses, depths, stochastic boosting, classification, or Random Forest.",
        ],
    }


def main() -> None:
    record = run_experiment()
    print("Synthetic nonlinear regression; selection uses validation only.")
    print("rate  stage  train_RMSE  validation_RMSE")
    for comparison in record["results"]["learning_rate_comparison"]:
        for stage in (0, 1, 5, 20, 100, 300):
            print(
                f'{comparison["learning_rate"]:4.2f}  {stage:5d}  '
                f'{comparison["train_rmse_by_stage"][stage]:10.6f}  '
                f'{comparison["validation_rmse_by_stage"][stage]:15.6f}'
            )
        print(f'Best validation stage: {comparison["best_stage"]}')
    results = record["results"]
    print(f'Selected: rate={results["selected_learning_rate"]}, stage={results["selected_stage"]}')
    print(f'Selected test RMSE: {results["selected_test_rmse"]:.6f}')
    print(f'Constant baseline test RMSE: {results["constant_baseline_test_rmse"]:.6f}')
    output = Path(__file__).resolve().parent / "outputs" / "experiment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Generated record: {output}")
    print("Interpretation candidate requires author review.")


if __name__ == "__main__":
    main()