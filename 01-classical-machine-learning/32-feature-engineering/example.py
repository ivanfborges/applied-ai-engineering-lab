"""Synthetic customers: train-fitted transforms and point-in-time aggregates."""
import inspect
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from from_scratch import pair_product

SEED = 42
RAW_NUMERIC = ["age", "income", "balance"]
CATEGORICAL = ["region", "channel"]
AGGREGATES = ["txn_count_30d", "txn_mean_30d", "txn_sum_30d", "txn_max_30d"]
VARIANTS = ("baseline", "aggregates", "engineered")


def aggregate_history(snapshots, transactions, window_days=30):
    """One snapshot per customer; events in [t-window, t), available before t."""
    if (isinstance(window_days, bool)
            or not isinstance(window_days, (int, np.integer)) or window_days <= 0):
        raise ValueError("window_days must be a positive integer.")
    snapshot_columns = {"customer_id", "prediction_time"}
    event_columns = {"customer_id", "event_time", "available_at", "amount"}
    if not snapshot_columns <= set(snapshots) or not event_columns <= set(transactions):
        raise ValueError("Missing required snapshot or transaction columns.")
    left = snapshots[["customer_id", "prediction_time"]].copy()
    events = transactions[["customer_id", "event_time", "available_at", "amount"]].copy()
    if (left.empty or left["customer_id"].isna().any()
            or left["customer_id"].duplicated().any()
            or events["customer_id"].isna().any()):
        raise ValueError("Require nonempty snapshots, unique snapshot IDs, and nonnull IDs.")
    for table, columns in [(left, ["prediction_time"]),
                           (events, ["event_time", "available_at"])]:
        for column in columns:
            table[column] = pd.to_datetime(table[column], utc=True, errors="raise")
            if table[column].isna().any():
                raise ValueError("Timestamps cannot be missing.")
    events["amount"] = pd.to_numeric(events["amount"], errors="raise")
    if not np.isfinite(events["amount"].to_numpy(dtype=float)).all():
        raise ValueError("Amounts must be finite.")
    if (events["available_at"] < events["event_time"]).any():
        raise ValueError("available_at cannot precede event_time in this example.")
    joined = left.merge(events, on="customer_id", how="inner", validate="one_to_many")
    eligible = (
        (joined["event_time"] >= joined["prediction_time"] - pd.Timedelta(days=window_days))
        & (joined["event_time"] < joined["prediction_time"])
        & (joined["available_at"] < joined["prediction_time"])
    )
    names = [f"txn_{stat}_{window_days}d" for stat in ["count", "mean", "sum", "max"]]
    history = joined.loc[eligible].groupby("customer_id")["amount"].agg(
        ["size", "mean", "sum", "max"]
    )
    history.columns = names
    # No-history means/maxima use zero by convention, accompanied by count=0.
    result = history.reindex(left["customer_id"].to_numpy()).fillna(0)
    result.index = snapshots.index
    result[names[0]] = result[names[0]].astype(int)
    return result


def make_synthetic_data(n_customers=2500, seed=SEED):
    """Labels deliberately depend on history and nonlinear terms; no external data."""
    if (isinstance(n_customers, bool) or not isinstance(n_customers, (int, np.integer))
            or n_customers < 20):
        raise ValueError("n_customers must be an integer of at least 20.")
    rng = np.random.default_rng(seed)
    cutoff = pd.Timestamp("2026-09-01", tz="UTC")
    customers = pd.DataFrame({
        "customer_id": np.arange(n_customers), "prediction_time": cutoff,
        "age": rng.integers(18, 75, n_customers),
        "income": rng.lognormal(11.0, 0.45, n_customers),
        "balance": rng.lognormal(9.0, 0.8, n_customers),
        "region": rng.choice(["north", "south", "southeast", "center"], n_customers),
        "channel": rng.choice(["web", "mobile", "branch"], n_customers),
    })
    counts = rng.poisson(8, n_customers)
    ids = np.repeat(customers["customer_id"].to_numpy(), counts)
    incomes = np.repeat(customers["income"].to_numpy(), counts)
    event_time = cutoff + pd.to_timedelta(rng.integers(-50, 11, len(ids)), unit="D")
    events = pd.DataFrame({
        "customer_id": ids, "event_time": event_time,
        "available_at": event_time + pd.to_timedelta(rng.integers(0, 5, len(ids)), unit="D"),
        "amount": rng.lognormal(np.log(incomes * 0.003), 0.7),
    })
    features = pd.concat([customers, aggregate_history(customers, events)], axis=1)
    age_signal = np.where(features["age"] < 25, 0.8,
                          np.where(features["age"] >= 60, 0.6, -0.3))
    logit = (-2 + age_signal
             + 0.35 * features["income"] * features["balance"] / 1e9
             + 0.004 * features["txn_mean_30d"]
             + 0.15 * (features["channel"] == "mobile"))
    probability = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
    labels = pd.Series(rng.binomial(1, probability), name="target", index=features.index)
    return features, labels, events


def build_pipeline(variant):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}.")
    numeric = RAW_NUMERIC + (AGGREGATES if variant != "baseline" else [])
    transforms = [
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ]
    if variant == "engineered":
        options = dict(n_bins=4, encode="onehot-dense", strategy="quantile",
                       subsample=None, random_state=SEED)
        # Explicit linear quantiles on newer releases; 1.4 uses this implicitly.
        if "quantile_method" in inspect.signature(KBinsDiscretizer).parameters:
            options["quantile_method"] = "linear"
        transforms.extend([
            ("age_bins", KBinsDiscretizer(**options), ["age"]),
            ("interaction", Pipeline([
                ("center", StandardScaler()),
                ("product", FunctionTransformer(pair_product, validate=True)),
                ("scale", StandardScaler()),
            ]), ["income", "balance"]),
        ])
    return Pipeline([
        ("features", ColumnTransformer(transforms, remainder="drop")),
        ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),
    ])


def run_experiment():
    features, labels, events = make_synthetic_data()
    train, holdout = train_test_split(np.arange(len(features)), test_size=0.25,
                                     stratify=labels, random_state=SEED)
    results = {}
    for variant in VARIANTS:
        model = build_pipeline(variant).fit(features.iloc[train], labels.iloc[train])
        scores = model.predict_proba(features.iloc[holdout])[:, 1]
        results[variant] = {
            "roc_auc": float(roc_auc_score(labels.iloc[holdout], scores)),
            "average_precision": float(average_precision_score(labels.iloc[holdout], scores)),
            "feature_count": int(model.named_steps["features"].transform(features.iloc[train]).shape[1]),
            "solver_iterations": int(model.named_steps["classifier"].n_iter_[0]),
        }
    return {
        "hypothesis": "Historical aggregates and nonlinear terms may improve a fixed logistic model's ranking.",
        "configuration": {
            "seed": SEED, "customers": len(features), "events": len(events),
            "train_rows": len(train), "holdout_rows": len(holdout),
            "holdout_prevalence": float(labels.iloc[holdout].mean()),
            "split": "stratified customer split; one snapshot per customer, no tuning",
            "history": "[t-30 days, t), with available_at < t; zero for absent history",
            "model": "LogisticRegression C=1, lbfgs, max_iter=2000",
            "bins": "4 training quantile bins, linear quantiles, no subsampling; raw age retained",
            "interaction": "center/scale income and balance, multiply, train-scale product; main effects once",
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "results": results,
        "interpretation_candidate": "Compare measured variants as evidence about this generator and fixed model only.",
        "limitations": [
            "The target deliberately rewards history, age regimes and an income-balance interaction.",
            "One seed and split, no uncertainty estimate or model selection; fixed C across feature spaces.",
            "Aggregates add historical source information absent from the baseline table.",
            "Bins and interaction are bundled; their individual effects are not isolated.",
            "Small in-memory joins, complete snapshots, no serving or repeated-customer evaluation.",
        ],
        "review_status": "Interpretation pending author review.",
    }


def main():
    record = run_experiment()
    for variant, result in record["results"].items():
        print(f"{variant:12s} ROC-AUC={result['roc_auc']:.4f} "
              f"AP={result['average_precision']:.4f} features={result['feature_count']}")
    path = Path(__file__).resolve().parent / "outputs" / "feature_comparison.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Ignored experiment record: {path}")
    print(record["review_status"])


if __name__ == "__main__":
    main()
