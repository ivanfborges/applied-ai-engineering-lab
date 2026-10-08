"""Numerical core for the Day 34 CPU-only visual experiments.

Mechanisms are known because we construct them. Real-world MAR versus MNAR
generally cannot be established conclusively from observed values alone.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
SEEDS = (11, 22, 33, 44, 55)
FEATURE_NAMES = ("Age-like", "Income-like", "Risk-like", "Behavior-like")
STRATEGIES = ("Median", "Median + indicator", "Scaled KNN", "Native NaN")
RATES = (0, .05, .10, .20, .30, .40, .50)
SHIFT_RATES = (.10, .20, .30, .40, .50, .60)


def matrix(X, complete=False):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or 0 in X.shape or np.isinf(X).any():
        raise ValueError("Expected a nonempty numeric matrix without infinity.")
    if complete and np.isnan(X).any():
        raise ValueError("Expected completely observed numerical data.")
    return X


def probability(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError("Missing probability must be a scalar in [0, 1].")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Missing probability must be numeric.") from error
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Missing probability must be finite and in [0, 1].")
    return value


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def generate_dataset(n=2400, seed=RANDOM_STATE):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 100:
        raise ValueError("n must be an integer of at least 100.")
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(42, 12, n), 18, 80)
    age_z = (age - 42) / 12
    behavior = rng.normal(size=n)
    log_income = 3.4 + .45 * age_z + .35 * behavior + .35 * rng.normal(size=n)
    income = np.exp(log_income)
    risk = .65 * behavior - .4 * age_z + .65 * rng.normal(size=n)
    target_probability = sigmoid(1.5 * risk - .65 * behavior + .35 * age_z + .7 * (log_income - 3.4))
    y = (rng.random(n) < target_probability).astype(int)
    return np.column_stack((age, income, risk, behavior)), y


def missingness_probability(X, mechanism):
    X = matrix(X, complete=True)
    if X.shape[1] < 2:
        raise ValueError("Mechanisms need age-like and income-like columns.")
    if mechanism == "MCAR":
        return np.full(len(X), .30)
    if mechanism == "MAR":
        # Age stays observable; parameters are fixed, not fitted on held-out data.
        return .05 + .75 * sigmoid((X[:, 0] - 45) / 5)
    if mechanism == "MNAR":
        # Read the original income before hiding it.
        return .05 + .75 * sigmoid((np.log(np.maximum(X[:, 1], 1e-12)) - 3.65) / .20)
    raise ValueError("mechanism must be MCAR, MAR, or MNAR.")


def inject_mechanism(X, mechanism, seed=RANDOM_STATE, fraction=1.0):
    X = matrix(X, complete=True)
    fraction = probability(fraction)
    p = missingness_probability(X, mechanism)
    mask = np.random.default_rng(seed).random(len(X)) < fraction * p
    result = X.copy()
    result[mask, 1] = np.nan
    return result, mask, p


def inject_mcar(X, rate, seed):
    X = matrix(X, complete=True)
    mask = np.random.default_rng(seed).random(X.shape) < probability(rate)
    result = X.copy()
    result[mask] = np.nan
    return result


def inject_mar(X, seed=RANDOM_STATE):
    return inject_mechanism(X, "MAR", seed)


def inject_mnar(X, seed=RANDOM_STATE):
    return inject_mechanism(X, "MNAR", seed)


def imputed_versions(X):
    """Descriptive, same-sample filling; not a held-out predictive experiment."""
    X = matrix(X)
    if np.isnan(X).all(axis=0).any():
        raise ValueError("Each feature needs at least one observed value.")
    result = {"Mean": SimpleImputer(strategy="mean").fit_transform(X),
              "Median": SimpleImputer(strategy="median").fit_transform(X)}
    scaler = StandardScaler().fit(X)
    filled = KNNImputer(n_neighbors=7).fit_transform(scaler.transform(X))
    result["KNN"] = scaler.inverse_transform(filled)
    return result


def model_for(strategy, seed):
    # Holding the learner fixed isolates preprocessing/routing choices.
    learner = HistGradientBoostingClassifier(
        max_iter=65, max_leaf_nodes=9, min_samples_leaf=20,
        learning_rate=.1, l2_regularization=1, early_stopping=False,
        random_state=seed,
    )
    if strategy == "Median":
        return make_pipeline(SimpleImputer(strategy="median"), learner)
    if strategy == "Median + indicator":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), learner)
    if strategy == "Scaled KNN":
        return make_pipeline(StandardScaler(), KNNImputer(n_neighbors=5), learner)
    if strategy == "Native NaN":
        return learner
    raise ValueError("Unknown missing-data strategy.")


def split_dataset(seed, n=2000):
    X, y = generate_dataset(n, seed)
    return train_test_split(X, y, test_size=.30, stratify=y, random_state=seed)


def auc(model, X, y):
    value = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    if not np.isfinite(value):
        raise ValueError("Evaluation produced a nonfinite ROC-AUC.")
    return value


def run_missingness_robustness_experiment(seeds=SEEDS, rates=RATES, n=2000):
    rows = []
    for seed in seeds:
        train, test, y_train, y_test = split_dataset(seed, n)
        for rate in rates:
            masked_train = inject_mcar(train, rate, seed + 1000)
            masked_test = inject_mcar(test, rate, seed + 2000)
            for strategy in STRATEGIES:
                model = model_for(strategy, seed)
                model.fit(masked_train, y_train)
                rows.append(dict(seed=int(seed), rate=float(rate), strategy=strategy,
                                 auc=auc(model, masked_test, y_test),
                                 realized_train_rate=float(np.isnan(masked_train).mean()),
                                 realized_test_rate=float(np.isnan(masked_test).mean())))
        print(f"Performance sweep: seed {seed} complete.", flush=True)
    return rows


def run_missingness_shift_experiment(seeds=SEEDS, rates=SHIFT_RATES, n=2000):
    rows = []
    for seed in seeds:
        train, test, y_train, y_test = split_dataset(seed, n)
        masked_train = inject_mcar(train, .10, seed + 1000)
        # Fit once; exactly the same complete test values and labels at every rate.
        for strategy in ("Median", "Median + indicator", "Native NaN"):
            model = model_for(strategy, seed).fit(masked_train, y_train)
            for rate in rates:
                masked_test = inject_mcar(test, rate, seed + 2000)
                rows.append(dict(seed=int(seed), rate=float(rate), strategy=strategy,
                                 auc=auc(model, masked_test, y_test),
                                 realized_train_rate=float(np.isnan(masked_train).mean()),
                                 realized_test_rate=float(np.isnan(masked_test).mean())))
        print(f"Frozen-model shift: seed {seed} complete.", flush=True)
    return rows


def summarize(rows):
    summary = []
    keys = list(dict.fromkeys((row["strategy"], row["rate"]) for row in rows))
    for strategy, rate in keys:
        values = np.array([row["auc"] for row in rows if row["strategy"] == strategy and row["rate"] == rate])
        if not np.isfinite(values).all():
            raise ValueError("Summary received nonfinite metrics.")
        summary.append(dict(strategy=strategy, rate=rate, mean=float(values.mean()),
                            std=float(values.std(ddof=1)) if len(values) > 1 else 0,
                            repeats=int(len(values))))
    return summary


def run_feature_dropout(seeds=SEEDS, n=2000):
    scenarios = {
        "All available": (), "Age unavailable": (0,), "Income unavailable": (1,),
        "Risk unavailable": (2,), "Behavior unavailable": (3,),
        "Risk + income unavailable": (1, 2),
    }
    rows = []
    for seed in seeds:
        train, test, y_train, y_test = split_dataset(seed, n)
        model = model_for("Median", seed).fit(train, y_train)
        baseline = auc(model, test, y_test)
        for name, columns in scenarios.items():
            masked = test.copy()
            masked[:, list(columns)] = np.nan
            value = auc(model, masked, y_test)
            rows.append(dict(seed=int(seed), scenario=name, auc=value,
                             delta_from_baseline=value - baseline))
    return rows


def run_indicator_experiment(seed=RANDOM_STATE, n=2400):
    rng = np.random.default_rng(seed)
    context, measurement = rng.normal(size=(2, n))
    latent_regime = rng.integers(0, 2, n)
    y = (rng.random(n) < sigmoid(1.3 * context + 2.3 * latent_regime - 1.1)).astype(int)
    # The unobserved regime causes both missingness and target propensity.
    # The mask does not look at realized outcomes.
    missing = rng.random(n) < np.where(latent_regime == 1, .85, .05)
    X_complete = np.column_stack((context, measurement))
    X = X_complete.copy()
    X[missing, 1] = np.nan
    train_ids, test_ids = train_test_split(np.arange(n), test_size=.3, stratify=y, random_state=seed)
    curves, scores = {}, {}
    for indicator in (False, True):
        label = "With indicator" if indicator else "Without indicator"
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=indicator),
            StandardScaler(), LogisticRegression(C=1, max_iter=500, random_state=seed),
        ).fit(X[train_ids], y[train_ids])
        prediction = model.predict_proba(X[test_ids])[:, 1]
        scores[label] = float(roc_auc_score(y[test_ids], prediction))
        fpr, tpr, _ = roc_curve(y[test_ids], prediction)
        curves[label] = (fpr, tpr)
    return dict(complete=X_complete[test_ids], masked=X[test_ids], y=y[test_ids],
                missing=missing[test_ids], auc=scores, curves=curves)


def distribution_stats(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Distribution statistics require a nonempty finite vector.")
    return dict(n=int(len(values)), mean=float(values.mean()), variance=float(values.var()),
                std=float(values.std()), median=float(np.median(values)))
