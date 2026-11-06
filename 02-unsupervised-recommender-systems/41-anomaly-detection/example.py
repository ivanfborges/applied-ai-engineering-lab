"""Normal-only fitting/calibration and held-out synthetic telemetry evaluation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from from_scratch import RobustMAD, calibrate_threshold, score_vector


def make_telemetry(seed=41):
    """IID synthetic observations, not a time-series replay or measured API data."""
    rng = np.random.default_rng(seed)

    def normal(n):
        z = rng.standard_normal((n, 2))
        return np.column_stack((250 + 35 * z[:, 0], 40 + 8 * (0.8 * z[:, 0] + 0.6 * z[:, 1])))

    train, calibration, normal_test = normal(800), normal(200), normal(300)
    relationship = rng.normal([285, 25], [12, 4], size=(25, 2))
    extreme = rng.normal([375, 70], [15, 6], size=(15, 2))
    test = np.vstack((normal_test, relationship, extreme))
    groups = np.array(["normal"] * 300 + ["relationship"] * 25 + ["extreme"] * 15)
    return train, calibration, test, groups


def fit_detectors(train, seed=41):
    """This interface accepts training features only, never calibration/test labels."""
    return {
        "MAD": RobustMAD().fit(train),
        "IsolationForest": IsolationForest(
            n_estimators=200, max_samples=256, contamination="auto", random_state=seed, n_jobs=1,
        ).fit(train),
        "OneClassSVM": make_pipeline(
            RobustScaler(), OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"),
        ).fit(train),
    }


def anomaly_scores(name, detector, X):
    if name == "MAD":
        values = detector.anomaly_score(X)
    elif name == "IsolationForest":
        values = -detector.score_samples(X)
    elif name == "OneClassSVM":
        values = -detector.decision_function(X)
    else:
        raise ValueError(f"unknown detector: {name}")
    return score_vector(values)


def evaluate_policy(groups, scores, threshold):
    groups = np.asarray(groups)
    scores = score_vector(scores)
    if groups.shape != scores.shape or not np.isin(groups, ["normal", "relationship", "extreme"]).all():
        raise ValueError("groups must match scores and use the documented synthetic group names")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    y = groups != "normal"
    if not y.any() or y.all():
        raise ValueError("evaluation requires both normal rows and anomalies")
    alerts = scores > threshold
    tn, fp, fn, tp = confusion_matrix(y, alerts, labels=[False, True]).ravel()
    return {
        "threshold": float(threshold), "ap": float(average_precision_score(y, scores)),
        "alerts": int(alerts.sum()), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)), "fpr": float(fp / (tn + fp)),
        "recall_by_injection": {
            group: float(alerts[groups == group].mean())
            for group in ("relationship", "extreme") if np.any(groups == group)
        },
    }


def run_experiment(seed=41):
    train, calibration, test, groups = make_telemetry(seed)
    results = []
    for name, detector in fit_detectors(train, seed).items():
        cal_scores = anomaly_scores(name, detector, calibration)
        test_scores = anomaly_scores(name, detector, test)
        for alpha in (0.05, 0.02, 0.005):
            threshold = calibrate_threshold(cal_scores, alpha)
            results.append({
                "detector": name, "alpha": alpha,
                "calibration_alerts": int(np.sum(cal_scores > threshold)),
                **evaluate_policy(groups, test_scores, threshold),
            })
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "hypothesis": "Featurewise MAD can miss relationship violations; changing a frozen score threshold changes alert burden, not ranking.",
        "configuration": {
            "seed": seed, "data": "IID synthetic correlated latency/queue telemetry",
            "features": ["latency_ms", "queue_depth"], "normal_location": [250, 40],
            "normal_scales": [35, 8], "normal_correlation": 0.8,
            "train_normal": 800, "calibration_normal": 200, "test_normal": 300,
            "relationship_injections": {"n": 25, "location": [285, 25], "scales": [12, 4]},
            "extreme_injections": {"n": 15, "location": [375, 70], "scales": [15, 6]},
            "MAD": "max absolute deviation / (1.482602218505602 * feature MAD)",
            "IsolationForest": {"n_estimators": 200, "max_samples": 256, "contamination": "auto", "n_jobs": 1},
            "OneClassSVM": {"scaler": "training-fitted RobustScaler defaults", "kernel": "rbf", "nu": 0.05, "gamma": "scale"},
            "threshold": "normal calibration quantile, method=higher; strict score > threshold",
            "alphas": [0.05, 0.02, 0.005], "test_prevalence": 40 / 340,
        },
        "results": results,
        "interpretation_status": "author review required",
        "interpretation_candidate": "Compare injection-specific recall and false alerts before preferring a model; the result is conditional on this generator and seed.",
        "limitations": "One IID synthetic seed; clean reference assumed; artificial prevalence; no seasonality, drift, incident replay, runtime benchmark or production validation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    run_name = datetime.now(timezone.utc).strftime("experiment-%Y%m%dT%H%M%S%fZ.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / run_name)
    args = parser.parse_args()
    report = run_experiment()
    print("Synthetic telemetry; higher scores = more anomalous; author review required")
    print("detector         alpha     AP alerts   precision recall    FPR")
    for row in report["results"]:
        print(f"{row['detector']:16} {row['alpha']:.3f} {row['ap']:.4f} {row['alerts']:6} "
              f"{row['precision']:11.4f} {row['recall']:.4f} {row['fpr']:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Run record: {args.output}")


if __name__ == "__main__":
    main()
