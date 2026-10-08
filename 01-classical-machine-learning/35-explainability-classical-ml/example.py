"""Inspect a fixed forest on synthetic data; write an ignored numerical report."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance as sklearn_permutation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from from_scratch import pdp_ice, permutation_importance

SEED = 35
FEATURES = ("signal", "signal_proxy", "context", "noise")


def make_data(n=1800, *, seed=SEED, proxy_noise=0.15, interaction=1.6):
    """Synthetic Bernoulli outcome; proxy and noise have no direct target terms."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 20:
        raise ValueError("n must be an integer of at least 20.")
    if not np.isfinite(proxy_noise) or proxy_noise < 0 or not np.isfinite(interaction):
        raise ValueError("proxy_noise must be finite and nonnegative; interaction finite.")
    rng = np.random.default_rng(seed)
    signal, context = rng.normal(size=(2, n))
    X = np.column_stack(
        [signal, signal + proxy_noise * rng.normal(size=n), context, rng.normal(size=n)]
    )
    probabilities = expit(1.4 * signal + 0.9 * context + interaction * signal * context)
    return X, rng.binomial(1, probabilities)


def class_index(model, label=1):
    classes = np.asarray(model.classes_)
    if classes.ndim != 1:
        raise ValueError("Only single-output classifiers are supported.")
    matches = np.flatnonzero(classes == label)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one class matching {label!r}.")
    return int(matches[0])


def explain_probability(model, background, rows, *, label=1):
    """Explicit interventional TreeSHAP with class and reconstruction checks."""
    try:
        import shap
    except ImportError as error:
        raise ImportError('Install the shared extra: python -m pip install -e ".[explainability]"') from error

    background = np.asarray(background, dtype=float)
    rows = np.asarray(rows, dtype=float)
    for values in (background, rows):
        if (values.ndim != 2 or values.shape[0] == 0
                or values.shape[1] != model.n_features_in_ or not np.isfinite(values).all()):
            raise ValueError("Background and rows must be finite nonempty model-width matrices.")
    if len(background) > 100:
        raise ValueError("Use at most 100 background rows to avoid implicit masker subsampling.")
    index = class_index(model, label)
    explainer = shap.TreeExplainer(
        model, data=background, feature_perturbation="interventional", model_output="probability"
    )
    explanation = explainer(rows)
    values = np.asarray(explanation.values)
    bases = np.asarray(explanation.base_values)
    expected_shape = (len(rows), rows.shape[1], len(model.classes_))
    if values.shape != expected_shape or bases.shape != (len(rows), len(model.classes_)):
        raise ValueError(f"Unexpected class-specific SHAP shapes: {values.shape}, {bases.shape}.")
    contributions, baseline = values[:, :, index], bases[:, index]
    predictions = model.predict_proba(rows)[:, index]
    reconstruction = baseline + contributions.sum(axis=1)
    background_mean = model.predict_proba(background)[:, index].mean()
    np.testing.assert_allclose(reconstruction, predictions, rtol=0, atol=1e-6)
    np.testing.assert_allclose(baseline, background_mean, rtol=0, atol=1e-6)
    return {
        "version": shap.__version__,
        "output": f"P(y={label})",
        "feature_perturbation": "interventional",
        "background_rows": len(background),
        "explained_rows": len(rows),
        "background_prediction_mean": float(background_mean),
        "max_reconstruction_error": float(np.max(np.abs(reconstruction - predictions))),
        "mean_absolute_contributions": np.abs(contributions).mean(axis=0).tolist(),
        "first_row": {
            "inputs": rows[0].tolist(), "baseline": float(baseline[0]),
            "contributions": contributions[0].tolist(), "prediction": float(predictions[0]),
        },
    }


def run_experiment():
    X, y = make_data()
    train_ids, test_ids = train_test_split(
        np.arange(len(y)), test_size=0.25, stratify=y, random_state=SEED
    )
    X_train, X_test, y_train, y_test = X[train_ids], X[test_ids], y[train_ids], y[test_ids]
    model = RandomForestClassifier(
        n_estimators=120, min_samples_leaf=5, max_features="sqrt", random_state=SEED, n_jobs=1
    ).fit(X_train, y_train)
    positive = class_index(model)
    predict = lambda rows: model.predict_proba(rows)[:, positive]
    auc = roc_auc_score(y_test, predict(X_test))
    library = sklearn_permutation(
        model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=1
    )
    groups = [(0,), (1,), (2,), (3,), (0, 1)]
    scratch = permutation_importance(model, X_test, y_test, scoring="roc_auc", groups=groups)
    reference = X_test[:150]
    response = partial_dependence(
        model, reference, [0], method="brute", kind="both", response_method="predict_proba",
        grid_resolution=21, percentiles=(0.05, 0.95)
    )
    grid = response["grid_values"][0]
    direct = pdp_ice(predict, reference, 0, grid)
    np.testing.assert_allclose(direct["ice"], response["individual"][0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(direct["pdp"], response["average"][0], rtol=0, atol=1e-12)
    background_positions = np.random.default_rng(SEED).choice(len(X_train), 100, replace=False)
    background_ids = train_ids[background_positions]
    shap_result = explain_probability(model, X[background_ids], X_test[:40])
    importance_rows = [
        {
            "feature": name, "mdi": float(model.feature_importances_[index]),
            "sklearn_auc_decrease": float(library.importances_mean[index]),
            "sklearn_repeat_std": float(library.importances_std[index]),
            "scratch_auc_decrease": float(scratch["importances_mean"][index]),
            "scratch_repeat_std": float(scratch["importances_std"][index]),
        }
        for index, name in enumerate(FEATURES)
    ]
    report = {
        "hypothesis": "Different explanation definitions can allocate correlated signal differently; joint permutation can reveal reliance on the pair.",
        "configuration": {
            "synthetic": True, "n_rows": len(y), "train_rows": len(train_ids),
            "test_rows": len(test_ids), "seed": SEED, "features": FEATURES,
            "generator_logit": "1.4*signal + 0.9*context + 1.6*signal*context",
            "proxy_noise_std": 0.15, "model": model.get_params(),
            "scoring": "roc_auc", "permutation_repeats": 10,
            "pdp_reference": "first 150 held-out rows", "pdp_percentiles": [0.05, 0.95],
            "shap_background_ids": background_ids.tolist(),
            "shap_explanation_ids": test_ids[:40].tolist(),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "shap": shap_result["version"]},
        "result": {
            "test_auc": float(auc), "signal_proxy_correlation": float(np.corrcoef(X_test[:, :2].T)[0, 1]),
            "importance": importance_rows,
            "joint_signal_pair_auc_decrease": float(scratch["importances_mean"][-1]),
            "joint_signal_pair_repeat_std": float(scratch["importances_std"][-1]),
            "pdp_ice": {
                "grid": grid.tolist(), "pdp": direct["pdp"].tolist(),
                "ice": direct["ice"].tolist(), "reference_ids": test_ids[:150].tolist(),
                "prediction_units": "positive-class probability",
                "mean_absolute_signal_proxy_gap_after_replacement": float(np.abs(grid[None, :] - reference[:, 1, None]).mean()),
                "mean_absolute_original_signal_proxy_gap": float(np.abs(reference[:, 0] - reference[:, 1]).mean()),
            },
            "shap": shap_result,
        },
        "interpretation_candidate": "Compare the pair's joint score decrease with individual decreases; inspect dependence violations before interpreting response curves. Additive reconstruction validates bookkeeping only.",
        "review_status": "Interpretation candidate pending author review.",
        "limitations": [
            "One synthetic split and seed; no benchmark, causal effect, or stability claim.",
            "The proxy pair is dependent; replacement breaks its relationship to other inputs.",
            "Repeat standard deviations describe shuffle variability, not population uncertainty.",
            "Library and scratch use different random shuffle streams; identical draws are not claimed.",
            "SHAP describes 40 rows against one 100-row training background, not the whole population.",
            "No model or feature selection uses these held-out diagnostics.",
        ],
    }
    return report


def main():
    report = run_experiment()
    output = Path(__file__).resolve().parent / "outputs" / "explainability.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result = report["result"]
    print(f"Synthetic held-out ROC-AUC: {result['test_auc']:.6f}")
    for row in result["importance"]:
        print(f"{row['feature']:14s} MDI={row['mdi']:.6f} library PI={row['sklearn_auc_decrease']:.6f} scratch PI={row['scratch_auc_decrease']:.6f}")
    print(f"Joint signal-pair PI: {result['joint_signal_pair_auc_decrease']:.6f}")
    print(f"SHAP maximum reconstruction error: {result['shap']['max_reconstruction_error']:.3e}")
    print("PDP/ICE agreement with sklearn: passed")
    print(f"Ignored report: {output}")
    print(report["review_status"])


if __name__ == "__main__":
    main()