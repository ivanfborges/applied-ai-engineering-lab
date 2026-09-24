"""Synthetic logistic classification: probability, odds, and fixed decisions."""

import platform

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, log_loss, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from from_scratch import classify, fit_logistic, logit, make_synthetic_data


def make_pipeline(n_train, l2=0.02):
    """Match mean BCE + l2/2 * ||w||^2 for unweighted binary training."""
    if (isinstance(n_train, (bool, np.bool_))
            or not isinstance(n_train, (int, np.integer)) or n_train < 1):
        raise ValueError("n_train must be a positive integer")
    if not np.isscalar(l2) or not np.isfinite(l2) or l2 <= 0:
        raise ValueError("l2 must be finite and positive")
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0 / (n_train * l2), solver="lbfgs", tol=1e-10, max_iter=2000,
        )),
    ])


def main():
    X, y = make_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=21,
    )
    model = make_pipeline(len(X_train)).fit(X_train, y_train)
    scaler = model.named_steps["scale"]
    classifier = model.named_steps["classifier"]
    positive_index = np.flatnonzero(classifier.classes_ == 1).item()
    probabilities = model.predict_proba(X_test)[:, positive_index]
    fitted = fit_logistic(scaler.transform(X_train), y_train)
    if not fitted.converged:
        raise RuntimeError("Scratch solver did not converge")
    scratch_probabilities = fitted.predict_proba(scaler.transform(X_test))
    # Undo standardization so one-unit odds ratios refer to original units.
    raw_coef = classifier.coef_[0] / scaler.scale_
    raw_intercept = classifier.intercept_[0] - raw_coef @ scaler.mean_

    print(f"Python {platform.python_version()}; NumPy {np.__version__}; sklearn {sklearn.__version__}")
    print(f"Synthetic data: train={len(y_train)}, test={len(y_test)}, seed=21")
    print(f"Positive prevalence: train={y_train.mean():.4f}, test={y_test.mean():.4f}")
    print(f"l2=0.02; sklearn C={classifier.C:.8f}")
    print(f"Test log loss: sklearn={log_loss(y_test, probabilities):.8f}; scratch={log_loss(y_test, scratch_probabilities):.8f}")
    baseline = np.full(len(y_test), y_train.mean())
    print(f"Train-prevalence baseline test log loss: {log_loss(y_test, baseline):.8f}")
    print(f"Maximum test probability difference: {np.max(np.abs(probabilities - scratch_probabilities)):.3e}")
    print(f"Scratch objective: {fitted.objective_history[0]:.8f} -> {fitted.objective_history[-1]:.8f}")
    print(f"Scratch iterations={fitted.n_iter}; gradient infinity norm={fitted.gradient_norm:.3e}")
    print(f"Raw coefficients: {raw_coef}; intercept={raw_intercept:.6f}")
    print(f"Odds ratios per raw feature unit: {np.exp(raw_coef)}")
    print("Boundary at threshold t: raw_coef @ x + raw_intercept = logit(t)")
    print("Fixed illustrative thresholds; no threshold is selected from test results.")
    print("threshold  logit(t)  positives  accuracy  precision  recall  TN FP FN TP")
    for threshold in (0.2, 0.5, 0.8):
        predicted = classify(probabilities, threshold)
        counts = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
        print(f"{threshold:9.1f} {logit(threshold):9.4f} {predicted.sum():10d} "
              f"{accuracy_score(y_test, predicted):9.4f} "
              f"{precision_score(y_test, predicted, zero_division=0):10.4f} "
              f"{recall_score(y_test, predicted, zero_division=0):7.4f} "
              + " ".join(str(value) for value in counts))


if __name__ == "__main__":
    main()