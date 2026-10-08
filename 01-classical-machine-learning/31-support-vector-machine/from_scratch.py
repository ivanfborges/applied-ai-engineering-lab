"""Educational linear SVM with L2 regularization and mean hinge loss."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np


def positive_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def feature_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.all(np.isfinite(X)):
        raise ValueError("X must be a nonempty finite two-dimensional matrix")
    return X


def signed_labels(y, n_samples):
    y = np.asarray(y, dtype=float)
    if y.shape != (n_samples,) or not np.all(np.isin(y, [-1, 1])):
        raise ValueError("y must contain one -1 or +1 label per row")
    return y


def objective_and_subgradient(X, y, w, b, C=1.0):
    """Return ||w||^2/2 + C mean(hinge) and a valid subgradient."""
    X = feature_matrix(X)
    y = signed_labels(y, len(X))
    w = np.asarray(w, dtype=float)
    if w.shape != (X.shape[1],) or not np.all(np.isfinite(w)):
        raise ValueError("w must be finite with one coefficient per feature")
    b_array = np.asarray(b, dtype=float)
    if b_array.ndim != 0 or not np.isfinite(b_array):
        raise ValueError("b must be a finite scalar")
    b = float(b_array)
    C = positive_number(C, "C")
    margins = y * (X @ w + b)
    losses = np.maximum(0.0, 1.0 - margins)
    # At margin == 1 choose zero from the hinge subdifferential [-1, 0].
    active = margins < 1.0
    # Normalize by ALL rows, including those with zero hinge loss.
    grad_w = w - C * (X[active].T @ y[active]) / len(X)
    grad_b = -C * float(np.sum(y[active])) / len(X)
    loss = 0.5 * float(w @ w) + C * float(losses.mean())
    if not np.isfinite(loss) or not np.all(np.isfinite(grad_w)) or not np.isfinite(grad_b):
        raise ValueError("nonfinite objective or subgradient; rescale inputs")
    return loss, grad_w, grad_b


class LinearSVM:
    """Deterministic batch subgradient descent; accepts {-1, +1} labels."""

    def __init__(self, C=1.0, learning_rate=0.1, epochs=5000):
        self.C = positive_number(C, "C")
        self.learning_rate = positive_number(learning_rate, "learning_rate")
        if (isinstance(epochs, (bool, np.bool_))
                or not isinstance(epochs, (int, np.integer)) or epochs <= 0):
            raise ValueError("epochs must be a positive integer")
        self.epochs = int(epochs)

    def fit(self, X, y):
        X = feature_matrix(X)
        y = signed_labels(y, len(X))
        if len(np.unique(y)) != 2:
            raise ValueError("fit requires both -1 and +1 classes")
        w, b = np.zeros(X.shape[1]), 0.0
        history = []
        best_loss = np.inf
        for step in range(self.epochs + 1):
            loss, grad_w, grad_b = objective_and_subgradient(X, y, w, b, self.C)
            history.append(loss)
            if loss < best_loss:
                best_loss, best_w, best_b = loss, w.copy(), b
            if step < self.epochs:
                rate = self.learning_rate / np.sqrt(step + 1.0)
                w -= rate * grad_w
                b -= rate * grad_b
        # Nonsmooth descent can oscillate; retain the best observed iterate.
        self.coef_, self.intercept_ = best_w, best_b
        self.objective_ = best_loss
        self.loss_history_ = np.asarray(history)
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        if not hasattr(self, "coef_"):
            raise RuntimeError("fit must be called before prediction")
        X = feature_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count from training")
        scores = X @ self.coef_ + self.intercept_
        if not np.all(np.isfinite(scores)):
            raise ValueError("nonfinite prediction scores; rescale inputs")
        return scores

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0.0, 1, -1)


def main():
    import sklearn
    from sklearn.datasets import make_blobs
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, labels = make_blobs(n_samples=240, centers=[[-2, -1], [2, 1]],
                          cluster_std=1.1, random_state=42)
    y = 2 * labels - 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_train)
    train, test = scaler.transform(X_train), scaler.transform(X_test)
    model = LinearSVM(C=1.0, learning_rate=0.1, epochs=10000).fit(train, y_train)
    # SVC sums hinge losses: C_library = C_mean / n_train matches our objective.
    reference = SVC(kernel="linear", C=model.C / len(train), tol=1e-9).fit(train, y_train)
    reference_loss = objective_and_subgradient(
        train, y_train, reference.coef_[0], reference.intercept_[0], model.C)[0]
    prediction = model.predict(test)
    reference_prediction = reference.predict(test)
    report = {
        "hypothesis": "Correct mean-hinge subgradients approach a matched linear SVC objective on standardized synthetic data.",
        "configuration": {
            "generator": "sklearn.datasets.make_blobs (synthetic)", "n_samples": 240,
            "centers": [[-2, -1], [2, 1]], "cluster_std": 1.1,
            "seed": 42, "split_seed": 42, "stratified_test_fraction": 0.25,
            "train_rows": len(train), "test_rows": len(test), "C_mean": model.C,
            "C_library": model.C / len(train), "learning_rate": model.learning_rate,
            "epochs": model.epochs, "schedule": "learning_rate / sqrt(step + 1)",
            "reference_tolerance": 1e-9,
            "preprocessing": "StandardScaler fitted on training rows only",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__},
        "result": {
            "initial_objective": float(model.loss_history_[0]),
            "scratch_objective": model.objective_, "reference_objective": reference_loss,
            "objective_gap": model.objective_ - reference_loss,
            "scratch_test_accuracy": float(accuracy_score(y_test, prediction)),
            "reference_test_accuracy": float(accuracy_score(y_test, reference_prediction)),
            "prediction_agreement": float(np.mean(prediction == reference_prediction)),
        },
        "interpretation_candidate": "Use the objective gap and prediction agreement as evidence for this controlled numerical example only.",
        "review_status": "pending author review",
        "limitation": "One synthetic split and finite iteration budget; no convergence certificate, kernels, dual coefficients, calibration or production validation.",
    }
    output = Path(__file__).resolve().parent / "outputs" / "linear_solver.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Ignored run record: {output}")


if __name__ == "__main__":
    main()
