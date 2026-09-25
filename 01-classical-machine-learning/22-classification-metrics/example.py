"""Compare a majority baseline with logistic regression on synthetic labels."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from from_scratch import classification_metrics


SEED = 22


def report(name, y_true, y_pred):
    metrics = classification_metrics(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    checks = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    expected_matrix = [
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ]
    if not np.array_equal(matrix, expected_matrix):
        raise AssertionError("Confusion counts disagree with scikit-learn")
    if not all(np.isclose(metrics[key], value) for key, value in checks.items()):
        raise AssertionError("Classification metrics disagree with scikit-learn")
    print(
        f"{name}: TN={metrics['tn']} FP={metrics['fp']} "
        f"FN={metrics['fn']} TP={metrics['tp']}"
    )
    print("  " + "  ".join(f"{key}={metrics[key]:.4f}" for key in checks))


def main():
    X, y = make_classification(
        n_samples=4000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.95, 0.05],
        flip_y=0,
        class_sep=1.2,
        random_state=SEED,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

    print("Synthetic binary data; positive class=1; fixed threshold=0.5")
    print(f"Train: {len(y_train)} rows, {int(y_train.sum())} positives")
    print(f"Test: {len(y_test)} rows, {int(y_test.sum())} positives")
    report("Always-negative baseline", y_test, np.zeros_like(y_test))
    report("Logistic regression", y_test, y_pred)


if __name__ == "__main__":
    main()
