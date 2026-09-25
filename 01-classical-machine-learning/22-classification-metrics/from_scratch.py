"""Educational binary classification metrics derived from four counts."""

import numpy as np


def classification_metrics(y_true, y_pred):
    """Return binary counts and metrics with class 1 designated positive.

    Undefined precision or recall is set to zero. Empty inputs are rejected.
    """
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("y_true and y_pred must have equal, nonzero lengths")
    if not np.isin(actual, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("y_true and y_pred must contain only binary labels 0 and 1")

    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    n = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


if __name__ == "__main__":
    result = classification_metrics([1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 0])
    for name, value in result.items():
        print(f"{name}: {value}")
