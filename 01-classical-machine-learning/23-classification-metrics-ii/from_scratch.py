"""Small, educational binary ranking and threshold calculations."""

import numpy as np


def _binary_scores(y_true, scores):
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if len(labels) == 0 or len(labels) != len(values):
        raise ValueError("labels and scores must have equal, nonzero lengths")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only binary values 0 and 1")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return labels, values


def pairwise_roc_auc(y_true, scores):
    """Fraction of positive-negative pairs correctly ordered; ties count half."""
    labels, values = _binary_scores(y_true, scores)
    positives = values[labels == 1]
    negatives = values[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("ROC-AUC requires both classes")
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def threshold_metrics(y_true, scores, threshold):
    """Return counts and retrieval metrics for score >= threshold."""
    labels, values = _binary_scores(y_true, scores)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    predicted = values >= threshold
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision,
                recall=recall, f1=f1, alerts=tp + fp)


def select_threshold(y_true, scores, min_precision):
    """Maximize validation recall subject to precision and at least one alert.

    Resolve ties by precision, then by the higher threshold. Search observed
    score cutoffs, which are all distinct nonempty prediction sets.
    """
    labels, values = _binary_scores(y_true, scores)
    if not np.isfinite(min_precision) or not 0 < min_precision <= 1:
        raise ValueError("min_precision must be in (0, 1]")
    if not np.any(labels == 1):
        raise ValueError("threshold selection requires positive examples")
    candidates = []
    for threshold in np.unique(values):
        metrics = threshold_metrics(labels, values, threshold)
        if metrics["alerts"] and metrics["precision"] >= min_precision:
            candidates.append((metrics["recall"], metrics["precision"], threshold))
    if not candidates:
        raise ValueError("no nonempty threshold satisfies minimum precision")
    return float(max(candidates)[2])


if __name__ == "__main__":
    labels = [1, 0, 1, 0]
    scores = [0.8, 0.8, 0.6, 0.2]
    print(f"Pairwise ROC-AUC: {pairwise_roc_auc(labels, scores):.3f}")
    print(f"At threshold 0.6: {threshold_metrics(labels, scores, 0.6)}")

