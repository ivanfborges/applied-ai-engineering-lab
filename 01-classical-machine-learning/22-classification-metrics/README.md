# Classification Metrics I

Day 22 studies how a binary classifier's decisions become a confusion matrix and how accuracy, precision, recall, and F1 compress that matrix. Class **1** is the positive class throughout this topic.

| Actual / predicted | Negative (0) | Positive (1) |
|---|---:|---:|
| Negative (0) | True negative (TN) | False positive (FP) |
| Positive (1) | False negative (FN) | True positive (TP) |

A single accuracy number can conceal missed rare positives. In alerting, fraud screening, document routing, or binary quality gates, the useful measure depends on the consequences of false alarms and misses. Inspecting the counts first also reveals the volume of downstream work.

## Read and run

| File | Purpose |
|---|---|
| [notes.md](notes.md) | Formulas, denominator rules, threshold and prevalence effects, trade-offs, and an executed experiment |
| [example.py](example.py) | Fixed-threshold logistic regression versus an always-negative baseline on synthetic data |
| [from_scratch.py](from_scratch.py) | Educational NumPy implementation of the four counts and derived metrics |
| [tests/test_metrics.py](tests/test_metrics.py) | Count, edge-case, input-validation, and scikit-learn parity checks |
| [interview_questions.md](interview_questions.md) | Practical evaluation questions and concise answers |
| [visualize_classification_metrics.py](visualize_classification_metrics.py) | Generate nine synthetic visual views, including an animated threshold sweep |
| [VISUAL_GUIDE.md](VISUAL_GUIDE.md) | Visual questions, selected previews, and executed experiment records |
| [tests/test_visual_lab.py](tests/test_visual_lab.py) | Threshold, F1, prevalence, and imbalance checks |

Use the shared dependencies in [pyproject.toml](../../pyproject.toml). From the repository root:

```bash
python -m pip install -e ".[dev]"
python 01-classical-machine-learning/22-classification-metrics/example.py
python 01-classical-machine-learning/22-classification-metrics/from_scratch.py
python -m pytest -q 01-classical-machine-learning/22-classification-metrics/tests
```

`example.py` and `from_scratch.py` print to the terminal and create no assets. The example generates all data with `make_classification`, uses a stratified 3,000/1,000 train/test split, fits scaling on training data only, and applies a fixed 0.5 threshold. It checks the NumPy metrics against scikit-learn on the same predictions. The scratch implementation accepts nonempty one-dimensional 0/1 labels and uses zero when precision or recall has a zero denominator; it is for learning, not a replacement for library metrics.

## Visual lab

Run the [visual generator](visualize_classification_metrics.py) from the repository root:

```bash
python 01-classical-machine-learning/22-classification-metrics/visualize_classification_metrics.py
```

It creates seven PNGs, one GIF, and one offline interactive Plotly HTML file in `assets/`. The [visual guide](VISUAL_GUIDE.md) names the question answered by each output and records the executed synthetic experiments, interpretation candidates, and limitations. Four small previews are deliberately public; the other outputs are regenerated locally.

![Confusion-matrix outcomes linked to metric denominators](assets/confusion_matrix_foundation.png)

![Threshold movement changes predictions, errors, and metrics](assets/threshold_tradeoff.gif)

The animation and threshold curves use validation data; a validation-selected threshold is evaluated once on held-out test data. The maximum-F1 threshold is specific to this synthetic grid and is not an operational recommendation.

## What the executed example shows

With seed 22 and 5% positive synthetic data, the always-negative baseline obtained 0.9500 accuracy and 0 recall. Logistic regression obtained 0.9610 accuracy, 0.7895 precision, 0.3000 recall, and 0.4348 F1. Its confusion counts were TN=946, FP=4, FN=35, TP=15. The complete hypothesis, setup, results, **interpretation candidate for author review**, and limitations are in [the experiment record](notes.md#executed-example).

## Key takeaways

- Accuracy is the share of all decisions that are correct; compare it with class prevalence and a simple baseline.
- Precision conditions on predicted positives. Recall conditions on actual positives.
- F1 combines precision and recall, but ignores TN and does not encode operational error costs.
- A threshold changes the confusion matrix. This example describes one fixed threshold; threshold selection belongs on validation data, with final evaluation on held-out test data.
- Counts, class definition, prevalence, and downstream costs belong beside scalar scores.

This study follows [logistic regression](../21-logistic-regression/) and [validation boundaries](../17-validation-and-leakage/). The [roadmap](../../ROADMAP.md) continues with curves, threshold tuning, and calibration on Day 23.
