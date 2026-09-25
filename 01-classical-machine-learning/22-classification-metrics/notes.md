# Notes: classification metrics from the confusion matrix

## Count first, then summarize

Let class 1 mean the event of interest. Use true labels for rows and predicted labels for columns:

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | TN | FP |
| Actual 1 | FN | TP |

The four cells are disjoint, so `TN + FP + FN + TP = N`. Stating the positive class is essential: swapping its meaning changes precision and recall. The matrix records absolute workload and missed cases, while a scalar metric loses some of that information.

| Metric | Formula | Conditioning population | Main question |
|---|---|---|---|
| Accuracy | `(TP + TN) / N` | All examples | How often is the decision correct? |
| Precision | `TP / (TP + FP)` | Predicted positives | How reliable are positive decisions? |
| Recall (sensitivity, TPR) | `TP / (TP + FN)` | Actual positives | How many positive cases were found? |
| F1 | `2TP / (2TP + FP + FN)` | Positive decisions and actual positives | How well do precision and recall balance? |

Precision and recall have different denominators. A classifier can be selective and precise while missing many positives. Conversely, flagging most rows can raise recall while burdening a review team with false alarms.

F1 is the harmonic mean `2PR/(P+R)` when `P+R > 0`. It falls sharply if either precision or recall is small. Its count form shows that true negatives do not enter directly. Adding correctly rejected negatives while TP, FP, and FN stay fixed leaves F1 unchanged, even though accuracy rises. F1 is a symmetric summary of precision and recall, not a measured cost function or a universal choice for imbalanced data.

## Undefined denominators and reporting policy

If no row is predicted positive, `TP + FP = 0` and precision is mathematically undefined. If the evaluation set contains no actual positives, `TP + FN = 0` and recall is undefined. In the educational implementation, both cases are **reported as 0** to match scikit-learn with `zero_division=0`. If `2TP + FP + FN = 0`, F1 is also reported as 0. This convention allows a stable table; it does not mean the undefined rate was empirically measured as zero. The code rejects empty input entirely and explicitly requires one-dimensional labels in {0, 1}.

For a tiny positive count, report both the rate and its denominator. Precision of 1.0 on one alert and precision of 1.0 on ten thousand alerts are very different amounts of evidence. Uncertainty and subgroup variation require additional analysis.

## Why prevalence matters

In a set with 95% negatives, an always-negative classifier gets 95% accuracy while detecting no positives. Accuracy is calculated correctly; it emphasizes the majority class because every row contributes equally. Compare it with the majority baseline and with the cost of errors before using it for model selection.

Let `pi = P(Y=1)` be prevalence, `TPR = P(pred=1 | Y=1)`, and `FPR = P(pred=1 | Y=0)`. Then, when the denominator is nonzero,

`precision = (TPR * pi) / (TPR * pi + FPR * (1 - pi))`.

Thus, if TPR and FPR really stay fixed, lowering prevalence generally lowers precision. In deployment, those conditional rates may also shift; the equation is a conditional thought experiment, not a guarantee.

## Thresholds and decision costs

A score becomes a label under the rule `pred=1 if score >= t`. On one fixed scored dataset, increasing `t` can only remove predicted positives: TP and FP cannot increase, while FN and TN cannot decrease. Recall therefore cannot increase as the threshold rises (assuming actual positives exist). Precision has **no guaranteed monotonic direction**: removing a true positive can lower it, and removing a false positive can raise it.

A threshold should reflect the action attached to a positive prediction. For example, with limited review capacity one might seek high precision subject to a minimum recall; with costly misses one might reverse the constraint. F1 assumes neither a review capacity nor a monetary cost. Select thresholds using validation data, then evaluate the chosen policy once on untouched test data. Tuning a threshold on the test set makes that test result optimistic.

The Day 22 code fixes `t=0.5` before test evaluation and makes no threshold recommendation. Day 23 covers threshold tuning, ROC/PR curves, and calibration.

## Executed example

**Hypothesis:** for a rare positive class, an always-negative baseline can show high accuracy while missing every positive; a fitted classifier may detect positives, but its error counts must be inspected.

**Configuration:** `example.py` generated 4,000 synthetic rows using scikit-learn `make_classification` with 10 features, 6 informative and 2 redundant features, class weights [0.95, 0.05], `flip_y=0`, `class_sep=1.2`, and seed 22. A stratified split reserved 3,000 rows for training and 1,000 for testing (150 and 50 positives respectively). A `StandardScaler` plus default-regularized `LogisticRegression(max_iter=1000)` was fit on training data. The 0.5 threshold and always-negative baseline were fixed before the single test evaluation. The scratch results were checked against scikit-learn on identical labels.

**Result:**

| Decision rule | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always negative | 950 | 0 | 50 | 0 | 0.9500 | 0.0000* | 0.0000 | 0.0000 |
| Logistic regression, `t=0.5` | 946 | 4 | 35 | 15 | 0.9610 | 0.7895 | 0.3000 | 0.4348 |

*No positive predictions were made; precision is undefined mathematically and reported as zero under the stated convention.*

**Interpretation candidate — author review:** in this single synthetic split, baseline accuracy concealed complete failure to detect positives. The fitted classifier found 15 of 50 positives while generating four false alarms, yet missed 35 positives. The counts convey more of that trade-off than the 0.011 accuracy difference.

**Limitations:** this is one seed, one split, one synthetic data generator, one fitted model, and one fixed threshold. It is neither a benchmark nor evidence of production performance, calibration, group reliability, or a preferred operating point. The generator's class distribution and feature structure may not represent an application. The interpretation above requires the author's review before being treated as a personal conclusion.

## Common mistakes and next questions

- Reporting F1 alone hides whether misses or false alarms dominate. Show precision, recall, and counts together.
- Comparing precision across samples with different prevalence can mislead; document sampling and target prevalence.
- Ignoring absolute FP and FN volume can misstate operational burden.
- Reversing the positive label changes the business meaning of the rates.
- Treating `0.5` as inherently optimal confuses a model default with a decision policy.
- Evaluating on training data or choosing a threshold on the test set breaks the intended generalization check.
- A global score may hide weak performance in important groups or time periods.

The separate [visual lab](VISUAL_GUIDE.md) executed a validation-grid threshold comparison and one held-out test evaluation. Its results and limitations are recorded there.

**Suggested, not executed here:** repeat across seeds or representative time periods; check subgroup counts and prevalence drift. Each would need its own recorded configuration, results, and limitations.
