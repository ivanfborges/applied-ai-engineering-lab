# Notes: ranking, probabilities, and decisions

## Ranking metrics

For score $s$ and threshold $t$, predict positive when $s \ge t$. At each threshold, the ROC curve plots $\mathrm{TPR}=TP/(TP+FN)$ against $\mathrm{FPR}=FP/(FP+TN)$. ROC-AUC is the area under that curve. Equivalently, it is the share of positive-negative pairs where the positive has a higher score, with ties contributing one half. Both classes must be present. The educational implementation in [from_scratch.py](from_scratch.py) uses all pairs, so its work grows as $O(N_+N_-)$.

A precision-recall (PR) curve plots $\mathrm{precision}=TP/(TP+FP)$ against recall, which equals TPR. Here **AP** means scikit-learn's noninterpolated average precision: $\sum_n (R_n-R_{n-1})P_n$. Trapezoidal area under the sampled PR curve is a different calculation and can give a different number. State which one was used instead of calling both simply “PR-AUC.” A random-ranking AP baseline is approximately the positive prevalence.

With prevalence $\pi=P(Y=1)$, Bayes' rule gives

$$
\mathrm{precision} =
\frac{\mathrm{TPR}\,\pi}
{\mathrm{TPR}\,\pi+\mathrm{FPR}(1-\pi)}.
$$

For example, at 1% prevalence, TPR 0.90 and FPR 0.05 imply precision about 0.154. This is an arithmetic illustration, not a measured dataset result. ROC-AUC remains a valid ranking measure under imbalance, but its FPR denominator can conceal a large absolute alert volume. Compare PR summaries only with their prevalence and evaluation population.

## Probability quality

Calibration asks whether cases assigned probability $p$ have event frequency near $p$. A reliability table or diagram compares mean score with observed event rate in bins. The [example](example.py) prints five equal-frequency bins for both original and transformed test scores; the bins are summaries, not a proof of individual-level calibration. Quantile bins provide similar counts but can cover very different score widths, and sparse positives make event rates noisy.

The Brier score is the mean squared probability error, $N^{-1}\sum_i(p_i-y_i)^2$; lower is better. It combines calibration and discrimination effects, so Brier alone cannot diagnose which changed. Log loss more strongly penalizes confident wrong probabilities. A strictly increasing transformation such as $p^4$ preserves score ordering and therefore ROC-AUC and AP, while changing the numerical probability meaning. The experiment deliberately applies this transform after training; it is not a proposed calibration method.

Sigmoid (Platt) scaling fits a low-dimensional sigmoid mapping; isotonic regression learns a more flexible monotone mapping and can overfit smaller calibration sets. Either requires calibration data separate from its evaluation data, or a suitable cross-validated design. Recheck calibration after prevalence or covariate shifts. A globally calibrated model can still be poor in a relevant segment.

## Decision policy

A threshold changes precision, recall, F1, and alert volume without refitting the model. The example selects the observed validation score cutoff with the highest recall among cutoffs with at least one alert and precision at least 0.60. Ties favor higher precision, then the higher threshold. Its test precision is free to fall below 0.60: a validation constraint is an estimate, not a test guarantee. The test set must not be used to adjust that threshold.

For calibrated probabilities and only two constant error costs, predicting positive has expected cost $(1-p)C_{FP}$, while predicting negative has cost $pC_{FN}$. Choose positive when $p > C_{FP}/(C_{FP}+C_{FN})$. This simplified cutoff assumes correct probabilities, fixed costs, no review capacity limit, and no benefit or intervention effects beyond those costs. In real operations, capacity, segment impacts, delayed labels, and changing prevalence may require a different policy, including top-$K$ review.

## Executed synthetic experiments

### 1. Ranking-preserving probability distortion

- **Hypothesis:** Raising strictly positive scores to the fourth power preserves ranking summaries but can change probability error.
- **Configuration:** `make_classification` with 5,000 rows, 12 features (5 informative, 2 redundant), requested class weights 95%/5%, class separation 1.3, label flip rate 0.01, seed 23. Stratified train/validation/test sizes were 3,000/1,000/1,000 with split seeds 23 and 24. A `StandardScaler` and `LogisticRegression(max_iter=1000)` were fitted on training data. The same 1,000 test labels were evaluated with original scores and scores raised to power four.
- **Result:** There were 54 positives in the test split. ROC-AUC was 0.9304 for both score versions; AP was 0.5875 for both. Brier score was 0.0321 for original scores and 0.0430 for transformed scores. In the highest quantile calibration bin, original mean prediction/event rate was 0.237/0.235; transformed was 0.051/0.235.
- **Interpretation candidate — author review required:** This controlled transformation illustrates that identical ranking can coexist with different probability quality. The fourth-power scores understate risk in the highest bin here.
- **Limitation:** This is one synthetic generator, seed, fitted model, and test split. Binned rates are noisy, and a change in Brier score does not isolate calibration from all other effects.

### 2. Validation-constrained threshold

- **Hypothesis:** A threshold chosen for validation precision at least 0.60 can yield a different precision and alert count on untouched test data.
- **Configuration:** The same trained model and split above. All observed validation score cutoffs were checked; eligible cutoffs needed at least one alert and precision at least 0.60. Highest recall won, with deterministic ties by precision and threshold. That cutoff was then applied once to test scores.
- **Result:** Selected threshold 0.363422. Validation: TP=26, FP=16, FN=28, TN=930; 42 alerts, precision 0.6190, recall 0.4815. Test: TP=29, FP=12, FN=25, TN=934; 41 alerts, precision 0.7073, recall 0.5370.
- **Interpretation candidate — author review required:** The selected policy met its validation constraint and happened to exceed that precision on this test split. The observed difference illustrates sampling variability rather than a guaranteed improvement.
- **Limitation:** The 0.60 target is illustrative, not a stated business requirement. It omits cost, review capacity, confidence intervals, and population shift. A single split cannot establish policy stability.

## Common mistakes and next experiments

Avoid tuning on the test set, treating 0.5 as universal, equating AP with every PR-area calculation, inferring calibration from AUC, and reporting only rates when alert counts drive workload. Before using a threshold operationally, estimate uncertainty and inspect performance by relevant segment.

**Suggested, not executed here:** vary prevalence while holding class-conditional score distributions fixed; compare sigmoid and isotonic calibration on a dedicated calibration split; evaluate a top-$K$ review policy under a stated capacity and cost model.
