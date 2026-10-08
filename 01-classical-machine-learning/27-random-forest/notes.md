# Random Forest notes

## Why the extra randomness matters

A decision tree greedily partitions the training data. A small change in rows can change an early split and all its descendants. Bagging fits multiple trees on different bootstrap samples and averages their predictions. If one strong predictor appears at the root of almost every tree, their errors may remain similar. Random Forest also restricts split search to a newly sampled feature subset at **each node**, creating more diversity. Restricting too aggressively can weaken individual trees; `max_features` controls this trade-off.

For regression predictions from \(B\) trees with equal variance \(\sigma^2\) and equal pairwise correlation \(\rho\), the average has variance

\[
\operatorname{Var}(\bar T)=\sigma^2\left(\rho+\frac{1-\rho}{B}\right).
\]

This identity follows by expanding the variance of a sum under those assumptions. It describes prediction variance at a fixed input, not a guaranteed change in classification accuracy. As \(B\) grows, the correlation term remains. Bootstrap sampling and feature subsets aim to lower correlation while keeping useful trees.

## Bootstrap and out-of-bag predictions

For \(n\) training rows, one tree draws \(n\) indices **with replacement**. The probability that a given row is never drawn is \((1-1/n)^n\), approaching \(e^{-1}\approx0.368\). Thus about 63.2% of original rows are represented at least once for large \(n\), despite the bootstrap sample containing \(n\) draws.

Let \(O_i\) be the set of trees whose bootstrap samples omit row \(i\). An OOB classifier aggregates predictions from \(O_i\) only, then scores the training rows that have OOB predictions. The scikit-learn example uses `oob_score_`, whose classifier default is accuracy. With too few trees, some rows may receive no OOB prediction; the scratch example reports coverage explicitly. OOB uses only training data and is separate from the held-out test score.

OOB is a reasonable internal check when independent rows are sampled from a stable population. Patient visits, customer histories, repeated documents, or future observations require group-aware or time-aware validation. Fit preprocessing inside the training boundary and keep a final test set for evaluation. Repeatedly selecting hyperparameters against OOB also makes it a tuning signal, so it is no longer an untouched final assessment.

## Feature importance answers different questions

For a classification split at node \(t\), Gini impurity is \(1-\sum_k p_k^2\). The impurity decrease is the parent impurity minus the child impurities weighted by their row counts. Mean decrease in impurity (MDI) adds the weighted decreases attributed to a feature across the fitted trees and normalizes the result. It is fast, but uses training splits and can favor variables with many candidate thresholds.

For a score where larger is better, permutation importance is \(M_{\text{baseline}}-M_{\text{shuffled }j}\). The example computes it on untouched test rows after fitting, using accuracy as the score. This measures how much that model and metric depend on feature \(j\) under a particular shuffle. With correlated features, another feature may substitute for the shuffled one; shuffling can also create implausible combinations. Neither method measures causal effect or proves a feature should be collected in production.

## Applications and limits

Random Forest is a practical nonlinear baseline for numeric tabular data with interactions. It usually needs no feature scaling, though missing values, categories, leakage, and class imbalance still require deliberate handling. Many deep trees increase training cost, memory, and prediction latency. Trees do not extrapolate smoothly beyond observed feature regions. Classification scores may require calibration before a threshold-driven decision; accuracy alone may be inadequate for rare events.

Compared with a single tree, the forest sacrifices a compact rule for stability. Compared with ordinary bagging, it also samples features at each split. Extra Trees add more split randomness; gradient boosting builds trees sequentially to correct previous errors. These are modeling choices to compare with an appropriate validation design, not a fixed performance ranking.

Common mistakes include describing 63.2% as the number of bootstrap *draws*, treating OOB as a universal replacement for validation, reading feature importance as causality, and assuming that more trees prevent every kind of overfitting. The scratch implementation's stump samples features once because it has one node; this should not be generalized to full forests.

## Executed synthetic example

**Hypothesis:** A forest can use the known signal features in a noisy nonlinear synthetic task; OOB and held-out accuracy provide distinct checks, while feature-importance rankings depend on the method.

**Configuration:** `example.py` generates 800 rows from five independent uniform numeric features. The target is an axis-aligned rule using `signal_a`, `signal_b`, and `signal_c`, with 10% independently flipped labels. `noise_d` and `noise_e` do not enter the rule. A seeded stratified 75/25 train/test split is used. The model has 150 trees, `max_features="sqrt"`, bootstrap and OOB enabled, and `min_samples_leaf=2`. Permutation importance uses test accuracy, 10 repeats, and a fixed seed.

**Result:** The executed command printed OOB accuracy 0.903 on training rows and held-out accuracy 0.855 on 200 test rows. MDI values were signal_a 0.501, signal_b 0.167, signal_c 0.199, noise_d 0.065, and noise_e 0.068. Held-out permutation accuracy decreases (mean +/- repeat SD over 10 shuffles) were signal_a +0.231 +/- 0.022, signal_b +0.099 +/- 0.021, signal_c +0.130 +/- 0.021, noise_d +0.001 +/- 0.002, and noise_e -0.001 +/- 0.002.

**Interpretation candidate for author review:** In this single run, both methods ranked the three rule features above the two unused features, while OOB accuracy exceeded held-out accuracy by 0.048. This is consistent with the known generator, but the gap is a split-specific observation and neither ranking establishes causal importance.

**Limitation:** One seed and one data generator cannot establish a stable effect size, deployment performance, or universal ranking. The test set is used here for demonstration, not model selection.

## Executed stump-mechanics check

**Hypothesis:** The educational stump ensemble should produce OOB predictions for the synthetic training rows when fitted with enough bootstrap learners.

**Configuration:** `from_scratch.py` generated 240 rows and four independent uniform numeric features with a nonlinear rule and 10% label flips. It fitted 80 randomized stumps, sampling two candidate features per stump, with seed 27.

**Result:** The command printed OOB coverage 1.000 and OOB accuracy 0.796 on covered training rows.

**Interpretation candidate for author review:** Every row received at least one eligible OOB vote in this run. The accuracy describes this one-split ensemble on this generated sample.

**Limitation:** Stumps cannot represent the same interactions as recursive trees; this result should not be compared as a benchmark against the scikit-learn example, which uses a different generator and model.

## Further experiments to run

- Vary `max_features` and record tree agreement alongside accuracy to examine the strength/diversity trade-off.
- Add a near-duplicate signal feature and compare both importance measures; correlated predictors may split credit.
- Vary tree count and record OOB coverage and score variability over independent seeds.

These are proposed experiments, with no results claimed here.

