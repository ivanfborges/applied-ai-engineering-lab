# Decision tree interview questions

## How is a numeric split chosen?

For each feature, inspect thresholds between distinct training values. For every legal split, compute parent impurity minus the child impurities weighted by their row counts. Choose the largest positive local reduction and repeat recursively. This greedy search need not produce the globally best tree. The [scratch code](from_scratch.py) shows this directly.

## Why weight child impurity?

Without weights, a tiny pure leaf could appear as valuable as a large well-separated group. A child with \(N_L\) of \(N\) rows contributes \(N_L/N\) of its impurity to the split score.

## How do Gini and entropy differ?

Gini is \(1-\sum p_k^2\); entropy is \(-\sum p_k\log_2p_k\). Both are zero at purity and peak for balanced classes. Entropy reduction is information gain. They may choose different splits, so any claim that they are equivalent requires an actual comparison.

## Why can a deep tree overfit?

Repeated splits can isolate mislabeled or rare training observations. Training impurity falls, but the resulting tiny regions can be unstable for new rows. A large train-to-validation gap is a diagnostic, not proof of the sole cause; also check split design and leakage.

## How do pre-pruning and cost-complexity pruning differ?

Depth and minimum-leaf controls stop growth. Cost-complexity pruning starts with a grown tree and trades training leaf impurity against a penalty for leaf count. Choose either control with validation or cross-validation, not by repeatedly inspecting the test set.

## Are a tree's probabilities and feature importances trustworthy?

A leaf fraction is an empirical training proportion and may be extreme in small leaves. Check calibration for probability-sensitive decisions. Impurity importance reflects training split use; it can favor features with many candidate thresholds and redistribute across correlated features. Neither quantity establishes causality.

## Why is scaling often unnecessary, and what still needs care?

A monotonic transformation preserves the order of a numeric feature and usually the possible partitions. Missing values, categorical encoding, label imbalance, and feature availability still need explicit handling.

## When would you use a single tree in an AI system?

It can serve as an inspectable tabular baseline or a low-cost routing rule after validation. For predictive quality, compare against simpler models and tree ensembles on the same leakage-safe split. Monitor rule stability, leaf sizes, and relevant class-specific or cost metrics.
