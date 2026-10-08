# Decision tree notes

## From a rule to a partition

At a node containing rows \(D\), a candidate rule \(x_j \le t\) sends rows to \(D_L\) or \(D_R\). For numeric features, candidate thresholds can sit between consecutive distinct observed values. The algorithm chooses the feature and threshold with the largest *local* impurity reduction, then repeats within each child. The rule at the root can change after a modest training-data change, and every downstream rule depends on that choice.

A leaf predicts its majority training class. Its empirical class fraction can also be reported as a score, but small leaves can give extreme fractions with high sampling uncertainty. Evaluate calibration if decisions depend on probability quality.

## Gini, entropy, and weighted gain

For class proportions \(p_1,\ldots,p_K\) at a node:

\[
G(D)=1-\sum_{k=1}^K p_k^2,\qquad
H(D)=-\sum_{k:p_k>0}p_k\log_2p_k.
\]

Both are zero for a pure node. For balanced binary classes, Gini is \(0.5\) and entropy is \(1\) bit. The zero-probability terms in entropy are treated as zero by continuity.

For either impurity \(I\), a split's reduction is

\[
\Delta I=I(D)-\frac{|D_L|}{|D|}I(D_L)-\frac{|D_R|}{|D|}I(D_R).
\]

With entropy, this reduction is information gain. With Gini, it is Gini impurity reduction. Weighting by child size matters: a single pure row should not count as much as a large child. A reduction on training data does not establish that the split improves prediction on new data.

**Worked Gini split.** A parent with six positives and four negatives has Gini \(1-(0.6^2+0.4^2)=0.48\). Suppose a candidate split produces a left child with five positives and one negative, and a right child with one positive and three negatives. Their Gini values are \(5/18\) and \(3/8\). Weighted child Gini is \(6/10(5/18)+4/10(3/8)=19/60\), giving a reduction of \(49/300\), approximately \(0.1633\). Other candidate rules must still be compared.

The [scratch implementation](from_scratch.py) searches each distinct numeric midpoint and keeps the first maximal positive Gini gain. It rejects empty splits, respects a minimum leaf count and maximum depth, stops at pure nodes, and predicts a leaf majority. It uses no sample weights, missing-value strategy, categorical split search, pruning, or efficient sorted-scan algorithm.

## Controlling complexity

An unrestricted tree can isolate noisy labels and reach very high training accuracy. A depth cap limits sequential rules; a minimum leaf size prevents tiny terminal regions; minimum split size and maximum leaf count are other pre-pruning controls. These controls trade flexibility for stability. Their values should be selected using validation data or cross-validation, respecting groups and time order when applicable.

Cost-complexity pruning starts from a fitted tree and minimizes a criterion of the form

\[
R_\alpha(T)=\sum_{\ell\in\mathrm{leaves}(T)}
\frac{N_\ell}{N}I(\ell)+\alpha\,|\mathrm{leaves}(T)|.
\]

The first term is training leaf impurity; the second penalizes the number of leaves. Larger \(\alpha\) favors a smaller subtree. In scikit-learn, the training-only pruning path supplies candidate `ccp_alpha` values. The [example](example.py) chooses among them on a separate validation set. It breaks validation-score ties in favor of the larger alpha. The held-out test data plays no role in choosing alpha.

## Executed synthetic comparison

**Hypothesis.** With independent label noise, an unrestricted tree may fit individual flipped labels, while limiting or pruning its branches may improve held-out accuracy.

**Configuration.** Seed 26; 600 code-generated rows with two uniform features in \([-1,1]\). The clean binary label follows \(x_0<-0.45\) or \(x_0>0.1\) and \(x_1>-0.2\); each label is independently flipped with probability 0.12. Stratified splits yield 360 training, 120 validation, and 120 test rows. The metric is accuracy. All trees use Gini. The fixed pre-pruned tree has `max_depth=3` and `min_samples_leaf=10`. The unrestricted tree has default growth controls. The pruning path from training data produced 23 distinct alpha candidates; the best validation score was 0.858, with selected `ccp_alpha=0.003750`.

| Tree | Depth | Leaves | Training accuracy | Test accuracy |
|---|---:|---:|---:|---:|
| Unrestricted | 15 | 72 | 1.000 | 0.808 |
| Fixed pre-pruned | 3 | 7 | 0.878 | 0.867 |
| Validation-pruned | 7 | 13 | 0.906 | 0.892 |

**Interpretation candidate — for author review.** On this one split, the unrestricted tree's perfect training fit and lower test accuracy are consistent with fitting some label noise; the constrained trees have higher test accuracy. This does not show that the selected alpha is universally best, nor that pruning always helps.

**Limitations.** The rule, noise rate, seed, split, and metric are fixed and synthetic. The two constrained models were not compared through a common hyperparameter search. Accuracy is suitable only for this fairly balanced demonstration and does not cover cost-sensitive or probability-sensitive use. Repeated seeds, cross-validation, and a separate calibration analysis would test whether the pattern is stable. The author should review the interpretation before treating it as a personal conclusion.

## Use and limits

Trees can expose threshold logic and interactions in tabular classification. Monotonic rescaling of a numeric feature generally preserves the order of candidate partitions, so feature standardization is usually unnecessary. This does not solve missing values or inappropriate feature encoding. A single tree may be attractive where inspectable rules or low inference cost matter; compare it with a simple linear baseline and ensembles when predictive quality matters.

The split criterion optimizes a training surrogate, not an application cost. With imbalanced labels, inspect class-specific metrics and decision costs rather than accuracy alone. Impurity-based feature importance sums training impurity reductions; it can favor features with many potential thresholds, and correlated features can substitute for one another. It is neither a causal claim nor a complete explanation of prediction behavior. For regression, constant predictions within leaves also limit extrapolation beyond observed target patterns.

## Suggested next experiments

These remain **suggestions**, not executed results:

- Compare Gini and entropy using identical training and validation splits; inspect root rules as well as predictive metrics.
- Sweep maximum depth and minimum leaf size, recording both training and validation behavior.
- Repeat the pruning comparison over several seeds, then report variation rather than a single favorable split.
- Examine leaf size versus empirical probability extremity and evaluate calibration if probabilities matter.
