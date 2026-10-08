# ExtraTrees interview questions

**What changes relative to Random Forest?**

Threshold candidate generation. RF searches thresholds within a random feature
subset; ET proposes random thresholds and retains the best valid candidate.
Both can use bootstrap. The defaults alone do not define the algorithms.
[Official ensemble guide](https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees).

**Why can less optimized trees improve an ensemble?**

The variance of an average depends on covariance between members as well as
each member's variance. Under equal variance and equal pairwise correlation,
it is sigma^2[rho + (1-rho)/M]. Lower covariance can compensate for weaker
members, but greater bias can offset the benefit. The formula's probability
space and assumptions matter; see the [derivation](notes.md#variance-reduction-state-the-probability-space).

**What evidence would show that extra randomness helped?**

Compare threshold strategies with bootstrap and other controls fixed, using
identical evaluation rows. Inspect tree strength, disagreement, and the task's
ensemble metrics. Repeat samples or folds for stability. Our
[executed comparison](notes.md#executed-synthetic-comparison) gives mixed metric
rankings, not a universal winner or a variance estimate.

**Does high disagreement establish low error correlation?**

No. Disagreement counts differing hard predictions across rows. Error
correlation requires labels and a defined error statistic; training-set
prediction variance requires repeated fits. Random guessing can produce
disagreement while adding little useful information.

**Is class prediction equivalent to majority vote?**

Not when the ensemble averages leaf probabilities first. Probabilities 0.49,
0.49, and 0.99 average above 0.5, although two hard predictions are negative.
The scratch implementation and its tests expose this difference.

**Why allow a zero-gain split in the scratch model?**

XOR has no useful one-feature first split by Gini gain, but useful splits in
the children. Depth and leaf constraints limit growth while allowing that
interaction. The truth-table demo is training-only evidence.

**Can ExtraTrees use OOB scoring?**

Yes, with bootstrap enabled. A full-data tree omits no training row and cannot
supply OOB predictions. Even genuine OOB evaluation does not repair time or
group leakage.
[ET API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).

**How would you choose between models with similar scores?**

Define costs for false positives and negatives, inspect probability quality,
and compare latency, memory, slice performance, and stability under deployment
validation. Use development folds for tuning. A small observed score advantage
on one test split is not a basis for claiming general superiority.

**Where might this fit in an LLM application?**

A metadata-based quality classifier could route cases to review or fallback.
Use only features available at the routing decision, evaluate on future or
group-held-out cases as appropriate, and validate calibration if predicted
probabilities determine actions. This study implements no such deployment.
