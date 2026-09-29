# KNN: geometry, validation, and practical limits

## The prediction rule

For a query `x`, compute its distance to each training row, select the `k` smallest distances, and vote on their labels. The educational implementation in [from_scratch.py](from_scratch.py) uses Euclidean distance and uniform votes. Its class-tie rule selects the first class in sorted order; equal distances are resolved by training-row order. Other implementations or weighting choices can differ.

For numerical features, Euclidean distance is

```text
d(x, z) = sqrt(sum_j (x_j - z_j)^2).
```

A large change in one coordinate can dominate the sum. Standardizing a feature with training mean `mu_j` and training standard deviation `s_j` gives `(x_j - mu_j) / s_j`. This changes the metric: each coordinate contributes in units of its training spread. It does not prove that all standardized coordinates are equally useful. A noisy coordinate can still distort neighbors. For mixed categorical and numerical data, missing values, sparse text, or embeddings, representation and distance need separate design and validation.

## Bias, variance, and votes

With `k=1`, the decision can closely follow local noise or mislabeled rows. Increasing `k` smooths predictions but may erase a narrow class region. Class imbalance can make uniform votes favor the majority class even when a minority example is close. Distance-weighted votes change that trade-off. The fraction of positive neighbors is a local score, but should not be treated as a calibrated probability without checking calibration.

Choose `k`, preprocessing, metric, and any weighting using training-only cross-validation or a validation set. A `Pipeline` keeps scaling within each fit, including each cross-validation fold. Reserve the test split for evaluating the selected procedure. This page's fixed-`k` comparison illustrates the geometry; it is not a model-selection study.

## Dimensionality and systems costs

For many roughly independent, similarly distributed coordinates, pairwise distances can concentrate as dimension grows: differences between “near” and “far” become less pronounced relative to the total distance. This is one form of the curse of dimensionality, not a universal theorem for every dataset. Irrelevant features also increase the chance that the selected neighbors are near for the wrong reasons. Feature selection or a justified lower-dimensional representation can help, but must be fitted within the training boundary.

A simple exact implementation stores `n` training rows with `d` features, taking `O(nd)` feature storage. A direct query computes `n` distances at `O(nd)` arithmetic cost, then selects `k` neighbors. Tree indexes may help in favorable lower-dimensional settings; their advantage often weakens in high dimension. Approximate nearest-neighbor search trades retrieval accuracy for latency or memory and needs its own evaluation. Changes to the reference set can change predictions without retraining model coefficients.

## Executed synthetic comparison

**Hypothesis.** An independent feature with much larger numerical units will dominate raw Euclidean distance; fitting a scaler on training data may improve KNN's ability to use the signal feature.

**Configuration.** [example.py](example.py) generated 400 synthetic rows with seed 24. The predictive coordinate is standard normal; the independent nuisance coordinate has standard deviation 100. The binary label is the sign of the predictive coordinate plus independent normal noise with standard deviation 0.35. A stratified split used 300 training and 100 test rows. Both models used uniform-vote Euclidean KNN with fixed `k=5`; one used raw features, the other used `StandardScaler` fitted on training rows through a pipeline.

**Result.** Running the example yielded test accuracy 0.590 for raw KNN and 0.820 for scaled KNN. The scratch script returned `low` and `high` on its two one-dimensional queries.

**Interpretation candidate — author review required.** In this constructed setting, the unit difference appears to obscure the predictive coordinate under raw Euclidean distance; train-fitted scaling changes the neighborhoods and coincides with higher held-out accuracy.

**Limitations.** This is one seed and one split on deliberately constructed data, with no uncertainty interval or tuned `k`. Scaling does not remove the independent nuisance feature, and the result does not establish superiority on real data or in high dimensions. The test comparison is descriptive; choosing between procedures for deployment would require a development-set selection policy followed by fresh test evaluation.

## Common mistakes and practical checks

- Scaling the whole dataset before splitting or cross-validation leaks test or validation statistics.
- Comparing distance-based models without documenting feature units and metric makes the comparison hard to interpret.
- Reporting accuracy alone can hide minority-class errors; inspect a confusion matrix and task-relevant metrics.
- Treating neighbor votes as trustworthy probabilities skips calibration assessment.
- Assuming an exact search remains cheap as the training set or query rate grows overlooks inference cost.
- Dropping features or using dimensionality reduction based on test performance contaminates evaluation.

An interview-ready summary: KNN is a local, instance-based predictor. Its behavior comes from the representation and distance as much as from `k`; preprocessing must stay inside the training boundary, and both dimensionality and query cost matter in production.
