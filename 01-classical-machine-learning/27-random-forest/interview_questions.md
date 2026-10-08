# Random Forest interview questions

## Why can a forest be more stable than one tree?

A tree can change its early splits after small training-data changes. A forest averages predictions from trees fitted on different bootstrap samples and candidate-feature subsets. Averaging reduces prediction variance when tree errors are imperfectly correlated. It does not guarantee better accuracy on every dataset.

## What is the difference between bagging and Random Forest?

Bagging changes the rows each tree sees by sampling with replacement. A standard Random Forest adds random candidate-feature selection at every split. If a few strong features dominate, this can reduce similarity among trees, though a very small subset may weaken each tree.

## Does a bootstrap tree train on only 63.2% of rows?

It receives as many draws as the training set has rows. For large samples, about 63.2% of original rows appear at least once, while duplicates fill the remaining positions. About 36.8% of original rows are omitted for that tree.

## How is an OOB prediction made, and when can it mislead?

For each training row, aggregate predictions only from trees whose bootstrap samples omitted that row. Score the resulting predictions against training labels. OOB does not protect against repeated entities crossing tree samples, temporal leakage, target leakage, or preprocessing fitted before the correct split. Few trees may also leave rows without OOB votes.

## Does scikit-learn classify by majority vote?

The classifier averages per-tree class probabilities and chooses the class with the largest mean. A hard majority vote is a useful conceptual simplification and is what the educational stump model implements; the two rules can disagree.

## What does feature importance measure?

Mean decrease in impurity credits features for reducing training impurity in fitted trees. Permutation importance measures the change in a chosen score on an evaluation set when one feature is shuffled. Both are model- and data-dependent; neither establishes causality. MDI may favor high-cardinality features, while correlated predictors can make individual permutation scores appear small.

## What changes when the number of trees or candidate features changes?

More trees generally reduce Monte Carlo variation, with more training, storage, and inference cost. Larger candidate subsets can strengthen individual trees but make them more alike. The useful setting depends on data and should be evaluated with a validation design that respects groups and time.

## How would you evaluate a forest for an AI-system quality gate?

Define the decision and cost of errors first. Split telemetry by time or entity when needed, keep features available at decision time, compare with a simple baseline, and choose metrics appropriate to prevalence and capacity. If the gate uses a probability threshold, check calibration and choose the threshold on validation data. Test on a final untouched period or group; inspect latency and drift before deployment.

