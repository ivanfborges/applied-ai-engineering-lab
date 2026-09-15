# Validation as an experiment about deployment

## What is being estimated?

For a fixed predictor f and deployment distribution P, the target risk is

$$
R_P(f) = \mathbb{E}_{(X,Y)\sim P}[L(f(X),Y)].
$$

An independent holdout estimates it by averaging losses:

$$
\widehat R(f) = \frac{1}{m}\sum_{i=1}^{m} L(f(x_i),y_i).
$$

Independence from model development and a relevant sampling distribution are
different requirements. An untouched random sample from last year's population
can still misrepresent next month's deployment. More rows reduce some sampling
noise; they do not correct a mismatched population.

Training estimates coefficients and transformation statistics. Validation also
influences the final system through feature choices, hyperparameters, thresholds,
early stopping and prompt edits. Test data must remain outside those decisions.
The point of a final evaluation is independence of decisions, not a literal
prohibition on rerunning an unchanged script to verify reproducibility.

For a simple three-way random split, reserve 20% for test, then 25% of the
remaining 80% for validation: this gives 60/20/20, not 55/25/20. Ratios should
follow training needs, independent entity counts and required evaluation
precision. For rare classes, count positive cases as well as total rows.

The executable classification example instead reserves 20% and uses five-fold
CV on the other 80%. Each fit sees 1,280 training rows and 320 validation rows.
After selecting C, the entire pipeline is refitted on 1,600 development rows
and evaluated on the 400 test rows.

## K-fold mechanics and aggregation

For ordinary K-fold, each row is validated once and trained on K-1 times.
The estimator must be fitted afresh in each fold. The educational splitter
rotates index blocks, distributing the remainder so validation sizes differ
by at most one. Its shuffled indices need not match scikit-learn because the
random generators differ; unshuffled membership is tested against KFold.

For fold scores M_k, descriptive summaries are

$$
\bar M = \frac{1}{K}\sum_{k=1}^{K} M_k, \qquad
s = \sqrt{\frac{\sum_k(M_k-\bar M)^2}{K-1}}.
$$

Training sets overlap, so fold scores are dependent. Neither s nor
s/sqrt(K) automatically supplies a confidence interval for production
performance. Repeated CV explores sensitivity to partitioning without creating
new independent observations.

A size-weighted average of fold mean losses equals pooled out-of-fold mean
loss for additive per-row losses. An unweighted average instead weights folds
equally. AUC, F1 and other nonadditive metrics generally do not obey this
identity. Pooled AUC also compares scores from different fitted models, whose
scales may differ. State whether the target is a typical row, customer or
time window; unequal groups can make these very different estimands.

Larger K increases the training fraction and fitting cost. It does not
monotonically reduce uncertainty. Leave-one-out uses almost all rows per fit,
but is expensive and can be unstable. Ordinary K-fold coverage does not extend
to expanding temporal CV: early rows may never be validated.

## Choosing a split for the actual question

**IID-like classification.** Stratification approximately preserves class
proportions, helping keep metrics defined. It does not remove entity dependence,
create more positive examples or reproduce temporal drift. Check class support
in every training and validation fold, especially for AUC.

**New entities.** Keep the entire customer, patient, device, document family or
case in one partition. Apply the same rule to the final holdout and inner
tuning folds. GroupShuffleSplit holds out a fraction of groups rather than
necessarily the same fraction of rows. GroupKFold prevents group overlap;
StratifiedGroupKFold attempts class balance subject to group constraints.

**Existing entities at a later time.** A customer appearing in training and
validation is not inherently invalid. If deployment predicts that customer's
future transactions using their legitimate history, chronological validation
can be the relevant target. Assess new-customer performance separately.
If the target is both new entities and future periods, use explicit time
cutoffs and group exclusions; neither ordinary group CV nor time CV alone
enforces both constraints.

**Temporal prediction.** Sort by prediction timestamp and reproduce when
training would actually occur. Expanding windows retain history; rolling
windows limit history to adapt to drift, at the cost of fewer training rows.
Validation windows should match the forecast horizon and retraining cadence.

TimeSeriesSplit uses row order and a gap measured in rows. Equally spaced
samples are needed for its equal-sized windows to represent comparable
durations; irregular events need timestamp-based cutoffs.
[TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

A temporal gap is not universal leakage protection. Suppose a label for time
t becomes available at t+2 and training occurs strictly before validation
starts. Then require t_train + 2 < min(t_validation). The example's gap of two
rows implements precisely this assumption. Real label windows can overlap
variable intervals: exclude training labels unavailable at the cutoff and
purge overlapping outcome intervals as needed. Feature values also need
point-in-time availability checks, including ingestion delay and revisions.

## Leakage audit

| Failure | Why the estimate changes | Prevention |
|---|---|---|
| Global scaling, imputation or PCA | Held-out covariates influence fitted statistics | Fit learned transformations within each training fold |
| Feature selection using all labels | Validation answers influence chosen columns | Put the selector inside the pipeline |
| Resampling before splitting | Synthetic or duplicated relatives cross partitions | Resample only the fold's training data |
| Post-outcome feature | Predictor contains information unavailable at decision time | Audit feature provenance and availability timestamp |
| Full-period customer aggregate | Future events enter a historical prediction | Build as-of aggregates; exclude current outcome |
| Near-duplicates across partitions | Validation measures recognition of related material | Group related examples before splitting |
| Repeated test-driven changes | Final holdout becomes a development objective | Freeze decisions; obtain fresh evaluation data when needed |

The deliberate invalid control in the code selects columns using every label
before CV. The safe version puts both selection and scaling inside the fitted
pipeline. A small or absent observed score difference would not legitimize a
leaking procedure. Pipelines isolate their own learned steps; they do not audit
upstream features. See
[scikit-learn's leakage examples](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## Selection bias and nested CV

Selecting the best configuration also selects favorable validation noise:

$$
\lambda^* = \arg\min_{\lambda}\widehat R_{CV}(f_{\lambda}).
$$

Consequently the winning CV score is a development statistic, not an untouched
generalization estimate. Nested CV runs the full search inside each outer
training fold and evaluates the selected pipeline on that outer holdout.
Refit on all outer training rows before scoring. Group/time restrictions apply
at both levels, including all preprocessing and threshold choices.

Outer results estimate the selection procedure trained at the outer training
size, rather than the exact final model trained on every row. Repeatedly
changing the search based on outer results also makes those results part of
development. A final independent holdout is useful when available. Nested
evaluation costs roughly outer folds times inner folds times configurations,
plus refits; it is not implemented in this compact example.

## RAG and agent evaluation

For unseen-document generalization, group queries and derived chunks by source
document, case or near-duplicate family before tuning. But a held-out query's
source document can legitimately be in the retrieval index if that is the
deployment contract. Withholding all answer-bearing documents would instead
test a different, potentially unanswerable task.

Keep reference answers and evaluation traces out of tuning and tool-visible
memory. Separate prompt-development queries from final evaluation queries,
reset persistent agent state between independent cases, and reproduce index
access permissions and time cutoffs. These are evaluation-design applications,
not experiments performed here.

## Executed experiment record

Executed locally with Python 3.11.0rc2, NumPy 2.4.6, scikit-learn 1.9.0, seed 42.
Commands are in the topic README. Values below are rounded from terminal output.
All interpretation candidates require author review; none are attributed to
the author's personal experience or conclusions.

### 1. Development-only model selection

- **Hypothesis:** a complete pipeline can select regularization using only
  development folds and produce a separate test estimate.
- **Configuration:** 2,000 synthetic classification rows, 20 features (8
  informative, 4 redundant), requested class weights 0.85/0.15, generator
  defaults otherwise; stratified 80/20 split; shuffled stratified five-fold CV;
  StandardScaler and LogisticRegression, max_iter=1000; C in
  {0.01, 0.1, 1, 10}; selection metric ROC-AUC; sequential fitting.
- **Result:** both development and test positive proportions 0.1525; selected
  C=10; fold AUCs 0.821615, 0.809172, 0.759997, 0.855712, 0.811582.
  Mean 0.811616, sample SD 0.034333; final test AUC 0.753663.
- **Interpretation candidate:** the separate holdout yields a lower estimate
  on this draw; the selected CV score should not be presented as final evidence.
- **Limitation:** one synthetic dataset and split cannot isolate how much of the
  difference comes from sampling variation versus selection optimism. No
  deployment threshold, calibration or business utility was evaluated.

### 2. Supervised feature-selection leakage

- **Hypothesis:** selecting features with held-out random labels can make a
  noise-only problem appear predictive.
- **Configuration:** 300 rows, 2,000 independent standard-normal features and
  independent Bernoulli(0.5) labels; select 20 columns with ANOVA F scores;
  shuffled stratified five-fold CV; scaled logistic regression with default
  C=1 and max_iter=1000. Same folds for both procedures.
- **Result:** invalid global-selection AUCs 0.784598, 0.839286, 0.781250,
  0.710790, 0.844271 (mean 0.792039). Fold-local AUCs 0.397321, 0.407366,
  0.618304, 0.310345, 0.462736 (mean 0.439214).
- **Interpretation candidate:** feature search exploits chance label
  associations when validation answers influence selection.
- **Limitation:** this high-dimensional control exaggerates opportunities for
  supervised selection bias. Fold-local AUC below 0.5 on one draw does not
  establish systematically worse-than-chance performance. The CV comparisons
  are demonstrations, not a new untouched test evaluation.

### 3. Repeated customer signatures

- **Hypothesis:** random row folds reward recognizing known customers and
  overstate performance for a new-customer deployment target.
- **Configuration:** 100 customers, 10 rows each, eight standard-normal
  signature features per customer plus independent Gaussian noise SD=0.01;
  independent Bernoulli(0.5) customer labels, shared by their rows; 1-nearest
  neighbor; shuffled five-fold KFold versus five-fold GroupKFold; accuracy.
- **Result:** random-fold accuracies all 1.0; overlapping customer counts
  91, 88, 88, 85, 87. Group-fold accuracies 0.455, 0.550, 0.450, 0.625,
  0.765 (mean 0.569); overlapping customer counts all zero.
- **Interpretation candidate:** customer recognition explains the perfect
  random-fold result under this intentionally constructed process.
- **Limitation:** real customer labels are not independent coin flips and
  signatures need not be this stable. Group-fold mean 0.569 is a single finite
  draw, not proof of transferable predictive signal.

### Structural demonstrations

The temporal schedule produced training/validation ranges 0..12 / 15..19,
0..17 / 20..24, and 0..22 / 25..29. A final fit at time 30 would use rows
0..27, leave 28..29 as the gap, and score 30..34. No temporal model was fitted
and no forecast metric was measured.

The educational 11-row, three-fold run produced validation sizes 4, 4 and 3.
These are partition checks, not predictive experiments.

## Follow-up status after the visual phase

- Repeat the noise and customer experiments across predeclared seeds; summarize
  paired differences without selecting the most dramatic seed.
- Compare five versus ten folds with fixed data and search space, recording
  runtime as well as score dispersion.
- Executed in the [visual lab](VISUAL_GUIDE.md): gradual concept drift with
  rolling and expanding validation against a later untouched period, and
  global versus fold-local scaling. The visual guide records results and
  limitations. The repeated-seed and five-versus-ten-fold proposals above
  remain unexecuted.
