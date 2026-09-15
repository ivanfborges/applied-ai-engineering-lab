# Technical Notes

## Start with the decision contract

An ML problem is well framed when another team could determine exactly when a
prediction is made, what it predicts, and what action consumes it.

| Question | Retention example |
|---|---|
| Decision | Which customers should receive limited retention outreach? |
| Prediction unit | One customer snapshot |
| Prediction time | End of the current observation window |
| Target | No paid renewal within the following 30 days |
| Horizon | 30 days |
| Available inputs | Tenure, usage, support activity, inactivity, and plan known at the snapshot |
| Output | Estimated churn probability |
| Action | Prioritize eligible customers under a capacity constraint |

A compact temporal formulation is

\[
X_t \longrightarrow P(Y_{t+h}=1 \mid X_t),
\]

where \(X_t\) contains only information available at prediction time \(t\),
and \(h\) is the outcome horizon. A column can exist in an analytical table
and still be invalid if it was populated after \(t\).

The statistical objective is not training loss by itself. For parameters
\(\theta\), empirical training risk is

\[
\widehat R_{train}(\theta)
= \frac{1}{n}\sum_{i=1}^{n}
L\left(y_i, f_\theta(x_i)\right),
\]

but the operational concern is expected loss under the future production
distribution:

\[
R_{production}(\theta)
= \mathbb E_{(X,Y)\sim P_{production}}
\left[L\left(Y,f_\theta(X)\right)\right].
\]

Validation is useful only to the extent that it approximates this second
quantity and the decisions built on top of it.

## Data collection is part of model design

Before training, inspect how examples enter the dataset:

- **Population:** who is eligible for prediction and who is absent?
- **Observation process:** are values measured, self-reported, inferred, or
  created by a previous policy?
- **Label process:** when does the label mature, and can it be wrong or
  censored?
- **Entity structure:** can one customer, patient, device, or document
  contribute several correlated rows?
- **Time:** were all features reconstructed as they were known at the
  historical prediction moment?
- **Intervention history:** did earlier decisions change which labels became
  observable?

The executable example uses generated data, so it cannot validate any of these
real collection mechanisms. Its semantic column names should not be mistaken
for a real churn dataset.

## Splits encode the generalization claim

The example uses a random stratified split because its generated rows are
independent and have no temporal ordering. That assumption would often be
wrong for real customer histories.

Choose the boundary that matches deployment:

| Production question | Candidate validation design |
|---|---|
| New independent rows from a stable population | Random stratified split |
| Future behavior | Forward or temporal split |
| Unseen customers, patients, devices, or documents | Grouped split |
| Both unseen entities and future time | A design respecting both constraints |
| Limited independent data | Cross-validation that still respects time or groups |

Row disjointness is necessary but not sufficient. If the same entity or future
information crosses partitions, the effective evidence is not independent.

The Day 16 example creates:

- **training data:** fits preprocessing and estimator parameters;
- **training folds:** compare regularization values;
- **validation data:** selects the recall-constrained decision threshold;
- **test data:** evaluates the locked model and threshold once.

Repeatedly changing the pipeline after inspecting test results turns the test
set into another validation set.

## Baselines answer an incremental-value question

A metric has meaning only relative to a reference. Useful baselines include:

- the class prior or majority class;
- a historical average;
- the current business rule;
- a transparent heuristic;
- a simple linear model.

The example uses `DummyClassifier(strategy='prior')`. Its constant score has
ROC-AUC 0.5 when both classes are present. This is a ranking reference, not a
complete business baseline: a current retention policy could be much stronger
and should be compared in a real project.

## Learned preprocessing belongs inside the pipeline

Imputation values, scaling statistics, vocabularies, selected features, and
dimensionality-reduction components are fitted parameters. If they are learned
from all rows before a split, validation information has entered training.

The executable pipeline nests:

```text
numerical -> median imputation -> standardization
categorical -> most-frequent imputation -> one-hot encoding
combined features -> logistic regression
```

`GridSearchCV` clones and fits the complete pipeline in each training fold.
The validation fold therefore cannot influence that fold's imputation,
scaling, or categories. `handle_unknown='ignore'` defines behavior for a
category unseen during training, but monitoring should still report unexpected
category rates.

## Model selection and decision policy are different

For binary classification, logistic regression estimates a score interpreted
as

\[
p(x) = P(Y=1\mid X=x).
\]

The downstream action applies a threshold:

\[
\widehat y_t =
\begin{cases}
1, & p(x) \ge t,\\
0, & p(x) < t.
\end{cases}
\]

Changing \(t\) does not retrain the probability model. It changes the error
trade-off and the workload created by the system. If false-positive and
false-negative costs are known, a simplified decision objective is

\[
\operatorname{Cost}(t)
= C_{FP}FP(t) + C_{FN}FN(t).
\]

Capacity can be a constraint as well: the number of predicted positives may
not exceed the cases a team can review.

The example chooses the highest validation threshold reaching a specified
recall. This is an educational policy, not a claim that 75% recall is correct
for retention. It also assumes validation performance is representative and
probabilities are sufficiently stable. A real project should evaluate
calibration, cost uncertainty, capacity, subgroup behavior, and threshold
robustness.

## From evaluated model to inference artifact

Deployment requires more than serializing coefficients. A usable artifact or
release should identify:

- preprocessing and estimator versions;
- feature names, types, units, and allowed missingness;
- category and unknown-value behavior;
- decision threshold and the policy owner;
- training-data and code version;
- evaluation evidence and approval state;
- runtime dependency versions.

The example stores the fitted scikit-learn pipeline, threshold, ordered feature
columns, and small metadata record in one ignored pickle bundle. It then
reloads the bundle and repeats inference. This demonstrates component
consistency, not a production registry, serving API, security boundary, or
backward-compatibility guarantee.

Pickle can execute code during deserialization. Only trusted local artifacts
should be loaded, and dependency compatibility still needs explicit control.

Batch inference is usually preferable when fresh millisecond decisions are
unnecessary: it is simpler to operate, audit, and retry. Online inference adds
latency budgets, availability targets, concurrency, feature freshness, and
fallback behavior.

## Monitoring closes the lifecycle

Monitoring should cover several layers:

| Layer | Example signals | Question |
|---|---|---|
| Service | latency, throughput, error rate, resource use | Is inference available and timely? |
| Contract | missing fields, type errors, unknown categories, range violations | Are requests valid? |
| Inputs | distributions, correlations, population mix | Has \(P(X)\) changed? |
| Predictions | score distribution, positive rate, calibration proxies | Is model behavior changing? |
| Delayed labels | ROC-AUC, precision, recall, calibration, slice performance | Is predictive quality changing? |
| Decision outcomes | outreach capacity, conversion, cost, guardrails | Is the system improving the intended decision? |

Covariate shift means

\[
P_{old}(X) \ne P_{new}(X),
\]

while concept drift means

\[
P_{old}(Y\mid X) \ne P_{new}(Y\mid X).
\]

Input drift does not prove performance degradation, and stable marginals do
not prove the conditional relationship is stable. Retraining should be
triggered and promoted through evaluated criteria, not merely because a
calendar interval elapsed.

## Common failure modes

- Starting from an available table instead of a decision and prediction time.
- Using outcome-derived or future values as features.
- Fitting preprocessing or feature selection before partitioning.
- Randomly splitting observations that are grouped or temporal.
- Reporting accuracy without class prevalence, error costs, or a baseline.
- Selecting a threshold on the final test set.
- Treating the highest cross-validation score as decisive despite uncertainty,
  latency, maintenance, calibration, or explainability costs.
- Reimplementing training transformations differently in serving.
- Monitoring only infrastructure while labels and business outcomes degrade.
- Automatically retraining without validating the replacement.

## Visual laboratory experiment design

The public visual lab now executes controlled demonstrations of target
leakage, temporal and grouped splitting, class-imbalanced baselines,
threshold-dependent cost, model complexity, repeated validation search,
cross-validation, covariate and concept drift, training-serving skew, and
multi-layer monitoring. The hypotheses, exact configurations, observed
outputs, interpretation candidates, and limitations are recorded in the
[topic README](README.md#executed-visual-evidence).

The simulations isolate one mechanism at a time. This makes causal structure
inside the generator visible, but it does not estimate how frequently or how
strongly these problems occur in a real deployment.

## Proposed extensions, not executed

1. **Fresh final holdout:** repeat development across several seeds, lock the
   process, and evaluate once on a separately generated sample.
2. **Probability calibration:** compare threshold cost before and after
   calibration using development data only.
3. **Combined entity and time boundary:** validate future observations from
   entities absent during training.
4. **Subgroup policy audit:** compare calibration, recall, workload, and cost
   across synthetic groups without implying a real fairness conclusion.
5. **Delayed-label response policy:** map monitoring signals to investigate,
   rollback, retrain, or wait actions under different label delays.

Results from these extensions should receive the same execution record and
author review before publication.
