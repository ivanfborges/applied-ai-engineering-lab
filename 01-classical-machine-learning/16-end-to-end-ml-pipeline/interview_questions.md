# Senior Interview Questions

## 1. How would you start an ML project?

Start with the decision, not an algorithm. Define the prediction unit, target,
prediction moment, outcome horizon, available information, consumer action,
latency requirement, error costs, and success criteria. Then investigate the
data and label-generation process, choose a validation boundary that resembles
production, establish a baseline, and only then compare models. Before release,
define feature consistency, artifact versioning, observability, fallback, and
retraining or rollback criteria.

## 2. How do you decide whether a feature is leakage?

Ask whether that exact value, computed by that exact process, would have been
available when the production prediction is made. A feature can be valid for
one horizon and leaky for another. Also inspect subtler paths: global
preprocessing, target encoding, feature selection, duplicated entities, and
historical tables updated after the prediction time.

## 3. Why can a random split be wrong?

It assumes observations are sufficiently exchangeable. It can overstate
generalization when production predicts future periods or unseen customers,
patients, devices, locations, or documents. Use forward validation for future
predictions, group-aware validation for new entities, or a design that
respects both. The split should reproduce the boundary the deployed model must
cross.

## 4. Why keep preprocessing inside a scikit-learn pipeline?

Imputation values, scale parameters, categories, selected features, and PCA
components are learned parameters. A pipeline refits them only on each
cross-validation training fold and packages the fitted transformations with
the estimator for inference. It reduces two risks: validation leakage and
training-serving inconsistency. It does not automatically fix a wrong split or
future-derived raw feature.

## 5. What does a baseline provide?

It answers whether complexity buys incremental value and acts as a debugging
reference. Suitable baselines include the class prior, a historical average, a
current rule, or a simple interpretable estimator. The most frequent class may
be enough to expose misleading accuracy, while a current business policy is
often the operationally relevant comparison.

## 6. Why separate model probability from the decision threshold?

The model estimates a score or probability; the threshold converts it into an
action. False-positive cost, false-negative cost, review capacity, safety
constraints, and probability calibration determine the useful operating
point. A default threshold of 0.5 has no universal business meaning.
Threshold selection is itself a model-development choice and must not use the
final test set.

## 7. How do validation and test sets differ?

Validation evidence guides hyperparameters, features, thresholds, and other
development choices. The test set estimates the performance of the locked
process. If a team repeatedly changes the process after reading test results,
the test set has become another validation set and its estimate is optimistic.
A fresh holdout may then be necessary.

## 8. Why not always choose the highest cross-validation score?

A small score difference may be sampling noise or immaterial relative to
latency, memory, calibration, interpretability, training cost, operational
complexity, and maintenance risk. Compare fold variability and error slices,
and decide what improvement would matter to the downstream decision before
optimizing. Model selection is multi-objective system design.

## 9. What belongs in a deployable model release?

At minimum: fitted preprocessing, estimator, decision threshold, feature
schema and semantics, dependency versions, training-data lineage, code and
artifact versions, evaluation record, and approval state. The serving path
also needs schema validation, logging, security controls, fallback behavior,
and a way to roll back. Serializing `predict()` behavior alone is not the
whole release.

## 10. How would you prevent training-serving skew?

Reuse the fitted preprocessing artifact or the same versioned transformation
implementation; define feature contracts with names, types, units, time
semantics, missingness, and category behavior; compare offline and online
feature values on sampled entities; and test raw request rows through the
complete saved pipeline before promotion. Monitor contract violations and
unknown-category rates after release.

## 11. What would you monitor after deployment?

Monitor service health, request contracts, input distributions, prediction
distributions, action rates, delayed-label model quality, calibration, subgroup
performance, and business outcomes. Input drift is not proof of model
degradation, so connect alerts to diagnosis and response playbooks. Label delay
must be explicit because it determines when true performance becomes
observable.

## 12. When should a model be retrained?

Retraining candidates include sustained performance loss, meaningful drift,
enough new representative data, changes in product behavior, target
definition, regulations, or feature availability. Retraining should create a
challenger that passes the same validation, guardrail, and approval process;
it should not automatically replace production simply because time passed.

## 13. Offline metrics improved but business outcomes did not. What do you check?

Investigate metric-objective mismatch, threshold and capacity, leakage,
population or concept drift, training-serving skew, latency, failed downstream
adoption, historical-policy bias, missing production features, feedback loops,
and experiment design. Also verify that the intervention can causally change
the outcome: better prediction does not guarantee a better action.

## 14. How would you explain the complete pipeline in under a minute?

An end-to-end ML pipeline starts by defining the decision, target, prediction
time, horizon, and information available at inference. I then validate how
examples and labels are generated, choose a split that reproduces the
production boundary, establish a baseline, and fit preprocessing and the model
inside a reproducible validation workflow. Hyperparameters and decision
thresholds use development evidence; the test set stays untouched until the
process is locked. Deployment versions the full inference contract, and
monitoring connects service, data, prediction, delayed-label, and business
signals to rollback and retraining decisions.
