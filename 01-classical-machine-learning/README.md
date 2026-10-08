# Classical Machine Learning — Days 16–35

Twenty implemented studies connect model assumptions, numerical implementations,
validation, feature availability and operational decisions. This is a study
curriculum, not twenty production deployments. Return to the [lab overview](../README.md)
for Foundations, the [roadmap](../ROADMAP.md) and later planned modules.

## Choose a technical question

- **Lifecycle and trust:** start with 16–17 for preprocessing, validation,
  inference persistence, leakage and deployment-aligned boundaries.
- **Inspect the mathematics:** 18–21 connect OLS, gradient descent,
  regularization and logistic loss to controlled numerical checks.
- **Separate score from action:** 22–23 and 33 distinguish ranking,
  probability quality, prevalence and validation-selected thresholds.
- **Compare inductive biases:** 24–31 connect neighbors, generative classifiers,
  trees, boosting and margins to their assumptions and implementation limits.
- **Challenge the data and explanation:** 32–35 cover point-in-time features,
  imbalance, missingness, reliance, response curves and attribution.

## Topic index

| Day | Study | Main implementation evidence |
|---:|---|---|
| 16 | [End-to-end pipeline](16-end-to-end-ml-pipeline/) | Fold-local preprocessing, validation policy, persisted inference and lifecycle lab |
| 17 | [Validation and leakage](17-validation-and-leakage/) | Split logic, random-label controls, group separation and temporal availability |
| 18 | [Linear regression theory](18-linear-regression-theory/) | OLS geometry, residual diagnostics and interactive numerical/app checks |
| 19 | [Linear regression from scratch](19-linear-regression-from-scratch/) | OLS/GD parity and controlled feature-scaling comparison |
| 20 | [Regularization](20-regularization/) | Fold-local tuning, proximal optimization and separate coefficient-path visual lab |
| 21 | [Logistic regression](21-logistic-regression/) | Stable binary loss, optimization, odds/threshold behavior and visual lab |
| 22 | [Classification metrics I](22-classification-metrics/) | Confusion-derived metrics, prevalence and threshold/capacity views |
| 23 | [Classification metrics II](23-classification-metrics-ii/) | Pairwise AUC, AP, probability distortion and validation threshold policy |
| 24 | [K-nearest neighbors](24-k-nearest-neighbors/) | Voting rules, scaling and distance geometry |
| 25 | [Naive Bayes](25-naive-bayes/) | Multinomial count model, smoothing and synthetic text routing |
| 26 | [Decision trees](26-decision-trees/) | Impurity/split logic, growth controls and validation-selected pruning |
| 27 | [Random forest](27-random-forest/) | Bootstrap stumps, OOB behavior, feature importance and tree correlation |
| 28 | [ExtraTrees](28-extratrees/) | Random recursive splits, disagreement and threshold/bootstrap comparison |
| 29 | [Gradient boosting](29-gradient-boosting/) | Residual-fitting stumps and validation-only stage selection |
| 30 | [XGBoost, LightGBM and CatBoost](30-xgboost-lightgbm-catboost/) | Numeric/mixed workflows, early stopping, Newton gain and small three-library integration tests |
| 31 | [Support vector machines](31-support-vector-machine/) | Mean-hinge derivatives, matched linear objectives and fold-local C/gamma search |
| 32 | [Feature engineering](32-feature-engineering/) | Train-fitted transforms and event/availability-aware transaction aggregates |
| 33 | [Imbalanced data](33-imbalanced-data/) | Training-only resampling, educational SMOTE and validation-selected cost policies |
| 34 | [Missing data](34-missing-data/) | Training-fitted imputation, indicators and frozen-model feature-loss stress checks |
| 35 | [Explainability](35-explainability-classical-ml/) | Grouped permutation, PDP/ICE agreement, probability SHAP reconstruction and eight-tab lab |

## Evidence and review status

Each topic includes public tests and scoped experiment records. **Implementation,
measured observations, delegated technical review and personal author review
are separate states.** On October 8, 2026, the author delegated an AI-assisted
technical review of nine interpretation items from the study-script records in
32, 33 and 35. Their bounded wording and scope are documented in the
[curation record](../docs/CURATION.md#delegated-technical-review--october-8-2026).
Visual interpretations in those topics and candidates in 16–31 and 34 remain
pending. This index does not certify
every statement; test success is not scientific approval.
The earlier [curation record](../docs/CURATION.md) covers a dated selection,
not approval of all later studies.

Most datasets are favorable synthetic controls: inspect the target generator,
split boundaries, selection policy and limitations before interpreting a score.
Single seeds, different representations or unequal tuning budgets do not
establish universal model rankings. Scratch implementations are educational.

## Run and validate

Follow each topic's commands. For the complete validation dependencies:

```bash
python -m pip install -e ".[dev,imbalance,explainability,boosting]"
python scripts/validate_repo.py all
```

Use the [validation contract](../docs/validation.md) to choose smaller checks
and understand test isolation, optional dependencies and coverage limits.
Visual generators are separate from study scripts and may use different
configurations. Generated outputs stay ignored unless deliberately curated
as linked public previews; a guide is not a claim that all assets are versioned.
