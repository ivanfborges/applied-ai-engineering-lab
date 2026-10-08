# Day 30 — XGBoost, LightGBM and CatBoost

Which boosting framework fits a tabular prediction problem, and what changes
when categorical features become important? These libraries share sequential
additive tree learning, but differ in split search, growth policy, categorical
representation, and training controls. This study connects those decisions to
a small executable comparison and a second-order split calculation.

For binary classification the ensemble produces a **logit**:

$$F_t(x)=F_{t-1}(x)+\eta f_t(x),\qquad p_t(x)=\sigma(F_t(x)).$$

A tree adds a score correction; its leaves are not standalone probabilities.
Boosted trees are useful for risk, churn, demand, and structured routing signals
around AI systems. Such applications still need representative labels, features
available at decision time, and validation that matches deployment.

## Differences to investigate

| Framework | Mechanisms | Practical controls |
|---|---|---|
| XGBoost | Regularized gradient/Hessian split objective; histogram training; depthwise or loss-guided growth | `max_depth`, `min_child_weight`, `gamma`, L1/L2 penalties, sampling |
| LightGBM | Histogram splits and best-first leaf growth; categorical partitions; optional GOSS and feature bundling | `num_leaves`, `max_depth`, minimum leaf support, binning and sampling |
| CatBoost | Ordered categorical statistics; optional ordered boosting; symmetric trees by default | `depth`, leaf regularization, categorical configuration, boosting and growth policies |

Histograms are shared techniques. XGBoost also supports native categoricals;
this example deliberately uses one-hot encoding to expose preprocessing choices.
CatBoost's ordered target statistics and ordered boosting solve related but
distinct problems; its CPU boosting default is `Plain`, so the example requests
`Ordered` explicitly. LightGBM's example uses ordinary GBDT, not GOSS.
See the [official sources](references.md) and [technical notes](notes.md).

## Files and evidence

| File | What it adds |
|---|---|
| [example.py](example.py) | Numeric and mixed synthetic workflows, validation early stopping, model selection, and selected-model test evaluation |
| [from_scratch.py](from_scratch.py) | Stable logistic derivatives, L2 leaf weights, regularized split gain, and one-feature Newton stump |
| [notes.md](notes.md) | Derivation, categorical intuition, capacity trade-offs, mistakes, and executed experiment records |
| [tests/test_boosting.py](tests/test_boosting.py) | Finite differences, direct objective oracle, split constraints, data boundaries, and controlled XGBoost agreement |
| [interview_questions.md](interview_questions.md) | Questions tied to the math and workflow choices |
| [references.md](references.md) | Original papers and official API documentation |

## Run from the repository root

Install the topic-specific optional extra from the canonical
[pyproject.toml](../../pyproject.toml):

```bash
python -m pip install -e ".[dev,boosting]"
python 01-classical-machine-learning/30-xgboost-lightgbm-catboost/example.py
python 01-classical-machine-learning/30-xgboost-lightgbm-catboost/example.py --dataset mixed
python 01-classical-machine-learning/30-xgboost-lightgbm-catboost/from_scratch.py
python -m pytest -q 01-classical-machine-learning/30-xgboost-lightgbm-catboost/tests
```

LightGBM 4.7 or newer is required for its `eval_X`/`eval_y` validation API.

Both datasets are **synthetic and generated in code**; no dataset is downloaded.
Each library sees the same 1,440 training and 480 validation rows. Validation
log loss selects boosting rounds and then the candidate model; only that
selected candidate is scored on the 480 test rows. No refit is performed.
ROC-AUC and Brier score are diagnostics, not alternative selection criteria.

The numeric case has eight features. The mixed case has four numeric features,
80 possible merchant categories, and four channels. It uses train-fitted dense
one-hot features for XGBoost, pandas categories fitted to the training vocabulary
for LightGBM, and raw strings for CatBoost. Unknown one-hot categories produce
all-zero category blocks; unknown LightGBM categories become missing values.
These are deliberately different workflows, not an isolated algorithm benchmark.

The scripts write `outputs/numeric_experiment.json` and
`outputs/mixed_experiment.json` relative to this folder, overwriting them on
reruns. Those generated records include configurations, split indices, package
versions, measured results, limitations, and pending review status. They remain
ignored; the intentional public summaries are in the
[executed records](notes.md#executed-experiment-records). No visual phase or
public output previews are included.

## Takeaways and limits

- Loss curvature and regularization determine leaf corrections, while split
  gain determines whether an extra leaf is worth adding.
- A shared depth limit does not make asymmetric, depthwise, and symmetric
  trees equally expressive. Equal learning rates do not equalize training.
- Native categorical handling reduces preprocessing work but cannot repair
  temporal leakage, invalid features, or an inappropriate evaluation split.
- Early stopping and library selection reuse validation information. Keep
  final test labels outside those decisions.
- Probability quality and ranking quality answer different questions. Log
  loss and Brier score alone do not isolate calibration.
- The executed results are single-seed synthetic demonstrations. Their
  interpretations remain **pending author review** and establish no universal
  accuracy, speed, or production-readiness ranking.

The scratch code implements one numerical feature and one Newton step. It
omits full boosting, recursion, categoricals, missing values, histogram search,
L1 regularization, and acceleration. It is an educational implementation.

Prerequisites: [gradient boosting intuition](../29-gradient-boosting/),
[decision trees](../26-decision-trees/), and
[classification evaluation](../23-classification-metrics-ii/).
