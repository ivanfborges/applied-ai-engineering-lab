# End-to-End Machine Learning Pipeline

## Overview

An end-to-end ML pipeline connects a decision to data, a reproducible
evaluation, a deployable inference contract, and production feedback. Model
training is only one stage:

```text
decision -> prediction contract -> data -> split -> baseline
         -> training and validation -> test -> deployment -> monitoring
```

This study frames a synthetic customer-retention problem as:

> At a customer snapshot, estimate the probability of churn within 30 days
> using only information available at that snapshot, so a retention workflow
> can prioritize outreach.

The prediction unit, prediction time, horizon, action, and error costs must be
defined before an algorithm is chosen. Those choices determine valid labels,
features, partitions, metrics, and serving requirements.

## Concepts and relevance

- **Problem framing:** translate a business decision into a target, unit,
  horizon, prediction moment, and action.
- **Data contracts:** validate feature names, types, missingness, categories,
  and label definitions at training and inference boundaries.
- **Evaluation design:** make train, validation, and test partitions represent
  the population, entity, and time boundary expected after deployment.
- **Baselines:** compare model complexity with a trivial or current-policy
  reference before claiming value.
- **Leakage-safe learning:** fit imputation, scaling, encoding, and model
  parameters only on the appropriate training fold.
- **Model and policy separation:** estimate probability with a model, then
  select a decision threshold from operational costs or constraints.
- **Deployment:** version the preprocessing pipeline, estimator, threshold,
  schema, and metadata together.
- **Monitoring:** observe service health, input quality, prediction behavior,
  delayed label performance, and business outcomes.

The same system-level reasoning applies to credit risk, fraud, document
intelligence, forecasting, retrieval, and other prediction-backed workflows.
The split and metrics must match the use case; this example is not a universal
pipeline template.

## Executable study

The example creates **deterministic synthetic data** in memory. It uses no
external or public dataset. The generator simulates customer tenure, product
usage, support activity, inactivity, plan, missing values, and a binary
30-day churn outcome. These names are educational abstractions rather than
measurements from a real business.

The implementation:

1. creates stratified 60/20/20 train, validation, and test partitions;
2. fits a prior-based `DummyClassifier` reference;
3. runs five-fold model selection on training data;
4. keeps imputation, scaling, one-hot encoding, and logistic regression inside
   one scikit-learn `Pipeline`;
5. chooses the highest validation threshold that reaches 75% recall;
6. evaluates once on the untouched test partition;
7. saves the fitted pipeline, threshold, feature contract, and metadata;
8. reloads the artifact and checks prediction equivalence on a raw row with a
   missing numeric value and an unseen category.

## Files

- [`notes.md`](notes.md): framing, validation assumptions, leakage,
  thresholding, deployment, monitoring, limitations, and proposed extensions.
- [`example.py`](example.py): complete deterministic training and inference
  workflow.
- [`simulation.py`](simulation.py): deterministic temporal, grouped, leakage,
  imbalance, threshold, complexity, validation-search, drift, serving, and
  monitoring experiments.
- [`visualizations.py`](visualizations.py): reusable Matplotlib/Plotly
  renderers, six bounded preview generators, and the concept-drift animation.
- [`streamlit_app.py`](streamlit_app.py): interactive laboratory with eight
  conditionally rendered sections and focused controls.
- [`tests/`](tests/): data-contract, partition, threshold, simulation,
  rendering, unknown-category, serialization, and Streamlit smoke checks.
- [`interview_questions.md`](interview_questions.md): senior-level questions
  connecting model development to system ownership.
- [`references.md`](references.md): primary and official material for deeper
  study.

A first-principles classifier is intentionally omitted. Reimplementing
logistic regression would answer an algorithm question, while this topic is
about connecting lifecycle boundaries; later roadmap topics cover estimator
internals directly.

## Run

From the repository root, install the shared dependencies if needed:

```bash
python -m pip install -e .[dev]
```

Run the complete example:

```bash
python 01-classical-machine-learning/16-end-to-end-ml-pipeline/example.py
```

Start the interactive visual laboratory:

```bash
streamlit run 01-classical-machine-learning/16-end-to-end-ml-pipeline/streamlit_app.py
```

Regenerate the selected visual candidates:

```bash
python 01-classical-machine-learning/16-end-to-end-ml-pipeline/visualizations.py
```

Run the focused tests:

```bash
python -m pytest -q 01-classical-machine-learning/16-end-to-end-ml-pipeline/tests
```

The example writes `outputs/inference_bundle.pkl` relative to the topic
folder. That model artifact remains ignored. Six preview filenames in the same directory are allowed by Git rules,
but their files are not currently published; other generated outputs stay local. Load pickle artifacts only from trusted sources because
deserialization is not a safe interchange format for untrusted files.

## Visual laboratory

The app groups the requested questions into a navigable learning sequence:

1. lifecycle and prediction-time boundary;
2. random, temporal, and grouped validation boundaries;
3. target leakage, class imbalance, baselines, and metric choice;
4. probability scores, threshold policy, confusion counts, precision–recall,
   configurable business cost, and a rotatable 3D probability surface;
5. underfitting, reasonable fit, high complexity, and validation-set
   selection optimism;
6. cross-validation with fold-local preprocessing;
7. covariate drift versus concept drift;
8. training-serving skew, multi-layer monitoring, and evaluated retraining.

Every major view states what it shows, why it matters, what can fail, and the
senior interview takeaway. Controls are local to the active section so hidden
experiments do not run on every interaction.

### Local visual previews

The preview files are not included in this checkout. Generate them with the
`visualizations.py` command above and inspect the local `outputs/` directory.
The generator covers lifecycle, temporal splits, leakage, threshold trade-offs
and drift. No absent image is presented here as a published result.

## Executed experiment record

- **Hypothesis:** under this constructed outcome process, a fold-safe logistic
  pipeline should rank outcomes better than a constant prior baseline; a
  threshold selected for 75% validation recall should trade precision for
  more positive predictions; serialization should preserve inference exactly.
- **Configuration:** seed 16; 2,400 synthetic rows; observed positive rate
  0.2317; stratified 1,440/480/480 train/validation/test rows; median and
  most-frequent imputation; numerical standardization; unknown-safe one-hot
  encoding; logistic regression; `C` candidates 0.1, 1, and 10; five shuffled
  stratified training folds scored by ROC-AUC; 75% validation-recall
  constraint.
- **Observed result:** `C=0.1` was selected with mean training-CV ROC-AUC
  0.7321. The validation-selected threshold was 0.1953, with validation
  precision 0.3457 and recall 0.7568. On the untouched test partition, the
  prior baseline ROC-AUC was 0.5000; model ROC-AUC was 0.6993, precision
  0.3401, recall 0.7568, F1 0.4693, and positive-prediction rate 0.5146. The
  reloaded-bundle probability differed from the pre-serialization value by
  0.000e+00 for the checked inference row.
- **Interpretation candidate — author review required:** the observed run is
  consistent with the generator containing learnable ranking signal and with
  the recall constraint producing a relatively broad intervention set.
  Whether that operating point is useful cannot be decided without real
  intervention costs and capacity.
- **Limitations:** this is one deterministic synthetic draw and one final test
  partition, not a benchmark or estimate of deployment value. The generator
  omits delayed and noisy labels, selection effects, entity history, temporal
  drift, feedback loops, fairness constraints, real feature ownership, and
  service behavior. Exact pickle round-trip agreement checks serialization,
  not cross-version portability or production readiness.

## Executed visual evidence

- **Hypotheses:** random splitting should look more favorable than a future
  holdout under gradual concept drift; row splitting should leak repeated
  entity identity; a target-derived future proxy should create invalid score
  inflation; trying more equal-quality candidates should increase the selected
  validation score without improving independent performance; concept drift
  should reduce old-model quality even when feature marginals stay similar.
- **Configuration:** seed 16 throughout; 1,200 timestamped observations with a
  1.3-radian progressive boundary rotation; 100 entities with eight rows each;
  1,400 observations for target leakage; 1,600 observations at configured 8%
  class prevalence for baseline comparison; 180 repeated validation-search
  simulations with 180 validation observations and 5–500 equal-quality
  candidates; covariate shift 1.0 and concept rotation 1.0; and 12 synthetic
  monitoring weeks. The concept animation contains 24 frames.
- **Observed results:** random-split ROC-AUC was 0.788 versus 0.730 for the
  temporal split. Row splitting shared 92 entities and produced ROC-AUC 0.743;
  group-aware splitting shared none and produced 0.588. Valid-feature
  ROC-AUC was 0.795, while the explicitly invalid future target proxy produced
  1.000. The majority baseline achieved accuracy 0.917 and recall 0.000.
  With equal true candidate accuracy 0.70, the mean best validation score rose
  from 0.739 with five candidates to 0.799 with 500, while the corresponding
  independent score was 0.699 and 0.694. Covariate drift gave feature-1
  Wasserstein distance 0.932 with old-model ROC-AUC 0.856; concept drift kept
  that distance at 0.068 while old-model ROC-AUC fell to 0.675. In the
  monitoring simulation, delayed ROC-AUC changed from 0.841 in week 1 to
  0.729 in week 12.
- **Interpretation candidates — author review required:** these constructed
  outputs are consistent with deployment-shaped validation being stricter
  under drift, repeated identities inflating row-level evidence, validation
  search selecting noise, and input-only monitoring missing conditional
  change. They do not establish effect sizes for real systems.
- **Limitations:** each experiment encodes a deliberately controlled mechanism
  and fixed seed. The leakage proxy is intentionally invalid and nearly reveals
  the label; the grouped experiment uses one-hot entity identity; all
  equal-quality validation candidates share a known simulated accuracy; drift
  is a smooth parametric construction; and the monitoring signals are
  synthetic. These choices isolate concepts but omit real label delay,
  selection, intervention feedback, fairness, policy changes, and operational
  dependencies.

## Key takeaways

- Design the pipeline backwards from the decision and prediction-time
  information boundary.
- A split is a claim about future generalization, not clerical preprocessing.
- Put learned transformations inside cross-validation to prevent leakage.
- Keep the final test set outside model and threshold selection.
- Treat probability estimation and decision policy as separate artifacts.
- Deployment starts the feedback loop; it does not finish the ML lifecycle.
