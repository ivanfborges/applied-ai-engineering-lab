# Day 33 — Imbalanced Data

How should a classifier handle a rare positive class when missed cases,
false alerts, and review volume have different consequences? This study
separates four decisions: class weighting changes the loss, resampling changes
training exposure, thresholding changes actions, and metrics define evaluation.

The example compares logistic regression with unchanged training data,
balanced class weights, random oversampling, random undersampling, and
educational numerical SMOTE. Each model is evaluated at 0.5 and at a threshold
selected **only on validation data** to minimize `FP + 10 × FN`.
Those costs are illustrative units, with no business or monetary claim.

These ideas apply to fraud alerts, defect detection, document escalation, and
human review routing. A rare class does not automatically require balancing;
positive sample count, feature geometry, error costs, and deployment prevalence
matter.

## Implementation

| File | Purpose |
|---|---|
| [notes.md](notes.md) | Weighted loss, SMOTE assumptions, prevalence, threshold derivation, pitfalls, and the executed experiment record |
| [example.py](example.py) | Five training strategies and two decision policies on a fixed synthetic train/validation/test split |
| [from_scratch.py](from_scratch.py) | Tested educational random resampling, SMOTE interpolation, and exhaustive empirical cost threshold selection |
| [tests/test_imbalanced_data.py](tests/test_imbalanced_data.py) | Geometry, class preservation, invalid inputs, threshold ties and endpoints, training-only scaling, and test isolation |
| [interview_questions.md](interview_questions.md) | Questions about weighting, calibration, leakage, and operational decisions |
| [references.md](references.md) | Official documentation and the original SMOTE paper |

Data are **entirely synthetic**, generated with `make_classification`;
no external dataset or download is used. The 6,000 rows have 12 numerical
features and are split 60/20/20 with stratification and seed 42. Label noise
means the requested 3% positive mixture is not exactly the observed prevalence:
the recorded validation and test sets each contain 40 positives out of 1,200.

Scaling fits on original training rows before resampling. Validation and test
sets retain their original distribution. All five models and thresholds freeze
before test scoring; the script does not select a model winner from test results.
The educational sampler requires meaningful continuous numerical geometry and
does not provide a production pipeline or categorical handling.

## Run from the repository root

Use the shared [dependency configuration](../../pyproject.toml):

```bash
python -m pip install -e ".[dev]"
python 01-classical-machine-learning/33-imbalanced-data/example.py
python 01-classical-machine-learning/33-imbalanced-data/from_scratch.py
python -m pytest -q 01-classical-machine-learning/33-imbalanced-data/tests
```

The example prints class counts and test metrics. It overwrites
`outputs/comparison.json` relative to the script, an ignored report containing
configuration, versions, validation selections, confusion counts, measurements,
limitations, and review status. The scratch script prints a small interpolation
and cost-policy demonstration. No figures or notebooks are required.

## Recorded example behavior

Selected rows from the executed seed-42 run:

| Training | Policy | Threshold | Test AP | Test recall | Test alert rate | Test cost |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Fixed | 0.5000 | 0.5841 | 0.2750 | 0.0108 | 292 |
| Baseline | Validation cost | 0.1630 | 0.5841 | 0.6500 | 0.0500 | 174 |
| Balanced weights | Validation cost | 0.7235 | 0.4615 | 0.7250 | 0.0842 | 182 |
| Educational SMOTE | Validation cost | 0.7257 | 0.4496 | 0.7500 | 0.0925 | 181 |

AP means average precision, not trapezoidal PR area. Full results, including
undersampling, oversampling, Brier scores, and the majority baseline, are in the
[experiment record](notes.md#executed-experiment).
These are descriptive measurements from one synthetic split, not benchmarks.
The study-script interpretations received a scoped
[delegated AI-assisted technical review](../../docs/CURATION.md#delegated-technical-review--october-8-2026).
The separate visual experiments remain pending.

## Key takeaways

Compare against an always-negative baseline and report positive counts.
Changing a threshold preserves ranking metrics for the same scores but changes
recall, precision, and workload. Weighting and resampling can change both ranking
and probability estimates. SMOTE interpolation adds assumptions, not independent
evidence. Fit preprocessing and resampling inside training boundaries; choose
policies on validation data and report final behavior on untouched test data.

Earlier studies on [validation](../17-validation-and-leakage/) and
[classification metrics](../23-classification-metrics-ii/) provide the evaluation
background. Calibration and capacity constraints need separate evaluation
before probabilities or policies are used operationally.
