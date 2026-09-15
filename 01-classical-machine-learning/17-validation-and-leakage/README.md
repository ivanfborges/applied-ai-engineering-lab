# Train/Validation/Test Split, Cross-Validation and Leakage

Day 17 studies how an offline evaluation can represent the conditions under
which a system will make predictions. A high score is useful only if the
evaluation respects the intended population, entity boundary, and information
available at prediction time.

Training fits parameters. Validation selects features, hyperparameters and
decision policies. A final test set evaluates the frozen procedure. Here,
cross-validation supplies the validation partitions within development data;
it does not require an additional permanent validation set.

## Choose the boundary before the percentage

| Deployment question | Appropriate starting point | What still needs checking |
|---|---|---|
| Another independent observation from the same population? | Random holdout; stratified CV for classification | Duplicates, class support, sampling bias |
| An unseen customer, patient or document? | Group holdout and group CV | Group definition, class balance, group weighting |
| A later period? | Chronological holdout and expanding or rolling windows | Feature timestamps, label delay, retraining schedule |
| Performance of an extensively tuned procedure on limited data? | Nested CV | Both loops must respect groups/time |

These distinctions matter for customer models, forecasting, document
intelligence and prompt evaluation. A pipeline can isolate learned
preprocessing within folds, but it cannot repair a feature derived from the
future or an inappropriate split. See the
[scikit-learn validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).

## Files and execution

- [example.py](example.py): development-only tuning, an intentionally leaking
  feature-selection control, random versus group validation, and temporal indices.
- [from_scratch.py](from_scratch.py): ordinary K-fold index rotation with explicit
  input validation; educational only, without stratification, groups or time.
- [notes.md](notes.md): risk estimates, assumptions, leakage audit, executed
  experiment records, limitations and proposed follow-ups.
- [tests/test_validation.py](tests/test_validation.py): partition coverage,
  reproducibility, input errors, group boundaries, label timing and fold-local scaling.
- [interview_questions.md](interview_questions.md): scenario-based questions and answers.
- [references.md](references.md): official documentation and its role in the study.

From the repository root, use the shared environment defined in
[pyproject.toml](../../pyproject.toml), following the
[root setup instructions](../../README.md):

```bash
python 01-classical-machine-learning/17-validation-and-leakage/example.py
python 01-classical-machine-learning/17-validation-and-leakage/from_scratch.py
python -m pytest -q 01-classical-machine-learning/17-validation-and-leakage/tests
```

All data is synthetic and generated in memory with seed 42; no external dataset
or download is required. The two introductory scripts print to the terminal
and save no artifacts.
The temporal example checks scheduling only; it makes no forecasting claim.

## What the executed examples show

On the recorded local run, global feature selection on random labels produced
mean CV AUC 0.7920, versus 0.4392 with selection inside each fold. For repeated
synthetic customers, random-fold accuracy was 1.0000 versus 0.5690 with group
folds. These are controlled demonstrations, not performance benchmarks.

**Interpretation candidates for author review:** the controls illustrate how
held-out labels and entity signatures can inflate an estimate for the wrong
generalization question. A single seed cannot establish the typical size of
either effect. Full configurations and limitations are in [notes.md](notes.md).

The practical takeaways are to define the prediction boundary first, fit every
learned transformation inside the training partition, preserve a final holdout,
and treat fold dispersion as a diagnostic rather than an automatic confidence
interval.

## Visual Validation Lab

The [visual guide](VISUAL_GUIDE.md) adds a 16-view executable lab: split and
fold animations, customer boundaries, temporal windows under drift,
preprocessing and target leakage, validation overfitting, actual nested CV,
and an offline Plotly 3D comparison. It also covers RAG evaluation boundaries.

```bash
python 01-classical-machine-learning/17-validation-and-leakage/validation_visual_lab.py --all
```

Open `outputs/index.html` afterward. Generated figures, GIFs, interactive HTML
and experiment records remain ignored. The guide records observed results,
limitations and preview candidates for author review.
