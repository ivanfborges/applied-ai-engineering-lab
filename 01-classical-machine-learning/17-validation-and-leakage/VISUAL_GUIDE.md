# Day 17 - Visual Validation Lab

This lab makes validation membership and information availability visible.
It connects split design to the population, time horizon and feature access
that a production system will actually encounter.

The lab contains 16 selectable views, producing 17 PNGs, six GIFs and one
interactive Plotly HTML. A local gallery presents the narrative and links to
the experiment records. Everything runs offline with synthetic data.

## Install and run

Use the repository environment and the dependencies in
[pyproject.toml](../../pyproject.toml). No additional package, API key, data
download, notebook or server is required. From the repository root:

```bash
python -m pip install -e ".[dev]"
python 01-classical-machine-learning/17-validation-and-leakage/validation_visual_lab.py
```

The command opens a numbered terminal menu. Choose a view, choose 17 to
generate everything, or enter q to quit. For automation:

```bash
python 01-classical-machine-learning/17-validation-and-leakage/validation_visual_lab.py --all
python 01-classical-machine-learning/17-validation-and-leakage/validation_visual_lab.py --view 4
python 01-classical-machine-learning/17-validation-and-leakage/validation_visual_lab.py --view 14
python 01-classical-machine-learning/17-validation-and-leakage/validation_visual_lab.py --list
```

The default output path is resolved relative to the script, so running from
another working directory is supported. Open `outputs/index.html` in a browser.
Click an image for its full size; open the 3D view to rotate, zoom and inspect
row-level hover information. Plotly JavaScript is embedded in the HTML, so it
does not need a CDN or network connection.
[Plotly's offline HTML documentation](https://plotly.com/python/interactive-html-export/).

The CLI prints the question and hypothesis before each view, then the
interpretation candidate, limitation and saved filenames. `--seed 42` is the
default. Use `--output PATH` to keep another seed's artifacts in a separate
directory; otherwise a rerun replaces that view's files and record. In
noninteractive execution, select `--all` or `--view N` explicitly.

## Follow the narrative

Blue means training, orange validation, and green final test or outer holdout.
Red identifies an invalid information source. Gray means unused in that fit.
Diagrams distinguish schematic coordinates from real feature/time axes.

| Menu | Question answered | Main generated artifact |
|---:|---|---|
| 1 | Where does each row go, and what is each partition for? | `train_val_test_split.gif`, `train_val_test_split.png` |
| 2 | How does the validation block rotate through five model fits? | `kfold_animation.gif` |
| 3 | Do folds preserve the rare class? | `stratified_kfold.png` |
| 4 | Does the model recognize customers already in training? | `group_leakage.png`, `group_membership.png` |
| 5 | Does training contain observations from validation's future? | `temporal_split.png` |
| 6 | Which history survives retraining? | `expanding_window.gif`, `rolling_window.gif` |
| 7 | What changes as the conditional relationship drifts? | `concept_drift.gif`, `concept_drift_comparison.png` |
| 8 | Which rows or labels influence a learned transformation? | `preprocessing_leakage.png`, `preprocessing_scores.png`, `preprocessing_influence.gif` |
| 9 | Is a high-scoring model using a post-outcome feature? | `target_leakage.png`, `target_information_timeline.png` |
| 10 | Can March know the full-year purchase average? | `temporal_feature_leakage.png` |
| 11 | Can choosing configurations fit validation noise? | `validation_overfitting.png` |
| 12 | Where does inner tuning stop and outer evaluation begin? | `nested_cv.png`, `nested_cv_scores.png` |
| 13 | How sensitive is an estimate to the split seed? | `holdout_variability.png` |
| 14 | How does a temporal boundary slice feature space? | `temporal_split_3d.html` |
| 15 | What is available at the prediction timestamp? | `production_information_boundary.png` |
| 16 | Are chunks and repeatedly inspected questions independent? | `rag_document_leakage.png` |

A useful first session is 1, 2, 4, 5, 6, 8, 11 and 15. Views 12 and 14
add the hierarchy of selection and the geometry of a temporal boundary.

## Experimental design and its limits

**Classification and fold rotation.** Shared classification datasets use
`make_classification` with ten features, five informative and two redundant,
requested class weights 0.8/0.2 and class separation 0.8; other generator
parameters retain library defaults. Models use a fresh StandardScaler and
LogisticRegression with C=1 and max_iter=1000 unless stated otherwise.
View 2 trains on four of five shuffled ordinary folds and displays computed
AUCs. Display ordering groups rows by their held-out fold; it is not chronology.
View 3 isolates stratification using exactly 180 negative and 20 positive labels.

**Entity boundaries.** View 4 has 100 synthetic customers and 20 rows per
customer. Eight-dimensional customer signatures are standard normal, transaction
noise has SD 0.06, and a random binary label is shared by a customer's rows.
The classifier is 1-nearest neighbor in all eight dimensions. The scatter
shows two features for the first eight customers to keep identity readable;
the membership heatmaps include all 100 customers. Stable identity is
deliberate, not an observed property of a real dataset. Group separation is
appropriate for the declared new-customer target; existing-customer future
prediction may instead require time-aware evaluation.

**Temporal drift.** Views 6 and 7 use 600 ordered observations with
`x ~ Normal(0, 1)` and
`y = (1 - 2*t/599)*x + Normal(0, 0.25)`. Features are available and labels
arrive immediately. A linear regression using x is fitted under random,
expanding and rolling schedules. The first 480 rows are development data;
four temporal validation windows contain 60 rows each. The rolling history
contains 120 rows. Both predeclared history policies are evaluated on the
last 120 rows, without using those scores to choose a policy.

The drift animation refits using earlier rows and freezes training at row
479 throughout the final test period. The true slope line is known only
because the generator is controlled; it is not a learned forecast.
Random and temporal scores differ in evaluation periods and training sizes,
so their gap is not a clean estimate of one isolated causal effect.
For delayed labels, add availability cutoffs and gaps as discussed in
[notes.md](notes.md); chronological row ordering alone is insufficient.

**Preprocessing.** View 8 generates 500 two-feature normal rows, shifts feature
1 by +4 in the final 100 rows, and sets the label using the sign of
`x1 + x2 + Normal(0,1)`. Three expanding folds with 100-row validation windows
compare global scaling with a fold-local pipeline, using logistic C=0.1.
Histograms show feature 1 and the first training fold. A second control uses
240 rows, 500 noise features, random labels, ANOVA selection of 15 columns,
and four stratified folds. It separates supervised feature-selection leakage
from the often subtler effect of scaling leakage.

The preprocessing animation shifts a copied held-out block to demonstrate
sensitivity of the global mean; it is not another trained-model experiment.
With other seeds, a validation window can lack a class; the code raises an
explicit error instead of presenting an undefined AUC.

**Target and temporal features.** View 9 compares available features with an
extra `y + Normal(0, 0.3)` measurement that is defined to arrive after the
outcome. Both use the same stratified 70/30 split of 900 synthetic rows.
Confusion matrices use a predeclared threshold of 0.5. The high-scoring
invalid control is not deployable. View 10 illustrates an as-of aggregate
for one customer's twelve monthly synthetic purchases; it fits no model.

**Validation overfitting.** View 11 uses 200 hypothetical null classifiers
with independent normal prediction scores, 80 balanced validation labels and
2,000 balanced independent test labels. This is a reproducible simulation of
candidate predictions, not 200 trained ML models. Every prefix winner is
chosen solely by validation AUC; ties retain the earliest candidate. Test
scores are inspected afterward as an audit of the predeclared selection
path. Real searches have correlated candidates, so the observed gap is not
a generic estimate of search optimism.

**Nested CV.** View 12 executes three outer stratified folds, each containing
three inner folds and C in {0.01, 0.1, 1, 10}, on 360 rows. The inner indices
are mapped back to original rows for the diagram and tested for isolation.
The winning pipeline is refitted on the entire outer training set before
outer scoring. This assesses the tuning procedure, not a single final model.
See the [scikit-learn nested CV example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

**Partition variability.** View 13 compares sixteen stratified holdouts with
sixteen means of stratified five-fold CV, on the same 400-row dataset.
Individual fits use 80% of the rows in both procedures. Reusing this dataset
measures partition sensitivity; it does not establish population bias,
independent confidence intervals, or that CV always has smaller variance.

**RAG.** The document diagram targets unseen-document generalization.
A test question's source can legitimately be available in the retrieval index
when that matches production. Keep reference answers out of tuning, group
near-duplicate families when relevant, and reserve questions independently
of prompt development. No RAG system or LLM is executed in this lab.

## Executed evidence

The default-seed run completed all 16 views locally with Python 3.11.0rc2,
NumPy 2.4.6, scikit-learn 1.9.0, matplotlib 3.11.0, Plotly 6.9.0 and Pillow
12.3.0. Results are rounded from the generated records, not hardcoded into
plots:

| Experiment | Observed result |
|---|---|
| Ordinary five-fold CV | Mean AUC 0.8348 |
| Customer validation | Random accuracy 1.0000; grouped accuracy 0.5500 |
| Future test under drift | Expanding MSE 0.9928; rolling MSE 0.2352 |
| Scaling leakage | Global AUC 0.9441; fold-local AUC 0.9425 |
| Post-outcome feature | Available-feature AUC 0.6237; invalid-feature AUC 0.9948 |
| Null-candidate search | Winner 122: validation AUC 0.6881; independent test AUC 0.4993 |
| Nested CV | Mean outer AUC 0.8762 |
| Partition sensitivity | Holdout SD 0.0535; CV-mean SD 0.0086 |

**Interpretation candidates require author review.** The group and
post-outcome controls are consistent with exploiting information excluded by
their declared deployment contracts. The scaling difference is small in this
run. The null-candidate winner's validation result does not persist on
independent test observations. These statements describe controlled synthetic
runs, not benchmarks or personal conclusions.

Every completed view writes `outputs/experiment_NN.json` containing its
question, hypothesis, seed, configuration, package versions, result,
interpretation candidate, limitation and artifact filenames. Structural
diagrams explicitly report that no model was fitted. The gallery reads these
records rather than inventing captions from assumed outcomes.

## Source files and validation

- [validation_visual_lab.py](validation_visual_lab.py): CLI, experiment records
  and local gallery.
- [visual_experiments.py](visual_experiments.py): synthetic data, model fitting,
  metrics and numerical boundary checks.
- [visualizations.py](visualizations.py): Matplotlib, Pillow and Plotly rendering.
- [tests/test_visual_validation.py](tests/test_visual_validation.py): split
  coverage, grouping, temporal isolation, transformation statistics, nested
  index mapping, validation-only selection and artifact smoke checks.

```bash
python -m pytest -q 01-classical-machine-learning/17-validation-and-leakage/tests
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
```

GIF generation uses Pillow with at most 20 requested frames per animation.
The full run uses small fixed datasets, sequential fitting and no service.
The HTML embeds Plotly and is substantially larger than the PNG/GIF files.
Generating a single view avoids rerunning the full lab.

## Output policy and preview candidates

All generated files remain ignored and regenerable. No new preview is
automatically versioned or linked as a public asset. The root README is not
modified by this visual phase.

For later editorial selection, the strongest candidates are:

- `kfold_animation.gif`: validation rotation with actual fold scores.
- `group_leakage.png`: identity, membership and evaluation in one comparison.
- `production_information_boundary.png`: the final availability principle.

Keep the large interactive HTML, local gallery and experiment JSON records
local by default. Review contact sheets named `review_*` are intermediate
inspection files. Before embedding a preview in public Markdown, intentionally
unignore that exact file and verify it is a Git candidate.

The central design question is: **does validation reproduce the information
boundary and generalization problem the system will face in production?**
