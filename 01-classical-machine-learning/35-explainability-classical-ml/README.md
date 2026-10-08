# Day 35 — Explainability for Classical ML

Which inputs does a fitted model rely on, how does its response vary, and how
can one prediction be attributed to its features? These are different questions.
This study connects feature importance, permutation importance, partial
dependence (PDP), individual conditional expectation (ICE), and SHAP to their
measurement definitions and failure modes.

| Method | Quantity inspected | Important boundary |
|---|---|---|
| Mean decrease in impurity (MDI) | Weighted split impurity reductions in the forest | Training structure; split-selection bias |
| Permutation importance | Change in a selected held-out score after shuffling | Frozen model, metric and perturbation dependent |
| PDP | Average prediction under feature replacement | Marginal averaging may leave joint data support |
| ICE | Each reference row's response under replacement | Heterogeneity depends on output scale |
| SHAP | Allocation of an output relative to a reference baseline | Background, dependence convention and class must be explicit |

These tools help investigate model shortcuts, leakage, cohort behavior, and
individual predictions in tabular applications such as document routing.
A successful explanation identity does not establish causal influence,
calibration, fairness, or an appropriate decision policy.

## The executable study

Data are **entirely synthetic**, with 1,800 rows and four numeric features:
an informative signal, its noisy proxy, context, and independent noise.
The Bernoulli target uses a logistic function with a signal–context interaction.
A fixed random forest is fitted on 1,350 training rows; 450 held-out rows supply
ROC-AUC and permutation diagnostics. There is no tuning or feature selection
from these diagnostics and no external dataset or download.

The example compares MDI with library and educational permutation estimates,
including a shared shuffle of the signal pair. It computes numerical PDP/ICE
on the same grid and rows with both implementations. Interventional TreeSHAP
explains positive-class probabilities for 40 held-out rows against 100 sampled
training rows and checks both reconstruction and the background baseline.

| File | Purpose |
|---|---|
| [notes.md](notes.md) | Definitions, formulas, assumptions, limitations and executed experiment records |
| [example.py](example.py) | Synthetic forest inspection and an ignored numerical report |
| [from_scratch.py](from_scratch.py) | Educational individual/group permutation and direct PDP/ICE helpers |
| [tests/test_explainability.py](tests/test_explainability.py) | Known-function checks, group semantics, invalid inputs, library agreement and class-specific SHAP checks |
| [interview_questions.md](interview_questions.md) | Questions linking attribution definitions to practical decisions |
| [references.md](references.md) | Official inspection documentation and original papers |

## Run from the repository root

Dependencies are centralized in [pyproject.toml](../../pyproject.toml).
SHAP is an optional extra specific to explainability:

```bash
python -m pip install -e ".[dev,explainability]"
python 01-classical-machine-learning/35-explainability-classical-ml/example.py
python 01-classical-machine-learning/35-explainability-classical-ml/from_scratch.py
python -m pytest -q 01-classical-machine-learning/35-explainability-classical-ml/tests
```

The example prints metrics and overwrites `outputs/explainability.json` relative
to its own file. This ignored report includes versions, configuration, selected
row indices, importance estimates, numerical response curves, SHAP checks,
a hypothesis, and limitations. The scratch script prints a known linear
control. No generated asset is needed to read this study.

The executed forest run achieved held-out ROC-AUC **0.815720**. Joint permutation
of signal and proxy reduced ROC-AUC by **0.256499**; individual scratch estimates
were 0.182565 and 0.019186. The largest SHAP reconstruction error was
**5.48e-08**. See the [full experiment record](notes.md#executed-forest-experiment)
for configuration and boundaries. The study-script interpretations received a
scoped [delegated AI-assisted technical review](../../docs/CURATION.md#delegated-technical-review--october-8-2026),
covering these recorded measurements from one synthetic run;
the separate lab's interpretations remain pending.

## What to retain

Importance belongs to a fitted model and inspection protocol, rather than to
a feature in isolation. Individual permutation and joint permutation answer
different reliance questions; neither equals retraining without the feature.
PDP is the mean of ICE only for the same rows, grid and output. Marginal
replacement can violate feature dependencies even within a percentile grid.
SHAP contributions must be read in their stated output units relative to the
chosen background, and additive reconstruction is a bookkeeping check.

Use [validation boundaries](../17-validation-and-leakage/) before interpreting
a model, [Random Forest](../27-random-forest/) for tree importance, and
[correlation versus causation](../../00-foundations/13-correlation-causation/)
for the distinction between prediction and intervention.

## Interactive visual learning lab

Explore the [eight-tab Streamlit lab](app.py) to move tree thresholds, animate
shuffles, watch PDP averaging, select ICE curves, compare 3D model/PDP surfaces,
construct two-feature Shapley values, inspect probability SHAP, and investigate
correlation, unrealistic replacement and deliberately invalid leakage.

Training controls are applied together through a form. Feature, observation,
grid position and centering reuse the fitted forest. SHAP backgrounds and
frozen-model diagnostics have separate bounded caches; hidden tabs do not
compute their expensive views.

After installing the shared dependencies, run from this topic directory:

```powershell
streamlit run app.py
python export_gifs.py
python -m pytest -q tests
```

See the [visual lab guide](VISUAL_LAB.md) for Windows/VS Code setup, equations,
all controls, export policy, executed measurements and suggested experiments.

| Visual-lab file | Purpose |
|---|---|
| [visual_core.py](visual_core.py) | Reproducible training, Gini splits, reliance, surfaces, Shapley/SHAP and pitfall experiments |
| [visualizations.py](visualizations.py) | Interactive Plotly views and Matplotlib PNG panels |
| [export_gifs.py](export_gifs.py) | Three educational GIFs, three 200-dpi PNGs, offline 3D HTML and an experiment record |
| [tests/test_explanations.py](tests/test_explanations.py) | Numerical identities and smoke checks for every tab and optional branch |

All generated exports remain ignored and regenerate locally. No preview asset
is required to read the documentation. The full Day 35 suite passed **63 tests**,
including the existing study tests and the new visual-lab checks. The exported
probability SHAP reconstruction error was **4.69e-08** for the lab's separately
specified configuration. Interpretations remain pending author review.

