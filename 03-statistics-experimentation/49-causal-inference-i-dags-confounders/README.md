# Day 49 — Causal Inference I: DAGs and Confounders

Which variables should we control for when estimating the effect of enabling
an AI assistant? Adding predictors can remove confounding, remove part of the
effect of interest, or introduce collider bias. Prediction quality does not
choose a causal adjustment set.

This study connects DAG assumptions to executable path checks and a **fully
synthetic** structural causal model (SCM). The outcome is an arbitrary
operational performance score, not processing time or a measured product KPI.
The target is the total average treatment effect (ATE), in score units.

## Graph and adjustment choices

The complete edge list is $C\to A$, $C\to Y$, $A\to Y$, $A\to M$,
$M\to Y$, $A\to S$, $U\to S$, and $U\to Y$.

| Variable | Role in this assumed graph | Consequence for total-effect estimation |
|---|---|---|
| $A$ | Binary assistant enablement | Intervention of interest |
| $C$ | Binary case complexity | Blocks $A\leftarrow C\to Y$ |
| $M$ | Continuous adoption proxy; mediator | Adjustment removes the mediated pathway |
| $S$ | Continuous feedback-participation proxy; collider | Adjustment opens $A\to S\leftarrow U\to Y$ |
| $U$ | Latent capability; independent of assignment given $C$ | Causes $S$ and $Y$; not itself a treatment confounder here |
| $Y$ | Performance score | Outcome |

The original backdoor criterion accepts $\{C\}$ for this graph. The empty set
does not block confounding; sets containing $M$ or $S$ violate its exclusion of
treatment descendants. These are claims conditional on the graph, not facts
learned from the generated sample. The baseline console example adjusts the
collider as a continuous covariate. The separate visual lab also explores
selection by intervals and high-S thresholds.

## Run from the repository root

Use stable Python 3.11+ and the canonical dependencies in
[pyproject.toml](../../pyproject.toml). No specialized causal framework is needed.

```powershell
python -m pip install -e ".[dev]"
python 03-statistics-experimentation/49-causal-inference-i-dags-confounders/example.py --n-samples 50000 --seed 49 --repetitions 10
python -m pytest -q 03-statistics-experimentation/49-causal-inference-i-dags-confounders/tests/test_causal_inference.py
```

The example prints graphical decisions, first-seed treatment counts and
coefficients, oracle intervention truth, and repeated-sample mean, SD and
Monte Carlo standard error (MCSE) of the mean. It checks FWL/scikit-learn
agreement on every regression. It writes no reports or visual assets.

## What is implemented

- [from_scratch.py](from_scratch.py): validated tiny DAGs, simple-path
  d-separation, the original sufficient backdoor criterion, intercept-aware
  FWL residualization and discrete-stratum standardization.
- [example.py](example.py): declared SCM, four scikit-learn regressions,
  standardization and repeated samples with deterministic seeds.
- [tests](tests/test_causal_inference.py): chains, forks, colliders and their
  descendants, multiple confounding paths, a pretreatment collider, hidden
  confounding, hand-weighted standardization, invalid inputs and OLS agreement.
- [notes.md](notes.md): identification, formulas, assumptions, population
  derivations, dated results, limitations and interview questions.
- [references.md](references.md): primary theory and further reading.

## Interactive visual laboratory

The [visual guide](VISUAL_LAB.md) connects eight modules in a single local
learning sequence: DAG paths, confounding, colliders, mediation, observation
versus intervention, binary 3D geometry, regression adjustments/FWL and bias
versus sample size. Every module includes a question, interaction, mathematical
connection and senior-level application. Educational text is in English.

From the repository root:

```powershell
python 03-statistics-experimentation/49-causal-inference-i-dags-confounders/visualize_causal_inference.py --all --fast
python -m pytest -q 03-statistics-experimentation/49-causal-inference-i-dags-confounders/tests
```

Or, from this topic folder, use `python visualize_causal_inference.py --all`.
The script prints the exact paths. Open the generated `index.html` directly
in a browser; each module HTML also works on its own with Plotly and synthetic
data embedded. No server or internet connection is needed after installation.
The 3D module uses browser WebGL; software rendering was sufficient locally.

Files added for the visual phase:

- [causal_visual_core.py](causal_visual_core.py): NumPy exogenous tapes, pandas
  SCM, OLS/FWL, selection summaries, NetworkX-backed graph checks and repeated
  bias studies. Baseline draws agree with the original console simulator.
- [visualize_causal_inference.py](visualize_causal_inference.py): HTML,
  Matplotlib/Pillow GIFs and PNGs, CLI, configuration and computed run records.
- [causal_lab.js](causal_lab.js): embedded offline controls, downstream
  regeneration, QR regression and Plotly rendering; browser values are checked
  against independent Python calculations.
- [visual tests](tests/test_visual_core.py): scientific invariants, 240
  external graph comparisons, modified equations, convergence, reproducibility
  and bounded exports into temporary test directories.

The three GIF exports are `assets/confounding_explained.gif`,
`assets/collider_bias_explained.gif` and `assets/mediation_explained.gif` inside
the generated output directory. DAG and coefficient-comparison PNGs are also
generated. All exports remain ignored, pending the author's preview selection;
no generated asset is linked as a public GitHub preview.

## Expected quantities and evidence boundaries

By construction, the population total effect is $2.0$ and the direct path
coefficient is $1.3$. Population regression slopes are $2.9$ without controls,
$2.0$ with $C$, $1.3$ with $C,M$, and $1.04$ with $C,S$. Those are analytical
targets, distinct from finite-sample output; see the
[dated executed study](notes.md#executed-synthetic-study--2026-11-12).
**Interpretation candidates require author review.**

FWL verifies an algebraic identity, not exchangeability. Standardization needs
both treatment arms in each target stratum. A graph cannot certify measurement
quality, overlap, intervention consistency or absence of hidden confounding.
The direct-effect interpretation requires the additional linear, additive,
independent-disturbance assumptions used here.

Path enumeration is deliberately transparent and can be exponentially costly;
it is not a production graph engine or a general causal-identification solver.
The numerical routines provide point estimates, without observational-data
confidence intervals. No real dataset, deployment or product-benefit claim is
involved. The central lesson is to define the estimand and identification
assumptions before choosing the estimator.

[Module index](../README.md) · [Day 13 foundations](../../00-foundations/13-correlation-causation/)
