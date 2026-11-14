# Day 50 — Causal Inference II: Propensity Score and Matching

How can we compare treated and untreated units when adoption was selective?
This study estimates assignment probabilities from baseline covariates, then
uses matching and inverse probability weighting (IPW) to construct adjusted
comparisons. The propensity model predicts treatment, not its causal effect.

The example is **fully synthetic**, with two observed common causes, binary
treatment and a known effect of 2 arbitrary outcome units by default. It could
represent selective AI-assistant enablement and an operational performance
score; it contains no customer data or measured business benefit. It builds on
[Day 49's adjustment-set reasoning](../49-causal-inference-i-dags-confounders/).

## Which effect does each method target?

| Method | Target and design |
|---|---|
| Unadjusted difference | Association; compared with ATE truth only as a confounding diagnostic |
| 1:1 propensity matching | ATT among treated units; nearest logit score, with replacement |
| Caliper matching | Contrast for retained treated units; rejected units are counted explicitly |
| Normalized IPW ATE | Difference of weighted means targeting the original eligible population |
| Normalized IPW ATT | Treated mean minus controls weighted by propensity odds |

ATE and ATT coincide in the default constant-effect generator. They need not
coincide in practice. The optional heterogeneous-effect configuration reports
sample-specific truths, including the truth for retained treated units.

Matching uses an inclusive **absolute logit-distance caliper of 0.2**, rather
than 0.2 times a standard deviation. Controls may appear repeatedly; their
multiplicity contributes to balance, the estimate and control ESS. No scores
are clipped and no IPW observations are silently trimmed.

## Run from the repository root

Use stable Python 3.11+ and the canonical dependencies in
[pyproject.toml](../../pyproject.toml). NumPy, SciPy and scikit-learn are enough
for the example; no dedicated causal framework or new optional extra is needed.

```powershell
python -m pip install -e ".[dev]"
python 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/example.py --n-samples 4000 --seed 50 --repetitions 10
python -m pytest -q 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/tests/test_propensity.py
```

The program compares assignment strengths 1 and 3 using seeds 50–59. It prints
first-seed estimates, each covariate's signed SMD, score quantiles by arm,
empirical common-range exclusions, maximum weights, arm-specific ESS, match
distances and retained counts. Repeated errors include SD, Monte Carlo standard
error (MCSE) of the mean error and RMSE. It writes no reports or assets.

## Files and evidence

- [from_scratch.py](from_scratch.py): transparent NumPy matching, ATE/ATT
  weights, normalized contrasts, ESS and fixed-reference SMD.
- [example.py](example.py): declared data-generating process, full-sample
  L2 logistic fit and paired-seed overlap stress study.
- [tests/test_propensity.py](tests/test_propensity.py): hand-weighted
  calculations, scikit-learn nearest-neighbor agreement, exact heterogeneous
  strata, inclusive calipers, reused controls, diagnostics and invalid inputs.
- [notes.md](notes.md): identification, estimands, formulas, design trade-offs,
  dated execution record, primary references and interview discussion.

The [executed study](notes.md#executed-synthetic-study--2026-11-12) records
versions, configuration and results. In the baseline first seed, the unadjusted
contrast was 4.3450, matching ATT 1.9670 and IPW ATE 2.0664, against truth 2.
With weaker practical overlap, first-seed IPW ATE treated ESS fell from 971.2
to 33.1, and uncalipered matching reused one control 504 times.
The two historical-study interpretations received a scoped AI-assistant
technical review ([50-A/B](../../docs/CURATION.md)), not personal author
approval. Separate visual interpretations remain pending.

## Interactive visual learning lab

The separate [visual guide](VISUAL_LAB.md) follows nine chapters from selective
adoption to assignment geometry, overlap, matching, weighting, balance,
effective information, repeated estimation and hidden confounding. It uses
synthetic **resolution time in minutes**, with a negative effect meaning time
saved. Its generator and records are distinct from the positive-score study
above; historical results in `notes.md` remain unchanged.

```powershell
python 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/main.py --quick --seed 50
python 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/main.py --full --seed 50
python 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/main.py --interactive --seed 50
python -m pytest -q 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/tests/test_visual_lab.py
```

The [entrypoint](main.py) generates ten PNGs and six Pillow GIFs, including a
rotating 3D propensity surface and matching/IPW animations. The native
Matplotlib viewer has separation, caliper, effect and seed sliders, a
heterogeneity checkbox and mouse rotation. Headless exports use Agg; the viewer
needs a GUI backend such as TkAgg. FFmpeg and network access are unnecessary.

[visual_core.py](visual_core.py) separates the synthetic model and numerical
evidence from [visualizations.py](visualizations.py). The
[visual tests](tests/test_visual_lab.py) check effects, targets, support,
matching/weighting, actual control events and bounded Pillow exports. Outputs
and finite JSON run records remain ignored pending author preview selection.
No generated asset is currently linked as a public preview. Interpretation
review remains pending.

## What these diagnostics establish

Measured balance and overlap help assess the comparison design; they cannot
establish no unmeasured confounding, consistency or absence of interference.
Inspect each important baseline covariate, distributional features and the
target population before interpreting an effect. High treatment-prediction AUC
is insufficient evidence of a good causal design.

The matching routine takes $O(n_T n_C)$ time and $O(n)$ auxiliary memory, making
it educational rather than a production matcher. The fitted logistic family
matches the simulated assignment mechanism, but L2 shrinkage and finite samples
still matter. There are no observational-data confidence intervals, matching
variance estimator, unmeasured-confounding sensitivity analysis or real-data
causal claims. MCSE concerns repeated-simulation precision only. A small SMD
does not certify identification; discarding treated units changes the reported
population even when the default constant effect hides that distinction.

[Module index](../README.md) · [Technical notes](notes.md)
