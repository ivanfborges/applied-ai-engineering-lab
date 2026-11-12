# Visual learning lab: from selective adoption to causal comparisons

This is a local Matplotlib learning sequence, not a collection of unrelated
charts. Follow chapters 01–08 in order, then use the optional hidden-confounder
chapter and the interactive workbench. All observations are synthetic.

## Install and run

From the repository root, use stable Python 3.11+ and the canonical
[dependency configuration](../../pyproject.toml):

```powershell
python -m pip install -e ".[dev]"
python 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/main.py --quick --seed 50
python 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/main.py --full --seed 50
python 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/main.py --interactive --seed 50
```

Or, from this topic directory, run `python main.py --quick`, `--full` or
`--interactive`. This repository's Python minimum is 3.11 even though the lab
does not require features beyond Python 3.10. There is no separate requirements
file or duplicate version list. No Streamlit, FFmpeg, cloud service, browser,
network request or dedicated causal framework is used by the lab.

An existing ignored output directory is created automatically with a unique
UTC timestamp. To choose a location or render selected chapters:

```powershell
python 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/main.py --quick --seed 71 --only 2 4 6 --output-dir 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/outputs/custom-seed71
```

The generator prints each chapter and the final absolute directory. Outputs
are organized as `figures/*.png`, `animations/*.gif`, and `run.json`. An existing
`run.json` prevents overwriting a completed record: choose a new directory.
Generation uses the headless Agg backend. Interactive mode uses a GUI backend,
normally TkAgg on Windows; Python must include tkinter/Tcl/Tk. Matplotlib's
toolbar can save the interactive figure. Mouse-drag the 3D axes to rotate them.

| Mode | Main units | Assignment frames | Surface rotation | Independent repetitions per effect scenario | GIF fps / dpi |
|---|---:|---:|---:|---:|---|
| Quick | 600 | 18 | 24 views | 24 | 4 / 85 |
| Full | 2,500 | 56 | 72 views | 120 | 6 / 110 |

Matching uses a separate readable illustration of 60 units in quick mode or
100 in full mode; every treated unit is processed, one per frame, followed by
a completed frame. PNG resolution is at least 140 dpi. Figures are closed after
each chapter. Pillow buffers one GIF's frames at a time; full mode deliberately
uses more memory and time than quick mode. Reduce scope with `--only` when needed.

## Source layout

- [visual_core.py](visual_core.py): synthetic potential outcomes, assignment,
  score fitting, population truth integration, diagnostics and independent
  repeated simulations. Reuses [from_scratch.py](from_scratch.py) for matching,
  weighting, SMD and ESS rather than reimplementing estimators in charts.
- [visualizations.py](visualizations.py): the nine learning chapters, consistent
  styling, `FuncAnimation`/`PillowWriter` exports and native Matplotlib controls.
- [main.py](main.py): headless/interactive separation, quick/full presets,
  selected exports and finite JSON run records.
- [tests/test_visual_lab.py](tests/test_visual_lab.py): numerical contracts,
  independent sampling, structural support failure, hidden-variable exclusion,
  widget mouse events and bounded exports into temporary directories.

## Study contract and sign convention

An independent unit is a support team's ticket-resolution summary. This
abstracts away within-team clustering; it is not a ticket-level estimate with
unmodeled repeated observations from the same team. AI is enabled once after
baseline measurement, with a fixed outcome horizon. There are no spillovers in
the generating model.

Draw centered baseline covariates $Z_0,Z_1\sim\operatorname{Uniform}(-2,2)$
independently. Report complexity $X_0=5+2Z_0$ (index) and baseline workload
$X_1=30+10Z_1$ (open tickets). All these features precede treatment. Independent
noise is $\epsilon\sim\operatorname{Uniform}(-3,3)$ minutes. The main design is

$$
e_a(Z)=\operatorname{expit}[-0.3+a(0.9Z_0+0.65Z_1)],\qquad
T=\mathbf{1}\{V<e_a(Z)\},\quad V\sim\operatorname{Uniform}(0,1),
$$

$$
Y(0)=60+10Z_0+6Z_1+\epsilon,\qquad
Y(1)=Y(0)+\tau(Z),\qquad Y=Y(0)+T\tau(Z).
$$

The constant scenario sets $\tau=-8$ minutes: both ATE and ATT are −8. The
heterogeneous scenario sets $\tau(Z)=-8-3Z_0$: more complex teams benefit more,
and selective adoption changes ATT. Negative effects mean **shorter resolution
time**, not deterioration. There is no post-treatment baseline covariate.
Changing effect magnitude in the viewer holds both assignment and $Y(0)$ fixed.

The propensity model fits both centered baseline covariates with full-sample
L2 logistic regression (`C=1`, `lbfgs`, `max_iter=1000`, `tol=1e-9`). This is the
correct functional family for the main generator, with finite-sample shrinkage.
Convergence warnings raise errors. We never clip probabilities or truncate
weights. The experiment is educational and favorable to observed-confounder
adjustment; it does not estimate a real assistant's value.

## Learning sequence

Every chapter identifies what is shown, what changes, how to read it and the
limitation that prevents overinterpretation. Embedded captions preserve those
boundaries when a figure is viewed outside this guide.

| Chapter and question | What is visible / what changes | Interpretation and limitation |
|---|---|---|
| **01. Why can AI teams look slower even if AI helps?** | Baseline 2D scatter, baseline group means, observed resolution means and true/naive contrasts. Assignment strength increases on a fixed exogenous tape. | Baseline selection enters the observed contrast; color changes reflect actual recomputed assignment. One sample path does not establish a general bias curve. |
| **02. What does a propensity score represent geometrically?** | Estimated sigmoid surface over complexity/workload, true wireframe and assignment-colored observations; a 2D contour projection. The GIF rotates the view of the same surface. | Height is adoption probability, not outcome or effect. Two purple stars have equal fitted scores with different covariates. Fitted and true scores can differ. Mouse rotation is available in the viewer. |
| **03. Where are credible comparators missing?** | Treated/control score densities move from strong to moderate to poor practical overlap; empirical common ranges are shaded. A separate structural panel fixes true probability to one above complexity 7.5. | Histogram intersection is a bin-dependent descriptive diagnostic; overlapping ranges can conceal sparse regions. True structural nonpositivity prevents full-population identification, so that panel computes no adjusted estimate. |
| **04. How is a missing counterfactual approximated?** | All controls are candidates. Each treated row is highlighted; its actual fitted-logit neighbor is ringed, a link is added or marked rejected, and the final retained sample and distance histogram are shown. | Matching uses replacement and an inclusive absolute logit caliper of 0.2. Area reflects control multiplicity in the final sample. A score neighbor can be distant in covariate space; rejected treated units change the population. |
| **05. What does weighting change?** | Fixed covariate coordinates and treatment labels; marker area grows in proportion to interpolated weights. Actual weighted complexity densities and contrasts update. An extra static ATT panel changes the weighting target. | Only the final animation frame is the stated IPW estimator. Weights alter contributions, not observed values. No resampling substitute, probability clipping or weight truncation is used. |
| **06. Is measured balance adequate?** | Love plot of both baseline covariates before adjustment, after full matching and after ATE IPW; marginal complexity/workload histograms reveal more than means. | Every SMD uses the same original unweighted pooled sample SD. The 0.1 reference is a heuristic. Mean balance does not prove exchangeability or full distributional balance. |
| **07. How much information survives extreme weights?** | Score densities, actual weight histograms on a log scale, raw counts versus arm ESS and a running IPW estimate. Assignment separation increases. | ESS is calculated per arm, not pooled. A low ESS flags concentration; a large nominal sample does not remove it. The estimate can move nonmonotonically and no fixed amount of bias is claimed. |
| **08. Are methods answering the same question?** | Estimator and target-specific error distributions across independent simulations, separately for constant and heterogeneous effects; population ATE/ATT lines. | Matching error uses ATT; IPW error uses ATE. Naive error is benchmarked to ATE as a diagnostic. Histograms and MCSE are simulation evidence, not observational confidence intervals. |
| **09. Can measured balance miss confounding?** | A separate hidden-variable generator; observed SMDs, oracle-only U imbalance and effect estimates. | U is withheld from score estimation. Its availability to the simulator does not imply it could be diagnosed in real data. This chapter illustrates an untestable identification assumption. |

## Weighting, matching and diagnostics

Matching uses 1:1 nearest neighbors on $\log(\hat e)-\log(1-\hat e)$ with
replacement and deterministic first-control tie-breaking. The final control
weight is its use count. All matching indices refer to original row positions
in the illustrated dataset. Full matching estimates ATT; the caliper contrast
describes retained treated units, with their known mean synthetic effect
reported separately. The visual illustration and 600/2,500-unit study are
different samples and are labeled accordingly.

ATE IPW uses

$$
w_i=\frac{T_i}{\hat e_i}+\frac{1-T_i}{1-\hat e_i}.
$$

ATT IPW uses $w_i=T_i+(1-T_i)\hat e_i/(1-\hat e_i)$. Both contrasts normalize
treated and control contributions separately; see the
[analytical derivations](notes.md#ipw-and-normalization). The IPW animation's
intermediate weight is $w_i(s)=(1-s)+sw_i$ with $s\in[0,1]$; marker **area** and
histogram calculations use this same displayed weight. No observations move.

For each baseline covariate, SMD divides the weighted mean difference by the
original unweighted pooled sample SD. Reused controls contribute their full
multiplicity. The displayed effective sample size is

$$
\operatorname{ESS}_g=\frac{(\sum_{i:T_i=g}w_i)^2}{\sum_{i:T_i=g}w_i^2},\qquad g\in\{0,1\}.
$$

The original routine rescales weights internally for stable ESS arithmetic;
that algebraic normalization does not truncate relative contributions. The
workbench explicitly shows when a zero caliper retains no treated units.

## Repeated simulations and population truth

Chapter 08 draws independent child random streams via NumPy `SeedSequence`;
datasets within a scenario are independent, and the two effect scenarios use
different streams. It retains estimates, generating seeds, sample truths and
population-target errors in `run.json`. Known population ATE and ATT come from
48-point tensor Gauss–Legendre integration of the uniform covariate population.
Tests compare integration orders; quadrature is numerical, not a real-data
identification procedure.

For method $m$ and repetition $r$, error is
$\hat\tau_{m,r}-\tau_m$, using ATT for full matching and ATE for IPW. Reported
mean error, error SD, RMSE and MCSE of mean error describe the repeated
experiment. MCSE is error SD divided by $\sqrt{R}$; it is not an uncertainty
interval for an observational effect. Caliper matching is illustrated
separately and is not silently benchmarked against full-population ATT.

The sweep chapters reuse a single tape and thus do **not** constitute
independent repetitions. The optional hidden-variable chapter adds
$U\sim\operatorname{Uniform}(-1,1)$, $1.5U$ to assignment log odds and $12U$ to
$Y(0)$, while fitting scores on $Z_0,Z_1$ only. The structural support chapter
sets true $e=1$ when $Z_0>1.25$ and deliberately refuses adjusted estimation.

## Interactive workbench

The native viewer has separation, absolute caliper, effect and seed sliders,
a heterogeneous-effect checkbox and a reset button. Changes recompute scores,
matches, weights and diagnostics, then redraw the true/estimated surface,
overlap, Love plot and contrasts. All calculations use the same core as saved
figures. Seed changes generate a new tape; other sliders retain the current
seed. The rotation angle is preserved when a slider redraws the surface.

The control-click smoke mode exercises a native mouse slider event, enables
heterogeneity, checks a zero caliper and saves a GUI screenshot before closing:

```powershell
python 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/main.py --interactive --smoke-test --output-dir 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/outputs/gui-smoke
```

## Validation and artifact review

```powershell
python -m pytest -q 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/tests/test_visual_lab.py
python -m pytest -q 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/tests/test_propensity.py
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

Generated PNGs, GIFs, screenshots and JSON records remain **ignored regenerable
artifacts**, pending author selection. Review `02_propensity_surface`,
`04_matching`, `05_ipw_ate` and `06_balance` first: they explain probability
geometry, counterfactual comparisons, contribution weighting and diagnostics.
Do not publish a preview merely because it rendered. Check readability, sign
convention, sample size, estimand and limitation captions before deliberately
unignoring a small subset. No generated asset is linked as a public preview.

This visual phase has a separate minutes-outcome generator and execution
record. The historical positive-score results in [notes.md](notes.md) remain
unchanged. Interpretation candidates require author review; rendering, tests
and a functioning slider do not establish causal identification or product value.

## Executed visual study — 2026-11-12

**Status:** executed locally by Codex. All interpretation candidates below
require **author review**. This record describes the visual minutes-outcome
generator, separately from the historical score-outcome experiment.

**Hypotheses:** selective enablement can reverse the observed contrast's sign;
weaker practical overlap can concentrate weights and impair comparisons;
heterogeneous effects require different ATE/ATT references; observed balance
can coexist with hidden-confounder bias.

**Configuration:** Windows x64, stable Python 3.11.14, NumPy 2.4.6,
SciPy 1.17.1, scikit-learn 1.9.1, Matplotlib 3.11.2, Pillow 12.3.0 and pytest
9.1.1 in the existing ignored reference environment. Quick mode uses seed 50,
600 main units, 18 assignment frames, 24 rotating views and 24 independent
datasets per effect scenario. The matching illustration has 60 units and
20 treated, producing 21 GIF frames including completion. No dependency was
installed or version changed for this lab.

Executed from the repository root:

```powershell
.local/environments/s08-reference-2026-10-08/Scripts/python.exe 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/main.py --quick --seed 50 --output-dir 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/outputs/visual-2026-11-12-final
.local/environments/s08-reference-2026-10-08/Scripts/python.exe 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/main.py --interactive --smoke-test --seed 50 --output-dir 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/outputs/gui-smoke-final-2026-11-12
```

An initial quick export in `outputs/visual-2026-11-12-quick` completed in
202.2 seconds and is retained as an earlier render. Visual inspection found an
overlap-legend/caption collision and clipped hidden-confounder tick labels.
Their layout was corrected and checked with artist-bound assertions before the
final export. A DejaVu semibold fallback in that initial run was also removed
by requesting the supported bold weight explicitly. Numerical results were
unchanged; these are rendering changes, not new causal experiments.

The final quick export completed in 210.4 seconds: ten 1792×1120 PNGs and six
1088×680 GIFs, with 18/24/18/21/18/18 frames for chapters 01/02/03/04/05/07.
All assets and every GIF frame decoded successfully, and each GIF's first and
last frames differ. Numerical records were compared directly with the initial
run and were identical. PNGs and representative GIF frames were inspected;
the corrected overlap and hidden-confounder PNGs were inspected again.

Final local validation: 65 topic tests passed (44 analytical and 21 visual),
without skips or warnings. Repository syntax parsed 339 Python files; internal
links checked 1,113 targets across 228 Markdown files. Dependency checking found
no broken requirements. `git diff --check` found no whitespace errors; Git's
existing ROADMAP line-ending warning remained. These are focused local checks,
not the complete repository test suite, Linux CI or GitHub-render verification.

**First-sample results (main n=600, seed 50):**

| Assignment strength | Naive contrast | Matching ATT | IPW ATE | IPW ESS control / treated | Maximum IPW weight |
|---:|---:|---:|---:|---|---:|
| 1 | +6.0892 min | −7.6599 min | −7.8891 min | 209.8 / 160.8 | 18.98 |
| 4 | +13.5041 min | −4.9217 min | +3.0896 min | 99.5 / 64.0 | 40.96 |

The true effect is −8 min for every unit in this sweep. Histogram intersection
fell from 0.5192 at strength 1 to 0.1866 at strength 4, with 1.17% versus 46.33%
outside the empirical common score range. At strength 4, the caliper retained
125 of 281 treated units; this is a population restriction. In the separate
60-unit matching illustration, 15 of 20 treated were retained, using eight
distinct controls with maximum reuse four.

**Repeated results (n=600, 24 independent datasets per scenario):** population
ATE is −8 min in both scenarios; ATT is −8 min for the constant effect and
approximately −9.58891 min for the heterogeneous effect. Each error below uses
the appropriate population target; naive uses ATE as an association diagnostic.

| Scenario | Method | Mean error (min) | Error SD | Mean-error MCSE | RMSE |
|---|---|---:|---:|---:|---:|
| Constant | Naive | 13.3526 | 0.9674 | 0.1975 | 13.3861 |
| Constant | Matching ATT | 0.1157 | 0.3664 | 0.0748 | 0.3769 |
| Constant | IPW ATE | 0.3847 | 0.6438 | 0.1314 | 0.7383 |
| Heterogeneous | Naive | 11.9658 | 0.9796 | 0.2000 | 12.0041 |
| Heterogeneous | Matching ATT | 0.1194 | 0.3046 | 0.0622 | 0.3212 |
| Heterogeneous | IPW ATE | 0.4499 | 0.6157 | 0.1257 | 0.7522 |

For the separate hidden-confounder sample, IPW ATE was −3.0598 min although
both observed-covariate absolute SMDs were below 0.01; the true ATE remained
−8 min. The oracle U row shows why measured balance cannot certify no hidden
confounding. This is one illustrative sample, not a quantified sensitivity
analysis for a real dataset.

**Interpretation candidates — author review required:** the observed positive
naive contrasts coexist with known negative effects because baseline selection
is substantial. The strong-separation path has less effective information and
larger error at its endpoint, without proving monotonic behavior across samples.
The heterogeneous experiment separates two valid target populations. The hidden
example shows that a visually reassuring Love plot can miss an identification
failure. None of these findings establishes business benefit from an actual AI
assistant or superiority of a method in general.

**Limitations:** these are synthetic independent team summaries, a favorable
parametric assignment family in the main study, modest repetition counts and
no observational confidence intervals. The matching scene is smaller than the
main study and should not be read as the same sample. Structural support loss
is illustrated separately rather than hidden in estimated probabilities.
Full-quality generation is a configured mode, with separate evidence from
the executed quick exports; no full-mode timing or full-suite CI is inferred.

The native TkAgg smoke used a real mouse slider event, toggled heterogeneity,
set a zero caliper and redrew the 3D axes before saving its screenshot and
closing. Tests also exercise a reset-button mouse event. This checks selected
native interactions, not every possible parameter combination or operating
system. All generated assets and GUI screenshots remain ignored.

## API references

- [Matplotlib Slider example](https://matplotlib.org/stable/gallery/widgets/slider_demo.html)
- [FuncAnimation](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html)
- [PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html)
- [Causal theory references](notes.md#primary-references)
