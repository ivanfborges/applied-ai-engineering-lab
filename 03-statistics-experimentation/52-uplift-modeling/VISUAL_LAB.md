# Offline uplift modeling laboratory

This is a separate visual experiment from the historical analytical study in
[notes.md](notes.md). Its nonlinear two-feature generator, sample sizes, models
and seeds differ. All data are synthetic. Interpretations require author review.

## Run locally

Install the shared dependencies from the repository root in stable Python 3.11+:

```powershell
python -m pip install -e ".[dev]"
```

Then enter this topic directory and run one command:

```powershell
python visualize_uplift.py
```

The generator trains three small forests, runs the sampling studies, exports
four GIFs with Pillow, writes `visualizations/index.html`, and opens it in the
default browser. No server, GPU, FFmpeg, external dataset or network is needed.
Plotly, data, UI code and image bytes are embedded in that single HTML file,
so it can also be copied elsewhere and opened offline. Math is rendered as
plain readable notation, without a remote MathJax dependency.

For headless generation from the repository root:

```powershell
python 03-statistics-experimentation/52-uplift-modeling/visualize_uplift.py --no-open
```

Useful controls:

```powershell
python visualize_uplift.py --no-open --no-gifs
python visualize_uplift.py --seed 521 --repetitions 40 --output-dir visualizations/alternate
python visualize_uplift.py --train 3000 --test 1500 --frames 16 --no-open
```

`--no-gifs` retains every interactive module. A GIF writer failure produces a
static PNG fallback and records its reason, while other exports continue.
Changing seeds/sample sizes requires regeneration; browser sliders operate
on analytical functions or already generated data and do not silently refit.
Regeneration replaces the current dashboard/animations in the chosen output
directory and writes a **new timestamped run record** under its `runs/`.
Historical analytical JSON files under `outputs/` are untouched.

Generated HTML, GIFs, PNG fallbacks, run records and browser screenshots remain
ignored. No public preview is selected. Source files and this guide are the
public reproducibility artifacts; there is no Markdown link to an ignored export.

## A guided tour

| Module | Manipulate | What to learn |
|---|---|---|
| 1. Potential outcomes | Explanation stage, play/pause, explicit oracle reveal | Known probabilities, realized binary potential outcomes and observed outcomes are different objects. Assignment reveals only one realization. |
| 2. Response vs effect | Treatment strength, baseline/treatment/CATE/estimated ranking, animation | Loyal customers can have high response with little incremental effect; fixed-trial estimates must not masquerade as re-estimated effects when a hypothetical probability changes. |
| 3. CATE geometry | Strength, synchronized 3D cameras, zoom/hover | Treatment strength updates both treatment probability and its exact difference from control. Color scales are fixed and zero-centered. |
| 4. S/T mechanics | Effect/error view | Both learners contrast two predictions; both arm errors enter the estimated effect. Same RCT, base settings and independent test set. |
| 5. Sampling | Total N, segment, animated N sweep | Subgroup arm counts determine precision. Inspect approximate pointwise Newcombe intervals over independent trials. |
| 6. Uplift ranking | Targeted percentage | Population-normalized IPW gain, random reference, AUUC and centered Qini area. Every ranking ends at the same estimated ATE. |
| 7. Decisions | Conversion value, contact cost, capacity | Predicted, IPW-estimated and true synthetic policy values differ. Positive effects can be uneconomic. Capacity may remain unused. |
| 8. Identification | Assignment regime | Confounding changes feature populations; extreme weights can destabilize adjustment. Known propensities are a simulation advantage. |
| 9. Production | Pipeline stage, monitoring traffic, propensity logging | Prediction, intervention benefit and decision economics are separate. Measurement and support must survive deployment. |

Each module includes its question, equation, explanation, what to observe,
common misconception and production implication. Navigation and controls
support keyboards. At narrow widths charts stack vertically.

## Mathematical contract

Features are engagement $X_1$ and price sensitivity $X_2$, independent uniform
variables on $[-2.5,2.5]$, interpreted in illustrative standardized units.

$$
L_0=-0.3+0.85X_1-0.65X_2+0.25\sin(X_1X_2),
\qquad h=0.9\tanh X_2-0.7\tanh X_1+0.5\sin(X_1X_2).
$$

$$
\mu_0=\operatorname{sigmoid}(L_0),\qquad
\mu_1=\operatorname{sigmoid}(L_0+s h),\qquad \tau=\mu_1-\mu_0.
$$

The analytical strength slider covers $s\in[0,2]$; model fits use $s=1$.
There is no probability clipping in these response surfaces. Every CATE view
uses a red/cream/blue scale centered on zero with limits $[-1,1]$ probability
points; 0.05 equals five percentage points. Probability surfaces use $[0,1]$.
The potential-outcome animation generates a binary pair for one synthetic
person only; this is a simulation device, not an observed experiment pair.

Training and test data are generated independently at seeds 520 and 521 by
default: 6,000 train / 3,000 test, assignment $e=0.5$. One S model and two T
models use 100 trees, depth 7, minimum leaf size 25 and one worker. Models are
reused across views. No features contain post-treatment data. Synthetic truth
enters diagnostics only, never fitting, ranking or policy construction.

For the independent test set, $Z_i=T_iY_i/e_i-(1-T_i)Y_i/(1-e_i)$ and
$\widehat G(q)=n^{-1}\sum_{i\in S_q}Z_i$. The denominator is **all eligible
test people**, including those not selected. AUUC uses trapezoids through
complete score-tie blocks. Qini area is
$\mathrm{AUUC}-\widehat G(1)/2$, without further normalization. The slider
interpolates uniformly within score ties. Policy selection instead uses stable
input-order ties, independent of outcomes/assignment. Neither procedure uses
holdout outcomes to pick model hyperparameters. Exploring policies on this
test set is educational; final selection would require new evaluation data.

The policy maximizes a uniform-value, equal-cost additive plug-in objective
within capacity: take up to the top $\lfloor n q\rfloor$ scores with
$v\hat\tau_i>c$. Every contact incurs $c$. Predicted conversions sum
$\hat\tau$, true synthetic conversions sum $\tau$, and IPW conversions sum
$Z$ over selected people. Net value is $v$ times conversions minus contact
costs. These are **incremental** values relative to treating nobody. Bars show
totals; the table normalizes by all eligible people. Random targeting is the
expectation over uniform subsets, not a separately executed intervention.
The grid boundary uses the selected cohort score cutoff, not a population
capacity guarantee; equal-score boundary ties remain cohort-specific.

Sampling Module 5 uses four illustrative segments rather than the continuous
surface generator: $(p_0,p_1)$ are $(.85,.87)$, $(.25,.55)$, $(.50,.50)$ and
$(.65,.55)$. Each experiment samples segment membership and 50/50 assignment.
For each of $N=200,1000,5000,20000$, 30 independent trials report unpooled
proportion differences and approximate 95% Newcombe intervals combining two
Wilson score intervals. They are pointwise, not multiplicity-adjusted, and
target the segment-population effect. Empirical coverage has Monte Carlo error.

Module 8 uses 30 independent 2,000-person datasets per regime. Assignment is
0.5, or $\operatorname{clip}(\operatorname{sigmoid}(a(X_1-1.2X_2)),.001,.999)$
with $a=.8$ or $3.5$. Those bounds define the assignment mechanism; the IPW
estimator never silently clips weights. True propensities justify adjustment
here because all assignment drivers are observed. Error comparisons subtract
the known CATE average of each synthetic sample. No fitted propensity nuisance
model or doubly robust extension is used; no cross-fitting claim is made.
Unobserved confounding, interference and positivity failures remain distinct
identification problems.

## Sources and implementation

- [uplift_visual_core.py](uplift_visual_core.py): DGP, S/T fitting, interval,
  sampling, overlap and policy calculations; reuses the historical study's
  tested [IPW/gain helpers](from_scratch.py).
- [visualize_uplift.py](visualize_uplift.py): generator, local bundling and
  timestamped records.
- [uplift_animations.py](uplift_animations.py): four bounded GIFs and fallbacks.
- [visual_lab.html](visual_lab.html) and [visual_lab.js](visual_lab.js): source
  template and browser controls, embedded during generation.
- [tests/test_visual_lab.py](tests/test_visual_lab.py): numerical and temporary
  rendering contracts. [References](references.md) includes primary methods
  and supplier documentation.

Run focused checks from the repository root:

```powershell
python -m pytest -q 03-statistics-experimentation/52-uplift-modeling/tests
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

## Executed visual record — 2026-11-12

**Hypotheses.** Response ranking need not concentrate treatment benefit;
small subgroup samples yield noisy contrasts; poor overlap can increase IPW
variability. These are educational questions, not claims of universal learner
superiority.

**Execution.** Codex ran the full default generator locally on Windows with
`--no-open`, using the existing stable environment
`.local/environments/s08-reference-2026-10-08/Scripts/python.exe`. No dependency
installation was needed. Versions: Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1,
scikit-learn 1.9.1, Plotly 6.9.0, Matplotlib 3.11.2 and Pillow 12.3.0.
Configuration: seed 520, 6,000 training / 3,000 independent test rows, 31×31
grid, 100-tree forests, 30 replications per sampling/assignment design and
24 requested animation frames at 8 fps. Generation took 77.66 seconds in
this environment; that is an observed run time, not a performance guarantee.

The self-contained HTML is 8,965,172 bytes. Four GIFs were generated without
fallback: potential outcomes (66,436 bytes), CATE rotation (1,105,805), model
contrast (679,378) and policy threshold (306,062). Pillow coalesces repeated
potential-outcome stages into five encoded frames with their durations;
the other three contain 24 frames each. Representative first/middle/final
frames were inspected. All exports remain regenerable ignored artifacts.
The timestamped run record preserves configuration, versions, metrics and
all sampling/overlap replication results.

| Main fitted RCT diagnostic | Measured value |
|---|---:|
| Treated train / test counts | 2,943 / 1,516 |
| Known synthetic test mean CATE | -0.018719 |
| IPW holdout ATE | -0.015333 |
| S-Learner test CATE RMSE | 0.075451 |
| T-Learner test CATE RMSE | 0.078807 |
| T-Learner ranking Qini area | 0.032550 |
| Control-response ranking Qini area | -0.028432 |

At value 100, contact cost 5 and capacity 30%, the fixed cost-aware policy
selected 900 test people. Predicted incremental conversions were 142.0692,
IPW-estimated conversions 82, and true expected synthetic conversions
110.1051. Contact cost was 4,500; predicted/IPW/true synthetic incremental
net totals were 9,706.9231 / 3,700 / 6,510.5052. These estimates differ;
none is an observed production outcome or a policy-value confidence interval.

For the persuadable segment, mean Newcombe interval widths at total N of
200, 1,000, 5,000 and 20,000 were 0.483173, 0.229307, 0.103058 and
0.051723. Empirical coverage across 30 replications was 93.33%, 100%, 100%
and 90%. These small coverage counts do not validate nominal 95% coverage;
the module explicitly labels the procedure approximate and pointwise.

| Assignment regime, 30 datasets each | Naive mean error ± MCSE | Known-propensity IPW mean error ± MCSE | IPW RMSE |
|---|---:|---:|---:|
| Randomized | -0.000911 ± 0.003579 | -0.001352 ± 0.005347 | 0.028827 |
| Observed-feature assignment | 0.242939 ± 0.003449 | 0.003551 ± 0.009425 | 0.050881 |
| Poor overlap | 0.327522 ± 0.003995 | -0.048442 ± 0.065439 | 0.355711 |

Errors subtract each dataset's known sample-mean CATE. MCSE is the standard
deviation of the 30 errors divided by $\sqrt{30}$, not an interval for one
person or one deployed policy. In the first poor-overlap dataset, control
and treatment ESS were 6.88 and 161.32, and the largest realized weight was
1,000 despite 2,000 nominal observations.

**Interpretation candidates — author review required.** The new generator
puts response and treatment-benefit signals in conflicting directions, so
response ranking is a poor comparator in this particular run. This differs
from the historical analytical generator; do not pool their results. S had
slightly smaller synthetic CATE RMSE in this one fit, which establishes no
general superiority. Smaller subgroup samples gave wider intervals. Known
propensity adjustment reduced the observed-feature assignment error here,
while poor overlap made IPW variable. These are simulation observations under
fully observed assignment drivers, not guarantees under hidden confounding.

**Verification.** The 23 visual tests passed without skips or warnings, covering
surface identities/bounds, assignment, exact IPW expectation, gain endpoints,
interval boundaries, cost/capacity, temporary HTML bundling and simulated GIF
failure recovery. Chromium was driven through Playwright with networking
disabled, loading the actual `file://` HTML without a server. Thirty-three
checks covered all nine modules, real keyboard controls, dropdowns, animation
play, synchronized 3D drag/zoom, 15 independent Python/JavaScript surface
point comparisons, four targeting fractions, eight policy configurations,
three assignment regimes and a 430-pixel mobile layout. Zero JavaScript
errors and zero external HTTP(S) requests were observed. One Canvas2D
readback performance warning came from browser rendering.

The combined Day 52 suite then passed **67 tests**, without skips or warnings.
Repository validators parsed 355 public Python files and checked 1,173 local
links in 237 Markdown files. JavaScript syntax, dependency consistency and
`git diff --check` passed. A 746-file initial-hash comparison confirmed that
existing changes outside the authorized Day 52 documentation/index updates
and historical analytical records were preserved.

This is Chromium-only local evidence; cross-browser, GitHub rendering and
production behavior were not tested. Intervals do not cover model CATE,
whole gain curves or adaptive policy selection. No causal forest, fitted
observational nuisance model, hidden-confounding correction or deployment
is claimed. The original analytical record remains separate and unchanged.
