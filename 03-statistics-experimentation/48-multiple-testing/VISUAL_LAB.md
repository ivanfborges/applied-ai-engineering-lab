# Day 48 visual laboratory

This is a local, offline laboratory of **synthetic** statistical mechanisms.
It uses NumPy/SciPy simulation, statsmodels reference checks, Matplotlib/Pillow
exports and an interactive Plotly surface. No services, FFmpeg or external
datasets are needed. Interpretation candidates require author review.

The original [baseline study](notes.md#executed-synthetic-study--2026-11-12)
is a separate historical record. Its 10,000 repetitions and random stream must
not be conflated with this lab's quick/full configurations or effect/correlation
sweeps. The baseline files and measurements are preserved.

## Run on Windows

Use stable Python 3.11+ in an activated isolated environment. From the repo root:

```powershell
python -m pip install -e ".[dev,experimentation]"
$topic = "03-statistics-experimentation/48-multiple-testing"
python "$topic/visualize_day48.py" --help
python "$topic/visualize_day48.py" --mode quick
Start-Process "$topic/outputs/quick/index.html"
```

The generated index links the interactive HTML and displays the PNGs and GIFs.
Open the HTML locally; GitHub's source view does not execute an HTML document.
Every HTML dependency is bundled. Parameter sliders, hover, rotation and zoom
work without an internet connection or server.

For a single question or a larger run:

```powershell
python "$topic/visualize_day48.py" --experiment peeking --output-dir "$topic/outputs/peeking-review"
python "$topic/visualize_day48.py" --mode full --output-dir "$topic/outputs/full-review"
python "$topic/visualize_day48.py" --experiment tradeoffs --hypotheses 50 --nonnull-proportion 0.1 --effect 2 --alpha 0.05 --repetitions 5000 --frames 30 --output-dir "$topic/outputs/tradeoffs-review"
python -m pytest -q "$topic/tests"
```

The CLI refuses to overwrite a nonempty directory unless `--overwrite` is
explicit. Prefer a new output directory when preserving a dated experiment.
The `--mode full` command above is an available configuration, not a claim
that it was executed. All generated outputs stay ignored by default. Public
previews are selected copies; running the CLI never updates them automatically.

| Parameter | Quick default | Full default | Meaning |
|---|---:|---:|---|
| Seed | 48 | 48 | PCG64; experiment-specific `SeedSequence([seed, experiment_id])` |
| Repetitions | 2,000 | 20,000 | Independent families/paths per simulated condition |
| Null tests in A | 1,000 | 5,000 | Independent individual null tests, not families |
| Animation frames | 24 | 48 | Maximum requested frames; GIF encoder may merge identical frames |
| Hypotheses per family | 20 | 20 | Configurable via `--hypotheses` |
| Candidate signal fraction | 0.2 | 0.2 | Integer count is `floor(m × proportion)` |
| Non-null Z shift | 3 | 3 | Mean of a unit-variance test statistic; not a business uplift |
| Error level | 0.05 | 0.05 | Alpha for FWER methods, q for FDR methods |
| Planned looks / final n | 10 / 1,000 | 10 / 1,000 | Nested cumulative samples, fixed before outcomes |

The effect sweep uses 0, 1, 2, 3, 4 plus the configured effect if different.
The correlation sweep uses 0, 0.25, 0.5, 0.75 and 0.9. The same noise is reused
across effects/correlations, making comparisons paired; points across the
sweep are not independent. Repetitions within a condition are independent.
Memory is bounded by requiring `hypotheses × repetitions <= 2,000,000`;
frames are limited to 60. Exports run serially using the Agg backend.

## A — Why can a true null be rejected?

Outputs: `01_false_positives.gif` and `01_false_positives.png`.

Generate independent $Z\sim N(0,1)$ and $p=2\Phi(-|Z|)$. These continuous
null p-values are uniform on $[0,1]$. Each frame reveals more tests, highlights
$p\leq\alpha$ and updates false-positive count and proportion. Every rejection
is false because every null is defined to be true in the generator.

The running proportion fluctuates around the nominal individual error level;
finite samples need not equal it. The plot is one sequence of null tests,
not a family-level FWER estimate or evidence that a particular real-world
discovery is false. Reproduce with `--experiment null`.

## B — How does family risk accumulate?

Outputs: `02_fwer_growth.png` and `02_fwer_surface.html`.

These are analytical figures, not simulations:

$$
\mathrm{FWER}(m,\alpha)=1-(1-\alpha)^m.
$$

Every null is true, tests are independent, and the individual rejection
probability equals the threshold. The static plot compares 0.01, 0.05 and
0.10 and marks $m=20,\alpha=0.05$. The 3D surface exposes both independent
variables. Move the two sliders to change the selected family marker and
alpha slice; hover shows exact values. Reset restores both parameters and
camera. Mouse drag rotates and mouse wheel zooms.

The nonlinearity follows the chance of avoiding every error. It does not
describe arbitrary correlated tests. Reproduce with `--experiment fwer`.

## C — What precisely changes between correction rules?

Outputs: `03_correction_comparison.png` and `03_bh_step_up.gif`.

Use the **first family** from independent normal statistics, with the first
configured signal count shifted by the selected effect. The family is not
chosen to maximize differences between methods. Two-sided p-values remain
attached to their original hypothesis identities.

The static panels plot raw ordered p-values against uncorrected, Bonferroni,
Holm and BH thresholds. Triangles identify non-nulls and circles true nulls;
filled markers identify actual rejections. The Holm first failure is marked:
later ranks are not rejected, even if their own boundary would pass.

The animation starts unsorted, moves each identity to its sorted rank, reveals
the BH line and scans for the **largest** qualifying rank. It then rejects
all ranks through that cutoff. Independent per-rank decisions or stopping at
the first failure would implement a different procedure. The deterministic
counterexample $p=(0.03,0.04),q=0.05$ is covered by tests: BH rejects both.

Bonferroni/Holm target FWER; BH targets FDR under independence or suitable
PRDS. One family's discovery count cannot establish power or FDR. Actual
decisions and adjusted values are checked against statsmodels, including BY.
Reproduce with `--experiment corrections`.

## D — How does error protection trade off against detection?

Outputs: `04_error_control_tradeoffs.png` and `04_discovery_simulation.gif`.

Repeat independent normal families with known truth. Track true positives
$S$, false positives $V$, false negatives, discoveries $R=S+V$ and
$\mathrm{FDP}=V/\max(R,1)$. Empirical FDR is the average of all FDP values,
including zero-discovery runs. It is not the pooled ratio of false to total
discoveries. Power is the mean fraction of actual non-nulls rejected.

The static effect sweep compares FWER, FDR, power, and mean true/false discovery
counts. At **zero effect every hypothesis is null**; power is undefined and
that point is omitted from power curves. At positive effects the configured
integer signal count determines the denominator. If it is zero, power remains
undefined throughout. This is an important boundary, not missing data to fill.

Shading and whiskers are pointwise $\pm1.96\,\mathrm{MCSE}$ with
$\mathrm{MCSE}=s/\sqrt B$, using sample standard deviation. Probability bands
are clipped to $[0,1]$ for display. These approximate Monte Carlo uncertainty
bands are neither simultaneous confidence bands nor treatment-effect intervals.
At an empirical boundary a zero MCSE is not proof that an event is impossible.

The animation contrasts individual BH FDP points with running mean FDP for
all four methods and accumulating mean discovery counts. It shows why realized
FDP and expected FDP differ; the running averages are not a sequential testing
procedure. Stricter error control can sacrifice detection, but this generator
does not prescribe the appropriate error budget for a product decision.
Reproduce with `--experiment tradeoffs`.

## E — Why does stopping at significance change the procedure?

Outputs: `05_peeking_problem.gif` and `05_peeking_error_rates.png`.

Generate independent normal **block sums**, with variance equal to the number
of additional observations. Cumulating them is distributionally equivalent to
summing independent $N(0,1)$ observations. At look $n$,
$Z_n=\sum_{j\leq n}X_j/\sqrt n$ has its correct fixed-time null distribution.
Looks share observations, so their p-values are not independent.

All three policies see the same paths: final-horizon testing, first ordinary
$p\leq\alpha$ crossing, and first $p\leq\alpha/L$ crossing over the $L$ planned
looks. The rates and paired naive-minus-fixed difference include MCSE.

For the animation, select the **first** path in the simulated pool that rejects
naively but not at the final horizon. Its index, pool size and selection rule
are recorded in the figure and `run.json`. If no such path exists, use the
first path and disclose that fallback. Never resample until a desired result
appears. Selection illustrates a mechanism, not its frequency. The gray
continuation after naive stopping is counterfactual; the blue observed path
freezes at the stopping look. The final fixed-horizon decision appears only
when its horizon is reached.

The bar chart uses the **whole unselected pool**. Finite-look Bonferroni is
valid with valid marginal p-values at prespecified looks, but conservative.
It is not an optimized group-sequential or anytime-valid design, nor does
correction across metrics alone cover temporal multiplicity. Reproduce with
`--experiment peeking`.

## F — What does dependence change, and what does it not prove?

Output: `06_correlation_effects.png`.

Use a common-factor multivariate normal generator with unit marginal variance
and covariance $(1-\rho)I+\rho\mathbf1\mathbf1^T$. Eigenvalues are checked for
positive semidefiniteness. Only nonnegative equicorrelations are generated.
The validator knows the feasible negative boundary $-1/(m-1)$, but negative
correlations are not part of this experiment.

Compare uncorrected, Bonferroni, Holm, BH and BY under the global null and a
mixed family. Global-null FWER equals FDR; mixed-family panels distinguish
FWER, FDR and power. Bands show pointwise Monte Carlo uncertainty.

Bonferroni and Holm provide strong FWER control with valid null p-values under
arbitrary dependence; BY provides FDR control under arbitrary dependence.
BH's guarantee requires independence or suitable PRDS. Positive correlation
among the underlying **two-sided** Z statistics is not presented as proof of
PRDS for every relevant null p-value configuration. These plots record behavior
in one feasible model and cannot establish a universal guarantee. Reproduce
with `--experiment dependence`. Mathematical references are in the
[topic README](README.md#references).

## Validation and publication boundaries

The [numerical core](day48_visual_core.py) contains no rendering or file writes;
the [generator](visualize_day48.py) handles CLI and exports. The
[tests](tests/test_visual_lab.py) cover analytical FWER, vectorized adjustment
agreement with statsmodels and the original educational core, BH's step-up
case, FDR denominators, PSD matrices, generated p-values, effect-zero truth,
shared monitoring paths, reproducibility and a small full export in temporary
directories. GIF validation decodes every frame and requires multiple distinct
images. PNG validation decodes content; the report records sizes and hashes.

`run.json` records configuration, versions, actual signal count, experiment
statistics, reference-check errors, trajectory selection and artifact validation.
Software tests, visual inspection, browser interaction, scientific interpretation,
public preview selection, CI and publication are separate evidence states.
The CLI does not assert author approval or production readiness.

## Executed quick run — 2026-11-12

**Configuration:** seed 48; 2,000 repetitions per simulated condition; 20
hypotheses; four candidate signals; selected effect 3; alpha/q 0.05; 1,000
individual null tests; 24 requested frames; ten nested looks at 100, 200, ...,
1,000. Run timestamp: `2026-11-12T09:40:10.531569+00:00`.

**Environment:** Windows x64; stable Python 3.11.14; NumPy 2.4.6; SciPy 1.17.1;
statsmodels 0.15.0; Matplotlib 3.11.2; Pillow 12.3.0; Plotly 6.9.0.
No dependencies were installed or changed. Actual generation command:

```powershell
.local/environments/day46/Scripts/python.exe 03-statistics-experimentation/48-multiple-testing/visualize_day48.py --mode quick --output-dir 03-statistics-experimentation/48-multiple-testing/outputs/visual-2026-11-12-quick
```

The environment path records this local run, not a portable setup requirement.
The generated directory contains six PNGs, four GIFs, the standalone surface,
`index.html` and `run.json`. It remains ignored. GIFs contain respectively
24 (null), 17 (BH), 24 (discovery accumulation) and 11 (peeking) distinct decoded
frames. Identical visual stages are merged by the GIF encoder; configured
frame timing is retained. Full mode was not executed.

**A/B hypotheses and observations:** Valid independent null tests still produce
false positives, and their family-level risk accumulates. A produced 48
false rejections out of 1,000 tests: proportion 0.0480, MCSE approximately
0.006763. B is analytical: at 20 tests and alpha 0.05, FWER is 0.641514.
**Interpretation candidate — author review required:** The observed individual
rate is compatible with the nominal level; it does not make the family-level
risk 5%. **Limitation:** A is one sequence, while B requires independence and
exact marginal rejection probabilities.

**C hypothesis and observation:** Different correction algorithms apply
different ordered thresholds. The first generated family rejected 5 hypotheses
uncorrected and 4 each with Bonferroni, Holm and BH. In the bounded reference
checks (one family for C; 32 for D), maximum absolute discrepancy from
statsmodels was at most $2.23\times10^{-16}$, with matching decisions.
**Interpretation candidate — author review required:** Correct methods need
not produce different counts in every family. Their target error rates and
rules differ even here. **Limitation:** This family is not evidence about
average power; it was not replaced to force a more dramatic contrast.

**D hypothesis:** FWER protection and expected-discovery protection imply
different detection trade-offs. At positive effect 3, the independent mixed
family produced these estimates (MCSE in parentheses):

| Method | FWER | FDR | Average non-null power |
|---|---:|---:|---:|
| Uncorrected | 0.5480 (0.011131) | 0.161633 (0.003682) | 0.854375 (0.003951) |
| Bonferroni | 0.0335 (0.004025) | 0.014442 (0.001998) | 0.491875 (0.005596) |
| Holm | 0.0380 (0.004276) | 0.015458 (0.001996) | 0.505375 (0.005732) |
| BH | 0.1305 (0.007534) | 0.039160 (0.002447) | 0.628000 (0.006266) |

**Interpretation candidate — author review required:** BH's FWER and FDR
measure different events, while its observed power lies above the two FWER
procedures here. Holm rejects at least the Bonferroni set by construction.
**Limitation:** This is a single sparsity configuration and finite effect grid,
not a universal performance ranking or a paired significance test of power
differences. Zero-effect points use the correct global-null truth mask.

**E hypothesis:** A significance-based stopping rule changes Type I error.

| Monitoring policy | Null rejection rate | MCSE |
|---|---:|---:|
| Final horizon | 0.0610 | 0.005353 |
| Naive first crossing | 0.2050 | 0.009029 |
| Finite-look Bonferroni | 0.0295 | 0.003784 |

Paired naive-minus-fixed difference: 0.1440, MCSE 0.007853. Selected path:
index 8 (zero-based) in the unfiltered pool of 2,000 paths. Its first ordinary
crossing occurs at $n=200$, $p\approx0.02029$, while its final
$p\approx0.12399$ does not reject.
**Interpretation candidate — author review required:** Additional unadjusted
crossing opportunities increased the null rejection rate in this run. The
fixed-horizon estimate is about 2.05 MCSE above 0.05; finite Monte Carlo results
need not equal the theoretical level. No seed or observations were changed to
make the rate closer to its target. **Limitation:** The animated path is
selected for explanation; rates use all paths. No sequential alternative,
stopping-time efficiency or anytime-valid procedure was evaluated.

**F hypothesis:** Dependence changes family-level behavior despite valid
marginal null p-values. Global-null raw FWER was 0.6350 (MCSE 0.010768) at
rho 0 and 0.1515 (0.008019) at rho 0.9. Mixed-family BH FDR was 0.039195
(0.002390) and 0.022912 (0.003083), respectively. The smallest covariance
eigenvalue over this grid was approximately 0.1.
**Interpretation candidate — author review required:** The independent-null
formula is consistent with the rho-zero estimate but would not describe the
whole correlation sweep. **Limitation:** This positive equicorrelation model
does not establish BH validity under arbitrary dependence or prove PRDS for
general two-sided tests. Intermediate points and all methods are retained in
the ignored run manifest.

**Preview selection:** The BH GIF (189,329 bytes), trade-off PNG (330,947 bytes)
and peeking GIF (265,447 bytes) were inspected and deliberately unignored under
`assets/`. They answer three distinct questions and are the recommended public
figure candidates. The standalone HTML (4,984,397 bytes) is also deliberately
unignored as a downloadable offline interactive artifact. Public copies match
their generated originals; automatic exports never overwrite them. Remaining
outputs and inspection contact sheets stay ignored.

**Verification for this visual phase:** 100 topic tests passed (67 original
study tests plus 33 visual tests), with no skips or test warnings. The small
all-experiment export smoke test wrote only to a temporary directory. Syntax
parsing passed for 326 public Python files; local-link validation passed for
1,062 links in 221 Markdown files. `pip check` found no broken requirements.
The global-null analytical calibration and family-level identities were also
checked against the saved quick-run report, allowing finite Monte Carlo error.

All six PNGs and initial/middle/final decoded frames of all four GIFs were
visually inspected. The bundled HTML was inspected in headless Chrome
157.0.8103.50 using the existing Playwright installation in an offline context:
all ten gallery images decoded; both sliders updated marker, alpha slice and
readout at their endpoints; reset restored defaults; real mouse drag and wheel
changed the camera; real hover displayed a data tooltip. The audit observed
zero JavaScript/console errors and zero HTTP(S) requests. One Canvas2D
`willReadFrequently` performance warning was recorded. This is Chromium-only
evidence, not cross-browser, GitHub rendering, CI or author review.

The Playwright CLI first closed its session during browser startup; the audit
therefore used the existing Playwright library directly. An initial audit
looked for the wrong tooltip DOM selector; the final check captured an actual
mouse-triggered Plotly hover event and inspected the visible tooltip. Neither
issue changed statistical data or generated artifacts. Browser logs and
inspection contact sheets remain ignored.

The root EN/PT READMEs were intentionally preserved for this request. After
review, update the Day 48 paragraph under `Current Status` and the Day 48 row
under `Implemented studies` in the English root README, and the existing
Day 48 paragraph in the Portuguese root README. Point to this topic's visual
guide and describe the six experiments, four GIFs and offline 3D interaction.
The current totals (48 studies; module 3 at 4/8) do not change. Interpretation
review must remain pending until explicitly completed by the author.
