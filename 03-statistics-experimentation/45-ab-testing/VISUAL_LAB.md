# Day 45 visual lab — navigation, mechanisms and evidence

This lab has **12 independent teaching modules**. All observations are synthetic;
analytical curves are labeled separately. Interpretations require author review.
The original nonvisual study record in notes.md remains a separate execution.

**Local verification — 2026-11-11:** all twelve modules completed on Python
3.13.12, followed by 75 passing tests and offline browser checks across 42
scenarios. The final light render took 69.94 seconds and peaked at 256.88 MiB
in its monitored Python process, using one CPU core. This is one observed local
run, not a performance guarantee. The earlier interrupted run is preserved as
a separate historical record below. Run generation, installation and tests
sequentially; do not resume concurrent heavy jobs.

## Run locally

Target: Windows, Python 3.13+, VS Code, CPU only. Dependencies use the repository's
canonical [pyproject.toml](../../pyproject.toml); no topic requirements file or
application framework is needed. There are no cloud services, credentials,
external datasets, FFmpeg or ImageMagick dependencies.

From the repository root, activate your Python environment and run:

```powershell
python -m pip install -e ".[dev]"
python 03-statistics-experimentation/45-ab-testing/visualize_ab_testing.py --help
python 03-statistics-experimentation/45-ab-testing/visualize_ab_testing.py --quick
python 03-statistics-experimentation/45-ab-testing/visualize_ab_testing.py --all
python 03-statistics-experimentation/45-ab-testing/visualize_ab_testing.py --module confidence_intervals --seed 45
python -m pytest -q 03-statistics-experimentation/45-ab-testing/tests/test_visual_lab.py
```

Open the printed `index.html` path directly in a browser. Each module has its
own standalone HTML with embedded Plotly JavaScript, a PNG companion, and a GIF
when the concept benefits from a process animation. Internet access and a server
are not needed to use the generated files.

`--quick` (also the default) generates modules 1, 3, 4, 8 and 9 with 600
repetitions and lightweight GIFs. `--all` defaults to 2,000 repetitions. Use
`--module` with the names below to regenerate one module. Rendering budgets:

| Quality | DPI | Maximum GIF steps | FPS |
|---|---:|---:|---:|
| `--quality light` (default) | 85 | 8 | 3 |
| `--quality normal` | 110 | 20 | 5 |
| `--quality high` | 140 | 32 | 7 |

The number of distinct encoded frames can be smaller than the step budget when
integer steps repeat. The generator requires at least two frames for every GIF.
`--no-gifs` explicitly skips animation rendering for numerical/browser checks.
Higher quality uses more time and memory; no GPU is used. Normal and high
profiles were not rendered in the final verification.

The renderer limits numerical-library threads to one and simulates clustered
experiments in batches of 128. The recorded verification also used a private
Windows wrapper with one-core affinity and a 512 MiB process-memory cutoff.
That wrapper is not part of the public CLI; the CLI itself does not impose an
operating-system RAM quota. Light mode and serial execution are recommended.

## Parameters and offline controls

```powershell
python 03-statistics-experimentation/45-ab-testing/visualize_ab_testing.py --module sampling_distributions --seed 91 --baseline 0.10 --treatment 0.105 --alpha 0.01 --n-per-arm 20000 --repetitions 1000 --no-gifs
```

This alternative configuration is a suggested exploration, not a recorded run.

Seeds, repetitions, baseline, treatment, alpha and per-arm sample size are CLI
parameters. Streams use seed/module/scenario IDs, so running one module does
not alter another module's random draws. Repetitions are independent experiments,
not repeated looks, except for the explicitly paired peeking module.

HTML dropdowns switch **precomputed results**. Play cycles the same scenarios.
They cannot fit arbitrary new parameter combinations. Use the CLI for free-form
updates, then open the new run. Power curves and the MDE axis cover predefined
ranges; the 3D surface has a fixed displayed baseline and alpha. Rotation and
zoom change the view, not the mathematical parameters.
The power module explores its own labeled effect grid rather than the single
`--treatment` value. Treatment-effect intervals use confidence level
$1-\alpha$ (95% by default); Monte Carlo frequency intervals remain 95%.

Every run creates a new directory under `visualizations/outputs/`, with files
named `01_randomization.png`, `01_randomization.html`, etc., plus `index.html`
and `run.json`. An explicit `--output` must identify a new or empty directory.
Existing runs are never overwritten. A failed generation can leave partial
artifacts; an index and run.json are written only after all selected modules
finish. Select a new run path after fixing a failure.

## Question → mechanism → interpretation

| Module / CLI name | Visualization and mathematical meaning | Expected behavior and interpretation | Assumptions and limitations |
|---|---|---|---|
| **1** `randomization` | Animated user assignment, counts, activity histograms, standardized baseline differences, interactive distributions of engagement/activity/history and a simulator-only latent trait. Rerandomizations hold the same population fixed. | Differences fluctuate around zero across allocations; finite samples need not balance exactly. | Synthetic correlated covariates, 320 users. Latent values are available only to the simulator. Covariate balance cannot prove assignment validity. |
| **2** `potential_outcomes` | Oracle and observed-only views of binary $Y(0),Y(1)$; individual effects, expected effect, finite-population ATE and randomized estimate. | Randomization estimates an average, while the individual counterfactual remains unobserved. | 2,400 users; an explicit 8 pp increment makes the mechanism visible. A shared uniform draw imposes monotonicity. This is a separate DGP, not the 1 pp homogeneous example. |
| **3** `sampling_distributions` | Independent binomial A/B repetitions under null and alternative. Separate axes show the estimator in pp and pooled Z without units. Accumulation GIF, critical Z thresholds and a first, unselected observed experiment. | Histograms stabilize with repetitions; the alternative shifts. Rejection means incompatibility with the equality null under the model, not certainty of benefit. | Balanced independent users; empirical pooled SE per experiment. Only Z has fixed test thresholds. A single fixed threshold on the effect axis would not reproduce every estimated-variance test. |
| **4** `confidence_intervals` | Repeated Wald intervals with colors determined by whether each contains the fixed true effect. Compare 1,000 with configured users per arm. | Larger samples narrow intervals; repeated coverage is approximately nominal under regularity. | First 80 intervals are displayed; full repetition coverage is separately reported. No posterior probability interpretation. Wald coverage is approximate. |
| **5** `power` | Normal power curves, Monte Carlo points with exact 95% simulation intervals, 80/90% reference lines, and users per arm against MDE. Dropdown changes baseline/alpha. | Smaller effects need substantially more users for fixed power; roughly inverse-square scaling locally. | Analytical power uses pooled null threshold and alternative variance. It is not exact power of a discrete test. Equal allocation, independent binary users, fixed horizon. |
| **6** `power_surface` | Rotatable, hoverable 3D surface: users per arm × true effect in pp × analytical power. An 80% contour and a 2D PNG companion expose the design boundary. | Larger n and larger absolute effect increase power in the displayed positive-effect domain. | Fixed displayed baseline/alpha. Surface is an analytical approximation, not fitted Monte Carlo evidence. |
| **7** `errors` | Null/alternative normal sampling densities with both Type I tails, Type II area and power. Play/dropdown changes alpha and effect. | At fixed separation, smaller alpha reduces Type I error and increases Type II error. | Explicit theoretical normal Z model with alternative mean and variance; not posterior densities over hypotheses. |
| **8** `aa_testing` | Null p-value histogram, accumulation GIF and cumulative rejection frequency with the nominal alpha reference. | Random noise produces false positives; empirical frequency fluctuates and stabilizes with repetitions. | Normal-test p-values are only approximately uniform with discrete binomial counts. This does not exercise a production platform. |
| **9** `peeking` | First five unselected p-value paths, false-positive rates for paired policies, first-crossing times and unrejected experiments censored at the horizon. GIF reveals predefined looks. | Any-crossing rejection can exceed final-only rejection on the same data. | 30 planned looks through the same maximum users per arm. Gray bars are censoring, not significant stops. Valid sequential inference is a separate method. |
| **10** `srm` | Exact binomial SRM for healthy intended 50/50, actual 40/60 against intended 50/50, and the same 40/60 counts against intended 40/60. | Mismatch depends on the intended allocation. A warning requires investigation before interpreting effects. | 3,000 assigned users, independent Bernoulli assignment; a passing test does not prove validity. Blocked assignment needs another count model. |
| **11** `unit_of_analysis` | Beta-binomial event clusters, matched effect estimates with naive request vs user-mean Welch intervals, and repeated null rejection rates at ICC 0, .05 and .20. | Treating correlated events as independent understates uncertainty. | 500 independent users per arm, 20 events each. Equal events preserve the same estimand. Unequal cluster sizes require an explicit weighting/estimand decision. |
| **12** `ai_product` | Synthetic primary-effect interval/test, latency, cost, safety, retrieval proxy, docs and tokens; metric ledger and predefined gate dashboard. | Statistical evidence, worthwhile impact and operational constraints answer different questions. | One synthetic request per assigned user. Latency/cost gates use point estimates; safety uses a one-sided exact upper bound. No joint assurance, safety certification or production approval. |

## Statistical contract

For binary user outcomes the pooled null standard error is

$$
SE_0=\sqrt{\hat p(1-\hat p)(1/n_A+1/n_B)},\quad
\hat p=(x_A+x_B)/(n_A+n_B),\quad
Z=(\hat p_B-\hat p_A)/SE_0.
$$

The p-value is $2\Phi(-|Z|)$. The Wald interval uses the separate unpooled
$SE_\Delta=\sqrt{\hat p_A(1-\hat p_A)/n_A+\hat p_B(1-\hat p_B)/n_B}$.
Their distinction and the difference between percentage points and relative
lift also appear in the [original notes](notes.md).

Sparse tables are refused explicitly, not silently dropped or assigned a
convenient p-value. Requiring five observed successes and failures per arm is
only a heuristic. At rare/extreme rates use more suitable score/exact inference.
No clipping is used to hide invalid Wald endpoints. The lab bounds sample sizes
and repetition counts; it is not a production inference library.

For normal power, let $\delta=p_1-p_0$, $\bar p=(p_0+p_1)/2$,
$c=z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)/n}$ and
$s=\sqrt{[p_0(1-p_0)+p_1(1-p_1)]/n}$. The plotted approximation is

$$
\Pr(\text{reject})\approx\Phi((-c-\delta)/s)+1-\Phi((c-\delta)/s).
$$

Monte Carlo power uses actual pooled-test rejections from independent binomial
samples. Its uncertainty is simulation uncertainty, not a treatment-effect CI.

For event ICC $\rho$, beta-binomial user risks have concentration
$1/\rho-1$. With $m$ events/user the theoretical design effect is
$1+(m-1)\rho$. The correct comparison uses independent user means and Welch
standard errors/df; the naive request Z-test is displayed as the counterexample.

## Illustrative AI launch policy (predefined, synthetic)

- Primary equality test rejects at configured alpha with a positive lower bound.
- Primary lower Wald bound exceeds a minimum worthwhile **0.5 pp** improvement.
- Treatment/control P95 latency ratio is at most **1.25**.
- Incremental cost per eligible assigned user is at most **USD 0.005**.
- Treatment unacceptable-response probability has a one-sided exact 95% upper
  bound at most **0.3%**.

A passing display means only **candidate for human review**. A failing display
means **HOLD for investigation**. Retrieval quality is a generated diagnostic
proxy, not an evaluated public retrieval dataset. Safety events are generated
Bernoulli indicators, not a validated safety evaluation. The experiment does
not claim these hypothetical product assumptions describe a real RAG system.

## Output policy and preview candidates

All generated assets remain ignored. No output has been automatically selected,
unignored or linked as a public asset. After inspecting your own run, the
strongest candidate set to review is:

1. `04_confidence_intervals.gif` — repeated coverage and interval width.
2. `09_peeking.gif` — paired fixed-horizon versus naive stopping.
3. `01_randomization.gif` — assignment and realized baseline differences.
4. `03_sampling_distributions.png` — estimator versus Z and rejection mechanics.
5. `11_unit_of_analysis.png` — the independence error and user-level correction.

Keep the large self-contained HTML files regenerable locally, including the
3D surface. Public preview decisions and root README changes require author
review; neither is performed by this generator.

## Verification and execution record

### Final reviewed-source execution — 2026-11-11

Hypotheses: randomization balances characteristics in expectation; regular
Wald intervals have approximately nominal repeated coverage; pooled tests
control fixed-horizon null rejection approximately at alpha; repeated naive
peeking and ignoring within-user dependence can inflate false positives.
These are simulation questions, not claims about a deployed experiment system.

Configuration: seed **45**, **2,000 independent repetitions**, **15,000 users
per arm**, control **10%**, treatment **11%**, two-sided **alpha 0.05**, light
rendering. Module-specific departures are listed in the navigation table.
Python **3.13.12**; NumPy **2.5.4**, SciPy **1.18.1**, Matplotlib **3.11.2**,
Plotly **6.9.0**, Pillow **12.3.0**, pytest **9.2.0**. The required numerical,
visual and test dependencies were installed in an isolated environment; their
requirements were taken from the canonical project dependency declarations.

Executed from the repository root (with the monitored Python environment):

```powershell
python 03-statistics-experimentation/45-ab-testing/visualize_ab_testing.py --all --quality light --repetitions 2000 --seed 45 --output 03-statistics-experimentation/45-ab-testing/visualizations/outputs/final-2026-11-11
python 03-statistics-experimentation/45-ab-testing/visualize_ab_testing.py --quick --repetitions 100 --no-gifs --output 03-statistics-experimentation/45-ab-testing/visualizations/outputs/quick-smoke-py313-2026-11-11
python -m pytest -q 03-statistics-experimentation/45-ab-testing/tests
```

The final bundle contains **12 PNGs, 5 GIFs, 12 module HTMLs, index.html and
run.json**: 31 files, including 29 module assets totaling 59.33 MiB. Each GIF
contains **eight distinct frames**. The full run took **69.94 s**, with monitored
Python peak **256.88 MiB**; quick/no-GIF smoke took **21.93 s**, peak **197.13 MiB**.
These process measurements exclude the browser and other system processes.

The following are actual results from that final synthetic run. Interpretation
candidates in this table all require **author review**; limitations in the
navigation table continue to apply. Counts refer to all repetitions, not just
the first intervals or paths shown on screen.

| Module | Recorded result | Interpretation candidate / limit |
|---|---|---|
| Randomization | 156 Control / 164 Treatment users. Realized standardized differences ranged from -0.200 to -0.127 across four characteristics; reallocation mean differences were within 0.0021 of zero. | A realized imbalance can coexist with balance in expectation; this synthetic exercise cannot validate production assignment. |
| Potential outcomes | Expected ATE 8 pp; finite-population ATE 9 pp; randomized estimate 11.745 pp. | Expected, finite-population and estimated effects differ. Counterfactual pairs are simulator-only; the observed-only HTML hides oracle effects. |
| Sampling distributions | Null rejected 101/2,000 (5.05%); alternative rejected 1,620/2,000 (81.0%). First unselected alternative estimate 1.240 pp, pooled Z 3.509, p 0.000450. | Estimator and Z have different units and sampling distributions. One significant synthetic draw is not evidence about a real product. |
| Confidence intervals | Coverage: 1,891/2,000 (94.55%) at n=1,000, and 1,890/2,000 (94.50%) at n=15,000. Mean widths: 5.372 and 1.388 pp. | Larger samples narrow intervals. These empirical coverages have Monte Carlo uncertainty and are not forced to equal 95%. Displayed first-80 coverage differs from full-run coverage. |
| Power / MDE | At 1 pp and n=15,000: analytical power 80.653%; Monte Carlo 1,588/2,000 = 79.4%, exact 95% simulation interval [77.56%, 81.15%]. Approximate 80% planning target: 14,751 users/arm. | Analytical and empirical values are distinct; their agreement is conditional on the independent-binomial design and approximation. |
| Power surface | Analytical grid: 1,000–40,000 users/arm and 0–2.5 pp effect, baseline 10%, alpha 0.05, with 80% contour. | This surface is a model calculation, not a separate empirical power experiment. |
| Errors | At a 1 pp effect, analytical Type II probability 19.347%, power 80.653%. | Areas are probabilities of data under specified sampling models, not posterior probabilities of hypotheses. |
| A/A | 101/2,000 false positives: 5.05%; exact 95% simulation interval [4.13%, 6.10%], versus theoretical nominal 5%. | Compatible with approximate null control in this configuration; not proof of exact size or platform validity. |
| Peeking | Same 30 looks, maximum n=15,000/arm: fixed horizon 98/2,000 (4.9%); naive any-crossing 543/2,000 (27.15%); 1,457 paths censored without rejection. | Naive repeated fixed-horizon tests inflate false positives here. The example is not a valid sequential testing procedure. |
| SRM | Healthy 1,537/3,000 Treatment against intended 50%: p=0.183. Biased 1,819/3,000 against intended 50%: p=1.73e-31; same count against intended 60%: p=0.491. | Mismatch depends on intended allocation; investigate a warning before interpreting outcomes. |
| Correlated events | ICC 0: naive/user false-positive rates 4.8%/5.0%; ICC .05: 15.2%/5.35%; ICC .20: 38.9%/4.75%. At ICC .20, theoretical design effect 4.8. | Request independence fails under clustering. Independent-user Welch inference behaves differently; equal cluster size is part of this design. |
| AI product | Task completion 10.0467% → 10.6800%; +0.6333 pp, +6.304% relative lift; Wald interval [-0.0564, 1.3231] pp, p=0.07193. P95 latency 1.735 → 2.927 s; cost USD 0.01278 → 0.02231 per assigned user; Treatment safety upper bound 0.3695%. | Predefined policy: **HOLD**; all five gates fail. The primary result is not significant at .05. These synthetic outcomes establish neither a benefit nor safe deployment. |

The 75 tests include independent SciPy reference comparisons for the pooled
test and user-level Welch inference, interval/sample-size checks, independent
random streams, Monte Carlo null/coverage checks, invalid inputs, sparse counts,
negative-effect visibility, configurable confidence labels, oracle hiding and
GIF/surface rendering. Dependency consistency passed `pip check`.
The four topic scripts compiled with `py_compile`; repository validation parsed
298 Python files and checked 971 internal links in 210 documents.
`git diff --check` and a separate whitespace scan of all 12 public Module 3 file
candidates passed. The CLI help command was also executed.

An offline Chromium audit using a cached Playwright library checked **12 pages
and 42 scenario selections**, compared displayed traces with scenario data,
exercised Play/Pause, and changed the 3D camera through mouse rotation and zoom.
It recorded **zero JavaScript/console errors and zero HTTP(S) page requests**.
The artifact audit decoded all PNGs and GIFs, checked standalone HTML resources,
and verified distinct GIF frames. Representative PNGs, browser screenshots and
first/middle/last GIF frames were visually inspected. This is local Chromium
verification; other browsers, normal/high rendering profiles and CI were not run.

### Historical interruption and pre-review runs

The complete intermediate run `validated-2026-11-11` was retained separately.
Visual inspection led to presentation corrections before `final-2026-11-11`:
readable potential-outcome pairs, hidden oracle effects in the observed-only
view and a clearer error-region legend. Final measurements above refer only
to the final bundle; intermediate assets are not selected previews.

The quick run `quick-2026-11-11` contains five PNGs, five GIFs, five module HTMLs,
an index and run.json. Its recorded modules are randomization, sampling
distributions, confidence intervals, A/A testing and peeking. These outputs
remain ignored and are not selected public previews.

The separate `all-2026-11-11` directory contains only partial artifacts: three
PNGs, three module HTMLs and one GIF, without a completed run.json or index.
It is not a completed lab run.

A focused visual test execution on Python 3.11.14 reported 26 passed before
later source edits. At that stop state, full rendering, final tests and browser
checks were incomplete. Browser CLI attempts had failed on Windows cache-write
permissions, and Python 3.13 dependency installation had encountered cache
rename failures. No precise peak usage or dominant process was measured in the
interrupted session. The recovery above used serial work, cache-free dependency
installation and new output directories; it does not rewrite these earlier runs.
