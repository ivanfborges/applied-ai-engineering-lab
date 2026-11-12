# DiD visual learning laboratory

This is the separately authorized visual extension of Day 51. Every outcome
is a synthetic productivity score, with higher values meaning higher
productivity. No real assistant, team outcome, ticket dataset or production
guardrail is measured. **Interpretations require author review.**

## Start locally

From the repository root, activate a Python 3.11+ environment, then:

```powershell
cd 03-statistics-experimentation/51-difference-in-differences
pip install -r requirements.txt
streamlit run app.py
```

The topic's Streamlit config uses a native light theme, system sans-serif
fonts, loopback binding and disabled usage telemetry. No runtime download,
API, credentials or cloud account is needed. Installation may need package
access if the canonical dependencies are not already installed. For execution
from the root, use `python -m streamlit run
03-statistics-experimentation/51-difference-in-differences/app.py
--server.address=127.0.0.1 --browser.gatherUsageStats=false` on one line.

The requirements file is a compatibility entry point to `../..[experimentation]`;
all bounds remain in [pyproject.toml](../../pyproject.toml). The lab was built
against the installed Streamlit 1.67 API, including lazy tabs and chart alt text.
Version bounds are not a frozen environment; the validation record lists actual
versions. The original [analytical run](notes.md#executed-synthetic-study--2026-11-12)
uses a different panel and remains unchanged.

## A path through the seven chapters

| Tab | Visual question | Experiment/control | Reading boundary |
|---|---|---|---|
| Counterfactual Intuition | Which treated post-outcome is missing? | Change effect, group baseline gap and noise; reveal purple synthetic truth | Dashed gray is estimated, purple is oracle-only truth; the selected green gap is period-specific |
| DiD Estimator | How do two changes become one contrast? | Four editable means; six-stage slider and Plotly play button; defaults 100→125 and 100→110 yield 15 | Illustrative cells are separate from the panel controls; arithmetic alone is not causal identification |
| Parallel Trends | How can a correct calculation miss the true effect? | Slope -2..2; true effect stays fixed | Error is decomposed into deterministic identification bias and realized measurement noise using synthetic truth |
| Regression & Uncertainty | What does a coefficient interval describe? | Statsmodels TWFE; shared noise control; optional slope and on-demand 100/200/500 independent panels | Coefficient CI and sampling histogram are different objects; neither validates parallel trends |
| Event Study | Do pre-period contrasts raise design concerns? | Trend/anticipation controls, pointwise intervals and pre-only placebo | Reference -1 is normalized; small pretests do not prove parallel trends; intervals are not simultaneous |
| When DiD Fails | Which assumption was broken? | Separate differential trend, control spillover and anticipation scenarios | Each scenario states a redesign; post ATT retains the declared no-rollout comparison |
| 3D Exploration | Where does the estimated gap sit in time/group/outcome space? | Rotate, zoom, hover, toggle traces, choose a post period, reveal oracle truth | Group is categorical; 2D is the primary teaching view, and 3D adds no identifying information |

The Event Study expander adds a conceptual early/late/never adoption heatmap.
It explains already-treated comparisons when effects evolve; no questionable
group-time ATT approximation or universal claim against TWFE is implemented.

## Transparent generating model

Seed 51 generates 120 independent team clusters, first 60 adopting at period 0
and 60 never adopting, with every team observed at periods -6 through 5.
There are six pre-periods and six post-periods. Treatment group membership is
fixed; `exposed = treated × post` marks actual own-treatment exposure.

For the no-rollout world:

$$
Y_{it}(0)=50+b_i+gD_i+\lambda_t+sD_it+\sigma\epsilon_{it},
\qquad b_i\sim N(0,3^2).
$$

Here $g$ is the group baseline offset (default 5), $s$ is the differential
untreated slope (default 0), and $\lambda_t=0.6t+u_t$, with shared
$u_t\sim N(0,0.3^2)$. Noise is stationary AR(1) with coefficient 0.6,
unit marginal standard deviation and independent innovations across teams;
$\sigma$ defaults to 1.5. Changes to effect, gap or noise reuse the same
standardized random draws. Baseline untreated trends are parallel in expectation,
not identical noisy sample paths.

The observed outcome extends this world with three disclosed mechanisms:

$$
Y_{it}=Y_{it}(0)+D_i[\tau 1(t\ge0)+a1(t\in\{-2,-1\})]
+(1-D_i)f\tau1(t\ge0).
$$

Default $\tau=4$ points. Spillover fraction $f$ gives controls a fraction of the
effect after rollout (default 0). Anticipatory gain $a$ affects treated teams at
-2 and -1 (default 0); it does not increase the post-treatment effect.
The post ATT compares treated rollout outcomes with the no-rollout world and
remains $\tau$. Under spillovers, outcomes require exposure definitions beyond
own treatment; controls' observed outcomes are not their no-rollout $Y(0)$.
`y0`, `y1`, `mean_y0` and the mechanism-specific contributions are retained for
teaching. None is an observed counterfactual in a real treated team.

## Counterfactual, bias and sampling noise

The estimated treated no-rollout path is the observed control trajectory plus
the **average pre-period group gap**. Its average post-period vertical gap
equals pooled DiD. This differs from anchoring only to the last pre-period:
the event study deliberately uses that different reference and target.

For the fixed windows, the analytical expected contamination is

$$
E[\widehat{DiD}]-ATT=6s-f\tau-\frac{a}{3}.
$$

Mean post time is 2.5 and mean pre time is -3.5. Anticipation affects two of six
pre-periods. With no measurement noise, these shifts are exact; random team
level differences and common time shocks cancel. With positive noise, a single
estimate's error also includes a noise contrast. The dashboard computes this
decomposition from the known generating components, not by estimating bias from
observational data. Negative treatment effects reverse the direction of control
spillovers; a zero effect produces no spillover contribution.

## Inference and responsiveness

The regression tab runs actual statsmodels OLS with unit/time indicators and
`treated:post`, clustered by team, CRV1 finite-sample correction and a Student
$t(119)$ reference. Manual and TWFE point estimates agree in this balanced
single-date design, including when the causal assumptions are violated.

Fast repeated-sampling intervals use unit-level pre/post changes and the exact
balanced-design CRV1 algebra. They retain the full regression parameter count
$K=G+T$ in the correction $G/(G-1)\times(N-1)/(N-K)$; they are tested against the
full statsmodels regression. This avoids hundreds of redundant dummy regressions.
The event study similarly uses exact saturated group-time contrasts relative
to -1, with $K=G+2T-2$ and team-clustered covariance retaining each team's shared
reference outcome. Tests compare every coefficient and interval with a full
saturated statsmodels fit. Reference -1 has no estimated interval.

The placebo uses only -6..-1, comparing -6..-4 with -3..-1 as if adoption began
at -3. No actual post-outcome is read. Anticipation may contaminate a pre-only
placebo despite correct date filtering. Isolated pointwise pre-intervals can
exclude zero by chance; the displayed count is not an omnibus test.

Only the active tab executes. Full regressions and sampling results have
bounded in-memory caches; simulation batches run only on button press. A changed
configuration hides saved sampling results until rerun. Independent panels use
`SeedSequence(5100).spawn(draws)` and report empirical SD, MCSE for the mean and
the fraction of intervals covering known truth. That fraction is a Monte Carlo
measurement, not a guarantee. One panel's CI does not describe the histogram.

## GIFs and publication boundaries

```powershell
python export_gifs.py
```

Default export: 48 frames, 6 fps, 960×600 pixels, Pillow writer, fixed axes and
consistent labels/colors. GIFs have no interactive runtime. The generator
protects existing exports; use a new `--out` directory or explicit `--overwrite`.

- `assets/did_counterfactual_animation.gif`: pre-outcomes → boundary → control
  evolution → estimated counterfactual → treated outcomes → estimated gap.
- `assets/parallel_trends_violation.gif`: slope 0→1.5, fixed ATT +4 and zero
  noise, so the growing error isolates identification bias.
- `assets/visual_run.json`: reproducible configuration, versions, numerical
  outputs, hypotheses and pending interpretation status.

The violation GIF's red segment is the counterfactual error at period 5;
the footer reports the pooled-window DiD error. These are distinct summaries.
Pillow can coalesce identical adjacent story frames while retaining their
display durations; the configured frame count is not necessarily the encoded count.

These are **ignored regenerable candidates**, not selected public previews.
No ignored asset is linked or embedded in public Markdown. The author selects
previews and root README publication updates separately. Source code and the
topic guide are reviewable without exposing private study or LinkedIn material.

## Visual-phase execution record — 2026-11-12

This new configuration is separate from the historical analytical run.
Interactive slider positions not listed here are explorations, not reported
experiments. Execution by Codex on Windows used Python 3.11.14, NumPy 2.4.6,
pandas 3.1.1, SciPy 1.17.1, statsmodels 0.15.0, Streamlit 1.67.0, Plotly 6.9.0,
Matplotlib 3.11.2 and Pillow 12.3.0; pytest 9.2.0 ran the tests.

Hypotheses: manual and full TWFE agree; differential untreated slope adds six
times that slope to pooled DiD; independently simulated valid panels center
near the configured ATT, while violating trends shifts that center.

The default seed-51 visual panel gave manual DiD and TWFE **4.396517** points,
team-cluster SE **0.254175** and 95% CI **[3.893224, 4.899809]** for true ATT 4.
The genuine-pre-only placebo was **-0.391474**, CI **[-0.981397, 0.198450]**.
One of five non-reference pre-period pointwise intervals excluded zero despite
the valid generating model. This is not an omnibus rejection or evidence that
the constructed post-treatment counterfactual assumption failed.

A separate repeated study ran 200 panels per scenario using root seed 5100,
with default effect 4, noise SD 1.5, baseline gap 5 and 120 teams. Datasets
are independent within each scenario; the two scenarios share seed streams to
isolate the added slope. Intervals use the same verified CRV1 algebra as the
full regression.

| Scenario | Slope | Mean DiD | Empirical SD | MCSE of mean | Intervals covering true ATT |
|---|---:|---:|---:|---:|---:|
| Parallel | 0 | 3.991450 | 0.236032 | 0.016690 | 192/200 (0.960; plug-in MCSE 0.013856) |
| Nonparallel | 0.8 | 8.791450 | 0.236032 | 0.016690 | 0/200 |

Zero observed coverage in 200 draws is not proof of exactly zero coverage
probability. The plug-in binomial MCSE degenerates to zero at an observed
fraction of zero or one, so it is not a useful uncertainty statement there.

Noise-free live-runtime controls also gave DiD **8.8** for slope 0.8, **2.0**
for spillover fraction 0.5, and **3.0** for anticipation gain 3. The post ATT
remained 4; only one violation was enabled in each case.

**Interpretation candidates — author review required:** the valid repeated
study is consistent with its constructed ATT, within Monte Carlo variation.
The paired slope shifts the estimate center by 4.8 without increasing its
empirical SD. Clustered uncertainty describes sampling variation around the
estimated contrast; it does not protect against that constructed identification
bias. No causal claim about a real assistant follows from these simulations.

Sampling configurations, raw numerical summaries and generated GIF manifests
are retained as ignored local records. New runs should use a fresh export
directory; preserve this dated record rather than silently replacing its values.

### Validation scope

The local validation covers numerical contracts, regression/covariance
agreement, temporary-directory GIF exports, Streamlit widget behavior and the
running application. The three isolated test files passed **36 + 28 + 2 = 66
tests**, without skips; the figure-encoding test also passed after adjusting
the initial 3D framing. Repository parsing covered **348 Python files** and
the link check covered **1,140 targets in 232 Markdown files**. Dependency
compatibility (`pip check`), `git diff --check` and the topic requirements
entry point's offline dry run (`--no-deps --no-build-isolation --no-index`)
passed. That dry run did not install or upgrade any dependency.

Both GIFs were generated at 960×600 and inspected at multiple stages. The story
GIF encoded 16 coalesced frames (232,270 bytes); the trend-violation GIF encoded
48 frames (951,734 bytes). Both run for approximately eight seconds. Their
ignored manifests and inspection sheets are local evidence, not public previews.

Browser inspection used Chrome 157.0.8103.51 through a
dedicated loopback CDP session after the Playwright CLI encountered a cache
permission restriction. The completed browser pass exercised all seven tabs,
the counterfactual reveal/effect control, all six animation frames, the slope
warning, 3D WebGL rotation and its time selector; it observed zero JavaScript
errors, failed requests or external requests. AppTest and live-agent checks
cover logic independently of the browser. Other browser engines and CI were
not run; this is local verification, not author interpretation approval.

During automated session cleanup, the Windows Python runtime logged an
`asyncio` pipe callback `ConnectionResetError` (`WinError 10054`). A subsequent
health request returned `ok` and a new live-agent session returned `ready`.
The stack contained runtime transport frames; the underlying disconnect
callback was not changed by this study. Browser JavaScript errors and app
execution exceptions were checked separately from that server-log observation.

## Limitations

Complete balanced panels, one common adoption date, constant own post-effect,
independent team clusters, fixed definitions and no missingness are intentional
simplifications. No clustered small-sample remedy, staggered ATT estimator,
sensitivity bound, simultaneous event band or selective-attrition correction is
implemented. Production exposure, instrumentation and assignment must be
audited separately. Neither a visual match nor a narrow interval validates the
untreated trend assumption. See [primary references](references.md) and
[technical notes](notes.md) for the methodological boundaries.
