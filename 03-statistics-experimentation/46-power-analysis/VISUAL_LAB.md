# Statistical power lab — visual guide

The lab asks whether a proposed experiment can answer its business question.
It uses **hypothetical population parameters**, **calculated approximations**
and **synthetic experimental estimates** with distinct labels. All views and
exports run locally, without accounts, APIs or production data.

## Run and reproduce

From this topic directory, with a stable Python 3.11+ environment activated:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
python generate_visuals.py
python -m pytest -q tests
```

The requirements file points to the repository's canonical
`.[dev,experimentation]` extra. Streamlit, NumPy, SciPy, pandas, Matplotlib,
Plotly and Pillow come from the root contract; statsmodels comes from the
experimentation extra. The local `.streamlit/config.toml` supplies the theme,
binds to `127.0.0.1` and disables usage statistics.

For another bounded export configuration:

```powershell
python generate_visuals.py --baseline 0.10 --effect 0.005 --alpha 0.05 --target 0.90 --seed 46 --frames 36 --repetitions 20000 --output-dir outputs/smaller-effect
```

This is a **suggested additional run**, not an observed result in this guide.
`--effect` is a probability difference (0.01 = one percentage point); the
dashboard effect controls use percentage points. Export supports 3–80 frames
and up to 100,000 simulation repetitions. Defaults are 36 frames, 20,000
repetitions and seed 46. Each GIF has 1000×580 frames; the PNG is 1700×986.

## Explore eight learning questions

| Module | What the interaction reveals | Boundary |
|---|---|---|
| Hypothesis distributions | Null/alternative curves, two-sided cutoffs, separate alpha, beta and power; move n to narrow uncertainty | The expected pooled null SE is frozen for planning; curve overlap is not beta |
| Power and sample size | Nonlinear power, 80%/90% references, selected n and target crossing; compare closed form and numerical solution | Both are normal approximations, despite integer output |
| 3D power surface | Rotate/zoom/hover over n, positive effect and power; a target plane and contour mark adequate sensitivity | Positive effects only; treatment probabilities stay strictly inside (0,1) |
| Effect size sensitivity | Log sample-size curve, inverse-square reference, relative uplift, business threshold and fixed-n MDE | Fixed-variance scaling is a reference, not an identity for Bernoulli rates |
| Monte Carlo simulation | Run H0 or H1 explicitly; view estimated differences, p-values, rejection rate and a 95% Monte Carlo interval | Simulation error and model uncertainty are different; snapshots retain their own inputs |
| Correlated observations | Clustered conversations, ICC, design effect and equivalent independent information | Equal-cluster heuristic; no cluster-randomized power guarantee |
| Business feasibility | Assigned versus mature samples, deadline, finish day, power and positive MDE at available n | Constant expected distinct traffic, complete eventual outcomes, 50/50 arms |
| Statistical vs. practical significance | Seeded effect estimates and intervals across n, zero and a separate business threshold | No p-value selection, equivalence claim or automatic deployment decision |

Every view provides four short readings: **what you are seeing**, **what
changes**, **mathematical interpretation** and **production implication**.
Analytical views rerun with controls; only the selected module computes.
Monte Carlo uses a bounded vectorized binomial generator and an explicit form
submission. It never runs just because a global slider changes. Result
configuration and a stale-input warning prevent relabeling an old run.

## Statistical model

Let $p_A$ and $p_B=p_A+\Delta$ be the population probabilities and $n$ the
independent sample size per arm. Write $\bar p=(p_A+p_B)/2$:

$$
SE_0=\sqrt{\frac{2\bar p(1-\bar p)}{n}},\qquad
SE_1=\sqrt{\frac{p_A(1-p_A)+p_B(1-p_B)}{n}}.
$$

The plotted null is $N(0,SE_0^2)$ at the declared common planning rate $\bar p$;
the alternative is $N(\Delta,SE_1^2)$. With $c=z_{1-\alpha/2}SE_0$,

$$
\alpha=P_{H_0}(|\hat\Delta|>c),\quad
\beta=P_{H_1}(-c\leq\hat\Delta\leq c),\quad \pi=1-\beta.
$$

Alpha is integrated under H0; beta and power are integrated under the specified
H1, including both tails. Critical boundaries change with n. In
[power_math.py](power_math.py), statsmodels calculates the same pooled-null,
separate-alternative approximation. The numerical solver inverts its two-tail
power; [from_scratch.py](from_scratch.py) retains the dominant-tail closed form
derived in [notes.md](notes.md). Sample sizes are rounded up and target power
is rechecked. Approximation agreement does not imply exact finite-sample power.

Simulations in [simulation.py](simulation.py) draw independent binomial counts
and use the **actual observed pooled SE** in the two-sided Z-test. This SE is
random, whereas the analytical approximation freezes it at a planning value.
Discreteness and this approximation can explain systematic differences beyond
Monte Carlo uncertainty. All-zero/all-one pooled tables receive p=1 and are
counted explicitly; this convention does not make sparse normal inference
valid. Expected success/failure counts below ten trigger a dashboard warning.
The heuristic threshold is a diagnostic, not a theorem certifying adequacy.

The rejection-rate interval uses statsmodels' 95% Wilson method and quantifies
Monte Carlo error. Interval examples for individual synthetic experiments use
an **unpooled Wald interval**; it is not an exact inverse of the pooled test and
can behave poorly at low counts. The color indicates whether that interval
excludes zero, while the table also reports the pooled test's p-value.

The app exposes the same two-sided test throughout. One-sided alternatives
would require different critical regions and planning quantiles; they are not
silently substituted for the selected two-sided design.

For equal conversation clusters of size $m$ and ICC $\rho$,

$$
DE=1+(m-1)\rho,\qquad N_{eff}\approx N_{raw}/DE.
$$

This is information about conversation means under that heuristic, not the
number of randomized users to insert into an independent-proportion calculator.
Unequal cluster sizes, few clusters and the chosen estimator require their
own planning model. The dot diagram shows a bounded schematic subset, not
simulated correlated Bernoulli outcomes.

The business view applies eligibility once to daily **potential** new users,
then applies the share admitted to the experiment, then splits arms equally.
It shifts the assigned curve by the maturation window rather than removing
nonresponders. The positive MDE solves for an improvement at available mature
n; a user-supplied required n can differ from the statistical requirement and
is explicitly labeled. The finish day uses expected traffic, without a
weekday floor; the original analytical script separately demonstrates a
minimum-enrollment constraint.

## Artifacts and publication policy

Default `outputs/` is ignored and contains:

```text
sampling_distributions.gif
power_growth.gif
effective_sample_size.gif
effect_size_sample_size.png
power_surface_3d.html
visual_run_<UTC timestamp>.json
```

The HTML embeds Plotly's JavaScript, enables rotation, zoom and numerical hover,
and uses no CDN. Open it directly in a browser, or serve it from localhost.
It remains a local regenerable artifact (about 5 MB), so public documentation
does not link to an ignored file. A unique JSON record preserves the run's
hypothesis, parameters, versions, summary, interpretation candidate and limits.

The three GIFs and PNG under [assets](assets/) are inspected, deliberately
unignored previews. They reproduce the documented default configuration,
independently of current dashboard controls. Regenerate into `outputs/`,
inspect first/middle/final frames, and deliberately replace selected previews
only after review. Tests render into temporary directories. They do not
overwrite previews or historical run records.

```text
46-power-analysis/
├── app.py                    # Streamlit entry point
├── power_math.py             # Planning, MDE, clustering and traffic
├── simulation.py             # Seeded binomial experiments and inference
├── visual_charts.py          # Shared interactive Plotly figures
├── generate_visuals.py       # Bounded, noninteractive exports
├── example.py / from_scratch.py
├── README.md / notes.md / VISUAL_LAB.md
├── requirements.txt          # Delegates to canonical root extras
├── .streamlit/config.toml    # Topic-local theme and loopback configuration
├── tests/                   # Five numerical, simulation, app and export files
├── assets/                  # Three selected GIFs and one PNG
└── outputs/                 # Ignored HTML, regenerated exports and dated runs
```

## Executed visual study — 2026-11-11

This record is separate from the earlier analytical study. Date is in
America/Sao_Paulo; its generated record has UTC timestamp
`20261112T013928879517Z`.

**Hypothesis:** at the base alternative, repeated pooled-Z rejection frequency
should be close to planned power; under H0, it should be close to nominal alpha,
allowing simulation uncertainty and finite-sample approximation error.

**Configuration:** $p_A=0.10$, $p_B=0.11$, two-sided $\alpha=0.05$, target
0.80, numerically calculated **14,751 users/arm**, 20,000 replications per mode,
seed 46. NumPy PCG64 generator; H0 uses both rates 0.10. The same seed is reset
per mode, so these are reproducible comparisons, not independent estimates
of a difference between the two rejection probabilities.

**Versions:** Windows, Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1,
statsmodels 0.15.0, pandas 3.1.1, Plotly 6.9.0, Matplotlib 3.11.2,
Pillow 12.3.0, Streamlit 1.67.0, pytest 9.2.0. The environment is isolated at
`.local/environments/day46`; no historical environment was upgraded.

| Mode | Rejections / 20,000 | Empirical rate | 95% Wilson Monte Carlo interval | Analytical approximation |
|---|---:|---:|---:|---:|
| H0 | 986 | 4.930% | 4.639%–5.239% | 5.000% |
| H1 | 15,994 | 79.970% | 79.410%–80.519% | 80.001% |

No degenerate pooled tables occurred. Three 36-frame GIFs, the PNG and the
self-contained HTML were generated. GIF first/middle/final frames and the PNG
were visually inspected; the selected previews are public Git candidates.

**Interpretation candidate — author review required:** these synthetic
rejection frequencies are compatible with their analytical approximations at
the base scenario within Monte Carlo uncertainty. This supports that limited
calibration example, not every probability regime or business experiment.

**Limitations:** all inputs are hypothetical; observations are independent,
the horizon is fixed and there is one comparison. No dependence, attrition,
interference, production gain, safety compliance or return on investment was
measured. Extra simulation draws do not resolve uncertainty in those assumptions.

## Validation completion — 2026-11-12

The final focused command `python -m pytest -q tests` passed **90 tests**, with
no skips or pytest warnings. It includes the original 40 analytical tests,
statsmodels agreement, null/alternative Monte Carlo checks, all eight Streamlit
views and reset, retained simulation configuration, and temporary-directory
exports. Syntax checks parsed **310 public Python files** and internal-link
checks verified **1,005 links across 213 Markdown files**. Dependency consistency
and `git diff --check` passed. Git emitted only LF/CRLF normalization notices.

A real headless Chrome session driven by Playwright completed **13 interaction
checks** across all eight modules: sample slider changes, reset, actual mouse
rotation of the 3D surface, both Monte Carlo modes, ICC, outcome maturation,
business threshold and invalid-probability handling. The final audit captured
zero JavaScript errors and zero external HTTP requests. Chrome emitted a
Canvas2D readback-performance advisory during Plotly rendering; it did not
prevent rendering or interaction.

The exported HTML was separately served from loopback HTTP and checked for
actual mouse rotation, scroll zoom and numerical hover, again with zero errors
or external requests in the final run. The automation tool blocks direct
`file:` navigation, so that path was not browser-verified. The file embeds its
JavaScript and favicon; serving it requires no external resources.

All 36 frames of each selected GIF were decoded, the PNG passed its integrity
check, and first/middle/final frames plus rendered dashboard/HTML screenshots
were inspected. Local browser evidence and intermediate screenshots remain
ignored. This is Windows/Chrome and focused local evidence, not cross-browser
validation, Linux CI, GitHub rendering or personal author approval.

## Manual experiments to try — proposed

1. Halve the effect from +1 to +0.5 pp, keeping alpha and target power fixed.
   Compare the requirement with the inverse-square reference and inspect the
   3D target crossing. Bernoulli variance prevents an exact fourfold identity.
2. At the default sample, run H0 and H1 with 1,000 then 20,000 repetitions.
   Compare Wilson interval widths without treating simulation repetitions as
   additional experiment participants.
3. Keep 1,000 users and ten conversations/user. Move ICC from zero to one;
   explain why raw count stays 10,000 while equivalent independent information
   approaches 1,000.
4. Hold the +1 pp business effect and a 14-day deadline fixed. Change eligible
   traffic admission and outcome delay; compare assigned and mature samples,
   achieved power and MDE without changing the original effect silently.
5. Set +0.2 pp as the population effect and +0.5 pp as the worthwhile threshold.
   Inspect synthetic intervals across n. Then move only the business threshold
   and explain why statistical evidence and utility answer different questions.

Naive repeated peeking and valid sequential methods are not implemented in
this lab. Cluster-aware, paired, rare-event and multiplicity-adjusted planning
also remain extensions. A nonsignificant result never establishes equivalence.

Primary references: [statsmodels two-proportion power](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.power_proportions_2indep.html),
[pooled proportions Z-test](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html)
and [binomial intervals](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html).
