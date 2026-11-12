# Day 47 visual lab: model, policy and evidence

This local educational lab uses synthetic independent users. It is separate
from the [historical textual experiment](notes.md#executed-synthetic-study--2026-11-12).
Its parameters, bounded costs and trajectories differ; sharing seed 47 does not
make the datasets identical. **Interpretation status: author review required.**

## Model contracts

Each arm receives exactly $n$ unique assigned users through a random permutation.
Every user submits one first request; all assigned users remain in conversion,
cost and request-safety denominators. Entry days are uniformly drawn from 0–27.
The observation snapshot ranges from calendar day 28 to 65, independently of
outcomes. Changing treatment parameters preserves control's random stream at
fixed sample size and seed. This is a reproducibility aid, not paired-arm data.

The main dataset is regenerated deterministically from controls. The sidebar
conversion and unsafe probabilities are **conditional on a served response**:

$$
p_{\mathrm{conversion}}=p_{\mathrm{served}}p_{\mathrm{conversion}\mid\mathrm{served}},
\qquad p_{\mathrm{incident}}=p_{\mathrm{served}}p_{\mathrm{incident}\mid\mathrm{served}}.
$$

Conversion is a synthetic completion event, not an independently verified
measure of answer correctness. No actual RAG retrieval, model call or quality
evaluator runs. All first-attempt end-to-end durations are fully observed
lognormal draws, including failures. No request timeout censoring, retries or
TTFT is simulated. Percentiles use the inverse empirical CDF (`inverted_cdf`),
while the 3D scene uses analytical population quantiles.

Per-request variable cost is independent Uniform$[0.5m,1.5m]$, with configured
mean $m$. Revenue is $v$ times conversion. The first-request cost is an
illustrative all-in variable budget, not a vendor token tariff; all requests,
including failures, incur cost. Fixed/shared costs and refunds are excluded.
Contribution per assigned user is revenue minus attributable variable cost,
not full profit. Cost per conversion is unavailable with no conversions.

## Coherent activity, churn and censoring

For configured marginal D7 activity $r_7$, define daily activity propensity
$a=(1+r_7)/2$ and engagement exit hazard $h=-\log(r_7/a)/7$. Draw a shared
exponential exit time per user, then sample daily activity with probability
$a$ only before that exit. Entry activity at D0 is fixed to true. Thus
$P(A_7)=a e^{-7h}=r_7$ and activity cannot resume after engagement exit, although
a user may skip a day and return before exiting. Other daily probabilities
follow this shared process; they are not independent arbitrary parameters.

Exact-day activity at $d$ means activity in $[d,d+1)$ and requires at least
$d+1$ elapsed observation days. Heatmap cells use mature members of each entry
week and remain blank if no member is eligible. The displayed count table
exposes changing denominators. Cohorts can differ by chance; the model contains
no calendar trend or informative observation timing.

Finite rolling return means any activity from day $d$ through D30. It is
monotone across $d$, uses only users with a complete 31-day horizon, and is
not truly unbounded return probability. Unbounded future return cannot be
observed with a finite window.

Baseline subscription membership is Bernoulli(0.6). Subscription cancellation
is an independent exponential time with a configured constant hazard. Free
use after cancellation is permitted, so activity and subscription survival
are different quantities. The subscription curve uses Kaplan–Meier on observed
cancellation or administrative censor time, assuming independent censoring.
It does not extrapolate beyond follow-up support or provide survival intervals.
Only matched subscription survival and cumulative subscription churn sum to one.

The "members remaining" area explicitly displays the simulator's complete
future as an oracle illustration. The censor plot displays actual observed
events versus administrative censoring for the first 100 starting subscribers.
Neither is a prediction from production data.

## Uncertainty and the illustrative decision policy

For a binomial count $k$ in $n$ independent users, `binomial_interval` uses
two-sided Clopper–Pearson bounds. A risk-difference interval combines an arm
interval at confidence $1-\alpha/2$ for each arm:

$$
[L_T-U_C,\ U_T-L_C].
$$

By a union bound its coverage is at least $1-\alpha$. It is deliberately
conservative, handles zero/all-success arms and avoids rare-event Wald collapse.
For example, zero events in both arms still leave uncertainty about their
population difference. It is not a posterior probability interval.

The policy divides its family error probability equally among five statistical
checks. No independence between checks is required for this Bonferroni bound.
Each check uses the relevant adjusted confidence; table bounds therefore differ
from the ordinary 95% exploratory conversion chart.

| Check | PASS | FAIL | Otherwise |
|---|---|---|---|
| Primary lift | Lower bound exceeds practical minimum | Upper bound is at or below minimum | INCONCLUSIVE |
| p95 latency maximum $L$ | Upper bound on $P(T>L)$ is at most 0.05 | Lower bound exceeds 0.05 | INCONCLUSIVE |
| Cost per conversion | Upper ratio bound at most maximum | Lower ratio bound exceeds maximum | INCONCLUSIVE |
| Error rate | Upper bound at most maximum | Lower bound exceeds maximum | INCONCLUSIVE |
| Safety degradation | Upper bound on treatment-control incidence at most tolerance | Lower bound exceeds tolerance | INCONCLUSIVE |

The lognormal duration population is continuous, so a p95 maximum can be tested
through exceedance probability rather than an interpolated sample percentile.
The cost ratio bounds combine exact conversion bounds and a Hoeffding interval
for mean cost with another equal error-budget split. For known support $[a,b]$:

$$
\epsilon=(b-a)\sqrt{\frac{\log(2/\alpha)}{2n}},\qquad
\mathrm{Cost/Conversion}\in
\left[\frac{L_{\mathrm{cost}}}{U_p},\frac{U_{\mathrm{cost}}}{L_p}\right].
$$

If $L_p=0$, the upper ratio bound is infinite. Bounds use the **known generating
support**, never the observed min/max. The cost model and independent-user
assumptions are essential; these bounds must not be applied to the historical
Gamma-cost example or copied into an unbounded production cost pipeline.

Safety uses detected incidents per all first requests. The synthetic detector
has configurable sensitivity and specificity fixed at one. Oracle latent
counts are identified separately. With imperfect detection the latent-safety
check remains INCONCLUSIVE even if detected-incidence bounds look favorable.
No classifier, judge or human assessment is represented as measured truth.
The configured critical share categorizes detected incidents; a detected
critical treatment incident or an explicitly injected confirmed incident causes
an immediate FAIL. Absence of a detected critical incident is not a safety proof.

Zero incidents in the separate exercise produce the exact one-sided bound
$1-\alpha^{1/n}$ and approximate $-\log(\alpha)/n$ (about $3/n$ at 95%). The
conditional zero-count exercise is distinguished from the realized binomial
sample and the main arm data. Confidence levels are configurable.

Any FAIL yields "Candidate violates one or more guardrails" (including the
primary policy requirement); otherwise an INCONCLUSIVE yields insufficient
evidence. Only all PASS yields the simulated-criteria recommendation. This
is a fixed-horizon policy demonstration. Changing controls, inspecting sample
paths repeatedly or selecting thresholds afterward does not establish valid
sequential inference or authorize production rollout.

## Multi-objective geometry and aggregation probes

The 3D scene displays 100 randomly generated population configurations plus
control and treatment. Conversion, unsafe incidence, mean cost and p95 are
known generating-model values. Feasibility checks the configured latency,
cost/conversion, error and relative safety constraints against population
values. It says nothing about how precisely a finite experiment estimates them.

The Pareto set is computed only within feasible configurations: maximize
conversion, minimize p95 duration, mean cost/request and unsafe/request. A point
is dominated only if another feasible point is at least as good on every
objective and strictly better on one. Identical ties survive. Point diameter
is $8+15(c-c_{\min})/(c_{\max}-c_{\min})$, with denominator one for an all-tied
set; this guarantees a positive mapping even for negative contribution.
The feasible-only filter retains the original size mapping and frontier.

The Simpson probe is a separate, deliberately constructed mixture with 1,000
users per group and worse treatment rates in both easy/hard segments. It does
not share the main experiment's randomization assumptions. The activity-volume
probe compares equal-user mean success with total successes/requests and shows
the integer counts. Neither treats repeated requests as independent users.

## Explicit GIF export and output policy

`animations.py` and section 09 generate only after an explicit invocation.
The latency GIF varies analytical lognormal sigma from 0.2 to 1.1 at fixed
treatment median; the uncertainty GIF shows a separate nested Bernoulli sample
with a fixed true marginal effect, 24 geometric sample sizes by default and
conservative pointwise intervals. Axes remain fixed across frames. Neither
animation claims monotone realized precision or sequential validity.

Each export uses a new UTC-stamped directory under ignored `outputs/`, with a
JSON manifest containing versions, configuration, sample-path results,
interpretation candidate and limitations. Old experiment records and exports
are preserved. No generated preview is selected or linked publicly.

## Validation record — 2026-11-12

Codex ran this phase locally on Windows using stable Python 3.11.14 in the
existing isolated environment `.local/environments/day46`. Versions: NumPy
2.4.6, pandas 3.1.1, SciPy 1.17.1, Streamlit 1.67.0, Plotly 6.9.0, Matplotlib
3.11.2, Pillow 12.3.0 and pytest 9.2.0. No dependency installation was needed.

**Hypothesis and configuration:** the default visual scenario uses seed 47,
3,000 users per arm, snapshot day 35, revenue 2, detection sensitivity 1 and
critical share 0. It is designed to show that conversion gains alone do not
resolve constraints and uncertainty. Arm values are the documented defaults
in `simulation.py`; these differ from the historical generator.

**Observed baseline:**

| Visual-model measure | Control | Treatment |
|---|---:|---:|
| Converted / assigned | 354/3,000 | 447/3,000 |
| Conversion | 11.8000% | 14.9000% |
| Exact D7 activity, all 3,000 mature | 36.7333% | 37.8667% |
| p95 duration (s) | 3.177444 | 6.120824 |
| Mean first-request cost | 0.039809 | 0.101025 |
| Cost per conversion | 0.337361 | 0.678023 |
| Contribution / assigned user | 0.196191 | 0.196975 |
| Detected unsafe incidents | 4 | 21 |

Observed lift was 3.10 percentage points; the ordinary conservative 95%
difference interval was approximately [0.29, 5.91] pp. Under the default
family-adjusted policy, primary lift was INCONCLUSIVE (approximately
[-0.4164, 6.6045] pp versus a 0.5 pp practical minimum). The latency exceedance
guardrail was FAIL; cost/conversion was PASS; request errors and safety
degradation were INCONCLUSIVE. No critical incident was detected. The final
illustrative recommendation was "Candidate violates one or more guardrails."

**Interpretation candidate — author review required:** the constructed example
shows why a favorable ordinary conversion interval can coexist with unresolved
primary-policy evidence and unacceptable latency. Serving contribution was
nearly unchanged under the fictional attribution rule; no business conclusion
or real safety certification follows. These are single-seed observations, not
power estimates or production effects.

**Commands and checks actually executed:** all Python commands used the stable
environment above; replace its executable prefix with `python` after activating
an equivalent environment.

```powershell
python -m pytest -q 03-statistics-experimentation/47-product-metrics-guardrails/tests
python 03-statistics-experimentation/47-product-metrics-guardrails/example.py --users-per-arm 3000 --seed 47
python 03-statistics-experimentation/47-product-metrics-guardrails/animations.py --users-per-arm 3000 --seed 47 --frames 24
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
python -m pip check
git diff --check
```

- Final focused tests: **90 passed**, no skips or pytest warning summary. They
  cover the original study, numerical contracts, all nine AppTest sections,
  controls/reset, hard stops, imperfect detection, explicit export behavior and
  six-frame temporary GIF renders.
- Original textual example rerun successfully retained its historical numeric
  results; its old JSON record was not overwritten.
- Repository syntax: 320 public Python files parsed. Internal paths: 1,034
  links checked in 217 Markdown files. Dependency compatibility passed.
  `git diff --check` passed, with Git's LF-to-CRLF notice for the module index.
- The loopback Streamlit agent API completed 12 transitions with `ready`
  status: nine sections, input changes, reset and the initial rerun. No runtime
  exception was reported. This is separate from AppTest and browser rendering.
- Chrome, driven by the existing local Playwright installation, passed 14
  browser checks: nine-section rendering, sample-size changes/reset, both Play
  animations, real 3D mouse rotation and zoom, feasible-only filtering and the
  critical-incident scenario. Zero JavaScript/console errors and zero external
  HTTP(S) requests. One Canvas2D readback performance advisory was recorded.
  The CLI daemon encountered sandbox/cache limitations; the same installed
  Playwright library completed the audit without new packages.
- Default GIF export created 24 distinct frames per GIF at 900 × 500 pixels:
  latency 639,969 bytes; uncertainty 684,994 bytes. Sampled frames were visually
  inspected. The run is dated 2026-11-12 08:34 UTC in its ignored manifest;
  no generated previews were made public.

**Limitations:** these checks do not establish full-suite CI, cross-browser
behavior, static type-checker coverage, GitHub rendering, model quality,
production readiness or personal author interpretation review. Retention curves
are descriptive and KM assumes independent censoring; safety depends on the
stated detector. The decision policy is conservative and specific to the known
synthetic cost support. The earlier textual experiment record remains intact.

## Primary references

- [SciPy binomial test and exact proportion intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).
- [Hoeffding (1963), Probability Inequalities for Sums of Bounded Random Variables](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf), the independent bounded-variable inequality used for serving cost.
- [Plotly animation documentation](https://plotly.com/python/animations/), including fixed axis ranges and animation controls.
- [Historical metric contracts and references](notes.md), including experimentation taxonomy, tail monitoring and safety limitations.
