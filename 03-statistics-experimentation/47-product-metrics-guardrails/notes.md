# Measurement contracts before product decisions

## Define the decision population

A metric name is insufficient. Record its decision role, population, analysis
unit, numerator, denominator, observation window, event source, direction,
missingness rule and important segments. Specify who owns the contract and
what changes require a new metric version.

For an experiment, begin with an assignment ledger whose eligibility is set
before treatment. Joining only successful response events can remove precisely
the failures caused by treatment. An exposed-only analysis needs its own causal
justification; it is not automatically intention-to-treat (ITT).
This example has one eligible request for every assigned user and no exclusions,
so its conversion denominator agrees with the synthetic assignment population.

The four metric roles follow the
[Microsoft experimentation taxonomy](https://www.microsoft.com/en-us/research/articles/patterns-of-trustworthy-experimentation-during-experiment-stage/):
primary outcome, guardrail, diagnostic and data quality. Which role a measure
serves depends on the intervention. Latency can be primary for an optimization
initiative and a constraint for a model-quality initiative. SRM belongs to
data integrity, rather than evidence of business benefit.

## Implemented contracts

All times are elapsed days from assignment. `followup_days` is a synthetic
administrative observation horizon, drawn independently of outcomes; it must
not be replaced by "time until the user stopped returning."

| Field / measure | Window, eligibility and missingness |
|---|---|
| `converted` | At least one target completion in `[0, 1)`; all users have at least one observed day; requires `served_ok` |
| `returned_d7` | Activity in `[7, 8)`; observed only for `followup_days >= 8`; earlier cohorts are missing, not inactive |
| `subscriber_at_start` | Subscription membership before treatment |
| `churned_7d` | Cancellation in `[0, 7)` among starting subscribers with `followup_days >= 7`; all other values missing |
| `latency_s` | Fully observed end-to-end duration of the first attempt, including failures |
| `cost`, `revenue` | Attributed variable serving cost and completion revenue in `[0, 1)`; fictional monetary units |
| `unsafe` | Synthetic output-incident indicator, completely observed for every request; failed requests have no output incident |

The validator requires one row per user and both arms. It rejects duplicate IDs
instead of silently choosing a conflicting event. In a real event pipeline,
deduplicate by stable event IDs, resolve assignment conflicts, join events to
the full ledger, enforce event-time windows, reconcile billing and track late
arrivals against an explicit watermark. An absent event is not necessarily a
negative outcome. This example receives a prepared ledger; it does not build
that pipeline or discover missing assignments.

Mature-cohort restriction prevents incomplete-window counting but does not
solve calendar confounding, informative censoring or missing logging. Use
comparable entry cohorts and appropriate censoring methods when needed.

## Rates, windows and weighting

For $N$ assigned users and $C$ target completions,

$$
\widehat p_{\mathrm{conv}} = \frac{C}{N}.
$$

For the mature exact-day cohort $M_7 = \{i: a_i \geq 8\}$, with $a_i$ the
observed follow-up horizon and $A_{i,7}$ activity in $[7,8)$,

$$
\widehat R_7 = \frac{\sum_{i\in M_7} A_{i,7}}{|M_7|}.
$$

Returning on exact Day 7 differs from returning at least once before Day 7.
Seven-day subscription churn uses a different cohort and event:

$$
\widehat q_7 =
\frac{\sum_{i\in S_0,\ a_i\geq7} L_{i,[0,7)}}{|\{i\in S_0:a_i\geq7\}|},
$$

where $S_0$ is the starting subscriber population and $L$ denotes cancellation.
It is generally not $1-\widehat R_7$. Activity, cancellation, revenue churn and
net revenue retention answer different questions. Subscription loss may coexist
with continued free use. The generator intentionally leaves retention and
cancellation independent rather than claiming a behavioral model.

For repeated requests, a request-weighted success rate and an equally weighted
mean user rate differ:

$$
\frac{\sum_i s_i}{\sum_i n_i}
\quad\ne\quad
\frac{1}{U}\sum_i\frac{s_i}{n_i}.
$$

A user with 9 successes in 10 requests and another with 0 in 1 produce $9/11$
and $0.45$, respectively. These are worked arithmetic examples, not measured
experiments. Repeated requests also violate independent-request inference in
a user-randomized test. Cluster-aware uncertainty or user-level statistics
are needed; the executable study uses one request per user.

## Latency and failures

The empirical CDF is $F_n(t)=n^{-1}\sum_i\mathbf{1}(t_i\leq t)$.
The implementation sorts observations and returns
$x_{(\lceil qn\rceil)}$ for $0<q\leq1$, using the minimum at $q=0$.
This inverse-ECDF convention agrees with
[NumPy `method="inverted_cdf"`](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html),
rather than its default interpolated quantile.

The p95 is a distribution threshold, not the mean of the slowest 5% or a
statement that 95% of requests take exactly that duration. Pooled quantiles
cannot generally be recovered by averaging per-host quantiles.

Track failed-request duration separately when its semantics differ from
successful completion; also measure errors and timeouts. A deadline-censored
timeout is not a completed response at that deadline. This simulation has
fully observed durations and no retries or timeout censoring. Streaming TTFT
and full response time would require separate telemetry.
[Google SRE](https://sre.google/sre-book/monitoring-distributed-systems/)
explains why latency distributions and errors both matter.

## Economics with consistent denominators

For first-day revenue $r_i$ and attributed variable cost $c_i$,

$$
\mathrm{Revenue/User}=\frac{\sum_i r_i}{N},\qquad
\mathrm{Contribution/User}=\frac{\sum_i(r_i-c_i)}{N},\qquad
\mathrm{Cost/Conversion}=\frac{\sum_i c_i}{C}.
$$

The contribution measure is an absolute amount per user, not a percentage
margin or full profit. Include costs of failures and nonconverters. Zero
conversions make cost/conversion unavailable. The aggregate ratio is not the
mean of a ratio restricted to converting users.

Here $r_i=2\,\mathbf{1}(\mathrm{converted}_i)$, an illustrative value rather
than a price or validated attribution model. Costs are independent Gamma
draws for every first request, including failed attempts. Real cost accounting
may also require retrieval, retries, tool calls, human review, support and
shared infrastructure. Token charges use consistent units:
$c=T_{\mathrm{in}}P_{\mathrm{in}}+T_{\mathrm{out}}P_{\mathrm{out}}+c_{\mathrm{other}}$.
No vendor pricing is assumed.

Incremental contribution is the treatment-control difference in contribution
per assigned user. Incremental cost per incremental conversion is a different
ratio and becomes unstable near a zero conversion difference. Neither should
be inferred from a conversion p-value.

## Uncertainty and guardrails

For independent user-level Bernoulli conversion, the educational interval is

$$
\widehat\Delta=\widehat p_T-\widehat p_C,\qquad
SE=\sqrt{\frac{\widehat p_T(1-\widehat p_T)}{n_T}
+\frac{\widehat p_C(1-\widehat p_C)}{n_C}},\qquad
CI=\widehat\Delta\pm z_{1-\alpha/2}SE.
$$

Absolute lift is in fractions (multiply by 100 for percentage points);
relative lift is $\widehat\Delta/\widehat p_C$. The code requires at least five
successes and failures per arm as a minimal approximation check. This heuristic
does not guarantee adequate coverage. Wald intervals can behave poorly near
boundaries; they are not clipped, and they are not used for rare safety events.
See [NIST's interval discussion](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
and [Day 45](../45-ab-testing/) for related inference and randomization checks.

A treatment point estimate below a maximum does not establish that its
population value is below that maximum. A nonsignificant degradation is also
not proof of non-inferiority. For a tolerated increase $\delta$ in a harm rate,
a planned comparison might require $U_{1-\alpha}(p_T-p_C)<\delta$ with a valid
one-sided bound. Latency quantiles, cost ratios and correlated safety events
need methods appropriate to their sampling structure. They have no comparative
intervals in this example.

The printed maxima (5 seconds p95, 0.08 cost/request, 0.5% unsafe/request,
2% error) are fixed illustrative values. There is no combined ship gate.
Passing all point checks would still leave primary-outcome relevance, guardrail
uncertainty, data quality, maturity, segment risks and authorization unresolved.
Predefine the decision rule, practical effect and tolerable degradation before
observing data; plan multiplicity, power and monitoring accordingly. Severe
privacy or unauthorized-action incidents may require immediate hard stops
regardless of any favorable aggregate outcome.

## Rare safety events and detection

For zero incidents in $n$ independent, fully observed Bernoulli opportunities,
solve $(1-p_U)^n=\alpha$:

$$
p_U=1-\alpha^{1/n}\approx\frac{-\log\alpha}{n}.
$$

At 95% one-sided confidence the approximation is about $3/n$. The implementation
uses `-expm1(log(alpha)/n)` for numerical stability and is tested against
[SciPy's exact binomial interval](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).
It applies only to the explicitly separate zero-incident exercise, not to an
arm with nonzero observed incidents or to a between-arm difference.

An undetected incident is not a safe outcome. If only outputs are reviewed,
report incident/output and review coverage separately; do not silently reuse
all requests as the reviewed denominator. This example assumes complete,
perfect synthetic labeling and defines output incidents per request, including
failures as nonincidents. It does not measure unauthorized actions on failures.
Real evaluation needs risk-specific rubrics, severity, representative sampling,
adversarial coverage and human calibration.
[NIST AI RMF 1.0's MEASURE function](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)
provides context for documented evaluation and uncertainty; it does not prescribe
the illustrative thresholds here.

## Executed synthetic study — 2026-11-12

**Hypothesis:** a configuration with higher generated conversion probability
can also have worse latency, cost and safety incidence. This is a constructed
trade-off, not an attempt to discover real product behavior.

**Execution:** Codex ran the command below on Windows using stable Python
3.11.14, NumPy 2.4.6 and pandas 3.1.1. The run timestamp was
2026-11-12 08:13:29 UTC. SciPy 1.17.1 and pytest 9.2.0 were used for the focused
checks, rather than for generating these observations.

```bash
python 03-statistics-experimentation/47-product-metrics-guardrails/example.py --users-per-arm 3000 --seed 47 --output-dir 03-statistics-experimentation/47-product-metrics-guardrails/outputs
```

**Configuration:** seed 47; 3,000 users per arm; exactly balanced labels randomly
permuted across users; one first request each; follow-up uniformly drawn from
integer days 1–14; baseline subscription probability 0.6; 2 fictional monetary
units of revenue per conversion. Every request has Gamma cost with shape 4
and scale equal to arm mean cost divided by 4.

| Generating parameter, not measured evidence | Control | Treatment |
|---|---:|---:|
| Request served probability | 0.990 | 0.985 |
| Conversion conditional on served | 0.12 | 0.15 |
| Exact D7 activity probability | 0.38 | 0.39 |
| Seven-day cancellation probability | 0.06 | 0.07 |
| Unsafe output conditional on served | 0.002 | 0.008 |
| Lognormal median latency (seconds) | 1.4 | 2.2 |
| Lognormal log-scale standard deviation | 0.50 | 0.65 |
| Mean first-request cost | 0.04 | 0.10 |

The marginal conversion probabilities are $0.990(0.12)=0.1188$ and
$0.985(0.15)=0.14775$; they differ from both the conditional parameters and
the observed sample rates. Unsafe incidence per request is likewise affected
by the served probability.

**Observed results:**

| Metric | Control | Treatment |
|---|---:|---:|
| Assigned users / first requests | 3,000 | 3,000 |
| Converted users | 341 | 468 |
| Conversion | 11.3667% | 15.6000% |
| D7 eligible users | 1,484 | 1,516 |
| Exact D7 retention | 40.9704% | 37.9288% |
| Churn eligible subscribers | 991 | 1,045 |
| Seven-day subscription churn | 6.2563% | 6.5072% |
| Error rate | 0.7667% | 1.4000% |
| p50 duration (seconds) | 1.399201 | 2.236882 |
| p95 duration (seconds) | 3.164144 | 6.117413 |
| Revenue per assigned user | 0.227333 | 0.312000 |
| Cost per first request | 0.039980 | 0.099219 |
| Contribution per assigned user | 0.187354 | 0.212781 |
| Cost per conversion | 0.351728 | 0.636016 |
| Unsafe incidents / all requests | 11/3,000 | 21/3,000 |
| Unsafe incidence per request | 0.3667% | 0.7000% |

Observed conversion lift was **4.2333 percentage points**, with approximate
two-sided 95% unpooled Wald interval **[2.5082, 5.9584] percentage points**;
relative lift was 37.2434%. These are conversion-only intervals, not joint
confidence statements about the scorecard.

The treatment exceeded the predeclared illustrative p95 maximum of 5 seconds,
cost/request maximum of 0.08 and unsafe/request maximum of 0.5%. Its observed
error rate was below the 2% point boundary. These comparisons have no
guardrail inference or release authorization attached.

In the separate analytical exercise, zero incidents in 1,000 independent
trials gave the exact one-sided 95% upper bound **0.299125%**. This is not an
interval for either arm's observed incidence.

**Interpretation candidates — author review required:** this single synthetic
run illustrates why an apparently favorable conversion interval does not
resolve the product decision. Contribution per user rose under the chosen
revenue assumptions even though cost per conversion also rose. Neither
observation cancels the illustrative latency or safety concerns. Observed D7
activity fell despite a slightly higher generating probability; this one draw
does not establish a retention regression.

**Limitations:** generator assumptions create the intended trade-off; there
are no real users, trustworthy production events, safety evaluators, long-term
economics, metric-specific guardrail intervals or segment evaluations.
Retention and cancellation are independent synthetic labels, follow-up is
administrative, and all first-attempt durations are completely observed.
Detection is assumed perfect. The numerical checks do not validate these
business contracts, authorize rollout or constitute personal author review.

The ignored JSON run record retains full precision, configuration and versions.
The small public table above is the intentional evidence summary; no generated
artifact or preview is selected for publication. The JSON is not linked as a
public asset. Unit-test runs, including their seed-99 CLI fixture, are software
checks and are not additional reported product experiments.

## Original suggested follow-ups — recorded before the visual phase

The later [visual lab](VISUAL_LAB.md) implements interactive tail, economic,
rare-event, cohort and aggregation demonstrations. The list below preserves
the original proposals; it is not an execution record for those later controls.

1. Hold the median fixed and vary lognormal spread to isolate tail latency.
2. Vary serving cost and revenue attribution to map contribution sensitivity.
3. Generate multiple requests per user and compare request/user weighting with
   cluster-aware uncertainty.
4. Introduce delayed events and a calendar shift to examine maturity versus
   informative missingness.
5. Simulate imperfect safety detection and severity-specific incidents; compare
   the detected rate with known latent incidence.

Extensions beyond the implemented visual controls remain proposals. The dated
visual validation record is separate, and neither phase contains deployment
conclusions or author-endorsed interpretations.
