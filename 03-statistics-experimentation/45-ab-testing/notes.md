# A/B testing: what makes the difference interpretable?

## Estimand and experimental contract

Let $Z_i$ indicate assignment and $Y_i(z)$ the binary outcome user $i$ would
have under assignment $z$. Only one potential outcome is observed:

$$
Y_i=Z_iY_i(1)+(1-Z_i)Y_i(0),\qquad
\tau=E[Y_i(1)-Y_i(0)].
$$

Random assignment makes $Z_i$ independent of potential outcomes under the
design. Consistency, no relevant cross-user interference, comparable measurement
and complete outcome windows are also needed. The mean difference estimates
the effect of assignment for the enrolled population. Generalizing to other
users or seasons needs further evidence. Randomization balances covariates in
expectation; it does not force identical groups in a particular realization.

Analyze users according to original assignment. A user who never opens the
feature still belongs in an ITT denominator. In this simulation every outcome
is observed; zero means measured noncompletion. Missing telemetry cannot simply
be relabeled as zero in a real experiment. Filtering to users who engaged after
assignment may select different populations. It does not identify the causal
effect among recipients without additional assumptions.

Sticky user assignment avoids switching across visits. It cannot remove
interference in shared organizations, marketplaces or collaborative agents.
Randomize at the relevant cluster or use a design that models spillovers.
Repeated requests from one user do not become independent because there are
many of them. Aggregate at the unit or use dependence-aware inference.

For an AI workflow, freeze eligibility, intervention, primary metric, observation
window, stopping rule and decision thresholds before collecting data. If both
retrieval and generation change, a valid test estimates the effect of that
package; it cannot attribute the effect to retrieval alone. Offline Recall@K
and NDCG support diagnostics and screening. Online task completion addresses
a different question.

## Deriving the numerical analysis

With $x_g$ completions among $n_g$ assigned users in group $g$:

$$
\hat p_g=\frac{x_g}{n_g},\qquad
\hat\Delta=\hat p_B-\hat p_A,\qquad
\text{relative lift}=\frac{\hat\Delta}{\hat p_A}.
$$

A difference of 0.01 means **1 percentage point**; against a 0.10 baseline it
means **10% relative lift**. Relative lift is undefined at a zero control rate;
the educational core refuses such sparse input rather than reporting infinity.

Under $H_0:p_A=p_B$, estimate the shared probability using pooled counts:

$$
\hat p=\frac{x_A+x_B}{n_A+n_B},\qquad
SE_0=\sqrt{\hat p(1-\hat p)\left(\frac1{n_A}+\frac1{n_B}\right)},
$$

$$
Z=\frac{\hat\Delta}{SE_0},\qquad
p_{\text{two-sided}}=2\Phi(-|Z|).
$$

Conditional on the null and its assumptions, the p-value is the probability of
a test statistic at least as extreme as observed. It is not a posterior
probability that the null is true or that the treatment has business value.

`erfc` computes the normal tail without subtracting a CDF near one. For a
two-by-two table of arm by completion, the uncorrected Pearson statistic equals
$Z^2$, so its one-degree-of-freedom p-value agrees. SciPy's default continuity
correction must be explicitly disabled for this comparison. Tests check the
statistic and p-value, unequal arm sizes and the sign reversal on swapping arms.
See the [official API and statistical sources](references.md).

For an illustrative unpooled Wald interval, do not constrain variance to $H_0$:

$$
SE_\Delta=\sqrt{\frac{\hat p_A(1-\hat p_A)}{n_A}
                  +\frac{\hat p_B(1-\hat p_B)}{n_B}},\qquad
CI=\hat\Delta\pm z_{1-\alpha/2}SE_\Delta.
$$

Repeated application of this procedure has approximate coverage under its
assumptions; it does not assign probability to the fixed parameter lying in
this observed interval. The interval is not the inversion of the pooled test,
so decisions can disagree near a boundary. Neither procedure is reliable
merely because the inputs are syntactically valid. Five observed successes and
failures per arm is only the minimum accepted here. Score-based difference
intervals and appropriate exact tests are alternatives to assess for low counts.
An exact binomial interval for a single proportion is not an exact interval for
the difference of two proportions.

## SRM and planning are different checks

Under independent allocation to B with probability $q$, conditional on the total
enrolled sample $N$, $n_B\sim\operatorname{Binomial}(N,q)$. The example calls
SciPy's two-sided `binomtest(n_B, N, q)` and uses a prespecified illustrative
0.001 investigation threshold, separate from the outcome-test alpha.

SRM can reveal allocation, eligibility, logging or post-assignment filtering
problems. A failure is a reason to investigate, not to discard random users
until the split looks balanced. Passing does not rule out all measurement bias.
The binomial allocation model must match the actual design: blocked or exactly
balanced randomization has a different count distribution. The repeated-count
simulations below have fixed group sizes and do not use SRM to validate them.

For equal arm sizes, define $\bar p=(p_A+p_B)/2$ and
$V_1=p_A(1-p_A)+p_B(1-p_B)$. The planning helper uses:

$$
n\approx\left\lceil
\frac{\left[z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)}
       +z_{1-\beta}\sqrt{V_1}\right]^2}{(p_B-p_A)^2}
\right\rceil.
$$

It distinguishes the null threshold variance from the alternative variance.
The closed-form calculation neglects the smaller rejection tail when solving
for $n$. Tests independently invert both normal tails with SciPy's root solver
and verify agreement to within one user per arm for 0.10 versus 0.11, alpha
0.05 and power 0.80. This verifies a normal approximation, not an exact binomial
power calculation. Unknown baseline rates, allocation imbalance, attrition and
cluster dependence can invalidate the plan. Day 46 covers planning in more depth.

An MDE is an effect used to specify detection power, not a guaranteed observed
effect and not automatically the smallest effect worth shipping. The example
separately specifies a 0.005 minimum worthwhile difference; it does not test a
null of $\Delta\le0.005$. A test against zero alone cannot establish that
business threshold. Latency, safety and cost guardrails remain unmeasured.

## Mistakes that survive a correct formula

| Shortcut | Consequence | Design response |
|---|---|---|
| Compare last month with this month | Seasonality or traffic changes can explain the difference | Concurrent randomized arms |
| Stop on the first $p<0.05$ | Ordinary fixed-horizon error control is lost | Predefined horizon or valid sequential design |
| Pick a favorable metric or subgroup after looking | Multiplicity and selection change false-positive behavior | Prespecify or label exploratory analysis and validate separately |
| Analyze requests as independent users | Dependence understates uncertainty | Align randomization, metric and inference units |
| Remove nonengaged users | Post-assignment selection changes the causal question | Preserve the assigned-user denominator |
| Treat nonsignificance as equivalence | Potentially useful or harmful effects can remain compatible | Read the interval; use a planned equivalence/noninferiority design if relevant |
| Ship on significance alone | Value and guardrails remain unresolved | Consider effect uncertainty and operational constraints |

## Dated experiment record — 2026-11-11

Executed locally by Codex on Windows using stable Python **3.11.14**, NumPy
**2.4.6**, SciPy **1.17.1** and pytest **9.1.1**. These are synthetic observations,
not author conclusions. **All interpretation candidates below require author
review.** Commands were run from the repository root using the existing local
reference environment; no dependencies were installed or changed.

```powershell
.local/environments/s08-reference-2026-10-08/Scripts/python.exe 03-statistics-experimentation/45-ab-testing/from_scratch.py
.local/environments/s08-reference-2026-10-08/Scripts/python.exe 03-statistics-experimentation/45-ab-testing/example.py --output 03-statistics-experimentation/45-ab-testing/outputs/run-2026-11-11.json
.local/environments/s08-reference-2026-10-08/Scripts/python.exe -m pytest -q 03-statistics-experimentation/45-ab-testing/tests/test_ab_testing.py
```

The JSON report stays ignored. Its path above is a local output destination,
not a public asset link. No previews were selected and no visual phase ran.

### 1. One user-randomized A/B experiment

**Hypothesis:** under independent binary outcomes, a hypothetical workflow with
generating completion probability 0.11 versus 0.10 may produce a detectable
positive difference. The tested null is equality, two-sided alpha 0.05.

**Configuration:** 30,000 users, NumPy `default_rng(45)`, `integers(0, 2)`
assignment followed by independent uniform outcome draws. The practical
threshold is 0.005; the separate SRM investigation threshold is 0.001. Every
assigned user has a completed observation window in the generator.

| Quantity | Observed result |
|---|---:|
| Control users / completions | 14,986 / 1,473 |
| Treatment users / completions | 15,014 / 1,594 |
| Control completion rate | 9.8292% |
| Treatment completion rate | 10.6168% |
| Absolute difference | +0.7876 percentage points |
| Relative lift | +8.0127% |
| Pooled Z | 2.251382 |
| Two-sided p-value | 0.02436134 |
| Unpooled Wald 95% interval | [+0.1020, +1.4731] percentage points |
| Exact binomial SRM p-value | 0.87612434 |

**Interpretation candidate — author review required:** equality is rejected by
the prespecified illustrative 5% test. The interval still includes improvements
below the 0.5 pp worthwhile threshold. Allocation counts do not trigger the
example's SRM threshold, which is not proof of valid instrumentation.

**Limitation:** one seed with known independent Bernoulli outcomes, no real
users, noncompliance, interference, missing telemetry or guardrail measurements.
Neither statistical detection nor the assumed 1 pp effect justifies launch.

### 2. Repeated A/A calibration

**Hypothesis:** for the ideal independent null generator, fixed-horizon tests
reject near the nominal 5% frequency and Wald coverage is near 95%.

**Configuration:** 2,000 independent experiments, exactly 15,000 users per arm,
$p_A=p_B=0.10$, stream seed 4501, alpha 0.05. Separate binomial counts are
equivalent to independent Bernoulli sums. No significance-based stopping or
SRM filtering is applied.

**Result:** 98/2,000 rejections, **4.90%**; Monte Carlo standard error **0.483
percentage points**; exact binomial 95% interval for the rejection probability
**[3.996%, 5.939%]**. Observed Wald coverage of the true zero difference was
**95.10%**.

**Interpretation candidate — author review required:** this run is compatible
with nominal rejection frequency under its specified generator. A single
rejection rate close to alpha is not a calibration guarantee.

**Limitation:** no production assignment or telemetry pipeline is exercised.
Coverage and rejection frequencies have Monte Carlo error. The interval above
quantifies simulation uncertainty, not uncertainty about a product effect.

### 3. Sample size and rejection frequency under an alternative

**Hypothesis:** with a fixed true 1 pp difference, increasing independent users
per arm increases detection probability.

**Configuration:** $p_A=0.10$, $p_B=0.11$, two-sided alpha 0.05; 2,000 independent
experiments per row, balanced fixed arm sizes. Each row uses a separate stream.

| Users per arm | Seed | Rejections | Rejection frequency | MC SE (pp) | Exact 95% MC interval | Wald coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 4502 | 240/2,000 | 12.00% | 0.727 | [10.607%, 13.505%] | 94.55% |
| 5,000 | 4503 | 743/2,000 | 37.15% | 1.080 | [35.027%, 39.310%] | 94.80% |
| 15,000 | 4504 | 1,620/2,000 | 81.00% | 0.877 | [79.211%, 82.698%] | 94.95% |

**Interpretation candidate — author review required:** the measured detection
frequencies increase in this run; at 15,000 per arm the simulation interval
includes the 80% planning target. The normal planning helper returned **14,751
users per arm** for the prespecified 0.10/0.11 probabilities, alpha 0.05 and
target power 0.80. That is an approximation, not a promised power level for
another design.

**Limitation:** balanced independent arms, one alternative and finite simulation
counts. These observations do not determine calendar duration, cluster-design
requirements or the economic value of an AI workflow. Sample-size and
effect-size sensitivity in unit tests verifies mathematical behavior, not an
additional empirical effect-size sweep.

### Numerical validation

The assumed-count standalone illustration (1,500/15,000 versus 1,650/15,000)
gave $Z=2.825039$, $p=0.00472748$ and Wald interval [0.3063, 1.6937] pp.
Those inputs are a worked calculation, not the user simulation above.
The focused suite passed **45 tests**, with no skips or warnings, covering
SciPy agreement, interval construction, sparse/invalid inputs, user uniqueness,
missing/nonbinary data, seeded assignment, SRM under a deliberately imbalanced
split, independent normal-power inversion and simulation accounting. This does
not establish scientific author approval, CI results or production correctness.

## Extensions proposed during the initial study

- Simulate correlated requests within users and compare naive request-level
  uncertainty with user aggregation or cluster-aware inference.
- Compare fixed-horizon inference with naive repeated checking under A/A; use
  a separate design to evaluate valid sequential alternatives.
- Introduce differential missing telemetry or noncompliance and examine why
  changing the denominator can bias an otherwise randomized comparison.
- Compare Wald coverage with score-based difference intervals at rare-event
  rates; do not bypass this implementation's sparse-input errors.

These suggestions were unexecuted when the nonvisual record above was made.
The later [visual lab](VISUAL_LAB.md) implements correlated-event and naive
peeking simulations with separate seeds and run records. Valid sequential
alternatives, differential missingness/noncompliance and score-based interval
comparisons remain proposed, not executed. The historical tables above are
preserved; visual runs do not replace them.
