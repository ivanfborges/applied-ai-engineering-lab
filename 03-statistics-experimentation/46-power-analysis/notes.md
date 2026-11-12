# Power, effect size and experiment feasibility

## Define the question before counting observations

Let $p_C$ and $p_T$ be population probabilities of a prespecified binary outcome.
The two-sided equality test uses $H_0:p_T=p_C$ and $H_1:p_T\ne p_C$.
Its power at a particular alternative is

$$
\pi(p_C,p_T)=P_{p_C,p_T}(\text{reject }H_0)=1-\beta(p_C,p_T).
$$

The significance level $\alpha$ controls false rejection under the null,
subject to calibration of the actual test. An 80% power target is conditional
on a chosen alternative; it is not a posterior probability that treatment is
better. A two-sided test can also detect harm.

For $p_C=0.10$ and $p_T=0.11$, the absolute effect is
$\Delta=p_T-p_C=0.01$: one percentage point. Relative uplift is
$\Delta/p_C=10\%$. Confusing a 1% relative improvement with one percentage point
changes the sample-size question substantially.

For continuous metrics, Cohen's $d=\Delta/\sigma$ expresses an effect relative
to noise. For binary metrics, the baseline supplies the Bernoulli variance
$p(1-p)$; an absolute difference alone does not determine power. This study
keeps probabilities explicit instead of replacing the model with a
variance-stabilized effect-size test.

The **smallest effect size of interest (SESOI)** expresses business value.
The **MDE** is the effect a particular design detects at a chosen power.
For a fixed sample, MDE can be found by inverting the power function over
feasible treatment probabilities; upward and downward effects need not have
the same MDE on the probability scale. This inversion is explained here but
is not implemented as a general-purpose solver in the original analytical
scripts. The later visual lab adds a bounded positive-effect inversion in
`power_math.py`; see [the visual guide](VISUAL_LAB.md).

To align the design with the decision, choose the SESOI independently of the
traffic constraint, then ask whether the resulting MDE is sufficiently small.
Testing against zero with high power at $\Delta$ does not establish that an
estimated benefit exceeds a deployment threshold of $\Delta$.

## The two standard errors in the planning approximation

For independent sample proportions,

$$
\hat\Delta=\hat p_T-\hat p_C,\qquad
V_1=\frac{p_C(1-p_C)}{n_C}+\frac{p_T(1-p_T)}{n_T}.
$$

The assumed alternative supplies $V_1$. For the pooled null rejection
threshold, the planning approximation substitutes the expected pooled rate

$$
\bar p=\frac{n_Cp_C+n_Tp_T}{n_C+n_T},\qquad
V_0=\bar p(1-\bar p)\left(\frac1{n_C}+\frac1{n_T}\right).
$$

Approximating $\hat\Delta$ by $N(\Delta,V_1)$ and treating the null standard
error as fixed gives rejection outside $[-c,c]$, where
$c=z_{1-\alpha/2}\sqrt{V_0}$. Thus

$$
\pi\approx
\Phi\left(\frac{-c-\Delta}{\sqrt{V_1}}\right)
+1-\Phi\left(\frac{c-\Delta}{\sqrt{V_1}}\right).
$$

`normal_power` evaluates the upper probability with SciPy's survival function
to avoid cancellation in `1 - cdf`. It retains both rejection tails, including
the remote one. At $\Delta=0$, $V_0=V_1$ and the expression returns $\alpha$.
The tests also integrate a normal density over the two rejection regions.

This is **not exact binomial power**. In an actual pooled Z-test the estimated
null standard error is random. Here it is replaced by a planning value, and
normality itself is approximate. Small counts or probabilities near the
boundaries may require exact methods or simulation of the intended analysis.
The code validates mathematical inputs; it does not certify design adequacy.

## Recovering the balanced sample-size formula

When $n_C=n_T=n$, write $v_0=2\bar p(1-\bar p)$ and
$v_1=p_C(1-p_C)+p_T(1-p_T)$. Ignoring the remote rejection tail and solving
for target power $q$ gives

$$
n\approx\frac{
\left(z_{1-\alpha/2}\sqrt{v_0}+z_q\sqrt{v_1}\right)^2
}{\Delta^2}.
$$

The numerator combines the rejection threshold and detection probability on
the noise scale; the squared effect appears in the denominator because standard
errors decrease as $1/\sqrt n$. `sample_size_per_arm` implements this formula
using only `statistics.NormalDist` and rounds upward. It supports nonzero
effects, interior probabilities and $q>\max(0.5,\alpha)$.

At fixed variances, halving $|\Delta|$ requires four times $n$. For Bernoulli
outcomes, changing the effect changes the variance, so the factor is approximate.
Higher target power or a smaller significance threshold also increases cost.

`required_sample_sizes` numerically solves the two-tail expression using
SciPy's bracketed Brent solver. Its ratio convention is
$r=n_T/n_C$, with $n_C$ capped at one billion during search. Both arm sizes are
rounded upward and their resulting approximate power is checked. Unequal
allocation can differ slightly from the requested ratio after rounding and
is not claimed to be the globally smallest integer design. Balanced sizing
tests check that one fewer unit per arm falls below the target.

Equal allocation is close to efficient when costs and variances are comparable;
it is not universally optimal. Different variances, treatment costs or exposure
constraints can justify unequal allocation. Compare total sample size and cost,
not just the smaller arm. The
[statsmodels power documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.power_proportions_2indep.html)
describes the same pooled-null/separate-alternative framework. Its arm and
ratio labels differ from this example, so they must be mapped explicitly when
making a future library comparison. The
[dominant-tail documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.samplesize_proportions_2indep_onetail.html)
explains the omitted-tail approximation; no statsmodels execution is claimed.

## Translate information requirements into business constraints

Let $U$ be daily distinct potential units, $e$ eligibility, and $s$ treatment
allocation share. Expected daily enrollment is $Ue(1-s)$ for control and $Ues$
for treatment. The accumulation estimate is

$$
D_{\mathrm{enroll}}=\left\lceil\max\left(
\frac{n_C}{Ue(1-s)},\frac{n_T}{Ues}\right)\right\rceil.
$$

The calendar helper uses
$D=\max(D_{\mathrm{enroll}},D_{\min})+L$, with a minimum enrollment period
$D_{\min}$ and final outcome delay $L$. Counts are expected arrivals, not a
guarantee that random assignment fills both arms on that date. A seven-day
floor illustrates a prespecified calendar constraint; it does not prove
weekday or seasonal representativeness.

Users must be new to this experiment, not daily active users counted again
each day. Eligibility is not a post-treatment completion filter: the example
assumes complete binary outcomes for all enrolled users. Missingness cannot
be fixed simply by discarding nonresponders or inflating sample size.

Under a deadline, compute achieved power at the original business effect and
report the sensitivity gap. Extending time, improving measurement or using a
valid variance-reduction/paired design may help. Increasing the assumed effect
to fit the deadline silently changes the question. Changing $\alpha$ or test
sidedness after observing results also changes error control.

Power does not supply an ROI model. Incremental serving expense can be
approximated from treatment enrollment times per-unit extra cost, but decision
value also needs benefit valuation, risk, maintenance and opportunity costs.

## Dependence, alternative designs and common mistakes

- **Conversation versus user:** user randomization plus repeated conversations
  needs user aggregation or cluster-aware inference. For equal cluster size
  $m$ and intraclass correlation $\rho$, $1+(m-1)\rho$ is a useful design-effect
  heuristic. Unequal sizes, few clusters and the chosen estimator require
  design-specific planning; multiplying this example's sample count is not a
  validated cluster design.
- **Offline RAG evaluation:** when both systems answer the same questions,
  paired outcomes and discordance drive sensitivity. Independent-group sizing
  wastes the pairing and may target the wrong variance.
- **Latency and spend:** skewed or heavy-tailed outcomes need a metric and
  estimator matched to their distributions, often pilot-based simulation.
  Binary conversion formulas do not transfer directly.
- **Many metrics or interim looks:** prespecify multiplicity and a valid
  sequential method, or keep one final fixed-horizon analysis. This calculator
  contains neither correction.
- **Observed-effect post-hoc power:** it largely restates the significance
  result. Report the estimate and confidence interval relative to the SESOI;
  failing to reject zero does not establish equivalence or noninferiority.
- **Selection bias:** a larger sample does not repair broken randomization,
  benchmark reuse, contamination or post-treatment exclusions.
- **Precision versus detection:** a target confidence-interval width and power
  for a test against zero are different sample-planning objectives.

## Executed analytical study — 2026-11-11

**Hypothesis:** in the declared model, smaller effects, stricter error tolerance
and unbalanced assignment increase sample requirements; a traffic deadline can
leave inadequate sensitivity to the original business effect.

**Configuration:** Windows, Python 3.11.14, SciPy 1.17.1. NumPy 2.4.6 and
pytest 9.1.1 were used for tests. Control probability 0.10, two-sided equality
test, baseline $\alpha=0.05$, target power 0.80, balanced allocation.
`example.py` and `from_scratch.py` were executed locally by Codex. These are
deterministic analytical calculations; no outcomes, random seed, public data,
Monte Carlo experiment or production traffic were used.

| Scenario (other inputs at baseline) | Control | Treatment | Total | Approximate power |
|---|---:|---:|---:|---:|
| Treatment 0.105 (+0.5 pp) | 57,763 | 57,763 | 115,526 | 0.800003 |
| Treatment 0.11 (+1 pp) | 14,751 | 14,751 | 29,502 | 0.800007 |
| Treatment 0.12 (+2 pp) | 3,841 | 3,841 | 7,682 | 0.800017 |
| Significance 0.01 | 21,950 | 21,950 | 43,900 | 0.800018 |
| Target power 0.90 | 19,747 | 19,747 | 39,494 | 0.900006 |
| Target power 0.95 | 24,421 | 24,421 | 48,842 | 0.950003 |
| Treatment:control ratio 2:1 | 11,125 | 22,249 | 33,374 | 0.800021 |

The standard-library dominant-tail formula also returned 14,751 per arm for
the baseline. Agreement after rounding in this case does not make the methods
identical or exact.

For 40% eligibility, 50/50 assignment, a seven-day minimum enrollment period
and two-day final outcome delay, the baseline plan produced:

| New distinct potential users/day | Calendar days |
|---:|---:|
| 500 | 150 |
| 2,500 | 32 |
| 10,000 | 10 |

At 2,500/day, a **14-day calendar deadline** allows twelve enrollment days,
or 6,000 expected users/arm. Power at the assumed 10% versus 11% probabilities
is **0.431307**. An invented extra serving cost of 0.03 currency units per
treatment user gives **442.53** units for the baseline full-sample plan.
This arithmetic excludes control cost, setup and benefit valuation.

**Interpretation candidate — author review required:** the hypothetical
14-day plan does not meet the chosen 80% sensitivity requirement for the
+1-point effect. The larger 2:1 plan illustrates a sample-efficiency trade-off
under these particular rates. Neither result is an author's conclusion or an
approved business recommendation.

**Limitations:** all probabilities, traffic and costs are assumptions. Results
are conditional on the approximation and design, with no empirical calibration
of the test or evidence about real assistant performance. Baseline uncertainty,
dependence, missing outcomes and traffic variability remain unmeasured.

Focused local verification: 40 tests passed with pytest 9.1.1, without skips or
warnings. An initial calendar-boundary test exposed a one-floating-point-step
rounding error at three days; the helper now handles that boundary, and tests
also retain upward rounding for a genuine fractional day. This verification
does not establish empirical Type I error or power of a finite-sample test.

## Follow-up studies — initial analytical-phase proposals

1. Vary the baseline over a defensible historical range at a fixed absolute
   business effect, retaining the uncertainty in the plan.
2. Simulate the actual pooled Z-test to estimate finite-sample rejection rates
   and Monte Carlo uncertainty, especially for rare outcomes.
3. Compare a paired binary design or cluster-aware estimator with its own
   data-generating assumptions rather than reusing independent-group power.
4. Specify a precision or noninferiority objective and compare its sample
   requirement with detection against zero.

Visual-phase update: proposal 2 has now been executed for the base scenario,
with a separate [dated record](VISUAL_LAB.md#executed-visual-study--2026-11-11).
Rare-outcome sweeps and proposals 1, 3 and 4 remain unexecuted suggestions.
The original analytical results above retain their original configuration.

## Interview discussion

**What does 80% power mean?** Across repetitions under the specified true
effect and assumptions, about 80% reject the null. It does not identify the
probability of the effect being real in the completed experiment.

**What if the team cannot collect the sample?** Report power at the effect
that matters and state which effects remain hard to detect. Discuss time,
measurement, variance reduction and valid alternative designs without changing
the original business requirement silently.

**Does a nonsignificant outcome rule out benefit?** No. Examine effect
estimates and intervals against the business threshold. Equivalence and
noninferiority require prespecified margins and their own tests.
