# Difference-in-Differences: identification before regression

## Estimand and derivation

Let $D_i$ indicate membership in the group adopting at the common intervention
date, and let $Y_{it}(0)$ and $Y_{it}(1)$ denote untreated and treated potential
outcomes. In the canonical two-period setting the target is

$$
ATT = E[Y_{i,post}(1)-Y_{i,post}(0)\mid D_i=1].
$$

Under no anticipation, the treated pre-outcome is untreated. Thus

$$
E[Y_{post}-Y_{pre}\mid D=1]
= ATT + E[Y_{post}(0)-Y_{pre}(0)\mid D=1].
$$

Subtracting the control change identifies ATT if

$$
E[Y_{post}(0)-Y_{pre}(0)\mid D=1]
= E[Y_{post}(0)-Y_{pre}(0)\mid D=0].
$$

This is parallel trends on the chosen outcome scale. It permits different
baseline levels and stable unobserved group differences. It excludes untreated
changes that differ systematically across groups. Additive parallel trends in
points need not imply parallel trends in log outcomes. ATT concerns treated
units in the specified period, not all future adopters. See the
[authors' DiD introduction](https://bcallaway11.github.io/did/articles/multi-period-did.html).

## What the regression does

The saturated two-group/two-period regression is

$$
Y_{it}=\beta_0+\beta_1 D_i+\beta_2 Post_t+
\tau D_i Post_t+\varepsilon_{it}.
$$

Its interaction coefficient is the difference of the four sample cell means.
For this complete panel with one shared rollout date, unit/time effects give

$$
Y_{it}=\alpha_i+\lambda_t+\tau D_i Post_t+\varepsilon_{it}.
$$

Every team is observed in all six periods, and the manual estimator uses
equal weights within each pre/post window. Here TWFE matches the difference
between the groups' mean unit-level pre/post changes. The target averages the
three post periods; it does not isolate the last period's effect. The generator
uses a constant effect of 4 points. Equality of estimates does not depend on
parallel trends: both methods can agree and share the same contamination.
An unbalanced panel or a different weighting policy need not preserve this
specific pooled-mean identity; the helper rejects that input rather than
silently changing its estimand.

## Inference contract

Treatment is assigned to support teams, and each team has repeated outcomes.
The generator adds stationary AR(1) noise with correlation 0.6 and marginal
standard deviation 1.5. `fit_twfe` clusters by team, uses CRV1 with
`use_correction=True`, `df_correction=True`, and explicitly selects `use_t=True`.
For 100 teams its interval uses a Student $t$ reference with 99 degrees of
freedom. The [statsmodels covariance documentation](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html)
specifies these switches; robust covariance defaults alone do not select that
$t$ reference. Tests independently reconstruct the critical-value calculation.

Independent clusters and sufficiently many assignment units remain substantive
requirements. The two named groups are not the two inference clusters here:
there are 100 synthetic team clusters. If an intervention is actually assigned
at a higher level, clustering by team may be too fine. A guard against fewer
than two teams per group only prevents a degenerate example; it is no assurance
of reliable small-cluster inference. Specialized small-cluster methods and
coverage simulations are outside this implementation. Clustered standard
errors do not fix violation of parallel trends.

## Design risks and diagnostics

| Risk | Consequence and practical check |
|---|---|
| Anticipation | Pre-outcomes may already be affected; audit training and announcement dates, not just feature activation |
| Spillovers | Never-treated teams may benefit through shared agents or knowledge; inspect team boundaries and exposure |
| Differential shocks | A bonus, staffing change or backlog cleanup only in treated teams can mimic a treatment effect |
| Composition/measurement change | Automation can remove easy tickets from the observed human queue; define outcomes on a stable incoming-request population |
| Endogenous timing | Rollout after a bad month may create regression to the mean; investigate assignment and longer histories |
| Post-treatment adjustment | Conditioning on adoption intensity or a mediator can change the estimand or induce selection bias |

Multiple pre-periods support trend diagnostics, but the relevant post-treatment
untreated path remains unobserved. A nonsignificant pretest can have low power;
selecting specifications after passing it can distort inference. See
[Roth (2022)](https://www.aeaweb.org/articles?id=10.1257/aeri.20210236).
The implemented placebo uses only periods -3, -2 and -1, pretending adoption
starts at -1. It compares the last pre-period with the average of the earlier
two. It is a descriptive falsification check, not an event-study lead test or
an equivalence test. The four printed intervals are separate marginal 95%
intervals; they are not adjusted simultaneous statements or a specification
selection rule.

## Why a trend violation contaminates the contrast

The paired scenario adds $sD_it$ to untreated outcomes. Then

$$
E[\widehat{\tau}_{DiD}]-\tau
=s(\bar t_{post}-\bar t_{pre}).
$$

The rollout windows have mean times 1 and -2: $s=0.8$ contributes 2.4 points.
The placebo windows have mean times -1 and -2.5: the same slope contributes
1.2 points despite no actual treatment. Because both scenarios reuse identical
noise, the differences between their realized estimates equal these shifts
up to floating-point precision. A single estimate's difference from truth
also includes sampling noise; it is not a Monte Carlo estimate of bias.

## Executed synthetic study — 2026-11-12

Hypotheses: (1) manual and TWFE estimates coincide for this balanced single-date
design; (2) adding a differential untreated slope shifts rollout and placebo
contrasts by the window-specific analytical amounts, while preserving numerical
agreement between estimators.

Configuration: seed 51; 100 teams, first 50 treated; periods -3 through 2;
actual adoption at 0; treatment effect +4 synthetic productivity points;
team intercepts drawn from $N(0,3^2)$ with a +2 treated level offset; common
time effects [-2, -1.1, 0, 1, 1.8, 2.2]; AR(1) noise as described above.
The two scenarios differ only in slope (0 versus 0.8). The pre-placebo cutoff
is -1, with all actual post-treatment observations excluded. This is an
educational nonrandom assignment rule, not a randomized trial.

Executed by Codex on Windows with Python 3.11.14, NumPy 2.4.6, pandas 3.1.1,
SciPy 1.17.1 and statsmodels 0.15.0. Command from the repository root:

```powershell
.local/environments/day46/Scripts/python.exe 03-statistics-experimentation/51-difference-in-differences/example.py
```

The existing local environment name identifies the interpreter used for this
new Day 51 run; this record does not replace any Day 46 historical result.

| Scenario | Comparison | Known treatment effect | Manual DiD = TWFE | Team-cluster SE | Marginal 95% interval | Analytical trend contribution |
|---|---|---:|---:|---:|---|---:|
| Parallel | Rollout | 4.000000 | 3.922205 | 0.297451 | [3.331996, 4.512413] | 0.000000 |
| Parallel | Pre-placebo | 0.000000 | -0.185955 | 0.367098 | [-0.914357, 0.542446] | 0.000000 |
| Differential trend | Rollout | 4.000000 | 6.322205 | 0.297451 | [5.731996, 6.912413] | 2.400000 |
| Differential trend | Pre-placebo | 0.000000 | 1.014045 | 0.367098 | [0.285643, 1.742446] | 1.200000 |

The baseline treated before/after change was 6.742409 points; the control
change was 2.820204. Their difference was 3.922205. Manual and TWFE agreement
passed the script's absolute tolerance of $10^{-10}$ in all four fits. The
paired changes were +2.4 points for rollout and +1.2 for placebo, as derived.

**Interpretation candidates — author review required:** the constructed
comparison removes common time changes, while it cannot separate the added
untreated differential slope from treatment. In the violated scenario the
interval excludes the known effect of 4 despite estimator agreement. Its SE
is unchanged because the added deterministic group/time component shifts the
coefficient without changing this balanced design's residuals. The placebo
detects this particular trend construction; it cannot certify the baseline
design or detect all possible post-only shocks.

Limitations: one seed and one constructed panel per scenario; no empirical
bias, coverage or power study; no real treatment assignment, measured AI
guardrails, adoption variation, missingness or production impact. Equality
tests and analytical paired shifts validate numerical behavior, not causal
identification on observational business data. No personal author conclusion
is recorded here.

## Staggered adoption and alternatives

The code intentionally does not accept team-specific dates. With staggered
adoption, conventional TWFE can use already-treated teams as controls and mix
cohorts with evolving effects. Define group-time $ATT(g,t)$ and a target
aggregation before selecting an estimator; consider Callaway–Sant'Anna or
Sun–Abraham methods. A single-cohort success here does not validate those
settings. The [multi-period introduction](https://bcallaway11.github.io/did/articles/multi-period-did.html)
explains the contaminated-comparison problem.

When randomization is feasible, a well-designed A/B test can offer a stronger
assignment basis. Matching/weighting from [Day 50](../50-causal-inference-ii-propensity-score-matching/)
targets observed covariate differences under different assumptions; balancing
levels does not itself establish untreated parallel changes. Conditional DiD
requires its own support and identification argument. With one treated
aggregate and a credible donor history, synthetic control may be worth
considering; a good pre-fit is still insufficient against differential shocks.

## Interview practice

**Why can baseline means differ?** Time-invariant group differences cancel in
the change contrast. Differential untreated changes do not.

**Does adding fixed effects make the estimate causal?** They remove the modeled
stable unit and shared time components. Causality still depends on the design,
especially the unobserved untreated trend.

**What does a narrow interval establish?** Conditional sampling precision for
the chosen model and dependence assumptions. It does not bound confounding bias
or prove quality and cost guardrails.

**How would you assess a RAG rollout?** Define assignment/exposure dates, stable
units and outcome denominators; choose a credible comparison, audit concurrent
changes and spillovers, examine pre-history, use appropriate inference and
measure quality/cost separately before interpreting productivity.

**Why not estimate a company-wide effect?** The target is the treated teams
over the selected post-periods. Different teams or later assistant versions
can respond differently.

## Suggested extensions — not executed

Simulate spillovers, selective attrition and post-only shocks (which can evade
pre-placebo checks); estimate event-time contrasts with a declared reference
period; run repeated independent panels to measure interval coverage and power;
compare cohort-aware estimators under staggered adoption. These are proposals,
not results from this study.
