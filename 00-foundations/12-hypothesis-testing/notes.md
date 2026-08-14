# Technical Notes

## 1. Start with the decision and estimand

A test is meaningful only after defining what is being estimated. In a paired
model comparison, one useful estimand is the population mean query-level score
difference

\[
\mu_d = E[D], \qquad D = Y_{\text{candidate}}-Y_{\text{baseline}}.
\]

The experimental unit is the query, not each token, judge call, or repeated
measurement produced for that query. If users contribute many correlated
queries, even the query may not be the independent unit; clustering must be
reflected in the design and analysis.

A two-sided formulation is

\[
H_0:\mu_d=0,
\qquad
H_1:\mu_d\ne0.
\]

A one-sided alternative such as \(H_1:\mu_d>0\) is justified only when the
direction was chosen in advance and effects in the opposite direction would
not trigger the same claim. Choosing direction after seeing the data changes
the error rate.

## 2. Test statistic and null distribution

For paired differences \(d_1,\ldots,d_n\),

\[
\bar d=\frac{1}{n}\sum_{i=1}^n d_i,
\qquad
s_d^2=\frac{1}{n-1}\sum_{i=1}^n(d_i-\bar d)^2,
\]

and the estimated standard error is

\[
SE(\bar d)=\frac{s_d}{\sqrt n}.
\]

The paired t statistic for a zero null mean is

\[
t=\frac{\bar d}{s_d/\sqrt n}.
\]

Under independent pairs and a normal population of differences, this statistic
has a Student-\(t\) distribution with \(n-1\) degrees of freedom under the
null. The test is often reasonably robust for a mean with larger samples, but
heavy tails, strong skew, influential observations, or dependence can make a
nominal p-value unreliable.

The paired design does not require the baseline and candidate scores
themselves to be independent. It requires the pairs to be independent of one
another under the simple analysis and focuses the distributional assumption on
the differences.

## 3. What the p-value says

For a two-sided statistic, the p-value is

\[
p=P(|T|\ge |t_{\text{obs}}|\mid H_0,\text{ model assumptions}).
\]

It answers a tail-area question conditional on the null model and the analysis
plan. It does not provide:

- the probability that \(H_0\) is true;
- the probability that the result occurred "by chance";
- the magnitude or usefulness of the effect;
- the probability that a replication will be significant;
- protection against a biased or leaked experiment.

At significance level \(\alpha\), a correctly calibrated procedure rejects a
true null at long-run rate \(\alpha\). The realized p-value is not the realized
false-positive probability.

## 4. Error rates and power

| Reality | Do not reject \(H_0\) | Reject \(H_0\) |
|---|---|---|
| \(H_0\) is true | correct | Type I error |
| a specified alternative is true | Type II error | correct |

The Type I rate is \(\alpha\). For a specific alternative parameter value,
the Type II rate is \(\beta\), and

\[
\text{power}=1-\beta.
\]

Power is not a property of a dataset or a test name alone. It is a function of
the effect to detect, sample size, variance, alpha, direction, and design. An
experiment should be planned around a minimum effect that would change a
decision, not around whatever effect happens to become significant.

For a simplified normal mean problem,

\[
n\approx
\left(
\frac{(z_{1-\alpha/2}+z_{1-\beta})\sigma}{\Delta}
\right)^2.
\]

This exposes two useful scaling rules: halving the target effect requires
roughly four times the sample, and doubling the noise also requires roughly
four times the sample. Exact planning must match the intended test and data
structure; Day 46 returns to formal power and sample-size analysis.

## 5. Effect size and confidence interval

For the paired design, Cohen's standardized effect is

\[
d_z=\frac{\bar d}{s_d},
\qquad
t=d_z\sqrt n.
\]

It is useful for comparing the effect with variability in paired differences,
but domain units are often more actionable. A score increase of 0.02 may be
understandable to a product team; a standardized effect of 0.3 may not be.

A two-sided \((1-\alpha)\) confidence interval for the mean difference is

\[
\bar d \pm t_{1-\alpha/2,n-1}\frac{s_d}{\sqrt n}.
\]

At matching alpha, the classical two-sided t test rejects zero exactly when
this interval excludes zero. The interval adds information: it shows the range
of effect sizes compatible with the procedure. It should not be read as a
posterior probability that this fixed interval contains the parameter.

Statistical and practical questions can disagree:

- An interval narrowly above zero can establish detectability while remaining
  entirely below a minimum useful improvement.
- An interval spanning zero and valuable positive effects is inconclusive, not
  evidence that the systems are equivalent.
- Demonstrating equivalence or non-inferiority requires margins and tests built
  for those claims; failure to reject a zero-effect null is insufficient.

## 6. Sign-flip test from first principles

Under a sharp paired null with a justified sign-exchangeability assumption,
the signs of the observed differences may be flipped:

\[
d_i^*=s_i d_i,\qquad s_i\in\{-1,+1\}.
\]

For a small sample, all \(2^n\) sign assignments form an exact reference
distribution. For larger samples, random assignments approximate it. The
Monte Carlo implementation uses

\[
\hat p=\frac{b+1}{B+1},
\]

where \(b\) simulated statistics are at least as extreme as observed among
\(B\) draws. The plus-one correction prevents an estimated p-value of zero and
accounts for the observed arrangement in the randomization argument.

This test is not assumption-free. Arbitrarily flipping signs may be invalid
when the null does not imply the required invariance, when pair construction is
wrong, or when observations across pairs are dependent. A randomization test
based on an actual treatment assignment mechanism has a stronger design-based
interpretation than a generic sign-flip test applied after the fact.

## 7. Choosing an analysis

| Situation | Candidate method | Central caution |
|---|---|---|
| Same examples scored by A and B | paired t or paired resampling test | analyze within-example differences |
| Independent groups with unequal variances | Welch t test | experimental units must be independent |
| Binary paired outcomes | McNemar test | discordant pairs drive inference |
| Independent proportions | two-sample proportion method | check count and approximation conditions |
| Skewed latency metric | bootstrap or suitable robust method | define whether mean, median, or tail is the estimand |
| Many metrics or segments | multiplicity-aware plan | primary outcomes should be predeclared |

Test selection follows the estimand, assignment/sampling mechanism, outcome
type, and dependence structure. It should not be based only on a normality test
or whichever method produces the preferred p-value.

## 8. Common failure modes

### Accepting the null

`p > 0.05` means the procedure did not reject. Low power, noisy measurement,
or a small sample may leave meaningful effects compatible with the data.

### Pseudo-replication

Ten users producing 10,000 messages do not automatically create 10,000
independent units. Ignoring within-user dependence usually makes uncertainty
too small.

### Repeated peeking

Stopping the first time a fixed-horizon p-value crosses 0.05 inflates the false
positive rate. Use a predefined horizon or a valid sequential design.

### Multiple comparisons

Testing many prompts, metrics, or segments increases false discoveries.
Bonferroni controls family-wise error conservatively; Benjamini-Hochberg
targets the false discovery rate. The right family and correction must be
defined in the analysis plan.

### Evaluation leakage

Repeatedly tuning on the benchmark makes it part of development. A test on the
same benchmark does not restore independence. Preserve a final evaluation set
or use a defensible nested process.

### Metric validity

An LLM judge, human rating rubric, or aggregate score introduces measurement
choices. A precise answer to the wrong metric is not useful evidence.

## 9. Engineering interpretation checklist

Before turning a test into an action, report:

1. target population, experimental unit, estimand, and pairing or clustering;
2. predeclared null, alternative, alpha, stopping rule, and primary metric;
3. effect estimate in domain units, confidence interval, and sample size;
4. test statistic, p-value, method, and key assumptions;
5. practical threshold, latency, cost, reliability, safety, and regressions;
6. missingness, evaluator uncertainty, leakage, multiplicity, and segment
   limitations.

The statistical result is evidence for this decision process, not the decision
itself.

## 10. Further extensions (not executed)

- Add skew and influential differences to compare the paired t and sign-flip
  procedures beyond the current normal synthetic model.
- Simulate repeated observations per user and compare naive with
  cluster-aware uncertainty.
- Add a valid sequential-testing design and contrast it with unplanned peeking.
- Simulate mixed null and alternative families to study false discovery rate,
  sensitivity, and dependence beyond the current all-null comparison.
- Compare a zero-effect test with equivalence testing using an explicit margin.

These remain suggestions until run and documented with configuration, observed
results, limitations, and author-reviewed interpretation.
