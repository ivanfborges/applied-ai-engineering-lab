# Technical Notes

## From observations to a sampling distribution

Let \(X_1,\ldots,X_n\) be IID observations with finite mean \(\mu\) and
variance \(\sigma^2\). The sample mean

\[
\bar X=\frac{1}{n}\sum_{i=1}^{n}X_i
\]

is itself random before the sample is observed. Linearity of expectation gives

\[
\operatorname{E}[\bar X]=\mu,
\]

so the mean is unbiased under this sampling model. Independence gives

\[
\operatorname{Var}(\bar X)
=\frac{1}{n^2}\sum_{i=1}^{n}\operatorname{Var}(X_i)
=\frac{\sigma^2}{n}.
\]

The standard deviation of the sampling distribution is therefore

\[
\operatorname{SE}(\bar X)=\frac{\sigma}{\sqrt n}.
\]

When \(\sigma\) is unknown, it is commonly estimated by the sample standard
deviation \(s\), giving the estimated standard error \(s/\sqrt n\).

Standard deviation and standard error answer different questions:

| Quantity | Describes | Shrinks just because \(n\) grows? |
|---|---|---|
| Standard deviation \(s\) | Variation among observations | No |
| Standard error \(s/\sqrt n\) | Estimated variation of the sample mean | Yes, under the assumed sampling structure |

## What the CLT contributes

The variance calculation gives the scale of the mean's fluctuations. The
classical CLT also characterizes their limiting shape:

\[
\frac{\sqrt n(\bar X-\mu)}{\sigma}
\xrightarrow{d}N(0,1).
\]

Equivalently, for a sufficiently accurate approximation,

\[
\bar X\approx N\left(\mu,\frac{\sigma^2}{n}\right).
\]

This statement is asymptotic. `n >= 30` is not a theorem: convergence can be
rapid for a symmetric light-tailed population and slow for highly skewed or
heavy-tailed data. The classical result also requires finite variance. More
general CLTs exist for some non-identical or dependent sequences, but their
conditions and variance calculations differ from the basic IID formula.

The Law of Large Numbers and the CLT play different roles:

- the Law of Large Numbers says where \(\bar X\) goes;
- the CLT describes the scaled fluctuations around that destination.

## Constructing and interpreting confidence intervals

If \(\sigma\) is known and the normal approximation is adequate, a two-sided
\((1-\alpha)\) interval for \(\mu\) is

\[
\bar X\pm z_{1-\alpha/2}\frac{\sigma}{\sqrt n}.
\]

When sampling from a normal population with unknown variance,

\[
T=\frac{\bar X-\mu}{s/\sqrt n}
\]

has a Student-\(t\) distribution with \(n-1\) degrees of freedom, producing

\[
\bar X\pm t_{1-\alpha/2,n-1}\frac{s}{\sqrt n}.
\]

The \(t\) critical value is larger than its normal counterpart for finite
degrees of freedom because estimating \(\sigma\) adds uncertainty. The two
critical values converge as sample size grows.

For a 95% frequentist procedure, approximately 95% of intervals constructed
over repeated samples contain the fixed true parameter when the assumptions
hold. After one interval has been observed, it is not generally correct to say
that the fixed parameter has a 95% probability of being inside it. A Bayesian
credible interval can make a posterior probability statement conditional on
its likelihood and prior, but it is a different inferential object.

Increasing the confidence level increases the critical value and interval
width. Increasing independent sample size decreases the margin of error only
at the square-root rate. An approximate target for the mean's sample size is

\[
n\approx\left(\frac{z_{1-\alpha/2}\sigma}{m}\right)^2,
\]

where \(m\) is the desired margin of error and \(\sigma\) must be supplied or
estimated from relevant prior evidence.

## Proportions and the Wilson interval

For Bernoulli data, the usual plug-in standard error is

\[
\sqrt{\frac{\hat p(1-\hat p)}{n}}.
\]

The corresponding Wald interval can extend outside \([0,1]\) and have poor
coverage for small samples or proportions near zero or one. The Wilson score
interval rearranges the score-test inequality and behaves better in those
settings. `from_scratch.py` implements Wilson's formula while keeping its
binomial sampling assumptions explicit.

For differences between two proportions or means, estimate the difference
directly and construct an interval for that difference. Two separate intervals
do not answer the comparison question as cleanly, and paired observations need
a paired analysis rather than an independence formula.

## Assumptions and failure modes

### Dependence and the sampling unit

For a mean with correlated observations,

\[
\operatorname{Var}(\bar X)
=\frac{1}{n^2}\left(
\sum_i\operatorname{Var}(X_i)
+2\sum_{i<j}\operatorname{Cov}(X_i,X_j)
\right).
\]

The covariance terms do not disappear. Treating 100 conversations from each
of 50 users as 5,000 independent units can make an interval far too narrow.
Depending on the design, remedies can include aggregation at the independent
unit, cluster-robust standard errors, cluster bootstrap, block bootstrap,
hierarchical models, or time-series methods.

### Bias and representativeness

A confidence interval quantifies uncertainty under a model and sampling
procedure. It does not automatically include selection bias, label error,
measurement error, leakage, missing segments, or distribution shift. A very
large biased sample can produce a precise interval around the wrong target.

### Heavy tails and nonlinear estimators

Finite variance is essential to the classical IID CLT. Extreme observations
can also make convergence too slow for the available sample size. Medians,
quantiles, ratios, ranking metrics, and other nonlinear estimators have their
own sampling behavior. Bootstrap methods can help when analytical formulas
are inconvenient, but resampling must preserve the relevant dependence and a
naive percentile bootstrap is not universally reliable.

### Multiple comparisons and practical significance

Nominal coverage applies to a specified procedure, not automatically to a
large collection of selectively reported intervals. Simultaneous inference or
multiplicity control may be necessary. With huge samples, negligible effects
can be estimated very precisely, so the interval should be compared with a
practical decision threshold rather than only with zero.

## Applied AI examples

- **LLM and RAG evaluation:** define whether the independent unit is a
  question, conversation, user, or document; report uncertainty for the metric
  and important slices; account for evaluator and repeated-user dependence.
- **Model comparison:** report an interval for the paired or independent metric
  difference, then evaluate both statistical and operational relevance.
- **Latency:** an interval for mean latency may be valid but irrelevant when
  the service objective concerns p95 or p99 latency. Use the estimand that
  matches the decision.
- **Monitoring:** distinguish a movement in the sampled metric from evidence of
  an underlying change, while also checking drift in the monitored population.

## Uncertainty communication checklist

A useful report states:

- the estimand and point estimate;
- interval endpoints, confidence level, and construction method;
- sample size and unit of independence;
- target population and sampling window;
- key assumptions and excluded uncertainty sources;
- effect scale and practical decision threshold;
- important segments or sensitivity analyses.

Avoid presenting an interval as a guarantee, as a posterior probability, or as
proof that the data-generating and evaluation processes are unbiased.

## Suggested extensions (not executed)

- Compare normal, uniform, exponential, and log-normal populations at common
  sample sizes.
- Repeat coverage experiments across \(n=5,10,30,100\) and compare normal,
  Student-\(t\), bootstrap, and robust procedures where appropriate.
- Generate clustered observations and compare row-level intervals with a
  cluster-aware method.
- Compare Wilson and Wald coverage for rare success probabilities.

These remain proposed experiments until their configurations and observed
results are executed and recorded.
