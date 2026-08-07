# Technical Notes: Statistical EDA

## EDA as a reasoning process

Rigorous EDA starts with the data-generating process and the intended decision,
not with `describe()` or a correlation matrix. A useful sequence is:

1. define the unit of observation and prediction-time boundary;
2. validate schema, units, ranges, duplicates, and missingness;
3. inspect univariate center, spread, shape, and tails;
4. compare robust and non-robust summaries;
5. examine relationships visually and numerically;
6. condition on meaningful subgroups and time;
7. investigate unusual observations and their lineage;
8. separate exploratory findings from confirmatory claims.

Sampling, confidence intervals, and formal hypothesis tests receive dedicated
treatment in later roadmap topics. EDA can generate hypotheses, but reusing the
same data to discover and confirm a claim exaggerates the evidence.

## Center

For observations \(x_1,\ldots,x_n\), the arithmetic mean is

\[
\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i.
\]

The mean uses every magnitude and estimates an expectation under suitable
sampling assumptions. Its sensitivity is also its weakness: one sufficiently
large observation can move it arbitrarily far.

The median is the middle order statistic, or the average of the middle pair
under the convention used here. It depends on ordering rather than the
distance of tail observations. Reporting both is often useful; a gap can signal
asymmetry, mixtures, or influential values, but does not diagnose the cause.

## Spread and estimator conventions

Population variance is

\[
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2.
\]

When estimating population variance from an IID sample, the common unbiased
estimator is

\[
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2.
\]

Estimating \(\bar{x}\) imposes
\(\sum_i(x_i-\bar{x})=0\), leaving \(n-1\) independent deviations. Dividing by
\(n-1\) corrects the downward bias of the variance estimator under the standard
framework. It does not make the sample standard deviation exactly unbiased.

`from_scratch.py` exposes this choice as `ddof`; the denominator is
\(n-\text{ddof}\). The standard deviation is the square root of variance and
returns to the original measurement unit. Both are sensitive to extremes.

### Robust spread

With first and third quartiles \(Q_1\) and \(Q_3\),

\[
\operatorname{IQR} = Q_3-Q_1.
\]

The unscaled median absolute deviation is

\[
\operatorname{MAD} =
\operatorname{median}\left(\lvert x_i-\operatorname{median}(x)\rvert\right).
\]

IQR and MAD describe the central distribution without allowing extreme
magnitudes to dominate. They also discard information that variance retains,
so “robust” does not mean universally superior.

Quantiles have multiple defensible finite-sample definitions. This topic uses
linear interpolation at index \((n-1)p\), matching the current NumPy/pandas
default for the shown calls. A reproducible report should state the convention.
The MAD in this topic is unscaled; multiplying it by a consistency factor is a
separate choice when estimating a Gaussian standard deviation.

## Shape and standardized moments

Population skewness is the standardized third central moment:

\[
\gamma_1 =
\frac{\operatorname{E}[(X-\mu)^3]}{\sigma^3}.
\]

The odd power retains direction: positive values usually indicate a longer or
more influential right tail and negative values a left tail. Zero skewness
means no third-moment asymmetry; it does **not** imply normality. A symmetric
bimodal distribution is a simple counterexample.

Population kurtosis is

\[
\gamma_2 =
\frac{\operatorname{E}[(X-\mu)^4]}{\sigma^4}.
\]

Excess kurtosis subtracts three, so a Gaussian population has excess kurtosis
zero. The fourth power makes the statistic strongly tail-sensitive. “Tail
weight and extreme-deviation contribution” is generally more useful than the
incomplete description “peakedness.”

There are several finite-sample bias corrections for skewness and kurtosis.
`from_scratch.py` deliberately computes uncorrected population moments, while
pandas applies its documented sample estimators. Exact comparisons must align
these definitions.

## Potential outliers are investigation targets

The IQR rule flags values outside

\[
[Q_1-1.5\operatorname{IQR},\ Q_3+1.5\operatorname{IQR}].
\]

It is a descriptive fence, not a probability statement or deletion rule. A
flagged value can be:

- a measurement, parsing, join, or unit error;
- a genuine tail observation;
- a minority population;
- a service failure, fraud event, or other target phenomenon;
- evidence that one global distribution is the wrong model.

A defensible workflow preserves the raw value, traces its provenance, checks
domain constraints, compares robust summaries, segments relevant populations,
and then chooses an action consistent with the use case. Options include
correction, exclusion with an audit trail, capping for a particular model,
transformation, robust modeling, or explicit tail modeling.

The common \(|z|>3\) heuristic is centered on the mean and scaled by the
standard deviation. It is especially easy to misuse on skewed or heavy-tailed
data because both inputs are influenced by the observations being flagged.

## Covariance and correlation

Sample covariance is

\[
s_{xy} = \frac{1}{n-1}
\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y}).
\]

It preserves units. Pearson correlation standardizes the cross-deviation:

\[
r =
\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_i(x_i-\bar{x})^2}
 \sqrt{\sum_i(y_i-\bar{y})^2}}.
\]

Pearson measures linear association and is undefined if either input is
constant. A value near zero only rules out strong linear association in that
sample. In `example.py`, \(y=x^2\) on a symmetric grid is deterministic but has
Pearson correlation approximately zero.

Spearman correlation applies Pearson correlation to ranks. The educational
implementation assigns average ranks to ties. It is appropriate for monotonic
relationships or ordinal data and is less sensitive to extreme magnitudes, but
it discards spacing information and still misses non-monotonic dependence.

Neither coefficient:

- establishes causality;
- proves independence in general;
- controls for confounding, shared trends, or subgroup composition;
- is automatically a feature-selection criterion.

High-dimensional correlation screening also creates multiplicity concerns.
Any confirmatory claim needs a pre-specified question, appropriate uncertainty
analysis, and independent validation.

## Missingness, time, leakage, and aggregation

Dropping missing values silently changes the analyzed population. Missingness
should be measured by column, time, source, subgroup, and—without violating the
prediction boundary—its relationship to outcomes. It may encode an upstream
failure or a meaningful collection process.

Time series can share trends or seasonality and therefore exhibit correlation
without a direct mechanism. Randomly mixing future and past observations can
also create leakage. EDA and preprocessing statistics must respect the same
information boundary as deployment.

Aggregate statistics can reverse or hide within-group relationships
(Simpson's paradox). In applied AI, useful segments include language, document
type, parser, model version, request class, region, and customer tier. Segment
selection should be domain-motivated rather than an unrestricted search for
interesting results.

## Applied AI checklist

For an inference, retrieval, or RAG system, useful empirical distributions may
include:

- document and chunk lengths;
- chunks per document and retrieved-context size;
- similarity or reranking scores, separated by relevance labels when known;
- input/output tokens and estimated request cost;
- latency percentiles by request class and system component;
- error, fallback, abstention, and evaluation scores;
- the same measures over time and across operationally important groups.

Means alone are rarely enough for bounded reliability objectives. Median, P95,
P99, maximum, error rate, and sample count answer different questions. A very
high sample maximum can be real or defective; investigate it rather than
allowing a dashboard to make the decision implicitly.

## Executed synthetic experiment

`example.py` compares identical seeded workloads before and after five
predefined additive latency spikes.

- **Hypothesis:** the spikes will change non-robust and tail-sensitive
  statistics more than the median.
- **Configuration:** seed 42; 1,000 observations; log-normal token counts;
  Poisson chunk counts; latency as a deterministic formula plus Gaussian
  noise; treatment spikes of 1,500–3,000 ms.
- **Result:** mean changed by +11.100 ms, median by +0.096 ms, sample standard
  deviation by +99.315 ms, P99 by +66.595 ms, and excess kurtosis by +151.792.
- **Interpretation:** the result is
  consistent with the robustness properties predicted by the definitions.
  Tail percentiles and a robust center expose different operational aspects of
  this workload.
- **Limitation:** this is one constructed distribution with explicit
  contamination. Its magnitudes and thresholds do not transfer to production.

## Suggested experiments not yet run

- Repeat the spike comparison across sample sizes and seeds; report the
  distribution of each statistic's change.
- Compare IQR, robust z-scores based on MAD, and ordinary z-scores on Gaussian
  and log-normal samples.
- Construct subgroups that exhibit Simpson's paradox, then compare aggregate
  and conditional associations.
- Introduce missingness mechanisms and measure how complete-case filtering
  changes the analyzed population.

These are proposals, not executed evidence or author conclusions.
