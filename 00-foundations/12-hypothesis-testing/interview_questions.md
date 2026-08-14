# Interview Questions

## 1. What is a p-value?

It is the probability, assuming the null model and analysis assumptions, of a
test statistic at least as extreme as the observed statistic. It is not the
probability that the null hypothesis is true or that the result happened by
chance.

## 2. What does `p > 0.05` mean?

It means the chosen procedure did not find enough evidence to reject at that
threshold. It does not prove equality. I would inspect the confidence interval
to see which meaningful effects remain compatible with the data and review
whether the experiment had adequate power.

## 3. Explain Type I error, Type II error, and power.

A Type I error rejects a true null; its controlled long-run probability is
alpha. A Type II error fails to reject for a specified alternative effect; its
probability is beta. Power is \(1-\beta\), the probability of rejection when
that specified alternative holds.

## 4. Why is power not one fixed number for an experiment?

Power changes with the true effect, variance, sample size, alpha, test
direction, and design. A power statement must name the target effect or effect
curve. "The experiment has 80% power" is incomplete without those conditions.

## 5. Why use a paired test for two models evaluated on the same queries?

Each query has a baseline and candidate score, so the comparison should use
their within-query difference. This preserves the design and removes shared
query difficulty. Treating the samples as independent discards that covariance
and usually answers the question less efficiently.

## 6. What assumptions matter for a paired t test?

The pairs must be correctly matched, the differences across independent units
must be independent, the outcome must support a meaningful mean difference,
and the mean's t reference distribution must be adequate. Normality concerns
the paired differences, not each score vector separately. Sampling and metric
validity remain external assumptions.

## 7. Is a permutation test assumption-free?

No. Its transformations must be valid under the null. A paired sign-flip test
requires an appropriate sign-exchangeability or randomization argument, while
an assignment-based test must follow the actual randomization scheme.

## 8. Why report an effect estimate and interval with a p-value?

The p-value measures evidence relative to a particular null, while the estimate
and interval communicate magnitude and uncertainty. A tiny, precisely measured
effect can be statistically detectable but operationally useless.

## 9. A candidate model has `p = 0.001`. Would you deploy it?

Not from that fact alone. I would inspect the effect and interval, practical
threshold, evaluation-set independence, segment and safety regressions,
latency, cost, reliability, multiplicity, and the assumptions behind the test.

## 10. You tested 50 metrics and three have `p < 0.05`. What is the issue?

Per-test alpha does not control errors across the family. Even under all true
nulls, false discoveries become likely. I would predeclare primary metrics and
apply a family-wise or false-discovery-rate procedure appropriate to the
decision and dependency structure.

## 11. Why is repeatedly peeking at a fixed-horizon test invalid?

The reference distribution assumes the planned stopping rule. Stopping when a
nominal p-value first crosses the threshold creates more opportunities for a
false rejection. A predefined horizon or a valid sequential method is needed.

## 12. How would you evaluate a new RAG reranker statistically?

I would define the query population, experimental unit, primary quality
metric, pairing, minimum useful improvement, alpha, sample size, and stopping
rule before evaluation. Both systems would be scored on an untouched query set.
For a continuous paired score I might use a paired t or justified resampling
analysis and report the difference, interval, p-value, assumptions, and sample
size. Deployment would also require latency, cost, faithfulness, safety, and
segment-level guardrails.

## Interview-ready summary

> Hypothesis testing measures how incompatible observed evidence is with a
> specified null model. I define the estimand, experimental unit, hypotheses,
> test, alpha, and stopping rule before inspecting results. I interpret the
> p-value under the null rather than as the probability the null is true, and I
> pair it with the effect estimate, confidence interval, power assumptions, and
> practical threshold. In AI evaluation, correct pairing, independent units,
> untouched test data, and metric validity are often more important than the
> choice between two reasonable test formulas.
