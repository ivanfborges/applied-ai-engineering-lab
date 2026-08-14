# Interview Questions

## 1. What does the Central Limit Theorem actually say?

For IID random variables with finite mean \(\mu\) and finite, nonzero variance
\(\sigma^2\), the standardized sample mean
\(\sqrt n(\bar X-\mu)/\sigma\) converges in distribution to a standard normal
random variable. It describes a sampling distribution, not a transformation
of the original observations into normal data.

## 2. How are standard deviation and standard error different?

Standard deviation describes variation among observations. Standard error
describes the repeated-sampling variation of an estimator. For an IID sample
mean, the population standard error is \(\sigma/\sqrt n\), usually estimated
by \(s/\sqrt n\).

## 3. Why does standard error shrink with \(\sqrt n\), not \(n\)?

Independent variances add, so the variance of a sum of \(n\) observations is
\(n\sigma^2\). Dividing the sum by \(n\) to form the mean divides variance by
\(n^2\), leaving \(\sigma^2/n\). Taking its square root gives
\(\sigma/\sqrt n\).

## 4. Is a sample size of 30 enough for the CLT?

There is no universal cutoff. Approximation quality depends on skewness, tail
behavior, dependence, the estimator, and the accuracy required for the
decision. Simulation, diagnostics, finite-sample theory, or an alternative
procedure may be needed.

## 5. What does a 95% confidence interval mean?

Under the model and sampling assumptions, a procedure with 95% coverage will
produce intervals containing the fixed true parameter in approximately 95% of
repeated samples. It does not generally assign a 95% posterior probability to
the parameter being inside one observed interval.

## 6. When should you use a Student-t interval instead of a z interval?

For inference about a mean from a normal population when the variance is
unknown, replacing \(\sigma\) with \(s\) yields a Student-\(t\) statistic with
\(n-1\) degrees of freedom. For large samples, \(t\) and normal critical values
are close. Neither choice repairs skewness, dependence, or biased sampling.

## 7. Why can a correctly calculated narrow interval still be misleading?

It may quantify only sampling variation while ignoring selection bias,
measurement error, label noise, leakage, misspecification, dependence, or
distribution shift. Precision around a biased estimand is not accuracy.

## 8. How would you communicate a model improvement?

Estimate the effect directly—for example, the paired accuracy difference—and
report its interval, confidence level, sample size, evaluation unit, and
method. Then discuss whether the interval includes zero, whether its plausible
effect sizes cross a practical threshold, and whether the evaluation design
supports generalization.

## 9. How would you apply confidence intervals to LLM evaluation?

First define the independent unit: question, conversation, user, document, or
another unit matching deployment. Estimate uncertainty for correctness,
hallucination, retrieval success, or human scores while accounting for
repeated users, evaluator effects, dataset composition, and important slices.
Cluster bootstrap or hierarchical methods may be more suitable than resampling
individual rows.

## 10. When is bootstrap preferable to an analytical CLT interval?

Bootstrap can be useful for an estimator with no convenient analytical
standard error, such as a median or nonlinear metric. It is not
assumption-free: the empirical sample must represent the target process, and
the resampling unit and scheme must preserve clusters, time dependence, or
pairing.

## 11. What is the difference between a confidence interval and a prediction interval?

A confidence interval describes uncertainty about a population quantity such
as a mean. A prediction interval describes uncertainty for a new observation
and includes both parameter uncertainty and observation-level variability, so
it is typically wider.

## 12. Why might 100,000 logged interactions not imply tiny uncertainty?

The rows may come from far fewer independent users, sessions, or documents.
Positive within-group correlation reduces effective information, and temporal
or selection effects can add error that the IID formula does not represent.
The analysis should identify the randomization or sampling unit before using
the row count as \(n\).

## Interview-ready summary

The CLT explains why the standardized error of many sample means approaches a
normal distribution under suitable conditions. That supports standard errors
and confidence intervals, which communicate the precision of an estimate
instead of only its point value. In applied AI, the calculation is usually the
easy part: the critical work is defining the target population, estimand,
sampling unit, dependence structure, and decision threshold. A narrow interval
does not compensate for biased or leaked evaluation data.
