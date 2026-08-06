# Probability Distributions — Senior Interview Questions

## 1. How do you choose a probability distribution for a target?

Start with support and the data-generating process. A binary target suggests
Bernoulli; successes among a fixed number of trials suggest Binomial; event
counts per exposure suggest Poisson or Negative Binomial; positive,
right-skewed measurements may suggest Log-normal or Gamma.

Then examine independence, exposure, rate stability, conditional variance,
excess zeros, temporal structure, and tail behavior. Compare plausible models
with residual diagnostics, calibration, out-of-sample likelihood, and metrics
aligned with the operational decision. A histogram alone is insufficient.

## 2. What is the relationship between Bernoulli and Binomial?

A Bernoulli variable represents one binary trial with success probability
\(p\). A Binomial variable is the sum of \(n\) independent Bernoulli trials
having the same \(p\):

$$
X_i\sim\operatorname{Bernoulli}(p),\qquad
K=\sum_{i=1}^{n}X_i\sim\operatorname{Binomial}(n,p).
$$

Bernoulli is the special case \(\operatorname{Binomial}(1,p)\).

## 3. Derive the Binomial probability mass function intuitively.

One particular sequence containing \(k\) successes and \(n-k\) failures has
probability \(p^k(1-p)^{n-k}\). There are \(\binom{n}{k}\) ways to choose the
positions of the \(k\) successes. Therefore:

$$
P(K=k)=\binom{n}{k}p^k(1-p)^{n-k}.
$$

This derivation depends on independent trials with a common success
probability.

## 4. When is Poisson a defensible count model?

It is a reasonable baseline for event counts over a known exposure when events
are approximately independent and the conditional rate is stable. Its
conditional mean and variance are both \(\lambda\).

Check for seasonality, bursts, dependence, heterogeneous rates, excess zeros,
and overdispersion. If conditional variance materially exceeds conditional
mean, Negative Binomial or a hierarchical count model may be more appropriate.

## 5. What is overdispersion, and why does it matter?

Overdispersion means the observed conditional variance exceeds what the model
allows. For Poisson, \(\operatorname{Var}(Y\mid X)=E[Y\mid X]\). Hidden
heterogeneity, clustered events, omitted variables, and temporal dependence
can increase variance.

A misspecified Poisson model can underestimate uncertainty, produce narrow
intervals, and make effects look more precise than they are. Diagnose
dispersion conditionally; an unconditional mixture can be overdispersed even
if each subgroup is Poisson.

## 6. Why are Poisson and Exponential related?

In a homogeneous Poisson process, Poisson describes the number of events in an
interval, while Exponential describes the waiting time to the next event. The
event \(T>t\) means zero arrivals have occurred by \(t\):

$$
P(T>t)=P(N(t)=0)=e^{-\lambda t}.
$$

This is the Exponential survival function.

## 7. What does Exponential memorylessness mean in practice?

It means:

$$
P(T>s+t\mid T>s)=P(T>t).
$$

After already waiting \(s\), the remaining wait has the same distribution as a
new wait. This follows from a constant hazard. It is elegant for simple
queueing baselines but often unrealistic for failures, abandonment, or LLM
service time, where risk or completion probability changes with elapsed time.

## 8. How is binary cross-entropy related to Bernoulli?

For a binary target \(y_i\) and predicted probability \(p_i\), the Bernoulli
negative log-likelihood is:

$$
-\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right],
$$

which is binary cross-entropy. Minimizing binary cross-entropy performs
maximum-likelihood estimation of conditional Bernoulli probabilities, subject
to the model and independence assumptions.

## 9. How is mean squared error related to the Normal distribution?

Assume:

$$
Y_i=\hat Y_i+\epsilon_i,\qquad
\epsilon_i\sim\mathcal{N}(0,\sigma^2)
$$

independently with constant variance. The Gaussian negative log-likelihood is,
up to constants and scale, the sum of squared residuals. Minimizing MSE is
therefore maximum likelihood under homoscedastic Gaussian residual noise.

This does not imply that predictors or the marginal target must be Normal.

## 10. Why might latency be Log-normal, and what could break that model?

Latency is positive and may reflect multiplicative factors such as workload,
request size, service load, and downstream delays. Taking logs converts
multiplication into addition, so an approximately Gaussian log latency can be
plausible.

The model can fail when retries, cold starts, cache hits, regions, or distinct
request paths create multiple regimes, or when the upper tail is heavier than
Log-normal. Validate the log-scale fit and decision-relevant tail probabilities
rather than relying only on skewness.

## 11. Why is exponentiating a log-scale prediction potentially biased?

In general:

$$
\exp(E[\log Y\mid X])\ne E[Y\mid X].
$$

Exponentiating a predicted Gaussian log mean gives the conditional median on
the original scale. Under a homoscedastic Log-normal model:

$$
E[Y\mid X]=\exp\left(X^\top\beta+\frac{\sigma^2}{2}\right).
$$

With heteroscedastic or non-Gaussian errors, retransformation may require
conditional variance modeling or a smearing estimator.

## 12. What is the difference between modeling a count and a rate?

A count has meaning only relative to exposure. Five incidents in ten requests
and five in one million requests represent very different rates. In Poisson
regression, exposure \(t_i\) is commonly an offset:

$$
\log\lambda_i=X_i^\top\beta+\log t_i.
$$

This models expected count while normalizing for observation time, traffic,
population, or another opportunity measure.

## 13. How would you validate a distributional assumption?

Combine domain reasoning with:

- empirical versus fitted PMF/PDF;
- Q–Q and P–P plots;
- conditional residual diagnostics;
- observed versus predicted mean–variance behavior;
- zero-frequency and tail-probability checks;
- calibration;
- out-of-sample log-likelihood;
- downstream decision performance.

Do not select solely through a normality test: large samples can reject
irrelevant deviations, and small samples can hide important tail mismatch.

## 14. Why is independence often problematic in AI evaluation?

Examples can be correlated because they share a document, user, conversation,
prompt template, model version, or infrastructure dependency. Five hundred
chunks from twenty documents do not provide the same independent information
as five hundred independent documents.

Ignoring clusters overstates effective sample size and understates
uncertainty. Use grouped splits, cluster-aware bootstrap or standard errors, or
hierarchical models as appropriate.

## 15. How do these distributions affect AI system design?

Bernoulli can model component success; Binomial can model batch-level pass
counts; Poisson can provide an arrival-count baseline; Exponential can provide
an inter-arrival baseline; Normal can model some aggregated estimation
uncertainty; and Log-normal can model some positive operational metrics.

Infrastructure decisions should use the full conditional distribution,
especially peak arrivals and service-time tails. Autoscaling, queue capacity,
timeouts, retry budgets, and SLOs are poorly served by averages alone.

## Interview-Ready Summary

> I choose a distribution from the variable's support and data-generating
> process, then validate its conditional assumptions. Bernoulli models one
> binary outcome, Binomial counts successes in fixed trials, Poisson models
> counts per exposure, and Exponential models waiting times in a homogeneous
> Poisson process. Normal is plausible for some additive symmetric variation,
> while Log-normal is plausible for some positive multiplicative processes.
> These choices define likelihoods and often ML losses, so I check dependence,
> calibration, mean–variance structure, zeros, tails, and out-of-sample fit. In
> production, distribution tails often matter more than averages for capacity,
> latency, and reliability decisions.
