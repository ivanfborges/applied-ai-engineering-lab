# Probability Distributions — Technical Notes

## 1. Intuition

A probability distribution is a model of uncertainty. It specifies:

- the values a random variable can take (its **support**);
- the probability of each value or interval;
- summaries such as expectation, variance, and quantiles;
- assumptions about the process that generated the observations.

Think of distribution families as a vocabulary for distinct mechanisms:

| Question | Candidate distribution | Variable type |
| --- | --- | --- |
| Did one document pass validation? | Bernoulli | Binary |
| How many of 100 documents passed? | Binomial | Bounded count |
| How many requests arrived in one minute? | Poisson | Unbounded count |
| How long until the next request? | Exponential | Positive waiting time |
| How does additive measurement noise vary? | Normal | Real-valued |
| How does positive, multiplicative latency vary? | Log-normal | Positive continuous |

A familiar histogram shape is supporting evidence, not a complete
justification. Support, exposure, dependence, conditional variance, and domain
mechanics are usually more informative.

## 2. Distribution Foundations

### 2.1 Discrete and continuous variables

A discrete variable has a probability mass function (PMF):

$$
p_X(x)=P(X=x), \qquad \sum_x p_X(x)=1.
$$

A continuous variable has a probability density function (PDF):

$$
f_X(x) \ge 0, \qquad \int_{-\infty}^{\infty} f_X(x)\,dx=1.
$$

For a continuous variable, \(P(X=x)=0\). Probability belongs to an interval:

$$
P(a \le X \le b)=\int_a^b f_X(x)\,dx.
$$

A density may exceed one; only its area over an interval must be between zero
and one.

### 2.2 CDF and survival function

The cumulative distribution function is defined for both discrete and
continuous variables:

$$
F_X(x)=P(X\le x).
$$

The survival function focuses on the upper tail:

$$
S_X(x)=P(X>x)=1-F_X(x).
$$

CDFs support quantile calculations and probability plots. Survival functions
are especially useful for latency, reliability, and time-to-event questions.

### 2.3 Expectation and variance

For a discrete variable:

$$
E[X]=\sum_x x\,p_X(x).
$$

For a continuous variable:

$$
E[X]=\int_{-\infty}^{\infty} x f_X(x)\,dx.
$$

Variance can be written in two equivalent forms:

$$
\operatorname{Var}(X)=E[(X-E[X])^2]=E[X^2]-E[X]^2.
$$

Expectation is a long-run average, not necessarily the most likely or typical
value. In a skewed distribution it can differ substantially from the median.

## 3. The Six Distributions

### 3.1 Bernoulli

One binary outcome is modeled as:

$$
X\sim\operatorname{Bernoulli}(p), \qquad X\in\{0,1\}.
$$

Its PMF is:

$$
P(X=x)=p^x(1-p)^{1-x}.
$$

Its moments are:

$$
E[X]=p,\qquad \operatorname{Var}(X)=p(1-p).
$$

The variance is largest at \(p=0.5\). Examples include pass/fail evaluation,
click/no-click, grounded/not-grounded, and tool success/failure.

**Assumptions and limits:** the outcome must be binary. When observations share
users, documents, prompts, or infrastructure, treating them as independent can
understate uncertainty.

### 3.2 Binomial

If \(X_1,\ldots,X_n\) are independent Bernoulli trials with the same \(p\),
their number of successes is:

$$
K=\sum_{i=1}^{n}X_i\sim\operatorname{Binomial}(n,p).
$$

The PMF is:

$$
P(K=k)=\binom{n}{k}p^k(1-p)^{n-k},
\qquad
\binom{n}{k}=\frac{n!}{k!(n-k)!}.
$$

Its moments are:

$$
E[K]=np,\qquad \operatorname{Var}(K)=np(1-p).
$$

The four central assumptions are a fixed number of trials, binary outcomes,
constant \(p\), and independence. Heterogeneous probabilities can motivate a
Beta-Binomial or hierarchical model.

### 3.3 Poisson

A count over a fixed time, space, or exposure can be modeled as:

$$
X\sim\operatorname{Poisson}(\lambda),\qquad X\in\{0,1,2,\ldots\}.
$$

The PMF is:

$$
P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!}.
$$

The mean–variance relationship is restrictive:

$$
E[X]=\operatorname{Var}(X)=\lambda.
$$

For a rate \(r\) and exposure \(t\), the expected count is \(rt\). In Poisson
regression, exposure is commonly included as an offset:

$$
\log E[Y_i\mid X_i]=X_i^\top\beta+\log(t_i).
$$

Classical homogeneous Poisson-process assumptions include independent events,
a constant rate, and independent increments in non-overlapping intervals.
Burstiness, seasonality, correlated incidents, heterogeneous rates, or excess
zeros violate the baseline model.

### 3.4 Exponential

The waiting time between events in a homogeneous Poisson process is:

$$
T\sim\operatorname{Exponential}(\lambda),\qquad t\ge 0.
$$

Its density, CDF, and survival function are:

$$
f(t)=\lambda e^{-\lambda t},
\qquad F(t)=1-e^{-\lambda t},
\qquad S(t)=e^{-\lambda t}.
$$

Its moments are:

$$
E[T]=\frac{1}{\lambda},
\qquad
\operatorname{Var}(T)=\frac{1}{\lambda^2}.
$$

It is memoryless:

$$
P(T>s+t\mid T>s)=P(T>t).
$$

Equivalently, its hazard \(h(t)=\lambda\) is constant. This is often too strong
for service time, hardware failure, user abandonment, or LLM generation
latency. Weibull and Gamma families are more flexible alternatives.

### 3.5 Normal

The Gaussian distribution is:

$$
X\sim\mathcal{N}(\mu,\sigma^2),
$$

with density:

$$
f(x)=\frac{1}{\sigma\sqrt{2\pi}}
\exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right].
$$

It is symmetric and defined over all real values. Standardization gives:

$$
Z=\frac{X-\mu}{\sigma}\sim\mathcal{N}(0,1).
$$

The Normal distribution arises in many sampling distributions through the
Central Limit Theorem, but that does not make raw real-world variables
Gaussian. It is plausible for some additive noise processes and conditional
residuals. It is poor for bounded counts or strongly skewed, positive variables
near zero because it can assign mass to impossible negative values.

Linear regression does not require predictors to be normally distributed.
Classical Gaussian inference concerns the conditional residual distribution,
along with assumptions such as correct specification and constant variance.

### 3.6 Log-normal

A positive variable is Log-normal when:

$$
\log Y\sim\mathcal{N}(\mu,\sigma^2),\qquad Y>0.
$$

Its density is:

$$
f(y)=\frac{1}{y\sigma\sqrt{2\pi}}
\exp\left[-\frac{(\log y-\mu)^2}{2\sigma^2}\right].
$$

Important summaries are:

$$
\operatorname{Median}(Y)=e^\mu,
$$

$$
E[Y]=e^{\mu+\sigma^2/2},
$$

$$
\operatorname{Var}(Y)
=\left(e^{\sigma^2}-1\right)e^{2\mu+\sigma^2}.
$$

Here, \(\mu\) is the mean on the log scale, not the arithmetic mean of \(Y\).
Multiplicative positive effects become additive after taking logs, which can
make this family plausible for file size, duration, cost, or latency.

Right skew alone does not prove Log-normality. Production latency may be
multimodal or heavier-tailed because of distinct request paths, retries, cold
starts, and queueing regimes.

## 4. Relationships and Approximations

### Bernoulli to Binomial

Bernoulli is \(\operatorname{Binomial}(1,p)\), and a Binomial variable is a sum
of independent, identically distributed Bernoulli variables.

### Binomial to Poisson

When \(n\) is large, \(p\) is small, and \(np=\lambda\) remains fixed:

$$
\operatorname{Binomial}(n,p)\approx\operatorname{Poisson}(\lambda).
$$

This is useful for rare events across many opportunities, but not when event
probabilities vary materially or trials are dependent.

### Poisson to Exponential

Let \(N(t)\) be a Poisson process with rate \(\lambda\), and let \(T\) be the
time to its first event. Then:

$$
P(T>t)=P(N(t)=0)=e^{-\lambda t},
$$

which is the Exponential survival function. Poisson models **how many** events;
Exponential models **how long until** an event.

### Normal to Log-normal

If \(X\) is Normal, then \(Y=e^X\) is Log-normal. Normal variation is additive
on its own scale; Log-normal variation is additive on the log scale and
multiplicative on the original scale.

## 5. Likelihood and Machine-Learning Objectives

For independent observations \(x_1,\ldots,x_n\), likelihood is:

$$
L(\theta)=\prod_{i=1}^{n}p(x_i\mid\theta),
\qquad
\ell(\theta)=\sum_{i=1}^{n}\log p(x_i\mid\theta).
$$

Logs turn products into sums and improve numerical stability.

### Bernoulli and binary cross-entropy

For \(y_i\in\{0,1\}\) and predicted probability \(p_i\):

$$
-\ell
=-\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

This is binary cross-entropy. Minimizing it is Bernoulli maximum likelihood
under conditional independence.

### Gaussian residuals and squared error

If:

$$
y_i=\hat y_i+\epsilon_i,\qquad
\epsilon_i\sim\mathcal{N}(0,\sigma^2),
$$

with constant \(\sigma^2\), the negative log-likelihood differs from the sum of
squared residuals only by constants and scaling. MSE therefore embeds a
homoscedastic Gaussian residual assumption.

### Poisson counts

Poisson regression models a positive conditional rate using a log link:

$$
\log\lambda_i=X_i^\top\beta.
$$

This respects non-negative integer targets and lets variance change with the
conditional mean, unlike ordinary least squares.

## 6. Maximum-Likelihood Estimates

For independent, uncensored observations:

| Distribution | MLE |
| --- | --- |
| Bernoulli | \(\hat p=\bar x\) |
| Binomial, known \(n\) | \(\hat p=\bar k/n\) |
| Poisson | \(\hat\lambda=\bar x\) |
| Exponential | \(\hat\lambda=1/\bar t\) |
| Normal mean | \(\hat\mu=\bar x\) |
| Normal variance | \(\hat\sigma^2=n^{-1}\sum_i(x_i-\bar x)^2\) |
| Log-normal | fit Normal MLEs to \(\log x_i\) |

The Normal variance MLE divides by \(n\). The common unbiased sample variance
divides by \(n-1\); these estimators answer different criteria.

## 7. Choosing and Validating a Distribution

Use a staged decision:

1. **Support:** binary, bounded count, non-negative count, positive continuous,
   or unrestricted real.
2. **Mechanism:** fixed trials, arrivals per exposure, waiting time, additive
   effects, or multiplicative effects.
3. **Conditional assumptions:** independence, constant probability/rate,
   mean–variance relationship, zeros, and tail behavior.
4. **Diagnostics:** empirical versus fitted PMF/PDF, Q–Q or P–P plots,
   probability calibration, residuals, and observed versus expected zero rate.
5. **Predictive validation:** out-of-sample log-likelihood and metrics tied to
   the downstream decision.

Do not rely on a normality test alone. With a large sample it can reject
operationally minor deviations; with a small sample it can miss consequential
tail differences.

## 8. Trade-offs and Alternatives

| Baseline | Limitation | Possible alternative |
| --- | --- | --- |
| Binomial | \(p\) varies across groups | Beta-Binomial or hierarchical Binomial |
| Poisson | Variance exceeds mean | Negative Binomial |
| Poisson | More zeros than expected | Zero-inflated or hurdle model |
| Poisson | Rate changes over time | Non-homogeneous process or time-series model |
| Exponential | Hazard changes with time | Weibull or Gamma |
| Normal | Symmetric but heavy tails | Student's \(t\) or robust methods |
| Normal | Positive and right-skewed target | Log-normal or Gamma |
| Log-normal | Multiple regimes or very heavy tails | Mixture, survival, or empirical model |

An unconditional mean–variance mismatch can arise from a mixture of subgroups
even when each subgroup is well described conditionally. Segment and model
known sources of heterogeneity before choosing a more complex family.

## 9. Applications in Data Science and AI Engineering

- **Bernoulli:** groundedness, validation pass, click, successful tool call.
- **Binomial:** pass counts in a fixed evaluation set or quality-control batch.
- **Poisson:** incidents per service-hour, requests per interval, malformed
  inputs per exposure.
- **Exponential:** baseline inter-arrival simulation and simple queueing models.
- **Normal:** conditional residuals, measurement error, and uncertainty of
  aggregated estimators under suitable conditions.
- **Log-normal:** candidate model for positive latency, processing time, file
  size, token use, and workflow cost.

For capacity planning, a homogeneous Poisson/Exponential baseline is often only
a starting point. Seasonality, peaks, queue length, concurrency, dependencies,
and service-time tails influence the actual system.

## 10. Limitations and Common Mistakes

- Choosing a distribution only from a histogram.
- Confusing one Bernoulli trial with a Binomial success count.
- Treating correlated chunks from one document as independent observations.
- Pooling customers, languages, or model versions despite different rates.
- Ignoring exposure when comparing counts.
- Applying Poisson despite overdispersion, burstiness, or zero inflation.
- Treating a PDF height as a point probability.
- Using a Normal model that assigns material probability to impossible values.
- Assuming Exponential memorylessness without examining the hazard or process.
- Calling all positive skewed data Log-normal.
- Exponentiating a predicted log mean and interpreting it as an arithmetic
  mean. Under a homoscedastic Log-normal model:

  $$
  E[Y\mid X]=\exp\left(X^\top\beta+\frac{\sigma^2}{2}\right).
  $$

- Fitting transformations or distribution parameters on train and test data
  together, which leaks information.
- Reporting only the mean for skewed operational data. Median, p90, p95, p99,
  segment-level metrics, and tail expectations can drive SLO decisions.

## 11. Suggested Experiments

1. Compare \(\operatorname{Binomial}(1000,0.005)\) with
   \(\operatorname{Poisson}(5)\) across their PMFs.
2. Mix Poisson samples with rates 2 and 10 and observe unconditional
   overdispersion.
3. Compare mean, median, p95, p99, and negative-value frequency for Normal and
   Log-normal synthetic latency candidates.
4. Verify Exponential memorylessness by comparing
   \(P(T>5\mid T>3)\) with \(P(T>2)\), then repeat with Log-normal samples.
5. Simulate calibrated Bernoulli outcomes, distort the predicted
   probabilities, and compare calibration and log loss at similar accuracy.

## 12. Visual Laboratory Map

The executable visual laboratory connects each mathematical statement to an
observable behavior:

- `static_visualizations.py` compares theoretical PMFs/PDFs with deterministic
  synthetic samples, isolates parameter effects, and emphasizes moments,
  location summaries, and upper tails.
- `generate_animations.py` shows probability mass moving continuously and makes
  the Binomial-to-Normal and Binomial-to-Poisson limiting relationships visible.
- `interactive_dashboard.py` provides immediate parameter manipulation,
  empirical/theoretical diagnostics, distribution relationships, and clearly
  labeled synthetic production scenarios.
- `notebook.ipynb` is a short experiment path for changing code and recording
  observations.

### What to inspect visually

1. For a Bernoulli variable, variance peaks at \(p=0.5\), when neither outcome
   dominates.
2. For a Binomial variable, \(p=0.5\) produces symmetry; probabilities near zero
   or one produce skew relative to the bounded support.
3. For Poisson, both center and spread grow with \(\lambda\), while
   \(P(X=0)=e^{-\lambda}\) falls rapidly.
4. For Exponential, the PDF and survival function change with \(\lambda\), but
   the hazard remains horizontal at \(h(t)=\lambda\).
5. For Normal, increasing \(\sigma\) lowers and widens the density because the
   total area must remain one.
6. For Log-normal, increasing log-scale variance separates mode, median, and
   mean and stretches the upper tail.
7. In the median-matched latency comparison, p95 and p99 can differ materially
   even when central behavior looks similar.

### Reading the 3D surfaces

The Plotly surfaces encode a probability function over a parameter:

- Normal: density over value and either \(\sigma\) or \(\mu\);
- Log-normal: density over positive value and log-scale spread;
- Binomial: probability mass over success count and \(p\), with fixed \(n\);
- Poisson: probability mass over event count and \(\lambda\).

The vertical axis remains probability density or mass. Rotation helps inspect
the parameter trajectory; it does not add a new statistical variable.
