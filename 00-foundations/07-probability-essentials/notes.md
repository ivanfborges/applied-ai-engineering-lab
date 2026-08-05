# Probability Essentials — Technical Notes

## 1. Intuition: uncertainty after evidence

Before observing a transaction, an investigator has a base probability that it
is fraudulent. After a detector raises an alert, the relevant question changes
from the marginal probability

\[
P(F)
\]

to the conditional probability

\[
P(F \mid A).
\]

Detector documentation often reports \(P(A \mid F)\), the probability of an
alert given fraud. That is sensitivity or recall, not the probability of fraud
given an alert. The direction of conditioning matters because each expression
uses a different reference population.

Probability formalizes this process of restricting a population, incorporating
evidence, and quantifying remaining uncertainty. Expected value then summarizes
the long-run average outcome, while variance measures how widely outcomes can
move around that average.

## 2. Sample spaces and events

A random experiment is a process whose outcome is uncertain from the modeler's
perspective. Its **sample space** \(\Omega\) contains all possible outcomes. An
**event** is a subset of \(\Omega\).

For a binary detector:

\[
\Omega = \{\text{alert}, \text{no alert}\}.
\]

For events \(A\) and \(B\):

- \(A \cup B\): at least one of the events occurs;
- \(A \cap B\): both events occur;
- \(A^c\): \(A\) does not occur;
- \(A \setminus B\): \(A\) occurs without \(B\).

A probability measure satisfies:

\[
P(A) \ge 0,\qquad P(\Omega)=1,
\]

and countable additivity for disjoint events. For two events, the general
addition rule is

\[
P(A \cup B) = P(A) + P(B) - P(A \cap B).
\]

The intersection is subtracted because it appears once in each marginal term.

## 3. Conditional probability

For \(P(B)>0\),

\[
P(A \mid B) = \frac{P(A \cap B)}{P(B)}.
\]

Conditioning replaces the original sample space with the outcomes in which
\(B\) occurred. Rearranging gives the multiplication rule:

\[
P(A \cap B) = P(A \mid B)P(B)
             = P(B \mid A)P(A).
\]

### A denominator check

Suppose 1,000 transactions include 100 alerts and 20 alerted frauds. Then

\[
P(F \mid A)=\frac{20}{100}=0.20,
\]

not \(20/1000\). A useful debugging habit is to state the denominator in words:
"among alerted transactions" in this example.

## 4. Independence and mutual exclusivity

Events \(A\) and \(B\) are independent when learning that one occurred does not
change the probability of the other:

\[
P(A \mid B)=P(A).
\]

Equivalently, when the relevant probabilities are defined,

\[
P(A \cap B)=P(A)P(B).
\]

Mutually exclusive events instead satisfy

\[
P(A \cap B)=0.
\]

If both events have positive probability, they cannot be both mutually
exclusive and independent. Observing one mutually exclusive event makes the
other impossible, so it provides maximal information rather than no
information.

### Conditional independence

Two events may be dependent marginally but independent after conditioning on a
third variable \(C\):

\[
P(A \cap B \mid C)=P(A \mid C)P(B \mid C).
\]

This distinction is central to Naive Bayes, Bayesian networks, and causal
reasoning. For example, umbrella use and traffic delays may be associated
because rain affects both; conditioning on weather can reduce that association.

### Pairwise versus mutual independence

For three events, checking each pair is not sufficient for mutual independence.
Mutual independence also requires

\[
P(A \cap B \cap C)=P(A)P(B)P(C),
\]

and the corresponding factorization for every subset. This matters when a
collection of signals shares a higher-order dependency that pairwise checks do
not reveal.

## 5. Total probability and Bayes' theorem

If \(B_1,\ldots,B_n\) form a partition—exactly one of them occurs—then

\[
P(A)=\sum_{i=1}^{n}P(A \mid B_i)P(B_i).
\]

For fraud versus legitimate transactions:

\[
P(A)=P(A \mid F)P(F)+P(A \mid F^c)P(F^c).
\]

Bayes' theorem follows by writing the same intersection in two directions:

\[
P(F \mid A)
=\frac{P(A \mid F)P(F)}{P(A)}.
\]

Substituting the binary total-probability expansion:

\[
P(F \mid A)
=
\frac{P(A \mid F)P(F)}
{P(A \mid F)P(F)+P(A \mid F^c)P(F^c)}.
\]

The components have distinct roles:

- **prior** \(P(F)\): belief or prevalence before the current evidence;
- **likelihood** \(P(A \mid F)\): compatibility of the evidence with fraud;
- **evidence** \(P(A)\): marginal probability of observing an alert;
- **posterior** \(P(F \mid A)\): updated probability after the alert.

### Base-rate example

Let

\[
P(F)=0.01,\qquad
P(A \mid F)=0.90,\qquad
P(A \mid F^c)=0.05.
\]

Then

\[
P(A)=0.90(0.01)+0.05(0.99)=0.0585
\]

and

\[
P(F \mid A)=\frac{0.90(0.01)}{0.0585}\approx0.1538.
\]

Although the detector finds 90% of frauds, only about 15.38% of its alerts are
fraudulent under this population. The many legitimate transactions create more
false positives than the rare fraudulent transactions create true positives.

## 6. Random variables and distributions

A random variable maps outcomes to numbers. A binary event can be represented
by an indicator:

\[
X =
\begin{cases}
1, & \text{if the event occurs},\\
0, & \text{otherwise}.
\end{cases}
\]

For a discrete variable, the probability mass function is

\[
p_X(x)=P(X=x), \qquad \sum_x p_X(x)=1.
\]

For a continuous variable with density \(f_X\), probability is assigned to
intervals:

\[
P(a\le X\le b)=\int_a^b f_X(x)\,dx.
\]

A density value is not itself a probability and may exceed 1; its integral over
the full support is 1. For a continuous distribution, \(P(X=x)=0\) at any
single point even though values near \(x\) can occur.

The cumulative distribution function works for both cases:

\[
F_X(x)=P(X\le x).
\]

## 7. Expected value

For a discrete random variable,

\[
\mathbb{E}[X]=\sum_x xP(X=x).
\]

For a continuous random variable,

\[
\mathbb{E}[X]=\int_{-\infty}^{\infty}x f_X(x)\,dx,
\]

provided the expectation exists.

Expected value is a long-run average, not necessarily a likely value. A fair
six-sided die has expectation 3.5, which cannot appear on one roll.

For \(X\sim\operatorname{Bernoulli}(p)\),

\[
\mathbb{E}[X]=1p+0(1-p)=p.
\]

Linearity of expectation does not require independence:

\[
\mathbb{E}[aX+bY]=a\mathbb{E}[X]+b\mathbb{E}[Y].
\]

This makes aggregate expected cost or event counts easier to calculate even
when their components are dependent.

### Conditional and total expectation

\(\mathbb{E}[X\mid Y=y]\) is the expected value within the subgroup defined by
\(Y=y\). Overall expectation can be recovered through

\[
\mathbb{E}[X]=\mathbb{E}\!\left[\mathbb{E}[X\mid Y]\right].
\]

For example, overall latency can be decomposed into cache-hit and cache-miss
latencies, weighted by the probability of each path. This is often more
diagnostic than one global average.

## 8. Variance, standard deviation, and covariance

Let \(\mu=\mathbb{E}[X]\). Population variance is

\[
\operatorname{Var}(X)=\mathbb{E}[(X-\mu)^2].
\]

Expanding the square gives the computational identity

\[
\operatorname{Var}(X)
=\mathbb{E}[X^2]-\mathbb{E}[X]^2.
\]

The identity is algebraically useful, although the direct one-pass computation
can be numerically unstable when values are large and their variance is small.
Tested libraries use more careful implementations.

Standard deviation returns to the original unit:

\[
\sigma=\sqrt{\operatorname{Var}(X)}.
\]

For a Bernoulli variable:

\[
\operatorname{Var}(X)=p(1-p),
\]

which is maximal at \(p=0.5\) and approaches zero near 0 or 1.

Under an affine transformation:

\[
\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X).
\]

Adding a constant shifts the mean without changing spread; scaling by \(a\)
scales squared deviations by \(a^2\).

Covariance measures linear co-movement:

\[
\operatorname{Cov}(X,Y)
=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])].
\]

Consequently,

\[
\operatorname{Var}(X+Y)
=\operatorname{Var}(X)+\operatorname{Var}(Y)
+2\operatorname{Cov}(X,Y).
\]

Independence implies zero covariance when the moments exist, but zero covariance
does not generally imply independence. Nonlinear dependence can remain hidden.

### Population versus sample variance

For observations \(x_1,\ldots,x_n\), the population-style calculation is

\[
\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2.
\]

The conventional unbiased estimator of population variance is

\[
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2.
\]

The \(n-1\) denominator applies Bessel's correction because the sample mean was
estimated from the same observations. In NumPy, `np.var(x, ddof=0)` uses \(n\)
and `np.var(x, ddof=1)` uses \(n-1\); pandas `Series.var()` uses \(n-1\) by
default.

## 9. From probability to decisions

Probabilities describe uncertainty; utilities and costs select actions. Suppose
the posterior fraud probability is \(p\), manual review costs \(C_R\), and
missing a fraud costs \(C_M\). Under a simplified model:

\[
\text{cost(no review)}=pC_M,\qquad
\text{cost(review)}=C_R.
\]

Review when

\[
p > \frac{C_R}{C_M}.
\]

This threshold is only as trustworthy as the probability calibration and cost
model. Real policies may also include investigator capacity, delayed outcomes,
legal requirements, unequal subgroup impact, and a nonzero cost for fraud that
passes review.

## 10. Assumptions and trade-offs

### Representative and stable probabilities

Empirical probabilities estimate a particular population and time period.
Selection bias, seasonality, product changes, and adversarial adaptation can
change the prior or likelihoods. A valid historical posterior may be
miscalibrated after deployment.

### Independence

Independence simplifies joint probabilities and variance calculations, but it
is often the most fragile assumption. Services can share regions, databases,
credentials, deployments, or upstream APIs. Features can encode the same
underlying signal. Multiplying marginal probabilities in these settings can
underestimate systemic risk or double-count evidence.

### Mean versus robustness

Expected value is convenient and additive, but rare extreme outcomes can
dominate it. For latency, cost, and loss, pair the mean with median, upper
percentiles, dispersion, and explicit tail-risk measures.

### Monte Carlo versus exact calculation

Simulation handles complex dependencies and provides intuition, but introduces
sampling error and can miss very rare events without enough draws. Exact
calculation is preferable when the event structure is tractable. When using
simulation, report the seed, number of draws, and uncertainty—not just one
realization.

### Probability versus likelihood

The expression \(P(\text{data}\mid\theta)\) is a probability model over possible
data for fixed \(\theta\). Once data are observed and the expression is viewed
as a function of \(\theta\), it is a likelihood. A likelihood need not sum or
integrate to 1 over \(\theta\).

## 11. Applications in AI engineering

- **Classification:** estimate \(P(Y=1\mid X=x)\) for ranking, thresholding,
  review routing, or expected-loss minimization.
- **Calibration:** check whether cases assigned probability 0.8 are positive
  approximately 80% of the time in deployment-like data.
- **RAG evaluation:** separate \(P(\text{correct answer}\mid\text{good
  retrieval})\) from \(P(\text{good retrieval})\) to locate pipeline
  bottlenecks. Similarity and reranker scores are not automatically
  probabilities.
- **Reliability engineering:** model shared causes explicitly instead of
  multiplying component availability estimates without justification.
- **A/B testing:** interpret observed average differences together with their
  sampling variability and experimental design.
- **Cost optimization:** combine cache paths, model routing, retries, token
  costs, and failure costs using total expectation.

## 12. Limitations

Probability calculations quantify uncertainty under a model; they do not repair
bad assumptions. Results can be misleading when:

- the sample does not represent the decision population;
- labels or features leak future information;
- dependence is omitted;
- estimates are poorly calibrated;
- the process changes over time;
- important outcomes or costs are excluded;
- the problem is causal but only associational probabilities are modeled;
- rare-event data are too sparse for reliable estimates.

High-stakes use also requires validation, monitoring, auditability, uncertainty
communication, and appropriate human oversight.

## 13. Common mistakes

1. **Reversing the conditional.** \(P(A\mid F)\) is not \(P(F\mid A)\).
2. **Ignoring the base rate.** Sensitivity alone cannot determine positive
   predictive value.
3. **Using the wrong denominator.** Conditioning restricts the population.
4. **Confusing exclusivity with independence.** Disjoint nonzero events are
   necessarily dependent.
5. **Treating a density as a probability.** Continuous probabilities are areas
   over intervals.
6. **Calling any score a probability.** Logits, similarity scores, and model
   confidence require a probabilistic interpretation and calibration evidence.
7. **Reporting only the mean.** Dispersion, subgroup behavior, and tail events
   may control the operational risk.
8. **Mixing variance conventions.** State whether the divisor is \(n\) or
   \(n-1\).
9. **Assuming zero covariance means independence.** It rules out linear
   co-movement, not every dependence.
10. **Defaulting to a 0.5 threshold.** The decision boundary should reflect
    costs, capacity, calibration, and policy.

## 14. Suggested experiments

1. Run `example.py` with fraud rates of 0.001, 0.01, and 0.10. Keep detector
   behavior fixed and compare \(P(F\mid A)\).
2. Reduce the false-positive rate from 0.05 to 0.01. For rare targets, this may
   improve alert precision more than a small recall increase.
3. Re-run with smaller transaction counts and different seeds. Observe Monte
   Carlo variability around the exact posterior.
4. Simulate two service failures first independently and then with a shared
   regional-outage event. Compare the empirical joint failure rates.
5. Generate log-normal latency and compare mean, median, standard deviation,
   and the 90th, 95th, and 99th percentiles.

