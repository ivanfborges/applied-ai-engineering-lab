# Probability Essentials — Senior Interview Questions

## 1. What is conditional probability?

Conditional probability is the probability of an event within the restricted
population where another event occurred:

\[
P(A\mid B)=\frac{P(A\cap B)}{P(B)},\qquad P(B)>0.
\]

The denominator is the conditioning event. In production analysis, useful
examples include fraud given an alert, latency given a cache miss, and answer
correctness given successful retrieval.

## 2. Why are \(P(A\mid B)\) and \(P(B\mid A)\) different?

They use different reference populations. In fraud detection,
\(P(\text{alert}\mid\text{fraud})\) is recall, whereas
\(P(\text{fraud}\mid\text{alert})\) is positive predictive value. Bayes'
theorem relates them through the prior and marginal evidence, but they are
generally unequal.

## 3. Explain the base-rate fallacy.

It is the error of interpreting evidence while neglecting how prevalent the
hypothesis was before that evidence. A sensitive detector can generate mostly
false-positive alerts when the positive class is rare because the large
negative population contributes many false positives. The posterior must
include the prior:

\[
P(H\mid E)=\frac{P(E\mid H)P(H)}{P(E)}.
\]

## 4. What does independence mean, and how would you challenge it in a system design?

Events are independent when \(P(A\cap B)=P(A)P(B)\), equivalently when
conditioning on one does not change the other's probability. In a system
design, I would look for shared regions, networks, databases, credentials,
deployment pipelines, traffic patterns, and upstream providers. Any common
cause can correlate failures and invalidate a simple product of marginal
reliabilities.

## 5. Are mutually exclusive events independent?

Not when both have positive probability. Mutual exclusivity gives
\(P(A\cap B)=0\), while independence requires
\(P(A\cap B)=P(A)P(B)>0\). Observing one exclusive event makes the other
impossible, so the first event changes the probability of the second.

## 6. What is conditional independence?

\(A\) and \(B\) are conditionally independent given \(C\) when

\[
P(A\cap B\mid C)=P(A\mid C)P(B\mid C).
\]

Variables can be dependent overall because of a shared cause but independent
within strata of that cause. Conditional independence enables factorization in
Naive Bayes and graphical models, but it must be justified rather than assumed
from weak marginal correlation.

## 7. What is expected value, and can it be an impossible outcome?

Expected value is a probability-weighted long-run average:

\[
\mathbb{E}[X]=\sum_x xP(X=x)
\]

for a discrete variable. It need not be a possible or typical outcome. A fair
die has expectation 3.5, and a skewed loss distribution can have a mean far
above its median.

## 8. Why does linearity of expectation not require independence?

Expectation is a linear operator:

\[
\mathbb{E}[aX+bY]=a\mathbb{E}[X]+b\mathbb{E}[Y].
\]

The result follows from the linearity of sums or integrals. Dependence matters
for variance because \(\operatorname{Var}(X+Y)\) contains a covariance term.

## 9. Why is variance needed if the mean is known?

The mean describes center; variance describes dispersion. Two services can have
the same average latency but very different consistency and tail behavior.
Variance is still incomplete for skewed or heavy-tailed data, so I would also
inspect percentiles, worst cases, subgroup behavior, and the frequency of
operationally severe events.

## 10. What is the difference between population and sample variance?

Population-style variance divides squared deviations by \(n\). The conventional
unbiased estimator of population variance divides by \(n-1\), applying Bessel's
correction because the sample mean is estimated from the same data. In code, I
would make the convention explicit because NumPy and pandas have different
defaults.

## 11. Does zero covariance imply independence?

No. Independence implies zero covariance when the required expectations exist,
but zero covariance rules out only linear co-movement. Nonlinear dependence may
remain. Jointly Gaussian variables are an important special case where zero
covariance does imply independence.

## 12. Why is a classification threshold of 0.5 often inappropriate?

The threshold should reflect false-positive and false-negative costs,
probability calibration, class prevalence, review capacity, subgroup impact,
and policy constraints. With calibrated probability \(p\), review cost \(C_R\),
and missed-fraud cost \(C_M\), a simplified rule reviews when
\(p>C_R/C_M\), not necessarily when \(p>0.5\).

## 13. How would you determine whether model probabilities are trustworthy?

I would evaluate reliability diagrams, calibration curves, Brier score, and log
loss on production-like data. I would check subgroup and temporal calibration,
not only one aggregate curve. I would also verify that the score has a
probabilistic interpretation and monitor for prevalence or likelihood shift
after deployment.

## 14. A classifier has 99% accuracy. Is it good?

There is not enough information. If the positive class prevalence is 1%, an
always-negative classifier also has 99% accuracy. I would inspect the confusion
matrix, precision, recall, calibration, decision costs, subgroup performance,
validation design, and the metric that matches the operational goal.

## 15. How does Bayes' theorem connect to distribution shift?

The posterior is proportional to likelihood times prior:

\[
P(H\mid E)\propto P(E\mid H)P(H).
\]

If deployment prevalence changes, the prior changes and old posteriors can
become miscalibrated even if class-conditional detector behavior is stable. If
the likelihoods also shift, a prior-only correction is insufficient. I would
monitor both prevalence and conditional performance and recalibrate or retrain
as evidence warrants.

## 16. How would you apply probability to a RAG pipeline?

I would treat retrieval, ranking, grounded generation, and validation as
uncertain stages. Conditional evaluation can estimate answer correctness given
good retrieval separately from retrieval success, revealing whether the
bottleneck is retrieval or generation. I would not treat cosine similarity,
reranker outputs, or an LLM's self-reported confidence as calibrated
probabilities without empirical validation.

## 17. What is the difference between probability and likelihood?

For fixed parameters, \(P(\text{data}\mid\theta)\) is a probability model over
possible data. With observed data fixed, the same expression viewed as a
function of \(\theta\) is a likelihood. A likelihood need not be normalized over
the parameter space and should not automatically be interpreted as a posterior
probability.

## 18. When would you prefer simulation to an exact probability calculation?

I would use simulation when dependencies, repeated paths, or system rules make
an analytical calculation unwieldy. It is useful for sensitivity analysis and
tail-risk experiments, but it introduces Monte Carlo error. I would use enough
draws for the event rarity, report the seed and sample size, quantify simulation
uncertainty, and compare with an exact result whenever one is available.

## Interview-ready summary

Probability is the formal language for uncertainty. Events describe outcomes,
conditioning restricts the relevant population after evidence, and
independence determines whether joint probabilities can be factorized. Expected
value summarizes long-run average outcomes, while variance quantifies
instability around that average.

Bayes' theorem combines a prior with the likelihood of evidence to produce a
posterior. In production, I would not trust that posterior from formulas alone:
I would validate base rates, calibration, sampling, dependence, temporal
stability, and the costs that turn probabilities into decisions.

