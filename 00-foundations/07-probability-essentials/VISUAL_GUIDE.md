# Visual Probability Lab — Study Guide

This guide explains the interactive and generated visuals in
`visual_lab.py`. Every scenario uses synthetic data. Run
`python generate_visual_assets.py` before opening the application if you want
all GIF and PNG previews to be available.

## Event operations grid

### Concept

The 100 cells form the sample space \(\Omega=\{1,\ldots,100\}\). Event \(A\)
contains values divisible by 2 and event \(B\) contains values divisible by 5.
Marker shape identifies membership; size and opacity identify the selected
operation.

### What the controls change

The operation selector switches among \(A\), \(B\), \(A\cup B\), \(A\cap B\),
\(A^c\), and \(B^c\). The underlying sample space and event definitions stay
fixed, which makes the operations directly comparable.

### What to observe

The intersection appears in both events. It must be subtracted once when the
union is computed because it was included in both marginal counts.

### Mathematical interpretation

\[
P(A\cup B)=P(A)+P(B)-P(A\cap B).
\]

For this sample space, \(P(A)=0.50\), \(P(B)=0.20\),
\(P(A\cap B)=0.10\), and \(P(A\cup B)=0.60\).

### Production connection

Event algebra underlies cohort construction, monitoring filters, eligibility
rules, and compound model-evaluation metrics.

### Interview takeaway

Only omit the intersection term when the events are known to be mutually
exclusive.

## Conditional-probability denominator animation

### Concept

Conditioning changes the reference population. The animation starts from
\(\Omega\), fades observations outside \(B\), and then emphasizes the subset
that is also in \(A\).

### What the controls change

The application controls total population, counts in \(A\), counts in \(B\),
and counts in \(A\cap B\). Invalid intersections are rejected using the count
form of the Fréchet bounds.

### What to observe

The denominator changes from the full population to \(|B|\). The numerator is
then \(|A\cap B|\), not the count of all observations in \(A\).

### Mathematical interpretation

\[
P(A\mid B)=\frac{P(A\cap B)}{P(B)},\qquad P(B)>0.
\]

The valid intersection range is

\[
\max(0, |A|+|B|-N)\le |A\cap B|\le\min(|A|,|B|).
\]

### Production connection

Useful conditional views include fraud given an alert, latency given a cache
miss, and answer correctness given successful retrieval.

### Interview takeaway

State the denominator in words. This prevents both denominator mistakes and
reversing \(P(A\mid B)\) with \(P(B\mid A)\).

## Independence comparison

### Concept

The visual compares a 2×2 joint distribution under negative dependence,
independence, and positive dependence while holding the marginal probabilities
fixed.

### What the controls change

Users set \(P(A)\), \(P(B)\), the dependence parameter \(\delta\), sample size,
and random seed. The application restricts \(\delta\) to the mathematically
valid interval implied by the Fréchet bounds.

### What to observe

At \(\delta=0\), the theoretical intersection equals the product of the
marginals. Empirical simulation approaches the theoretical table as sample size
grows.

### Mathematical interpretation

\[
P(A\cap B)=P(A)P(B)+\delta.
\]

Independence corresponds to \(\delta=0\). Mutual exclusivity instead requires
\(P(A\cap B)=0\), which is generally a dependence relationship.

### Production connection

Shared regions, databases, credentials, and deployment pipelines introduce
common failure causes. Multiplying component reliabilities can therefore be
optimistic.

### Interview takeaway

Zero covariance does not generally prove independence; it only rules out linear
co-movement.

## Bayes base-rate population animation

### Concept

The fraud detector's true-positive and false-positive rates remain fixed while
the prior fraud prevalence changes. The grid distinguishes true positives,
false positives, false negatives, and true negatives with marker shapes,
labels, and colors.

### What the controls change

The interactive version controls prior prevalence, true-positive rate,
false-positive rate, and population size. The generated animation varies only
the prior so the base-rate effect is isolated.

### What to observe

When fraud is rare, false positives from the large legitimate population can
outnumber true positives. As prevalence rises, the posterior share of fraud
among alerts rises even though detector behavior is unchanged.

### Mathematical interpretation

\[
P(F\mid A)=
\frac{P(A\mid F)P(F)}
{P(A\mid F)P(F)+P(A\mid F^c)P(F^c)}.
\]

The denominator is the total alert probability: true-positive probability plus
false-positive probability.

### Production connection

Alert precision changes when deployment prevalence changes. Monitoring only
recall and false-positive rate is insufficient.

### Interview takeaway

\(P(A\mid F)\) is detector recall; it is not \(P(F\mid A)\).

## Bayes 3D surface

### Concept

The surface maps prior prevalence and false-positive rate to posterior alert
precision. Height is a real probability variable, so the third dimension has a
mathematical role rather than being decorative.

### What the controls change

The true-positive-rate slider regenerates the surface. The selected prior and
false-positive rate move a labeled scenario marker across it.

### What to observe

The surface is steep for rare events: small changes in prevalence or
false-positive rate can produce large posterior changes. Reducing false
positives is particularly important at low prevalence.

### Mathematical interpretation

For fixed TPR \(t\), prior \(\pi\), and false-positive rate \(f\):

\[
\operatorname{posterior}=
\frac{t\pi}{t\pi+f(1-\pi)}.
\]

Undefined zero-denominator cells are handled safely rather than divided by
zero.

### Production connection

The surface is a sensitivity-analysis tool for prevalence shift, detector
tuning, and operational alert volume.

### Interview takeaway

A posterior is conditional on both model behavior and population composition.
It is not an intrinsic property of the model.

## Expected-value balance point

### Concept

Bars show probability mass, labels show each weighted contribution
\(xP(X=x)\), and the vertical line or fulcrum marks the expected value.

### What the controls change

Users choose a preset—fair die, biased die, Bernoulli, synthetic profit/loss,
or synthetic AI request cost—and can edit both values and probabilities.
Invalid distributions are explained instead of plotted.

### What to observe

The fair die balances at 3.5 even though 3.5 cannot be rolled. In skewed
distributions, a rare extreme outcome can move the expected value away from the
most probable outcome.

### Mathematical interpretation

\[
\mathbb{E}[X]=\sum_x xP(X=x).
\]

Every bar contributes its outcome multiplied by its probability.

### Production connection

Expected values summarize average request cost, fraud loss, review load, or
revenue. They should be paired with tail-risk measures for capacity planning.

### Interview takeaway

Expected value is a long-run probability-weighted average, not necessarily the
mode, median, or an observable single outcome.

## Variance spread animation

### Concept

The mean remains fixed at zero while the standard deviation increases. A
density curve and rescaled synthetic observations show the distribution
spreading away from the center.

### What the controls change

The application controls a selected standard deviation and compares it with
narrow and medium reference distributions. A second control moves symmetric
outer observations to expose squared-deviation contributions.

### What to observe

Central location remains unchanged while variance, standard deviation, and tail
percentiles grow. Large deviations contribute disproportionately because they
are squared.

### Mathematical interpretation

\[
\operatorname{Var}(X)=\mathbb{E}[(X-\mu)^2]
=\mathbb{E}[X^2]-\mathbb{E}[X]^2.
\]

Standard deviation is \(\sqrt{\operatorname{Var}(X)}\) and returns to the
original unit.

### Production connection

Two AI systems can have similar average quality while one has much worse
low-quality tails. Latency systems can share a mean while having different
capacity risk.

### Interview takeaway

Variance measures dispersion, not center, and its unit is squared.

## Bernoulli variance curve

### Concept

The curve plots the variance of a Bernoulli event across every valid success
probability.

### What the controls change

The probability slider moves a labeled marker along the fixed curve and updates
variance and standard deviation numerically.

### What to observe

Variance is zero at \(p=0\) and \(p=1\), where the outcome is certain. It reaches
its maximum at \(p=0.5\), where success and failure are equally plausible.

### Mathematical interpretation

\[
X\sim\operatorname{Bernoulli}(p)
\quad\Longrightarrow\quad
\operatorname{Var}(X)=p(1-p).
\]

### Production connection

Binary events near a 50/50 rate have the greatest outcome uncertainty. Very
rare events have low Bernoulli variance per observation but can still create
high business loss.

### Interview takeaway

Low Bernoulli variance near \(p=0\) does not imply low business risk; magnitude
of loss is a separate quantity.

## Monte Carlo convergence animation

### Concept

The animation progressively reveals a fair-die empirical mean and its absolute
error. The application also supports cumulative estimation of the Bayesian
fraud posterior.

### What the controls change

Users select experiment type, number of independent paths, maximum sample size,
and base random seed. Seeds are incremented across paths to keep every run
reproducible.

### What to observe

Small-sample estimates are unstable. Paths become more concentrated around the
theoretical value with more data, but random fluctuation remains and error need
not decrease monotonically.

### Mathematical interpretation

The law of large numbers motivates

\[
\widehat{\theta}_n\longrightarrow\theta
\]

under appropriate assumptions as \(n\) grows. It does not state that every
successive estimate is closer than the previous one.

### Production connection

Monte Carlo is useful when exact dependence structures are unwieldy, but rare
events require enough draws and explicit simulation-error reporting.

### Interview takeaway

Empirical frequency estimates theoretical probability; it is not the
theoretical probability itself.

## Expected-cost threshold

### Concept

The chart compares a fixed review cost with a non-review cost that increases
linearly with posterior fraud probability. Their intersection is the decision
threshold.

### What the controls change

Users set review cost, missed-fraud cost, and the selected posterior. The chart
updates both costs, their intersection, and the lower-cost action.

### What to observe

Below the intersection, not reviewing is cheaper in the simplified model.
Above it, review is cheaper. If the threshold exceeds one, review is never the
lower-cost action under the stated costs.

### Mathematical interpretation

\[
\operatorname{Cost}(\mathrm{review})=C_R,\qquad
\operatorname{Cost}(\mathrm{no\ review})=pC_M,
\]

so

\[
p^*=\frac{C_R}{C_M}.
\]

### Production connection

Real decision rules also include capacity, delayed outcomes, legal policy,
unequal subgroup impact, and the possibility that review itself makes errors.

### Interview takeaway

A 0.5 classification threshold is not universal. It encodes a particular cost
and calibration setup that rarely matches production exactly.

## Production-connections dashboard

### Concept

The final section applies the same probability foundations to calibration, RAG
decomposition, distributed-system reliability, LLM quality variation, and
prior shift.

### What the controls change

RAG controls change retrieval quality and conditional answer correctness.
Reliability controls change component success and shared-region failure.
Monitoring controls change deployment prevalence.

### What to observe

Multi-stage outcomes decompose through total probability; shared failures break
naive independence; similar average quality can hide different tails; and
posterior probabilities move when priors shift.

### Mathematical interpretation

For RAG:

\[
P(C)=P(C\mid G)P(G)+P(C\mid G^c)P(G^c).
\]

For two independent services:

\[
P(S_1\cap S_2)=P(S_1)P(S_2),
\]

but this factorization fails when a shared cause is omitted.

### Production connection

Monitoring should cover calibration, conditional performance, priors,
dependencies, dispersion, and tails—not only offline accuracy or one global
mean.

### Interview takeaway

Similarity scores, logits, reranker outputs, and LLM self-confidence are not
automatically calibrated probabilities.

