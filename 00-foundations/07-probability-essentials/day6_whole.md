# Day 7 — Probability Essentials

## 1. Executive overview

Probability is the mathematical framework used to reason about uncertainty.

In Data Science and AI Engineering, uncertainty appears everywhere:

* whether a transaction is fraudulent;
* whether a retrieved document is relevant;
* whether a user will click or convert;
* whether a classification result is correct;
* whether a model improvement is real or random variation;
* whether an observed feature changes the probability of an outcome;
* what the expected cost of a model decision will be;
* how uncertain a prediction or experiment result is.

The central concepts for this study session are:

* **events**: possible outcomes or groups of outcomes;
* **conditional probability**: probability after observing information;
* **independence**: whether one event changes the probability of another;
* **expected value**: long-run average outcome;
* **variance**: dispersion around the expected value;
* **Bayes’ theorem**: updating beliefs after observing evidence.

These concepts form the basis of:

* statistical inference;
* hypothesis testing;
* Bayesian modeling;
* probabilistic classifiers;
* Naive Bayes;
* logistic regression interpretation;
* uncertainty estimation;
* A/B testing;
* anomaly detection;
* ranking and recommendation;
* decision theory;
* reliability analysis;
* model evaluation and calibration.

At a senior level, knowing the formulas is not enough. You should also understand:

1. what assumptions are being made;
2. which direction a conditional probability represents;
3. when independence is realistic;
4. how base rates affect predictions;
5. why expected outcomes are not necessarily typical outcomes;
6. how uncertainty should influence engineering decisions.

---

## 2. Core intuition

Imagine a fraud detection system.

Before observing anything about a transaction, you have a **prior probability** that it is fraudulent. For example:

[
P(F) = 0.01
]

where (F) represents the event “the transaction is fraudulent.”

This means that, historically, approximately 1% of transactions are fraudulent.

Now suppose your model generates an alert. You want to know:

[
P(F \mid A)
]

where (A) represents the event “the model generated an alert.”

This means:

> What is the probability that the transaction is fraudulent, given that the model raised an alert?

The model documentation might instead give you:

[
P(A \mid F)
]

which means:

> What is the probability that the model raises an alert, given that the transaction is fraudulent?

These two probabilities are not the same.

The first is what an investigator usually wants. The second is typically the model’s sensitivity or recall.

Bayes’ theorem connects them.

The essential intuition is:

> Probability starts with uncertainty, incorporates evidence, and produces an updated degree of belief.

Expected value then helps answer:

> On average, what outcome or cost should we expect?

Variance helps answer:

> How much can actual outcomes vary around that average?

---

## 3. Theoretical foundations

### 3.1 Random experiment

A random experiment is a process whose exact outcome is uncertain before it occurs.

Examples:

* predicting whether a user will churn;
* rolling a die;
* observing whether a model response is accepted;
* measuring the latency of an API request;
* observing whether a retrieved chunk is relevant.

The experiment does not need to be fundamentally random. It is enough that the outcome is uncertain from the modeler’s perspective.

---

### 3.2 Sample space

The **sample space**, usually represented by (\Omega), is the set of all possible outcomes.

For a binary classifier:

[
\Omega = {\text{positive}, \text{negative}}
]

For a six-sided die:

[
\Omega = {1,2,3,4,5,6}
]

For API latency, the sample space may be all non-negative real numbers:

[
\Omega = [0,\infty)
]

---

### 3.3 Events

An event is a subset of the sample space.

For a die:

[
A = {2,4,6}
]

could represent the event “an even number is rolled.”

Events can be combined using set operations.

#### Union

[
A \cup B
]

means that event (A), event (B), or both occur.

#### Intersection

[
A \cap B
]

means that both (A) and (B) occur.

#### Complement

[
A^c
]

means that (A) does not occur.

---

### 3.4 Probability axioms

A probability function (P) must satisfy three axioms.

#### Non-negativity

[
P(A) \geq 0
]

for every event (A).

#### Normalization

[
P(\Omega) = 1
]

The probability that some possible outcome occurs is 1.

#### Additivity

For mutually exclusive events (A) and (B):

[
P(A \cup B) = P(A) + P(B)
]

Mutually exclusive means:

[
A \cap B = \varnothing
]

where (\varnothing) is the empty set.

---

### 3.5 General addition rule

When events are not mutually exclusive:

[
P(A \cup B)
===========

P(A) + P(B) - P(A \cap B)
]

The intersection must be subtracted because it was counted twice.

For example:

* (A): user used the search feature;
* (B): user used the chatbot;
* (A \cap B): user used both.

Simply adding (P(A)) and (P(B)) would double-count users who used both features.

---

### 3.6 Conditional probability

The conditional probability of (A) given (B) is:

[
P(A \mid B)
===========

\frac{P(A \cap B)}{P(B)}
]

provided that:

[
P(B) > 0
]

The symbol (P(A \mid B)) is read as:

> Probability of (A), given (B).

Conditioning changes the reference population.

Suppose:

* 1,000 transactions exist;
* 100 generated alerts;
* 20 of the alerted transactions were fraudulent.

Then:

[
P(F \mid A) = \frac{20}{100} = 0.20
]

The denominator is no longer all 1,000 transactions. It is only the 100 alerted transactions.

---

### 3.7 Multiplication rule

Rearranging the conditional probability formula gives:

[
P(A \cap B)
===========

P(A \mid B)P(B)
]

Similarly:

[
P(A \cap B)
===========

P(B \mid A)P(A)
]

Therefore:

[
P(A \mid B)P(B)
===============

P(B \mid A)P(A)
]

This relationship leads directly to Bayes’ theorem.

---

### 3.8 Independence

Events (A) and (B) are independent when observing one does not change the probability of the other.

Formally:

[
P(A \mid B) = P(A)
]

Equivalently:

[
P(B \mid A) = P(B)
]

or:

[
P(A \cap B) = P(A)P(B)
]

These definitions are equivalent when the relevant probabilities are non-zero.

#### Example

Suppose two independent services each fail with probabilities:

[
P(A) = 0.02
]

and:

[
P(B) = 0.03
]

Then the probability that both fail is:

[
P(A \cap B)
===========

# 0.02 \times 0.03

0.0006
]

This calculation is only valid if their failures are genuinely independent.

If both depend on the same cloud region, database, network, or deployment pipeline, independence is probably unrealistic.

---

### 3.9 Independence is not mutual exclusivity

Mutually exclusive events cannot occur together:

[
P(A \cap B) = 0
]

Independent events may occur together, but the occurrence of one does not alter the probability of the other:

[
P(A \cap B) = P(A)P(B)
]

Two non-zero-probability events cannot be both mutually exclusive and independent.

If:

[
P(A)>0
]

and:

[
P(B)>0
]

then independence implies:

[
P(A \cap B)>0
]

while mutual exclusivity implies:

[
P(A \cap B)=0
]

---

### 3.10 Pairwise versus mutual independence

For three events (A), (B), and (C), pairwise independence means:

[
P(A \cap B)=P(A)P(B)
]

[
P(A \cap C)=P(A)P(C)
]

[
P(B \cap C)=P(B)P(C)
]

However, mutual independence additionally requires:

[
P(A \cap B \cap C)
==================

P(A)P(B)P(C)
]

Pairwise independence does not necessarily imply mutual independence.

This distinction appears in feature modeling and probabilistic graphical models.

---

### 3.11 Law of total probability

Suppose events (B_1, B_2, \dots, B_n) form a partition of the sample space.

A partition means:

* the events are mutually exclusive;
* one of them must occur.

Then:

[
P(A)
====

\sum_{i=1}^{n}
P(A \mid B_i)P(B_i)
]

This formula calculates the overall probability of (A) by considering every possible path through which (A) can occur.

For a binary fraud problem:

[
P(A)
====

P(A \mid F)P(F)
+
P(A \mid F^c)P(F^c)
]

An alert may occur because:

1. the transaction is fraudulent and the detector correctly alerts;
2. the transaction is legitimate and the detector produces a false positive.

---

### 3.12 Bayes’ theorem

Bayes’ theorem is:

[
P(A \mid B)
===========

\frac{P(B \mid A)P(A)}{P(B)}
]

Where:

* (P(A)) is the **prior probability**;
* (P(B \mid A)) is the **likelihood of the evidence under (A)**;
* (P(B)) is the **evidence** or marginal probability;
* (P(A \mid B)) is the **posterior probability**.

Using the law of total probability:

[
P(A \mid B)
===========

\frac{P(B \mid A)P(A)}
{\sum_i P(B \mid A_i)P(A_i)}
]

For binary classification:

[
P(F \mid A)
===========

\frac{P(A \mid F)P(F)}
{P(A \mid F)P(F)+P(A \mid F^c)P(F^c)}
]

---

### 3.13 Random variables

A random variable maps outcomes to numerical values.

For a binary event:

[
X =
\begin{cases}
1, & \text{if the event occurs} \
0, & \text{otherwise}
\end{cases}
]

This is a Bernoulli random variable.

A random variable may be:

* **discrete**, such as the number of retrieved relevant documents;
* **continuous**, such as model latency or response time.

---

### 3.14 Probability mass function

For a discrete random variable (X), the probability mass function is:

[
p_X(x)=P(X=x)
]

The probabilities must satisfy:

[
p_X(x)\geq 0
]

and:

[
\sum_x p_X(x)=1
]

---

### 3.15 Probability density function

For a continuous random variable (X), a probability density function (f_X(x)) describes relative probability density.

Probabilities are calculated over intervals:

[
P(a \leq X \leq b)
==================

\int_a^b f_X(x),dx
]

For a continuous variable:

[
P(X=x)=0
]

This does not mean the value cannot occur. It means that a single point has zero area under a continuous density.

---

### 3.16 Cumulative distribution function

The cumulative distribution function is:

[
F_X(x)=P(X\leq x)
]

It answers:

> What is the probability that (X) is less than or equal to (x)?

The CDF applies to both discrete and continuous random variables.

---

## 4. Mathematical, statistical or logical foundations

### 4.1 Deriving Bayes’ theorem

Start with the conditional probability definition:

[
P(A \mid B)
===========

\frac{P(A \cap B)}{P(B)}
]

Also:

[
P(B \mid A)
===========

\frac{P(A \cap B)}{P(A)}
]

Rearranging the second expression:

[
P(A \cap B)
===========

P(B \mid A)P(A)
]

Substitute this result into the first expression:

[
P(A \mid B)
===========

\frac{P(B \mid A)P(A)}{P(B)}
]

That is Bayes’ theorem.

It is not a separate probability law. It follows directly from the definition of conditional probability.

---

### 4.2 Fraud detection example

Suppose:

[
P(F)=0.01
]

where (F) is the event that a transaction is fraudulent.

The detector has 90% sensitivity:

[
P(A \mid F)=0.90
]

The false-positive rate is 5%:

[
P(A \mid F^c)=0.05
]

The probability that an arbitrary transaction generates an alert is:

[
P(A)
====

P(A \mid F)P(F)
+
P(A \mid F^c)P(F^c)
]

Since:

[
P(F^c)=1-P(F)=0.99
]

we obtain:

[
P(A)
====

0.90(0.01)+0.05(0.99)
]

[
P(A)=0.009+0.0495=0.0585
]

The probability that an alerted transaction is actually fraudulent is:

[
P(F \mid A)
===========

\frac{P(A \mid F)P(F)}{P(A)}
]

[
P(F \mid A)
===========

\frac{0.90(0.01)}{0.0585}
]

[
P(F \mid A)
\approx 0.1538
]

Therefore, even with 90% sensitivity:

[
P(F \mid A)\approx15.38%
]

Most alerts are still false positives because fraud is rare.

This is the **base-rate effect**.

---

### 4.3 Expected value

For a discrete random variable (X), the expected value is:

[
\mathbb{E}[X]
=============

\sum_x xP(X=x)
]

Where:

* (X) is the random variable;
* (x) is a possible value of (X);
* (P(X=x)) is the probability of that value;
* (\mathbb{E}[X]) is the probability-weighted average.

For a continuous random variable:

[
\mathbb{E}[X]
=============

\int_{-\infty}^{\infty}
x f_X(x),dx
]

where (f_X(x)) is the probability density function.

Expected value represents a long-run average, not necessarily a value that will be observed.

For a fair six-sided die:

[
\mathbb{E}[X]
=============

# \frac{1+2+3+4+5+6}{6}

3.5
]

A roll of 3.5 is impossible, but the long-run average approaches 3.5.

---

### 4.4 Expected value of a Bernoulli variable

Let:

[
X \sim \operatorname{Bernoulli}(p)
]

where:

* (X=1) with probability (p);
* (X=0) with probability (1-p).

Then:

[
\mathbb{E}[X]
=============

1 \cdot p + 0 \cdot (1-p)
]

[
\mathbb{E}[X]=p
]

For a binary target, the expected value is the positive-class probability.

This helps explain why probabilistic binary models estimate values between 0 and 1.

---

### 4.5 Linearity of expectation

For random variables (X) and (Y):

[
\mathbb{E}[X+Y]
===============

\mathbb{E}[X]+\mathbb{E}[Y]
]

More generally:

[
\mathbb{E}[aX+bY]
=================

a\mathbb{E}[X]+b\mathbb{E}[Y]
]

where (a) and (b) are constants.

Importantly, linearity of expectation does not require (X) and (Y) to be independent.

This property is useful when calculating expected:

* infrastructure cost;
* number of failures;
* total fraud loss;
* number of relevant retrieved documents;
* aggregate conversion volume.

---

### 4.6 Conditional expectation

The conditional expectation of (X) given event (A) is:

[
\mathbb{E}[X \mid A]
]

This represents the expected value of (X) within the subpopulation where (A) occurred.

Examples:

[
\mathbb{E}[\text{transaction amount} \mid \text{fraud}]
]

[
\mathbb{E}[\text{latency} \mid \text{cache miss}]
]

[
\mathbb{E}[\text{answer quality} \mid \text{retrieval succeeded}]
]

Conditional expectation is frequently more useful than an overall average because production systems often contain heterogeneous subgroups.

---

### 4.7 Law of total expectation

If (Y) represents a grouping variable:

[
\mathbb{E}[X]
=============

\mathbb{E}\left[\mathbb{E}[X \mid Y]\right]
]

For a discrete (Y):

[
\mathbb{E}[X]
=============

\sum_y
\mathbb{E}[X \mid Y=y]P(Y=y)
]

For example, overall expected latency can be decomposed by request type:

[
\mathbb{E}[L]
=============

\mathbb{E}[L \mid C]P(C)
+
\mathbb{E}[L \mid C^c]P(C^c)
]

where:

* (L) is latency;
* (C) is a cache hit;
* (C^c) is a cache miss.

---

### 4.8 Variance

Variance measures expected squared deviation from the mean.

Let:

[
\mu = \mathbb{E}[X]
]

Then:

[
\operatorname{Var}(X)
=====================

\mathbb{E}\left[(X-\mu)^2\right]
]

Where:

* (X) is the random variable;
* (\mu) is its expected value;
* (X-\mu) is the deviation from the mean;
* the square prevents positive and negative deviations from canceling.

An equivalent formula is:

[
\operatorname{Var}(X)
=====================

\mathbb{E}[X^2]-\mathbb{E}[X]^2
]

#### Derivation

Starting from:

[
\operatorname{Var}(X)
=====================

\mathbb{E}\left[(X-\mu)^2\right]
]

Expand the square:

[
(X-\mu)^2=X^2-2\mu X+\mu^2
]

Apply expectation:

[
\operatorname{Var}(X)
=====================

## \mathbb{E}[X^2]

2\mu\mathbb{E}[X]
+
\mu^2
]

Since:

[
\mathbb{E}[X]=\mu
]

we obtain:

[
\operatorname{Var}(X)
=====================

\mathbb{E}[X^2]-2\mu^2+\mu^2
]

Therefore:

[
\operatorname{Var}(X)
=====================

\mathbb{E}[X^2]-\mu^2
]

or:

[
\operatorname{Var}(X)
=====================

\mathbb{E}[X^2]-\mathbb{E}[X]^2
]

---

### 4.9 Standard deviation

The standard deviation is:

[
\sigma = \sqrt{\operatorname{Var}(X)}
]

Variance uses squared units. Standard deviation returns to the original unit.

If latency is measured in milliseconds:

* variance is measured in squared milliseconds;
* standard deviation is measured in milliseconds.

This makes standard deviation easier to interpret operationally.

---

### 4.10 Variance of a Bernoulli variable

For:

[
X \sim \operatorname{Bernoulli}(p)
]

we know:

[
\mathbb{E}[X]=p
]

Because (X) can only be 0 or 1:

[
X^2=X
]

Therefore:

[
\mathbb{E}[X^2]=p
]

Using:

[
\operatorname{Var}(X)
=====================

\mathbb{E}[X^2]-\mathbb{E}[X]^2
]

we obtain:

[
\operatorname{Var}(X)
=====================

p-p^2
]

[
\operatorname{Var}(X)
=====================

p(1-p)
]

The Bernoulli variance is highest when:

[
p=0.5
]

because outcomes are most uncertain around an even split.

Variance approaches zero as (p) approaches 0 or 1.

---

### 4.11 Variance under scaling

For a constant (a):

[
\operatorname{Var}(aX)
======================

a^2\operatorname{Var}(X)
]

Adding a constant does not change variance:

[
\operatorname{Var}(X+b)
=======================

\operatorname{Var}(X)
]

Therefore:

[
\operatorname{Var}(aX+b)
========================

a^2\operatorname{Var}(X)
]

Adding 100 milliseconds to every request changes the average latency but not the spread.

Multiplying every latency by 2 multiplies the variance by 4.

---

### 4.12 Covariance

Covariance measures whether two random variables vary together.

[
\operatorname{Cov}(X,Y)
=======================

\mathbb{E}\left[
(X-\mathbb{E}[X])
(Y-\mathbb{E}[Y])
\right]
]

Equivalent form:

[
\operatorname{Cov}(X,Y)
=======================

## \mathbb{E}[XY]

\mathbb{E}[X]\mathbb{E}[Y]
]

If (X) and (Y) are independent:

[
\operatorname{Cov}(X,Y)=0
]

However, zero covariance does not generally imply independence.

Independence is a stronger condition.

---

### 4.13 Variance of a sum

For two random variables:

[
\operatorname{Var}(X+Y)
=======================

\operatorname{Var}(X)
+
\operatorname{Var}(Y)
+
2\operatorname{Cov}(X,Y)
]

If (X) and (Y) are independent:

[
\operatorname{Cov}(X,Y)=0
]

and therefore:

[
\operatorname{Var}(X+Y)
=======================

\operatorname{Var}(X)
+
\operatorname{Var}(Y)
]

This matters when aggregating uncertain costs, traffic, model errors, or service latencies.

---

### 4.14 Population variance versus sample variance

Given observations:

[
x_1,x_2,\dots,x_n
]

the population-style variance estimate is:

[
\frac{1}{n}
\sum_{i=1}^{n}
(x_i-\bar{x})^2
]

where:

* (n) is the number of observations;
* (x_i) is observation (i);
* (\bar{x}) is the sample mean.

The usual unbiased sample variance is:

[
s^2
===

\frac{1}{n-1}
\sum_{i=1}^{n}
(x_i-\bar{x})^2
]

The denominator (n-1) applies Bessel’s correction.

In NumPy:

```python
np.var(values, ddof=0)  # Divide by n
np.var(values, ddof=1)  # Divide by n - 1
```

In pandas, `.var()` uses `ddof=1` by default.

---

### 4.15 Expected loss and decision-making

Probability alone does not determine the best action.

Suppose a fraud alert has posterior probability:

[
p=P(F\mid A)
]

Let:

* (C_R) be the cost of reviewing a transaction;
* (C_M) be the cost of missing a fraudulent transaction.

The expected cost of not reviewing is approximately:

[
pC_M
]

The cost of reviewing is:

[
C_R
]

A simple decision rule is to review when:

[
pC_M>C_R
]

or:

[
p>\frac{C_R}{C_M}
]

This connects probability to decision theory.

A probability threshold should therefore reflect business costs, not only a conventional value such as 0.5.

---

## 5. Practical applicability

### 5.1 Classification

A probabilistic classifier estimates quantities such as:

[
P(Y=1\mid X=x)
]

where:

* (Y) is the target;
* (X) represents input features;
* (x) is a specific feature vector.

This probability can support:

* ranking;
* threshold selection;
* human review;
* expected-loss minimization;
* uncertainty communication.

A hard label discards information that a probability score retains.

---

### 5.2 Fraud and anomaly detection

Fraud is usually rare, making base rates critical.

A detector may have:

* high recall;
* low false-positive rate;
* poor positive predictive value.

The posterior probability of fraud depends on:

* prevalence;
* sensitivity;
* false-positive rate.

Ignoring prevalence produces misleading conclusions.

---

### 5.3 Medical, legal and risk-related systems

Bayesian updates appear whenever evidence changes the probability of a hypothesis.

Examples:

* probability of a condition after a test result;
* probability of document relevance after retrieval evidence;
* probability of policy violation after automated checks;
* probability of model failure after monitoring signals.

High-stakes systems require more than probability estimates. They also require calibration, validation, auditability and appropriate human oversight.

---

### 5.4 A/B testing

Probability and expectation are used to reason about:

* conversion rates;
* expected uplift;
* uncertainty in observed differences;
* probability that a variant is better;
* risk of false conclusions.

Observed averages alone are insufficient. Their variance and sample size determine how much confidence should be placed in the result.

---

### 5.5 RAG systems

Probability concepts appear implicitly in RAG pipelines:

* probability that a retrieved document is relevant;
* probability that the answer is supported by context;
* conditional answer quality given retrieval quality;
* expected cost across retrieval and generation stages;
* probability of failure at each component.

For example:

[
P(\text{correct answer})
]

can be decomposed conceptually into:

[
P(\text{correct answer} \mid \text{good retrieval})
P(\text{good retrieval})
]

plus the corresponding term for poor retrieval.

This decomposition helps locate system bottlenecks.

However, similarity scores from a vector database are not automatically calibrated probabilities.

---

### 5.6 Reliability engineering

Suppose a system depends on multiple components.

If component failures are independent:

[
P(\text{all succeed})
=====================

\prod_i P(\text{component } i \text{ succeeds})
]

But production failures are often correlated because components share:

* cloud regions;
* credentials;
* databases;
* deployment pipelines;
* upstream APIs;
* network infrastructure.

Assuming independence can substantially underestimate systemic risk.

---

### 5.7 LLM evaluation

Expected value is useful when aggregating quality scores across a dataset:

[
\mathbb{E}[\text{quality score}]
]

Variance helps reveal whether quality is stable or highly inconsistent.

Two systems may have the same mean quality while having very different reliability.

For production AI, a slightly lower mean with lower tail risk may be preferable to a higher mean with catastrophic failures.

---

### 5.8 Cost optimization

Expected cost can combine:

* token cost;
* model-routing probabilities;
* retry probability;
* cache-hit probability;
* human-review cost;
* failure cost.

For example:

[
\mathbb{E}[\text{request cost}]
===============================

P(C)C_C
+
P(C^c)C_G
]

where:

* (C) is the event of a cache hit;
* (C_C) is the cost when cached;
* (C_G) is the cost of generating a new response.

---

### 5.9 When probability is not enough

Probability calculations may be misleading when:

* inputs are not representative of production;
* events are non-stationary;
* dependencies are ignored;
* probabilities are poorly calibrated;
* the cost function is unknown;
* the population changes over time;
* important causal variables are omitted;
* the underlying process is adversarial.

Probability quantifies uncertainty under assumptions. It does not make incorrect assumptions safe.

---

## 6. Common pitfalls and mistakes

### 6.1 Confusing (P(A\mid B)) with (P(B\mid A))

This is the most common mistake.

[
P(\text{alert}\mid\text{fraud})
]

is not the same as:

[
P(\text{fraud}\mid\text{alert})
]

The first is recall. The second is precision or positive predictive value.

---

### 6.2 Ignoring the base rate

A highly sensitive detector can still produce mostly false positives when the event is rare.

Always consider:

[
P(A)
]

before interpreting:

[
P(A\mid B)
]

---

### 6.3 Treating mutually exclusive events as independent

Mutually exclusive means events cannot occur together.

Independent means one event does not affect the probability of another.

These concepts are fundamentally different.

---

### 6.4 Assuming feature independence without justification

Naive Bayes assumes conditional independence of features given the class.

This can work surprisingly well, but the assumption is often unrealistic.

Correlated signals may cause evidence to be counted multiple times.

---

### 6.5 Interpreting expected value as the most likely outcome

Expected value is a probability-weighted average.

It may:

* be impossible to observe;
* be far from the median;
* be dominated by rare extreme outcomes.

For skewed cost distributions, report more than the mean.

---

### 6.6 Ignoring variance and tail behavior

Two models can have the same average performance but different reliability.

Always inspect:

* standard deviation;
* percentiles;
* worst-case behavior;
* subgroup performance;
* tail latency;
* catastrophic failure frequency.

---

### 6.7 Using the wrong variance convention

NumPy and pandas may use different defaults.

```python
np.var(values)       # ddof=0
pd.Series(values).var()  # ddof=1
```

Know whether you are calculating:

* population variance;
* unbiased sample variance.

---

### 6.8 Assuming zero covariance implies independence

Independence implies zero covariance when moments exist.

The reverse is not generally true.

Nonlinear dependence may exist even when covariance is zero.

---

### 6.9 Treating model scores as calibrated probabilities

A score of 0.8 does not necessarily mean that 80% of similar cases are positive.

Probabilities should be checked using:

* calibration curves;
* Brier score;
* expected calibration error;
* reliability diagrams.

Similarity scores, logits, reranker scores and LLM self-confidence are not automatically probabilities.

---

### 6.10 Leakage in probability estimates

If future information enters the model features, estimated probabilities can appear excellent offline and fail in production.

Examples:

* using a case outcome generated after the prediction timestamp;
* normalizing with statistics from the full dataset;
* splitting repeated users across train and test;
* retrieving documents created after the target event.

---

### 6.11 Selection bias

Conditional probabilities depend on the population being conditioned on.

For example:

[
P(\text{fraud}\mid\text{manually reviewed})
]

may not represent:

[
P(\text{fraud}\mid\text{all transactions})
]

because manually reviewed transactions were selected by a previous system.

---

### 6.12 Independence assumptions in distributed systems

Multiplying service failure probabilities assumes independence.

Shared dependencies can make this calculation dangerously optimistic.

Model correlated failure modes explicitly or test them through failure injection.

---

### 6.13 Forgetting non-stationarity

Probabilities estimated from historical data may change because of:

* user behavior;
* seasonality;
* policy changes;
* adversarial adaptation;
* product changes;
* model feedback loops.

Production probabilities require monitoring and recalibration.

---

### 6.14 Applying a 0.5 threshold automatically

A threshold of 0.5 is appropriate only under particular assumptions about:

* class balance;
* costs;
* calibration;
* utility.

In many real systems, the optimal threshold is much lower or higher.

---

## 7. Important comparisons

### 7.1 Marginal versus conditional probability

Marginal probability:

[
P(A)
]

describes the probability of (A) without conditioning on additional information.

Conditional probability:

[
P(A\mid B)
]

describes the probability of (A) within the subpopulation where (B) occurred.

Use conditional probability when context or evidence changes the relevant population.

---

### 7.2 Independence versus conditional independence

Ordinary independence:

[
P(A\cap B)=P(A)P(B)
]

Conditional independence given (C):

[
P(A\cap B\mid C)
================

P(A\mid C)P(B\mid C)
]

Two variables may be dependent overall but independent after conditioning on another variable.

For example, umbrella use and traffic accidents may both increase on rainy days. They may be associated marginally, but much less associated after conditioning on weather.

Conditional independence is fundamental to:

* Naive Bayes;
* Bayesian networks;
* graphical models;
* causal reasoning.

---

### 7.3 Probability versus likelihood

Probability treats the model or parameter as fixed and the data as variable:

[
P(\text{data}\mid\theta)
]

Likelihood uses the same expression but views it as a function of the parameter (\theta) for observed data.

[
L(\theta;\text{data})
=====================

P(\text{data}\mid\theta)
]

A likelihood is not generally a normalized probability distribution over (\theta).

This distinction matters in maximum likelihood estimation.

---

### 7.4 Prior, likelihood and posterior

Prior:

[
P(H)
]

represents belief before observing the current evidence.

Likelihood:

[
P(E\mid H)
]

represents how compatible the evidence (E) is with hypothesis (H).

Posterior:

[
P(H\mid E)
]

represents updated belief after observing evidence.

Bayes’ theorem combines them:

[
P(H\mid E)
\propto
P(E\mid H)P(H)
]

The proportionality hides the normalization term (P(E)).

---

### 7.5 Expected value versus median

Expected value minimizes expected squared error.

The median minimizes expected absolute error.

For symmetric distributions, they may be similar.

For skewed distributions, the mean may be strongly influenced by rare extreme values.

Examples:

* latency;
* transaction amounts;
* insurance losses;
* token usage;
* cloud costs.

---

### 7.6 Variance versus mean absolute deviation

Variance:

* squares deviations;
* penalizes large deviations strongly;
* has useful algebraic properties;
* is sensitive to outliers.

Mean absolute deviation:

* uses absolute deviations;
* is easier to interpret robustly;
* is less sensitive to extreme observations;
* is less algebraically convenient.

---

### 7.7 Frequentist versus Bayesian interpretation

A frequentist interpretation treats probability as long-run frequency under repeated experiments.

A Bayesian interpretation treats probability as a degree of belief under uncertainty.

In practice:

* frequentist methods often focus on estimators, sampling distributions and confidence intervals;
* Bayesian methods combine prior distributions with observed data to produce posterior distributions.

Both use the same probability foundations.

---

### 7.8 Accuracy versus posterior probability

Accuracy is an aggregate metric:

[
\frac{\text{correct predictions}}{\text{total predictions}}
]

Posterior probability is case-specific:

[
P(Y=1\mid X=x)
]

A model can have high accuracy while producing poor probabilities, especially with severe class imbalance.

---

### 7.9 Confidence versus calibration

Confidence is the score a model reports.

Calibration measures whether reported probabilities match observed frequencies.

A calibrated model predicting 0.8 for many cases should be correct approximately 80% of the time in that group.

A model may rank examples well but still be poorly calibrated.

---

## 8. Practical Python example

This example simulates a fraud detector and compares the theoretical posterior probability with an empirical Monte Carlo estimate.

It also demonstrates expected value, variance and the base-rate effect.

```python
"""
Day 7 — Probability Essentials

Demonstrates:
- conditional probability;
- Bayes' theorem;
- expected value;
- variance;
- Monte Carlo estimation;
- base-rate effects.

Dependencies:
    pip install numpy pandas matplotlib
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bayes_posterior(
    prior: float,
    true_positive_rate: float,
    false_positive_rate: float,
) -> float:
    """
    Compute P(Fraud | Alert).

    Parameters
    ----------
    prior:
        P(Fraud)
    true_positive_rate:
        P(Alert | Fraud)
    false_positive_rate:
        P(Alert | Not Fraud)
    """
    probability_alert = (
        true_positive_rate * prior
        + false_positive_rate * (1.0 - prior)
    )

    return true_positive_rate * prior / probability_alert


def simulate_transactions(
    n_transactions: int,
    fraud_rate: float,
    true_positive_rate: float,
    false_positive_rate: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic transaction dataset."""
    rng = np.random.default_rng(seed)

    is_fraud = rng.random(n_transactions) < fraud_rate

    alert_probability = np.where(
        is_fraud,
        true_positive_rate,
        false_positive_rate,
    )

    has_alert = rng.random(n_transactions) < alert_probability

    return pd.DataFrame(
        {
            "is_fraud": is_fraud,
            "has_alert": has_alert,
        }
    )


def main() -> None:
    n_transactions = 200_000

    fraud_rate = 0.01
    true_positive_rate = 0.90
    false_positive_rate = 0.05

    df = simulate_transactions(
        n_transactions=n_transactions,
        fraud_rate=fraud_rate,
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
    )

    theoretical_posterior = bayes_posterior(
        prior=fraud_rate,
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
    )

    alerted = df[df["has_alert"]]

    empirical_posterior = alerted["is_fraud"].mean()

    empirical_fraud_rate = df["is_fraud"].mean()
    fraud_variance = df["is_fraud"].var(ddof=0)

    theoretical_bernoulli_variance = fraud_rate * (1.0 - fraud_rate)

    confusion_table = pd.crosstab(
        df["is_fraud"],
        df["has_alert"],
        rownames=["Actual fraud"],
        colnames=["Alert"],
    )

    print("Confusion table")
    print(confusion_table)
    print()

    print(f"Theoretical P(Fraud | Alert): {theoretical_posterior:.4f}")
    print(f"Empirical P(Fraud | Alert):   {empirical_posterior:.4f}")
    print()

    print(f"Theoretical E[Fraud]: {fraud_rate:.4f}")
    print(f"Empirical E[Fraud]:   {empirical_fraud_rate:.4f}")
    print()

    print(
        "Theoretical Var(Fraud): "
        f"{theoretical_bernoulli_variance:.6f}"
    )
    print(f"Empirical Var(Fraud):   {fraud_variance:.6f}")

    # Show how the base rate changes the posterior probability.
    base_rates = np.linspace(0.001, 0.20, 200)

    posterior_probabilities = [
        bayes_posterior(
            prior=rate,
            true_positive_rate=true_positive_rate,
            false_positive_rate=false_positive_rate,
        )
        for rate in base_rates
    ]

    plt.figure(figsize=(9, 5))
    plt.plot(base_rates, posterior_probabilities)
    plt.xlabel("Prior fraud probability P(Fraud)")
    plt.ylabel("Posterior probability P(Fraud | Alert)")
    plt.title("Base-rate effect on posterior probability")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

### Expected interpretation

With the initial configuration:

[
P(F)=0.01
]

[
P(A\mid F)=0.90
]

[
P(A\mid F^c)=0.05
]

the posterior should be close to:

[
P(F\mid A)\approx0.1538
]

The exact simulation result varies slightly because of random sampling.

The chart shows that the same detector produces very different posterior probabilities depending on the prior fraud rate.

This is why metrics cannot be interpreted independently of the deployment population.

---

## 9. From-scratch implementation when useful

The following implementation calculates conditional probability, expected value, variance and a binary Bayes update without relying on statistical libraries.

```python
"""
Simplified from-scratch probability utilities.

Educational implementation only.
Not designed as a replacement for tested scientific libraries.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def validate_probabilities(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 1:
        raise ValueError("Probabilities must be a one-dimensional array.")

    if np.any(probabilities < 0):
        raise ValueError("Probabilities cannot be negative.")

    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("Probabilities must sum to 1.")


def expected_value(
    values: Sequence[float],
    probabilities: Sequence[float],
) -> float:
    x = np.asarray(values, dtype=float)
    p = np.asarray(probabilities, dtype=float)

    if x.shape != p.shape:
        raise ValueError(
            "Values and probabilities must have the same shape."
        )

    validate_probabilities(p)

    return float(np.sum(x * p))


def variance(
    values: Sequence[float],
    probabilities: Sequence[float],
) -> float:
    x = np.asarray(values, dtype=float)
    p = np.asarray(probabilities, dtype=float)

    mean = expected_value(x, p)

    return float(np.sum(((x - mean) ** 2) * p))


def conditional_probability(
    event_a: Sequence[bool],
    event_b: Sequence[bool],
) -> float:
    a = np.asarray(event_a, dtype=bool)
    b = np.asarray(event_b, dtype=bool)

    if a.shape != b.shape:
        raise ValueError("Events must have the same number of observations.")

    probability_b = np.mean(b)

    if probability_b == 0:
        raise ValueError("P(B) must be greater than zero.")

    probability_a_and_b = np.mean(a & b)

    return float(probability_a_and_b / probability_b)


def binary_bayes_update(
    prior: float,
    probability_evidence_given_hypothesis: float,
    probability_evidence_given_not_hypothesis: float,
) -> float:
    """
    Calculate P(H | E) for a binary hypothesis.

    prior:
        P(H)

    probability_evidence_given_hypothesis:
        P(E | H)

    probability_evidence_given_not_hypothesis:
        P(E | not H)
    """
    for value in (
        prior,
        probability_evidence_given_hypothesis,
        probability_evidence_given_not_hypothesis,
    ):
        if not 0 <= value <= 1:
            raise ValueError("All probability values must be between 0 and 1.")

    probability_evidence = (
        probability_evidence_given_hypothesis * prior
        + probability_evidence_given_not_hypothesis * (1 - prior)
    )

    if probability_evidence == 0:
        raise ValueError("The probability of the evidence cannot be zero.")

    posterior = (
        probability_evidence_given_hypothesis * prior
        / probability_evidence
    )

    return posterior


def main() -> None:
    die_values = [1, 2, 3, 4, 5, 6]
    die_probabilities = [1 / 6] * 6

    die_mean = expected_value(die_values, die_probabilities)
    die_variance = variance(die_values, die_probabilities)

    print(f"Expected die value: {die_mean:.4f}")
    print(f"Die variance: {die_variance:.4f}")

    fraud_posterior = binary_bayes_update(
        prior=0.01,
        probability_evidence_given_hypothesis=0.90,
        probability_evidence_given_not_hypothesis=0.05,
    )

    print(f"P(Fraud | Alert): {fraud_posterior:.4f}")

    fraud = np.array([True, False, True, False, False])
    alert = np.array([True, True, False, False, True])

    empirical_probability = conditional_probability(
        event_a=fraud,
        event_b=alert,
    )

    print(f"Empirical P(Fraud | Alert): {empirical_probability:.4f}")


if __name__ == "__main__":
    main()
```

### Why this implementation matters

The functions expose the internal operations that libraries often hide:

[
\mathbb{E}[X]
=============

\sum_x xp(x)
]

[
\operatorname{Var}(X)
=====================

\sum_x (x-\mu)^2p(x)
]

[
P(A\mid B)
==========

\frac{P(A\cap B)}{P(B)}
]

[
P(H\mid E)
==========

\frac{P(E\mid H)P(H)}
{P(E\mid H)P(H)+P(E\mid H^c)P(H^c)}
]

The implementation is educational rather than production-grade. Mature numerical libraries provide better performance, testing and numerical stability.

---

## 10. Suggested experiments

### Experiment 1 — Change the base rate

Run the fraud simulation with:

```python
fraud_rate = 0.001
fraud_rate = 0.01
fraud_rate = 0.10
```

Keep the detector characteristics constant.

Observe how:

[
P(F\mid A)
]

changes dramatically even though:

[
P(A\mid F)
]

remains unchanged.

This demonstrates that model precision depends on the deployment population.

---

### Experiment 2 — Reduce the false-positive rate

Compare:

```python
false_positive_rate = 0.10
false_positive_rate = 0.05
false_positive_rate = 0.01
```

For rare events, reducing false positives can improve operational usefulness more than slightly increasing recall.

---

### Experiment 3 — Compare expected cost across thresholds

Create several hypothetical thresholds with different:

* true-positive rates;
* false-positive rates.

Calculate:

[
\text{Expected cost}
====================

C_{FP}P(FP)
+
C_{FN}P(FN)
]

where:

* (C_{FP}) is the cost of a false positive;
* (C_{FN}) is the cost of a false negative;
* (P(FP)) is the probability of a false positive;
* (P(FN)) is the probability of a false negative.

Select the threshold with the lowest expected cost rather than the highest accuracy.

---

### Experiment 4 — Simulate correlated failures

Generate two service failures using:

1. independent random variables;
2. a shared “regional outage” variable.

Compare the empirical probability that both services fail.

This demonstrates how shared dependencies invalidate simple independence assumptions.

---

### Experiment 5 — Compare mean and median under skew

Generate latency data:

```python
latency = np.random.lognormal(mean=4.5, sigma=0.8, size=10_000)
```

Calculate:

```python
np.mean(latency)
np.median(latency)
np.std(latency)
np.percentile(latency, [90, 95, 99])
```

Observe why average latency alone is insufficient for production reliability analysis.

---

## 11. Senior interview questions

### Question 1 — What is conditional probability?

Conditional probability is the probability of an event within a restricted population defined by another event.

[
P(A\mid B)
==========

\frac{P(A\cap B)}{P(B)}
]

It answers how likely (A) is after learning that (B) occurred.

In production systems, conditioning is essential because global metrics often hide subgroup behavior. Examples include fraud probability given an alert, latency given a cache miss, or answer accuracy given successful retrieval.

---

### Question 2 — What is the difference between (P(A\mid B)) and (P(B\mid A))?

They represent different directions of conditioning.

[
P(A\mid B)
]

is the probability of (A) after observing (B).

[
P(B\mid A)
]

is the probability of (B) after observing (A).

For fraud detection:

* (P(\text{alert}\mid\text{fraud})) is recall;
* (P(\text{fraud}\mid\text{alert})) is precision.

Bayes’ theorem relates them, but they are usually not equal.

---

### Question 3 — Explain the base-rate fallacy.

The base-rate fallacy occurs when someone interprets evidence without considering how common the underlying event is.

A test may have high sensitivity but still produce mostly false positives when the positive class is rare.

The posterior probability depends on:

[
P(H\mid E)
==========

\frac{P(E\mid H)P(H)}{P(E)}
]

The prior (P(H)) cannot be ignored.

---

### Question 4 — What does independence mean?

Two events are independent when observing one does not alter the probability of the other.

[
P(A\mid B)=P(A)
]

Equivalent formulation:

[
P(A\cap B)=P(A)P(B)
]

In system design, independence assumptions should be challenged because components often share infrastructure and failure modes.

---

### Question 5 — Are mutually exclusive events independent?

Generally, no.

For mutually exclusive events:

[
P(A\cap B)=0
]

For independent events:

[
P(A\cap B)=P(A)P(B)
]

If both events have positive probability, then:

[
P(A)P(B)>0
]

Therefore they cannot be both mutually exclusive and independent.

---

### Question 6 — What is expected value?

Expected value is the probability-weighted average of a random variable.

For a discrete variable:

[
\mathbb{E}[X]
=============

\sum_x xP(X=x)
]

It represents a long-run average under repeated sampling.

In real systems, expected value is useful for calculating expected revenue, infrastructure cost, fraud loss or review load. It should not automatically be interpreted as the most common or typical outcome.

---

### Question 7 — Why is variance important if we already know the mean?

The mean describes central tendency, while variance describes dispersion.

Two systems may have the same average latency or model quality but very different stability.

A production system with slightly better average performance but high variance and severe tail failures may be less desirable than a more consistent system.

---

### Question 8 — Why does variance square deviations?

Squaring:

* prevents positive and negative deviations from canceling;
* penalizes large deviations more strongly;
* produces mathematically convenient properties;
* makes variance differentiable and useful in optimization.

The downside is sensitivity to outliers and squared units.

---

### Question 9 — What is the difference between population and sample variance?

Population variance divides by (n):

[
\frac{1}{n}
\sum_{i=1}^{n}(x_i-\bar{x})^2
]

The conventional unbiased sample variance divides by (n-1):

[
\frac{1}{n-1}
\sum_{i=1}^{n}(x_i-\bar{x})^2
]

The (n-1) denominator compensates for the fact that the sample mean is estimated from the same data.

---

### Question 10 — Why does linearity of expectation not require independence?

Expectation is a linear operator.

[
\mathbb{E}[X+Y]
===============

\mathbb{E}[X]+\mathbb{E}[Y]
]

This property follows directly from weighted sums or integrals.

Independence becomes relevant for simplifying variance:

[
\operatorname{Var}(X+Y)
]

because covariance terms appear.

---

### Question 11 — Does zero covariance imply independence?

No, not in general.

Independence implies zero covariance when the expectations exist, but two variables can have nonlinear dependence with zero covariance.

For jointly Gaussian variables, zero covariance does imply independence, but that is a special case.

---

### Question 12 — How would you use Bayes’ theorem in a real ML system?

I would use it when evidence must update the probability of a hypothesis and base rates matter.

Examples include:

* fraud probability after an alert;
* probability of relevance after observing retrieval signals;
* probability of system failure after a monitoring event;
* probability of a class after multiple diagnostic signals.

In implementation, I would also validate whether the inputs are calibrated and whether independence assumptions are justified.

---

### Question 13 — Why is a 0.5 classification threshold often inappropriate?

The threshold should reflect:

* class prevalence;
* false-positive cost;
* false-negative cost;
* operational capacity;
* model calibration;
* regulatory or business constraints.

A 0.5 threshold assumes a particular symmetric decision setup that rarely matches real production systems.

---

### Question 14 — How does probability connect to RAG evaluation?

RAG can be treated as a multi-stage uncertain system.

Answer correctness depends on events such as:

* relevant evidence being retrieved;
* the evidence being ranked highly;
* the generator using the evidence correctly;
* validation detecting unsupported claims.

Conditional evaluation can separate:

[
P(\text{correct answer}\mid\text{good retrieval})
]

from:

[
P(\text{good retrieval})
]

This helps determine whether the bottleneck is retrieval or generation.

---

### Question 15 — How would you evaluate whether model probabilities are trustworthy?

I would assess calibration using:

* reliability diagrams;
* calibration curves;
* Brier score;
* log loss;
* expected calibration error;
* subgroup calibration;
* temporal calibration.

I would also verify calibration on production-like data, because prevalence shift can affect posterior probabilities.

---

### Question 16 — A model has 99% accuracy. Is it good?

Not enough information is available.

If the positive class occurs in only 1% of cases, predicting every observation as negative also produces 99% accuracy.

I would inspect:

* class distribution;
* confusion matrix;
* precision;
* recall;
* calibration;
* expected cost;
* performance by subgroup;
* production decision requirements.

---

### Question 17 — How would you model a multi-service reliability problem?

I would identify:

1. component success and failure events;
2. system topology;
3. shared dependencies;
4. conditional failure probabilities;
5. correlated regional or upstream failures;
6. recovery and retry behavior.

I would avoid multiplying component probabilities unless independence was defensible. For complex systems, simulation or a probabilistic graphical model may be more appropriate.

---

### Question 18 — How can distribution shift affect Bayes’ theorem in production?

The posterior depends on the prior:

[
P(H\mid E)
\propto
P(E\mid H)P(H)
]

If the class prevalence changes, the prior changes.

Even if the likelihood terms remain approximately stable, the posterior probability may no longer be calibrated.

This is why models deployed under prior probability shift may need recalibration or explicit prior adjustment.

---

## 12. Interview-ready explanation

Probability provides the formal language for reasoning under uncertainty. I use events to represent possible outcomes, conditional probability to update the relevant population after observing evidence, and independence to determine whether probabilities can be factorized safely.

Expected value represents the long-run probability-weighted average of an outcome, while variance measures how much outcomes fluctuate around that average. In production systems, both matter because two models can have similar average performance but very different reliability or tail risk.

Bayes’ theorem connects prior probability, observed evidence and posterior probability. A common application is fraud detection: the probability that a detector raises an alert for fraudulent transactions is not the same as the probability that an alerted transaction is fraudulent. The posterior also depends strongly on the base rate.

In a real project, I would use these concepts for probabilistic classification, threshold selection, expected-cost optimization, A/B testing, reliability analysis, model calibration and multi-stage AI evaluation. I would also validate assumptions such as independence, stationarity and representative sampling before trusting the resulting probabilities.

---

## 13. GitHub file structure

```text
day-07-probability-essentials/
├── README.md
├── notes.md
├── notebook.ipynb
├── example.py
├── from_scratch.py
├── interview_questions.md
├── references.md
└── outputs/
    └── base_rate_effect.png
```

### Suggested responsibility of each file

```text
README.md
```

High-level objective, concepts, execution instructions and key takeaways.

```text
notes.md
```

Detailed theoretical notes, formulas, derivations, production considerations and common pitfalls.

```text
notebook.ipynb
```

Interactive simulations and visualizations:

* conditional probability;
* base-rate effect;
* expected value;
* variance;
* correlated failures;
* Monte Carlo convergence.

```text
example.py
```

Executable fraud detection simulation using NumPy, pandas and matplotlib.

```text
from_scratch.py
```

Educational implementations of:

* expected value;
* variance;
* conditional probability;
* binary Bayes update.

```text
interview_questions.md
```

Conceptual, mathematical and production-oriented interview questions.

```text
references.md
```

Books, courses, papers and documentation consulted.

```text
outputs/base_rate_effect.png
```

Generated visualization showing how the prior changes the posterior probability.

---

## 14. Suggested README.md content

You can paste the following content directly into the topic folder’s `README.md`.

# Day 7 — Probability Essentials

## Objective

This module reviews the probability concepts that support statistical inference, machine learning and reliable AI system design.

The main goal is to understand how to reason about uncertainty, update probabilities after observing evidence and connect probabilistic quantities to real engineering decisions.

## Concepts Covered

* Sample spaces and events
* Conditional probability
* Independence and conditional independence
* Law of total probability
* Bayes' theorem
* Expected value
* Conditional expectation
* Variance and standard deviation
* Covariance
* Base-rate effects
* Expected-loss decision rules

## Why It Matters

Probability appears throughout applied machine learning and AI engineering:

* probabilistic classification;
* fraud and anomaly detection;
* A/B testing;
* model calibration;
* risk analysis;
* retrieval and generation evaluation;
* reliability engineering;
* threshold selection;
* cost-sensitive decisions.

A model metric should not be interpreted without considering the population, class prevalence, uncertainty and business cost of errors.

## Repository Files

* `notes.md`: theoretical foundations, formulas and production considerations.
* `notebook.ipynb`: interactive experiments and visualizations.
* `example.py`: Monte Carlo simulation of a fraud detector.
* `from_scratch.py`: simplified probability functions implemented with NumPy.
* `interview_questions.md`: senior-level questions and answers.
* `references.md`: recommended learning resources.

## Installation

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux or macOS

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install numpy pandas matplotlib jupyter
```

## Running the Example

```bash
python example.py
```

Run the educational implementation:

```bash
python from_scratch.py
```

Open the notebook:

```bash
jupyter notebook notebook.ipynb
```

## Main Experiment

The practical example simulates a fraud detection system with:

* a low fraud base rate;
* a defined true-positive rate;
* a defined false-positive rate;
* an empirical Monte Carlo estimate;
* a theoretical Bayesian posterior.

The experiment demonstrates that a detector with high sensitivity may still produce mostly false-positive alerts when the target event is rare.

## Key Takeaways

1. Conditional probabilities depend on the direction of conditioning.
2. Model performance must be interpreted together with the base rate.
3. Independence is a strong assumption and should be justified.
4. Expected value describes a long-run average, not necessarily a typical outcome.
5. Variance and tail behavior are essential for evaluating production reliability.
6. Classification thresholds should reflect expected cost rather than defaulting to 0.5.
7. Model scores should not be treated as probabilities without calibration analysis.

## Production Perspective

In production AI systems, probability estimates can become unreliable because of distribution shift, selection bias, data leakage, dependence between components and poor calibration.

Reliable probabilistic decisions require both mathematical correctness and validation of the assumptions behind the model.

The README can later be expanded with generated charts and links to the notebook results.

## 15. LinkedIn post idea

Um modelo pode identificar 90% das fraudes e, ainda assim, a maioria dos alertas gerados por ele ser falsa.

Parece contraditório, mas é um efeito direto da probabilidade base.

Quando o evento que queremos detectar é raro, não basta perguntar:

“Qual é a chance de o modelo gerar um alerta quando existe fraude?”

Também precisamos responder:

“Qual é a chance de realmente existir fraude quando o modelo gera um alerta?”

As duas perguntas parecem semelhantes, mas representam probabilidades diferentes.

Esse é um dos pontos mais importantes do Teorema de Bayes e aparece diretamente em problemas de fraude, diagnóstico, detecção de anomalias, classificação e avaliação de sistemas de IA.

A principal conclusão prática é que métricas não devem ser interpretadas isoladamente. Prevalência, custo dos erros, calibração e contexto de produção alteram completamente a decisão.

Documentei os conceitos, fórmulas, armadilhas de interpretação e uma simulação em Python no meu repositório de estudos em Applied AI Engineering.

Esse post pode ser publicado junto de um gráfico mostrando como a probabilidade posterior muda conforme a taxa-base aumenta.

## 16. 30–60 minute checklist

### 30-minute essential path

* [ ] Read the executive overview and core intuition — 3 minutes.
* [ ] Review conditional probability and the multiplication rule — 5 minutes.
* [ ] Derive Bayes’ theorem from conditional probability — 5 minutes.
* [ ] Calculate the fraud example manually — 5 minutes.
* [ ] Review expected value and variance — 5 minutes.
* [ ] Run `example.py` — 5 minutes.
* [ ] Answer three interview questions aloud — 2 minutes.

### 45-minute recommended path

* [ ] Complete the 30-minute path.
* [ ] Review independence versus mutual exclusivity — 4 minutes.
* [ ] Review total probability and total expectation — 4 minutes.
* [ ] Run `from_scratch.py` — 4 minutes.
* [ ] Change the fraud base rate and rerun the experiment — 3 minutes.

### 60-minute complete path

* [ ] Complete the 45-minute path.
* [ ] Generate the base-rate visualization — 3 minutes.
* [ ] Simulate a lower false-positive rate — 3 minutes.
* [ ] Compare mean, median and percentiles on skewed latency data — 4 minutes.
* [ ] Write three production takeaways in `notes.md` — 3 minutes.
* [ ] Practice the interview-ready explanation without reading — 2 minutes.

### Minimum deliverables for the GitHub commit

* [ ] `README.md`
* [ ] `notes.md`
* [ ] `example.py`
* [ ] `from_scratch.py`
* [ ] `interview_questions.md`
* [ ] Generated base-rate chart
* [ ] At least one personal observation from the experiments
