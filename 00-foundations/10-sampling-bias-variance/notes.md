# Technical Notes: Sampling, Bias, and Variance

## From a population to a decision

Statistical reasoning begins by naming the target population and estimand.
Without them, “representative” has no operational meaning.

```text
target population
    -> sampling frame and inclusion mechanism
    -> observed sample
    -> estimator
    -> estimate and uncertainty
    -> decision
```

A **parameter** is a fixed, usually unknown population quantity such as a mean
\(\mu\) or rate \(p\). An **estimator** \(\hat\theta\) is a rule applied to a
random sample. An **estimate** is the value returned for one realized sample.
The distribution of estimates over hypothetical repetitions is the
estimator's **sampling distribution**.

## Three meanings of bias

### Sampling bias

Sampling bias concerns who or what enters the data. If inclusion \(S=1\)
depends on an outcome or a related characteristic, then the sample distribution
can differ systematically from the target:

\[
P(S=1 \mid X, Y) \neq P(S=1).
\]

Undercoverage, convenience sampling, self-selection, non-response, and
survivorship are common mechanisms. The important object is the selection
process, not just the realized sample histogram.

### Estimator bias

For an estimator \(\hat\theta\) of parameter \(\theta\),

\[
\operatorname{Bias}(\hat\theta)=E[\hat\theta]-\theta.
\]

This expectation is defined under an assumed sampling process. The sample mean
can be unbiased for the distribution actually sampled and still be wrong for a
different target population. “The sample mean is unbiased” is therefore an
incomplete statement unless the target and design are explicit.

### Model bias

Model bias is systematic prediction error from assumptions or restrictions in
a fitted model—for example, fitting a line to a nonlinear signal. It is not a
synonym for sampling or estimator bias. A more flexible model may reduce model
bias without repairing coverage or selection problems.

## Variance, standard error, and MSE

Estimator variance measures instability across samples:

\[
\operatorname{Var}(\hat\theta)
=E[(\hat\theta-E[\hat\theta])^2].
\]

Its square root is the **standard error**. Standard deviation describes the
spread of observations; standard error describes the spread of an estimator.

For IID observations with variance \(\sigma^2\),

\[
E[\bar X]=\mu,
\qquad
\operatorname{Var}(\bar X)=\frac{\sigma^2}{n},
\qquad
SE(\bar X)=\frac{\sigma}{\sqrt n}.
\]

Halving standard error therefore requires approximately four times as many
independent observations. This relationship addresses random variation, not a
systematic population mismatch.

For squared error,

\[
\operatorname{MSE}(\hat\theta)
=E[(\hat\theta-\theta)^2]
=\operatorname{Var}(\hat\theta)
+\operatorname{Bias}(\hat\theta)^2.
\]

An unbiased estimator is not automatically preferable: a small bias can be
worth accepting if it produces a sufficiently large variance reduction.
Regularization is a familiar modeling example of that trade-off.

## Finite populations and dependence

For a simple random sample of size \(n\), drawn without replacement from a
finite population of size \(N\), the mean variance is

\[
\operatorname{Var}(\bar X)
=\frac{\sigma^2}{n}\frac{N-n}{N-1}.
\]

The finite population correction approaches one when \(n\) is small relative
to \(N\), and reaches zero when the entire population is observed.

The familiar \(\sigma^2/n\) result also assumes independence. In general,

\[
\operatorname{Var}\left(\sum_i X_i\right)
=\sum_i\operatorname{Var}(X_i)
+2\sum_{i<j}\operatorname{Cov}(X_i,X_j).
\]

Repeated requests from a user, frames from a video, documents from one case,
or transactions from one customer can be positively correlated. A row-level
bootstrap or random split then overstates the effective independent sample and
can leak entity-specific information. Use a group-aware design when the
deployment target is unseen entities and a temporal design when the target is
the future.

## Comparing sampling designs

### Simple random sampling

Each possible fixed-size sample has the same probability. It offers clear
design-based interpretation, but a rare or high-risk group may contribute too
few cases for subgroup evaluation.

### Stratified sampling

Partition the population into strata and sample within every stratum. If
\(W_h=N_h/N\) and \(\bar X_h\) is a stratum sample mean, the population mean
estimator is

\[
\bar X_{\text{strat}}=\sum_{h=1}^{H}W_h\bar X_h.
\]

The weights are target-population shares, not necessarily sample shares.
Oversampling a rare group can support diagnosis while population weighting
preserves the aggregate estimand. Variance can fall when strata explain
between-unit heterogeneity, but allocation quality and correct weights matter.

### Cluster sampling

Select groups such as sites, organizations, or schools, then observe units
inside selected clusters. It can lower collection cost, but within-cluster
correlation usually makes it less statistically efficient than an equally
sized independent sample. Variance estimation must reflect the design.

### Systematic sampling

Choose a random start and every \(k\)-th unit. This is operationally simple but
can align with periodic workloads, batch jobs, or sorted records. Randomizing
the start does not neutralize every periodic structure.

### Convenience sampling

Use readily available observations, such as voluntary feedback or already
reviewed incidents. It can be useful for discovery and debugging, but its raw
frequency usually should not be interpreted as a population prevalence.

## Weighting and effective sample size

If unit \(i\) has known inclusion probability \(\pi_i\), a design weight is
often proportional to \(1/\pi_i\). A normalized weighted mean is

\[
\hat\mu_w=\frac{\sum_i w_iX_i}{\sum_iw_i}.
\]

This correction depends on valid inclusion probabilities or a credible model
for them. It cannot identify groups absent from the sampling frame, and it can
amplify noise when a few units have very large weights.

Kish's approximation summarizes weight concentration:

\[
n_{\text{eff}}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
\]

Equal positive weights give \(n_{\text{eff}}=n\). Unequal weights reduce it.
This is not a universal effective-sample-size formula: clustering,
autocorrelation, estimator choice, and the relationship between weights and
outcomes may require a design-specific calculation.

## Data quality is broader than sample size

Sampling quality asks whether the observed distribution supports the target
inference. Other dimensions remain distinct:

- measurement validity and stable definitions;
- label and annotation quality;
- missingness and response mechanisms;
- duplicates and dependent observations;
- temporal, geographic, language, and device coverage;
- lineage, freshness, and prediction-time availability.

Millions of precisely measured observations can still omit the population that
matters. A representative sample can still contain incorrect labels. These
failures require different diagnostics and remedies.

## Applied AI patterns

### Training data

Oversampling rare outcomes can improve optimization or batch composition, but
it changes the empirical class prior. Validate against production prevalence,
check calibration, and choose decision thresholds using deployment costs.

### LLM and RAG evaluation

Maintain separate views rather than one overloaded score:

- a probability sample or well-weighted approximation for expected production
  performance;
- a diagnostic set balanced across capabilities and failure modes;
- dedicated rare-event, safety, or high-risk suites.

A benchmark made from clean developer-written queries may underrepresent
misspellings, ambiguity, multilingual traffic, long conversations, or document
types that fail during ingestion.

### Monitoring and annotation

Uniform traffic sampling estimates common behavior efficiently. Triggered and
risk-stratified samples expose rare failures. Do not pool their unweighted
rates. For annotation, uncertainty or diversity sampling may maximize learning
per label, but such active samples are not automatically prevalence estimates.

## Assumptions in the executable example

The example intentionally fixes a narrow design:

- a finite synthetic population with exactly two observed segments;
- independent code-generated spending values conditional on segment;
- sampling without replacement within each trial;
- known empirical population shares for stratified weighting;
- an eightfold premium selection weight as a visible selection mechanism;
- population variance (`ddof=0`) across the complete set of simulated
  estimates, so the empirical MSE identity uses one convention consistently.

The example does not establish how a real inclusion mechanism works. Its
stratified variance advantage follows from the constructed segment separation
and allocation; another population or allocation could behave differently.

## Common mistakes

- Claiming a large dataset is representative because its row count is large.
- Calling an estimator unbiased without stating the target and sampling
  assumptions.
- Reporting a balanced benchmark average as expected production performance.
- Oversampling a class and interpreting unadjusted probabilities as production
  probabilities.
- Treating repeated rows from one entity as independent.
- Using random splits for future prediction or group-generalization tasks.
- Applying weights without inspecting their distribution, origin, and
  sensitivity.
- Assuming the central limit theorem repairs selection bias, leakage, bad
  labels, or distribution mismatch.
- Tuning repeatedly on a nominal test set, making it part of model selection.

## Suggested extensions (not executed here)

- Repeat the simulation across sample sizes and compare empirical variance to
  the finite-population formula.
- Make premium prevalence rare and compare subgroup coverage under random and
  stratified sampling.
- Increase the premium selection multiplier while tracking bias and variance.
- Compare row-level with customer-level bootstrap uncertainty in clustered
  synthetic data.
- Perturb or clip unequal weights and track both the estimate and effective
  sample size.

These remain proposed investigations until their configurations and results
are executed and recorded.
