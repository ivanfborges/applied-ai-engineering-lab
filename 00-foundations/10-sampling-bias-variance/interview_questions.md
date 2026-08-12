# Interview Questions: Sampling, Bias, and Variance

## 1. What is the difference between sampling bias and estimator bias?

Sampling bias is a mismatch created by the inclusion mechanism: some relevant
units are systematically over- or underrepresented. Estimator bias is the
mathematical difference \(E[\hat\theta]-\theta\) under a stated sampling
process. An estimator can be unbiased for the sampled distribution yet
misleading for a different target population.

## 2. Why does more data not necessarily fix bias?

For an IID sample mean, variance decreases as \(\sigma^2/n\). If selection
changes the distribution being sampled, increasing \(n\) can make the estimate
more stable around the wrong expectation. Sample size addresses random
uncertainty; representativeness addresses the target of inference.

## 3. Standard deviation or standard error?

Standard deviation describes variability among observations. Standard error
describes variability of an estimator over repeated samples. The distinction
matters because a heterogeneous population can have a large standard deviation
while a mean estimated from many independent observations has a small standard
error.

## 4. Why can stratification reduce variance?

When strata explain important heterogeneity and units within each stratum are
relatively homogeneous, estimating stratum means separately removes
between-stratum composition noise. The aggregate must use target-population
weights. Poor strata, poor allocation, or incorrect weights can remove the
advantage.

## 5. Stratified sampling versus cluster sampling?

Stratified sampling draws units from every stratum to guarantee representation
or improve precision. Cluster sampling selects only some groups, often to
reduce collection cost. Because units within a cluster tend to be correlated,
the same row count usually contains less independent information.

## 6. When would you oversample a rare class?

When the natural prevalence provides too few examples for effective training,
diagnosis, or subgroup evaluation. I would keep the altered training or
diagnostic distribution separate from a representative validation view, and I
would check probability calibration and thresholds against production
prevalence.

## 7. What does effective sample size tell you?

For survey weights, Kish's approximation
\((\sum w_i)^2/\sum w_i^2\) measures loss of efficiency from weight
concentration. Equal weights recover the nominal size; unequal weights reduce
it. It is an approximation, not a replacement for dependence- and
design-aware variance estimation.

## 8. Why can row-level cross-validation be invalid?

Rows from the same user, patient, document, case, or conversation may be
dependent. Putting one entity in both train and validation lets the model use
entity-specific patterns and yields an optimistic estimate for unseen-entity
generalization. Use group-aware splits when the production unit is a group and
time-aware splits when the production task predicts the future.

## 9. How would you sample queries for an LLM or RAG evaluation?

I would define the estimand first. A production-weighted sample estimates
expected traffic performance; a balanced capability set diagnoses categories;
and a targeted risk suite tests rare critical failures. I would preserve those
views separately, track their sampling frames and weights, and segment by
language, query type, document source, complexity, and time where relevant.

## 10. Would you always choose an unbiased estimator?

No. Squared-error risk is variance plus squared bias. A slightly biased
estimator can have lower MSE if it reduces variance enough. The decision also
depends on the loss function: a small average bias may still be unacceptable
for safety, fairness, or regulated subgroup guarantees.

## 11. What assumptions support \(SE(\bar X)=\sigma/\sqrt n\)?

The usual expression assumes independent observations with a common variance
under the stated sampling model. Sampling without replacement from a
non-negligible finite population introduces a correction. Clustering,
autocorrelation, unequal weights, or other complex designs require an adjusted
variance estimator.

## 12. A model scores well offline and poorly in production. How would you
investigate sampling?

Compare the evaluation sampling frame with production across time, geography,
language, device, entity, use-case complexity, and rare failure modes. Check
whether inclusion or annotation depends on outcomes, whether repeated entities
cross splits, whether test examples influenced model selection, and whether
the offline metric was weighted for the intended production estimand.

## Interview-ready summary

Sampling determines which population an estimate can speak about. Estimator
bias is systematic error under a stated design, while estimator variance is
instability across repeated samples; for squared error they combine as
\(MSE=Variance+Bias^2\). More independent, representative data reduces
variance, but a larger convenience sample can become precisely wrong. In an AI
system I separate representative production evaluation from balanced
diagnostics and rare-risk suites, restore population weights when needed, and
make group or temporal boundaries match the deployment unit.
