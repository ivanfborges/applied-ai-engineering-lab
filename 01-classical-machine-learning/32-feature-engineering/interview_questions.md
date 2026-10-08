# Feature engineering interview questions

### Why can features help a linear classifier without adding new information?

A product or bin indicator expands the score functions the classifier can
represent. The score remains linear in its coefficients, while becoming
nonlinear in raw inputs. Historical aggregation is a different case: it adds
source information missing from a customer-only matrix.

### Does convergence make scaling irrelevant?

No. Even when optimization converges, scaling changes the coefficients needed
for an equivalent score and therefore their regularization cost. For KNN and
RBF kernels it also changes distance geometry. Trees generally need no change
of units because threshold partitions depend on order.

### When is ordinal encoding defensible?

When the order is meaningful. Integer codes still impose equal logit increments
in an additive linear model. If only order is justified, consider one-hot
indicators or an explicitly constrained model rather than assuming spacing.

### Why does target encoding need more than smoothing?

A training row's own label can influence its category statistic. Cross-fitting
or temporally prior statistics address that dependence; smoothing addresses
estimation variance. Folds must also respect entity/time boundaries. Neither
outer validation targets nor test targets may enter the transform.

### What happens to an unseen category in this example?

Its category block becomes all zeros. Prediction remains executable, but that
fallback has not been validated for unseen populations. A production policy
may instead use a learned infrequent bucket, monitor unknown rates, or reject
unsupported categories.

### What does retaining raw age alongside age bins mean?

A common within-bin slope plus bin-specific jumps in the logit. It preserves
some resolution but introduces discontinuities. Quantile intervals need not
match meaningful age regimes, and the bin count should be validated.

### Why avoid duplicate main-effect columns?

Identical columns allow a coefficient to be split between them, reducing its
L2 penalty. A polynomial feature branch that repeats income and balance can
silently change the effective penalty. This implementation returns one product
column and keeps each numeric main effect once.

### How do you specify a transaction feature precisely?

Define the entity, prediction cutoff, event-time window, availability cutoff,
amount/refund semantics, event deduplication, and no-history policy. Here the
window is `[t-30d,t)`, availability is `<t`, and absent-history mean/max are
zero accompanied by count zero. Test boundary equality and late arrivals.

### Can aggregation happen before splitting?

Customer-local summaries of eligible events can: they do not estimate population
parameters or use labels. Global frequencies and target means cannot be treated
the same way. Learned transformations need training-only fitting. A pipeline
cannot fix a future-contaminated input table.

### What does this measured comparison establish?

On one synthetic customer split the three prespecified representations produced
different ranking scores. The target deliberately favors history and nonlinear
terms. The final comparison bundles bins and interaction, uses fixed C, and
has no uncertainty estimate. It establishes neither general superiority nor
business value. The interpretation is pending author review.

### What would you check before serving these features?

Check schema and versioned definitions, correct timestamp conversion, event
availability/revisions, consistent refund and no-history behavior, unknown
categories, and parity between offline and serving outputs. A repeated-customer
task also needs an evaluation split matching the deployment question. The small
join here is an educational implementation.
