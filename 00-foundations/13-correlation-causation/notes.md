# Technical Notes

## Association and intervention answer different questions

An association describes a feature of a joint distribution. A causal effect
compares outcomes under different interventions. Observing (X=x) preserves
the natural process that produced (X); applying (do(X=x)) replaces that
process with an external assignment.

This is why

\[
P(Y \mid X=x) \neq P(Y \mid do(X=x))
\]

in general. Equality requires causal conditions, not merely a better-fitting
statistical model.

A useful workflow is:

1. define the intervention and outcome;
2. define the estimand, such as an average treatment effect;
3. draw the assumed data-generating graph;
4. determine whether the design identifies the estimand;
5. only then choose an estimator and diagnostics.

Prediction reverses part of this priority: a downstream proxy can be highly
predictive even when it is useless or harmful as an intervention target.

## Correlation measures

For random variables (X) and (Y),

\[
\operatorname{Cov}(X,Y)
= E[(X-E[X])(Y-E[Y])]
\]

and Pearson population correlation is

\[
\rho_{X,Y}
= \frac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y}.
\]

The sample version centers both vectors and normalizes their dot product.
Pearson captures linear association and is sensitive to outliers, restricted
ranges, and nonlinear structure.

Spearman's rho is Pearson correlation applied to ranks:

\[
\rho_s = \operatorname{Corr}(R(X), R(Y)).
\]

It captures monotonic relationships and is useful for ordinal data. Ranking
reduces the influence of magnitudes, but it does not make the measure immune to
outliers, ties, selection, or confounding.

Ignoring ties, Kendall's tau compares concordant and discordant pairs:

\[
\tau = \frac{C-D}{\binom{n}{2}}.
\]

Tau-b adjusts the denominator for ties. It has a direct rank-agreement
interpretation but, like Pearson and Spearman, remains symmetric and
non-causal.

Zero Pearson correlation does not imply independence. If (X) has a symmetric
distribution around zero and (Y=X^2), the positive and negative linear
components cancel even though (Y) is determined by (X).

## Causal roles in a DAG

A directed acyclic graph records assumptions about causal direction. The graph
comes from domain knowledge, temporal order, system behavior, and study design;
it is not discovered merely by inspecting a correlation matrix.

### Confounder

\[
X \leftarrow Z \rightarrow Y
\]

(Z) is a common cause of treatment (X) and outcome (Y). The path creates
association even if (X) has no effect on (Y). A valid adjustment set blocks
all relevant backdoor paths without opening new ones.

### Mediator

\[
X \rightarrow M \rightarrow Y
\]

(M) transmits part of the effect. Adjusting for it can remove that pathway,
changing a total-effect question into a direct-effect question. Direct effects
require their own definitions and assumptions; “control for the mediator” is
not a neutral operation.

### Collider

\[
X \rightarrow C \leftarrow Y
\]

(C) is a common effect. The path is blocked until the analysis conditions on
(C) or, in many structures, one of its descendants. Selection, filtering,
stratification, or regression adjustment can therefore create an association
that was absent in the source population.

### Post-treatment variables

Variables measured after treatment can be mediators, colliders, consequences
of the outcome, or combinations of these roles. Temporal position alone does
not determine the role, but it is a warning that adjustment may change the
estimand or add bias.

## Omitted-variable bias

Suppose the data generator is

\[
Y = \beta X + \gamma Z + \varepsilon,
\]

but the fitted model omits (Z). In the simple linear setting, the expected
coefficient on (X) contains

\[
\widetilde{\beta}
= \beta
+ \gamma\frac{\operatorname{Cov}(X,Z)}{\operatorname{Var}(X)}.
\]

The second term is omitted-variable bias. Its sign depends on both the effect
of (Z) on (Y) and the association between (X) and (Z). Bias can be
positive, negative, or large enough to reverse the coefficient's sign.

Regression adjustment can recover the constructed coefficient in this topic's
example because:

- the common cause is observed without error;
- the graph contains no other relevant backdoor path;
- the relationships are linear and additive;
- the exposure has sufficient variation at values of the confounder;
- the noise does not introduce another treatment-outcome common cause.

These are assumptions of the example, not guarantees supplied by ordinary
least squares.

Partial correlation residualizes (X) and (Y) against controls and
correlates the residuals. It describes remaining linear association. It is
causally useful only when the controls form a valid adjustment set.

## Potential outcomes and identification

Let (Y_i(1)) be unit (i)'s outcome under treatment and (Y_i(0)) its outcome
without treatment. The average treatment effect is

\[
ATE = E[Y(1)-Y(0)].
\]

Only one potential outcome is observed for each unit. Identification connects
the missing counterfactual comparison to observable data. Common assumptions
include:

- **Consistency:** the observed outcome under the received treatment equals
  the corresponding potential outcome, and treatment versions are defined
  adequately.
- **Exchangeability:** treatment is independent of potential outcomes,
  unconditionally under ideal randomization or conditionally on an adequate
  observed adjustment set.
- **Positivity:** each treatment has positive probability for the covariate
  patterns to which the estimate applies.
- **No relevant interference:** one unit's treatment does not alter another
  unit's outcome, unless the estimand and design model that interference.
- **Measurement and model adequacy:** treatment, outcome, and adjustment
  variables represent the intended concepts, and estimation choices do not
  contradict the identification strategy.

Under a valid observed adjustment set (Z), the backdoor adjustment formula is

\[
P(Y \mid do(X=x))
= \sum_z P(Y \mid X=x,Z=z)P(Z=z).
\]

No estimator can test away unmeasured confounding in general. Sensitivity
analysis, negative controls, alternative designs, and triangulation help assess
how conclusions depend on assumptions.

## Causal traps

### Reverse causality

High-risk users may receive more support, so support contacts can correlate
with churn. The plausible direction (churn\ risk \rightarrow support)
must be considered before treating support as harmful.

### Shared time trends

Two unrelated quantities that grow over time can correlate strongly.
Detrending or differencing may address some association artifacts, but neither
operation identifies a causal effect without a credible design.

### Multiple comparisons

Searching thousands of features creates chance correlations. Holdout
validation, replication, and multiplicity control address false discoveries;
they still do not establish causal direction.

### Simpson's paradox

An aggregate association can reverse within every subgroup because treatment
groups contain different subgroup mixtures. The paradox does not say whether
the aggregate or stratified estimate is causal. That depends on whether the
grouping variable is a confounder, mediator, collider, or effect modifier and
on the target estimand.

### Proxy and measurement bias

Thumbs-up rate may stand in for satisfaction but also depend on UI placement,
survey exposure, and user segment. A rigorous estimator cannot repair an
outcome that fails to represent the decision objective.

### Feature importance as intervention value

SHAP values, permutation importance, and predictive coefficients explain a
model or improve prediction under their respective definitions. They do not
estimate (P(Y \mid do(X=x))). A feature may be predictive because it is
downstream of the outcome or shares causes with it.

## Applied AI questions

For a proposed RAG `top_k` change, first specify:

- treatment: the policy that assigns retrieval depth;
- unit: query, session, user, or account;
- outcome: a predefined quality metric plus latency, cost, and safety
  guardrails;
- assignment: randomized policy if feasible, including how routing and query
  difficulty are handled;
- interference and repeated measures: whether queries or users are clustered;
- estimand: average effect overall or within a defined traffic segment.

Production telemetry is valuable for generating hypotheses and monitoring.
When the decision is “what will happen if we change the system?”, the evidence
must match that intervention.

## Educational implementation limits

[`from_scratch.py`](from_scratch.py) uses centered sums and
`numpy.linalg.lstsq` to expose the mechanics. It intentionally omits standard
errors, confidence intervals, heteroskedasticity handling, clustered data,
nonlinear models, propensity methods, and sensitivity analysis. Use mature
statistical and causal-inference tools for real analyses.
