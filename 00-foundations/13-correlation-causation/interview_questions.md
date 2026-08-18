# Interview Questions

## 1. Why does correlation not imply causation?

Correlation is a property of an observed joint distribution; causation compares
outcomes under interventions. They can differ because of confounding, reverse
causality, selection, aggregation, time trends, and measurement processes.
The stronger answer is not the slogan but a plausible data-generating graph
and a design that identifies the requested intervention effect.

## 2. Can Pearson correlation be zero when variables are dependent?

Yes. With symmetric (X) and (Y=X^2), positive and negative linear
components cancel, so Pearson correlation is zero even though (Y) is
determined by (X). Pearson measures linear association, not arbitrary
dependence.

## 3. How do Pearson, Spearman, and Kendall correlation differ?

Pearson measures linear association in the original values. Spearman computes
Pearson correlation on ranks and targets monotonic association. Kendall's tau
compares concordant and discordant pairs and has a rank-agreement
interpretation. None determines causal direction or protects against
confounding.

## 4. What is a confounder?

A confounder is a common cause in a path such as
(X \leftarrow Z \rightarrow Y). It makes treatment groups systematically
different before treatment. Conditioning on an adequate set of pre-treatment
confounders can block backdoor paths under assumptions, but merely adding
correlated covariates is not sufficient.

## 5. Why can adding variables to a regression make a causal estimate worse?

A variable might be a mediator, collider, post-treatment measurement, or
descendant of a collider rather than a confounder. Controlling for a mediator
can remove part of the effect of interest; controlling for a collider can open
a non-causal path. Variable selection should follow the causal question and
graph, not predictive fit alone.

## 6. What is collider bias?

For (X \rightarrow C \leftarrow Y), (C) is a common effect. The path
between (X) and (Y) is initially blocked. Conditioning or selecting on
(C) can make its causes statistically dependent. Hiring only candidates
above a threshold based on technical and communication skill can create a
negative skill correlation among hires even when the skills are independent
in the applicant population.

## 7. What does Simpson's paradox teach?

An aggregate association can reverse within subgroups because treatment and
outcome groups have different compositions. It teaches that aggregation is
part of the data-generating process. It does not automatically tell us which
estimate to use; the graph and estimand determine whether stratification
removes confounding or introduces a different bias.

## 8. Why is a regression coefficient not automatically causal?

Regression estimates a conditional association. A causal reading additionally
requires a well-defined intervention and estimand, an identification strategy,
adequate control of confounding, positivity, consistency, appropriate temporal
ordering, valid measurement, and a suitable model. The algorithm cannot verify
all those conditions from fit statistics.

## 9. Why does randomization help?

Proper random assignment makes treatment independent of potential outcomes in
expectation. It breaks systematic links from pre-treatment common causes to
treatment, allowing group differences to estimate the assigned-treatment
effect under assumptions such as consistency, limited interference, and valid
measurement. Noncompliance, attrition, spillovers, and implementation errors
can still complicate interpretation.

## 10. What is the difference between (P(Y\mid X)) and
(P(Y\mid do(X)))?

(P(Y\mid X)) describes outcomes among units whose treatment was observed.
(P(Y\mid do(X))) describes outcomes after externally assigning treatment,
which cuts the natural causes of treatment. They coincide only under
appropriate causal conditions.

## 11. How would you test whether increasing RAG `top_k` improves quality?

Define the treatment policy, unit, target traffic, quality metric, and latency,
cost, and safety guardrails before analysis. Production `top_k` may be
confounded by query difficulty and routing. When feasible, randomize among
acceptable `top_k` policies, preserve query- or user-level clustering in the
analysis, and report effect uncertainty and heterogeneous effects. If
randomization is unavailable, justify a graph and identification strategy and
perform sensitivity analyses.

## 12. Is SHAP a causal method?

Not by itself. SHAP attributes a model prediction according to a specified
value function and background distribution. It does not estimate the outcome
under an intervention. Predictive importance can suggest hypotheses, but it is
not intervention value.

## 13. When is correlation enough?

Correlation is appropriate for descriptive EDA, redundancy checks, monitoring,
and hypothesis generation when no intervention claim is made. Causal reasoning
is needed when the question contains “what happens if we change (X)?”

## 14. Give a concise senior-level explanation.

Correlation describes observed co-movement; causation concerns a defined
intervention and counterfactual outcome. Before estimating an effect, I specify
the estimand and reason about treatment assignment, confounders, mediators,
colliders, selection, and time ordering. I prefer randomization when feasible.
For observational data, I state the graph and identification assumptions,
choose an estimator consistent with them, check overlap and robustness, and
avoid presenting model fit as causal evidence.
