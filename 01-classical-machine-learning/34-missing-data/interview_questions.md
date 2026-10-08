# Missing-data interview questions

**1. Why does MAR not mean uniformly random missingness?**

MAR states conditional independence from missing values given observed
information. Different observed groups can have very different missing rates.
In the code, `x1` missingness depends on always-observed `x2`.

**2. Can an association between absence and age prove MAR?**

No. It can contradict MCAR, but dependence on an unavailable value may remain
after conditioning on age. MAR versus MNAR generally needs assumptions,
process knowledge, external information, or sensitivity analysis.

**3. What exactly goes wrong with mean imputation?**

If the observed-column mean fills all gaps, completed sample variance is
`(n_obs-1)/(n-1)` times observed-only sample variance. The mean is preserved
relative to the observed entries, which may already be selected. Coefficients,
covariance, and naive uncertainty estimates can be distorted.

**4. Why scale before KNN imputation?**

Distance uses the coordinates observed for both rows. Larger numerical units
otherwise dominate neighbor selection. Fit scaling on observed training values,
preserve NaNs, and retain training-only donors. Validate whether available
coordinates describe useful proximity.

**5. Is an iterative imputer automatically multiple imputation?**

No. A deterministic iterative result is one completed dataset. Proper multiple
imputation uses plausible stochastic draws, appropriate analysis models, and
pooling that accounts for within- and between-imputation uncertainty.

**6. What happens if a previously complete feature disappears at inference?**

A fit-time-only indicator schema does not add an indicator for it. The median
can still fill the gap. Native histogram trees with no training NaNs for that
feature route NaNs to the larger child. Neither behavior demonstrates acceptable
predictions; the example's `x2` outage tests this boundary.

**7. Can native NaN handling solve MNAR bias?**

It solves input handling and may use predictive missingness structure. It does
not identify unavailable values or eliminate bias for an inferential estimand.
In this comparison the classifier also changes, which prevents isolating a
native-routing effect from differences between learners.

**8. When is it appropriate to include the outcome in imputation?**

For some statistical inference tasks, an observed outcome belongs in the
imputation model to preserve associations. A predictive feature pipeline cannot
require the future outcome at serving time. Explain which task and information
boundary you mean before recommending target-aware imputation.

**9. Why does a missing-rate dashboard alone fail to establish robustness?**

Identical marginal rates can hide different joint patterns and populations.
Measure critical-feature availability, counts and errors by pattern, probability
quality, and downstream workload once outcomes become available. Stress tests
should freeze fitted models and keep evaluation labels and rows comparable.

**10. A document amount suddenly becomes 30% missing. What is the response?**

Investigate extraction errors, source mix, layout changes, and schema/join
failures; distinguish valid absence from technical failure. Compare availability
with training, evaluate the affected cohort, and use a previously evaluated
fallback or human review when necessary. Filling every gap is not sufficient
evidence to continue automated decisions.

**11. Does a larger test AUC make a workflow the chosen model?**

Selection belongs on validation data under a prespecified objective. The example
selects scaled KNN from validation, even though native boosting has higher
matched test AUC. Repeatedly choosing from test results converts the test set
into development data.
