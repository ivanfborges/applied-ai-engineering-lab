# Linear Regression: Interview Questions

## 1. Derive OLS and state the rank condition.

Minimize RSS = (y - Dβ)ᵀ(y - Dβ). Setting its gradient to zero yields
DᵀDβ̂ = Dᵀy. Full column rank gives a unique coefficient vector. Solve the
least-squares system numerically without explicitly inverting DᵀD. With rank
deficiency, fitted training values remain unique, while individual
coefficients need not be.

## 2. Training residuals have zero mean. Have you verified the model?

No. With an intercept, that is an algebraic property of the fitted OLS
objective. It holds for a misspecified model too. The
[quadratic experiment](notes.md#experiment-2-missing-curvature-despite-training-orthogonality)
shows this directly. Inspect residual structure and held-out performance;
neither alone establishes exogeneity.

## 3. Which assumption supports unbiased coefficients, and which supports BLUE?

For a full-rank specified model, zero conditional error mean supports
conditional unbiasedness. Constant error variance and uncorrelated errors add
the Gauss-Markov minimum-variance guarantee among linear unbiased estimators.
Normal errors are unnecessary for either statement. Gaussian errors support
classical exact finite-sample inference.

## 4. What changes when errors are heteroskedastic?

Under exogeneity, coefficients remain conditionally unbiased, but usual
homoskedastic standard errors are generally invalid. Robust covariance
estimation addresses uncertainty under appropriate conditions, not omitted
variables or predictive misspecification. Dependence requires additional
care, such as cluster or temporal covariance treatment.

## 5. Does an x coefficient of 3 always mean a one-unit change adds 3?

In an additive model, it changes the fitted response by 3, holding other
predictors fixed. With x² or interactions, the change depends on the input.
The units and observed support matter. None of these statements alone gives
a causal effect.

## 6. Can unstable coefficients coexist with good predictions?

Yes. Near-collinear features can redistribute coefficients while preserving
their combined contribution on familiar inputs. That does not guarantee
stable predictions when the feature relationship changes. Examine the
intended interpretation and deployment distribution before dropping
predictors or adding regularization.

## 7. Why can test R² be negative?

The prediction RSS can exceed the sum of squared deviations from the test
target mean. That mean is a scoring reference, not a model learned from
training data. Compare against a training-mean baseline as well. With an
intercept, exact training OLS has nonnegative R² for nonconstant targets.

## 8. How do leverage, an outlier, and influence differ?

Leverage concerns unusual predictor values; a response outlier has a large
residual; influence concerns sensitivity of the fitted model to an
observation. A point can have high leverage and a small residual because it
pulls the fitted model toward itself. Check data validity and sensitivity
before removing it.

## 9. Why is a prediction interval wider than a mean-response interval?

It includes uncertainty in the estimated mean plus noise in a new
observation. Both depend on modeling assumptions; neither automatically
captures distribution shift or a missing mechanism.

## 10. Would a low latency RMSE justify deploying the model?

Check the decision first: mean latency forecasting and tail-latency
guarantees are different requirements. Verify feature availability, temporal
validation, errors by workload and concurrency, and behavior outside the
training range. Output-token counts observed after a call cannot be treated
as known before that call. No deployment recommendation is inferred from the
synthetic example.

## Short explanation to practice

OLS finds the linear combination of a chosen feature set that minimizes
squared residuals. Its training geometry holds even with a poor feature set.
Statistical interpretation requires assumptions about the conditional mean,
rank, and error process. I would assess residual structure, representative
held-out error, coefficient stability, and feature availability, and keep
causal claims separate from predictive associations.
