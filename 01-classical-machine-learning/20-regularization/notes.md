# Regularization: objectives, geometry, and evidence

## One objective, explicit conventions

Let \(X\in\mathbb R^{n\times p}\), \(y\in\mathbb R^n\), intercept \(b\), and
coefficients \(w\). This study uses

\[
J(b,w)=\frac{1}{2n}\|y-b\mathbf1-Xw\|_2^2
+\alpha\rho\|w\|_1+\frac{\alpha(1-\rho)}2\|w\|_2^2.
\]

The intercept is unpenalized. With \(\alpha>0\), \(\rho=0\) gives Ridge,
\(\rho=1\) gives Lasso, and \(0<\rho<1\) gives ElasticNet. With \(\alpha=0\),
the objective is OLS regardless of \(\rho\). ElasticNet's `l1_ratio` is
**exactly** \(\rho\) for this convention.

Scikit-learn's ElasticNet uses this normalization. Ridge instead minimizes
SSE plus `alpha` times the squared coefficient norm, so matching the pure-L2
objectives requires \(\alpha_{\text{Ridge}}=n\alpha\). Equal numerical alpha
values across these APIs are not equal penalties. See the
[ElasticNet objective](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
and [Ridge objective](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

The example tunes each API's native alpha separately. The Ridge grid therefore
does not represent a common normalized penalty grid with Lasso/ElasticNet.
Its fixed native alpha also corresponds to different normalized strengths
during CV fitting and full-development refitting because their sample counts
differ. That is the usual native-parameter search, not a matched-penalty study.

## Ridge: weak directions and bias

Centering \(X,y\) removes the intercept from the optimization. Writing the
centered arrays as \(X_c,y_c\),

\[
(X_c^\top X_c+n\alpha I)\hat w=X_c^\top y_c,\qquad
\hat b=\bar y-\bar X^\top\hat w.
\]

For \(\alpha>0\), the coefficient system is positive definite even when the
design is rank deficient. Solve the linear system or use SVD; explicitly
forming an inverse adds numerical error and unnecessary work.

With \(X_c=UDV^\top\), Ridge has

\[
\hat w=V\,\mathrm{diag}\left(\frac{d_j}{d_j^2+n\alpha}\right)U^\top y_c.
\]

Relative to an identifiable OLS direction, the shrinkage multiplier is
\(d_j^2/(d_j^2+n\alpha)\). Small singular values are damped most. This explains
why nearly duplicate predictors can produce large opposing OLS coefficients
while Ridge prefers a smaller combination with similar fitted predictions.
Null directions have zero contribution for positive alpha.

The L2 norm is nonincreasing as the penalty strengthens, but this does **not**
guarantee that every coefficient's absolute value decreases: mixing singular
directions back into the feature basis can change signs or magnitudes.
Ridge can have zeros by symmetry or zero signal; it has no L1-style threshold
that generally makes whole ranges of weak signals exactly zero.

Under a correctly specified fixed-design linear model with mean-zero,
homoscedastic errors of variance \(\sigma^2\), write
\(A=X_c^\top X_c\). For coefficients, conditional on \(X_c\),

\[
E[\hat w\mid X_c]-w=-n\alpha(A+n\alpha I)^{-1}w,
\]
\[
\mathrm{Cov}(\hat w\mid X_c)
=\sigma^2(A+n\alpha I)^{-1}A(A+n\alpha I)^{-1}.
\]

These equations describe fixed alpha, not the additional randomness of
data-driven tuning. In identified eigendirections the variance is reduced,
at the cost of bias toward zero. A prediction's error at a fixed input has
the decomposition

\[
E_{D,\epsilon}[(y-\hat f_D(x))^2]
=(E_D[\hat f_D(x)]-f(x))^2
+\mathrm{Var}_D(\hat f_D(x))+\sigma^2(x).
\]

Reducing variance can outweigh increased squared bias. Excessive shrinkage
underfits; regularization does not guarantee a lower realized test error.
One fitted model cannot estimate this repeated-sampling decomposition.

## Lasso: why zero is an optimum

For a zero coefficient, the subgradient of absolute value is the interval
\([-1,1]\). Set \(r=y_c-X_cw\). Lasso's optimality conditions are

\[
X_{c,j}^\top r/n=\alpha\,\mathrm{sign}(w_j)\quad(w_j\ne0),
\]
\[
|X_{c,j}^\top r/n|\le\alpha\quad(w_j=0).
\]

A range of residual correlations is therefore compatible with a zero
coefficient. For centered orthonormal columns, \(X_c^\top X_c/n=I\),
let \(z=X_c^\top y_c/n\). Then

\[
\hat w_j=S(z_j,\alpha),\qquad
S(z,t)=\mathrm{sign}(z)\max(|z|-t,0).
\]

For ElasticNet in the same orthonormal setting,

\[
\hat w_j=\frac{S(z_j,\alpha\rho)}{1+\alpha(1-\rho)}.
\]

The tests use this independent analytic solution. For general correlated
designs, thresholding the OLS coefficients once is not the Lasso solution.
At \(\alpha\ge\|X_c^\top y_c\|_\infty/n\), pure Lasso admits the all-zero
coefficient vector, while retaining the target mean as its intercept.

The constrained view minimizes squared error subject to an L1 or L2 budget.
L1's two-dimensional feasible set has corners on the axes, where a loss
contour often first touches it; the L2 boundary is smooth. A corresponding
constraint budget exists for a penalized optimum, but there is no universal
numeric conversion between that budget and alpha.

## Correlated features and selection

If two columns provide almost the same information, many coefficient
allocations predict similarly. Lasso may keep one and discard the other;
small data perturbations can alter that choice. With perfectly duplicate
columns, its coefficients may be nonunique even when fitted predictions
are unique.

A positive L2 component makes the coefficient objective strictly convex and
encourages similar weights for equally scaled, similarly predictive columns.
For identical columns, the unique ElasticNet solution assigns equal weights.
This grouping tendency does not guarantee that every correlated feature
will be selected or that feature identities will be stable under sampling.

A sparse predictive representation is useful for review and sometimes for
feature-computation cost. Actual cost savings require removing unnecessary
upstream computations too. Zero coefficients are not evidence of absent
causal effects, scientific irrelevance, or dispensability under future shift.
Correlated proxies with zero direct coefficients in a synthetic generator
can still carry useful predictive information.

## Scaling, priors, and interpretation

For \(z_j=(x_j-\mu_j)/s_j\), a standardized coefficient measures a change in
target units per training standard deviation of that predictor. Convert back
using

\[
w_{\text{original},j}=w_{\text{standardized},j}/s_j,\qquad
b_{\text{original}}=b_{\text{standardized}}
-\sum_j\mu_j w_{\text{standardized},j}/s_j.
\]

Changing units without scaling changes the effective regularization. Scaling
is itself a modeling choice: rare binary indicators, outliers, and domain
constraints can justify alternatives to unit variance. The example's reported
norms all refer to the same full-development standardized feature space.
They cannot be compared directly with the generator's raw-unit coefficients.

For independent Gaussian errors with known variance \(\sigma^2\), a Gaussian
coefficient prior with variance \(\tau^2\) yields Ridge MAP with
\(\alpha=\sigma^2/(n\tau^2)\) in this convention. A Laplace prior with scale
\(s\) yields Lasso MAP with \(\alpha=\sigma^2/(ns)\). These are statements about
a posterior **mode**. A continuous Laplace prior does not assign positive
posterior probability mass to an exactly zero coefficient.

Ordinary OLS standard errors and p-values do not become valid post-selection
inference just because a penalty was used. Prediction, uncertainty about
parameters, and causal identification remain different questions.

## Validation boundaries

Split off test rows first. Search over
`Pipeline(StandardScaler(), estimator)` so every fold learns its own
preprocessing statistics. A scaler before `LassoCV` or `ElasticNetCV` is fitted
once before those estimators' internal folds; use an outer search of the full
pipeline for strictly fold-local preprocessing. The implementation follows
the [official pipeline guidance](https://scikit-learn.org/stable/common_pitfalls.html).

Use a metric matched to the decision. Here it is mean fold RMSE, not the square
root of pooled mean squared error. Refit the chosen configuration on the full
development set and evaluate the reserved test set. The script selects the
family using CV before printing all four predeclared test results. Changing
the family or search grid after inspecting those test results would make the
test set part of development. CV scores optimized over many configurations
also carry selection optimism; nested CV is an option for repeated estimates.

IID shuffled folds are appropriate for this generator. Forecasting, repeated
customers, and delayed outcomes need temporal or group boundaries. Penalties
cannot compensate for invalid splits, mislabeled outcomes, unavailable
features, outliers under squared loss, or nonlinear misspecification.

## What the educational solver implements

The smooth part of the objective has gradient

\[
g(w)=X_c^\top(X_cw-y_c)/n+\alpha(1-\rho)w.
\]

Its Lipschitz constant is \(L=\|X_c\|_2^2/n+\alpha(1-\rho)\). The implementation
uses \(\eta=1/L\) and iterates

\[
w^+=S(w-\eta g(w),\eta\alpha\rho).
\]

Centering yields the unpenalized intercept; scaling remains the caller's
responsibility. If \(L=0\), the zero initial coefficients already satisfy
optimality. The solver measures the largest KKT violation: on active
coordinates it uses \(|g_j+\alpha\rho\,\mathrm{sign}(w_j)|\); on inactive
coordinates it uses \(\max(|g_j|-\alpha\rho,0)\).

The absolute stopping tolerance depends on data units. A tiny update alone
would be misleading with a tiny learning rate. A finite iteration budget
returns `converged=False` rather than implying success. Objective history
allows a descent check. The solver supports dense, finite, single-target
arrays only; it lacks sparse matrices, sample weights, acceleration,
warm-start paths, and a production dual-gap stopping rule. Its SVD step-size
calculation and stored history target small educational problems.

## Executed experiments

These runs were executed by Codex during implementation. Interpretation
candidates below require author review; none is an author's personal conclusion.
Commands are in [README.md](README.md). Environment: Windows, Python 3.11.0rc2,
NumPy 2.4.6, scikit-learn 1.9.0. This records the actual local environment;
the repository declares Python 3.11+.

### 1. Prediction, shrinkage, and sparsity

**Hypothesis:** regularization can reduce large opposing coefficients and
produce sparse fits without a large change in held-out prediction error on
a sparse linear problem with redundant predictors.

**Configuration:** `example.py`; NumPy generator seed 20; 500 rows and 30
initial standard-normal features. Columns 28 and 29 are replaced by columns
0 and 1 plus independent normal noise of SD 0.05. The target is generated
after this replacement: intercept 8, coefficients
`[4, -3, 2, 1.5, -1, 0, ...]`, Gaussian noise SD 4. Feature scales are then
multiplied by `geomspace(0.1, 100, 30)`, with the true coefficients converted
accordingly. No external dataset is used.

The split uses seed 20 and reserves 25% (125 rows) for test. Five shuffled
development folds use seed 20. Ridge searches 13 log-spaced native alphas
from \(10^{-3}\) to \(10^3\); Lasso and ElasticNet search 13 from \(10^{-3}\)
to \(10^1\). ElasticNet searches ratios 0.2, 0.5, and 0.8. L1 solvers use
100,000 maximum iterations and tolerance \(10^{-7}\); convergence warnings
are treated as errors. Ridge uses SVD. Scaling is inside the searched
pipeline; all candidates use the same splits.

**Result:** development CV selected Lasso before test evaluation.

| Model | Selected native alpha | Ratio | CV RMSE | Train RMSE | Test RMSE | Exact zeros / 30 | Coefficient L2 norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLS | 0 (no penalty) | — | 4.1640 | 3.7893 | 4.2396 | 0 | 16.1329 |
| Ridge | 31.6228 | — | 4.1436 | 3.8377 | 4.2616 | 0 | 4.0495 |
| Lasso | 0.215443 | 1 | 3.9971 | 3.8902 | 4.2385 | 20 | 5.0361 |
| ElasticNet | 0.215443 | 0.8 | 4.0293 | 3.8979 | 4.2494 | 17 | 3.8151 |

For the first correlated pair (x0, x28), standardized coefficients were
(10.0815, -6.3665) for OLS, (1.9057, 1.6529) for Ridge, (3.5490, 0) for
Lasso, and (2.0172, 1.4861) for ElasticNet.

**Interpretation candidate — author review required:** the chosen penalties
substantially changed parameter allocation while test RMSE stayed similar.
Lasso's test advantage over OLS was only about 0.0011 target units and does
not establish a meaningful advantage. Its 10 retained coefficients also
exceed the five direct effects in the generator; sparsity did not recover
exactly the generating support.

**Limitation:** one generator and split, a finite search grid, no repeated
sampling, and no interval on score differences. This comparison does not
measure coefficient variance, selection stability, or a bias-variance
decomposition. The test table must not become a second tuning stage.

### 2. Soft thresholding and convergence

**Hypothesis:** proximal updates can converge to a sparse ElasticNet optimum
without penalizing the intercept.

**Configuration:** `from_scratch.py`; seed 20; 300 rows, four independent
standard-normal predictors, target intercept 3, coefficients (4, 0, -2, 0),
normal noise SD 1. The predictors are standardized on these demonstration
rows. Alpha 0.1, ratio 0.8, KKT tolerance \(10^{-8}\), budget 50,000 updates.

**Result:** 12 updates; KKT residual \(1.712\times10^{-9}\); objective decreased
from 10.649449 to 1.164334. Standardized coefficients:
(3.994716, 0, -1.770931, 0); standardized-space intercept 2.627653.

**Interpretation candidate — author review required:** the proximal operation
produced exact zeros in the two noise coordinates in this sample, with a
small optimality residual. The intercept is the sample target mean in the
centered representation; it need not equal the generator's raw intercept.

**Limitation:** this is an optimization demonstration with no held-out data.
Standardized coefficients are not directly comparable to raw generating
coefficients. Convergence and correct zeros here do not establish general
support recovery or performance on large ill-conditioned inputs.

## Suggested experiments — not executed in this phase

- Repeat training samples while holding a common test design fixed; estimate
  prediction bias and variance separately from coefficient variance.
- Vary correlation and record how often Lasso selects each proxy. Track
  pair-level prediction contribution as well as individual coefficients.
- Compare dense and sparse true signals over several sample sizes; select
  penalties within each replicate's development data.
- Inspect coefficient paths using a common training-fitted scaler. A future
  visual phase can plot them; no visual artifacts were generated here.
