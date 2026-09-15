# Linear Regression: What the Fit Does and Does Not Establish

## 1. A model for a conditional mean

For $p$ predictors and $n$ observations, define the design matrix
$D=[\mathbf{1},X]$, including an intercept. It has $k=p+1$ columns.

$$
y=D\beta+\varepsilon,\qquad
E[y\mid D]=D\beta
$$

The conditional-mean statement is an assumption about the process, not a
consequence of calling a fitting method. With a wrong feature set, OLS can
still estimate a useful linear approximation without identifying the desired
structural relationship.

For an additive model, a coefficient of 3 means that increasing its predictor
by one unit changes the fitted mean by 3 target units while holding the other
predictors fixed. This comparison needs support in the data: near-duplicate
features may rarely vary separately. Rescaling a predictor changes its
coefficient's units, so raw coefficient magnitude is not feature importance.

The intercept is the prediction when all features are zero. If that input is
outside the observed domain, its practical interpretation may be weak.
Centering features on training means makes the intercept refer to a more
representative input.

“Linear” refers to coefficients. With
$m(x)=\beta_0+\beta_1x+\beta_2x^2$, the slope with respect to the original
input is $\beta_1+2\beta_2x$, not just $\beta_1$. With an interaction
$\beta_3x_1x_2$, the effect on the fitted mean of changing $x_1$ depends on
$x_2$. These remain linear regression models.

## 2. Derive the optimization before interpreting it

$$
RSS(\beta)=\|y-D\beta\|_2^2
$$

Dividing RSS by $n$ produces MSE with the same minimizer. Differentiating:

$$
\nabla RSS=-2D^\top(y-D\beta),\qquad
\nabla^2 RSS=2D^\top D.
$$

The Hessian is positive semidefinite, so a stationary point is a global
minimum. Setting the gradient to zero gives the normal equations:

$$
D^\top D\hat\beta=D^\top y.
$$

When $D$ has full column rank, the Hessian is positive definite and:

$$
\hat\beta=(D^\top D)^{-1}D^\top y.
$$

The inverse expresses the mathematics. Numerical software should solve the
least-squares system directly, for example through QR or SVD. Forming
$D^\top D$ squares the 2-norm condition number for full-rank $D$, magnifying
conditioning problems.

If the columns are dependent, coefficients are not unique. Fitted values on
the training rows remain the unique projection onto the column space.
Different minimizing coefficients can disagree at a new input that breaks
the training dependence. A pseudoinverse chooses one solution; it does not
create identification. NumPy's [least-squares API](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
returns a minimum-norm solution when minimizers are not unique.

## 3. Projection explains the residual trap

Let $H=D(D^\top D)^{-1}D^\top$ for full-rank $D$. Then:

$$
\hat y=Hy,\qquad e=y-\hat y=(I-H)y,\qquad D^\top e=0.
$$

Thus training residuals are orthogonal to every included design column.
An intercept makes their sum zero, up to numerical precision. These
properties also hold for the least-squares projection in a rank-deficient
design, using the appropriate pseudoinverse.

They hold even when important nonlinear terms or variables are missing.
Consequently, checking training residual correlation with an included feature
cannot establish exogeneity. Check patterns across feature ranges, omitted
transformations, groups, and time. Held-out residuals have no fitted
zero-sum or orthogonality constraint.

The theoretical error $\varepsilon=y-D\beta$ depends on the population
model. The fitted residual $e=y-D\hat\beta$ uses estimated coefficients.
Under the classical covariance assumptions:

$$
\operatorname{Cov}(e\mid D)=\sigma^2(I-H).
$$

Residuals therefore need not be independent or have identical variances even
when the underlying errors do:
$\operatorname{Var}(e_i\mid D)=\sigma^2(1-h_{ii})$. This is one reason to use
appropriately standardized residuals for detailed diagnostics.

## 4. Separate the guarantees

| Question | Sufficient conditions / interpretation |
|---|---|
| Can least squares be computed? | A finite design and response; normality and homoskedasticity are unnecessary. |
| Are all coefficients unique? | Full column rank of the design, including the intercept. Having more rows than columns alone is insufficient. |
| Is OLS conditionally unbiased for the specified $\beta$? | Full rank and $E[\varepsilon\mid D]=0$. |
| Is OLS BLUE? | The preceding conditions plus $\operatorname{Cov}(\varepsilon\mid D)=\sigma^2I$. |
| Are classical finite-sample t and F reference distributions exact? | Add conditional joint Gaussian errors to the classical model, with positive residual degrees of freedom. |
| Will prediction generalize? | Assess representative held-out data, feature availability, support, and stability of the process. BLUE is not a prediction guarantee. |

BLUE means minimum covariance among estimators **linear in the observed
response and unbiased**, conditional on the design. It is not a comparison
against every predictor or every biased estimator. These distinctions are
covered in [MIT's regression lecture](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/fc0c08db6b497cc3bd09020aec39f9b5_MIT18_S096F13_lecnote6.pdf).

Uncorrelated errors suffice for the classical covariance result; independence
is a stronger condition. Neither the predictors nor the marginal target must
be Gaussian. Normality is not needed for OLS unbiasedness.

### Exogeneity is not a residual test

From the normal-equation solution:

$$
\hat\beta=\beta+(D^\top D)^{-1}D^\top\varepsilon.
$$

Zero conditional error mean removes the expected second term. Omitted
determinants, reverse causality, measurement error, or selection can invalidate
that argument for an intended structural coefficient.

For a simple example, suppose
$y=\beta_0+\beta_1x+\gamma z+u$, with $u$ uncorrelated with $x$.
Omitting $z$ gives a population simple-regression slope:

$$
b_1=\beta_1+\gamma\frac{\operatorname{Cov}(x,z)}
                              {\operatorname{Var}(x)}.
$$

The added term requires both an effect of the omitted variable and its
association with the retained predictor. Fitted residual orthogonality does
not remove it. Causal claims also need an identification strategy; see
[correlation versus causation](../../00-foundations/13-correlation-causation/).

### Heteroskedasticity and dependence

Changing conditional error variance does not by itself bias coefficients
under exogeneity. It invalidates the usual homoskedastic standard-error
formula and removes the general Gauss-Markov efficiency guarantee.

Heteroskedasticity-consistent standard errors address variance estimation
under suitable conditions; they do not fix omitted-variable bias or change
the fitted predictions. Clustered or temporally dependent observations need
covariance estimators and evaluation splits suited to that dependence.
Weighted least squares can improve efficiency if its variance model is
appropriate. These are extensions, not implemented comparisons here.

## 5. Uncertainty and collinearity

Under the classical assumptions:

$$
\operatorname{Cov}(\hat\beta\mid D)=\sigma^2(D^\top D)^{-1},
\qquad
s^2=\frac{RSS}{n-k},\quad n>k.
$$

The estimated standard error of a coefficient is the square root of the
corresponding diagonal entry of $s^2(D^\top D)^{-1}$. The denominator
$n-k$ accounts for fitted parameters; training MSE uses $n$ instead.
No coefficient intervals or significance claims are computed in the example.

At a new design vector $d_0$, the estimated variance of the fitted mean is:

$$
s^2d_0^\top(D^\top D)^{-1}d_0.
$$

For a new independent observation from the same homoskedastic process, add
$s^2$. Prediction intervals are therefore wider than mean-response
confidence intervals, under these assumptions. Neither accounts for an
unmodeled regime change.

Near-collinear features make some directions weakly identified: small
response changes can cause large coefficient changes. Predictions can still
be stable for inputs close to the observed feature relationships, while
becoming unstable when those relationships change.

Ridge trades coefficient bias for potential variance reduction; Lasso can
produce sparse coefficients; Elastic Net combines penalties. Their gains
need validation. Scaling affects those penalties. Scaling an unpenalized
full-rank OLS design consistently preserves fitted predictions in exact
arithmetic, but can improve numerical conditioning. Regularization is the
subject of Day 20.

## 6. Diagnose what aggregate error hides

| Signal | Possible explanation | Follow-up |
|---|---|---|
| Residual mean changes across feature ranges | Missing curvature or interaction | Compare justified transformations using development data |
| Residual spread grows with input size | Heteroskedasticity | Examine conditional uncertainty and error slices |
| Runs of similarly signed residuals over time | Dependence or changing process | Temporal evaluation and regime analysis |
| Residuals differ by segment | Missing group effects or selection | Check group definitions, support, and feature availability |
| Large residual or sensitivity to one row | Unusual response or influential observation | Validate the observation and inspect leverage |

These are clues, not unique diagnoses. NIST explains why residual analysis
can reveal model inadequacy hidden by a single fit statistic in its
[model-validation guidance](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd44.htm).
The present example uses numerical summaries; a plot-based diagnostic study
is a separate extension.

Leverage $h_{ii}$ measures unusual predictor geometry. A response outlier
has a large residual. Influence describes the change in the fit when a point
is changed or removed. A high-leverage point can have a small residual
because it pulls the line toward itself. Do not delete observations solely
because they are inconvenient.

For a nonconstant evaluation target:

$$
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}
              {\sum_i(y_i-\bar y_{\mathrm{eval}})^2}.
$$

With an intercept, exact unpenalized training OLS cannot have negative R²;
adding columns cannot increase the optimized training RSS. Held-out R² can
be negative. For a constant target, the displayed formula is undefined;
library conventions may substitute finite values.

The evaluation-set mean in R² is a scoring reference. It is different from a
deployable constant model learned from the **training** mean. The example
reports that baseline separately. RMSE emphasizes large errors; MAE retains
target units with less emphasis on the largest residuals. Neither establishes
business usefulness or tail reliability.

## 7. Executed experiments

**Status:** run by Codex; interpretation candidates below require author
review. These are synthetic demonstrations, not author conclusions.

Command from the repository root:

~~~bash
python 01-classical-machine-learning/18-linear-regression-theory/example.py
~~~

Run environment: Windows; Python 3.11.0rc2, NumPy 1.26.4, SciPy 1.11.2,
scikit-learn 1.9.0. Each experiment starts its own
`np.random.default_rng(42)`, generates 500 observations, and uses an 80/20
shuffled split with `random_state=42`. All fitting uses only 400 training
rows; metrics use the remaining 100. No tuning or repeated seed selection was
performed. Small last-digit differences may occur across numerical libraries.

The diagnostic printed as the maximum column-normalized inner product is
$\max_j |D_j^\top e|/\|D_j\|_2$, treating a zero column as zero. It helps
compare differently scaled columns but is not a statistical test.

### Experiment 1: known linear mean

- **Hypothesis:** an OLS fit should approximate the generating coefficients
  and improve on a training-mean predictor in this specified synthetic setup.
- **Configuration:** independent $x_1\sim N(10,2^2)$,
  $x_2\sim N(5,1.5^2)$, and $\varepsilon\sim N(0,2^2)$;
  $y=5+3x_1-2x_2+\varepsilon$. Fit an intercept and both predictors.
- **Result:** intercept 4.602311; slopes 3.066012 and -2.050174.
  Test RMSE 1.890698, MAE 1.464149, R² 0.920035.
  Training-mean baseline test RMSE 6.689765.
  Training residual mean $1.308\times10^{-14}$; maximum normalized inner
  product $2.614\times10^{-13}$. Test residual mean -0.105184.
- **Interpretation candidate — author review required:** this run is
  consistent with recovery of the specified linear relationship and the
  expected training projection identities.
- **Limitation:** one sample does not verify unbiasedness across samples or
  interval coverage. The generating model was made favorable to OLS. The
  intercept refers to feature values far from their means.

### Experiment 2: missing curvature despite training orthogonality

- **Hypothesis:** fitting only $x$ leaves systematic residual structure;
  adding the prespecified $x^2$ term should reduce held-out error.
- **Configuration:** $x\sim U(-3,3)$,
  $y=3x+2x^2+\varepsilon$, with independent $\varepsilon\sim N(0,1)$.
  Fit an intercept with $x$, then with $x,x^2$, on identical training rows.
  Feature sets and residual bins were fixed before evaluation.
- **Result:**

| Quantity | x only | x and x² |
|---|---:|---:|
| Intercept | 5.686225 | -0.043474 |
| x coefficient | 2.896686 | 2.979421 |
| x² coefficient | Omitted | 1.995455 |
| Test RMSE | 5.287462 | 0.968058 |
| Test MAE | 4.493228 | 0.786984 |
| Test R² | 0.425371 | 0.980738 |
| Test correlation of residual with x² | 0.982070 | -0.005250 |
| Mean residual, x in [-3, -1), n=38 | 3.132048 | -0.139314 |
| Mean residual, x in [-1, 1), n=25 | -5.050183 | -0.022303 |
| Mean residual, x in [1, 3), n=37 | 2.367638 | 0.095141 |

Both training residual means were below $4\times10^{-15}$ in absolute
value, and both maximum normalized inner products were below
$9\times10^{-14}$.

- **Interpretation candidate — author review required:** the nearly exact
  training orthogonality coexists with missing curvature. Adding the known
  generating term substantially reduces held-out error in this run.
- **Limitation:** the correct feature was known from the synthetic generator.
  This is not a general model-selection procedure. Residual correlation is
  descriptive, not a significance test or proof that remaining errors are
  independent. Exploring new features after seeing test diagnostics would
  turn this test set into development data; final evaluation would then need
  a fresh holdout.

## 8. Applications, limits, and suggested extensions

A linear workload model can make units and conditional associations explicit.
For latency prediction, use features available at the prediction time:
realized output-token count may support retrospective analysis but is not
known before generation. Respect this distinction when comparing an offline
score with a serving use case.

Extrapolation extends the equation outside observed support; it does not
validate the trend. Queue saturation, caching, changed hardware, missing
interactions, and negative predictions for inherently positive outcomes can
make an additive model inappropriate. Monitor delayed target errors,
time/group slices, feature support, and retraining stability. Use the
[Day 17 validation guidance](../17-validation-and-leakage/) when designing
evaluation boundaries.

**Extensions to the terminal example:** the later [visual laboratory](VISUAL_GUIDE.md)
implements outlier movement, repeated-sample coefficient stability,
heteroskedastic residuals, and Ridge comparisons. Its executed configurations
and measurements are recorded separately. The list below describes broader
follow-ups; inference coverage and development-fold Ridge tuning remain
suggestions, not executed evidence.

- Add a high-leverage response outlier and compare coefficients and held-out
  errors before and after refitting.
- Add a near-duplicate feature and examine coefficient variability over
  repeated training samples, together with prediction variability.
- Generate increasing noise variance and study classical versus robust
  uncertainty estimates.
- Compare OLS with scaled Ridge using training/development folds, preserving
  a final untouched test set.

These require their own hypotheses, configurations, measured results, and
author review before any conclusions are published.
