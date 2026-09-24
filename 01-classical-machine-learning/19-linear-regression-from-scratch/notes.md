# Numerical least squares: derivation, decisions, and evidence

## One objective, two solvers

Let raw features be X with shape (n, p), and let A = [1, X] include the
intercept. Parameters theta have shape (p + 1,). The implementation uses

$$J(\theta)=\frac{1}{2n}\|A\theta-y\|_2^2,\qquad
\nabla J(\theta)=\frac{1}{n}A^\top(A\theta-y).$$

Setting the gradient to zero gives the normal equations:

$$A^\top A\hat\theta=A^\top y.$$

The familiar inverse expression exists only with full column rank. The
implementation uses `np.linalg.lstsq(A, y, rcond=None)` and exposes the rank
of A. It avoids explicitly forming or inverting the Gram matrix: for a
full-rank design, its 2-norm condition number is the square of that of A.
NumPy supplies the decomposition; our code supplies the regression formulation.

Geometrically, fitted values project y onto the column space of A, and
A-transpose times the training residual is zero up to numerical error.
With dependent columns, training fitted values remain unique but coefficients
need not be. `lstsq` selects the minimum-norm parameter vector for this
augmented design. Centering before solving can select a different parameter
vector in singular cases; compare fitted values rather than demand identical
coefficients. Off-design extrapolation need not agree.

Batch GD starts at zero and performs

$$\theta_{t+1}=\theta_t-\eta\nabla J(\theta_t).$$

The Hessian H = A-transpose A / n is positive semidefinite, so stationary
points minimize this quadratic. A sufficient constant-step convergence
condition in exact arithmetic is 0 < eta < 2/L, where L is the largest
eigenvalue of H. Convexity alone does not make every step size safe.
For singular designs, zero-initialized GD stays in the row space and approaches
the minimum-norm solution when the step is suitable and iterations suffice.

Here loss is **half-MSE**, not MSE. The gradient for MSE is twice as large;
reusing the same learning rate after changing that convention changes updates.
The public loss helper is tested against central finite differences.

## Scaling, stopping, and cost

For z_j = (x_j - mu_j) / s_j, translate fitted standardized parameters by

$$w_j = w_{z,j}/s_j,\qquad b=b_z-\sum_j\mu_jw_j.$$

Fit mu and s using training data only and preserve them for serving. OLS
predictions are invariant to invertible feature rescaling in exact arithmetic
with a full-rank design and intercept; numerical conditioning still matters.
Standardization often improves GD geometry, but cannot remove collinearity.

`fit_gradient_descent` stops when the gradient infinity norm is at most `tol`,
or returns `converged=False` when `max_iter` updates are exhausted. History
contains the initial objective and every subsequent objective, including the
returned parameters. Overflow or nonfinite arithmetic raises an error.
A small gradient is coordinate-dependent and can coexist with inaccurate
coefficients along weak-curvature directions. Small loss changes alone can
also reflect an excessively small step. Check predictions and conditioning.

For dense n >= p, a direct decomposition typically costs O(np^2); each batch
GD update costs O(np). This implementation stores the full design plus an
O(iterations) loss history, so it is not a streaming or minibatch solver.
The example computes L via a spectral norm solely to isolate conditioning;
that decomposition has a cost and is not a scalable learning-rate strategy.
No runtime benchmark is claimed.

## Statistical and practical boundaries

Optimization requires no Gaussian features or errors. For unbiased coefficient
estimation in the correctly specified model, zero conditional mean errors and
identifiability matter. The usual Gauss-Markov variance claim additionally
assumes a spherical conditional error covariance. Gaussian errors enable
classical exact small-sample inference; residuals themselves are fitted and
are not independent observations of errors. Heteroscedasticity or dependence
calls for appropriate inference and validation, not an automatic declaration
that every prediction is unusable.

Linear means linear in parameters: polynomial feature expansions still qualify.
Coefficients describe conditional associations, not causal interventions.
Squared loss is sensitive to large residuals, and extrapolation can violate
domain constraints. Inspect residual structure and segment errors before
using a baseline for demand, latency, or a fixed-embedding regression probe.
Ridge changes the objective and can stabilize coefficients; it belongs to
the next study, rather than being silently added to this OLS implementation.

## Executed experiments

Executed by Codex on 2026-09-21 with Python 3.11.0rc2, NumPy 2.4.6, and
scikit-learn 1.9.0. Command (repository root):

```bash
.venv/Scripts/python.exe 01-classical-machine-learning/19-linear-regression-from-scratch/example.py
```

### Solver equivalence

**Hypothesis:** direct OLS and adequately converged GD should reproduce the
scikit-learn fit on a well-conditioned, full-rank problem.

**Configuration:** NumPy `default_rng(42)`; X has 500 x 3 independent standard
normal entries; y = 4 + X @ [3.5, -2, 1.2] + Gaussian noise of standard
deviation 0.8. `train_test_split(test_size=0.2, random_state=42)` gives 400/100
rows. OLS and sklearn fit raw features; GD uses training-only `StandardScaler`,
zero initialization, step 0.1, at most 10,000 updates, and tolerance 1e-8.
Assertions use absolute prediction tolerances 1e-10 for OLS and 1e-6 for GD,
with relative tolerance zero; recovered GD parameters use 1e-6.

**Result:**

| Method | Test MSE | Test R2 | Max prediction difference from sklearn |
|---|---:|---:|---:|
| NumPy OLS | 0.572688892 | 0.962611855 | 1.243e-14 |
| Batch GD | 0.572688894 | 0.962611855 | 4.963e-08 |
| scikit-learn | 0.572688892 | 0.962611855 | 0 |

OLS parameters [intercept, slopes] were approximately
[3.89094398, 3.40354450, -1.98851462, 1.23097025]. GD converged in 200 updates;
training half-MSE decreased from 16.684530315 to 0.326403509.

**Interpretation candidate — author review required:** agreement supports
implementation correctness for this fixture. The parameter differences from
the known generator are compatible with finite-sample noise, not evidence
that one solver optimizes a different objective.

**Limitation:** one synthetic seed, split, and correctly specified model;
this does not establish performance on real data, causal validity, or speed.
Printed final digits can vary across environments.

### Feature units and convergence

**Hypothesis:** making one feature much larger will slow GD even with a stable
step, while training-only standardization will improve this fixture's geometry.

**Configuration:** same training rows and targets, second feature multiplied
by 10,000. Compare raw changed units with their standardized representation.
Each run uses its own eta = 1/L, zero initialization, tolerance 1e-8, and
2,000-update budget. Objective gaps are measured against direct OLS in the
same coordinates. No test observations participate in this experiment.

**Result:**

| Representation | Step | cond(A) | Updates | Converged | Final half-MSE | Gap to OLS |
|---|---:|---:|---:|---|---:|---:|
| Unscaled | 1.088e-08 | 1.010e+04 | 2000 | False | 15.200963246 | 1.487e+01 |
| Standardized | 9.243e-01 | 1.085e+00 | 11 | True | 0.326403509 | 0 to displayed precision |

**Interpretation candidate — author review required:** scaling improved
conditioning and progress toward the common optimum. The unscaled run
exhausted its budget despite a stable step; this is slow convergence, not
a different statistical model or demonstrated divergence.

**Limitation:** different coordinate systems use different gradient tolerances
and step sizes. The objective gap makes progress comparable, but update counts
are not a timing benchmark. Independent synthetic features make this case
especially favorable to standardization; correlated features can remain hard.

## Interview rehearsal and further work

- **Why not invert A-transpose A?** It squares the condition number and may be
  singular; a direct decomposition solves the actual least-squares problem.
- **Why can two correct fits have different coefficients?** Rank deficiency
  leaves unidentified directions; parameterization and minimum-norm conventions
  can choose different solutions with the same training fitted values.
- **Does GD convergence validate the model?** It validates an optimization
  stopping condition; held-out errors, residuals, and domain assumptions answer
  different questions.
- **How would you debug a deployment discrepancy?** First verify feature order,
  units, intercept, and the saved training scaler; then inspect distribution
  shift, missing values, residual patterns, and validation design.

The original nonvisual study left these as suggestions: sweep learning rates around
2/L; repeat near-collinear fits across noise realizations; add target outliers;
compare raw x with [x, x-squared] under a nonlinear generator. Singular fixtures
and divergence checks in tests are correctness checks, not empirical studies
of these broader effects. The subsequent [visual laboratory](VISUAL_GUIDE.md)
now records executed learning-rate, collinearity, outlier, and polynomial-feature
experiments separately, with configurations and interpretation caveats.

## References

- [NumPy lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html):
  least-squares, rank, and minimum-norm return semantics.
- [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html):
  reference estimator interface and ordinary least-squares objective.
- [Day 18 theory](../18-linear-regression-theory/): statistical interpretation
  and residual diagnostics preceding this implementation study.
