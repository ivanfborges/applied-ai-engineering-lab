# Gradient Descent: Technical Notes

## 1. Intuition

Gradient descent treats training as an iterative search in parameter space. At
the current parameter vector, the gradient measures the local sensitivity of
the objective to each parameter. Under the Euclidean norm, it points in the
direction of steepest increase, so minimization moves in the opposite
direction:

$$
\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t),
$$

where $\eta > 0$ is the learning rate.

The familiar "walking downhill in fog" analogy captures the local nature of
the method, but real objectives are not necessarily smooth bowls. They may
contain flat regions, saddle points, noisy gradient estimates, and directions
with very different curvature.

## 2. Regression Objective

For one feature, the model is

$$
\hat{y}_i = wx_i + b,
$$

with residual $r_i = \hat{y}_i-y_i$. This topic uses the half mean squared
error:

$$
J(w,b) = \frac{1}{2n}\sum_{i=1}^{n}(wx_i+b-y_i)^2.
$$

The factor $1/2$ simplifies derivatives and does not change the minimizer. It
does mean that `loss_history_` in `from_scratch.py` is half the conventional
MSE reported for test predictions in `example.py`.

Mean squared error is differentiable and penalizes large residuals
quadratically. That is convenient for optimization but makes the fit sensitive
to outliers.

## 3. Deriving the Gradients

Applying the chain rule to the coefficient:

$$
\frac{\partial J}{\partial w}
= \frac{1}{2n}\sum_{i=1}^{n}
2(wx_i+b-y_i)x_i
= \frac{1}{n}\sum_{i=1}^{n}x_i r_i.
$$

For the intercept:

$$
\frac{\partial J}{\partial b}
= \frac{1}{n}\sum_{i=1}^{n}r_i.
$$

Both derivatives must be computed from the same parameter state. Updating
$w$ before calculating the derivative for $b$ silently changes the algorithm.

For multiple features:

$$
\hat{\mathbf{y}} = X\mathbf{w} + b\mathbf{1},
$$

$$
\nabla_{\mathbf{w}}J
= \frac{1}{n}X^\top(X\mathbf{w}+b\mathbf{1}-\mathbf{y}),
$$

$$
\frac{\partial J}{\partial b}
= \frac{1}{n}\mathbf{1}^\top
(X\mathbf{w}+b\mathbf{1}-\mathbf{y}).
$$

The NumPy expression `X.T @ residuals / n` implements the vectorized
coefficient gradient.

## 4. Convexity, the Hessian, and Convergence

If the intercept is incorporated into the design matrix, the objective becomes

$$
J(\theta) = \frac{1}{2n}\lVert X\theta-y\rVert_2^2.
$$

Its Hessian is

$$
H = \nabla^2J(\theta) = \frac{1}{n}X^\top X.
$$

$H$ is positive semidefinite, so the objective is convex. Every local minimum
is global. If the design matrix has full column rank, $H$ is positive definite
and the minimizer is unique. With linearly dependent columns, different
parameter vectors can make identical predictions and attain the same minimum.

For fixed-step gradient descent on this quadratic, convergence is guaranteed
when the learning rate satisfies

$$
0 < \eta < \frac{2}{\lambda_{\max}(H)}.
$$

This is a stability condition, not a claim that every value in the interval is
equally fast. The best practical rate depends on the entire eigenvalue
spectrum.

The condition number

$$
\kappa(H)=
\frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}
$$

measures the ratio between the steepest and flattest positive-curvature
directions. A large condition number creates elongated contours. Updates then
zigzag across steep directions while making slow progress along flatter ones.

## 5. Why Feature Scaling Matters

Changing a feature's units changes the associated curvature and gradient
magnitude. If one feature is measured in fractions and another in millions, a
single learning rate may be too large for one direction and too small for the
other.

Standardization,

$$
x'_j = \frac{x_j-\mu_j}{\sigma_j},
$$

often improves conditioning and permits a more effective learning rate. The
scaler must be fitted on training data only; using the complete dataset leaks
information from validation or test observations.

Scaling improves optimization geometry. It does not repair multicollinearity,
bad labels, distribution shift, or a misspecified model.

## 6. Stopping and Diagnostics

Common stopping rules include:

- a maximum update count;
- a sufficiently small gradient norm;
- negligible objective improvement;
- validation-based early stopping;
- a time or compute budget.

This implementation uses the gradient norm:

$$
\lVert\nabla J(\theta)\rVert_2 \le \varepsilon.
$$

For this convex problem, a small gradient norm is evidence that the parameters
are near a global optimum. In a non-convex problem, it may instead indicate a
saddle point or a flat region.

Useful diagnostics answer different questions:

| Diagnostic | What it reveals |
| --- | --- |
| Training loss | Whether the stated objective is decreasing |
| Validation loss | Whether generalization improves |
| Gradient norm | Whether a first-order stationary region is near |
| Parameter history | Oscillation, drift, and differing convergence speeds |
| Learning rate | Whether behavior changed with the step schedule |
| NaN/Inf checks | Numerical overflow or invalid computation |

A flat loss curve alone is ambiguous: the optimizer may have converged, the
learning rate may be too small, gradients may be broken, or numerical precision
may be limiting progress.

## 7. Batch Variants

### Batch Gradient Descent

Uses all training observations for each update. Gradients are deterministic for
fixed data and parameters, but every update requires a complete dataset pass.

### Stochastic Gradient Descent

Uses one observation per update. Updates are cheap and support streaming, but
the path is noisy and generally requires shuffling and a learning-rate
schedule.

### Mini-batch Gradient Descent

Uses a subset of observations. It balances gradient variance, memory use,
update frequency, and efficient matrix operations. It is the standard choice
for accelerator-based deep learning.

An iteration is one parameter update. An epoch is one complete pass over the
dataset. For batch gradient descent, they usually coincide; for mini-batch
training, an epoch contains multiple iterations.

## 8. Assumptions

The optimization derivation assumes:

- a differentiable objective;
- finite numerical inputs and targets;
- a fixed training dataset during each full-batch update;
- a learning rate compatible with the objective's curvature;
- consistent shapes for features, parameters, predictions, and targets.

Statistical interpretation adds separate assumptions. Classical linear
regression inference can require linearity, independent errors,
homoscedasticity, and distributional assumptions. Gradient descent does not
validate any of them; it only finds parameters for the objective supplied.

## 9. Trade-offs and Comparisons

### Gradient Descent versus a Direct Least-Squares Solver

| Aspect | Gradient descent | Direct numerical solver |
| --- | --- | --- |
| Strategy | Iterative first-order updates | Solves a least-squares system |
| Result | Approximation depends on stopping | Numerical least-squares solution |
| Main controls | Rate, updates, batch size | Solver tolerances |
| Large/streaming data | Can use batches | Often less natural |
| Generality | Applies to many models/losses | Specific mathematical forms |
| Diagnostics | Exposes optimization path | Usually returns final solution |

Production libraries generally use QR, SVD, or other stable least-squares
methods rather than explicitly computing $(X^\top X)^{-1}$.

### First-order versus Second-order Methods

Gradient descent uses slope only. Newton's method also uses curvature:

$$
\theta_{t+1}
= \theta_t-H^{-1}\nabla J(\theta_t).
$$

Second-order methods can need fewer iterations but forming, storing, or solving
with the Hessian is expensive at high parameter counts.

Momentum reduces oscillation by accumulating a velocity from recent gradients.
Adaptive methods such as Adam maintain per-parameter moment estimates. They can
improve practical training dynamics, but they do not remove the need for
validation or guarantee better generalization.

## 10. Applications

Gradient-based optimization appears in:

- large-scale logistic regression;
- neural network and Transformer training;
- embedding and reranker training;
- matrix factorization and recommendation;
- differentiable calibration and ranking objectives;
- parameter-efficient and full-model fine-tuning.

It is normally absent from the request-time path of a prompt-only or RAG
application. The foundation model may have been trained with gradient-based
optimization, and fine-tuning may use it again, but retrieval and prompt
construction are usually inference operations.

## 11. Limitations

- It requires gradients or useful gradient estimates.
- A fixed rate can be slow on ill-conditioned objectives.
- Non-convex objectives may contain saddle points and flat regions.
- Full-batch updates can be expensive for large datasets.
- Squared error is sensitive to large residuals and may not reflect business
  costs.
- Convergence on training data says nothing by itself about leakage,
  calibration, fairness, robustness, latency, or production drift.
- Floating-point precision can cause overflow, underflow, or loss of useful
  gradient information.

## 12. Common Mistakes

- Adding the gradient instead of subtracting it.
- Forgetting the factor from the chosen loss convention.
- Omitting the feature multiplier in the coefficient derivative.
- Updating one parameter before computing all gradients.
- Mixing one-dimensional arrays with `(n, 1)` arrays and relying on
  broadcasting.
- Using a rate without considering feature scale.
- Fitting preprocessing on the full dataset.
- Monitoring only the final training loss.
- Treating a small gradient as proof of good validation performance.
- Comparing half-MSE training history directly with full MSE without noting the
  factor of two.
- Using a handwritten optimizer in production when a tested library solver is
  appropriate.

## 13. Suggested Experiments

1. Try learning rates from `1e-4` to `1.0` and compare loss and gradient-norm
   histories.
2. Add a second feature whose scale is several orders of magnitude larger, then
   compare training before and after standardization.
3. Add a few extreme target values and compare squared error with a robust loss.
4. Implement stochastic and mini-batch variants; compare update noise rather
   than claiming benchmark superiority.
5. Check the analytical gradient with centered finite differences.
6. Build a contour plot of $J(w,b)$ and overlay the recorded parameter path.

