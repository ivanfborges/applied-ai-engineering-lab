# Senior Interview Questions: Gradient Descent

## 1. What does the gradient represent?

The gradient is the vector of partial derivatives of a scalar objective with
respect to its parameters. Each component measures local sensitivity to one
parameter. Under the Euclidean norm, the vector points toward the steepest
local increase, so gradient descent moves in its negative direction.

## 2. Why does gradient descent reach a global minimum for least-squares linear regression?

The squared-error objective is a convex quadratic because its Hessian is
$X^\top X/n$, which is positive semidefinite. Therefore, every local minimum is
global. If the design matrix has full column rank, the Hessian is positive
definite and the minimizer is unique. Convergence still requires an appropriate
learning rate.

## 3. Derive the gradient for one-feature linear regression.

For

$$
J(w,b)=\frac{1}{2n}\sum_i(wx_i+b-y_i)^2,
$$

the derivatives are

$$
\frac{\partial J}{\partial w}
=\frac{1}{n}\sum_i x_i(wx_i+b-y_i)
$$

and

$$
\frac{\partial J}{\partial b}
=\frac{1}{n}\sum_i(wx_i+b-y_i).
$$

The feature multiplier in the coefficient derivative comes from the chain
rule.

## 4. What happens when the learning rate is too small or too large?

A very small rate can make progress impractically slow. A large rate can
overshoot, oscillate, or diverge. For a quadratic with Hessian $H$, fixed-step
gradient descent is stable when

$$
0 < \eta < \frac{2}{\lambda_{\max}(H)}.
$$

Feature scaling changes the eigenvalues, so it also changes the useful rate
range.

## 5. Why does standardization often accelerate convergence?

Differently scaled features create very different curvature across parameter
directions, producing an ill-conditioned Hessian and elongated loss contours.
Standardization often makes the curvature more balanced, reducing zigzagging
and allowing a useful global learning rate. It improves optimization geometry,
not data quality or model specification.

## 6. Why compute all gradients before updating any parameters?

One gradient-descent step is defined using the gradient at a single parameter
state. If the coefficient is updated before the intercept gradient is
calculated, the two derivatives come from different states. That creates a
different coordinate-style algorithm and can introduce subtle implementation
errors.

## 7. How do batch, stochastic, and mini-batch gradient descent differ?

Batch gradient descent uses the complete dataset and provides stable but
potentially expensive updates. Stochastic gradient descent uses one observation
and provides cheap, noisy updates. Mini-batch training uses a subset and
balances memory, vectorization, update frequency, and gradient variance. Its
hardware efficiency makes it standard in deep learning.

## 8. Why not always use gradient descent for linear regression?

For small or medium dense least-squares problems, mature numerical solvers are
simple, fast, and stable. Gradient descent becomes more attractive when data or
parameter counts are large, data arrives incrementally, batches are required,
or the objective has no direct solution. Production solvers avoid explicitly
forming $(X^\top X)^{-1}$ and typically use stable decompositions or iterative
least-squares methods.

## 9. How would you diagnose divergence?

Inspect:

- training and validation loss by update;
- gradient norms and parameter magnitudes;
- the learning rate and any schedule;
- feature and target ranges;
- NaN and infinity occurrences;
- batch ordering and batch-level variance;
- analytical gradients against finite differences for custom code.

Typical responses include reducing the rate, scaling inputs, correcting the
gradient, improving initialization, or applying gradient clipping where
appropriate. Clipping can contain symptoms but should not replace root-cause
analysis.

## 10. Does a small gradient guarantee a good model?

No. It indicates a first-order stationary region for the selected training
objective. The model can still have poor validation performance, leakage,
biased data, distribution-shift sensitivity, calibration problems, or an
objective that does not represent the product cost.

## 11. What if training loss decreases while validation loss increases?

That pattern suggests overfitting or a train/validation mismatch. Investigate
the split and leakage first, then consider regularization, early stopping,
model capacity, representative data, and data augmentation where appropriate.
Changing the optimizer alone does not address the generalization problem.

## 12. What is the difference between an iteration and an epoch?

An iteration is one parameter update. An epoch is one complete pass over the
training data. In batch gradient descent, one epoch normally produces one
update. If a dataset is divided into ten mini-batches, one epoch normally
contains ten updates.

## 13. How should optimization be monitored in a production training pipeline?

Track training and validation objectives, task metrics, learning rate, gradient
norm, parameter or update norms, epochs, iterations, throughput, resource use,
and non-finite values. Associate these metrics with code, data, configuration,
random seeds, and checkpoint versions. Alerting should distinguish optimization
failure from generalization failure.

## 14. How does gradient descent relate to distributed training?

Workers compute gradients on different mini-batches and aggregate them before
an update. Synchronous training uses a consistent step but can wait for slow
workers. Asynchronous training improves utilization but may apply stale
gradients. Communication volume, aggregation precision, failure recovery, and
reproducibility become core design constraints.

## 15. Give an interview-ready explanation.

Gradient descent is a first-order iterative method for minimizing a
differentiable objective. At each step it computes the loss gradient with
respect to the parameters and subtracts a learning-rate-scaled version of that
gradient. For least-squares linear regression, the objective is convex, so an
appropriate learning rate converges toward a global minimum. Feature scaling
matters because it changes the objective's conditioning. In practice, I would
monitor both optimization signals, such as loss and gradient norm, and
validation signals, because minimizing training loss does not guarantee a
reliable production model.

