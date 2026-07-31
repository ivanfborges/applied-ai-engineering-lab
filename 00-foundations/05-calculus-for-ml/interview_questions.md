# Interview Questions: Calculus for Machine Learning

## 1. What does a derivative represent in machine learning?

A derivative measures local sensitivity. It approximates how much a scalar
output changes for a small change in a scalar input. In training, a partial
derivative such as \(\partial L/\partial\theta_j\) measures how sensitive the
loss is to one parameter while the others are held fixed.

The important qualifier is *local*. A derivative does not describe the entire
loss landscape or identify a global optimum.

## 2. What is the difference between a partial derivative and a gradient?

A partial derivative is the sensitivity of a multivariate scalar function to
one selected input. A gradient is the vector containing all those partial
derivatives:

\[
\nabla_{\boldsymbol\theta}L=
\begin{bmatrix}
\partial L/\partial\theta_1\\
\vdots\\
\partial L/\partial\theta_p
\end{bmatrix}.
\]

It has the same number of components as the parameter vector.

## 3. Why does gradient descent use the negative gradient?

The first-order approximation is

\[
L(\boldsymbol\theta+\Delta\boldsymbol\theta)
\approx
L(\boldsymbol\theta)
+
\nabla L(\boldsymbol\theta)^\top\Delta\boldsymbol\theta.
\]

Choosing
\(\Delta\boldsymbol\theta=-\eta\nabla L(\boldsymbol\theta)\) makes the
first-order change
\(-\eta\|\nabla L(\boldsymbol\theta)\|_2^2\), which is non-positive. A
sufficiently small step should therefore reduce the loss locally.

## 4. Does a correct negative-gradient step always reduce loss?

No. The argument is based on a local approximation. Loss may increase when the
learning rate is too large, the minibatch gradient is noisy, momentum
overshoots, precision is unstable, or the measured loss includes stochastic
behavior. A correct direction does not validate an arbitrary step length.

## 5. What is the relationship between the chain rule and backpropagation?

The chain rule composes derivatives through nested functions. Backpropagation
is an efficient reverse-mode procedure that applies this rule through a
computational graph. Each operation combines an upstream sensitivity with its
local derivative and passes the result to its dependencies.

Backpropagation computes gradients. It is not the optimizer that updates the
parameters.

## 6. Why is reverse-mode automatic differentiation suitable for neural networks?

Neural networks typically map millions or billions of parameters to one scalar
loss. Reverse mode computes the gradient of that one output with respect to all
inputs at a cost comparable to a small multiple of the forward computation. It
uses vector-Jacobian products and does not need to materialize every full
Jacobian.

Forward mode is more attractive when there are few inputs and many outputs.

## 7. How do analytical, numerical, and automatic differentiation differ?

- **Analytical differentiation** derives a formula by hand. It is exact and
  interpretable but difficult to maintain for large computation graphs.
- **Numerical differentiation** estimates derivatives by perturbing values. It
  is approximate, expensive, and useful mainly as an independent check.
- **Automatic differentiation** applies exact elementary derivative rules to
  the executed computation and composes them with the chain rule. Its result is
  accurate up to floating-point precision.

Automatic differentiation is neither finite differencing nor symbolic
algebra.

## 8. What is a gradient check, and how would you perform one?

A gradient check compares an analytical or automatically differentiated value
with a finite-difference estimate:

\[
\frac{\partial L}{\partial\theta_j}
\approx
\frac{L(\theta_j+\varepsilon)-L(\theta_j-\varepsilon)}
{2\varepsilon}.
\]

Use a small deterministic problem, double precision where possible, and a
centered difference. Compare relative as well as absolute error. Do not choose
\(\varepsilon\) so large that truncation dominates or so small that
floating-point cancellation dominates.

## 9. What is the difference between a gradient, a Jacobian, and a Hessian?

A gradient is the first derivative of a scalar output with respect to a vector
input. A Jacobian contains first derivatives for a vector output with respect
to a vector input. A Hessian contains second derivatives of a scalar output
with respect to a vector input and represents local curvature.

Because notation conventions differ, explicitly stating function domains and
checking shapes is good interview and implementation practice.

## 10. What causes vanishing and exploding gradients?

Backpropagation multiplies local derivative factors. If many factors have
magnitudes below one, gradients can vanish; if many exceed one, they can
explode. Saturating activations, long dependency paths, poor initialization,
and unstable recurrent dynamics are common causes.

Residual connections, normalization, suitable activation functions, and
careful initialization improve gradient flow. Gradient clipping limits an
exploding update but may only hide the underlying instability.

## 11. ReLU is not differentiable at zero. Why can gradient training still work?

The non-differentiable location is isolated, and libraries choose a practical
subgradient convention there. Gradient methods do not require every component
to be smoothly differentiable at every possible value. Differentiability
almost everywhere is often sufficient.

This should not be generalized to arbitrary discrete operations. An `argmax`
or hard threshold over a region may provide no useful gradient path.

## 12. Why is \(\partial L/\partial z=p-y\) important in logistic regression?

When sigmoid \(p=\sigma(z)\) is composed with binary cross-entropy, their
derivatives simplify to \(p-y\) at the logit. Consequently,

\[
\frac{\partial L}{\partial\mathbf w}=(p-y)\mathbf x.
\]

This demonstrates how chain-rule factors can simplify and provides an
intuitive error signal. Production code should use a combined logits-based
loss for numerical stability.

## 13. How would you debug a training run whose loss becomes `NaN`?

Start with a small deterministic batch. Check data, targets, activations,
parameters, loss components, and gradients for non-finite values. Inspect
logarithms, exponentials, divisions, tensor shapes, and reduction choices.
Then examine learning rate, initialization, gradient norms, mixed-precision
scaling, normalization, and custom backward code.

Gradient clipping may stabilize a run, but the pre-clipping norms should still
be inspected to identify the root cause.

## 14. How does this calculus apply to LLM fine-tuning and LoRA?

Fine-tuning computes a scalar training loss and propagates its gradient through
the transformer. Full fine-tuning updates most or all weights. LoRA freezes the
base weights and propagates gradients to small trainable low-rank adapter
matrices.

Practical consequences include activation memory, optimizer-state memory,
gradient accumulation, checkpointing, mixed precision, clipping, learning-rate
schedules, and distributed gradient synchronization.

## 15. How would you optimize a RAG system if it is not differentiable end to end?

Separate trainable and non-trainable decisions. Gradients can train an
embedding model, cross-encoder reranker, or routing classifier. Discrete
choices such as chunk size, top-\(k\), metadata rules, prompt templates, and
agent topology are better evaluated with an offline test set plus controlled
search or experiments.

An entire production system does not need to be differentiable to be improved
systematically.

## 16. What does a zero gradient tell you?

It identifies a stationary or numerically flat point, not necessarily a
minimum. The point could be a local minimum, maximum, saddle point, flat
plateau, saturated activation, disconnected graph, or underflowed computation.
Curvature, nearby evaluations, and graph diagnostics provide additional
evidence.

## 17. What senior-level limitation matters more than computing the gradient correctly?

Objective and evaluation validity. An optimizer efficiently improves the loss
it receives, even when that loss is a poor proxy for the real goal or the
validation set contains leakage. Correct calculus cannot repair misaligned
metrics, invalid splits, biased data, or missing safety constraints.

## Interview-Ready Summary

Derivatives measure local sensitivity. For a scalar loss with many parameters,
the partial derivatives form a gradient, which points toward steepest local
increase under the Euclidean norm. Optimizers usually move in the opposite
direction. The chain rule propagates sensitivity through composed operations,
and backpropagation is its efficient reverse-mode implementation on a
computational graph. In practice, correct gradients are only one part of
successful training: step size, scaling, numerical stability, data quality,
and objective alignment remain critical.
