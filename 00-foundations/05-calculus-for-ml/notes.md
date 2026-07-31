# Notes: Derivatives, Gradients, and the Chain Rule

## 1. Intuition: Sensitivity and Local Models

A derivative answers a local counterfactual:

> If this input changed by a very small amount, how would the output change?

For a differentiable scalar function \(f\), a small perturbation \(\Delta x\)
can be approximated by

\[
f(x + \Delta x) \approx f(x) + f'(x)\Delta x.
\]

The derivative is therefore the coefficient of the best local linear
approximation. It describes behavior near the current point; it does not claim
that the function is linear everywhere.

In machine learning, replace \(x\) with a parameter vector \(\boldsymbol\theta\)
and \(f\) with a scalar loss \(L\). The same idea becomes

\[
L(\boldsymbol\theta + \Delta\boldsymbol\theta)
\approx
L(\boldsymbol\theta)
+
\nabla L(\boldsymbol\theta)^\top\Delta\boldsymbol\theta.
\]

This approximation is the bridge from calculus to gradient-based
optimization.

## 2. Derivatives

For \(f:\mathbb{R}\rightarrow\mathbb{R}\), the derivative at \(x\) is

\[
f'(x)
=
\lim_{h\rightarrow 0}\frac{f(x+h)-f(x)}{h},
\]

when the limit exists.

For \(f(x)=x^2\), \(f'(x)=2x\). At \(x=3\), a perturbation of \(0.01\) predicts
an output change of approximately \(6(0.01)=0.06\).

### Useful derivative rules

\[
\frac{d}{dx}c=0,
\qquad
\frac{d}{dx}x^k=kx^{k-1},
\qquad
\frac{d}{dx}e^x=e^x,
\qquad
\frac{d}{dx}\log x=\frac{1}{x}.
\]

For common activation functions,

\[
\sigma(z)=\frac{1}{1+e^{-z}},
\qquad
\sigma'(z)=\sigma(z)(1-\sigma(z)),
\]

\[
\frac{d}{dz}\tanh(z)=1-\tanh^2(z).
\]

ReLU is differentiable away from zero:

\[
\operatorname{ReLU}'(z)=
\begin{cases}
0, & z<0,\\
1, & z>0.
\end{cases}
\]

At zero, its classical derivative does not exist. Libraries choose a practical
subgradient convention, commonly zero.

## 3. Continuity, Differentiability, and Smoothness

Differentiability at a point implies continuity there, but continuity does not
imply differentiability. The function \(f(x)=|x|\) is continuous at zero but
has different left and right derivatives.

Many ML objectives are differentiable almost everywhere rather than
everywhere. Gradient methods can still work with subgradients or library-defined
conventions at isolated non-smooth points. This is different from inserting a
hard discrete operation, such as `argmax`, into a path that must carry
gradients: such an operation may eliminate the useful local sensitivity
altogether.

## 4. Partial Derivatives and Gradients

For a scalar function of several variables, a partial derivative changes one
coordinate while holding the others fixed. If

\[
f(x,y)=x^2+3xy+y^2,
\]

then

\[
\frac{\partial f}{\partial x}=2x+3y,
\qquad
\frac{\partial f}{\partial y}=3x+2y.
\]

For \(f:\mathbb{R}^d\rightarrow\mathbb{R}\), the gradient collects all partial
derivatives:

\[
\nabla f(\mathbf{x})=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\vdots\\
\frac{\partial f}{\partial x_d}
\end{bmatrix}.
\]

Under the Euclidean norm, the gradient points in the direction of greatest
local increase. Its negative is the direction of steepest local decrease.
These are local statements and depend on the geometry induced by the chosen
coordinate system and norm.

## 5. Directional Derivatives

For a unit vector \(\mathbf{u}\), the directional derivative is

\[
D_{\mathbf{u}}f(\mathbf{x})
=
\nabla f(\mathbf{x})^\top\mathbf{u}.
\]

This is the local rate of change when moving along \(\mathbf{u}\). By the
Cauchy-Schwarz inequality,

\[
\nabla f(\mathbf{x})^\top\mathbf{u}
\leq
\|\nabla f(\mathbf{x})\|_2,
\]

with equality when \(\mathbf{u}\) has the same direction as the gradient. This
explains the steepest-ascent interpretation.

## 6. Jacobians and Hessians

For a vector-valued function
\(\mathbf{f}:\mathbb{R}^n\rightarrow\mathbb{R}^m\), the Jacobian is

\[
J_{\mathbf f}(\mathbf{x})_{ij}
=
\frac{\partial f_i}{\partial x_j},
\]

and has shape \(m\times n\). It describes the sensitivity of every output
component to every input component.

For a scalar function \(f:\mathbb{R}^n\rightarrow\mathbb{R}\), the Hessian
contains second derivatives:

\[
H_f(\mathbf{x})_{ij}
=
\frac{\partial^2 f}{\partial x_i\partial x_j}.
\]

The gradient describes slope; the Hessian describes local curvature. Hessian
eigenvalues can help characterize minima, maxima, saddle points, and
ill-conditioning. A dense Hessian for \(p\) parameters contains \(p^2\)
entries, so explicitly constructing it is generally infeasible for modern
neural networks.

### Shape summary

| Object | Mapping | Derivative shape | Primary meaning |
| --- | --- | --- | --- |
| Derivative | scalar to scalar | scalar | one-dimensional sensitivity |
| Gradient | vector to scalar | input-sized vector | all first-order sensitivities |
| Jacobian | vector to vector | output by input matrix | all input-output sensitivities |
| Hessian | vector to scalar | input by input matrix | local curvature |

The numerator-layout Jacobian convention is used here. Other texts may use a
transpose convention, so shape checks matter more than memorizing notation.

## 7. The Chain Rule

For \(y=f(g(x))\),

\[
\frac{dy}{dx}
=
\frac{df}{dg}\frac{dg}{dx}.
\]

For a vector intermediate \(\mathbf z=g(\mathbf x)\) and scalar
\(L=h(\mathbf z)\), using the Jacobian convention above,

\[
\nabla_{\mathbf x}L
=
J_g(\mathbf x)^\top\nabla_{\mathbf z}L.
\]

The chain rule says that sensitivity along a dependency path is the product of
local sensitivities. When a value contributes through multiple paths, the
contributions add.

### A nonlinear scalar neuron

Consider

\[
z=wx+b,\qquad a=\tanh(z),\qquad L=(a-y)^2.
\]

The derivative with respect to the weight is

\[
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
=
2(a-y)(1-a^2)x.
\]

Each term is local to one operation. This decomposition is implemented in
`from_scratch.py`.

## 8. Computational Graphs and Backpropagation

A computational graph represents a complex expression as elementary
operations. A forward pass computes values and retains the intermediates
needed for differentiation. A backward pass starts from the scalar loss and
propagates sensitivities toward its dependencies.

Backpropagation is reverse-mode automatic differentiation applied to such a
graph. It is not an optimizer:

- backpropagation computes gradients;
- SGD, Adam, and other optimizers use gradients to update parameters.

Reverse mode is a good match for neural networks because they typically have
many parameters and one scalar loss. It computes vector-Jacobian products
without materializing every full Jacobian.

## 9. Linear Regression Gradient Derivation

For synthetic observations \((x_i,y_i)\), let

\[
\hat y_i=wx_i+b
\]

and define mean squared error

\[
L(w,b)=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2.
\]

With residual \(e_i=wx_i+b-y_i\), the chain rule gives

\[
\frac{\partial L}{\partial w}
=
\frac{2}{n}\sum_{i=1}^{n}e_i x_i,
\qquad
\frac{\partial L}{\partial b}
=
\frac{2}{n}\sum_{i=1}^{n}e_i.
\]

For \(X\in\mathbb{R}^{n\times d}\), weights
\(\mathbf w\in\mathbb{R}^d\), and residual vector
\(\mathbf e=X\mathbf w+b\mathbf 1-\mathbf y\),

\[
\nabla_{\mathbf w}L=\frac{2}{n}X^\top\mathbf e,
\qquad
\frac{\partial L}{\partial b}
=
\frac{2}{n}\mathbf 1^\top\mathbf e.
\]

The transpose maps the observation-sized residual vector back to one
sensitivity per feature. `example.py` checks this analytical result against
finite differences before using it.

## 10. Logistic Regression Simplification

For one binary observation,

\[
z=\mathbf w^\top\mathbf x+b,\qquad p=\sigma(z),
\]

\[
L=-y\log p-(1-y)\log(1-p).
\]

Applying the chain rule through binary cross-entropy and sigmoid yields

\[
\frac{\partial L}{\partial z}=p-y.
\]

Therefore,

\[
\frac{\partial L}{\partial \mathbf w}=(p-y)\mathbf x,
\qquad
\frac{\partial L}{\partial b}=p-y.
\]

Implementations normally combine logits and cross-entropy in one numerically
stable operation rather than computing probabilities and logarithms
separately.

## 11. Why the Negative Gradient Can Reduce Loss

Choose the update

\[
\Delta\boldsymbol\theta
=
-\eta\nabla L(\boldsymbol\theta),
\qquad \eta>0.
\]

Substitution into the first-order approximation gives

\[
L(\boldsymbol\theta+\Delta\boldsymbol\theta)
\approx
L(\boldsymbol\theta)
-
\eta\|\nabla L(\boldsymbol\theta)\|_2^2.
\]

For a sufficiently small step, this predicts a local decrease. The qualifier
is essential: a large learning rate can leave the region where the linear
approximation is accurate, and minibatch noise or optimizer momentum can also
increase the measured loss on a particular step.

Day 6 treats gradient-descent mechanics in depth. Here the update mainly
demonstrates why derivative information is useful.

## 12. Differentiation Strategies

| Strategy | Mechanism | Strength | Limitation |
| --- | --- | --- | --- |
| Analytical | derive a closed-form expression | exact and interpretable | manual work is error-prone at scale |
| Numerical | perturb inputs and evaluate the function | simple independent check | approximate and expensive |
| Symbolic | manipulate expressions | produces algebraic formulas | can create very large expressions |
| Automatic | apply local rules to executed operations | efficient and accurate to floating-point precision | requires a differentiable supported graph |

Centered finite differences estimate

\[
\frac{\partial L}{\partial\theta_j}
\approx
\frac{L(\theta_j+\varepsilon)-L(\theta_j-\varepsilon)}
{2\varepsilon}.
\]

Choosing \(\varepsilon\) involves a trade-off. A large value has truncation
error; an extremely small value suffers floating-point cancellation. Gradient
checks should use a small deterministic problem and, when available, double
precision.

## 13. Assumptions and Modeling Choices

Gradient-based training works best when:

- parameters are continuous;
- a scalar objective is differentiable or has useful subgradients;
- local sensitivity is informative;
- the numerical scale of inputs, activations, and losses is manageable;
- the computation graph preserves the required dependency paths.

Important choices include:

- **Loss definition:** determines what the optimizer actually improves.
- **Feature and parameter scale:** affects gradient magnitude and conditioning.
- **Batch size:** trades compute efficiency and gradient variance.
- **Learning rate:** controls whether the local approximation is respected.
- **Precision:** affects underflow, overflow, and finite-difference checks.
- **Frozen parameters:** restrict which paths receive and apply gradients.

No calculus result guarantees that the training objective matches a business
metric, causal goal, safety requirement, or user preference.

## 14. Applications

Gradients are central to:

- linear and logistic regression;
- neural-network and transformer training;
- embedding and cross-encoder reranker fine-tuning;
- parameter-efficient adaptation such as LoRA;
- matrix factorization and differentiable ranking objectives;
- custom losses and differentiable calibration;
- saliency and other gradient-based explainability methods.

An Applied AI system need not be differentiable end to end. For example, a RAG
system may use gradients to train an embedding model or reranker, while using
offline evaluations or search to choose chunk size, top-\(k\), prompt
templates, and workflow topology.

## 15. Trade-offs and Alternatives

Gradient methods scale to high-dimensional continuous parameter spaces, but
they provide local information and can be sensitive to conditioning,
initialization, learning rate, noise, and curvature.

Derivative-free methods may be preferable when:

- decisions are discrete;
- only a black-box API is available;
- evaluations are discontinuous or extremely expensive;
- the search space is small enough to enumerate.

Alternatives include grid search, random search, Bayesian optimization,
evolutionary methods, bandit algorithms, and other problem-specific searches.

First-order methods use gradients and scale well. Second-order methods use
curvature information and may converge faster locally, but full Hessian
storage and inversion are prohibitive at modern model sizes.

## 16. Limitations and Failure Modes

### Vanishing gradients

Repeated chain-rule factors with magnitude below one can make early-layer
gradients approach zero. Saturating activations and long dependency paths are
common causes. Residual connections, normalization, suitable activations, and
careful initialization help preserve signal.

### Exploding gradients

Repeated factors above one can make gradients extremely large, producing
unstable updates or `NaN` values. Lower learning rates, stable initialization,
normalization, gradient-norm monitoring, and clipping can help. Clipping treats
a symptom and may not remove the root cause.

### Stationary points and flat regions

A zero or very small gradient does not prove a global minimum. It may indicate
a local minimum, maximum, saddle point, flat plateau, saturation, or numerical
underflow.

### Non-differentiable decisions

Hard thresholds, `argmax`, integer choices, discrete sampling, and exact
ranking can interrupt useful gradients. A surrogate loss, soft relaxation,
straight-through estimator, or a non-gradient method may be needed.

### Objective misalignment

The optimizer minimizes the supplied loss. It does not know that the real goal
is answer correctness, calibrated risk, tail performance, retrieval quality,
latency, or business value unless those concerns are represented in training
and evaluation.

## 17. Common Mistakes

- Saying the gradient points toward a minimum; it points toward steepest local
  increase.
- Treating a local derivative as a global prediction.
- Forgetting an inner chain-rule factor.
- Confusing gradients with parameter updates or backpropagation with an
  optimizer.
- Ignoring tensor shapes and allowing unintended broadcasting.
- Assuming a correct gradient guarantees lower loss for any learning rate.
- Using finite differences as a scalable training algorithm.
- Evaluating gradient checks with an unsuitable \(\varepsilon\) or low
  precision.
- Breaking a computation graph with a detached value or discrete operation.
- Applying a probability activation twice when a loss expects logits.
- Looking only at loss while ignoring activations, parameter magnitudes, and
  per-layer gradient norms.
- Mistaking successful optimization for valid evaluation; leakage can make a
  model optimize misleadingly well.

## 18. Practical Debugging Checklist

When loss diverges or becomes `NaN`:

1. Check inputs, targets, parameters, and activations for non-finite values.
2. Reproduce the issue on one small deterministic batch.
3. Inspect loss terms involving logarithms, exponentials, and division.
4. Reduce the learning rate and inspect update-to-parameter ratios.
5. Monitor gradient norms by layer and before clipping.
6. Verify tensor shapes and reduction conventions such as sum versus mean.
7. Confirm that intended parameters require gradients and remain connected to
   the loss.
8. Compare a custom derivative with finite differences in double precision.
9. Check mixed-precision scaling, initialization, and normalization.
10. Revisit whether the objective and data split represent the intended task.
