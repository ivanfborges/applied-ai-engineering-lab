# Day 5 — Calculus for ML: Derivatives, Gradients and Chain Rule

## 1. Executive overview

Derivatives describe **how sensitive an output is to a small change in an input**. In Machine Learning, this becomes the mathematical foundation for answering questions such as:

* How much would the loss change if a model parameter changed slightly?
* Which parameter should be increased or decreased?
* How large should that update be?
* How does an error propagate through multiple layers of transformations?
* Why is a neural network failing to learn?
* Why are gradients exploding, vanishing or becoming `NaN`?

For a model with parameters (\theta) and loss function (L(\theta)), training is usually framed as:

[
\theta^\star = \arg\min_{\theta} L(\theta)
]

Where:

* (\theta) represents all trainable model parameters;
* (L) is the objective or loss function;
* (\theta^\star) is the set of parameters that minimizes the loss.

The gradient

[
\nabla_\theta L(\theta)
]

indicates the direction of greatest local increase in the loss. Therefore, optimization algorithms usually move in the opposite direction:

[
\theta_{t+1}
============

## \theta_t

\eta \nabla_\theta L(\theta_t)
]

Where:

* (t) is the current optimization step;
* (\eta) is the learning rate;
* (\nabla_\theta L) is the gradient of the loss with respect to the parameters.

In practice, this machinery appears in:

* linear and logistic regression;
* neural-network training;
* fine-tuning LLMs;
* embedding-model training;
* learned rerankers;
* computer vision models;
* differentiable ranking losses;
* reinforcement learning;
* calibration and probabilistic models;
* gradient-based explainability methods.

An Applied AI Engineer may rarely calculate these derivatives manually, but senior-level understanding is essential when debugging unstable training, selecting optimizers, freezing layers, implementing custom losses or explaining backpropagation in interviews.

---

## 2. Core intuition

Imagine the loss function as a landscape.

* Each coordinate represents a model parameter.
* The height represents the model loss.
* Training means finding a low point in this landscape.

For a model with two parameters (w_1) and (w_2), the loss is a surface:

[
L(w_1,w_2)
]

At a particular point, the gradient

[
\nabla L(w_1,w_2)
=================

\begin{bmatrix}
\frac{\partial L}{\partial w_1}\
\frac{\partial L}{\partial w_2}
\end{bmatrix}
]

points toward the steepest uphill direction.

To reduce the loss, gradient descent moves downhill:

[
-\nabla L(w_1,w_2)
]

The chain rule explains how to calculate that slope when the loss is produced by several nested transformations.

Consider:

[
x \rightarrow z \rightarrow a \rightarrow L
]

For example:

[
z = wx+b
]

[
a = \sigma(z)
]

[
L = \text{BinaryCrossEntropy}(a,y)
]

To understand how (w) affects (L), we follow the full dependency path:

[
\frac{\partial L}{\partial w}
=============================

\frac{\partial L}{\partial a}
\cdot
\frac{\partial a}{\partial z}
\cdot
\frac{\partial z}{\partial w}
]

That is the central mechanism behind backpropagation.

---

## 3. Theoretical foundations

### 3.1 Functions and local sensitivity

A function maps inputs to outputs:

[
y=f(x)
]

A derivative measures how much (y) changes when (x) changes slightly.

For a small perturbation (\Delta x):

[
f(x+\Delta x)
\approx
f(x)+f'(x)\Delta x
]

This is a first-order local approximation.

The derivative (f'(x)) acts as a local linear coefficient relating changes in (x) to changes in (f(x)).

---

### 3.2 Derivative

The derivative of a scalar-valued function of one scalar variable is:

[
f'(x)
=====

\lim_{h\to 0}
\frac{f(x+h)-f(x)}{h}
]

Where:

* (x) is the point at which the derivative is evaluated;
* (h) is a small perturbation;
* (f(x+h)-f(x)) is the corresponding output change;
* the ratio represents the rate of change.

For:

[
f(x)=x^2
]

We have:

[
f'(x)=2x
]

At (x=3):

[
f'(3)=6
]

This means that, locally, increasing (x) by approximately (0.01) increases (f(x)) by approximately:

[
6 \cdot 0.01=0.06
]

---

### 3.3 Continuity versus differentiability

Differentiability implies local smoothness.

A differentiable function must be continuous at that point, but a continuous function does not necessarily have to be differentiable.

For example:

[
f(x)=|x|
]

is continuous at (x=0), but the left and right derivatives differ:

[
\lim_{h\to 0^-}\frac{|h|}{h}=-1
]

[
\lim_{h\to 0^+}\frac{|h|}{h}=1
]

Therefore, the derivative at (x=0) is not uniquely defined.

This matters because models often contain non-smooth operations such as ReLU:

[
\operatorname{ReLU}(x)=\max(0,x)
]

At (x=0), deep-learning frameworks adopt a practical convention, usually setting the derivative to zero.

---

### 3.4 Partial derivatives

When a scalar function depends on multiple variables:

[
f(x_1,x_2,\ldots,x_n)
]

a partial derivative measures sensitivity to one variable while holding the others constant.

For:

[
f(x,y)=x^2+3xy+y^2
]

The partial derivative with respect to (x) is:

[
\frac{\partial f}{\partial x}=2x+3y
]

The partial derivative with respect to (y) is:

[
\frac{\partial f}{\partial y}=3x+2y
]

Each derivative isolates one input direction.

---

### 3.5 Gradient

The gradient combines all partial derivatives of a scalar-valued function:

[
\nabla f(\mathbf{x})
====================

\begin{bmatrix}
\frac{\partial f}{\partial x_1}\
\frac{\partial f}{\partial x_2}\
\vdots\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
]

Where:

* (\mathbf{x}\in\mathbb{R}^n) is the input vector;
* (f:\mathbb{R}^n\rightarrow\mathbb{R}) is a scalar-valued function;
* (\nabla f(\mathbf{x})\in\mathbb{R}^n).

The gradient has three important interpretations:

1. It points toward the direction of greatest local increase.
2. Its negative points toward the direction of greatest local decrease.
3. Its magnitude indicates how steep the function is locally.

---

### 3.6 Directional derivative

Suppose we do not want to move only along one coordinate. Instead, we want to know how the function changes in a direction (\mathbf{u}).

For a unit vector (\mathbf{u}), where

[
|\mathbf{u}|_2=1
]

the directional derivative is:

[
D_{\mathbf{u}}f(\mathbf{x})
===========================

\nabla f(\mathbf{x})^\top \mathbf{u}
]

Where:

* (D_{\mathbf{u}}f) is the rate of change in direction (\mathbf{u});
* (\nabla f(\mathbf{x})^\top) is the transposed gradient;
* the dot product projects the gradient onto the selected direction.

This formula also explains why the gradient is the steepest direction: the dot product is maximized when (\mathbf{u}) points in the same direction as (\nabla f).

---

### 3.7 Jacobian

A gradient applies to a scalar-valued output. For a vector-valued function:

[
\mathbf{f}:\mathbb{R}^n\rightarrow\mathbb{R}^m
]

the collection of first-order partial derivatives is the Jacobian:

[
J_{\mathbf{f}}(\mathbf{x})
==========================

\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n}\
\vdots & \ddots & \vdots\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
]

Its shape is:

[
m\times n
]

The Jacobian describes how each output component changes with respect to each input component.

This is relevant for:

* neural-network layers;
* vector embeddings;
* coordinate transformations;
* multi-output models;
* vector-valued activations.

---

### 3.8 Hessian

The Hessian contains second-order partial derivatives of a scalar-valued function:

[
H_f(\mathbf{x})
===============

\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2}
&
\cdots
&
\frac{\partial^2 f}{\partial x_1\partial x_n}\
\vdots
&
\ddots
&
\vdots\
\frac{\partial^2 f}{\partial x_n\partial x_1}
&
\cdots
&
\frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
]

The gradient describes slope. The Hessian describes curvature.

It can help distinguish:

* local minima;
* local maxima;
* saddle points;
* flat versus sharp regions.

For large neural networks, explicitly computing the full Hessian is usually impractical because it has (n^2) entries for (n) parameters.

---

### 3.9 Chain rule

For a composition:

[
y=f(g(x))
]

the derivative is:

[
\frac{dy}{dx}
=============

\frac{df}{dg}
\cdot
\frac{dg}{dx}
]

Equivalently:

[
\frac{d}{dx}f(g(x))
===================

f'(g(x))g'(x)
]

For vector-valued intermediate results, let:

[
\mathbf{z}=g(\mathbf{x})
]

and:

[
L=h(\mathbf{z})
]

where (L) is scalar. Then:

[
\nabla_{\mathbf{x}}L
====================

J_g(\mathbf{x})^\top
\nabla_{\mathbf{z}}L
]

This is the vector form of the chain rule commonly used in backpropagation.

---

### 3.10 Computational graphs and backpropagation

A computational graph decomposes a complex function into small operations.

Consider:

[
z=wx+b
]

[
a=\tanh(z)
]

[
L=(a-y)^2
]

The forward pass computes:

[
x \rightarrow z \rightarrow a \rightarrow L
]

The backward pass computes derivatives in reverse:

[
\frac{\partial L}{\partial a}
\rightarrow
\frac{\partial L}{\partial z}
\rightarrow
\frac{\partial L}{\partial w},
\frac{\partial L}{\partial b}
]

Backpropagation is not a separate optimization algorithm. It is an efficient application of the chain rule to a computational graph.

The optimizer, such as SGD or Adam, consumes the resulting gradients and updates the parameters.

---

## 4. Mathematical, statistical or logical foundations

### 4.1 Common derivatives

For a constant (c):

[
\frac{d}{dx}c=0
]

For a power:

[
\frac{d}{dx}x^k=kx^{k-1}
]

For the exponential:

[
\frac{d}{dx}e^x=e^x
]

For the natural logarithm:

[
\frac{d}{dx}\ln(x)=\frac{1}{x},
\qquad x>0
]

For the sigmoid function:

[
\sigma(z)=\frac{1}{1+e^{-z}}
]

Its derivative is:

[
\sigma'(z)
==========

\sigma(z)\left(1-\sigma(z)\right)
]

For the hyperbolic tangent:

[
\frac{d}{dz}\tanh(z)
====================

1-\tanh^2(z)
]

For ReLU:

[
\operatorname{ReLU}'(z)
=======================

\begin{cases}
0, & z<0\
1, & z>0
\end{cases}
]

The derivative at zero is undefined mathematically, but frameworks choose a convention.

---

### 4.2 Deriving gradients for linear regression

Consider a one-dimensional linear model:

[
\hat{y}_i=wx_i+b
]

Where:

* (x_i) is the (i)-th input;
* (y_i) is the observed target;
* (\hat{y}_i) is the prediction;
* (w) is the weight;
* (b) is the bias.

The mean squared error is:

[
L(w,b)
======

\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
]

Substituting the prediction:

[
L(w,b)
======

\frac{1}{n}
\sum_{i=1}^{n}
(wx_i+b-y_i)^2
]

Define the residual:

[
e_i=wx_i+b-y_i
]

Then:

[
L(w,b)=\frac{1}{n}\sum_{i=1}^{n}e_i^2
]

Using the chain rule:

[
\frac{\partial L}{\partial w}
=============================

\frac{1}{n}
\sum_{i=1}^{n}
2e_i
\frac{\partial e_i}{\partial w}
]

Since:

[
\frac{\partial e_i}{\partial w}=x_i
]

we obtain:

[
\frac{\partial L}{\partial w}
=============================

\frac{2}{n}
\sum_{i=1}^{n}
e_i x_i
]

Similarly:

[
\frac{\partial e_i}{\partial b}=1
]

Therefore:

[
\frac{\partial L}{\partial b}
=============================

\frac{2}{n}
\sum_{i=1}^{n}
e_i
]

The parameter updates are:

[
w_{t+1}
=======

## w_t

\eta
\frac{\partial L}{\partial w}
]

[
b_{t+1}
=======

## b_t

\eta
\frac{\partial L}{\partial b}
]

---

### 4.3 Multivariate linear regression

For a dataset represented by matrix (X):

[
\hat{\mathbf{y}}
================

X\mathbf{w}+b\mathbf{1}
]

Where:

* (X\in\mathbb{R}^{n\times d});
* (n) is the number of observations;
* (d) is the number of features;
* (\mathbf{w}\in\mathbb{R}^d);
* (\mathbf{1}\in\mathbb{R}^n) is a vector of ones;
* (\hat{\mathbf{y}}\in\mathbb{R}^n).

The MSE loss is:

[
L(\mathbf{w},b)
===============

\frac{1}{n}
\left|
X\mathbf{w}+b\mathbf{1}-\mathbf{y}
\right|_2^2
]

The gradients are:

[
\nabla_{\mathbf{w}}L
====================

\frac{2}{n}
X^\top
\left(
X\mathbf{w}+b\mathbf{1}-\mathbf{y}
\right)
]

[
\frac{\partial L}{\partial b}
=============================

\frac{2}{n}
\mathbf{1}^\top
\left(
X\mathbf{w}+b\mathbf{1}-\mathbf{y}
\right)
]

The transpose (X^\top) ensures that the resulting gradient has the same dimensionality as (\mathbf{w}).

---

### 4.4 Logistic regression and a useful chain-rule result

For binary classification:

[
z=\mathbf{w}^\top\mathbf{x}+b
]

[
p=\sigma(z)
]

Where (p) is the predicted probability.

Binary cross-entropy for one observation is:

[
L
=

## -y\log(p)

(1-y)\log(1-p)
]

Where:

* (y\in{0,1}) is the observed label;
* (p\in(0,1)) is the predicted probability.

After applying the chain rule through the cross-entropy and sigmoid:

[
\frac{\partial L}{\partial z}=p-y
]

Therefore:

[
\frac{\partial L}{\partial \mathbf{w}}
======================================

(p-y)\mathbf{x}
]

[
\frac{\partial L}{\partial b}=p-y
]

This is an important interview result. The sigmoid and binary cross-entropy derivatives simplify into the prediction error at the logit level.

In production, implementations usually combine sigmoid and cross-entropy into a numerically stable operation such as PyTorch's `BCEWithLogitsLoss`.

---

### 4.5 Gradient descent

The generic gradient-descent update is:

[
\theta_{t+1}
============

## \theta_t

\eta\nabla_\theta L(\theta_t)
]

A first-order Taylor approximation helps explain why this works:

[
L(\theta+\Delta\theta)
\approx
L(\theta)
+
\nabla_\theta L(\theta)^\top\Delta\theta
]

Select:

[
\Delta\theta=-\eta\nabla_\theta L(\theta)
]

Then:

[
L(\theta+\Delta\theta)
\approx
L(\theta)
---------

\eta
\left|
\nabla_\theta L(\theta)
\right|_2^2
]

For a sufficiently small positive learning rate (\eta), the first-order approximation predicts a reduction in the loss.

This is a local statement. A large learning rate can invalidate the approximation and increase the loss.

---

### 4.6 Why minibatches produce noisy gradients

The full empirical loss is:

[
L(\theta)
=========

\frac{1}{n}
\sum_{i=1}^{n}
\ell_i(\theta)
]

Its gradient is:

[
\nabla_\theta L
===============

\frac{1}{n}
\sum_{i=1}^{n}
\nabla_\theta \ell_i
]

A minibatch (B) estimates this gradient:

[
\widehat{\nabla_\theta L}
=========================

\frac{1}{|B|}
\sum_{i\in B}
\nabla_\theta \ell_i
]

Where (|B|) is the batch size.

This estimate is noisy but usually much cheaper. The noise can also help optimization move away from saddle points or narrow regions.

---

### 4.7 Reverse-mode automatic differentiation

For a model with millions of parameters and one scalar loss, reverse-mode automatic differentiation is especially efficient.

It computes products of the form:

[
\mathbf{v}^\top J
]

without materializing the full Jacobian.

These are known as vector-Jacobian products.

Backpropagation is effectively reverse-mode automatic differentiation specialized to computational graphs.

---

## 5. Practical applicability

### Where derivatives and gradients are useful

They are central when:

* the model has trainable continuous parameters;
* the objective can be differentiated or approximated by a differentiable function;
* the parameter space is too large for exhaustive search;
* local gradient information provides useful optimization directions.

Examples include:

* fitting linear and logistic regression;
* training deep neural networks;
* fine-tuning transformer models;
* parameter-efficient fine-tuning with LoRA;
* training embedding or reranking models;
* optimizing differentiable ranking losses;
* matrix factorization;
* probabilistic models;
* differentiable calibration;
* adversarial-example generation;
* saliency and gradient-based explainability.

---

### Applied AI and GenAI connections

In an LLM or RAG system, most application components are not trained end to end. However, gradients still matter when:

* fine-tuning an embedding model;
* fine-tuning an LLM;
* training a cross-encoder reranker;
* optimizing a classifier used for routing;
* adapting a reward or preference model;
* implementing LoRA;
* diagnosing unstable mixed-precision training;
* deciding which model layers to freeze;
* creating a custom contrastive loss.

By contrast, prompt selection, chunk-size tuning, retrieval parameters and agent workflow decisions are often optimized with experiments, search or evaluation loops rather than direct gradient descent.

---

### When gradient-based optimization may not make sense

It may be unsuitable when:

* parameters are discrete;
* the objective is non-differentiable;
* evaluations are extremely expensive;
* the system behaves as a black box;
* the objective is noisy or discontinuous;
* the decision space is small enough for exhaustive search;
* gradients are unavailable through an external API.

Examples:

* selecting a prompt template;
* deciding the number of retrieved chunks;
* choosing an agent topology;
* optimizing hard business rules;
* tuning discrete infrastructure configurations;
* optimizing a third-party API without access to internals.

Alternatives include:

* grid search;
* random search;
* Bayesian optimization;
* evolutionary algorithms;
* reinforcement learning;
* bandit methods;
* derivative-free optimization.

---

### Trade-offs

Gradient methods are scalable and efficient, but they are local.

They may be affected by:

* poor parameter initialization;
* bad feature scaling;
* noisy gradients;
* saddle points;
* flat regions;
* exploding or vanishing gradients;
* poorly chosen learning rates;
* objectives misaligned with business requirements.

A mathematically correct gradient does not guarantee a useful production model. It only means the chosen loss is being optimized correctly.

---

## 6. Common pitfalls and mistakes

### 6.1 Saying that the gradient points toward the minimum

The gradient points toward the greatest local increase.

The negative gradient points toward the greatest local decrease:

[
-\nabla L
]

---

### 6.2 Confusing derivative, partial derivative and gradient

* A derivative usually refers to a scalar function of one scalar variable.
* A partial derivative varies one input while holding the others fixed.
* A gradient collects partial derivatives for a scalar-valued multivariate function.
* A Jacobian applies to vector-valued outputs.
* A Hessian contains second-order derivatives of a scalar function.

---

### 6.3 Forgetting the chain rule

For:

[
L=(wx+b-y)^2
]

an incorrect answer might differentiate only the square and forget the derivative of the inner expression.

The correct result is:

[
\frac{\partial L}{\partial w}
=============================

2(wx+b-y)x
]

The factor (x) comes from differentiating (wx+b-y) with respect to (w).

---

### 6.4 Ignoring tensor shapes

Many gradient bugs are shape bugs.

For:

[
X\in\mathbb{R}^{n\times d}
]

and:

[
\mathbf{w}\in\mathbb{R}^{d}
]

the expression:

[
X^\top(\hat{\mathbf{y}}-\mathbf{y})
]

returns a vector in (\mathbb{R}^d), matching the shape of (\mathbf{w}).

Unexpected broadcasting can produce code that runs but computes the wrong objective.

---

### 6.5 Treating differentiability as global smoothness

A function can be differentiable almost everywhere but not everywhere.

ReLU is the common example. Optimization can still work because the non-differentiable point is handled by a subgradient convention.

---

### 6.6 Using non-differentiable operations inside training

Operations such as:

* `argmax`;
* hard thresholds;
* discrete sampling;
* exact sorting;
* integer decisions;

can break gradient flow.

Training may require:

* softmax instead of `argmax`;
* differentiable approximations;
* straight-through estimators;
* policy-gradient methods;
* surrogate losses.

---

### 6.7 Vanishing gradients

When chain-rule terms repeatedly have magnitudes smaller than one, their product can approach zero.

For a deep composition:

[
\frac{\partial L}{\partial x}
=============================

\frac{\partial L}{\partial h_k}
\prod_{j=1}^{k}
\frac{\partial h_j}{\partial h_{j-1}}
]

If many factors are small, early layers receive almost no learning signal.

Mitigations include:

* ReLU-family activations;
* residual connections;
* normalization;
* careful initialization;
* gated architectures;
* shorter effective gradient paths.

---

### 6.8 Exploding gradients

When the chain-rule factors repeatedly have magnitudes above one, gradients can become extremely large.

Symptoms include:

* unstable loss;
* `NaN` values;
* very large parameter updates;
* divergence.

Mitigations include:

* gradient clipping;
* lower learning rates;
* stable initialization;
* normalization;
* mixed-precision loss scaling;
* monitoring gradient norms.

---

### 6.9 Assuming a correct gradient guarantees a lower loss

A gradient only provides a local direction.

The loss can still increase because:

* the learning rate is too high;
* the minibatch gradient is noisy;
* momentum carries the parameters past a good region;
* numerical precision causes instability;
* the objective changes due to stochastic layers.

---

### 6.10 Using finite differences as the training mechanism

Finite differences approximate:

[
f'(x)
\approx
\frac{f(x+\varepsilon)-f(x-\varepsilon)}{2\varepsilon}
]

This is useful for gradient checking but scales poorly.

For (p) parameters, it requires approximately (2p) forward evaluations to estimate the full gradient.

Backpropagation can compute gradients for all parameters with a cost on the same order as a small number of forward passes.

---

### 6.11 Choosing the wrong objective

The optimizer minimizes the supplied loss, not the business goal.

Examples:

* optimizing token-level accuracy while caring about full-answer correctness;
* optimizing cross-entropy while ignoring severe class imbalance;
* optimizing retrieval similarity without measuring downstream answer quality;
* minimizing average error while tail-risk cases are business-critical.

This is not a calculus error, but it is one of the most important senior-level limitations of gradient-based training.

---

### 6.12 PyTorch-specific mistakes

Common implementation mistakes include:

* forgetting `optimizer.zero_grad()`;
* accidentally detaching tensors with `.detach()`;
* converting tensors to NumPy during the forward graph;
* using `torch.no_grad()` during training;
* modifying tensors in place;
* calling `.backward()` multiple times without retaining the graph;
* forgetting `model.train()` or `model.eval()`;
* applying sigmoid before `BCEWithLogitsLoss`;
* inspecting only loss without monitoring gradient norms.

---

### 6.13 Leakage and evaluation errors

Data leakage is not caused by derivatives, but gradient-based models can exploit leaked information extremely well.

A low training or validation loss is meaningless when:

* future information leaks into features;
* preprocessing is fitted before the train-test split;
* entities overlap incorrectly across splits;
* evaluation data influences model or prompt selection.

Optimization quality cannot compensate for an invalid evaluation design.

---

## 7. Important comparisons

### Derivative-related objects

| Concept            |  Input | Output | Meaning                                    |
| ------------------ | -----: | -----: | ------------------------------------------ |
| Derivative         | Scalar | Scalar | Sensitivity of one scalar to another       |
| Partial derivative | Vector | Scalar | Sensitivity to one selected input          |
| Gradient           | Vector | Scalar | All first-order partial derivatives        |
| Jacobian           | Vector | Vector | Sensitivity of every output to every input |
| Hessian            | Vector | Scalar | Second-order curvature information         |

---

### Analytical, numerical and automatic differentiation

| Approach                   | How it works                              | Advantages                                         | Limitations                                  |
| -------------------------- | ----------------------------------------- | -------------------------------------------------- | -------------------------------------------- |
| Analytical differentiation | Manually derives a formula                | Exact and interpretable                            | Error-prone and hard for large graphs        |
| Numerical differentiation  | Uses finite perturbations                 | Simple and useful for checking                     | Slow and numerically sensitive               |
| Symbolic differentiation   | Manipulates mathematical expressions      | Produces symbolic formulas                         | Expression explosion                         |
| Automatic differentiation  | Applies chain rule to executed operations | Efficient and exact up to floating-point precision | Requires supported differentiable operations |

Automatic differentiation is not symbolic differentiation and is not finite-difference approximation.

It decomposes the program into elementary operations and applies exact local derivative rules.

---

### Forward mode versus reverse mode

| Mode         | Efficient when           | Typical use                                              |
| ------------ | ------------------------ | -------------------------------------------------------- |
| Forward mode | Few inputs, many outputs | Sensitivity with respect to a small number of variables  |
| Reverse mode | Many inputs, few outputs | Neural networks with millions of parameters and one loss |

Deep learning primarily uses reverse mode.

---

### Full-batch, stochastic and minibatch gradient descent

| Method     | Gradient source        | Advantages                                          | Disadvantages                  |
| ---------- | ---------------------- | --------------------------------------------------- | ------------------------------ |
| Full batch | Entire dataset         | Stable gradient                                     | Expensive and memory intensive |
| Stochastic | One observation        | Cheap updates                                       | Very noisy                     |
| Minibatch  | Subset of observations | Efficient hardware utilization and manageable noise | Requires batch-size tuning     |

---

### First-order versus second-order optimization

| Type         | Information            | Examples                     | Trade-off                              |
| ------------ | ---------------------- | ---------------------------- | -------------------------------------- |
| First order  | Gradient               | SGD, Adam, RMSProp           | Scales well to large models            |
| Second order | Gradient and curvature | Newton, quasi-Newton, L-BFGS | Faster local convergence but expensive |

Newton's method uses:

[
\theta_{t+1}
============

## \theta_t

H^{-1}\nabla L
]

Where (H^{-1}) is the inverse Hessian.

For modern deep networks, explicitly storing or inverting the Hessian is generally infeasible.

---

### Gradient descent versus derivative-free optimization

Use gradient descent when:

* gradients are available;
* variables are continuous;
* the parameter space is large;
* the objective is sufficiently smooth.

Use derivative-free methods when:

* the system is a black box;
* decisions are discrete;
* evaluations are expensive;
* gradients are unavailable or unreliable.

---

## 8. Practical Python example

The following example fits a linear model using gradients derived manually.

Create `example.py`:

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def mean_squared_error(
    x: np.ndarray,
    y: np.ndarray,
    weight: float,
    bias: float,
) -> float:
    predictions = weight * x + bias
    errors = predictions - y
    return float(np.mean(errors**2))


def compute_gradients(
    x: np.ndarray,
    y: np.ndarray,
    weight: float,
    bias: float,
) -> tuple[float, float]:
    """
    Compute the analytical gradients of MSE for:

        prediction = weight * x + bias
    """
    predictions = weight * x + bias
    errors = predictions - y

    d_loss_d_weight = 2.0 * np.mean(errors * x)
    d_loss_d_bias = 2.0 * np.mean(errors)

    return float(d_loss_d_weight), float(d_loss_d_bias)


def numerical_gradient_check(
    x: np.ndarray,
    y: np.ndarray,
    weight: float,
    bias: float,
    epsilon: float = 1e-6,
) -> tuple[float, float]:
    """
    Estimate gradients using centered finite differences.
    This is useful for validation, not for model training.
    """
    numerical_weight_gradient = (
        mean_squared_error(x, y, weight + epsilon, bias)
        - mean_squared_error(x, y, weight - epsilon, bias)
    ) / (2.0 * epsilon)

    numerical_bias_gradient = (
        mean_squared_error(x, y, weight, bias + epsilon)
        - mean_squared_error(x, y, weight, bias - epsilon)
    ) / (2.0 * epsilon)

    return numerical_weight_gradient, numerical_bias_gradient


def main() -> None:
    rng = np.random.default_rng(42)

    x = rng.uniform(-3.0, 3.0, size=200)
    noise = rng.normal(0.0, 0.8, size=x.shape)

    true_weight = 3.2
    true_bias = -1.5
    y = true_weight * x + true_bias + noise

    weight = 0.0
    bias = 0.0
    learning_rate = 0.05
    epochs = 150

    analytical_gradients = compute_gradients(x, y, weight, bias)
    numerical_gradients = numerical_gradient_check(x, y, weight, bias)

    print("Initial gradient check")
    print(f"Analytical dw: {analytical_gradients[0]:.8f}")
    print(f"Numerical  dw: {numerical_gradients[0]:.8f}")
    print(f"Analytical db: {analytical_gradients[1]:.8f}")
    print(f"Numerical  db: {numerical_gradients[1]:.8f}")

    losses: list[float] = []

    for epoch in range(epochs):
        loss = mean_squared_error(x, y, weight, bias)
        d_weight, d_bias = compute_gradients(x, y, weight, bias)

        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias

        losses.append(loss)

        if epoch % 25 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss:.4f} | "
                f"Weight: {weight:.4f} | "
                f"Bias: {bias:.4f}"
            )

    print("\nFinal parameters")
    print(f"Estimated weight: {weight:.4f}")
    print(f"Estimated bias:   {bias:.4f}")
    print(f"True weight:      {true_weight:.4f}")
    print(f"True bias:        {true_bias:.4f}")

    sorted_indices = np.argsort(x)
    sorted_x = x[sorted_indices]
    fitted_y = weight * sorted_x + bias

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, alpha=0.6, label="Synthetic observations")
    plt.plot(sorted_x, fitted_y, linewidth=2, label="Fitted model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Trained with Gradient Descent")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Optimization Curve")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

Install and run:

```bash
pip install numpy matplotlib
python example.py
```

### What this example demonstrates

The analytical gradients:

[
\frac{\partial L}{\partial w}
=============================

\frac{2}{n}\sum_i(\hat{y}_i-y_i)x_i
]

[
\frac{\partial L}{\partial b}
=============================

\frac{2}{n}\sum_i(\hat{y}_i-y_i)
]

are compared with centered finite differences.

The two results should be extremely close. This is a standard technique for validating custom backward implementations.

---

### Small PyTorch autograd example

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 4.0, 6.0])

weight = torch.tensor(0.0, requires_grad=True)
bias = torch.tensor(0.0, requires_grad=True)

predictions = weight * x + bias
loss = torch.mean((predictions - y) ** 2)

loss.backward()

print("Loss:", loss.item())
print("dL/dw:", weight.grad.item())
print("dL/db:", bias.grad.item())
```

PyTorch records the executed tensor operations and constructs a computational graph. Calling `loss.backward()` applies reverse-mode automatic differentiation.

---

## 9. From-scratch implementation when useful

This example manually backpropagates through one nonlinear neuron:

[
z=wx+b
]

[
a=\tanh(z)
]

[
L=(a-y)^2
]

Create `from_scratch.py`:

```python
from __future__ import annotations

import math


def forward(
    x: float,
    target: float,
    weight: float,
    bias: float,
) -> tuple[float, float, float]:
    z = weight * x + bias
    activation = math.tanh(z)
    loss = (activation - target) ** 2
    return z, activation, loss


def manual_backward(
    x: float,
    target: float,
    weight: float,
    bias: float,
) -> dict[str, float]:
    z, activation, loss = forward(x, target, weight, bias)

    # L = (a - y)^2
    d_loss_d_activation = 2.0 * (activation - target)

    # a = tanh(z)
    d_activation_d_z = 1.0 - activation**2

    # Chain rule: dL/dz = dL/da * da/dz
    d_loss_d_z = d_loss_d_activation * d_activation_d_z

    # z = w*x + b
    d_z_d_weight = x
    d_z_d_bias = 1.0

    # Chain rule through the complete graph
    d_loss_d_weight = d_loss_d_z * d_z_d_weight
    d_loss_d_bias = d_loss_d_z * d_z_d_bias

    return {
        "z": z,
        "activation": activation,
        "loss": loss,
        "d_loss_d_activation": d_loss_d_activation,
        "d_activation_d_z": d_activation_d_z,
        "d_loss_d_z": d_loss_d_z,
        "d_loss_d_weight": d_loss_d_weight,
        "d_loss_d_bias": d_loss_d_bias,
    }


def numerical_gradient(
    x: float,
    target: float,
    weight: float,
    bias: float,
    parameter: str,
    epsilon: float = 1e-6,
) -> float:
    if parameter == "weight":
        loss_plus = forward(x, target, weight + epsilon, bias)[2]
        loss_minus = forward(x, target, weight - epsilon, bias)[2]
    elif parameter == "bias":
        loss_plus = forward(x, target, weight, bias + epsilon)[2]
        loss_minus = forward(x, target, weight, bias - epsilon)[2]
    else:
        raise ValueError("parameter must be 'weight' or 'bias'")

    return (loss_plus - loss_minus) / (2.0 * epsilon)


def main() -> None:
    x = 1.5
    target = 0.8
    weight = 0.4
    bias = -0.2

    results = manual_backward(x, target, weight, bias)

    numerical_d_weight = numerical_gradient(
        x=x,
        target=target,
        weight=weight,
        bias=bias,
        parameter="weight",
    )

    numerical_d_bias = numerical_gradient(
        x=x,
        target=target,
        weight=weight,
        bias=bias,
        parameter="bias",
    )

    print("Forward pass")
    print(f"z:          {results['z']:.8f}")
    print(f"activation: {results['activation']:.8f}")
    print(f"loss:       {results['loss']:.8f}")

    print("\nBackward pass")
    print(f"dL/da: {results['d_loss_d_activation']:.8f}")
    print(f"da/dz: {results['d_activation_d_z']:.8f}")
    print(f"dL/dz: {results['d_loss_d_z']:.8f}")

    print("\nGradient check")
    print(f"Manual dL/dw:    {results['d_loss_d_weight']:.8f}")
    print(f"Numerical dL/dw: {numerical_d_weight:.8f}")
    print(f"Manual dL/db:    {results['d_loss_d_bias']:.8f}")
    print(f"Numerical dL/db: {numerical_d_bias:.8f}")


if __name__ == "__main__":
    main()
```

The full weight derivative is:

[
\frac{\partial L}{\partial w}
=============================

\frac{\partial L}{\partial a}
\cdot
\frac{\partial a}{\partial z}
\cdot
\frac{\partial z}{\partial w}
]

Substituting each local derivative:

[
\frac{\partial L}{\partial w}
=============================

2(a-y)
\left(1-a^2\right)
x
]

The implementation demonstrates exactly how deep-learning frameworks propagate gradients through computational graphs.

---

## 10. Suggested experiments

### Experiment 1 — Change the learning rate

Try:

```python
learning_rate = 0.001
learning_rate = 0.05
learning_rate = 0.5
learning_rate = 1.0
```

Observe:

* slow convergence;
* stable convergence;
* oscillation;
* divergence.

Relate the behavior to the local approximation used by gradient descent.

---

### Experiment 2 — Change feature scale

Multiply the input by 100:

```python
x = x * 100
```

Run training without changing the learning rate.

Then standardize it:

```python
x = (x - x.mean()) / x.std()
```

Observe how feature scale changes gradient magnitude and optimization stability.

---

### Experiment 3 — Compare analytical and numerical gradients

Test several values of:

```python
epsilon = 1e-2
epsilon = 1e-4
epsilon = 1e-6
epsilon = 1e-10
```

A large (\varepsilon) creates approximation error. An extremely small (\varepsilon) introduces floating-point cancellation.

---

### Experiment 4 — Replace `tanh`

In `from_scratch.py`, replace `tanh` with sigmoid or ReLU and derive the new backward pass.

Observe:

* saturation with sigmoid;
* zero gradients for negative ReLU inputs;
* different gradient magnitudes.

---

### Experiment 5 — Add minibatches

Modify the regression example to calculate gradients using random minibatches.

Compare:

* optimization-curve noise;
* number of updates;
* final parameter estimates;
* sensitivity to batch size.

Suggested values:

```python
batch_size = 1
batch_size = 16
batch_size = 64
batch_size = len(x)
```

---

## 11. Senior interview questions

### 1. What does a gradient represent?

The gradient is the vector of first-order partial derivatives of a scalar-valued function with respect to its inputs. It points toward the direction of greatest local increase, while its negative indicates the steepest local decrease under the Euclidean norm.

In model training, (\nabla_\theta L) indicates the local sensitivity of the loss to every trainable parameter.

---

### 2. Why does gradient descent use the negative gradient?

Using the first-order Taylor approximation:

[
L(\theta+\Delta\theta)
\approx
L(\theta)
+
\nabla L(\theta)^\top\Delta\theta
]

Choosing:

[
\Delta\theta=-\eta\nabla L(\theta)
]

makes the linear term negative:

[
-\eta|\nabla L(\theta)|_2^2
]

Therefore, for a sufficiently small learning rate, the loss is expected to decrease locally.

---

### 3. What is the relationship between the chain rule and backpropagation?

Backpropagation is an efficient algorithm for applying the chain rule through a computational graph.

Each operation computes:

1. its forward output;
2. a local derivative;
3. the product of the upstream gradient and the local derivative.

Reverse-mode automatic differentiation reuses intermediate results, avoiding repeated calculations.

---

### 4. Is backpropagation an optimizer?

No.

Backpropagation computes gradients. Optimizers use those gradients to update parameters.

Examples:

* Backpropagation: computes (\nabla_\theta L).
* SGD, Adam and RMSProp: decide how to update (\theta).

---

### 5. Why is reverse-mode automatic differentiation appropriate for neural networks?

A neural network often has millions or billions of input parameters but produces one scalar loss.

Reverse mode efficiently computes the gradient of one scalar output with respect to many inputs. Its cost is proportional to a small multiple of the forward computation rather than proportional to the number of parameters.

---

### 6. How is automatic differentiation different from numerical differentiation?

Numerical differentiation estimates derivatives through perturbations, such as finite differences. It is approximate and requires repeated function evaluations.

Automatic differentiation decomposes executed operations and applies exact derivative rules through the chain rule. Its results are exact up to floating-point precision.

---

### 7. What causes vanishing gradients?

Vanishing gradients arise when backpropagation repeatedly multiplies by derivatives with magnitudes smaller than one.

This is common with saturated sigmoid or tanh activations and long computational paths.

It can prevent early layers from receiving useful learning signals.

---

### 8. What causes exploding gradients?

Exploding gradients occur when repeated Jacobian products increase gradient magnitude excessively.

This can be caused by:

* unstable initialization;
* recurrent computations;
* high learning rates;
* unnormalized activations;
* poorly conditioned objectives.

Gradient clipping limits the update magnitude but does not necessarily solve the underlying cause.

---

### 9. ReLU is not differentiable at zero. Why does training still work?

The non-differentiable point is isolated. Frameworks assign a practical subgradient convention at zero, commonly zero.

Gradient-based optimization does not require perfect global differentiability at every point. Differentiability almost everywhere is usually sufficient for practical training.

---

### 10. What is a gradient check?

A gradient check compares an analytical or automatically differentiated gradient with a numerical finite-difference estimate.

For one parameter:

[
\frac{\partial L}{\partial \theta}
\approx
\frac{L(\theta+\varepsilon)-L(\theta-\varepsilon)}
{2\varepsilon}
]

It is useful when implementing custom losses, layers or backward operations.

It should use small models and double precision when possible because finite differences are computationally expensive and sensitive to numerical error.

---

### 11. Why might loss increase even when the gradients are correct?

Possible causes include:

* excessive learning rate;
* stochastic minibatch noise;
* momentum overshooting;
* bad parameter scaling;
* mixed-precision instability;
* exploding gradients;
* a highly curved region;
* an optimizer state that is no longer appropriate.

A correct gradient describes local sensitivity, not a guaranteed improvement for an arbitrary step size.

---

### 12. What is the difference between a gradient and a Jacobian?

A gradient is the derivative of a scalar-valued function with respect to a vector.

A Jacobian is the derivative of a vector-valued function with respect to another vector.

The gradient can be understood as a special case of a Jacobian when the output dimension is one.

---

### 13. What role does the Hessian play?

The Hessian describes local curvature.

It can indicate:

* whether a stationary point is a minimum, maximum or saddle point;
* whether different parameter directions have very different curvature;
* how well-conditioned the optimization problem is.

Full Hessians are generally too expensive for large neural networks, so practical methods use approximations, Hessian-vector products or first-order optimizers.

---

### 14. How would you debug a model whose loss becomes `NaN`?

I would inspect:

1. input data for `NaN`, infinity and extreme values;
2. loss inputs, especially logarithms and divisions;
3. learning rate;
4. gradient norms by layer;
5. activation and parameter magnitudes;
6. mixed-precision loss scaling;
7. custom loss and backward implementations;
8. normalization and initialization;
9. whether clipping only hides a deeper instability.

I would also run a small deterministic batch and enable anomaly detection when supported.

---

### 15. How does this apply to fine-tuning an LLM?

Fine-tuning computes the loss between model outputs and training targets, then backpropagates gradients through the transformer.

With full fine-tuning, gradients update most or all model parameters. With LoRA, the base model is frozen and gradients update only low-rank adapter parameters.

Key production considerations include:

* memory required for activations and optimizer states;
* gradient accumulation;
* mixed precision;
* clipping;
* learning-rate schedules;
* frozen versus trainable layers;
* checkpointing;
* distributed gradient synchronization.

---

### 16. How would you optimize a RAG system when most components are non-differentiable?

I would separate trainable and non-trainable components.

Gradient-based optimization could be used for:

* embedding fine-tuning;
* reranker training;
* classifier or routing-model training.

For discrete application decisions such as chunk size, top-(k), prompt template and agent workflow, I would use an evaluation dataset combined with search, controlled experiments or Bayesian optimization.

The entire system does not need to be differentiable to improve it systematically.

---

## 12. Interview-ready explanation

Derivatives measure how sensitive one quantity is to a small change in another. In Machine Learning, when the loss depends on many model parameters, those partial derivatives are collected into a gradient.

The gradient points toward the greatest local increase in the loss, so gradient-based optimizers update the parameters in the opposite direction. The chain rule makes this possible even when the loss is produced by many nested transformations. Backpropagation is essentially an efficient application of the chain rule through a computational graph.

I would use gradient-based optimization when the model has continuous trainable parameters and a differentiable objective, such as regression, neural-network training, embedding fine-tuning or LLM adaptation. In a real project, I would also consider feature scale, learning rate, gradient stability, numerical precision and whether the training loss is actually aligned with the business or system-level metric.

---

## 13. GitHub file structure

Suggested folder:

```text
applied-ai-engineering-lab/
└── fundamentals/
    └── 05-calculus-for-ml/
        ├── README.md
        ├── notes.md
        ├── notebook.ipynb
        ├── example.py
        ├── from_scratch.py
        ├── interview_questions.md
        ├── references.md
        └── requirements.txt
```

### File responsibilities

```text
README.md
```

High-level explanation, setup instructions, key concepts and conclusions.

```text
notes.md
```

Detailed theoretical and mathematical notes:

* derivatives;
* partial derivatives;
* gradients;
* directional derivatives;
* Jacobians;
* Hessians;
* chain rule;
* gradient descent;
* backpropagation.

```text
notebook.ipynb
```

Interactive experiments:

* function and tangent visualization;
* gradient-descent trajectory;
* learning-rate comparison;
* numerical gradient checking;
* vanishing and exploding gradients.

```text
example.py
```

Linear regression trained with manually derived gradients.

```text
from_scratch.py
```

Manual forward and backward pass through a nonlinear neuron.

```text
interview_questions.md
```

Senior-level conceptual, mathematical and production questions.

```text
references.md
```

Suggested sources:

* *Deep Learning* — Goodfellow, Bengio and Courville;
* *Mathematics for Machine Learning* — Deisenroth, Faisal and Ong;
* *Dive into Deep Learning*;
* PyTorch automatic-differentiation documentation;
* CS231n notes on backpropagation and optimization;
* *The Matrix Calculus You Need for Deep Learning*.

```text
requirements.txt
```

```text
numpy
matplotlib
torch
jupyter
```

PyTorch can be omitted when running only the NumPy examples.

---

## 14. Suggested `README.md` content

# Calculus for Machine Learning

This module explores the calculus foundations behind gradient-based Machine Learning, including derivatives, partial derivatives, gradients, the chain rule and backpropagation.

## Objectives

The goal is to understand:

* how derivatives represent local sensitivity;
* how gradients extend derivatives to multivariate functions;
* why the negative gradient is used for loss minimization;
* how the chain rule propagates derivatives through nested operations;
* how backpropagation computes gradients efficiently;
* how analytical, numerical and automatic differentiation differ;
* why gradients may vanish, explode or become numerically unstable.

## Contents

* `notes.md`: theoretical and mathematical foundations;
* `example.py`: linear regression trained with manually derived gradients;
* `from_scratch.py`: manual forward and backward pass through a nonlinear neuron;
* `notebook.ipynb`: visual experiments and optimization analysis;
* `interview_questions.md`: senior-level interview preparation;
* `references.md`: recommended learning resources.

## Running the examples

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Run the gradient-descent example:

```bash
python example.py
```

Run the manual backpropagation example:

```bash
python from_scratch.py
```

## Key takeaways

* A derivative measures local sensitivity.
* A gradient contains the partial derivatives of a scalar-valued function.
* The gradient points toward the steepest local increase.
* Gradient descent moves in the opposite direction.
* The chain rule connects local derivatives across nested computations.
* Backpropagation is reverse-mode automatic differentiation over a computational graph.
* Correct optimization still depends on learning rate, data quality, numerical stability and objective design.

---

## 15. LinkedIn post idea

Em Machine Learning, treinar um modelo significa responder repetidamente a uma pergunta: como cada parâmetro contribuiu para o erro atual?

As derivadas medem essa sensibilidade. O gradiente reúne essa informação para todos os parâmetros, enquanto a regra da cadeia permite propagar o erro por várias transformações — exatamente o que acontece no backpropagation de uma rede neural.

O ponto mais importante é que o gradiente não “encontra a solução”. Ele oferece uma direção local. A qualidade do treinamento ainda depende da taxa de aprendizado, da escala dos dados, da estabilidade numérica e, principalmente, de uma função de perda alinhada ao problema real.

Documentei no GitHub a derivação dos gradientes de uma regressão linear, uma implementação manual de backpropagation e experimentos de verificação numérica.

---

## 16. 30–60 minute checklist

### 30-minute version

**0–5 minutes — Conceptual review**

* [ ] Define derivative, partial derivative and gradient.
* [ ] Explain why the gradient points uphill.
* [ ] Explain why optimization uses the negative gradient.

**5–15 minutes — Mathematical derivation**

* [ ] Derive (\partial L/\partial w) for one-dimensional linear regression.
* [ ] Derive (\partial L/\partial b).
* [ ] Explain each chain-rule factor.

**15–25 minutes — Implementation**

* [ ] Run `example.py`.
* [ ] Compare analytical and numerical gradients.
* [ ] Inspect the loss curve.

**25–30 minutes — Interview practice**

* [ ] Answer “Is backpropagation an optimizer?”
* [ ] Explain automatic versus numerical differentiation.
* [ ] Give the interview-ready explanation aloud.

---

### 60-minute version

**0–10 minutes — Foundations**

* [ ] Review derivatives and partial derivatives.
* [ ] Explain gradients geometrically.
* [ ] Review Jacobians and Hessians.
* [ ] Explain the directional derivative.

**10–25 minutes — Mathematics**

* [ ] Derive the linear-regression gradients.
* [ ] Explain the logistic-regression result (\partial L/\partial z=p-y).
* [ ] Derive the manual nonlinear-neuron backward pass.
* [ ] Connect the derivation to computational graphs.

**25–40 minutes — Code**

* [ ] Run `example.py`.
* [ ] Run `from_scratch.py`.
* [ ] Confirm the numerical gradient checks.
* [ ] Read the PyTorch autograd example.

**40–50 minutes — Experiment**

Choose one:

* [ ] compare learning rates;
* [ ] change feature scale;
* [ ] test different finite-difference values;
* [ ] replace `tanh` with sigmoid or ReLU;
* [ ] implement minibatch training.

**50–60 minutes — Portfolio and interviews**

* [ ] Create the GitHub folder.
* [ ] Add the README.
* [ ] Commit the executable examples.
* [ ] Answer five interview questions aloud.
* [ ] Explain how gradients apply to LLM fine-tuning and RAG components.
