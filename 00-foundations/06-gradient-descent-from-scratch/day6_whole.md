# Day 6 — Gradient Descent from Scratch

## 1. Executive overview

Gradient Descent is an iterative optimization algorithm used to find parameter values that minimize an objective function.

In supervised machine learning, that objective is usually a loss function measuring how different the model predictions are from the expected outputs. Instead of calculating the optimal parameters directly, Gradient Descent repeatedly:

1. evaluates the current error;
2. calculates how the error changes with respect to each parameter;
3. adjusts the parameters in the direction that reduces the error.

For a simple linear regression model,

[
\hat{y} = wx + b
]

Gradient Descent estimates:

* (w): the slope or feature coefficient;
* (b): the intercept.

Although linear regression has a closed-form solution, implementing Gradient Descent for this problem is valuable because the same underlying mechanism appears in:

* logistic regression;
* neural network training;
* matrix factorization;
* embedding models;
* recommendation systems;
* ranking and reranking models;
* fine-tuning;
* deep learning;
* large language model training.

In production, engineers rarely implement a raw optimizer themselves. Libraries such as PyTorch, TensorFlow and JAX handle automatic differentiation and optimized parameter updates. However, senior engineers must still understand what the optimizer is doing, how learning rate affects convergence, why feature scaling matters and how to diagnose unstable training.

A key distinction is:

> Gradient Descent solves an optimization problem. It does not, by itself, guarantee that the resulting model generalizes well.

You may successfully minimize the training loss and still produce a poor model because of overfitting, leakage, distribution shift or an incorrectly chosen objective.

---

## 2. Core intuition

Imagine standing on a mountain in dense fog and trying to reach the lowest point.

You cannot see the entire landscape, but you can determine the local slope beneath your feet. The gradient tells you the direction of steepest increase. Therefore, to move downhill, you walk in the opposite direction.

In optimization terminology:

* your position represents the current model parameters;
* the altitude represents the loss;
* the slope represents the gradient;
* the step size represents the learning rate.

The update is:

[
\text{new position}
===================

## \text{current position}

\text{step size}
\times
\text{slope}
]

Or, for a model parameter (\theta):

[
\theta_{t+1}
============

## \theta_t

\eta \nabla J(\theta_t)
]

where:

* (\theta_t) is the parameter vector at iteration (t);
* (\theta_{t+1}) is the updated parameter vector;
* (J) is the objective or loss function;
* (\nabla J(\theta_t)) is the gradient evaluated at the current parameters;
* (\eta) is the learning rate.

The learning rate determines how aggressively the optimizer moves:

* too small: stable but slow convergence;
* too large: oscillation or divergence;
* appropriate: fast and stable convergence.

The analogy has an important limitation: real machine learning loss surfaces may contain saddle points, flat regions, noise and poorly conditioned directions. They are not always smooth bowls.

For linear regression with mean squared error, however, the objective is convex, which makes it an excellent environment for understanding the optimization mechanics.

---

## 3. Theoretical foundations

### 3.1 Model

For one input variable, linear regression assumes:

[
\hat{y}_i = wx_i + b
]

where:

* (x_i) is the input for observation (i);
* (y_i) is the true target;
* (\hat{y}_i) is the predicted target;
* (w) is the coefficient;
* (b) is the intercept.

The residual for observation (i) is:

[
e_i = \hat{y}_i - y_i
]

A positive residual means the prediction is above the true value. A negative residual means it is below the true value.

### 3.2 Objective function

For regression, a common objective is Mean Squared Error:

[
\text{MSE}
==========

\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
]

where (n) is the number of training observations.

For mathematical convenience, optimization derivations often use:

[
J(w,b)
======

\frac{1}{2n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
]

The factor (1/2) does not change the location of the minimum. It only cancels the factor (2) that appears when differentiating the squared term.

### 3.3 Gradient

The gradient contains one partial derivative for each trainable parameter:

[
\nabla J(w,b)
=============

\begin{bmatrix}
\frac{\partial J}{\partial w} \
\frac{\partial J}{\partial b}
\end{bmatrix}
]

Each component answers a local sensitivity question:

* how would the loss change if (w) increased slightly?
* how would the loss change if (b) increased slightly?

The gradient points toward the direction of steepest increase. Gradient Descent moves in the negative-gradient direction.

### 3.4 Iterative update

At iteration (t):

[
w_{t+1}
=======

## w_t

\eta
\frac{\partial J}{\partial w}
]

[
b_{t+1}
=======

## b_t

\eta
\frac{\partial J}{\partial b}
]

The algorithm stops when one of these conditions is reached:

* a fixed number of iterations;
* the loss improvement becomes negligible;
* the gradient norm becomes sufficiently small;
* a validation metric stops improving;
* a time or compute budget is exhausted.

### 3.5 Convexity

For linear regression with squared error, the objective is convex.

Convexity means that every local minimum is also a global minimum. When the feature matrix has full column rank, the solution is unique.

This does not mean every learning rate will converge. An excessively large step can still cause Gradient Descent to overshoot and diverge.

### 3.6 Batch variations

#### Batch Gradient Descent

Uses the entire training set to calculate each gradient.

Advantages:

* deterministic for fixed data and initialization;
* stable gradient estimates;
* straightforward convergence analysis.

Disadvantages:

* expensive for large datasets;
* one parameter update may require scanning the full dataset;
* less suitable for online learning.

#### Stochastic Gradient Descent

Uses one observation per update.

Advantages:

* inexpensive updates;
* suitable for streaming data;
* gradient noise may help escape some problematic regions in non-convex optimization.

Disadvantages:

* noisy trajectory;
* loss may fluctuate;
* sensitive to ordering and learning-rate schedules.

#### Mini-batch Gradient Descent

Uses a subset of observations per update.

It is the dominant strategy in deep learning because it balances:

* computational efficiency;
* vectorization;
* GPU utilization;
* gradient quality;
* memory constraints.

---

## 4. Mathematical, statistical or logical foundations

### 4.1 Deriving the gradient for the coefficient

Start with:

[
J(w,b)
======

\frac{1}{2n}
\sum_{i=1}^{n}
(wx_i+b-y_i)^2
]

We differentiate (J) with respect to (w).

Using the chain rule:

[
\frac{\partial J}{\partial w}
=============================

\frac{1}{2n}
\sum_{i=1}^{n}
2(wx_i+b-y_i)x_i
]

The (2) cancels the (1/2):

[
\frac{\partial J}{\partial w}
=============================

\frac{1}{n}
\sum_{i=1}^{n}
x_i(\hat{y}_i-y_i)
]

Interpretation:

* (\hat{y}_i-y_i) is the prediction error;
* multiplying by (x_i) measures how much the coefficient contributed to that error;
* averaging over all observations produces the loss sensitivity with respect to (w).

### 4.2 Deriving the gradient for the intercept

Differentiate the same objective with respect to (b):

[
\frac{\partial J}{\partial b}
=============================

\frac{1}{2n}
\sum_{i=1}^{n}
2(wx_i+b-y_i)
]

Therefore:

[
\frac{\partial J}{\partial b}
=============================

\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)
]

Because:

[
\frac{\partial(wx_i+b-y_i)}{\partial b} = 1
]

The intercept gradient is simply the average prediction error.

### 4.3 Parameter updates

The updates become:

[
w_{t+1}
=======

## w_t

\eta
\left[
\frac{1}{n}
\sum_{i=1}^{n}
x_i(\hat{y}_i-y_i)
\right]
]

[
b_{t+1}
=======

## b_t

\eta
\left[
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)
\right]
]

### 4.4 Matrix formulation

For multiple features, define:

* (X \in \mathbb{R}^{n \times p}): feature matrix;
* (n): number of observations;
* (p): number of model parameters, including the intercept if it is added to (X);
* (\theta \in \mathbb{R}^{p}): parameter vector;
* (y \in \mathbb{R}^{n}): target vector;
* (\hat{y}=X\theta): prediction vector.

The objective is:

[
J(\theta)
=========

\frac{1}{2n}
\lVert X\theta-y \rVert_2^2
]

Here, (\lVert \cdot \rVert_2) is the Euclidean norm.

The gradient is:

[
\nabla J(\theta)
================

\frac{1}{n}
X^\top(X\theta-y)
]

The update is:

[
\theta_{t+1}
============

## \theta_t

\eta
\frac{1}{n}
X^\top(X\theta_t-y)
]

This is the vectorized form implemented with NumPy.

### 4.5 Hessian and convergence

For this quadratic objective, the Hessian is:

[
H
=

# \nabla^2 J(\theta)

\frac{1}{n}X^\top X
]

The Hessian describes the curvature of the objective.

Let:

[
e_t = \theta_t-\theta^*
]

where (\theta^*) is the optimal parameter vector.

For linear regression, the parameter error evolves as:

[
e_{t+1}
=======

(I-\eta H)e_t
]

where (I) is the identity matrix.

For fixed-rate Gradient Descent to converge on this quadratic objective, a sufficient condition is:

[
0 < \eta < \frac{2}{\lambda_{\max}(H)}
]

where (\lambda_{\max}(H)) is the largest eigenvalue of the Hessian.

This gives a more precise explanation of why learning rates can be unstable. If the learning rate is too large relative to the curvature, parameter updates overshoot the minimum.

### 4.6 Condition number

The condition number of the Hessian is:

[
\kappa(H)
=========

\frac{\lambda_{\max}(H)}
{\lambda_{\min}(H)}
]

where (\lambda_{\min}(H)) is the smallest positive eigenvalue.

A large condition number means the loss surface is much steeper in some directions than in others. The contours become elongated rather than circular.

Consequences include:

* zigzagging updates;
* slow convergence;
* strong sensitivity to learning rate;
* different parameters converging at very different speeds.

Feature standardization frequently improves the conditioning of the optimization problem.

### 4.7 Gradient norm

A useful diagnostic is:

[
\lVert \nabla J(\theta) \rVert_2
]

A small gradient norm indicates that the optimizer is near a stationary point.

For convex linear regression, that stationary point is a global minimum. For non-convex neural networks, it could also be a saddle point or one of many local minima.

---

## 5. Practical applicability

### When Gradient Descent is useful

Gradient-based optimization is appropriate when:

* the objective is differentiable or approximately differentiable;
* the parameter space is too large for direct analytical optimization;
* data volume makes matrix inversion impractical;
* training is performed incrementally or in mini-batches;
* the model is non-linear and has no closed-form solution;
* automatic differentiation is available;
* GPU or distributed training is required.

Examples include:

* logistic regression on large datasets;
* feed-forward neural networks;
* convolutional and recurrent networks;
* Transformers;
* embedding and ranking models;
* recommendation systems;
* matrix factorization;
* fine-tuning task-specific models;
* differentiable calibration or scoring components.

### When it may not make sense

Gradient Descent may be unnecessary or inappropriate when:

* a small linear regression can be solved directly;
* the objective is non-differentiable and cannot be approximated effectively;
* the search space is discrete;
* derivative-free optimization is more suitable;
* the dataset is small and a second-order solver converges faster;
* model training time is insignificant compared with implementation complexity.

For a small ordinary least squares problem, the analytical solution is:

[
\theta^*
========

(X^\top X)^{-1}X^\top y
]

In practice, numerical libraries generally use QR decomposition or singular value decomposition instead of explicitly calculating the inverse.

### Production trade-offs

#### Compute versus convergence

Larger batches produce more accurate gradients but require more memory and computation per update.

#### Training speed versus stability

A larger learning rate can reduce training time but increases the risk of divergence.

#### Optimization versus generalization

The optimizer that reaches the lowest training loss fastest is not automatically the optimizer that produces the best validation performance.

#### Reproducibility versus throughput

Deterministic execution may require fixed seeds and deterministic kernels, sometimes reducing hardware performance.

#### Exact convergence versus budget

Production training usually stops based on validation performance, cost or time—not because the mathematical optimum has been reached exactly.

### Connection to Applied AI systems

In your context, Gradient Descent appears indirectly when:

* fine-tuning a model in Vertex AI;
* training an embedding model;
* fitting a reranker;
* calibrating a classification component;
* training a document classifier;
* optimizing a neural OCR component;
* adapting a model with parameter-efficient fine-tuning.

For systems based entirely on prompting or RAG inference, Gradient Descent is usually not part of the request-time pipeline. It was used during the training of the foundation model and may appear again during fine-tuning, but retrieval and prompt construction themselves are generally not gradient-based.

---

## 6. Common pitfalls and mistakes

### Confusing the loss with the optimizer

Mean squared error defines what the model should minimize. Gradient Descent defines how parameters are updated to minimize it.

Changing the optimizer does not necessarily change the objective.

### Using the wrong update sign

Gradient Descent subtracts the gradient:

[
\theta \leftarrow \theta-\eta\nabla J(\theta)
]

Adding the gradient performs gradient ascent and increases the objective.

### Choosing a learning rate without checking scale

A learning rate that works for standardized features may diverge when one feature ranges from (0) to (1) and another ranges from (0) to (10^6).

### Ignoring feature scaling

Poorly scaled features can create an ill-conditioned objective and cause zigzagging or extremely slow convergence.

### Implementing the gradient incorrectly

Common errors include:

* forgetting to average over (n);
* omitting the feature multiplier in (\partial J/\partial w);
* mixing residual definitions;
* using the updated coefficient when computing the intercept update;
* relying on accidental NumPy broadcasting;
* producing column vectors and one-dimensional arrays inconsistently.

Both gradients should be computed from the same parameter state before applying either update.

### Monitoring only the final loss

A final number does not reveal whether training:

* diverged and recovered;
* oscillated;
* plateaued;
* converged too slowly;
* produced exploding gradients.

Track at least:

* training loss;
* validation loss;
* gradient norm;
* learning rate;
* iteration or epoch;
* optionally parameter values.

### Evaluating on the training data

A low training loss measures optimization success, not generalization. Keep a validation or test split.

### Data leakage

Examples include:

* fitting a scaler using the complete dataset before splitting;
* using future observations to construct current features;
* including target-derived information in the feature matrix;
* selecting hyperparameters based on the test set.

Gradient Descent will efficiently optimize a leaked problem. The optimizer cannot detect that the experimental design is invalid.

### Optimizing the wrong metric

Training with MSE may not align with the real business objective.

For example, a production system may care more about:

* absolute error;
* asymmetric costs;
* tail errors;
* ranking quality;
* calibration;
* latency-adjusted quality;
* false-negative rate.

### Treating convergence as proof of model correctness

The optimizer may converge even when:

* the model is misspecified;
* important variables are missing;
* outliers dominate the squared loss;
* the training distribution differs from production;
* the target itself is noisy or biased.

### Using Gradient Descent when a simpler solver is better

For small linear regression problems, a closed-form or library solver is usually more convenient and numerically robust.

The from-scratch implementation is educational, not a production recommendation.

### No shuffle in SGD or mini-batch training

If data is sorted by class, date or target, fixed sequential batches can produce biased update patterns.

### Excessive iterations without early stopping

More iterations increase compute cost and may cause overfitting in flexible models. Convergence criteria should be explicit.

### Forgetting numerical precision

Using low-precision data types can cause unstable optimization, particularly with:

* very small gradients;
* large feature values;
* deep networks;
* mixed-precision training.

---

## 7. Important comparisons

### Gradient Descent versus analytical linear regression

| Aspect                | Gradient Descent                      | Analytical solver                           |
| --------------------- | ------------------------------------- | ------------------------------------------- |
| Strategy              | Iterative optimization                | Direct numerical solution                   |
| Large datasets        | More scalable                         | Matrix operations may become expensive      |
| Exactness             | Approximate, convergence-dependent    | Solves the least-squares system numerically |
| Hyperparameters       | Learning rate, iterations, batch size | Few optimizer hyperparameters               |
| Online learning       | Possible                              | Generally not natural                       |
| General model support | Broad                                 | Limited to specific formulations            |
| Educational value     | Exposes optimization mechanics        | Exposes statistical solution                |

For large (p), explicitly forming and solving systems involving (X^\top X) can become expensive. Iterative methods may be preferable.

### Batch GD versus SGD versus mini-batch GD

| Method        | Gradient source | Main advantage            | Main limitation            |
| ------------- | --------------- | ------------------------- | -------------------------- |
| Batch GD      | Full dataset    | Stable updates            | Expensive updates          |
| SGD           | One observation | Cheap and online          | Highly noisy               |
| Mini-batch GD | Small subset    | Efficient on accelerators | Batch size requires tuning |

### Gradient Descent versus Newton's method

Newton's method uses both gradient and curvature:

[
\theta_{t+1}
============

## \theta_t

H^{-1}\nabla J(\theta_t)
]

It may converge in fewer iterations, but calculating or approximating the Hessian can be expensive.

Gradient Descent uses only first-order information and scales more naturally to millions or billions of parameters.

### Gradient Descent versus momentum

Momentum maintains a velocity term:

[
v_t
===

\beta v_{t-1}
+
\nabla J(\theta_t)
]

[
\theta_{t+1}
============

\theta_t-\eta v_t
]

where (\beta) controls how much previous gradients influence the current update.

Momentum can reduce oscillation and accelerate movement through consistent directions.

### Gradient Descent versus Adam

Adam maintains moving estimates of the first and second moments of gradients. It adapts the effective learning rate for each parameter.

Advantages:

* often converges quickly during early training;
* handles differently scaled gradients;
* widely used in Transformer training.

Limitations:

* more optimizer state;
* more hyperparameters;
* rapid optimization does not guarantee superior generalization;
* weight decay must be implemented carefully, commonly through AdamW.

### Gradient Descent versus coordinate descent

Gradient Descent updates all parameters together. Coordinate descent updates one parameter or block of parameters at a time.

Coordinate descent can be particularly effective for some sparse or regularized problems, such as Lasso regression.

---

## 8. Practical Python example

This example:

* generates synthetic regression data;
* creates training and test sets;
* trains the from-scratch model;
* compares it with scikit-learn;
* plots the fitted line;
* plots loss convergence;
* plots coefficient convergence.

Save it as `example.py`.

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from from_scratch import LinearRegressionGD


def create_dataset(
    n_samples: int = 250,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple synthetic linear regression dataset."""
    rng = np.random.default_rng(random_state)

    x = rng.normal(loc=0.0, scale=1.5, size=(n_samples, 1))
    noise = rng.normal(loc=0.0, scale=1.0, size=n_samples)

    true_weight = 4.2
    true_intercept = -1.5

    y = true_weight * x[:, 0] + true_intercept + noise
    return x, y


def main() -> None:
    x, y = create_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
    )

    gd_model = LinearRegressionGD(
        learning_rate=0.05,
        max_iterations=2_000,
        tolerance=1e-12,
    )
    gd_model.fit(x_train, y_train)

    sklearn_model = LinearRegression()
    sklearn_model.fit(x_train, y_train)

    gd_predictions = gd_model.predict(x_test)
    sklearn_predictions = sklearn_model.predict(x_test)

    gd_mse = mean_squared_error(y_test, gd_predictions)
    sklearn_mse = mean_squared_error(y_test, sklearn_predictions)

    print("Gradient Descent")
    print(f"  Coefficient: {gd_model.coefficients_[0]:.4f}")
    print(f"  Intercept:   {gd_model.intercept_:.4f}")
    print(f"  Test MSE:    {gd_mse:.4f}")
    print(f"  Iterations:  {gd_model.n_iterations_}")

    print("\nscikit-learn")
    print(f"  Coefficient: {sklearn_model.coef_[0]:.4f}")
    print(f"  Intercept:   {sklearn_model.intercept_:.4f}")
    print(f"  Test MSE:    {sklearn_mse:.4f}")

    x_line = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    y_line = gd_model.predict(x_line)

    plt.figure(figsize=(8, 5))
    plt.scatter(x_train[:, 0], y_train, alpha=0.55, label="Training data")
    plt.plot(x_line[:, 0], y_line, linewidth=2, label="GD prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Trained with Gradient Descent")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(gd_model.loss_history_)
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.title("Gradient Descent Convergence")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

    coefficient_history = np.asarray(gd_model.coefficient_history_)

    plt.figure(figsize=(8, 5))
    plt.plot(
        coefficient_history[:, 0],
        label="Estimated coefficient",
    )
    plt.axhline(
        y=4.2,
        linestyle="--",
        label="Data-generating coefficient",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Coefficient value")
    plt.title("Coefficient Convergence")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

Install the dependencies with:

```bash
pip install numpy matplotlib scikit-learn
```

Run:

```bash
python example.py
```

The Gradient Descent parameters and scikit-learn parameters should be similar. They will not necessarily equal the exact data-generating values because the dataset contains random noise and only a finite sample.

---

## 9. From-scratch implementation when useful

Save this as `from_scratch.py`.

The implementation supports multiple features, although the example uses one.

```python
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class LinearRegressionGD:
    """Linear regression optimized with batch gradient descent.

    This implementation is educational and is not intended to replace
    production-grade implementations from scikit-learn or similar libraries.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1_000,
        tolerance: float = 1e-10,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than zero.")

        if tolerance < 0:
            raise ValueError("tolerance cannot be negative.")

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.coefficients_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None
        self.n_iterations_: int = 0

        self.loss_history_: list[float] = []
        self.gradient_norm_history_: list[float] = []
        self.coefficient_history_: list[NDArray[np.float64]] = []
        self.intercept_history_: list[float] = []

    @staticmethod
    def _prepare_features(x: ArrayLike) -> NDArray[np.float64]:
        features = np.asarray(x, dtype=np.float64)

        if features.ndim == 1:
            features = features.reshape(-1, 1)

        if features.ndim != 2:
            raise ValueError("x must be a one- or two-dimensional array.")

        if not np.all(np.isfinite(features)):
            raise ValueError("x contains NaN or infinite values.")

        return features

    @staticmethod
    def _prepare_target(
        y: ArrayLike,
        expected_rows: int,
    ) -> NDArray[np.float64]:
        target = np.asarray(y, dtype=np.float64).reshape(-1)

        if target.shape[0] != expected_rows:
            raise ValueError("x and y must have the same number of rows.")

        if not np.all(np.isfinite(target)):
            raise ValueError("y contains NaN or infinite values.")

        return target

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> "LinearRegressionGD":
        features = self._prepare_features(x)
        target = self._prepare_target(y, features.shape[0])

        n_samples, n_features = features.shape

        coefficients = np.zeros(n_features, dtype=np.float64)
        intercept = 0.0

        self.loss_history_ = []
        self.gradient_norm_history_ = []
        self.coefficient_history_ = []
        self.intercept_history_ = []

        previous_loss = np.inf

        for iteration in range(self.max_iterations):
            predictions = features @ coefficients + intercept
            residuals = predictions - target

            # Objective:
            # J = (1 / 2n) * sum((prediction - target)^2)
            loss = float(
                np.dot(residuals, residuals) / (2.0 * n_samples)
            )

            # Gradients:
            # dJ/dw = (1/n) * X^T (prediction - target)
            # dJ/db = (1/n) * sum(prediction - target)
            coefficient_gradient = (
                features.T @ residuals
            ) / n_samples

            intercept_gradient = float(np.mean(residuals))

            gradient_norm = float(
                np.sqrt(
                    np.dot(
                        coefficient_gradient,
                        coefficient_gradient,
                    )
                    + intercept_gradient**2
                )
            )

            self.loss_history_.append(loss)
            self.gradient_norm_history_.append(gradient_norm)
            self.coefficient_history_.append(coefficients.copy())
            self.intercept_history_.append(intercept)

            # Compute both gradients before updating either parameter.
            coefficients -= (
                self.learning_rate * coefficient_gradient
            )
            intercept -= (
                self.learning_rate * intercept_gradient
            )

            self.n_iterations_ = iteration + 1

            loss_improvement = abs(previous_loss - loss)

            if loss_improvement < self.tolerance:
                break

            previous_loss = loss

        self.coefficients_ = coefficients
        self.intercept_ = intercept

        return self

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        if self.coefficients_ is None or self.intercept_ is None:
            raise RuntimeError("The model must be fitted before prediction.")

        features = self._prepare_features(x)

        if features.shape[1] != self.coefficients_.shape[0]:
            raise ValueError(
                "The number of prediction features does not match "
                "the fitted model."
            )

        return features @ self.coefficients_ + self.intercept_
```

### Important implementation decisions

The model initializes all parameters to zero. This is safe for linear regression because the objective is convex. In neural networks, initializing all weights identically would create a symmetry problem.

The gradient calculations are vectorized:

```python
coefficient_gradient = features.T @ residuals / n_samples
```

This is preferable to explicit Python loops because NumPy delegates the matrix operations to optimized numerical libraries.

The implementation records:

* loss;
* gradient norm;
* coefficient trajectory;
* intercept trajectory;
* number of iterations.

These diagnostics make optimization behavior visible rather than treating training as a black box.

### Optional numerical gradient check

A useful experiment is to validate the analytical gradient using finite differences.

For parameter (\theta_j):

[
\frac{\partial J}{\partial \theta_j}
\approx
\frac{
J(\theta_j+\epsilon)
--------------------

J(\theta_j-\epsilon)
}{
2\epsilon
}
]

where (\epsilon) is a small perturbation.

The analytical and numerical gradients should be close. Gradient checking is expensive and not normally used during production training, but it is valuable when implementing custom differentiable operations.

---

## 10. Suggested experiments

### Experiment 1 — Vary the learning rate

Test values such as:

```python
learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.5, 1.0]
```

Observe:

* number of iterations;
* oscillation;
* divergence;
* final loss;
* gradient norm.

Expected insight: learning rate controls both stability and convergence speed.

### Experiment 2 — Create badly scaled features

Add a second feature with a much larger scale:

```python
x_large = x * 100_000
```

Train once without standardization and once after:

[
x_{\text{scaled}}
=================

\frac{x-\mu}{\sigma}
]

where:

* (\mu) is the training-feature mean;
* (\sigma) is the training-feature standard deviation.

Expected insight: scaling changes the geometry of the objective and can dramatically improve convergence.

Fit the scaler using only the training data.

### Experiment 3 — Compare batch, stochastic and mini-batch training

Implement:

* batch size (n);
* batch size (1);
* batch size (16) or (32).

Compare:

* loss smoothness;
* number of parameter updates;
* wall-clock time;
* final validation performance.

Expected insight: a noisier gradient can still produce efficient optimization.

### Experiment 4 — Add outliers

Replace a few target values with extreme values:

```python
y_with_outliers = y.copy()
y_with_outliers[:5] += 30
```

Observe how the regression line changes.

Expected insight: squared error assigns disproportionately high influence to large residuals. Consider comparing it with mean absolute error or Huber loss.

### Experiment 5 — Plot the loss surface

For simple (w) and (b), calculate (J(w,b)) over a grid and produce:

* a contour plot;
* a three-dimensional surface;
* the Gradient Descent trajectory.

Expected insight: parameter updates follow the negative gradient and the geometry changes with feature scaling.

---

## 11. Senior interview questions

### 1. What exactly does the gradient represent?

The gradient is a vector of partial derivatives of the objective with respect to each trainable parameter. It represents the local direction of steepest increase in the objective. Gradient Descent moves in the opposite direction to reduce the objective.

Each component also measures local parameter sensitivity: how much the loss would change under a small change to that parameter.

---

### 2. Why does Gradient Descent find the global minimum for linear regression?

Linear regression with squared error has a convex quadratic objective. Therefore, any stationary minimum is global.

When the design matrix has full column rank, the Hessian is positive definite and the minimum is unique. If features are linearly dependent, multiple parameter vectors may produce the same minimum loss.

---

### 3. Why does feature scaling affect Gradient Descent?

Features with different scales produce different curvature across parameter directions. This can make the Hessian poorly conditioned and the loss contours elongated.

Gradient Descent then oscillates across steep directions while progressing slowly through flatter directions. Standardization often produces a better-conditioned optimization problem and permits a more effective learning rate.

---

### 4. How would you diagnose a diverging training process?

I would inspect:

* loss over iterations;
* gradient norms;
* learning rate;
* parameter magnitudes;
* activation or input ranges;
* NaN and infinity occurrences;
* batch-level variability.

Typical interventions include reducing the learning rate, scaling inputs, applying gradient clipping, improving initialization and verifying the gradient implementation.

For deep models, I would also inspect mixed-precision behavior and numerical overflow.

---

### 5. Why not always use the analytical solution for linear regression?

The analytical formulation is useful for relatively small ordinary least squares problems, but its numerical solution involves matrix factorization whose cost grows quickly with the number of features.

Gradient-based and iterative solvers are more suitable when:

* the dataset is very large;
* the feature space is high-dimensional;
* data arrives incrementally;
* the loss includes components without a simple closed form;
* the model is not linear;
* distributed or mini-batch training is required.

Also, production numerical libraries usually avoid explicitly calculating ((X^\top X)^{-1}).

---

### 6. What is the difference between an epoch and an iteration?

An iteration is one parameter update.

An epoch is one complete pass over the training dataset.

For batch Gradient Descent, one epoch usually corresponds to one iteration. For mini-batch training with ten batches, one epoch contains ten iterations.

---

### 7. What happens if the learning rate is too large?

The optimizer may overshoot the minimum, oscillate or diverge.

For a quadratic objective, stability depends on the relationship between the learning rate and the largest eigenvalue of the Hessian. A typical convergence condition is:

[
0<\eta<\frac{2}{\lambda_{\max}(H)}
]

---

### 8. Does reaching a zero or very small gradient guarantee a good model?

No.

It only indicates that optimization has reached a stationary region for the chosen training objective.

The model may still have:

* poor validation performance;
* data leakage;
* bias;
* distribution-shift sensitivity;
* an inappropriate objective;
* weak calibration;
* unacceptable latency or cost.

Optimization quality and model quality are related but distinct concerns.

---

### 9. Why are mini-batches standard in deep learning?

Mini-batches provide a practical balance between noisy stochastic updates and expensive full-dataset gradients.

They enable:

* efficient matrix operations;
* GPU utilization;
* manageable memory consumption;
* frequent parameter updates;
* scalable distributed training.

The batch size also changes optimization dynamics and can affect generalization.

---

### 10. How would you monitor Gradient Descent in a production training pipeline?

I would log:

* training and validation loss;
* task-specific validation metrics;
* learning rate;
* gradient norm;
* epoch and iteration;
* throughput;
* GPU or accelerator utilization;
* checkpoint identifiers;
* data and code versions;
* random seeds;
* optimizer configuration;
* early-stopping state.

In Vertex AI, these metrics could be sent to experiment tracking or an external platform such as MLflow. Model artifacts should be associated with dataset, code and hyperparameter versions.

---

### 11. How do you choose a learning rate?

I would begin with established defaults for the optimizer and model family, then validate empirically.

Useful approaches include:

* logarithmic search;
* learning-rate range tests;
* warm-up;
* decay schedules;
* validation-based selection;
* adaptive optimizers.

The correct value depends on feature scaling, batch size, curvature, optimizer and model architecture.

---

### 12. What is the difference between minimizing training loss and minimizing expected risk?

Training loss is empirical risk:

[
\hat{R}(f)
==========

\frac{1}{n}
\sum_{i=1}^{n}
L(f(x_i),y_i)
]

Expected risk is performance over the unknown data-generating distribution:

[
R(f)
====

\mathbb{E}_{(X,Y)}
[
L(f(X),Y)
]
]

Gradient Descent directly optimizes empirical risk. Validation methodology, regularization and representative data are needed to improve expected performance.

---

### 13. What role does randomness play in SGD?

Random sampling introduces noise into the gradient estimate.

This noise:

* reduces the cost of each update;
* creates fluctuations around the optimum;
* can help exploration in non-convex objectives;
* makes reproducibility more difficult;
* usually requires learning-rate decay for stable convergence.

The mini-batch gradient is an estimate of the full gradient.

---

### 14. What would you do if training loss decreases but validation loss increases?

That is a classic overfitting signal.

Possible responses include:

* early stopping;
* stronger regularization;
* more representative data;
* data augmentation;
* reducing model capacity;
* validating the split strategy;
* investigating leakage or distribution mismatch.

Changing the optimizer alone may not solve the underlying generalization problem.

---

### 15. How would Gradient Descent be used in a distributed system?

Workers calculate gradients over different mini-batches. The gradients are then aggregated before updating the parameters.

Two common architectures are:

* synchronous updates, where workers wait for each other;
* asynchronous updates, where workers update parameters independently.

Synchronous training is easier to reason about but can suffer from slow workers. Asynchronous training improves utilization but may use stale gradients.

Communication cost, gradient aggregation and fault tolerance become major system-design concerns.

---

## 12. Interview-ready explanation

> Gradient Descent is a first-order iterative optimization algorithm used to minimize a differentiable objective function. At each step, it calculates the gradient of the loss with respect to the model parameters and moves those parameters in the opposite direction, scaled by a learning rate.
>
> For linear regression with mean squared error, the objective is convex, so Gradient Descent can converge to the global minimum when the learning rate is appropriate. Feature scaling is important because poorly scaled features make the objective ill-conditioned and slow down convergence.
>
> In a real project, I would use a gradient-based optimizer when the model does not have a practical analytical solution, the dataset is too large for direct solvers or the model is trained in mini-batches, as with logistic regression, neural networks, embedding models and fine-tuning. I would monitor both optimization metrics, such as loss and gradient norm, and generalization metrics on a validation set, because minimizing training loss does not guarantee production performance.

---

## 13. GitHub file structure

```text
day-06-gradient-descent/
├── README.md
├── notes.md
├── notebook.ipynb
├── from_scratch.py
├── example.py
├── interview_questions.md
├── references.md
├── requirements.txt
└── outputs/
    ├── regression_fit.png
    ├── loss_convergence.png
    ├── parameter_convergence.png
    └── loss_surface.png
```

Suggested responsibilities:

* `README.md`: objective, usage and main conclusions;
* `notes.md`: theory, derivations and production trade-offs;
* `notebook.ipynb`: interactive experiments and visualizations;
* `from_scratch.py`: reusable NumPy implementation;
* `example.py`: executable end-to-end demonstration;
* `interview_questions.md`: senior questions and concise answers;
* `references.md`: books, papers and documentation;
* `requirements.txt`: minimal dependencies;
* `outputs/`: generated visual artifacts.

Suggested `requirements.txt`:

```text
numpy
matplotlib
scikit-learn
jupyter
```

---

## 14. Suggested README.md content

# Gradient Descent from Scratch

## Objective

This project implements batch gradient descent for linear regression using NumPy. The goal is to explore the mathematical mechanics of gradient-based optimization and visualize how model parameters and training loss evolve during convergence.

## Concepts covered

* Linear regression
* Mean squared error
* Analytical gradients
* Batch gradient descent
* Learning rate
* Convergence criteria
* Gradient norm
* Feature scaling
* Convex optimization
* Comparison with scikit-learn

## Project structure

```text
.
├── README.md
├── notes.md
├── notebook.ipynb
├── from_scratch.py
├── example.py
├── interview_questions.md
├── references.md
├── requirements.txt
└── outputs/
```

## Installation

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the example

```bash
python example.py
```

The script:

1. generates a synthetic regression dataset;
2. trains a linear model with a NumPy implementation of gradient descent;
3. compares the result with scikit-learn;
4. visualizes the fitted regression line;
5. visualizes loss and parameter convergence.

## Key takeaways

* Gradient descent updates model parameters in the opposite direction of the loss gradient.
* The learning rate controls the speed and stability of convergence.
* Feature scaling can significantly improve the conditioning of the optimization problem.
* Convergence on the training objective does not guarantee good validation or production performance.
* Batch, stochastic and mini-batch gradient descent provide different trade-offs between computational cost and gradient variance.

## Scope

The implementation is intentionally educational. Production systems should rely on numerically tested libraries and include validation, experiment tracking, reproducibility controls and monitoring.

---

## 15. LinkedIn post idea

Nos últimos dias, revisitei um dos mecanismos mais fundamentais de Machine Learning: o Gradient Descent.

A ideia parece simples: calcular em qual direção o erro aumenta e atualizar os parâmetros na direção contrária.

Mas, quando implementamos o algoritmo do zero, alguns pontos ficam muito mais claros:

* por que uma taxa de aprendizado alta pode fazer o treinamento divergir;
* por que features em escalas muito diferentes prejudicam a convergência;
* como acompanhar não apenas a loss, mas também os gradientes e os parâmetros;
* e por que minimizar o erro de treino não significa necessariamente construir um bom modelo.

Implementei uma regressão linear usando apenas NumPy e comparei o resultado com o scikit-learn. Também visualizei a evolução da loss e dos coeficientes durante o treinamento.

É um exercício simples, mas ajuda a consolidar conceitos que aparecem em praticamente todo treinamento moderno, de regressões a redes neurais e Transformers.

A implementação e as anotações técnicas estão documentadas no meu repositório de estudos no GitHub.

O post pode ser publicado em conjunto com o gráfico da loss ou com uma animação do deslocamento dos parâmetros sobre a superfície de erro.

---

## 16. 30–60 minute checklist

### 0–10 minutes — Understand the mechanics

* Review the linear model (\hat{y}=wx+b).
* Review the MSE objective.
* Derive (\partial J/\partial w).
* Derive (\partial J/\partial b).
* Explain why the update subtracts the gradient.

### 10–25 minutes — Implement from scratch

* Create `from_scratch.py`.
* Implement prediction.
* Implement the loss.
* Implement vectorized gradients.
* Implement the parameter update.
* Record loss and gradient history.

### 25–40 minutes — Run and visualize

* Generate synthetic data.
* Train the implementation.
* Plot the regression line.
* Plot the loss curve.
* Compare parameters with scikit-learn.

### 40–50 minutes — Experiment

Run at least two variations:

* use a very small learning rate;
* use an excessively large learning rate;
* create badly scaled features;
* add outliers;
* compare batch and stochastic updates.

### 50–60 minutes — Consolidate for interviews

You should be able to explain:

* what the gradient represents;
* why feature scaling affects convergence;
* why linear regression has a global minimum;
* why a converged training loss does not guarantee generalization;
* when Gradient Descent is preferable to a direct solver.

Final repository actions:

```bash
git status
git add day-06-gradient-descent
git commit -m "Add Day 6 gradient descent from scratch"
git push
```
