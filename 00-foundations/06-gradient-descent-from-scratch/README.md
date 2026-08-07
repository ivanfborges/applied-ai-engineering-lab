# Gradient Descent from Scratch

## Overview

Gradient descent is a first-order optimization algorithm that iteratively
updates model parameters to reduce a differentiable objective. This topic uses
linear regression with squared error because its convex loss surface makes the
optimization mechanics easy to inspect without hiding them behind a framework.

The code trains a linear model with batch gradient descent using NumPy, compares
the result with scikit-learn's ordinary least-squares implementation, and saves
plots showing the fitted line and the optimization trajectory.

All observations are generated synthetically with a fixed random seed. The
example is educational and does not report benchmark results.

## Concepts Covered

- Linear regression and residuals
- Mean squared error and the half-MSE objective
- Analytical and vectorized gradients
- Batch gradient descent
- Learning rate and convergence criteria
- Gradient norm and parameter histories
- Convexity, curvature, and conditioning
- Feature scaling
- Batch, stochastic, and mini-batch trade-offs
- Optimization versus generalization

## Why It Matters

The same parameter-update principle appears in logistic regression, matrix
factorization, neural networks, embedding models, rerankers, and model
fine-tuning. Understanding it makes learning-rate selection, feature scaling,
gradient diagnostics, and unstable-training investigations less
trial-and-error.

Gradient descent only solves the stated optimization problem. A low training
loss does not protect a system from leakage, overfitting, distribution shift,
biased data, or an objective that is misaligned with the product goal.

## Files

- `notes.md`: intuition, derivations, convergence conditions, assumptions,
  trade-offs, applications, limitations, and common mistakes.
- `from_scratch.py`: educational NumPy implementation of batch gradient
  descent for one or more regression features.
- `example.py`: deterministic synthetic-data experiment, scikit-learn
  comparison, and convergence visualizations.
- `visualize_gradient_descent.py`: complete visual learning suite covering
  optimization geometry, convergence, learning rates, scaling, batching, and
  outlier sensitivity.
- `interview_questions.md`: senior-level conceptual, mathematical,
  diagnostic, and production questions with answers.
- `references.md`: books, course notes, and official library documentation.

## How to Run

From the repository root, install the shared dependencies if needed:

```bash
python -m pip install -e ".[dev]"
```

Run the end-to-end example:

```bash
python 00-foundations/06-gradient-descent-from-scratch/example.py
```

To display the figures as well as save them:

```bash
python 00-foundations/06-gradient-descent-from-scratch/example.py --show
```

The script creates:

```text
00-foundations/06-gradient-descent-from-scratch/outputs/
├── convergence.png
└── regression_fit.png
```

Generated outputs are ignored by the repository-level `.gitignore`.

The first-principles module can also be run directly for a small smoke test:

```bash
python 00-foundations/06-gradient-descent-from-scratch/from_scratch.py
```

## Visual exploration

The visualization suite uses deterministic synthetic data to make optimization
behavior observable. It reuses the recorded histories from
`LinearRegressionGD` for the regression animation, convergence charts, loss
contours, three-dimensional surfaces, and single-step illustration. Additional
educational experiments compare learning rates, feature conditioning, batch
strategies, and MSE sensitivity to target outliers.

From the topic directory:

```bash
cd 00-foundations/06-gradient-descent-from-scratch
```

Generate the core optimization visuals:

```bash
python visualize_gradient_descent.py --mode core
```

Generate every visualization:

```bash
python visualize_gradient_descent.py --mode all
```

Generate and display the Matplotlib figures after saving them:

```bash
python visualize_gradient_descent.py --mode all --show
```

Without `--show`, the script uses a non-interactive backend, saves the files,
and exits without blocking. `--show` requires a working graphical Matplotlib
backend in the local Python environment.

Generated artifacts include:

- PNG charts for loss, gradient norm, parameters, loss-surface geometry,
  learning-rate behavior, feature scaling, batch strategies, outliers, and one
  gradient update;
- `regression_fitting.gif`, rendered with Pillow and no FFmpeg dependency;
- `loss_surface_interactive.html`, which can be opened directly in a browser
  without a Python server. It loads Plotly JavaScript from the CDN, so the
  interactive page needs internet access when opened.

All visualizations are educational, use synthetic data, and are not benchmark
results. Generated files are written to `outputs/`, which is ignored by the
repository-level `.gitignore`.

## Key Takeaways

- The gradient points toward the steepest local increase under the Euclidean
  norm; gradient descent subtracts it to reduce the objective.
- Learning rate controls both speed and stability. A step that is too large can
  oscillate or diverge even for a convex objective.
- Feature scale changes the curvature of the loss surface and can make a
  single global learning rate ineffective.
- For linear regression with squared error, the objective is convex, so a
  stationary minimum is global; rank deficiency can make it non-unique.
- Loss, gradient norm, and parameter histories reveal different aspects of
  convergence and should be interpreted together.
- Optimization success and model quality are distinct: validation design and
  production monitoring remain necessary.
