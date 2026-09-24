# Linear Regression from Scratch

Day 19 turns the [OLS theory from Day 18](../18-linear-regression-theory/)
into two transparent numerical implementations: a direct least-squares fit
and full-batch gradient descent. Both minimize the same objective; the example
compares their parameters and held-out predictions with scikit-learn.

This is useful for debugging optimization, choosing a regression baseline,
and understanding why feature units affect iterative learning. The direct
implementation delegates matrix decomposition to NumPy; it does not implement
SVD itself. These are educational routines, not production library replacements.

## What to inspect

- [from_scratch.py](from_scratch.py): input validation, intercept augmentation,
  direct OLS, explicit half-MSE gradient, and convergence diagnostics.
- [example.py](example.py): deterministic synthetic comparison, conversion of
  standardized coefficients back to original units, and a scaling experiment.
- [notes.md](notes.md): derivation, solver trade-offs, executed experiment
  records, limitations, interview prompts, and official references.
- [tests/test_linear_regression.py](tests/test_linear_regression.py): analytic
  fixtures, finite-difference gradients, sklearn parity, singular designs,
  convergence, and input/error contracts.

## Run from the repository root

Use the shared environment and dependencies in [pyproject.toml](../../pyproject.toml):

```bash
python -m pip install -e ".[dev]"
python 01-classical-machine-learning/19-linear-regression-from-scratch/example.py
python -m pytest -q 01-classical-machine-learning/19-linear-regression-from-scratch/tests
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
```

On Windows, `.venv/Scripts/python.exe` can replace `python`. The example uses
only synthetic data generated in memory; no dataset download is needed.
It prints metrics and optimizer diagnostics without writing output artifacts.
`from_scratch.py` is the reusable numerical module, not a separate CLI.

## Reading the comparison

The fixed example uses 500 observations, three independent Gaussian features,
noise standard deviation 0.8, and a 400/100 train/test split. Both random seeds
are 42. Scaling is fitted only on training rows. There is no hyperparameter
selection on the test set.

In the recorded run, all three methods had test MSE approximately 0.572689.
GD converged in 200 updates, with maximum held-out prediction difference from
scikit-learn of 4.963e-08. A separate training-only experiment changed one
feature's units by a factor of 10,000 and compared convergence before and after
standardization. See the complete configurations and review caveats in
[the experiment records](notes.md#executed-experiments).

## Key takeaways

- OLS and converged GD solve the same least-squares problem; scaling the loss
  changes the gradient and suitable learning rate, but not its minimizer.
- Solve the least-squares system directly rather than explicitly inverting
  the Gram matrix. Rank deficiency can make parameters nonunique.
- Compare coefficients in the same units and predictions within stated
  tolerances. Numerical agreement does not establish statistical validity.
- A finite iteration budget is not evidence of convergence. Inspect the
  returned `converged` flag and loss history.

The API accepts finite dense `X` of shape `(n, p)` and single-target `y` of
shape `(n,)`, always fits an intercept, and leaves preprocessing to the caller.
It intentionally omits sparse inputs, multiple targets, sample weights,
regularization, inference, and the sklearn estimator protocol.
