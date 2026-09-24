# Regularization: Ridge, Lasso and ElasticNet

Day 20 asks how a linear model should trade training fit for smaller or sparse
coefficients. Ridge stabilizes weakly identified directions with an L2 penalty;
Lasso can remove coefficients with L1; ElasticNet combines both. These are
useful baselines for correlated tabular features and downstream regression
on engineered AI-system signals.

The study connects penalty geometry, feature units, the bias-variance trade-off,
and soft thresholding to executable code. It continues
[Day 19's regression implementation](../19-linear-regression-from-scratch/)
and applies [Day 17's validation boundaries](../17-validation-and-leakage/).

## What to read and run

| File | Purpose |
|---|---|
| [notes.md](notes.md) | Objective conventions, SVD shrinkage, KKT conditions, assumptions, measured experiments, and limitations |
| [example.py](example.py) | OLS/Ridge/Lasso/ElasticNet comparison with scaling fitted inside each CV fold |
| [from_scratch.py](from_scratch.py) | Educational proximal-gradient solver with an unpenalized intercept and convergence diagnostics |
| [tests/test_regularization.py](tests/test_regularization.py) | Analytic solutions, scikit-learn parity, invalid inputs, and fold-local scaling |
| [interview_questions.md](interview_questions.md) | Questions about sparsity, correlated predictors, validation, and deployment |
| [references.md](references.md) | Official API and leakage guidance |

Use the repository environment and dependencies declared in
[pyproject.toml](../../pyproject.toml). From the repository root:

```bash
python -m pip install -e ".[dev]"
python 01-classical-machine-learning/20-regularization/example.py
python 01-classical-machine-learning/20-regularization/from_scratch.py
python -m pytest -q 01-classical-machine-learning/20-regularization/tests
```

Both examples print to the terminal and require no network, external dataset,
credentials, or generated assets. All data are explicitly synthetic.

The comparison uses 500 rows, 30 differently scaled predictors, five direct
effects, and two correlated proxies. It reserves 125 test rows, tunes on the
375 development rows with five-fold CV, and selects the model family before
test evaluation. The output includes CV/train/test RMSE, exact zero counts,
coefficient norms, and coefficients for the correlated pairs.

## What the executed example shows

In the recorded run, development CV selected Lasso. It set 20 of 30 coefficients
to zero; its test RMSE was 4.2385 versus 4.2396 for OLS. Ridge substantially
reduced the coefficient norm but had test RMSE 4.2616. These are observations
from one synthetic split, not evidence that one estimator generally wins.
The complete configuration and interpretation candidates awaiting author
review are in [the experiment record](notes.md#executed-experiments).

## Key takeaways

- A penalty changes the estimation objective; it does not repair leakage,
  missing predictors, outliers, or deployment shift.
- L1 admits exact zeros through its subgradient at zero. A zero coefficient
  is conditional on the representation and penalty, not a causal finding.
- Ridge shrinks singular directions. Individual correlated coefficients
  need not decrease monotonically along a regularization path.
- Scale inside the CV pipeline. A pipeline ending in an internally
  cross-validating estimator does not automatically make earlier steps
  fold-local.
- Compare objective conventions: this solver's pure-L2 `alpha` matches
  scikit-learn `Ridge(alpha=n * alpha)` on the same training rows.
- The first-principles solver is a teaching implementation for small dense
  arrays. Check `converged`; production work should use established solvers.
