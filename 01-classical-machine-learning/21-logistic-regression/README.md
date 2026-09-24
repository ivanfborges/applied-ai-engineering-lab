# Logistic Regression: From Log-Odds to Classification

Day 21 connects a linear score to a probability estimate and then to a class
decision. Logistic regression models the positive class through

$$
\log\frac{p(x)}{1-p(x)} = w^\top x+b,
\qquad p(x)=\sigma(w^\top x+b).
$$

The sigmoid is the inverse logit. Training minimizes Bernoulli negative
log-likelihood, or binary cross-entropy, optionally with regularization.
A threshold converts probabilities into labels; changing that threshold
moves the decision boundary without fitting new coefficients.

This is a useful baseline for binary tabular classification, sparse text
features, and linear probes over frozen embeddings. Its coefficients describe
conditional odds ratios; they do not establish causal effects. A probabilistic
output alone does not establish calibration or production suitability.

## Read and run

| File | Purpose |
|---|---|
| [notes.md](notes.md) | Odds, likelihood, gradients, boundaries, assumptions, and recorded experiments |
| [example.py](example.py) | Train-only scaling, scikit-learn/NumPy comparison, original-unit coefficients, and fixed thresholds |
| [from_scratch.py](from_scratch.py) | Stable sigmoid and logit, logit-based loss, and educational batch gradient descent |
| [tests/test_logistic_regression.py](tests/test_logistic_regression.py) | Finite-difference gradients, analytic optimum, solver parity, input checks, and preprocessing boundaries |
| [interview_questions.md](interview_questions.md) | Questions connecting the mathematics to modeling decisions |
| [references.md](references.md) | Official objective, API, numerical, and threshold guidance |

Use the shared environment defined in [pyproject.toml](../../pyproject.toml).
From the repository root:

```bash
python -m pip install -e ".[dev]"
python 01-classical-machine-learning/21-logistic-regression/example.py
python 01-classical-machine-learning/21-logistic-regression/from_scratch.py
python -m pytest -q 01-classical-machine-learning/21-logistic-regression/tests
```

Both scripts print to the terminal. All data are generated synthetically in
code; there are no downloads, credentials, or output assets. The NumPy solver
supports small dense binary problems with optional L2 regularization and an
unpenalized intercept. It exposes convergence diagnostics and is a teaching
implementation, not a replacement for a library solver.

The practical example reserves 300 of 1,200 rows for testing and fits scaling
on the 900 training rows. Regularization and thresholds are fixed in advance.
The test table demonstrates decision changes; it does not select a threshold.
The standalone scratch script uses all rows only to check optimization.

## Visual lab

The [14-view visual lab](VISUAL_GUIDE.md) connects sigmoid, log-odds, loss,
probability fields, moving thresholds, gradient descent, regularization,
nonlinear features, and calibration. It includes three GIFs and two locally
interactive 3D views.

![Logistic regression inference and training summary](visuals/12_full_pipeline.png)

Run all views from the repository root:

```bash
python 01-classical-machine-learning/21-logistic-regression/logistic_regression_visual_lab.py
```

The [generator](logistic_regression_visual_lab.py) creates outputs under
visuals/. Four PNG/GIF previews are deliberately unignored; other outputs
remain regenerable. Both HTML files use the adjacent local plotly.min.js
bundle. See the guide for individual commands, interpretation, measured
experiments, and publication review.

## Evidence and takeaways

In the executed synthetic split, scikit-learn and the scratch solver both
achieved test log loss 0.475924; the train-prevalence baseline scored 0.685106.
Their largest probability difference was approximately 1.19e-08. These are
reproducibility observations from one controlled example. Full configurations,
results, limitations, and **interpretation candidates for author review** are
in [the experiment record](notes.md#executed-experiments).

- A feature coefficient adds to log-odds and multiplies odds; its probability
  effect depends on the starting probability.
- At threshold 0.5 the boundary is score zero. At threshold t it is logit(t).
- Cross-entropy uses confidence information that a thresholded accuracy loses.
- Stable loss computation should operate on logits.
- Scaling changes the meaning of penalized coefficients; fit it on training
  data and state the units when reporting odds ratios.
- Convexity does not guarantee a finite unregularized optimum under separation.
- Calibration, threshold selection, and deployment validity need their own
  evidence.

This study builds on [likelihood and MAP](../../00-foundations/14-maximum-likelihood-map/),
[cross-entropy](../../00-foundations/15-entropy-cross-entropy-kl-divergence/),
[validation boundaries](../17-validation-and-leakage/), and
[regularization](../20-regularization/). Days 22–23 in the
[roadmap](../../ROADMAP.md) develop classification metrics and calibration.