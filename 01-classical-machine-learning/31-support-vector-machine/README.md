# Day 31 — Support Vector Machines

How does a classifier choose among many separating boundaries? A linear SVM
balances a wide geometric margin against violations of that margin. With signed
labels, its score is `f(x) = wᵀx + b`; hinge loss is `max(0, 1 - y f(x))`.
Correct predictions can still incur loss when they lie inside the margin.

This study connects maximum margin, support vectors, soft-margin optimization,
the dual formulation, and linear, polynomial, and RBF kernels. Kernels replace
inner products with a similarity function, enabling nonlinear boundaries
without explicitly constructing the transformed features.

An SVM is a useful candidate for supervised classification over text features,
scientific measurements, or fixed embeddings. A document router over embeddings
is one possible application; representative labels, class-specific evaluation,
and measured inference costs would be needed to validate that application.
This folder implements synthetic binary classification only.

## Files and scope

| File | What it contributes |
|---|---|
| [notes.md](notes.md) | Margin derivation, primal/dual objectives, scaling conventions, trade-offs, and executed experiment records |
| [example.py](example.py) | Joint RBF `C`/`gamma` search with scaling inside cross-validation folds and a reserved test set |
| [from_scratch.py](from_scratch.py) | NumPy mean-hinge objective, subgradients, and an educational linear optimizer |
| [tests/test_svm.py](tests/test_svm.py) | Numerical derivatives, kink behavior, analytic optima, library comparison, input errors, fold-local scaling, and dual-score reconstruction |
| [README_VISUALS.md](README_VISUALS.md) | Visual geometry, recorded experiments, limitations, and local regeneration |
| [visualize_svm.py](visualize_svm.py) | Static plots, parameter GIFs and self-contained 3D explorers |
| [tests/test_visualize_svm.py](tests/test_visualize_svm.py) | Numerical visual contracts and temporary-file rendering |
| [interview_questions.md](interview_questions.md) | Focused mathematical and operational questions with answers |
| [references.md](references.md) | Original research, official APIs, and synthetic data generators |

The scratch implementation is deliberately limited to dense finite features
and labels `{-1, +1}`. It has no kernels or dual coefficients and is not a
production library replacement. The library example accepts the generator's
`{0, 1}` labels directly.

## Run from the repository root

Dependencies are managed in [pyproject.toml](../../pyproject.toml):

```bash
python -m pip install -e ".[dev]"
python 01-classical-machine-learning/31-support-vector-machine/example.py
python 01-classical-machine-learning/31-support-vector-machine/from_scratch.py
python -m pytest -q 01-classical-machine-learning/31-support-vector-machine/tests
```

The first script prints training/CV scores for 16 RBF configurations, evaluates
the selected model and a prespecified linear baseline, and reports support-vector
counts. The second compares the scratch optimizer against a linear `SVC` with
matched objective normalization. Both datasets are synthetic; neither script
downloads external data or generates visual assets.

Each run overwrites its own ignored JSON record under `outputs/`, relative to
the script: `kernel_search.json` or `linear_solver.json`. These contain the
hypothesis, configuration, versions, results, limitations, and review status.
No generated artifacts are selected for versioning. Public measured summaries
are in the [experiment records](notes.md#executed-experiments).

## Separate visual laboratory

The [visual guide](README_VISUALS.md) connects margins, hinge loss, scaling,
kernel geometry and joint C/gamma behavior. Generate locally from the root:

```bash
python 01-classical-machine-learning/31-support-vector-machine/visualize_svm.py --skip-gifs
```

Omit `--skip-gifs` to include animations. Generated files stay ignored under
`outputs/`; no preview is linked until deliberately selected and unignored.
The visual experiments use separate configurations from the two study scripts.
Interpretations require author review; decision scores are not probabilities,
and the explicit finite lift is not an RBF feature mapping.

## What the executed examples establish

- On the fixed moons split, cross-validation selected `C=1`, `gamma=1`.
  Selected test accuracy was 0.96; the fixed linear baseline scored 0.91.
  This compares a tuned nonlinear model with an untuned reference on one
  synthetic task, and does not establish a general model ranking.
- On standardized synthetic blobs, the scratch objective was approximately
  `0.552029502`, with a gap of `2.58e-9` from the matched linear SVC objective.
  Predictions agreed on all 60 test rows. This is a controlled numerical
  check, not a convergence certificate.
- All measured interpretations remain **pending author review**.

## Key takeaways

Smaller `C` means a smaller penalty on hinge violations relative to the norm
penalty. RBF `gamma` sets the decay scale of similarity; validate it jointly
with `C`. Scaling changes both the distance geometry and the regularization
geometry, so fit preprocessing within training folds.

Support vectors are training observations with nonzero dual multipliers; in a
soft-margin model they can be on the margin, inside it, or misclassified.
Decision scores are not probabilities, and nonlinear score contours are not
equal-distance corridors in the original feature space. Kernel prediction
cost depends on support-vector count as well as feature dimension.

Prerequisites: [regularization](../20-regularization/),
[logistic regression](../21-logistic-regression/),
[validation and leakage](../17-validation-and-leakage/), and
[gradient descent](../../00-foundations/06-gradient-descent-from-scratch/).
