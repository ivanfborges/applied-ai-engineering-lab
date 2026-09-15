# Linear Regression Theory

Day 18 studies what ordinary least squares (OLS) estimates, what its
assumptions justify, and what residuals can reveal. A fitted equation can be a
useful baseline for a numerical target while its individual coefficients
remain unsuitable for causal interpretation.

The central question is: **does a successful least-squares fit mean the model
has captured the relationship?** The executable example contrasts a known
linear process with a missing quadratic term. Both satisfy the training
normal equations; their held-out errors behave differently.

## Visual laboratory

The [Visual Linear Regression Lab](linear_regression_visual_lab.py) adds 20
interactive views: fitted lines, signed residuals, RSS surfaces, animated
gradient descent, regression planes, projection geometry, diagnostics,
multicollinearity, outliers, extrapolation, Ridge, and generalization.

From the repository root:

~~~bash
python -m streamlit run 01-classical-machine-learning/18-linear-regression-theory/linear_regression_visual_lab.py
~~~

See the [visual guide](VISUAL_GUIDE.md) for the complete concept map,
installation, preview generation, executed measurements, and review limits.

## Concepts and practical relevance

- OLS objective, normal equations, projection, and rank.
- Coefficients as conditional associations, with units and support limits.
- Assumptions for estimation, unbiasedness, efficiency, and inference.
- Errors versus residuals; training constraints versus held-out diagnostics.
- Multicollinearity, omitted variables, outliers, and extrapolation.

These distinctions matter when building a workload or latency baseline,
explaining a numerical business outcome, or deciding whether a more complex
model addresses an actual failure. A mean-latency model does not establish a
tail-latency service guarantee.

## Study files

| File | Purpose |
|---|---|
| [notes.md](notes.md) | Derivation, assumptions, interpretation, limitations, and executed experiment records |
| [example.py](example.py) | Two deterministic synthetic experiments using scikit-learn; terminal diagnostics |
| [tests/test_linear_regression.py](tests/test_linear_regression.py) | Residual semantics, numerical geometry, and invalid-input checks |
| [interview_questions.md](interview_questions.md) | Questions about assumptions, diagnostics, and practical decisions |
| [references.md](references.md) | External technical sources consulted |
| [linear_regression_visual_lab.py](linear_regression_visual_lab.py) | Interactive Streamlit application with 20 conceptual views |
| [regression_math.py](regression_math.py) | Synthetic experiments and validated numerical helpers |
| [regression_charts.py](regression_charts.py) | Plotly charts, geometry, and animation |
| [generate_visual_assets.py](generate_visual_assets.py) | Reproducible PNG/GIF/HTML previews and experiment records |
| [VISUAL_GUIDE.md](VISUAL_GUIDE.md) | View map, run instructions, executed results, and limitations |
| [requirements.txt](requirements.txt) | Compatibility entry point for the shared dependencies |
| [tests/test_visual_lab.py](tests/test_visual_lab.py) | Numerical and 20-view application smoke tests |

This topic concentrates on visual theory. Its educational optimizer and
Ridge comparisons provide a bridge to the deeper Day 19 and Day 20 studies
in the [roadmap](../../ROADMAP.md); those future days are not marked complete.
The diagnostic helpers and visual experiments are not production numerical
or statistical inference software.

## Run from the repository root

Use the shared environment described in the [root setup](../../README.md#quick-start).
Dependency bounds remain in [pyproject.toml](../../pyproject.toml).

```bash
python 01-classical-machine-learning/18-linear-regression-theory/example.py
python -m pytest -q 01-classical-machine-learning/18-linear-regression-theory/tests
```

The example needs no downloads, credentials, command-line options, or output
directory. All observations are synthetic, in arbitrary units. Each
experiment generates 500 observations with NumPy seed 42 and uses a fixed
400/100 train/test split. Both feature sets in the second experiment are
specified in advance; the test set is not used to choose them.

## Executed evidence

These are measured results from the run recorded in [notes.md](notes.md),
not benchmarks or author-endorsed conclusions.

| Experiment/model | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| Known linear mean | 1.890698 | 1.464149 | 0.920035 |
| Quadratic process, fitting only x | 5.287462 | 4.493228 | 0.425371 |
| Same process, fitting x and x² | 0.968058 | 0.786984 | 0.980738 |

**Interpretation candidate — author review required:** training residual
orthogonality verifies an optimization property, not the adequacy of the
feature set. The quadratic experiment illustrates this distinction. One
synthetic split does not establish performance or model adequacy on real
workloads.

## Key takeaways

- OLS minimizes squared error; it does not require normally distributed inputs.
- Statistical guarantees depend on assumptions that fitting alone cannot verify.
- A coefficient's meaning depends on units, included variables, and available
  variation; significance would not establish causality.
- Structured residuals can motivate a better specification. Neither zero
  training residual mean nor a high R² proves that a model is appropriate.
- A polynomial can remain linear in its coefficients.

The terminal example writes no artifacts. The separate visual generator
produces seven PNGs, three GIFs, four offline HTML files, and an experiment
record in the ignored outputs directory. No public preview is selected or
unignored; review candidates and measured visual results are listed in the
[visual guide](VISUAL_GUIDE.md).
