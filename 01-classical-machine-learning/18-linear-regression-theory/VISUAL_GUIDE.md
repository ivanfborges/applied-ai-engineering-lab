# Visual Linear Regression Lab

The laboratory makes the mathematics and failure modes of regression
inspectable. All examples are synthetic; changing a control recalculates the
fit and displayed values. The three-axis projection view is in **observation
space**; the multiple-regression plane is in **feature–target space**.

## Run locally

From the repository root, install the shared dependencies if needed:

```bash
python -m pip install -r 01-classical-machine-learning/18-linear-regression-theory/requirements.txt
python -m streamlit run 01-classical-machine-learning/18-linear-regression-theory/linear_regression_visual_lab.py
```

The topic requirements file forwards to the root requirements file and
[pyproject.toml](../../pyproject.toml), preserving one dependency definition.
Run the installation command from the root because the editable package
requirement is relative to the working directory. No new dependencies,
credentials, data downloads, or cloud services are required.

Choose a view in the sidebar. Only that view executes. Drag to rotate Plotly
3D charts; hover for values. The gradient-descent view has Play, Pause, and
frame-selection controls. Larger repetitions take longer and are cached with
a bounded cache. Navigating between views may reset view-specific controls.

## Concept-to-view map

Each view has “What you are seeing,” “Why this happens,” and an interview
takeaway. Metrics label the evaluation sample; training metrics are not
reported as held-out performance.

| View | Question answered | Interaction or evidence |
|---|---|---|
| 01 · Fitting a line | How does noise change estimated coefficients? | True intercept/slope, noise, sample size, seed; fitted and generating lines; training RMSE/R² |
| 02 · Residual distances | What exactly gets squared? | Signed vertical segments, prediction markers, RSS/MSE/RMSE, residual sum |
| 03 · Why squared error? | Why are large errors influential? | Signed, absolute, and squared values for residual magnitudes 1, 2, 3, 5, 10 |
| 04 · OLS loss landscape | Where is the minimum in parameter space? | Rotatable RSS surface and contours; generating coefficients and exact OLS minimum |
| 05 · Gradient descent | How does an iterative method approach that minimum? | Synchronized line/path animation; step count and learning rate; loss history |
| 06 · Multiple regression plane | What does adding a second predictor look like? | Rotatable points and fitted plane; both true slopes adjustable |
| 07 · Geometric projection | Why is the residual orthogonal to the design? | Three response coordinates; column-space plane, vectors, right angle, computed Dᵀe |
| 08 · Residual diagnostics | What can a single fit statistic hide? | Actual/predicted, residual/fitted, histogram; no Gaussianity claim |
| 09 · Nonlinear misspecification | Can a linear coefficient model fit curvature? | Side-by-side x-only and x/x² fits and residuals |
| 10 · Heteroskedasticity | What changes when conditional variance grows? | Paired residual plots; adjustable variance-growth rate |
| 11 · Multicollinearity | Can unstable slopes give stable predictions? | Independent/correlated designs, coefficient variance, repeated-fit predictions, elongated loss contours |
| 12 · Outlier sensitivity | How far can one observation pull a fit? | Added-point position and response offset; before/after fits and clean holdout RMSE |
| 13 · Leverage vs residual | Can an influential point end with a small residual? | Leverage, final residual, slope change; same point controls |
| 14 · Extrapolation | What does the model know beyond observed support? | Identical training data; linear, saturating, or accelerating unseen continuation |
| 15 · OLS vs Ridge | What does shrinkage trade away? | Alpha, original-unit coefficients, paths, predictions, held-out RMSE, finite-ensemble squared bias and variance |
| 16 · Holding other variables constant | What does a conditional coefficient mean? | Move tokens or concurrency while fixing the other in a hypothetical latency equation |
| 17 · Intercept interpretation | Is the prediction at zero meaningful? | Switch from data around zero to data only above 100 |
| 18 · R² and the mean baseline | What variation does R² compare? | SST and RSS distances on the same sample; undefined constant-target R² |
| 19 · Training vs generalization | Can polynomial linear regression overfit? | Degrees 1–15, displayed curve, training/validation errors, design conditioning |
| 20 · Normal equations | Why avoid solving through the Gram matrix? | D, DᵀD, Dᵀy, pseudoinverse/lstsq/scikit-learn comparison; adjustable near-collinearity |

The assumptions discussion is distributed across views 07–11 and 14–16:
optimization identities do not establish exogeneity, Gaussian predictors are
unnecessary, variance assumptions affect inference, and observational
coefficients do not establish causal effects.

## Files and numerical choices

- [linear_regression_visual_lab.py](linear_regression_visual_lab.py): one
  application with conditional view selection and concise explanations.
- [regression_math.py](regression_math.py): deterministic generators,
  least-squares fits, bounded gradient descent, repeated samples, and metrics.
- [regression_charts.py](regression_charts.py): Plotly figures and animation.
- [generate_visual_assets.py](generate_visual_assets.py): Matplotlib/Pillow
  previews, self-contained interactive HTML, and experiment records.
- [tests/test_visual_lab.py](tests/test_visual_lab.py): numerical invariants,
  input boundaries, 20-view Streamlit smoke test, and control changes.

Main fits use NumPy least squares or scikit-learn. View 20 deliberately
computes `pinv(D.T @ D) @ D.T @ y` as a numerical caution, not the recommended
solver. Gradient descent minimizes MSE; its stability boundary is
`2 / largest_eigenvalue(2 * D.T @ D / n)`. The UI permits an unstable rate
to make divergence visible, with a finite-loss guard and bounded iterations.

Ridge uses training-only standardization and an unpenalized intercept.
Displayed coefficients are transformed back to original units. Repeated
experiments independently regenerate training samples and evaluate them at
one fixed fresh design. The bias–variance chart compares predictions against
the known conditional mean, excluding observation noise; the finite-ensemble
decomposition is checked numerically. Error bars in view 11 are variability
across fitted models, not confidence or prediction intervals.

The heteroskedastic comparison matches average conditional variance and
shares standardized noise draws. Increasing spread is therefore not merely
a comparison of different overall noise scales.

## Generate and review artifacts

From the repository root:

```bash
python 01-classical-machine-learning/18-linear-regression-theory/generate_visual_assets.py
```

Defaults: seed 42, 36 animation frames. Optional controls:

```bash
python 01-classical-machine-learning/18-linear-regression-theory/generate_visual_assets.py --seed 7 --frames 24
```

The default output location is the topic's `outputs/` folder, resolved
relative to the generator, regardless of the working directory.
Generation overwrites its named artifacts. Frames are bounded to 12–80;
sample sizes, repetitions, loss grids, and iteration counts are also bounded.

### Assets generated in this implementation

All are **regenerable artifacts**, ignored by Git. No public preview has been
selected or deliberately unignored. These filenames are plain text rather
than public links to ignored files.

| Type | Files in outputs/ |
|---|---|
| PNG | `ols_loss_surface.png`, `residual_misspecification.png`, `heteroskedasticity.png`, `multicollinearity_stability.png`, `ridge_coefficient_paths.png`, `geometric_projection.png`, `multiple_regression_plane.png` |
| GIF | `gradient_descent_ols.gif`, `outlier_influence.gif`, `regression_sample_growth.gif` |
| Offline HTML | `ols_loss_surface.html`, `gradient_descent_ols.html`, `geometric_projection.html`, `multiple_regression_plane.html` |
| Experiment record | `experiment_report.json` |

HTML files include Plotly itself so they open without network access; each is
approximately 5 MB and should remain local. PNGs and GIFs are the more
practical review candidates.

**Strong candidates for manual GitHub/LinkedIn review:** the gradient-descent
GIF, residual-misspecification PNG, multicollinearity PNG, outlier GIF, and
projection PNG. Choosing an asset does not imply approval of a conclusion or
publication. No LinkedIn draft was created or changed.

## Executed experiments and review notes

The generator was run with seed 42 and 36 frames. Its JSON report records
hypothesis, configuration, result, interpretation candidate, and limitation
for eight experiment groups: optimization, curvature, heteroskedasticity,
collinearity/Ridge, outliers, sample growth, projection, and the regression
plane.

Observed values from that run:

| Experiment and configuration | Measured result | Interpretation candidate — author review required |
|---|---|---|
| Gradient descent, n=80, noise SD=1, 90-step budget, rate=0.6 of stability boundary | MSE 50.116156 initially; 0.859169 finally; direct OLS 0.859169 | This controlled quadratic reaches the same objective value within numerical precision |
| Curvature, n=100, noise SD=1, mean=3x+2x² | Training RMSE 4.856305 for x only; 1.033726 for x/x² | The generating feature removes the visible omitted curvature; this is not a generalization result |
| Collinearity, 60 independent samples of n=80, noise SD=1 | Var(beta1) 0.018794 for independent features; 0.804978 at population correlation 0.99 | Slopes are less stable under the nearly dependent design |
| Ridge alpha=10, same 60 samples, fixed 250-row test design | OLS prediction variance 0.045935; Ridge 0.031023; MSE against true mean 0.045979 vs 0.132118 | Ridge reduces variance here, but its added squared bias outweighs that reduction |
| Added point at x=8, offset rising to 30, n=50, noise SD=0.7 | Slope changes from 1.436833 to 2.614747 | This contamination path pulls the fit; other point positions need not behave identically |
| Three-observation projection | Maximum absolute Dᵀe = 8.88e-16 | The numerical projection satisfies orthogonality to floating-point precision |

The remaining generated records include noise SD range 0.3265–3.8122 in the
heteroskedastic setup, final sample-growth coefficients [1.984677, 1.571299],
and plane coefficients [1.936237, 2.856799, -1.985180]. These are single-seed
descriptions, not benchmarks or author conclusions.

Executed environment: Python 3.11.0rc2, NumPy 1.26.4, scikit-learn 1.9.0,
Matplotlib 3.7.3, Plotly 6.9.0, Streamlit 1.60.0, Pillow 10.0.0. The pre-existing
Matplotlib installation is older than the shared declared minimum; the
generator nevertheless ran successfully in this environment. Installation
instructions continue to use the repository's supported dependency bounds.

## Validation and practical limits

```bash
python -m pytest -q 01-classical-machine-learning/18-linear-regression-theory/tests
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
```

The topic suite checks least-squares/library agreement, rank-deficient
projection, residual orthogonality, direct RSS evaluation, descent stability
and divergence, original-unit Ridge coefficients, the finite-ensemble
bias–variance identity, deterministic isolated randomness, nested polynomial
training losses, undefined R², invalid inputs, and app rendering.

All 20 views were exercised with Streamlit AppTest, including zero-noise and
constant-target controls, unstable descent, alpha zero, and a nearly singular
design. This checks Python execution and figure construction; it does not
replace manual inspection of browser hover, WebGL rotation, responsive
layout, and animation playback.

Author review remains necessary for:

- every interpretation candidate and proposed public preview;
- projection geometry and chart readability at the intended sharing size;
- extrapolation assumptions and practical feature availability;
- using repeated-sample variability as an illustration rather than a
  population guarantee;
- interpreting a validation-selected polynomial degree without reusing that
  validation score as a final untouched-test result.

No robust standard errors, causal identification, confidence-interval
coverage study, or production latency measurements are claimed.