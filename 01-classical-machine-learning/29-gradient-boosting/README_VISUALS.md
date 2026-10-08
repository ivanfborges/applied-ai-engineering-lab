# Gradient Boosting Visual Laboratory

Watch the ensemble evolve, then inspect the residual targets and corrections
that produced each stage. [visual_core.py](visual_core.py) implements the
squared-error boosting loop explicitly, using sklearn to fit weak regression
trees. [visual_lab.py](visual_lab.py) renders its history and runs separate
library comparisons. The existing [NumPy stump model](from_scratch.py) remains
a smaller first-principles alternative. These are educational implementations.

## Install and run

Use Python 3.11+ and the shared dependencies in
[pyproject.toml](../../pyproject.toml). No new dependencies were added.
NumPy, scikit-learn, Matplotlib, and Pillow provide the PNG/GIF views;
Plotly provides the offline HTML explorers.

From the repository root:

~~~bash
python -m pip install -e ".[dev]"
python 01-classical-machine-learning/29-gradient-boosting/visual_lab.py
~~~

Inside this topic folder, the main command is:

~~~bash
python visual_lab.py
~~~

The default command generates every view and, with a GUI-capable Matplotlib
backend, opens the saved one-step, learning-rate, and validation figures.
Close those windows to finish. Open either HTML directly in a browser; it
embeds JavaScript and needs neither a web server nor an internet connection.
No Kaleido, FFmpeg, credentials, or browser automation package is required.

Useful alternatives:

~~~bash
python visual_lab.py --headless
python visual_lab.py --headless --skip-html --skip-gifs
python visual_lab.py --headless --dpi 80 --output-dir outputs/low-resolution
~~~

Options are limited to --headless, --skip-html, --skip-gifs, --dpi (50–200),
and --output-dir. The default output path is relative to the generator;
custom relative paths resolve from the working directory. Matching output
files are overwritten. Skip flags do not remove files from earlier runs;
the JSON manifest lists only assets generated during the current run.

## Files and generated outputs

~~~text
29-gradient-boosting/
├── visual_core.py
├── visual_lab.py
├── README_VISUALS.md
├── tests/
│   └── test_visual_lab.py
└── outputs/                         # ignored, generated on demand
    ├── figures/                     # 12 PNGs + 2 HTML explorers
    ├── gifs/                        # 2 bounded animations
    ├── stage_history.npz             # inspectable numerical arrays
    └── visual_experiments.json       # actual measurements and review status
~~~

The earlier nonvisual example's experiment.json is separate and unchanged.
All paths below are relative to outputs/. They are regenerable artifacts,
rather than links to versioned public previews.

| View | Output | Educational question |
|---:|---|---|
| 1 | figures/01_initial_model.png | How does the training mean create a constant F0, and where are its errors? |
| 2 | figures/02_residual_learning.png | How do errors of F0 become targets for a shallow correction tree? |
| 3 | figures/03_one_boosting_step.png | How do the previous function, raw learner, shrinkage, and new sum connect at stage 5? |
| 4 | gifs/04_sequential_boosting.gif | How does the approximation evolve from F0 to F100? Thirteen actual stages are shown, with denser early sampling. |
| 5 | gifs/05_residual_evolution.gif | How do training and validation residuals change on fixed axes at stages 0, 1, 5, 10, 25, 50, and 100? |
| 6 | figures/06_additive_prediction.png | How do signed contributions accumulate at the grid point nearest x=1? |
| 7 | figures/07_learning_rate.png | How do rates 1.0, 0.3, 0.1, and 0.03 affect validation paths and predictions after the same 20-tree budget? |
| 8 | figures/08_tree_complexity.png | How do depths 1, 2, 4, and 8 change final functions and train/validation error at matched settings? |
| 9 | figures/09_train_validation_loss.png | Where is the measured validation minimum, and what happens afterward? A late-stage inset reveals smaller changes. |
| 10 | figures/10_functional_gradient_descent.png | How does updating a parameter vector relate to adding a function? This is a conceptual schematic. |
| 11 | figures/11_negative_gradient.png | What are the local loss slope, ideal gradient step, and actual tree approximation at one observation? |
| 12 | figures/12_pseudo_residuals.png | Do finite differences match y-F for squared error and y-sigmoid(F) for logistic loss? |
| 13 | figures/13_boosting_vs_bagging.png | Why do forest members predict original targets while boosting members supply dependent corrections? |
| 14 | figures/14_boosting_surface_3d.html | How does a two-feature piecewise approximation approach a known interacting surface at stages 0, 1, 5, 10, 25, 50, and 100? |
| 15 | figures/15_decision_stump.png | How do a fitted threshold and two leaf residual means approximate a gradient? |
| 16 | figures/16_stage_explorer.html | What prediction, residuals, latest correction, and RMSE belong to each stage 0–100? |

## What to look for

Start with views 1–3. Each learner fits errors of the **whole current ensemble**
and contributes a signed addition:

$$
r_{im}=y_i-F_{m-1}(x_i),\qquad
F_m(x)=F_{m-1}(x)+\eta h_m(x).
$$

Move the stage slider backward as well as forward. At stage m, the correction
panel shows the addition that produced F_m. Residuals are shown **after** the
update and supply the next learner's targets. Stage 0 has a zero correction.

Broad residual structure can diminish while noise and held-out errors remain.
A region-mean correction approximates many pointwise gradients; it may increase
an individual observation's loss even when total training squared error falls.
View 11 distinguishes the ideal pointwise gradient from the fitted tree value.

View 6 sums contributions without averaging. Changing learning rate changes
later targets and splits, rather than simply rescaling an already fitted model.
Inspect views 7–9 without assuming that smaller rates or particular depths
are universally better. The marked validation minimum selects a prefix after
training; it does not save training time as online early stopping would.

The general target is

$$
r_{im}=-\left.\frac{\partial L(y_i,F)}{\partial F}\right|_{F=F_{m-1}(x_i)}.
$$

View 12 checks the squared-error identity and six code-defined logistic
observations. Logistic signals update **raw logits**, rather than probabilities.
No classification booster or calibration experiment is performed.

The forest has 30 trees, depth 4, minimum leaf size 3, bootstrap enabled, and
seed 42. With one feature, feature subsampling cannot create feature diversity.
Its trees fit without using earlier members' errors, but their prediction
errors may still be correlated. No fair performance comparison is claimed.

The 3D experiment adds a feature interaction. Its display connects a 32 x 32
sampled grid; rendering interpolation should not be mistaken for smooth tree
predictions. Both surfaces use shared value/color ranges.

## Inspect the history

The main loop uses 100 stages, rate 0.2, depth 2, minimum leaf size 3, and
a fixed tree seed. Numerical arrays in stage_history.npz include features,
targets, known grid truth, split indices, predictions, residuals, corrections,
RMSE, and learning rate. No models are pickled.

Prediction and residual arrays have rows 0–100. Raw correction arrays have
rows 0–99, so correction row m-1 is h_m, before shrinkage:

~~~python
import numpy as np

with np.load("outputs/stage_history.npz", allow_pickle=False) as history:
    m = 5
    before = history["predictions_train"][m - 1]
    residual_targets = history["residuals_train"][m - 1]
    applied = history["learning_rate"] * history["corrections_train"][m - 1]
    after = history["predictions_train"][m]
    np.testing.assert_allclose(after, before + applied)
~~~

The comparison experiments use GradientBoostingRegressor. Its internal seed
sequence and split implementation need not match the explicit educational
loop. The original NumPy stump model was not modified.

## Theoretical expectation and observed results

**Theoretical expectation:** full-data squared-error residual-mean updates
with rate in (0,1] cannot increase training squared error in exact arithmetic.
Validation error has no corresponding guarantee, and noise need not vanish.
See [the derivation](notes.md#why-squared-error-produces-residuals).

**Executed configuration:** Codex ran the lab on 2026-10-06. The 1D dataset
has 300 evenly spaced values in [-3,3], y = 2 sin(x) + 0.4 x^2 +
Normal(0,0.35^2), seed 42. It uses the same 180/60/60
train/validation/reserved-test split as the original example. The visual lab
does not score reserved test rows.

The rate comparison uses 300 trees, depth 2, and minimum leaf size 3.
The depth comparison uses rate 0.1, 100 trees, and minimum leaf size 3.
The flexible path uses rate 0.1, 500 trees, depth 4, and minimum leaf size 1.
All library experiments use squared error, full-data fitting, seed 42,
the training-mean initialization, and no built-in early stopping.
No dataset noise or split adjustment was made to obtain the flexible path.

The 2D experiment uses 600 uniform observations in [-2.5,2.5]^2,
y = 1.5 sin(x1) + 0.4 x2^2 + 0.6 x1 x2 + Normal(0,0.3^2), seed 42,
a 450/150 train/validation split, 100 trees, rate 0.1, depth 3,
minimum leaf size 4, and full-data squared-error fitting.

**Observed results from the executed run:**

| Rate | Best validation stage | Validation RMSE there |
|---:|---:|---:|
| 1.0 | 16 | 0.358386 |
| 0.3 | 79 | 0.350641 |
| 0.1 | 234 | 0.349296 |
| 0.03 | 299 | 0.360346 |

The depth figure displays final-stage functions, with these measured errors:

| Depth | Train RMSE at 100 | Validation RMSE at 100 |
|---:|---:|---:|
| 1 | 0.351747 | 0.415983 |
| 2 | 0.249806 | 0.359360 |
| 4 | 0.206684 | 0.353025 |
| 8 | 0.163245 | 0.363298 |

The main loop's validation minimum was at stage 100 (the budget boundary): train RMSE 0.223717, validation RMSE 0.347656.

The flexible model's minimum was stage 57: train RMSE 0.163044, validation RMSE 0.391220. At stage 500, train RMSE was 0.010987 and validation RMSE was 0.439980.

The 2D model's final validation RMSE was 0.408207.

Finite differences (step 1e-5) had maximum absolute gradient differences 3.23e-11 for squared error and 3.70e-12 for logistic loss.

Environment: python 3.11.0rc2, numpy 2.4.6, scikit_learn 1.9.0, matplotlib 3.11.0.

**Interpretation candidates — author review required:** inspect whether the
flexible path supports fitting training-specific patterns after its validation
minimum. Slow-rate trajectories invite testing larger budgets instead of
declaring a generally best rate. Review depth comparisons alongside the known
function and synthetic noise, and assess whether the figures communicate
the intended mechanism before selecting previews.

**Limitations:** one synthetic function, noise realization, and split per
experiment; no uncertainty estimates, production data, or unbiased test score.
Validation minima are optimistic after stage/parameter search. The lab supplies
no classifier-quality result or fair forest-versus-boosting benchmark.
The 2D experiment does not establish extrapolation quality.
visual_experiments.json records hypotheses, configurations, actual results,
interpretation candidates, limitations, environment, and generated asset sizes.

## Validation and publication review

From the repository root:

~~~bash
python -m pytest -q 01-classical-machine-learning/29-gradient-boosting/tests
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
~~~

Tests cover sequential addition, residual targets and leaf means, numerical
gradients, repeatability, split isolation, history storage, PNG rendering,
slider frame contents, and headless CLI execution. Both HTML explorers were
also checked in local headless Chrome at stage 50; the 1D frame values matched
saved history. GUI window behavior was not manually exercised.

All 16 requested views are implemented. The 3D view is delivered as HTML
without a PNG-export dependency. The GIFs use 13 and seven frames, respectively;
frame spacing is intentionally unequal. Each HTML embeds several MB of
JavaScript to remain usable offline.

Recommended review candidates are the sequential GIF, one-step PNG, and
validation-loss PNG. The stage explorer is especially useful for local study.
These are suggestions only. **No output is intentionally unignored or linked
as a versioned preview.** QA contact sheets, screenshots, and isolated browser
profiles under outputs/qa_* are ignored intermediate artifacts. The root
README and publication decisions are unchanged in this visual phase.

Implementation references:
[Matplotlib animation](https://matplotlib.org/stable/api/animation_api.html),
[Plotly sliders](https://plotly.com/python/sliders/), and
[offline HTML export](https://plotly.com/python/interactive-html-export/).
