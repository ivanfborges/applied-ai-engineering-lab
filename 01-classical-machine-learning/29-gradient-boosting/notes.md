# Gradient boosting: loss-directed sequential correction

## From a constant to an additive function

For training observations $(x_i,y_i)$, define empirical risk
$R(F)=\sum_i L(y_i,F(x_i))$. Choose a starting constant that minimizes the
same loss:

$$
F_0(x)=\arg\min_c\sum_i L(y_i,c).
$$

At stage $m$, evaluate the negative prediction gradient at the **current
ensemble**, fit a limited learner to that signal, and add its contribution:

$$
r_{im}=-\left.
\frac{\partial L(y_i,F)}{\partial F}
\right|_{F=F_{m-1}(x_i)},\qquad
h_m\approx\arg\min_h\sum_i(r_{im}-h(x_i))^2.
$$

The resulting function is additive in learners:
$F_M=F_0+\sum_{m=1}^M\eta h_m$ in the simplified squared-error implementation.
Additive in learners does not mean linear in features: each learner can itself
be a nonlinear function.

A shallow tree cannot reproduce arbitrary gradients. It approximates an
improving direction using feature-defined regions. A weak learner here means
a restricted building block, such as a depth-one regression stump; it is not
a promise of a particular standalone accuracy.

The sample gradient gives desired corrections at training rows. The fitted
learner extends those corrections to new feature values. This is the connection
between function-space optimization and prediction on unseen observations.

## Why squared error produces residuals

With $L(y,F)=\tfrac12(y-F)^2$,

$$
\frac{\partial L}{\partial F}=F-y,\qquad
r=y-F,\qquad F_0=\bar y_{\mathrm{train}}.
$$

A tree leaf with training indices $R_j$ receives the optimal squared-error
correction

$$
\gamma_j
=\arg\min_\gamma\sum_{i\in R_j}(y_i-F_{m-1}(x_i)-\gamma)^2
=\frac{1}{|R_j|}\sum_{i\in R_j}r_{im}.
$$

Thus the stump's leaf means already encode the correction magnitude. No
additional line search is needed for this implementation. General formulations
can choose a global step by minimizing
$\sum_i L(y_i,F_{m-1}(x_i)+\gamma h_m(x_i))$ or optimize each leaf value
under the original loss. Shrinkage then scales that fitted correction by $\eta$.
Fitting gradients alone does not imply that leaf means are optimal for every
loss.

The [scikit-learn mathematical formulation](https://scikit-learn.org/stable/modules/ensemble.html#mathematical-formulation-of-gradient-boosting)
describes this connection between loss, gradient signal, and tree updates.

### A correction reduces training error under specific conditions

Let $r_i$ denote current residuals and let $h_i$ be the residual mean in the
leaf containing row $i$. Within every leaf,
$\sum_{i\in R_j}r_i h_i=|R_j|\bar r_j^2$. Therefore

$$
\sum_i(r_i-\eta h_i)^2
=\sum_i r_i^2-\eta(2-\eta)\sum_i h_i^2.
$$

For full-data, unweighted residual-mean leaves and $0<\eta\le1$, training
squared error cannot increase in exact arithmetic. This also holds for the
constant fallback leaf. It says nothing about validation error. Subsampling,
other losses, approximate leaf values, and floating-point arithmetic change
the conditions of this statement.

With $x=[0,1,2,3]$, $y=[1,1,5,5]$, the initial prediction is 3 and the first
stump splits at 1.5, predicting residual means -2 and +2. With $\eta=0.5$,
the first updated predictions are $[2,2,4,4]$. Subsequent stages learn
residuals of those updated predictions. They do not keep fitting the original
targets or reuse the first residual vector.

## What changes for classification?

For labels $y\in\{0,1\}$, use a raw logit $F$ and $p=\sigma(F)$:

$$
L(y,F)=\log(1+e^F)-yF,\qquad
-\frac{\partial L}{\partial F}=y-p.
$$

The starting logit is $\log(\bar y/(1-\bar y))$ when both classes are
present. A one-class sample requires special handling because that expression
is not finite.

Although $y-p$ resembles a residual, learners update the **logit**. They do
not directly add corrections to probabilities or vote for labels. Applying
the sigmoid maps the final logit to a probability. Loss-specific leaf
optimization, including Newton updates in some implementations, can differ
from simply taking the mean of $y-p$. This study explains that distinction
without implementing a classification booster.

Absolute-error regression instead uses a residual sign as a subgradient
away from zero; its optimal constant is a median. Loss choice changes both
the error signal and the correct region update. Squared error emphasizes
large target errors and is sensitive to outliers.

## Capacity, validation, and related ensembles

| Decision | Mechanism and trade-off |
|---|---|
| Learning rate | Smaller contributions change later residual targets; reducing the rate on an already fitted ensemble is not equivalent to retraining. |
| Stage count | More learners offer more chances to fit structure and noise; select jointly with learning rate. |
| Depth / leaf count | Controls each correction's complexity. Stumps on multiple features yield sums of univariate functions and cannot express arbitrary interactions. |
| Minimum leaf size | Limits corrections supported by very few observations. |
| Subsampling | Fitting on a row subset adds stochastic regularization; the full-data training-error identity above no longer applies directly. |
| Early stopping | Limits the number of additions using held-out performance; a practical patience rule need not select the exact curve minimum. |

Use leakage-safe validation appropriate to the deployment population:
grouped splits for repeated entities and forward splits for temporal outcomes.
Any learned encoding or preprocessing belongs inside the training boundary.
A random split is suitable for this code-defined interpolation demonstration;
it would not establish forecasting quality.

Training rounds depend on earlier rounds. Random Forest and ExtraTrees fit
trees without earlier trees' prediction errors as targets, then aggregate
their predictions. Forest members can be trained in parallel conditional on
the data, although their errors may remain correlated. Boosting constrains
parallelism across stages; tree-building operations and inference can still
be parallelized. Bias and variance outcomes depend on the data and capacity,
so neither ensemble has a universal performance advantage.

AdaBoost often introduces boosting through observation reweighting and
classification errors; it also has a loss-minimization interpretation.
XGBoost, LightGBM, and CatBoost are implementations with additional algorithmic
and engineering choices. Day 30 examines them separately.

## Educational implementation boundaries

[from_scratch.py](from_scratch.py) accepts shapes $(n,)$ and $(n,1)$, finite
values, a positive estimator count and minimum leaf size, and a learning rate
in $(0,1]$. Targets must be one-dimensional and aligned. Prediction before
fitting raises an explicit error.

The stump tries midpoints between distinct sorted feature values. Duplicates
remain together. Its split objective is the sum of within-leaf squared errors,
subject to the minimum leaf size. Exact score ties keep the first improving
candidate. If there is no improvement or no valid partition, prediction is
the target mean. Refitting resets fitted state.

The exhaustive search scans rows at each candidate, taking roughly $O(nu)$
work per stump for $u$ unique values, plus sorting. This favors readable
mechanics over efficient training. Very large numeric magnitudes can exceed
float64 arithmetic range even when inputs are finite; rescale such data.
The model supports no missing values, sample weights, sparse arrays, multiple
features, alternative losses, or production estimator API.

Its `staged_predict` includes stage 0 and returns independent snapshots.
scikit-learn's method starts after the first tree. The library also converts
tree input features to float32, so thresholds and ties can differ from the
NumPy search. See the [estimator API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).
The controlled test uses integer features representable in both precisions
and checks matched stump predictions at every stage to $10^{-12}$ tolerances.
Passing that check establishes agreement on its fixture, not general library
equivalence.

## Executed learning-rate and stage experiment

**Executed by Codex on 2026-10-06. Interpretations pending author review.**

**Hypothesis:** at a fixed tree budget, smaller learning rates may require
more stages; falling training error need not imply falling validation error.

**Configuration:** 300 evenly spaced values $x\in[-3,3]$,
$y=2\sin(x)+0.4x^2+\epsilon$, with independent
$\epsilon\sim\mathcal N(0,0.35^2)$ from NumPy seed 42.
First reserve 20% for test, then reserve 25% of development rows for validation,
using `train_test_split(random_state=42)` both times. This produces
180 training, 60 validation, and 60 test rows.

Each scikit-learn regressor uses squared error, 300 trees, depth 2,
minimum leaf size 3, full-data fitting (`subsample=1`), seed 42,
and `n_iter_no_change=None`. Only learning rate varies: 1.0, 0.1, 0.01.
Stage 0 uses the training mean. Choose the minimum validation RMSE over all
rates and stages 0–300; exact stage ties favor the earliest stage and rate
ties favor the first rate listed. Keep that fitted prefix without refitting
on validation rows. Test targets are used only after selection.

**Environment:** Python 3.11.0rc2, NumPy 2.4.6, scikit-learn 1.9.0.
Values below are computed against noisy observed targets, not the noiseless
generator. The ignored JSON stores exact split indices and all 301 measured
values per curve; no smoothing or forced monotonic validation trend is used.

| Learning rate | Best validation stage | Train RMSE there | Validation RMSE there | Train RMSE at 300 | Validation RMSE at 300 |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 16 | 0.257364 | 0.358386 | 0.105574 | 0.408962 |
| 0.1 | 234 | 0.218631 | 0.349296 | 0.209375 | 0.351667 |
| 0.01 | 300 | 0.339654 | 0.423260 | 0.339654 | 0.423260 |

Validation selected **rate 0.1, stage 234**. Its final test RMSE was
**0.374100**; the training-mean baseline's test RMSE was **1.869013**.
No alternative candidate was selected using test error.

**Interpretation candidate — author review required:** rate 1.0 reduced
training RMSE after stage 16 while its validation RMSE at stage 300 was
higher. This is consistent with corrections fitting training-specific noise.
Rate 0.01 had its validation minimum at the budget boundary, which suggests
testing a larger budget rather than concluding that small rates are poor.
Rate 0.1 won this particular validation search; it is not a generally optimal
learning rate.

**Limitations:** one synthetic function, noise realization, and split; only
60 evaluation rows and no uncertainty interval. Searching many prefixes
makes the best validation value optimistic. The selected test result is a
single evaluation of this selection procedure. Inspecting the full fitted
path chooses a stage after training and does not save training time as
online early stopping would. This run supplies no comparative evidence
about tree depth, robust losses, classification, or forests.

## Executed four-row mechanics check

**Hypothesis:** for a perfectly separable two-level target, each update with
$\eta=0.5$ halves the remaining residual, so MSE decreases by a factor of four.

**Configuration:** the four synthetic rows above, three stumps, minimum
leaf size 1, learning rate 0.5; training rows also serve as demonstration rows.

**Result:** running `from_scratch.py` produced MSE values
4, 1, 0.25, and 0.0625 at stages 0–3. Final predictions were
$[1.25,1.25,4.75,4.75]$.

**Interpretation candidate — author review required:** the observed trajectory
matches the closed-form update on this exactly representable target.
**Limitation:** this is a training-only arithmetic check with four rows and
provides no evidence of generalization.

## Practical uses and common mistakes

A structured model could combine retrieval scores, document age, and query
metadata to predict an answer-review decision. Labels must reflect the
desired decision, and features must be available before that decision.
Answer length or final latency can leak future information if used to decide
whether to generate an answer. Evaluate the routing policy's quality, latency,
and cost; no such system is implemented here.

Other traps include calling each tree a corrected full predictor, dividing
the boosted sum by the number of trees, choosing stage count on test data,
assuming that trees need no feature preparation, and treating low training
loss as evidence of deployment quality. Tree splits reduce the need for
feature scaling, but categorical encoding and missing-value handling still
depend on the implementation. Feature importance describes predictive use,
not a causal effect. Classification ranking does not establish probability
calibration.

## Suggested extensions — not executed

- Compare depths 1, 2, and 4 at matched rates and budgets; inspect interactions
  and overfitting on multiple seeds.
- Compare full-data and `subsample=0.7` training.
- Add target outliers and compare squared error with a robust loss.
- Implement patience-based early stopping and compare training cost with
  post-training prefix selection.
- Repeat selection across independent seeds or suitable grouped/temporal
  splits before drawing broader conclusions.

These remain proposals. No figures, animations, notebook, or visual companion
were generated in this implementation phase.