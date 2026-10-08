# Modern boosting: objectives and engineering choices

## From residuals to a Newton update

For squared error, fitting negative gradients recovers residual fitting.
For binary logistic loss, work in logit space instead:

$$\ell(y,z)=\log(1+e^z)-yz,\quad p=\sigma(z),\quad
 g=\frac{\partial\ell}{\partial z}=p-y,\quad
 h=\frac{\partial^2\ell}{\partial z^2}=p(1-p).$$

The code evaluates loss with `logaddexp` and sigmoid without exponentiating
large positive values. It computes the Hessian as
`sigmoid(z) * sigmoid(-z)` to retain tiny curvature when `sigmoid(z)` rounds to
one. At extreme margins even that curvature can underflow to zero; a leaf with
zero total curvature and zero L2 penalty is rejected explicitly.

For a proposed tree, discard the constant current loss and use the local
quadratic approximation:

$$Q(f)=\sum_i\left[g_i f(x_i)+\tfrac12 h_i f(x_i)^2\right]
       +\gamma T+\tfrac\lambda2\sum_{j=1}^T w_j^2.$$

The leaf membership sets define $G_j=\sum_{i\in I_j}g_i$ and
$H_j=\sum_{i\in I_j}h_i$. Within one leaf,

$$Q_j(w)=G_j w+\tfrac12(H_j+\lambda)w^2+\gamma.$$

Setting its derivative to zero gives

$$w_j^*=-\frac{G_j}{H_j+\lambda},\qquad
 Q_j(w_j^*)=-\tfrac12\frac{G_j^2}{H_j+\lambda}+\gamma.$$

Subtract the two-child optimum from the parent optimum:

$$\operatorname{gain}=\tfrac12\left[
 \frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}
 -\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma.$$

This derivation uses convex local curvature, positive denominators, fixed leaf
membership, no sample weights, and L2 regularization only. L1 introduces a
soft-thresholded gradient; the scratch implementation omits it. The derivation
follows the [XGBoost objective tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/model.html).

A positive gain is improvement in the **approximate regularized objective**,
not a guarantee of improved validation loss or the exact nonlinear objective.
The demo reports exact in-sample logistic loss separately. After constructing
a tree, shrinkage applies $\eta w$ to its leaves; the displayed gain is computed
before shrinkage. The demo sets $\eta=1$ for transparent arithmetic.

`fit_newton_stump` searches all distinct one-feature partitions. It stores the
largest observed value in the left partition and predicts with `<=`, avoiding
midpoint overflow and keeping duplicate values together. Equal positive gains
retain the earliest threshold. If no gain exceeds zero, it returns the optimized
parent leaf; that leaf can still change the current scores. Minimum child weight
means **sum of Hessians**, not a number of rows. A four-row leaf with confident
predictions may have less curvature than a smaller uncertain leaf.

## Growth policy, histograms, and capacity

XGBoost's `depthwise` policy favors nodes near the root; `lossguide` favors
leaves with larger loss changes. These are configurable choices, so describing
XGBoost as exclusively level-wise is inaccurate. Histogram mode aggregates
statistics over quantized numerical features. More bins allow finer candidates
at greater memory and computation cost. See the
[XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).

LightGBM grows leaves best-first. At a fixed leaf budget this can concentrate
capacity in a difficult region and produce a deep branch. Control leaf count,
maximum depth, minimum row support, and minimum curvature rather than assuming
all depth settings have the same meaning. Its histogram design groups candidate
values and can reuse histogram information between parent and child nodes.
See the [features guide](https://lightgbm.readthedocs.io/en/stable/Features.html).

GOSS keeps large-gradient rows and samples small-gradient rows, with reweighting
to estimate their contribution. It is an optional sampling strategy, not a loss
or an automatic property of every LightGBM fit. Exclusive Feature Bundling
compacts mutually exclusive sparse features; useful savings depend on feature
sparsity. The example specifies ordinary `gbdt` and `data_sample_strategy="bagging"`
with row sampling disabled. See
[LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html).

A symmetric CatBoost tree applies the same split at every node of a level. A
depth-three tree has eight leaf positions, even if some receive no rows. This
restricts local partition flexibility and permits compact evaluation. CatBoost
also offers other growth policies. The example explicitly uses `SymmetricTree`.
A depth-three XGBoost tree and an eight-leaf LightGBM tree need not represent
the same partitions or receive the same regularization.

All example models cap depth at three; LightGBM additionally caps leaves at
eight. Learning rate 0.05, a 400-round ceiling, and patience 30 make the comparison
readable. They do **not** equalize initial scores, leaf estimation, binning,
category statistics, minimum leaf support, or effective model capacity.

## Categorical statistics and prediction shift

For a nominal feature, arbitrary integer codes must not imply a meaningful
order. LightGBM can search categorical partitions using categorical codes;
XGBoost supports categorical partitions when configured appropriately. Native
support does not imply every numeric-coded column is recognized as categorical.
The mixed example marks LightGBM columns with a shared training-derived pandas
category dtype. Its absent held-out categories map to missing codes. See the
[LightGBM categorical guide](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html)
and [XGBoost categorical tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html).

Naive target means computed before splitting can expose held-out labels. Even
when computed on training rows only, a row contributing its own label to its
encoding creates self-inclusion bias; singleton categories make it obvious.

An educational ordered statistic for category $c_i$ at position $i$ in a
permutation $\pi$ is

$$s_i=\frac{\sum_{j:\pi(j)<\pi(i)}\mathbf1(c_j=c_i)y_j+aP}
 {\sum_{j:\pi(j)<\pi(i)}\mathbf1(c_j=c_i)+a}.$$

Here $a>0$ is prior strength and $P$ is a chosen prior. Earlier rows contribute;
the current row and later rows do not. This formula explains the idea rather
than reproducing CatBoost's full categorical machinery, which includes
combinations and multiple statistics. Keep validation/test labels outside
training statistics; ordered encodings are not a replacement for time-aware
splits or cross-fitting in an independently built encoder.

Ordered boosting addresses a related prediction-shift problem: a training row's
prediction used to construct its gradient can be influenced by its own target.
Permutation-prefix models approximate the information boundary faced by unseen
rows. This is separate from ordering categorical statistics. Both mechanisms
are explained in the [CatBoost paper](https://arxiv.org/abs/1810.11363).

The CPU boosting default is `Plain`; defaults vary with execution conditions.
This study requests `boosting_type="Ordered"` so the mechanism is an explicit
choice. Ordered boosting is not synonymous with chronological training. It
cannot fix future-derived features or entity overlap. See
[CatBoost training parameters](https://catboost.ai/docs/en/references/training-parameters/common).

## Validation, applications, and mistakes

Choose the split before tuning. Use entity separation for new-customer
prediction and a temporal boundary for future events. Random stratified splits
here assume independent synthetic rows and are not fraud or churn evaluations.
Scaling is usually unnecessary for tree thresholds, but preprocessing can still
leak information through imputation, category vocabularies, target encoding,
or feature aggregation windows.

The example monitors validation log loss for all three libraries and selects
the smallest validation loss, with a fixed candidate order resolving exact
ties. It does not compare all test scores and select a library afterward. The
constant test baseline uses only training prevalence. Best-prefix predictions
use XGBoost's selected iteration, LightGBM's selected iteration, and CatBoost's
retained best model. Validation estimates are optimistic after both stopping
and model selection. Changing settings after inspecting test results consumes
the test's evaluation boundary.

ROC-AUC measures ranking; log loss and Brier score assess probabilities. Their
changes combine calibration and discrimination effects. A calibration study
would need separate calibration data and reliability diagnostics. No calibration
model is fitted here. A useful operational evaluation would also assess
threshold costs, review capacity, delayed labels, subgroups, latency, and
features available at inference. Feature importance does not establish causality.

Common mistakes include assuming CatBoost prevents all leakage, treating IDs
as reusable categories, reading `min_child_weight` as a row count, believing
LightGBM's `subsample` takes effect without a positive sampling frequency,
and announcing a universal winner from untuned defaults or one split.

## Executed experiment records

Executed by Codex on **2026-10-07**. These are measured demonstrations;
all interpretation candidates are **pending author review**.

### Three-library workflows

**Hypothesis:** shared-row, limited-capacity boosting may improve probability
loss over a training-prevalence baseline. Categorical representations may alter
validation behavior without establishing a library ranking.

**Configuration:** seed 42; 2,400 independent synthetic rows per scenario;
stratified train/validation/test sizes 1,440/480/480. All candidates use CPU,
one thread, learning rate 0.05, a 400-round ceiling, depth three, and L2 penalty
parameter 1. LightGBM has at most eight leaves and minimum row support 20;
XGBoost has minimum Hessian mass 1. Row/column subsampling is disabled. CatBoost
uses ordered boosting, symmetric trees, no bootstrap, zero random split-score
strength, and one Newton leaf-estimation iteration. Other framework defaults
remain in effect. These settings do not equalize regularization or expressivity.

Early-stopping patience is 30, monitored metric is validation log loss, and
model choice minimizes that same metric. Predictions retain each selected
prefix; there is no refit. The test set is scored only for the chosen candidate
and the training-prevalence baseline. No timing measurements were collected.

The numeric generator is scikit-learn `make_classification`: eight features,
five informative, one redundant, class weights 0.65/0.35, label-flip probability
0.03, class separation 1, seed 42, and other generator defaults. The mixed
generator draws four independent standard-normal numeric values, uniform
merchant indices from 80 possibilities, and uniform channel indices from four.
Each merchant receives a fixed standard-normal effect drawn by the same seeded
RNG. Conditional binary labels are Bernoulli with sigmoid logit

$$z=-0.8+1.2x_0-0.9x_1+0.7x_2x_3+b_{\mathrm{merchant}}
    +0.6\mathbf1(\mathrm{channel}=1).$$

The two scenarios use different data-generating mechanisms; their metric
levels cannot be compared as an encoding ablation.

**Executed environment:** python 3.11.0rc2; platform Windows-10-10.0.19045-SP0; numpy 2.4.6; pandas 3.0.3; scikit-learn 1.9.0; xgboost 3.2.0; lightgbm 4.7.0; catboost 1.2.10.

#### Numeric scenario

**Result:** training input feature counts are XGBoost=8, LightGBM=8, CatBoost=8.

| Candidate | Selected rounds | Train log loss | Validation log loss | Validation ROC-AUC | Validation Brier |
|---|---:|---:|---:|---:|---:|
| XGBoost | 392 | 0.132145 | 0.271328 | 0.945968 | 0.079645 |
| LightGBM | 283 | 0.159390 | 0.273274 | 0.945665 | 0.080486 |
| CatBoost | 398 | 0.187133 | 0.286946 | 0.941199 | 0.085348 |

| Final evaluation | Test log loss | Test ROC-AUC | Test Brier |
|---|---:|---:|---:|
| XGBoost (validation selected) | 0.272537 | 0.948731 | 0.076866 |
| Training-prevalence baseline | 0.651231 | 0.500000 | 0.229336 |

**Interpretation candidate — pending author review:** the selected XGBoost
prefix has lower test log loss than the constant baseline on this synthetic
split. Validation losses are relatively close across candidates; no superiority
claim follows from one realization. XGBoost and CatBoost selected prefixes near
the round ceiling, so this run does not establish their best larger-budget
performance.

#### Mixed scenario

**Result:** training input feature counts are XGBoost=88, LightGBM=6, CatBoost=6.

| Candidate | Selected rounds | Train log loss | Validation log loss | Validation ROC-AUC | Validation Brier |
|---|---:|---:|---:|---:|---:|
| XGBoost | 392 | 0.411998 | 0.501028 | 0.815383 | 0.165942 |
| LightGBM | 249 | 0.331045 | 0.495614 | 0.824581 | 0.163639 |
| CatBoost | 376 | 0.473626 | 0.493024 | 0.824453 | 0.162044 |

| Final evaluation | Test log loss | Test ROC-AUC | Test Brier |
|---|---:|---:|---:|
| CatBoost (validation selected) | 0.555606 | 0.772960 | 0.189889 |
| Training-prevalence baseline | 0.665676 | 0.500000 | 0.236391 |

**Interpretation candidate — pending author review:** this configuration selects
CatBoost by validation log loss, and its test loss is below the constant
baseline. CatBoost's validation loss is lower than its reported training loss;
train and held-out categorical transformations need not behave identically,
and sampling variation also matters. This table alone does not identify the
cause. The test loss exceeds validation loss; model selection and finite-sample
variation are possible explanations, not established findings. Changing both
the generator and representations prevents attributing the different selected
library solely to categorical handling.


**Limitations shared by both runs:** one seed, one synthetic generator per
scenario, one IID split, no uncertainty intervals, and no systematic tuning.
Stopping and candidate selection reuse validation labels. Nominal capacity
limits do not equalize shapes, initial scores, bins, categorical statistics,
or leaf support. The mixed case's 80 merchants do not test industrial scale,
entity cold starts, temporal drift, or memory efficiency. Probability metrics
do not establish calibrated deployment probabilities. There is no runtime,
GPU, fairness, causal, or production-system conclusion.

Both cases were rerun after API compatibility fixes; the recorded metrics and
selected rounds were unchanged. Full precision values, split indices, chosen
parameters, and environment are regenerable in ignored JSON records. No output
assets are selected for versioning.

### One-stump regularization demonstration

**Hypothesis:** with a fixed separable partition and zero initial logits,
increasing L2 penalty shrinks the Newton leaf scores.

**Configuration:** synthetic x values `[1, 2, 2.5, 3, 5, 6, 7, 8]`; four zeros
then four ones as labels; all initial logits zero; one tree; learning rate 1;
gamma and minimum child weight zero. Use the same Python and NumPy versions
as above. There is no train/validation/test selection in this arithmetic demo.

**Result:** all three fits select stored threshold 3 with the `x <= 3` rule.
Initial mean logistic loss is 0.693147.

| L2 penalty | Left score | Right score | Approximate gain | Exact mean loss after update |
|---:|---:|---:|---:|---:|
| 0 | -2.000000 | 2.000000 | 4.000000 | 0.126928 |
| 1 | -1.000000 | 1.000000 | 2.000000 | 0.313262 |
| 10 | -0.181818 | 0.181818 | 0.363636 | 0.606365 |

**Interpretation candidate — pending author review:** the unchanged partition
and smaller score magnitudes are consistent with the closed-form L2 shrinkage
formula. Higher penalty gives a smaller exact in-sample loss reduction in this
one-step construction.

**Limitation:** this is eight separable rows evaluated in sample. It establishes
neither improved generalization from L2 nor monotonic loss decrease for arbitrary
Newton steps. Approximate gain includes regularization, whereas mean logistic
loss does not; their numerical values are not directly comparable.


## Suggested follow-up experiments — not executed

- Hold a validation protocol fixed and compare learning-rate/round budgets.
- Vary LightGBM leaf limits and minimum support one at a time, recording train
  and validation gaps rather than assuming overfitting.
- Increase merchant cardinality and rarity; measure preprocessing memory,
  unknown-category rates, repeated runtime, and held-out behavior.
- Compare XGBoost native categorical partitions with the one-hot workflow.
- Use separate calibration data to inspect reliability and decision costs.

These suggestions have no measured results in this study. No visual assets,
GPU/distributed training, or production deployment are implemented.
