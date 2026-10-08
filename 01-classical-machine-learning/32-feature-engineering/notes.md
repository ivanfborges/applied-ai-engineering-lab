# Representation design and information availability

## What changes when a feature changes?

For a score `f_theta(phi(X))`, both the representation `phi` and the model family
determine what can be learned. An invertible rescaling preserves information
but changes optimization and regularization geometry. Binning discards within-bin
resolution. A product is determined by its two inputs, yet can expose structure
an additive model cannot represent.

Aggregating a separate transaction table brings historical information into a
customer-only table. The source information was available to the system; it was
absent from the baseline feature matrix. An improvement after adding history
therefore is not solely evidence about a better transform of identical inputs.

## Encoding choices

| Encoding | Useful situation | Assumption or cost |
|---|---|---|
| One-hot | Low-cardinality nominal variables | More columns; unseen-category policy required |
| Ordinal | Genuine ordered categories | A linear model assumes equal step effects for integer codes |
| Frequency | Category prevalence is informative | Equal-frequency categories collide; fit counts on train only |
| Target statistics | High-cardinality predictive categories | Labels enter the transform; requires leakage controls |
| Hashing | Bounded feature width | Collisions and weaker interpretability |

For category `c`, frequency encoding is `N_c / N`. A smoothed target statistic is

$$
TE(c)=\frac{N_c\bar y_c+\alpha\bar y}{N_c+\alpha},\qquad \alpha\geq 0.
$$

Smoothing alone does not prevent a row's label from influencing its own feature.
Training rows need out-of-fold or temporally prior encodings; held-out rows use
statistics fitted on the relevant training partition. A generic random internal
split can still violate entity or temporal boundaries.
Scikit-learn's target encoder cross-fits `fit_transform`; it differs from
`fit(...).transform(...)`. This example implements one-hot encoding only.
See the [official TargetEncoder API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html).

The example retains all category indicators under L2 regularization. With an
intercept they are redundant, but the penalized model can still fit predictions.
Coefficients should not be read as uniquely identified unpenalized category
effects. Unknown region/channel values map to all-zero blocks; this is a defined
fallback, not evidence that unseen groups will be predicted accurately. Dense
encoding is appropriate for the seven indicators here, not millions of IDs.

## Scaling and the penalty

Training-only standardization uses population variance:

$$
\mu_j=\frac{1}{n}\sum_i x_{ij},\qquad
s_j^2=\frac{1}{n}\sum_i(x_{ij}-\mu_j)^2,\qquad
z_{ij}=\frac{x_{ij}-\mu_j}{s_j}.
$$

For constant training columns the scratch code uses `scale=1`, matching the
library convention. Training values then become zero; a different holdout value
need not become zero. Standardization does not imply Gaussianity or constrain
holdout values to a range.

If `x'_j=a_j x_j`, preserving the same linear score requires
`beta'_j=beta_j/a_j`. Consequently an L2 penalty on the new coefficient has a
different cost. Scaling matters even if an optimizer converges in both units.
For KNN and kernels it also changes the metric. Raw min-max scaling uses
`(x-min)/(max-min)`; held-out values can exceed the fitted range. Median/IQR
scaling reduces dependence on extreme observations but does not remove them.

Trees generally do not require changes of units: positive affine transforms
preserve ordering and the partitions available to exact threshold search.
Finite precision, histogram binning, and arbitrary nonlinear transforms can
affect actual implementations. Binary indicators can also be scaled, but doing
so changes their penalty weight; leaving them unscaled is a deliberate choice,
not a prohibition.

## Bins as a basis for nonlinear behavior

Given ordered thresholds `t_1,...,t_(K-1)`, the bin is the number of thresholds
less than or equal to `x`. One-hot bins let a linear score assign separate levels
to intervals. An integer bin ID instead imposes equal score increments across
successive codes.

Uniform bins use equal numerical widths; quantile bins target equal counts;
one-dimensional k-means bins adapt to clusters. Quantiles are not guaranteed to
balance counts when observations tie. Boundaries are learned on training rows,
then reused unchanged. The library example explicitly uses linear quantiles
and disables subsampling for this small dataset.
See [KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html).

The scratch implementation removes duplicate thresholds and thresholds at the
observed extremes. Equality goes to the right bin; values outside the training
range go to the outer bins. Constant input produces one bin. This simplified
tie policy is not an exact replacement for the library's handling of narrow
intervals.

The example retains raw age plus bin indicators. Raw age permits a common
within-bin linear slope; indicators add jumps. This is more expressive than
bins alone but discontinuous at boundaries. Four bins are prespecified, not
selected as optimal. Smooth splines are another candidate when abrupt jumps
are undesirable.

## Interaction means a conditional slope

A logistic score with a product is

$$
\eta=\beta_0+\beta_1x_1+\beta_2x_2+\beta_3x_1x_2,\qquad
p=\sigma(\eta).
$$

Then `d eta / d x_1 = beta_1 + beta_3 x_2`, while
`d p / d x_1 = p(1-p)(beta_1 + beta_3 x_2)`. The logit and probability slopes
are different. The model remains linear in its coefficients.

The example centers and scales income/balance, multiplies them, then scales
the product using training statistics. With the main effects and intercept,
a centered product spans the same unpenalized score space as a raw product:
expanding `(x_1-mu_1)(x_2-mu_2)` yields the raw product, main effects, and a
constant. Their penalized fits need not coincide.

Only the product is added. Including the original columns again through a
polynomial transformer would duplicate main effects: splitting coefficient
`b` across identical columns as `b/2,b/2` reduces their squared penalty from
`b^2` to `b^2/2`. Such duplicates alter regularization, not merely storage.
Pairwise interactions among `p` columns add `p(p-1)/2` terms, so choose a
small set with a substantive hypothesis. Ratios require a denominator policy;
the example avoids them.

## Historical aggregation contract

For customer `i` and prediction time `t_i`, define

$$
H_i=\{e: e.customer=i,\ t_i-W\leq e.event\_time<t_i,
\ e.available\_at<t_i\}.
$$

For amounts in `H_i`, compute count, sum, mean, and maximum. The event window
includes its start and excludes the prediction instant. Availability is also
strictly before prediction. Other contracts can permit equality; they must be
specified and tested. An event occurring yesterday but arriving tomorrow is
unavailable today.

No-history sums/counts are zero. Undefined means/maxima are assigned zero and
accompanied by count zero, distinguishing absent history from observed zero
amounts. Signed refund amounts are supported; the synthetic generator emits
positive amounts. Count conveys the amount of evidence behind a mean, not a
confidence interval. Sum, count, and mean are related and can introduce redundant
predictors; the fixed example includes all four for inspection.

The helper normalizes timestamps to UTC; naive timestamps are interpreted as
UTC. Convert local business timestamps before using this helper. It accepts
exactly one snapshot per customer, preserves snapshot row order/index, ignores
events for other customers, and rejects null IDs, missing timestamps,
nonfinite amounts, and availability before event time. Inputs must already be
deduplicated according to the source's event identity contract.

The in-memory join is educational. It does not implement event revisions,
backfilled customer attributes, multiple cutoffs per customer, rolling feature
stores, or low-latency serving. The synthetic snapshots are complete and assumed
known at their cutoff.

## Two leakage boundaries

Customer-local historical summaries can be computed before the split because
they use neither labels, future/late events, nor other customers' distributions.
This does not justify computing category means, global thresholds, imputers, or
target statistics before splitting. Those parameters belong inside training
partitions and, when evaluating with cross-validation, inside each fold.
The [official leakage guide](https://scikit-learn.org/stable/common_pitfalls.html)
explains train-only fitting and pipelines.

The tests mutate holdout values/categories and confirm unchanged means, scales,
bin boundaries, vocabularies, interaction scaling, and training representations.
A separate three-fold check verifies fold-local fitting. Event tests check
window endpoints, late availability, future-event invariance, and absent history.

Repeated snapshots of a customer require evaluation aligned with deployment:
group separation for new-customer generalization, or temporally valid evaluation
for future predictions of existing customers. This experiment uses disjoint
customers at one cutoff, not a temporal deployment simulation.

## Executed experiments

### Fixed representation comparison

**Hypothesis:** adding admissible history and explicit nonlinear terms may
improve ranking for a fixed logistic-regression model.

**Configuration:** executed by Codex on 2026-10-07; seed 42; 2,500 synthetic
customers; 20,152 events with offsets from -50 to +10 days around the UTC
2026-09-01 cutoff, and availability delays of 0–4 days. Stratified 75/25
customer split: 1,875 train and 625 holdout rows. Holdout prevalence 0.3648.
No tuning or model selection. Each variant uses `C=1`, `lbfgs`, and
`max_iter=2000`; four linear-quantile age bins with no subsampling. Numeric
scalers and category vocabularies fit on training rows only.

The artificial target logit is

$$
-2+a(age)+0.35\,income\,balance/10^9
+0.004\,txn\_mean\_{30d}+0.15\,1[channel=mobile],
$$

where `a=0.8` for age below 25, `0.6` for age at least 60, and `-0.3`
otherwise. Binary labels are Bernoulli draws from its sigmoid.

Environment: Python 3.11.0rc2, NumPy 2.4.6, pandas 3.0.3,
scikit-learn 1.9.0. A second run from the topic directory produced an identical
JSON record (SHA-256 comparison). Execution used the existing repository virtual environment;
no fresh-environment installation was tested.

| Variant | Features | ROC-AUC | Average precision | Solver iterations |
|---|---:|---:|---:|---:|
| Baseline | 10 | 0.667893 | 0.535180 | 7 |
| Added aggregates | 14 | 0.686685 | 0.600378 | 17 |
| Added bins and interaction | 19 | 0.705268 | 0.606240 | 25 |

AP is the recall-weighted average of precision increments, not trapezoidal
integration of the PR curve. Both reported metrics assess ranking; neither
establishes calibration or an operational threshold.

**Interpretation candidate — pending author review:** in this generator and
split, added history improved both ranking metrics, and the bundled nonlinear
terms improved them further. The comparison is consistent with exposing the
generator's signal to an additive model.

**Limitations:** deliberately favorable target specification; one seed and split;
no uncertainty estimate; no separate bin/interaction ablation; fixed C across
different feature spaces; no real-world labels or validated deployment.
Do not rank feature techniques generally or attribute the full improvement to
representation alone. Do not reuse this holdout for feature selection.

### Educational numerical demonstration

**Hypothesis:** training-fitted numerical transforms should match their
specified formulas on new rows.

**Configuration:** the scratch script uses four training rows
`(25,50000), (40,80000), (55,120000), (30,65000)` and holdout rows
`(35,70000), (60,130000)`; population scaling and four linear-quantile bins.

**Observed result:** means `[37.5,78750]`; income thresholds
`[61250,72500,90000]`; holdout bin IDs `[1,3]`; products of standardized
holdout columns approximately `[0.07323923,3.86075365]`.
The focused tests separately check StandardScaler parity and library bin
assignments on distinct values, plus the documented tie policy.

**Interpretation candidate — pending author review:** the printed transforms
illustrate reuse of training parameters. This is an arithmetic demonstration,
not evidence that these transforms improve prediction. **Limitation:** tiny,
dense, complete arrays; simplified bins; no sparse or missing-value handling.

## Suggested experiments — not executed

- Remove only the interaction, then only the bins, using development CV to
  isolate their effects. Reserve a new test set if choices change.
- Compare uniform, quantile, and k-means bins; inspect ties and interval counts.
- Compare scaled/unscaled logistic regression and a fixed tree ensemble, with
  convergence diagnostics as well as validation scores.
- Repeat across prespecified seeds and report variation rather than one score.
- Add deliberately future-contaminated aggregates as a labeled negative control,
  never as a valid feature pipeline.

Manual review should assess the generator's favorable assumptions, the
appropriateness of raw age plus bins, event-availability semantics, and the
scope of the measured interpretation.
