# ExtraTrees: split proposals, strength, and diversity

## What is randomized?

At every node, sample candidate features. For each selected nonconstant
numeric feature j, propose a threshold uniformly over that node's observed
range:

$$
t_j \sim U\left(\min_{i\in D}x_{ij},\max_{i\in D}x_{ij}\right).
$$

Evaluate the proposals against the labels, then retain the best valid one.
Random Forest instead searches thresholds within the selected features.
This is the defining difference; scikit-learn defaults to bootstrap for RF
and full-data trees for ET, but both expose the bootstrap option.
[Ensemble guide](https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees),
[RF API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html),
[ET API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).

For binary class frequency p, Gini impurity is G = 2p(1-p). A candidate
partition receives:

$$
G_{\mathrm{children}}
=\frac{n_L}{n}G_L+\frac{n_R}{n}G_R,
\qquad
\Delta G=G_{\mathrm{parent}}-G_{\mathrm{children}}.
$$

The scratch code minimizes weighted child impurity, equivalent to maximizing
gain because the parent is fixed. It uses x_j <= t on the left and x_j > t
on the right. Constant features and proposals violating the minimum leaf size
are skipped. One valid proposal is sufficient; the split need not be globally
optimal.

A deliberately useful edge case is XOR: every first split has zero gain,
but a second split can make the leaves pure. Rejecting all zero-gain splits
would hide that interaction. The scratch model allows them, subject to depth
and leaf limits.

## Variance reduction: state the probability space

At a **fixed input x**, consider scalar predictions T_m(x) over repeated
training datasets and algorithmic random draws. In general,

$$
\operatorname{Var}\left(\frac{1}{M}\sum_m T_m\right)
=\frac{1}{M^2}
\left[\sum_m\operatorname{Var}(T_m)
+2\sum_{i<j}\operatorname{Cov}(T_i,T_j)\right].
$$

If all trees have variance sigma^2 and equal pairwise correlation rho,
this simplifies to:

$$
\operatorname{Var}(\bar T)
=\sigma^2\left[\rho+\frac{1-\rho}{M}\right].
$$

The expression is exact under those assumptions. As M grows, only the
uncorrelated term disappears. Holding sigma^2 and rho fixed leaves a floor
rho sigma^2. Real trees need not satisfy equal variance or correlation, and
switching algorithms can change both quantities.

Under squared-error regression with independent zero-mean observation noise:

$$
\mathbb E[(Y-\hat f(x))^2]
=\operatorname{Bias}[\hat f(x)]^2+
\operatorname{Var}[\hat f(x)]+\sigma_\epsilon^2.
$$

Random proposals may increase bias while reducing covariance. Their net effect
is empirical. This squared-error identity is not a numerical decomposition of
classification accuracy, ROC-AUC, or log loss.

## What our diversity statistic does and does not measure

For M trees and N held-out rows, average unordered-pair disagreement is:

$$
D=\frac{1}{N\binom M2}
\sum_{n=1}^N\sum_{i<j}\mathbf 1[\hat y_i(x_n)\ne\hat y_j(x_n)].
$$

If row n has k_n positive votes, k_n(M-k_n) pairs disagree. Counting those
pairs computes D in O(MN) time without constructing an M by M by N array.
The tests compare this calculation with direct pair enumeration.

D describes hard predictions across this evaluation set. It is neither
prediction covariance over new training datasets nor pairwise correlation
of errors. Trees may disagree because they learned complementary signal or
because they became less useful. Average tree accuracy supplies a strength
diagnostic, but does not settle why disagreement changed. Estimating variance
would require repeated training samples evaluated at fixed inputs.

## Aggregation and implementation boundary

For class c, the ensemble probability is the mean of tree leaf frequencies:

$$
\hat p_c(x)=\frac1M\sum_m p_{m,c}(x),
\qquad
\hat y(x)=\arg\max_c\hat p_c(x).
$$

Hard voting discards the frequencies before averaging. For example, positive
probabilities 0.49, 0.49, and 0.99 produce a positive ensemble mean even though
two trees predict class 0. This is covered by a numerical test.

The scratch model uses a single seeded RNG, reset at fit, and samples features
again at every node. Every tree sees all training rows; different split draws
create different trees. A convex combination (1-u)low + u high, u uniform in
[0,1), implements the uniform threshold without subtracting extreme bounds.
Exact ties select class 0. Failed proposals stop a node instead of searching
additional features as optimized library implementations may do.

Scratch parameter conventions are deliberately narrow: integer max_features,
or None meaning floor(sqrt(p)) with a minimum of one. **In scikit-learn,
max_features=None means all p features**, so the meanings differ. Scratch
max_depth is a nonnegative integer; zero creates frequency-only roots.
Only binary 0/1 labels and finite nonempty dense inputs are accepted.
No claim of threshold, tree structure, or probability parity with sklearn
is made.

## Applications, choices, and common mistakes

A tabular quality classifier could combine retrieval similarity summaries,
source counts, document metadata, and latency measurements. Define its label
and ensure every feature exists at prediction time. Features from the final
answer can leak the outcome if the decision happens before generation.
Use group or temporal validation when cases or sessions repeat.

Smaller max_features can limit access to signal as well as change diversity.
Depth, minimum leaf size, and ensemble size also affect the strength/resource
trade-off. Adding trees cannot fix leakage or excessive bias. Uniform proposals
depend on the observed numeric range: outliers can consume much of it.
Positive affine rescaling preserves the ideal uniform proposal distribution;
a nonlinear monotone transform generally changes it, even if it preserves
feature order. Routine standardization is usually unnecessary for these
tree learners, but categorical representation and version-specific missing
data support still require care.

Log loss measures probability quality and can rank models differently from
accuracy or ROC-AUC. It combines several properties and does not by itself
establish calibration; inspect reliability and use independent calibration
data when probabilities drive decisions.
[Probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

OOB evaluation requires bootstrap omission; full-data trees provide no OOB
rows. Importance measures describe fitted predictive behavior, not causal
effects. Regression tree leaves average observed targets and do not provide
linear extrapolation outside the training target range.

| Variant | Main source of variation or correction | Aggregation |
|---|---|---|
| Bagged greedy trees | Bootstrap rows, often all features | Average predictions |
| Random Forest | Candidate features per node, optional bootstrap | Average predictions |
| ExtraTrees | Candidate features and random thresholds, optional bootstrap | Average predictions |
| Gradient boosting | Sequential updates against the current loss | Additive model |

The variants are design choices, not a ranking from weakest to strongest.
Threshold proposals can reduce search work, while actual runtime depends on
tree size, data, implementation, and parallelism.

## Executed synthetic comparison

**Execution date:** 2026-10-06. Executed by Codex; interpretations remain
**pending author review**.

**Hypothesis:** at fixed bootstrap settings, random thresholds may increase
tree disagreement and weaken individual trees; the ensemble need not improve
on every evaluation metric.

**Configuration:** the [example](example.py) generates 2,500 synthetic rows
with 20 features: 8 informative, 4 redundant, 0 repeated, and 8 remaining noise
features. There are two classes and two clusters per class, weights=None,
class_sep=1.0, flip_y=0.05, shuffle=True, and seed 42.
flip_y randomly reassigns labels; it is not a guarantee that exactly 5% change.
[Generator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html).

One stratified 75/25 split, seed 42, yields 1,875 training rows with class
counts [935, 940] and 625 held-out rows with counts [312, 313]. There is no
learned preprocessing. All four models use 120 trees, Gini, max_features="sqrt"
(four candidate features), max_depth=None, min_samples_split=2,
min_samples_leaf=2, n_jobs=1, and seed 42. Other estimator parameters retain
library defaults. The same seed reproduces each configuration; it does not
make RF and ET consume identical random draws.

**Environment:** Python 3.11.0rc2, NumPy 2.4.6, scikit-learn 1.9.0,
Windows-10-10.0.19045-SP0. Fit times cover model.fit only, with RF full-data,
RF bootstrap, ET full-data, and ET bootstrap run in that order. Measurements
were collected once, with focused validation running concurrently; scheduling
and cache effects limit timing interpretation.

**Result:** values rounded to four decimals from the actual run.

| Model | Bootstrap | Accuracy | ROC-AUC | Log loss | Mean tree accuracy | Disagreement | Fit seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| Random Forest | False | 0.8976 | 0.9597 | 0.2868 | 0.7920 | 0.2527 | 0.8832 |
| Random Forest | True | 0.8960 | 0.9533 | 0.3142 | 0.7677 | 0.2861 | 0.5611 |
| ExtraTrees | False | 0.9120 | 0.9588 | 0.3276 | 0.7585 | 0.2994 | 0.2612 |
| ExtraTrees | True | 0.8928 | 0.9532 | 0.3581 | 0.7291 | 0.3390 | 0.2471 |

Higher accuracy, AUC, and tree accuracy are better; lower log loss is better.
Disagreement has no universally preferred direction.

**Interpretation candidate -- author review required:** at each bootstrap
setting, ET had more hard-label disagreement and lower mean tree accuracy.
With bootstrap disabled, ET had higher ensemble accuracy but slightly lower
AUC and higher log loss than RF. With bootstrap enabled, ET had lower accuracy,
slightly lower AUC, and higher log loss. These mixed results are compatible
with a strength/diversity trade-off; they do not prove that ET reduced
training-set variance or error correlation. ET's measured fit times were
lower here, but this is not a reliable benchmark or a universal speed claim.

**Limitations:** one generator, dataset, split, and model seed; untuned
settings; no repeated-sampling variance estimate, confidence intervals,
calibration study, inference timing, or real deployment validation. Observing
these held-out values must not become a model-selection loop. Tuning would
require a development-validation scheme and a fresh final test boundary.

The full JSON run record stays in ignored outputs/comparison.json. Re-running
overwrites it, while this table records the execution above. No generated
asset is selected for versioning.

The scratch demo also ran: five depth-two trees with both candidate features
recovered the synthetic XOR training labels [0, 1, 1, 0], with positive
probabilities [0, 1, 1, 0]. This is a truth-table mechanics check and offers
no evidence about unseen rows.

## Suggested follow-ups -- not executed

- Repeat training samples and model seeds, evaluating the same fixed inputs,
  to study prediction variance separately from cross-row disagreement.
- Sweep max_features inside development folds and record both tree strength
  and ensemble metrics before touching a final test set.
- Vary irrelevant-feature count, label noise, and outliers one factor at a
  time; document each generator.
- Evaluate calibration on separate data, and repeat timings in isolation
  with fixed thread counts and several runs.
