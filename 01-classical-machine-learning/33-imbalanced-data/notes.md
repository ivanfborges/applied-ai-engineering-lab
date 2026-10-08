# Imbalance: learning, probabilities, and decisions

## Start with the positive count and the action

Let positive prevalence be $\pi=P(Y=1)$. A small proportion can still provide
many positive examples in a large dataset; the same proportion in a tiny
dataset may leave too few positives for reliable fitting or evaluation.
Label quality, minority subgroups, separation, and delayed outcomes matter.

An always-negative classifier has accuracy $1-\pi$ and zero positive recall.
Its high accuracy does not establish usefulness for finding rare events.
Choose the positive class according to the action, even when it is not the
least frequent label. The educational resamplers identify whichever binary
class has fewer rows; the example's positive class is label 1.

Document escalation may require recall subject to a review budget.
Fraud triage may use expected error cost or precision/recall at a fixed capacity.
Churn ranking does not establish that contacting a high-risk customer will
prevent churn: that is a separate intervention question.

## Changing the training objective

For binary logistic probabilities $p_i=\sigma(z_i)$, a class-weighted objective
has data-loss term

$$
L=-\sum_i\left[w_1 y_i\log p_i+
w_0(1-y_i)\log(1-p_i)\right],
$$

plus the chosen regularization penalty. The balanced heuristic is
$w_k=N/(K N_k)$. With 90 negatives and 10 positives, binary weights are
$w_0=100/180$ and $w_1=5$; aggregate weight is equal across classes.
These weights reflect frequency, not an elicited error-cost model.
See [scikit-learn class weights](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html).

A useful first-principles calculation: at a fixed $x$ with true positive
probability $p$, minimizing expected weighted log loss over an unrestricted
probability $q$ gives

$$
q=\frac{w_1p}{w_1p+w_0(1-p)},\qquad
\frac{q}{1-q}=\frac{w_1}{w_0}\frac{p}{1-p}.
$$

This follows by differentiating $-w_1p\log q-w_0(1-p)\log(1-q)$.
It explains why a weighted model's output need not be a posterior under the
natural prevalence. Finite samples, misspecification, and penalties prevent
treating this identity as an exact fitted-model correction. A threshold of
0.5 on ideal $q$ corresponds to $p\ge w_0/(w_0+w_1)$.

Duplicating each row a specified integer number of times is equivalent to
that row's weight in a summed data-loss term. Random duplication produces
random multiplicities, and implementations differ in loss normalization,
penalties, and optimization. Equal nominal `C` across different sample totals
does not isolate the effect of minority exposure.

## Changing the training distribution

| Method | What is retained or added | Main trade-off |
|---|---|---|
| Random oversampling | All original rows plus minority duplicates | More exposure without new independent information |
| Random undersampling | All minority rows plus a majority subset | Smaller fitting set, with discarded majority coverage |
| Numerical SMOTE | All original rows plus minority interpolations | Adds local geometry assumptions and can amplify noise |

SMOTE chooses a minority anchor $x_i$, one of its $k$ minority nearest neighbors
$x_j$, and $\lambda\sim U[0,1)$:

$$
x_{new}=x_i+\lambda(x_j-x_i).
$$

The same scalar interpolates all coordinates, so the new row lies on a line
segment, not in an arbitrary coordinate-wise box. Nearby endpoints do not
guarantee that the segment represents a valid minority observation.
Scaling changes which rows are neighbors. Correlated, redundant, or irrelevant
dimensions also change Euclidean geometry.
The mechanism originates in [Chawla et al. (2002)](https://www.jair.org/index.php/jair/article/view/10302).

The implementation excludes the anchor by index, including when duplicate
coordinates tie. It uses stable index order for tied distances, samples anchors
uniformly, and returns only generated rows. It requires at least $k+1$
minority rows. It allocates a dense pairwise distance matrix and a distance
difference tensor, so memory grows quadratically in minority count; it is for
small educational examples.

Interpolation of integer category codes or one-hot indicators can create
invalid categories. Disconnected clusters, outliers, noisy labels, and
high-dimensional neighborhoods can produce implausible rows. Mixed-feature
methods such as SMOTENC have different semantics; changing the sampler does
not prove domain validity.
See [imbalanced-learn oversampling](https://imbalanced-learn.org/stable/over_sampling.html).

## Metrics answer different questions

With confusion counts TP, FP, FN, and TN:

| Quantity | Definition | Question |
|---|---|---|
| Precision | $TP/(TP+FP)$ | What fraction of alerts are correct? |
| Recall / TPR | $TP/(TP+FN)$ | What fraction of positives are found? |
| FPR | $FP/(FP+TN)$ | What fraction of negatives become alerts? |
| Balanced accuracy | $(TPR+TNR)/2$ | How well is each class recognized equally? |
| Alert rate | $(TP+FP)/N$ | How much downstream work is created? |
| Error cost | $C_{FP}FP+C_{FN}FN$ | What loss follows from this policy? |

The example defines precision as zero when there are no alerts.
$F_\beta=(1+\beta^2)TP/((1+\beta^2)TP+\beta^2FN+FP)$;
$\beta=2$ emphasizes recall. This ratio does not encode a fixed monetary
false-negative/false-positive cost ratio, and it omits true negatives.

ROC-AUC measures ranking, including half credit for tied positive/negative
scores. It does not encode alert burden. If there are 100,000 negatives,
a 1% FPR creates 1,000 false alerts. Even 90 true positives would then give
precision $90/(90+1000)\approx0.0826$. These are illustrative counts, not
experiment measurements.

Under a fixed class-conditional score distribution,

$$
\operatorname{Precision}
=\frac{\pi\,TPR}{\pi\,TPR+(1-\pi)FPR}.
$$

This Bayes identity explains how precision changes with prevalence even when
TPR and FPR stay fixed. Real drift can change those rates too.

Average precision summarizes precision using recall increments:

$$
AP=\sum_n(R_n-R_{n-1})P_n.
$$

It differs from trapezoidal integration of the PR curve. A constant score
has AP equal to prevalence under this definition; finite-sample random rankings
can fluctuate. Report the prevalence and exact summary definition when
comparing datasets.
See [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

## Choosing a decision policy

With calibrated deployment probability $p$, constant error costs, zero cost
for correct decisions, and no capacity constraints,

$$
\operatorname{Cost}(+)=(1-p)C_{FP},\qquad
\operatorname{Cost}(-)=pC_{FN}.
$$

Choosing positive when its cost is no larger gives

$$
p\ge t^*=\frac{C_{FP}}{C_{FP}+C_{FN}}.
$$

For the illustrative costs 1 and 10, $t^*=1/11\approx0.0909$.
This analytical policy requires appropriate probabilities and cost assumptions.

The example instead minimizes **empirical validation cost** for each model.
It does not assume calibrated probabilities. It evaluates every distinct
validation score as a threshold, plus the next floating-point value above the
maximum for the no-alert policy. The minimum score covers all-alert decisions.
Prediction uses `score >= threshold`; tied scores always receive one decision.
An all-negative policy may require a threshold slightly greater than 1 when
a supplied score is exactly 1. Cost ties choose the highest threshold, yielding
the fewest validation alerts among minimizing policies.

Searching $m$ distinct scores on $n$ validation rows costs $O(mn)$ here.
A sorted cumulative-count implementation would scale better. The exhaustive
version is intentionally inspectable. The selected validation cost is optimistic
because it was minimized on those rows; test cost estimates behavior of the
frozen policy, within the synthetic sampling assumptions.
Threshold selection is a model-design step.
See [scikit-learn threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html).

Lowering a threshold cannot reduce recall or alert count on fixed scores.
Precision need not move monotonically. Neither ROC-AUC nor AP changes merely
because the decision threshold changes. Thresholding cannot repair a poor
ranking.

A top-$K$ policy can guarantee a review count but needs a tie rule, per-batch
definition, and delay semantics. A probability threshold does not guarantee
a fixed workload when volume or prevalence shifts. No capacity constraint is
implemented in this experiment.

## Probabilities and validation boundaries

Binary Brier loss is $N^{-1}\sum_i(p_i-y_i)^2$. It measures probability quality,
combining calibration, discrimination, and outcome uncertainty; a lower value
does not establish better calibration alone. Reliability bins need enough
positive outcomes to interpret. This example measures Brier loss but does not
fit a calibrator.
See [probability calibration](https://scikit-learn.org/stable/modules/calibration.html).

Under pure prior shift, if $P(X\mid Y)$ stays fixed and $q$ is the true posterior
under a resampled prior $\pi_s$, deployment odds at prior $\pi$ satisfy

$$
\frac{p}{1-p}
=\frac{q}{1-q}\,
\frac{\pi/(1-\pi)}{\pi_s/(1-\pi_s)}.
$$

This is a Bayes derivation under explicit assumptions. SMOTE changes minority
feature distribution; misspecified weighted models also need not satisfy this
posterior identity. Blind prior correction is therefore not implemented.

Split before fitting scaling, class-frequency weights, and resampling.
In cross-validation, repeat those operations inside each training fold.
Synthetic relatives or duplicates spread across folds contaminate evaluation.
An imbalanced-learn pipeline can place sampling inside fitting; a standard
scikit-learn transform pipeline does not itself support sampler semantics.
See [resampling pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html).

Keep validation and test representative of the intended deployment population.
Stratification stabilizes class counts for this IID generator; it does not solve
entity overlap, chronological availability, or temporal drift. For a time-based
application, define time and group boundaries before considering imbalance.
If calibration, model choice, and threshold choice all use scarce validation
positives, use an appropriate cross-fitting or nested design and account for
selection uncertainty.

## Executed experiment

**Executed by Codex on 2026-10-07. Interpretation candidates: pending author review.**

**Hypothesis.** Training interventions and cost threshold selection may change
positive recall, alert volume, and error cost differently; balancing need not
improve ranking.

**Configuration.** `example.py` generated 6,000 synthetic rows with 12 features
(5 informative, 2 redundant), two clusters per class, class separation 1.2,
requested mixture weights [0.97, 0.03], label-flip parameter 0.01, and generator
seed 42. Both stratified split seeds were 42. Training had [3480, 120] rows by
class; validation and test each had [1160, 40]. Scaling fit on the original 3,600
training rows. Logistic regression used `C=1`, `lbfgs`, `max_iter=2000`.
Balanced weights left counts unchanged; random oversampling and SMOTE fit on
[3480, 3480], while undersampling fit on [120, 120]. Sampler seed was 42; SMOTE
used $k=5$. Costs were $C_{FP}=1$, $C_{FN}=10$. Each strategy's threshold was
selected independently on validation, with no subsequent refit.

Environment: Python 3.11.0rc2, NumPy 2.4.6, scikit-learn 1.9.0.
Version or numerical changes can alter fitted scores near thresholds.
Commands, from the repository root:

```bash
python 01-classical-machine-learning/33-imbalanced-data/example.py
python 01-classical-machine-learning/33-imbalanced-data/from_scratch.py
```

**Results.** Test measurements below are rounded; all costs use 1,200 test rows.

| Training | Policy | Threshold | AP | ROC-AUC | Precision | Recall | Alert rate | Cost | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always negative | Fixed | 0.5000 | 0.0333 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 400 | 0.0333 |
| Baseline | Fixed | 0.5000 | 0.5841 | 0.8881 | 0.8462 | 0.2750 | 0.0108 | 292 | 0.0204 |
| Baseline | Validation cost | 0.1630 | 0.5841 | 0.8881 | 0.4333 | 0.6500 | 0.0500 | 174 | 0.0204 |
| Balanced weights | Fixed | 0.5000 | 0.4615 | 0.8995 | 0.1459 | 0.8500 | 0.1942 | 259 | 0.1284 |
| Balanced weights | Validation cost | 0.7235 | 0.4615 | 0.8995 | 0.2871 | 0.7250 | 0.0842 | 182 | 0.1284 |
| Random over | Fixed | 0.5000 | 0.4519 | 0.8987 | 0.1429 | 0.8250 | 0.1925 | 268 | 0.1292 |
| Random over | Validation cost | 0.7239 | 0.4519 | 0.8987 | 0.2843 | 0.7250 | 0.0850 | 183 | 0.1292 |
| Random under | Fixed | 0.5000 | 0.4563 | 0.8952 | 0.1398 | 0.8250 | 0.1967 | 273 | 0.1299 |
| Random under | Validation cost | 0.7515 | 0.4563 | 0.8952 | 0.2989 | 0.6500 | 0.0725 | 201 | 0.1299 |
| Educational SMOTE | Fixed | 0.5000 | 0.4496 | 0.9037 | 0.1518 | 0.8500 | 0.1867 | 250 | 0.1201 |
| Educational SMOTE | Validation cost | 0.7257 | 0.4496 | 0.9037 | 0.2703 | 0.7500 | 0.0925 | 181 | 0.1201 |

The always-negative classifier had test accuracy $1160/1200=0.9667$ and recall
zero. In the scratch demonstration, the toy scores [0.1, 0.3, 0.4, 0.9] with
labels [0, 1, 0, 1] selected threshold 0.3, one false positive, no false negatives,
and cost 1. The interpolation demo generated four rows lying on the defined
synthetic line; it estimates no predictive benefit.

**Interpretation candidates — author review required.**

- In this split, baseline threshold selection traded additional alerts for
  higher recall and lower illustrative error cost, without changing ranking.
- Balancing raised fixed-threshold recall, but it did not improve AP in this
  comparison. ROC-AUC and AP ordered training interventions differently.
- Weighted/resampled probability outputs had higher Brier loss here. The
  effective training prior is a possible explanation, but Brier loss alone
  cannot diagnose calibration or establish that mechanism.

**Limitations.** One IID synthetic seed, 40 test positives, and no uncertainty
interval or repeated-seed comparison. The target is not a real business process.
Fixed `C` across changing fit sizes/weight totals confounds minority exposure
with regularization effects. SMOTE plausibility is not established by this
generator. Costs are constant illustrative units; capacity, action benefit,
delayed labels, and drift are absent. No calibrator was fitted. Test rows are
descriptive comparisons of prespecified models, not justification for a
test-selected winner. Validation-optimal thresholds need not minimize test cost.

The JSON report regenerates under ignored `outputs/` and is not a public linked
asset. The small results table above is the intentional public record.

## Suggested extensions — not executed

Repeat the comparison across seeds with paired uncertainty estimates. Vary
positive counts and class overlap independently. Use a reviewed calibration
design before comparing analytical cost thresholds. Evaluate recall at fixed
review capacity with explicit ties. For mixed features, compare a categorical
sampler only after specifying valid feature combinations. Any new run needs
its own hypothesis, configuration, measurements, interpretation candidate, and
limitations.
