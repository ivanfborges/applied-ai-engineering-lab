# Logistic regression: model, objective, and decision

## Odds explain why sigmoid appears

For a binary target, let p = P(Y=1 | x). The odds p/(1-p) compare event and
non-event probabilities: p=0.8 means odds 4, whereas p=0.2 means odds 0.25.
Taking the natural logarithm maps probabilities inside (0,1) onto the real
line. Assume this logit is an affine function of supplied features:

$$
\operatorname{logit}(p)=\log\frac{p}{1-p}=z=w^\top x+b.
$$

Solving p/(1-p)=exp(z) gives p=1/(1+exp(-z)). Thus the logistic link specifies
which quantity is linear. The sigmoid also gives
dp/dz=p(1-p), with maximum slope 1/4 at p=1/2.
It is not the only possible probability link; a probit model uses a normal CDF.

With other supplied predictors held fixed, adding one unit to x_j adds w_j
to the logit and multiplies odds by exp(w_j). For an odds multiplier r,

$$
p_{\mathrm{new}}=\frac{rp}{1-p+rp}.
$$

Doubling odds changes p=0.2 to 1/3 and p=0.8 to 8/9. It does not double
probability. Without interactions or nonlinear transformations,
dp/dx_j=w_j p(1-p); with interactions, the derivative must include those terms.
Conditioning on correlated predictors can make a hypothetical isolated change
unrealistic. Coefficients are associations under the chosen model.

For standardized inputs u_j=(x_j-mu_j)/s_j, fitted parameters convert back to

$$
w_{j,\mathrm{raw}}=w_{j,\mathrm{scaled}}/s_j,\qquad
b_{\mathrm{raw}}=b_{\mathrm{scaled}}-\sum_j w_{j,\mathrm{raw}}\mu_j.
$$

An odds ratio from a standardized coefficient is per training standard
deviation. Undoing the units preserves predictions, but does not undo the
effect of having trained with a penalty in standardized coordinates.

## Likelihood, cross-entropy, and stable computation

For conditionally independent observations with y_i in {0,1},

$$
L(w,b)=\prod_i p_i^{y_i}(1-p_i)^{1-y_i}.
$$

Taking the negative average log yields the binary cross-entropy:

$$
J_{\mathrm{BCE}}=-\frac{1}{n}\sum_i
[y_i\log p_i+(1-y_i)\log(1-p_i)].
$$

Unpenalized minimization is maximum likelihood. Accuracy assigns the same
error to many differently confident predictions; log loss distinguishes them.
For y=1, p=0.9 incurs about 0.105 nats, while p=0.01 incurs about 4.605.
Cross-entropy is a proper probability scoring rule, but a good aggregate
score alone does not demonstrate calibration in every subgroup.

Algebraically the per-row loss is softplus(z)-yz. For binary labels it can
also be written softplus((1-2y)z). The implementation uses the latter form:

```python
np.logaddexp(0.0, (1.0 - 2.0 * y) * logits)
```

This avoids exponent overflow, log(0), and cancellation when subtracting
large nearly equal values. The sigmoid uses exp(-abs(z)), with the appropriate
branch for each sign. It does not clip the score and alter the objective.
Floating-point probabilities can round to exactly 0 or 1, even though the
mathematical sigmoid is strictly inside that interval for finite scores.
The logit helper rejects endpoints. Invalid shapes, nonfinite data, and
nonbinary labels raise explicit errors; arithmetic overflow also raises.
NumPy's [logaddexp documentation](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html)
describes the underlying stable operation.

## Gradient, curvature, and regularization

This study minimizes

$$
J_\lambda(w,b)=J_{\mathrm{BCE}}(w,b)+\frac{\lambda}{2}\|w\|_2^2
$$

with no intercept penalty. Differentiating a row loss with respect to z
gives p-y, so

$$
\nabla_w J_\lambda=X^\top(p-y)/n+\lambda w,\qquad
\partial_b J_\lambda=\operatorname{mean}(p-y).
$$

The sigmoid derivative cancels with terms from cross-entropy. A confidently
wrong observation can still have a large score gradient even when the
sigmoid's own slope is small.

Using the augmented design A=[X,1], the Hessian is

$$
H=A^\top \operatorname{diag}(p_i(1-p_i))A/n
+\lambda\,\operatorname{diag}(1,\ldots,1,0).
$$

It is positive semidefinite, making this objective convex. Convexity does
not ensure uniqueness or existence of a finite unpenalized minimizer.
Redundant columns affect identifiability; complete or quasi-complete
separation can drive unpenalized coefficients toward infinity. Positive L2
and both classes stabilize this dense model; the scratch fit rejects
single-class data because an unpenalized intercept could otherwise diverge.

Since p_i(1-p_i) <= 1/4, a gradient Lipschitz bound is

$$
L \le \|A\|_2^2/(4n)+\lambda.
$$

The scratch solver takes the reciprocal of the right-hand side as its step
size. It starts at zero, records the initial and each updated objective, and
stops when the full gradient's infinity norm is at most the tolerance.
This conservative bound makes descent understandable without tuning a learning
rate. Computing a spectral norm and repeatedly validating dense arrays is
acceptable here; it is not intended for large or sparse production workloads.
A false convergence flag must be inspected. A small gradient under unpenalized
separation does not prove that a finite maximum-likelihood estimate exists.

For unweighted L2 training, the
[scikit-learn objective](https://scikit-learn.org/stable/modules/linear_model.html#binary-case)
uses a penalty coefficient 1/(nC), so this comparison sets
C=1/(n*lambda). Both implementations use the same scaled rows and an
unpenalized intercept. This mapping must be reconsidered for sample/class
weights or a different objective normalization. The example uses lbfgs and
its default L2 behavior; it does not pass the deprecated penalty argument.
The [estimator API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
documents solver and parameter compatibility.

L1 encourages exact zeros; L2 generally shrinks rather than selects; ElasticNet
combines both. Those penalties and their trade-offs continue
[Day 20](../20-regularization/). This implementation deliberately covers L2 only.

## From probability to a decision boundary

For the explicit convention predict positive when p >= t, where 0<t<1,

$$
w^\top x+b \ge \operatorname{logit}(t).
$$

At t=0.5, the right side is zero. For fixed nonzero w, different scalar
thresholds produce parallel hyperplanes; their displacement along the
positive normal is the change in logit(t) divided by ||w||. If w=0, the model
is constant and a usual separating hyperplane does not exist.
With polynomial features or embeddings, the boundary is linear in the
supplied representation and can be nonlinear in original inputs.

Lowering a threshold on fixed scores cannot remove predicted positives.
Recall and false-positive rate cannot decrease on that fixed labeled set.
Precision need not change monotonically. No refitting is needed; probabilities
and probability-based log loss stay unchanged. This study's equality rule is
explicit and need not match a library's exact tie convention.

The example predeclares t=0.2, 0.5, and 0.8 for illustration. Selecting among
them using the test table would turn that test set into development data.
Choose any operational threshold using separate validation data or appropriate
cross-validation, then evaluate the locked policy on untouched test data.
See [scikit-learn's threshold guidance](https://scikit-learn.org/stable/modules/classification_threshold.html).

## Assumptions, use, and limitations

The conditional mean model is Bernoulli with log-odds linear in the supplied
features. Features need not be Gaussian or independent of one another, and
Bernoulli variance p(1-p) is not constant. Conditional independence of rows
supports the simple likelihood product; repeated entities or time dependence
also require suitable split and uncertainty designs. Correct feature timing,
target definitions, and representative evaluation matter as much as the solver.

A linear model can serve as a compact baseline for text or a linear probe on
frozen embeddings. A routing model needs labels reflecting the actual routing
objective and features available before the routing decision. A grounding
classifier needs credible grounding labels; retrieval similarity alone is
not proof of groundedness. Inference cost, calibration, and usefulness of
these applications are not measured in this study.

Common mistakes include interpreting odds ratios causally, comparing
coefficients in incompatible units, scaling before splitting, treating 0.5
as universal, and assuming class weighting creates calibrated probabilities.
Weights change the fitted objective. A case-control sample or deployment
prevalence shift can also change probability meaning. Accuracy can hide a
rare positive class, while discrimination and calibration answer different
questions. No single result below establishes all three: ranking quality,
probability quality, and decision utility.

Compared with squared-error linear regression, logistic regression constrains
probabilities and models a Bernoulli conditional distribution. OLS can still
be computed for binary labels; Gaussian errors are not needed just to
compute its solution, but its usual error model is mismatched and outputs
can leave [0,1]. Linear SVMs use a margin loss and require additional work for
probabilities. Trees can learn interactions and piecewise boundaries without
the same linear feature model. Their relative quality must be tested.

A sigmoid output layer on a fixed representation is logistic regression.
Multinomial logistic regression instead normalizes multiple class scores
with softmax for mutually exclusive classes. Independent sigmoids suit
multiple binary labels and do not force probabilities to sum to one.

## Executed experiments

These runs were executed by Codex during implementation. Interpretations
below are candidates for author review, not the author's conclusions.
They are synthetic demonstrations, not benchmarks.

Environment: Windows, Python 3.11.0rc2, NumPy 2.4.6, scikit-learn 1.9.0.
The environment reports a Python prerelease; dependencies remain centralized
in the repository configuration. Last digits may vary by platform or version.

### 1. Matched objectives and fixed threshold decisions

**Hypothesis:** the two optimizers should produce nearly identical
probabilities when optimizing the same objective. Threshold changes should
change the positive set without changing those probabilities.

**Configuration:** run example.py from the repository root using the README
command. NumPy default_rng(21) generates 1,200 independent two-dimensional
standard normal latent vectors. Their logits are -0.4+1.4*u1-1.1*u2;
labels are Bernoulli draws from their sigmoid probabilities.
Raw X=latent*[2,10]+[3,-4], with arbitrary feature units and no domain labels.
A stratified split with random_state=21 reserves 25% (300 rows).
StandardScaler fits on the 900 training rows only.
Both models use lambda=0.02, corresponding to C=0.05555556.
Scikit-learn uses lbfgs, max_iter=2000, tol=1e-10.
Scratch uses zero initialization, the curvature-based step, max_iter=10,000,
and gradient tolerance 1e-8. No class weights, parameter search, or calibration
fit is used. Thresholds are fixed at 0.2, 0.5, and 0.8 before evaluation.

**Results:** training positive prevalence 0.4356; test prevalence 0.4367.

| Quantity | Observed result |
|---|---:|
| Scikit-learn test log loss | 0.47592365 |
| Scratch test log loss | 0.47592365 |
| Constant train-prevalence baseline test log loss | 0.68510588 |
| Maximum absolute test probability difference | 1.191e-08 |
| Scratch objective, initial to final | 0.69314718 to 0.54838071 |
| Scratch updates | 26 |
| Final gradient infinity norm | 6.674e-09 |

The fitted original-unit coefficients were approximately [0.5049875,
-0.07735233], with intercept -2.174307 and odds ratios [1.6569648, 0.92556369].
They are penalized estimates on sampled labels, not exact recovery of the
generating parameters.

| Threshold | Positive predictions | Accuracy | Precision | Recall | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 240 | 0.6167 | 0.5333 | 0.9771 | 57 | 112 | 3 | 128 |
| 0.5 | 123 | 0.7800 | 0.7642 | 0.7176 | 140 | 29 | 37 | 94 |
| 0.8 | 22 | 0.6367 | 1.0000 | 0.1679 | 169 | 0 | 109 | 22 |

**Interpretation candidate — author review:** the small probability difference
supports the implementation and objective mapping on this case. The threshold
table illustrates how one probability model supports different decisions.
The feature model outperformed a constant baseline by log loss on this split.

**Limitations:** one seed and one split; data intentionally follow the model
family. No calibration, robustness, latency, confidence intervals, rare-event,
or nonlinear-model comparison was run. Precision 1.0 represents only 22
selected positives here, not guaranteed future precision. No threshold is
endorsed for an application.

### 2. Standalone optimization demonstration

**Hypothesis:** the conservative gradient step should decrease the penalized
training objective and reach the specified gradient tolerance.

**Configuration:** run from_scratch.py; the same 1,200 synthetic rows are
standardized using all rows and used entirely for training. Lambda=0.02,
zero initialization, max_iter=10,000, tol=1e-8. There is no test set in this
optimization-only demonstration.

**Results:** objective 0.69314718 to 0.53359838; 27 updates; converged=True;
gradient infinity norm 6.001e-09; learning rate approximately 3.579861.

**Interpretation candidate — author review:** this run is consistent with
descent and convergence under the stated step rule.

**Limitations:** training-objective evidence only. It does not measure
generalization, demonstrate convergence for every dataset, or validate a
deployment policy. Its final objective is not directly comparable to the
900-row training run as a model-quality claim.

## Visual experiments

The subsequent [visual lab](VISUAL_GUIDE.md) adds executed threshold,
optimization, regularization, representation, and calibration experiments.
Its datasets and configurations differ from the two runs above; those
earlier results and limitations refer only to their original scripts.

## Suggested follow-ups — not executed

- Extend the visual lab's fixed-degree polynomial comparison with degree
  selection inside development folds.
- Change prevalence and compare class weighting with threshold changes;
  evaluate calibration on representative held-out data.
- Examine unregularized separation by tracking coefficient norms and loss.
- Select a threshold against declared costs or capacity on validation data,
  then report its locked test performance.

Author review should focus on interpretation of coefficients, probability
quality, the threshold example, and the limits of any application claims.