# Support Vector Machines: geometry, optimization, and evidence

## Margin geometry

For binary labels $y_i\in\{-1,+1\}$, a linear score is
$f(x)=w^Tx+b$. The sign selects the class; the scratch model assigns an exact
zero score to `+1`. A functional margin $m_i=y_i f(x_i)$ changes if both $w$
and $b$ are multiplied by a positive constant, even though the boundary does
not change. It is therefore not itself a geometric distance or probability.

For nonzero $w$, project a displacement onto the normal vector $w/\|w\|$:

$$
\text{signed geometric margin}_i=\frac{y_i(w^Tx_i+b)}{\|w\|},
\qquad
\text{distance to hyperplane}=\frac{|w^Tx+b|}{\|w\|}.
$$

Canonical constraints set the closest separating points at functional margin
one. The parallel hyperplanes $f(x)=+1$ and $f(x)=-1$ are separated by
$2/\|w\|$; each lies $1/\|w\|$ from the boundary. Minimizing
$\|w\|^2/2$ under separation constraints maximizes that corridor. If $w=0$,
these distance formulas are undefined.

Hard-margin SVM requires linear separability:

$$
\min_{w,b}\frac12\|w\|^2
\quad\text{subject to } y_i f(x_i)\ge1.
$$

Overlapping classes make these constraints infeasible. Even separable data
can contain outliers that make the hard-margin solution fragile. A large
margin controls sensitivity within the chosen feature geometry; it does not
guarantee good generalization under label noise or distribution shift.

## Soft margin and hinge loss

With nonnegative slack variables, use $y_i f(x_i)\ge1-\xi_i$ and minimize
$\|w\|^2/2+C_{\mathrm{sum}}\sum_i\xi_i$. Eliminating slack yields

$$
J_{\mathrm{sum}}(w,b)=\frac12\|w\|^2+
C_{\mathrm{sum}}\sum_i\max(0,1-y_i f(x_i)).
$$

| Functional margin | Classification and hinge loss |
|---|---|
| $m_i\ge1$ | Correct, at or beyond the margin; loss zero |
| $0<m_i<1$ | Correct, inside the margin; loss $1-m_i$ |
| $m_i=0$ | On the decision boundary; loss one |
| $m_i<0$ | Incorrect; loss greater than one |

Increasing `C` makes violations more expensive relative to the norm penalty.
It weakens regularization in this convention. A smaller `C` can underfit;
a larger `C` can react strongly to noise. These are tendencies rather than a
guarantee that a particular test metric, contour, or support count changes
monotonically. The loss is convex but nonsmooth at $m_i=1$. Convexity removes
spurious local optima; finite iterative training can still be inaccurate,
and intercepts or dual representations need not always be unique.

### Normalization must match the implementation

The scratch model uses **mean hinge**:

$$
J_{\mathrm{mean}}=\frac12\|w\|^2+
\frac{C_{\mathrm{mean}}}{n}\sum_i\max(0,1-y_i f(x_i)).
$$

To compare with the summed-hinge `SVC` objective, set
$C_{\mathrm{sum}}=C_{\mathrm{mean}}/n$. Dividing $J_{\mathrm{sum}}$ by
$C_{\mathrm{sum}}n$ also gives
$\lambda\|w\|^2/2+\mathrm{mean}(\mathrm{hinge})$ with
$\lambda=1/(C_{\mathrm{sum}}n)$. These identities are exact for the stated
objectives, with an unregularized intercept. The same numeric `C` across
mean-loss and summed-loss code does not impose the same trade-off.

Let $A=\{i:y_i f(x_i)<1\}$. The selected subgradient is

$$
 g_w=w-\frac{C_{\mathrm{mean}}}{n}\sum_{i\in A}y_i x_i,
 \qquad
 g_b=-\frac{C_{\mathrm{mean}}}{n}\sum_{i\in A}y_i.
$$

The denominator is **all $n$ rows**, not $|A|$. Averaging only active rows
would change the loss weight as the active set changes and fail to optimize
the displayed objective. At exactly $m_i=1$, the implementation chooses the
zero hinge subgradient, a valid endpoint of the subdifferential. Even if no
rows violate the margin, the $w$ regularization gradient remains.

The optimizer initializes $w=b=0$, takes full-batch steps of size
$\eta_t=\eta_0/\sqrt{t+1}$, and retains the lowest observed objective.
This finite-budget schedule can oscillate near hinge kinks; it has no stopping
certificate here. Tests check finite differences away from kinks, a convex
supporting inequality at a kink, and the known optimum $w=\min(C,1),b=0$
for observations `[-1]` and `[+1]` with corresponding labels.

## Dual formulation and support vectors

For the summed-hinge objective, the dual is

$$
\max_\alpha\sum_i\alpha_i-
\frac12\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i,x_j),
\quad 0\le\alpha_i\le C_{\mathrm{sum}},\quad
\sum_i\alpha_i y_i=0.
$$

With the linear kernel, stationarity gives
$w=\sum_i\alpha_i y_i x_i$. A support vector has $\alpha_i>0$.
The ideal KKT conditions, assuming common positive `C`, imply:

| Dual multiplier | Functional margin |
|---|---|
| $\alpha_i=0$ | $m_i\ge1$ |
| $0<\alpha_i<C$ | $m_i=1$ |
| $\alpha_i=C$ | $m_i\le1$ |

Equality cases can be degenerate. In particular, a point at $\alpha_i=C$
can be exactly on the margin, so saying every such point is misclassified
is incorrect. Numerical solvers also use tolerances. Non-support rows have
no direct term in a fixed fitted decision function; removing them and
refitting preprocessing or changing sample-count normalization can still
change the model.

The scratch solver works in the primal and does not compute dual multipliers.
Its active hinge rows are not a substitute for `SVC.support_`.

## Kernels and their geometry

A valid positive semidefinite kernel represents an inner product in a feature
space: $K(x,z)=\langle\phi(x),\phi(z)\rangle$. For any finite sample, its Gram
matrix must be symmetric positive semidefinite; an arbitrary similarity
function is not necessarily a valid kernel for convex SVM optimization.
Prediction uses

$$
f(x)=\sum_{i\in SV}\alpha_i y_i K(x_i,x)+b.
$$

| Kernel | Formula | Inductive bias |
|---|---|---|
| Linear | $x^Tz$ | Linear boundary in supplied features |
| Polynomial | $(\gamma x^Tz+r)^d$ | Polynomial interactions; standard valid choices use nonnegative $\gamma,r$ and integer $d\ge1$ |
| RBF | $\exp(-\gamma\|x-z\|^2)$, $\gamma>0$ | Similarity decays with squared distance |

For RBF, distance $1/\sqrt{\gamma}$ gives similarity $e^{-1}$.
Larger `gamma` makes influence more local; smaller `gamma` spreads it more
broadly. `C` controls violation cost within that geometry, so tune both.
The [official parameter example](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)
provides further exploration. Here the grid uses numeric `gamma` explicitly.
For `gamma="scale"`, scikit-learn instead computes
`1 / (n_features * X.var())` on the data supplied to the SVC.
([SVC API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html))

In a kernel model, margin geometry belongs to the implicit feature space.
Contours $f(x)=-1,0,+1$ in the original input space need not be parallel or
have equal Euclidean spacing. A raw RBF decision score is not a distance in
input space. A geometric distance in feature space would additionally divide
by the norm of the feature-space weight vector.

## Scaling, evaluation, and operational choices

Scaling changes the RBF distance metric. For a linear SVM it changes the
meaning of the coefficient norm penalty. Fit the scaler on each training
fold within a pipeline; scaling the complete dataset before cross-validation
leaks validation statistics. On sparse text features, consider transformations
that preserve sparsity rather than dense centering.

Decision scores support ranking and threshold policies, but are not calibrated
probabilities. Calibration requires validation data or internal resampling
and additional work; logistic regression also needs empirical calibration
checks. Accuracy is used here because the synthetic classes are balanced.
For an operational router, evaluate class-specific recall, error costs, and
any probability or abstention policy on a representative split.

`LinearSVC` has a different solver and defaults to squared hinge with intercept
regularization; equal `C` does not ensure agreement with `SVC(kernel="linear")`.
([LinearSVC API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html))
Kernel SVC trains multiclass models one-versus-one even when the returned
scores use an `ovr` shape. This study's binary dual-score reconstruction does
not extend unchanged to that multiclass coefficient layout.
([SVM guide](https://scikit-learn.org/stable/modules/svm.html))

Kernel methods can require quadratic pairwise storage if a full Gram matrix
is materialized; cached solvers need not store all of it. Training cost depends
on the implementation and geometry. Dense RBF prediction evaluates distances
to support vectors, roughly $O(n_{SV}d)$ per row. Linear prediction can instead
use a weight vector with $O(d)$ cost. No timing or memory benchmark was run here.

Compare against linear models, trees, or kernel approximations according to
feature structure and measured constraints. Fixed embeddings plus a linear
classifier can be useful when nonlinear kernel training is impractical.
These are candidate designs, not application results demonstrated here.

## Executed experiments

Both scripts were run on 2026-10-07 using the existing repository virtual
environment: Python **3.11.0rc2**, NumPy **2.4.6**, scikit-learn **1.9.0**.
This records the installed prerelease Python accurately; it is not a claim of
validation across the repository's entire supported dependency range.
All interpretations below are **pending author review**.

### Joint C/gamma selection on synthetic moons

**Hypothesis.** Joint validation can identify useful nonlinear capacity;
training accuracy alone is insufficient for selecting the configuration.

**Command.** From the repository root:

```powershell
.venv/Scripts/python.exe 01-classical-machine-learning/31-support-vector-machine/example.py
```

**Configuration.** `make_moons(n_samples=400, noise=0.20, random_state=42)`;
stratified 75/25 train/test split with seed 42 (300/100 rows). Search
`C ∈ {0.1,1,10,100}` and `gamma ∈ {0.01,0.1,1,10}` using five shuffled
stratified training folds, seed 42. Scaling is fitted inside each fold;
selection maximizes mean CV accuracy, with the first candidate winning ties.
SVC tolerance is 0.001, probability estimation is disabled, and the winner is
refitted on all training rows. The prespecified linear baseline uses `C=1`
and its own training-fitted scaler.

**Results.** Cross-validation selected `C=1`, `gamma=1`. Mean CV accuracy was
0.980000 (fold standard deviation 0.019437), refitted training accuracy
0.983333, and final test accuracy 0.960000. The fixed linear baseline scored
0.910000 on test. The selected RBF model retained 65/300 training rows as
support vectors, 34 for class 0 and 31 for class 1.

Two grid candidates illustrate the selection distinction:

| C | gamma | Mean fold training accuracy | Mean CV accuracy |
|---:|---:|---:|---:|
| 1 | 1 | 0.981667 | 0.980000 |
| 100 | 10 | 1.000000 | 0.956667 |

Full candidate scores and class metrics are regenerable in the ignored
`outputs/kernel_search.json` record. Mean fold training scores above are
different from the final refitted training score.

**Interpretation candidate — author review required.** The selected nonlinear
model scored higher than the fixed linear reference on this split, and perfect
fold training accuracy did not yield the best CV score. The result is consistent
with a benefit from nonlinear capacity on this generator; it does not identify
a generally optimal `C`/`gamma` combination.

**Limitations.** One seed, a small finite grid, 100 synthetic test rows, balanced
classes, and unequal tuning budgets. CV fold standard deviation is not a
confidence interval and the selected CV score has selection optimism. No
calibration, production latency, drift, or general model superiority was tested.

### Controlled linear optimizer comparison

**Hypothesis.** The corrected mean-hinge subgradient can approach the objective
of a linear SVC when loss normalization and intercept treatment match.

**Command.** From the repository root:

```powershell
.venv/Scripts/python.exe 01-classical-machine-learning/31-support-vector-machine/from_scratch.py
```

**Configuration.** Synthetic `make_blobs`: 240 rows, two centers `[-2,-1]` and
`[2,1]`, cluster standard deviation 1.1, seed 42. A stratified 75/25 split
with seed 42 gives 180 training rows and 60 test rows; standardization fits
only training rows. Scratch `C_mean=1`, initial step size 0.1, 10,000 updates,
step schedule `0.1/sqrt(t+1)`, best observed iterate retained. Reference:
`SVC(kernel="linear", C=1/180, tol=1e-9)`, unregularized intercept.

**Results.** The initial mean-hinge objective was 1.000000000. The retained
scratch objective was 0.552029501812; the reference objective evaluated with
the same helper was 0.552029499228. The gap was `2.583922e-9`. Both classifiers
scored 1.000000 test accuracy and predictions agreed on all 60 rows.
Full precision values and configuration are in ignored
`outputs/linear_solver.json`.

**Interpretation candidate — author review required.** Objective proximity
and prediction agreement support the corrected implementation on this
controlled dataset. Equal predicted classes do not imply identical parameters,
dual variables, or behavior on other inputs.

**Limitations.** This is one small synthetic split with a finite optimization
budget, not an optimizer convergence guarantee or a production benchmark.
The scratch solver omits kernels, sparse matrices, sample/class weights,
calibration, dual coefficients, optimized solvers, and automatic stopping.

## Suggested follow-ups, not executed

Repeat the joint search across seeds to assess selection variability; extend
the scaling comparison across datasets; evaluate linear, polynomial, and RBF
kernels under equal tuning budgets; or study class weights and calibration on
a separate imbalanced generator. The separate [visual laboratory](README_VISUALS.md)
already records margin/kernel demonstrations and one feature-scaling comparison.
The larger comparative studies above remain suggestions, not executed results.
