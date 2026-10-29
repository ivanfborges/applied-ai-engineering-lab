# PCA: retained variance, lost directions and useful information

## Projection and reconstruction

Let $X\in\mathbb{R}^{n\times p}$ and let $X_c=X-\mathbf{1}\mu^\top$ use the
**training** mean. For sample covariance and its ordered eigendecomposition:

$$
S=\frac{X_c^\top X_c}{n-1},\qquad Sv_j=\lambda_jv_j,\qquad
\lambda_1\geq\cdots\geq\lambda_p\geq0.
$$

Maximizing $v^\top Sv$ subject to $v^\top v=1$ produces the first eigenvector;
successive directions impose orthogonality. With $V_k=[v_1,\ldots,v_k]$:

$$
Z=X_cV_k,\qquad \widehat X=ZV_k^\top+\mathbf{1}\mu^\top,\qquad
EVR_j=\frac{\lambda_j}{\sum_i\lambda_i}.
$$

The code stores components as **rows** (`components_ = V_k.T`), so projection
is `(X - mean_) @ components_.T`. Scores are coordinates of observations;
components are directions in feature space. Feature selection instead retains
original columns.

For centered SVD $X_c=U\operatorname{diag}(s)V^\top$,
$\lambda_j=s_j^2/(n-1)$ and the retained scores equal $U_ks_k$ columnwise.
Centering bounds rank by $\min(n-1,p)$. Keeping all available nonzero directions
reconstructs training rows; a model fitted with fewer than $p$ directions need
not reconstruct arbitrary future observations outside that fitted subspace.

The optimal rank-$k$ approximation satisfies:

$$
\|X_c-X_cV_kV_k^\top\|_F^2=(n-1)\sum_{j>k}\lambda_j.
$$

Our MSE divides this SSE by $np$, averaging over observations **and features**.
The eigenvalue-tail identity applies to training data, not to held-out MSE.
For fixed nested orthogonal subspaces, adding directions cannot increase
pointwise squared reconstruction error, including for held-out rows. Choosing
dimensions for downstream prediction still requires a task metric.

## Scaling changes the objective

Centering removes an offset; it does not reconcile measurement units.
`PCA` centers but does not standardize. Raw covariance PCA emphasizes numerical
variance; standardization is appropriate when comparable feature contributions
are a defensible modeling choice. It also gives small-variance noise more weight.

`StandardScaler` uses population variance (`ddof=0`), whereas PCA's eigenvalues
use sample covariance (`ddof=1`). With $p$ nonconstant standardized features,
the training sample-covariance trace is $pn/(n-1)$, not exactly $p$.
Sample-standardized covariance equals the correlation matrix; sklearn's
population-standardized covariance differs by $n/(n-1)$ when all features are
nonconstant. This common factor leaves eigenvectors and variance ratios intact.

Standardized reconstruction minimizes error in that transformed metric.
Use the fitted scaler's inverse transform to recover original feature units;
raw-unit MSE would assign a different weight to each feature. Never compare
raw and standardized MSE as if they measured the same loss.

## Interpreting directions without inventing meaning

`components_[j, i]` is the weight of feature $i$ in direction $j$. Large absolute
weights show a feature's contribution to that direction in the chosen scaling.
The term *loading* is ambiguous: some texts use direction weights, others use
$v_{ij}\sqrt{\lambda_j}$. For nonconstant feature $i$ and a positive-variance PC,
the feature-score correlation is $v_{ij}\sqrt{\lambda_j}/s_i$, using sample
standard deviation $s_i$ in the PCA input space. The terminal output calls its
numbers **direction weights** to make the convention explicit.

Multiplying a direction and its scores by $-1$ preserves reconstruction. Closely
spaced eigenvalues can make individual directions unstable across samples;
compare complete subspaces with projectors or principal angles. If a truncation
boundary splits an exact eigenvalue tie, even the selected subspace is not
unique. The tied-eigenvalue test retains the entire tied block.

Training scores are uncorrelated because $V_k^\top SV_k$ is diagonal. This
does not imply independence, and it need not hold on a shifted future dataset.
Ordinary PCA requires no Gaussian assumption, but it is sensitive to outliers,
batch effects, redundant measurements and the chosen population. Naming PC1
a business or biological factor requires domain and stability review.

## Choosing dimensions and validating a pipeline

A cumulative-variance threshold is a reconstruction heuristic. In this example,
`PCA(n_components=0.95, svd_solver="full")` chooses the smallest count that
**exceeds** the threshold under sklearn's strict-boundary convention. The count
is learned separately in each fold; it is not computed once on the full dataset.

Split first. Fit scaler, PCA and classifier on each training fold, transform
validation rows with those frozen objects, then select a candidate on mean CV
balanced accuracy. Refit that candidate on all outer-training rows and evaluate
once on the reserved test labels. The script uses identical folds and fixed
logistic-regression `C=1` for all candidates. Exact CV ties follow candidate
order: baseline, two PCs, then the 95% rule.
Balanced accuracy averages per-class recalls. This example compares fixed
representations, with no joint tuning of regularization and component counts.

Training-only diagnostics print raw/standardized spectra and the four largest
absolute standardized PC1 weights. Test features are used for reconstruction
diagnostics, without choosing another classifier or consulting their labels.
Those rows cease to be a fresh holdout if these diagnostics motivate later
changes; a new study would need a new evaluation boundary.

## Practical limits and alternatives

- PCA can remove useful low-variance signal. Denoising is justified only when
  the discarded directions actually behave like noise for the task.
- Whitening rescales retained scores by inverse standard deviation. It changes
  the downstream metric and can amplify small-eigenvalue noise; it is disabled here.
- Explicit covariance uses $O(p^2)$ storage and eigendecomposition typically
  $O(p^3)$ work, in addition to covariance construction. It squares the spectral
  condition number of the centered matrix. Full SVD avoids explicitly forming it.
- The educational class accepts dense finite real matrices with at least two
  training rows; it permits constant columns/rank deficiency but rejects zero
  total variance. It clips roundoff-negative covariance eigenvalues. It is not
  a replacement for sklearn's solvers or input ecosystem.
- Feature selection preserves original-variable meaning. TruncatedSVD avoids
  mandatory centering for sparse text; t-SNE/UMAP serve different neighborhood
  objectives. Ordinary PCA is a global linear representation.
- Compressing document embeddings changes distances and usually cosine
  similarities. Version one training-fitted transform for both documents and
  queries; validate retrieval quality and normalization. No retrieval, storage,
  latency or deployment gain is measured in this study.
- At inference, preserve feature order, units, fitted means/scales and PCA
  version. Reconstruction drift can be a diagnostic, but does not identify its
  cause or replace monitoring downstream outcomes.

## Interview checks

**Does retaining 95% of variance retain 95% of predictive information?** No.
Labels never enter PCA's objective. The synthetic counterexample isolates this
failure by assigning the label to a deliberately low-variance feature.

**Why can a full rotation be lossless but PCA compression lose information?**
A complete orthonormal basis is invertible. Truncation removes coordinates;
the inverse transform replaces discarded coordinates with zero relative to the
training mean.

**Why compare projectors rather than signed coefficients?** Signs are arbitrary,
and bases can rotate within a tied eigenspace. The projector describes a retained
subspace, provided the compared fits retain the same complete tied block.

**Should scaling always precede PCA?** Choose the metric from the application.
Mixed units often justify standardization; physically meaningful raw variance
may justify covariance PCA. Neither choice guarantees useful prediction.

**How would you justify PCA operationally?** Compare task quality, resource
cost and interpretability with an unreduced baseline using leakage-safe
selection. This study measures no resource costs or production outcomes.

## Suggested experiments — not executed

The [separate visual phase](VISUAL_LAB.md) now includes a fixed outlier
illustration and additional scale/reconstruction comparisons. The broader
follow-up studies below remain proposals.

Repeat the Wine comparison across outer splits; jointly tune `C` and dimension
within nested CV. Perturb the training population to assess subspace stability;
inject outliers to assess rotations. Test a standardized-PCA variant of the
synthetic counterexample. Evaluate embedding retrieval before/after compression
on a separate labeled query set. These are proposals, not reported results.

## Executed experiments — 2026-10-29

Executed by Codex on Windows x64 using stable Python **3.11.14**, NumPy **2.4.6**,
SciPy **1.17.1**, scikit-learn **1.9.1** and single-threaded library execution
for `example.py`. The NumPy-only script uses the same Python/NumPy environment.
This is the initial Day 39 implementation run, distinct from the source study's
unexecuted proposals and earlier Day 4 experiments. Dependency ranges are not
a numerical lock. Both commands printed terminal results and wrote no files:

```bash
python 02-unsupervised-recommender-systems/39-pca-in-practice/example.py
python 02-unsupervised-recommender-systems/39-pca-in-practice/from_scratch.py
```

**All interpretation candidates below require author review.** These measured
values do not constitute the author's conclusions or production evidence.

### A. Wine: variance threshold versus fixed predictive representations

**Hypothesis:** changing representation can change classification quality;
a variance threshold alone cannot establish the best predictor.

**Configuration:** UCI Wine via `load_wine()`, 178 rows, 13 features, three
classes. Stratified 75/25 split with seed 42: 133 training and 45 test rows;
class counts `[44, 53, 36]` and `[15, 18, 12]`. Five shuffled stratified folds,
seed 42, identical for all candidates. Fold-local `StandardScaler`, optional
full-SVD PCA with `whiten=False`, then logistic regression (`C=1`, `lbfgs`,
`max_iter=2000`). Selection uses mean CV balanced accuracy, exact ties following
the table's candidate order. Only the selected representation is evaluated on test labels.

**Measured result:**

| Candidate | CV balanced accuracy mean | Fold SD (`ddof=1`) | Dimensions in fold order |
|---|---:|---:|---|
| No PCA | 0.98121212 | 0.01719547 | 13, 13, 13, 13, 13 |
| Two PCs | 0.98121212 | 0.02747400 | 2, 2, 2, 2, 2 |
| 95% training variance | 0.98727273 | 0.01746044 | 9, 10, 10, 10, 10 |

CV selected the 95% rule. Refit on all 133 training rows, it retained ten PCs;
its final test balanced accuracy was **0.97222222**.

**Interpretation candidate — author review required:** two PCs and the full
representation shared a CV mean here, while the threshold representation had
the highest mean. The threshold's fold-specific dimensions illustrate that it
is a fitted rule. These small differences do not establish general superiority.

**Limitation:** one small outer split, dependent CV folds, fixed regularization,
no uncertainty interval and no paired statistical comparison. The selected
test result is not an estimate of the other candidates' held-out quality.

### B. Wine: scaling and reconstruction

**Hypothesis:** scaling changes the variance objective, and additional nested
directions reduce squared reconstruction loss without directly measuring labels.

**Configuration:** same outer-training rows as A; full-SVD PCA separately on
raw features and training-standardized features. Reconstruction models retain
two, threshold-selected ten, or all thirteen PCs. The fitted training scaler
transforms held-out rows. MSE averages squared standardized errors over all
entries, with no conversion to original measurement units.

**Measured result:** raw PC1 EVR was **0.99779501**, versus **0.35749925**
after standardization. The largest absolute standardized PC1 direction weights
were flavanoids `+0.427807`, total phenols `+0.392694`, OD280/OD315 of diluted
wines `+0.381598` and proanthocyanins `+0.310834`. Axis sign is arbitrary.

| Retained PCs | Training cumulative EVR | Training MSE | Held-out MSE |
|---:|---:|---:|---:|
| 2 | 0.54959175 | 0.4504082456 | 0.4693018797 |
| 10 | 0.96372013 | 0.0362798711 | 0.0570779209 |
| 13 | 1.00000000 | 0.0000000000 | 0.0000000000 |

Displayed zeros are rounded numerical residuals, not a claim of exact floating-
point equality.

**Interpretation candidate — author review required:** raw and standardized
spectra answer different geometric questions. Additional directions reduced
reconstruction loss here; neither the weights nor these losses validate
chemical mechanisms, class separation or predictive relevance.

**Limitation:** a single fitted population, no outlier/stability analysis, and
MSE tied to the chosen standardized metric. Held-out loss is not given by the
training eigenvalue-tail identity. Semantic interpretations of the weights
have not been validated.

### C. Synthetic: useful signal in a low-variance direction

**Hypothesis:** high retained variance can coexist with the loss of label signal.

**Configuration:** `default_rng(42)` draws 800 nuisance values from a normal
distribution with mean zero and SD 20, followed by 800 independent signal
values with mean zero and SD 0.1. The label is `signal > 0`. Stratified 75/25
split seed 42 gives 600 training and 200 test rows. The baseline standardizes
both features; the compressed pipeline fits one-PC PCA in **raw units** and
then standardizes that score. Both use the logistic-regression settings in A.
These are two fixed configurations, with no model selection on test labels.

**Measured result:** raw one-PC training EVR **0.99997235**; test balanced
accuracy **1.00000000** for the baseline and **0.46894689** after compression.

**Interpretation candidate — author review required:** this constructed setting
demonstrates that variance retention can prioritize a nuisance direction while
removing the useful label dimension. The finite-sample compressed result near
chance is compatible with that mechanism; its value below 0.5 is not a general
property of PCA.

**Limitation:** deliberately extreme scale imbalance, deterministic threshold
labels, one seed and no standardized-PCA control run. This does not show that
PCA generally damages prediction or that scaling always solves signal loss.

### D. Synthetic: educational projection and reconstruction

**Hypothesis:** correlated three-feature data can be well approximated by fewer
directions, while truncation retains nonzero reconstruction error.

**Configuration:** `default_rng(42)`, 120 independent three-dimensional standard
normal latent rows, right-multiplied by
`[[3,2,1],[0,0.5,0.2],[0,0,0.1]]`, then shifted by `[10,-4,2]`.
Fit on the first 90 rows, reconstruct the final 30 with frozen training means
and directions. No standardization; `CovariancePCA` uses sample covariance.

**Measured result:**

| PCs | Training cumulative EVR | Training MSE | Held-out MSE |
|---:|---:|---:|---:|
| 1 | 0.98391331 | 0.0627042015 | 0.0613699316 |
| 2 | 0.99948962 | 0.0019894188 | 0.0032201101 |
| 3 | 1.00000000 | 0.0000000000 | 0.0000000000 |

**Interpretation candidate — author review required:** the first direction
captures most training variance under this intentionally anisotropic generator;
the second lowers both measured losses, while the complete basis reconstructs
to numerical precision.

**Limitation:** a designed linear Gaussian generator, no labels, one seed and
no evidence for nonlinear structure, denoising quality or real-world utility.
