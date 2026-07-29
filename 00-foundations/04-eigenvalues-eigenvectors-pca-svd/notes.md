# Notes: Eigenvalues, Eigenvectors, PCA, and SVD

## 1. Intuition

A matrix represents a linear transformation. Most vectors change both length
and direction when transformed, but an eigenvector stays on the same line:

\[
Av = \lambda v, \qquad v \ne 0.
\]

The eigenvalue \(\lambda\) describes the action along that direction:

- \(|\lambda| > 1\): stretching;
- \(0 < |\lambda| < 1\): compression;
- \(\lambda < 0\): reversal as well as scaling;
- \(\lambda = 0\): collapse into the null space.

For data, imagine an elongated cloud whose natural axes do not match the
original feature axes. PCA rotates to orthogonal axes aligned with decreasing
variation. Keeping only the first axes compresses the data.

SVD applies a related idea to any matrix, including a rectangular data matrix:

\[
X = U\Sigma V^\top.
\]

It can be read as reorienting the input with \(V^\top\), scaling independent
directions with \(\Sigma\), and reorienting the output with \(U\).

## 2. Eigenvalues and Eigendecomposition

From \(Av = \lambda v\),

\[
(A-\lambda I)v=0.
\]

A nonzero solution exists only when \(A-\lambda I\) is singular:

\[
\det(A-\lambda I)=0.
\]

The roots of this characteristic equation are the eigenvalues. If a
\(d \times d\) matrix has \(d\) linearly independent eigenvectors, then

\[
A = Q\Lambda Q^{-1},
\]

where the columns of \(Q\) are eigenvectors and \(\Lambda\) contains their
eigenvalues.

Not every square matrix is diagonalizable over the real numbers. Eigenvalues
can be complex, and repeated eigenvalues may not provide enough independent
eigenvectors. PCA avoids many of these complications because its covariance
matrix is real and symmetric. The spectral theorem gives

\[
C = V\Lambda V^\top, \qquad V^\top V=I.
\]

Its eigenvalues are real and its eigenvectors can be chosen orthonormally.

## 3. Covariance Is Positive Semidefinite

Let \(X_c \in \mathbb{R}^{n \times d}\) be a data matrix after subtracting each
feature mean. Its sample covariance matrix is

\[
C = \frac{1}{n-1}X_c^\top X_c.
\]

It is symmetric. For any vector \(a\),

\[
a^\top Ca
= \frac{1}{n-1}a^\top X_c^\top X_c a
= \frac{1}{n-1}\lVert X_c a\rVert_2^2
\ge 0.
\]

Therefore, \(C\) is positive semidefinite and its eigenvalues are nonnegative.
Tiny negative values from a numerical solver may be floating-point artifacts.
The rank of \(C\) is at most \(\min(d,n-1)\), so zero eigenvalues are expected
when features are redundant or \(d \ge n\).

## 4. PCA as Variance Maximization

For a unit direction \(v\), the projected observations are \(X_cv\). Their
sample variance is

\[
\operatorname{Var}(X_cv)=v^\top Cv.
\]

The first principal direction solves

\[
\max_{v}\;v^\top Cv
\quad \text{subject to} \quad v^\top v=1.
\]

Using a Lagrange multiplier,

\[
\mathcal{L}(v,\lambda)=v^\top Cv-\lambda(v^\top v-1).
\]

Setting the derivative to zero gives

\[
Cv=\lambda v.
\]

At a unit eigenvector, the objective value is its eigenvalue. The maximizing
direction is therefore the eigenvector with the largest eigenvalue. Later
components repeat this optimization while remaining orthogonal to earlier
ones.

If \(W_k=[v_1,\ldots,v_k]\), then

\[
Z=X_cW_k
\]

contains the component scores, and

\[
\hat X=ZW_k^\top+\mu
\]

reconstructs the data in the original feature coordinates.

The feature coefficients in a direction are often called component loadings,
although terminology varies across software and disciplines. A large absolute
coefficient means that the original feature strongly contributes to that axis;
the sign is orientation, not importance.

## 5. Explained Variance

If covariance eigenvalues are ordered
\(\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d\), the explained variance
ratio is

\[
r_i=\frac{\lambda_i}{\sum_{j=1}^{d}\lambda_j},
\]

and cumulative explained variance through component \(k\) is

\[
R_k=\sum_{i=1}^{k}r_i.
\]

The total variance is

\[
\operatorname{tr}(C)=\sum_j C_{jj}=\sum_i\lambda_i.
\]

A rule such as retaining 95% of variance is only a heuristic. PCA does not see
the target, retrieval relevance judgments, or operational constraints. A
low-variance direction can be essential to a supervised task.

## 6. Singular Value Decomposition

For \(X\in\mathbb{R}^{n\times d}\), the reduced SVD is

\[
X=U\Sigma V^\top,
\]

where \(r=\operatorname{rank}(X)\),
\(U\in\mathbb{R}^{n\times r}\),
\(\Sigma\in\mathbb{R}^{r\times r}\), and
\(V\in\mathbb{R}^{d\times r}\). The columns of \(U\) and \(V\) are
orthonormal, and

\[
\sigma_1\ge\sigma_2\ge\cdots\ge\sigma_r>0.
\]

The right singular vectors are eigenvectors of \(X^\top X\):

\[
X^\top X=V\Sigma^2V^\top.
\]

The left singular vectors are eigenvectors of \(XX^\top\):

\[
XX^\top=U\Sigma^2U^\top.
\]

Unlike eigendecomposition, SVD exists for every real or complex matrix and does
not require a square or diagonalizable input.

## 7. PCA Through SVD

For centered data,

\[
X_c=U\Sigma V^\top.
\]

Substitution into the covariance matrix gives

\[
C
=\frac{1}{n-1}X_c^\top X_c
=V\frac{\Sigma^2}{n-1}V^\top.
\]

Consequently:

\[
\lambda_i=\frac{\sigma_i^2}{n-1},
\qquad
W_k=V_k,
\qquad
Z=X_cV_k=U_k\Sigma_k.
\]

Direct SVD is often preferable to explicitly building \(X_c^\top X_c\).
Forming the Gram or covariance matrix squares the 2-norm condition number,
can add rounding error, and requires storing a \(d\times d\) matrix.
Appropriate computational choices depend on matrix shape, sparsity, the number
of requested components, and available memory.

## 8. Low-Rank Approximation

Keeping the first \(k\) singular triplets gives

\[
X_k=U_k\Sigma_kV_k^\top.
\]

The Eckart-Young-Mirsky theorem states that this is a best rank-\(k\)
approximation under the spectral and Frobenius norms. For the Frobenius norm,

\[
\lVert X-X_k\rVert_F^2
=\sum_{i=k+1}^{r}\sigma_i^2.
\]

For centered PCA data, this is also

\[
(n-1)\sum_{i=k+1}^{r}\lambda_i.
\]

Thus variance retained and squared reconstruction error are two views of the
same orthogonal low-rank projection. The guarantee does not cover arbitrary
task losses, cosine-neighbor preservation, or nonlinear structure.

## 9. Centering, Scaling, and Whitening

### Centering

Classical PCA describes variation around the mean, so it centers each feature.
SVD itself does not center. Running SVD on raw data with a large offset can
make the dominant direction describe the mean rather than variation around it.
`sklearn.decomposition.PCA` centers automatically.

### Standardization

Standardization replaces feature \(j\) with

\[
x'_{ij}=\frac{x_{ij}-\mu_j}{s_j}.
\]

PCA on standardized variables corresponds to analyzing their correlation
structure. It is useful when arbitrary units would otherwise dominate, but it
is not mandatory. If all measurements share a meaningful physical scale,
standardizing may erase useful magnitude information.

The scaler and PCA must be fitted only on training data and kept in one
versioned preprocessing pipeline.

### Whitening

PCA scores have diagonal covariance \(\Lambda\). Whitening rescales them:

\[
Z_{\text{white}}=X_cV_k\Lambda_k^{-1/2}.
\]

The retained coordinates then have approximately unit variance. Whitening
removes relative scale, can reduce interpretability, and can amplify noise in
small-variance directions.

## 10. Assumptions and Modeling Choices

PCA is most defensible when:

- linear combinations represent the structure of interest;
- variance is a useful proxy for retained information;
- Euclidean distances and orthogonality are meaningful;
- feature scales have been handled deliberately;
- the fitting sample represents future inputs;
- outliers do not dominate the covariance estimate.

PCA does not require normally distributed data for the algebraic
transformation. Distributional assumptions become relevant for some
statistical inference or probabilistic interpretations.

## 11. Applications and Trade-offs

### Visualization

Two or three components can expose broad clusters, outliers, batch effects,
and class overlap. PCA preserves global variance, not necessarily local
neighborhoods or class separation, so a 2D view must not be overinterpreted.

### Denoising and Compression

Dropping low-energy directions can reduce storage or noise when the signal is
approximately low rank and noise occupies weaker directions. If rare signal
also has low variance, it will be lost.

### Multicollinearity

Orthogonal components can stabilize a linear model with correlated predictors,
but coefficients no longer map cleanly to original business variables.
Regularization may preserve more interpretability.

### Sparse Text

Centering a sparse TF-IDF matrix usually makes it dense. `TruncatedSVD`
operates without centering and is therefore common for latent semantic
analysis. It is not mathematically identical to PCA on centered data.

### Embeddings and Vector Search

Reducing embedding dimension can decrease index memory, serialization cost,
and distance-computation work. It may also change semantic neighborhoods. Fit
the transformation on a representative corpus, transform both documents and
queries identically, apply any required normalization after projection, and
measure Recall@k, MRR, nDCG, and end-to-end answer quality.

Refitting changes the coordinate system. Existing and newly transformed
vectors may become incompatible, so a vector index normally needs a versioned
migration and full rebuild.

### Recommendation Systems

A low-rank user-item matrix can expose latent factors, but plain SVD treats the
matrix as fully observed. Real interaction data includes missing-not-at-random
entries, user/item biases, implicit feedback, and temporal effects, which call
for objectives designed for those conditions.

## 12. Limitations

- PCA is linear and cannot unfold a nonlinear manifold.
- It is unsupervised and may discard low-variance predictive information.
- Squared deviations make it sensitive to outliers.
- Components can be difficult to explain or govern.
- Component directions can drift as the fitting population changes.
- Closely spaced or repeated eigenvalues make individual directions unstable,
  even when their combined subspace is stable.
- PCA does not select original features; it constructs new ones.
- Categorical encodings may create Euclidean geometry with little domain
  meaning.

Alternatives depend on the goal: feature selection for original-variable
interpretability, robust covariance methods for outliers, Kernel PCA or
autoencoders for nonlinear structure, and supervised methods such as LDA or
PLS when target preservation is central.

## 13. Common Mistakes

1. **Leakage:** fitting scaling or PCA before the train/test split.
2. **No centering:** confusing a large mean offset with high variation.
3. **Automatic scaling:** standardizing without considering units and domain
   meaning.
4. **Variance as task quality:** assuming 95% explained variance means 95% of
   predictive or retrieval performance.
5. **Causal interpretation:** treating a loading as a causal or actionable
   variable effect.
6. **Sparse densification:** centering a large sparse matrix.
7. **Wrong solver:** using general `eig` instead of symmetric `eigh` for a
   covariance matrix.
8. **Exact sign comparison:** treating \(v\) and \(-v\) as different PCA
   solutions.
9. **Unnecessary covariance construction:** building a huge \(d\times d\)
   matrix when direct, truncated, or randomized SVD is more suitable.
10. **Silent refitting:** changing production component semantics without
    versioning and migrating dependent data.

## 14. Sign and Subspace Indeterminacy

If \(v\) is an eigenvector, so is \(-v\). Singular-vector and PCA
implementations may therefore return opposite signs. Compare absolute
correlations, projection matrices, or reconstructions rather than raw signed
vectors.

With repeated eigenvalues, any orthonormal basis of the shared eigenspace is
valid. Even without exact equality, nearly tied eigenvalues can make individual
components unstable. In that case, compare the subspace using
\(W_kW_k^\top\), principal angles, or downstream behavior.

## 15. Practical Component Selection

Use multiple signals:

- cumulative explained variance and a scree plot;
- reconstruction error;
- downstream cross-validation metrics;
- retrieval-neighbor preservation;
- latency, memory, and storage budgets;
- stability across samples or time windows;
- interpretability and governance requirements.

The best component count is a model and system choice, not a universal
percentage.

## 16. Visual Laboratory Checkpoints

The `visual_lab/` module turns the identities above into falsifiable visual
checks:

- the eigenvector animation keeps each eigenvector on its invariant line while
  arbitrary vectors rotate;
- the covariance ellipse scales PCA arrows by \(\sqrt{\lambda_i}\), connecting
  eigenvalues to standard deviation along each direction;
- the 3D projection animation shows the residual removed along the discarded
  component;
- the SVD animation applies \(V^\top\), \(\Sigma\), and \(U\) separately and
  verifies that their composition equals the direct transformation;
- the low-rank image experiment calculates retained singular-value energy,
  reconstruction MSE, and a clearly qualified dense-storage estimate;
- the equivalence figure aligns component signs before comparing covariance
  eigendecomposition, direct SVD, and scikit-learn PCA;
- the pitfalls comparison changes one modeling condition at a time: centering,
  feature scale, or outlier leverage.

These visuals preserve the same limitations as the underlying methods. A
geometrically faithful PCA projection is not evidence that labels, causal
structure, semantic neighbors, or production performance are preserved.
