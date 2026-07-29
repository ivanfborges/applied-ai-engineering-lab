# Interview Questions: Eigenvalues, Eigenvectors, PCA, and SVD

## 1. What does an eigenvector represent?

It is a nonzero direction that a square linear transformation preserves:
\(Av=\lambda v\). The vector may be stretched, compressed, reversed, or
collapsed, but it is not rotated away from its line. The eigenvalue is the
corresponding scale factor.

## 2. Why are covariance-matrix eigenvalues nonnegative?

For \(C=X_c^\top X_c/(n-1)\) and any vector \(v\),

\[
v^\top Cv=\frac{1}{n-1}\lVert X_cv\rVert_2^2\ge 0.
\]

So \(C\) is positive semidefinite. For an eigenvector,
\(v^\top Cv=\lambda\lVert v\rVert^2\), which implies \(\lambda\ge0\).

## 3. Why does PCA choose covariance eigenvectors?

The variance after projecting centered data onto a unit direction \(v\) is
\(v^\top Cv\). Maximizing that Rayleigh quotient subject to \(v^\top v=1\)
leads through Lagrange multipliers to \(Cv=\lambda v\). The eigenvector with the
largest eigenvalue captures the most variance; subsequent orthogonal
eigenvectors capture the next-largest amounts.

## 4. How are PCA and SVD related?

For centered data \(X_c=U\Sigma V^\top\):

- the columns of \(V\) are the principal directions;
- the PCA scores are \(X_cV=U\Sigma\);
- covariance eigenvalues are \(\lambda_i=\sigma_i^2/(n-1)\).

Thus PCA can be computed by SVD of the centered data without explicitly forming
the covariance matrix.

## 5. Why is direct SVD often preferable to covariance eigendecomposition?

Constructing \(X_c^\top X_c\) squares the 2-norm condition number, can add
rounding error, requires materializing a \(d\times d\) matrix, and may do work
for directions that will be discarded. Direct, truncated, or randomized SVD
is often more stable or scalable. The best solver still depends on matrix
shape, sparsity, and the requested rank.

## 6. Does PCA require standardization?

No. Classical PCA requires thoughtful centering; standardization is a modeling
choice. Standardize when arbitrary units or scales should not determine the
components. Preserve the original scale when its magnitude has domain meaning.
`sklearn.decomposition.PCA` centers but does not standardize.

## 7. What are the two optimization views of PCA?

For orthogonal linear projections and squared Euclidean loss, retaining the
first \(k\) principal directions both maximizes retained variance and minimizes
reconstruction error. The error equals the sum of squared discarded singular
values. This equivalence does not imply optimality for a supervised or
retrieval metric.

## 8. Why can PCA hurt a classifier?

PCA is unsupervised and ranks directions by feature variance, not target
information. A low-variance direction may separate classes while high variance
may be nuisance variation. Select component count inside cross-validation and
compare against no PCA, feature selection, LDA, PLS, or regularization as
appropriate.

## 9. How do PCA and `TruncatedSVD` differ in scikit-learn?

PCA centers its input. `TruncatedSVD` normally operates on the uncentered
matrix and can preserve sparsity, which makes it useful for TF-IDF and
term-document matrices. Uncentered truncated SVD is therefore not identical to
classical PCA.

## 10. How would you select the number of components?

Combine cumulative explained variance, scree-plot structure, reconstruction
error, downstream cross-validation, stability, interpretability, and system
constraints. For vector search, include Recall@k, MRR or nDCG, end-to-end
answer quality, latency, and index memory. A fixed variance threshold is a
starting point, not a decision rule.

## 11. What is sign indeterminacy?

If \(v\) is an eigenvector, \(-v\) is equally valid. The same ambiguity occurs
for singular vectors and PCA directions. Compare absolute correlations,
projection matrices, reconstructed values, or subspaces rather than requiring
equal signed coefficients.

## 12. What happens when eigenvalues are repeated or nearly equal?

For a repeated eigenvalue, individual eigenvectors within its eigenspace are
not unique; any orthonormal basis of that subspace is valid. Nearly tied
eigenvalues can make individual components unstable across samples, even when
the combined principal subspace remains stable.

## 13. Why is PCA sensitive to outliers?

Covariance and reconstruction loss use squared deviations, so a small number
of extreme observations can rotate the dominant axes. Investigate data
quality, robust scaling, outlier treatment, or robust covariance/PCA methods,
then validate the impact rather than removing points automatically.

## 14. How would you deploy PCA safely in an ML pipeline?

Split first, then fit scaling and PCA only on the training partition inside
one pipeline. Version the fitted mean, scale, components, input schema, feature
order, and downstream model together. At inference, reject incompatible
schemas and never silently refit. Monitor input drift, component scores, and
task metrics.

## 15. How would PCA affect a vector-search system?

It may reduce index memory, transfer size, and distance-computation cost, but
it can change semantic neighborhoods and reduce recall. Fit on a representative
corpus, transform documents and queries with the same version, normalize at
the correct stage, and evaluate the quality-latency-memory frontier. A refit
usually requires transforming the whole corpus and rebuilding the index.

## 16. What is randomized SVD, and when would you use it?

It uses random projections to approximate the dominant subspace before solving
a smaller factorization. It is useful for a very large matrix when only a small
rank is needed and modest approximation error is acceptable. Accuracy depends
on spectral decay, oversampling, power iterations, and randomness, so it should
be validated and seeded when reproducibility matters.

## 17. How is PCA related to a linear autoencoder?

Under appropriate conditions, a linear autoencoder trained with squared
reconstruction loss learns the same optimal principal subspace. Its individual
encoder weights need not equal PCA eigenvectors because the latent basis may
be rotated. Nonlinear autoencoders are more expressive but add optimization,
architecture, reproducibility, and deployment complexity.

## 18. Give a two-minute senior-level explanation of PCA.

PCA is a linear, unsupervised dimensionality-reduction method. After centering
the data, it finds orthogonal directions that capture decreasing variance.
These are covariance eigenvectors, with eigenvalues equal to variance along
each direction. In practice I view PCA through SVD: for
\(X_c=U\Sigma V^\top\), \(V\) contains the directions, \(U\Sigma\) contains
the scores, and \(\sigma_i^2/(n-1)\) gives explained variance.

I use PCA when a lower-rank representation can help visualization,
compression, denoising, multicollinearity, or system cost. I do not assume
variance equals target relevance. I decide scaling from domain semantics, fit
preprocessing without leakage, choose rank with downstream and operational
metrics, and version the transformation because a refit changes the feature
space.
