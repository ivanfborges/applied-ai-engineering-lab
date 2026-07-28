# Notes: Vector Spaces, Bases, and Projections

## 1. Intuition

A vector is an abstract object that can be added to another vector and scaled. A coordinate array is one representation of that vector relative to a chosen basis. Changing the basis changes the coordinates, not the underlying vector.

Think of a point in a room. The usual left-right, forward-backward, and up-down directions form a basis for three-dimensional space. A rotated set of three independent directions is another valid basis. Projecting the point onto the floor keeps the component that the floor can represent and discards the perpendicular component.

Machine learning uses the same geometry:

- feature columns span the space of predictions available to a linear model;
- least squares projects a target onto that column space;
- PCA selects a lower-dimensional subspace and projects centered observations into it;
- embedding models map objects into learned coordinate spaces for geometric comparison.

## 2. Vector Spaces and Subspaces

A vector space (V) over a field such as (mathbb{R}) is closed under vector addition and scalar multiplication and satisfies the usual algebraic axioms: associativity, commutativity of addition, distributivity, a zero vector, and additive inverses.

The familiar example is (mathbb{R}^d), but fixed-shape matrices, polynomials, signals, and parameter collections may also form vector spaces.

A subset (S \subseteq V) is a subspace when:

1. (0 \in S);
2. (u + v \in S) for all (u,v \in S);
3. (a u \in S) for every scalar (a) and (u \in S).

The plane (z=0) is a subspace of (mathbb{R}^3). The plane (z=1) is not because it excludes the origin; it is an affine space, a translated subspace. This distinction explains why PCA centers data before finding a linear subspace.

## 3. Span, Independence, Basis, and Dimension

Given vectors (v_1,\ldots,v_k), a linear combination is

\[
\sum_{i=1}^{k} a_i v_i.
\]

Their span is the set of all such linear combinations:

\[
\operatorname{span}(v_1,\ldots,v_k)
=
\left\{\sum_{i=1}^{k}a_i v_i : a_i \in \mathbb{R}\right\}.
\]

The vectors are linearly independent if

\[
\sum_{i=1}^{k}a_i v_i=0
\]

implies that every (a_i=0). Dependence means at least one vector is redundant because it can be written from the others.

A basis is both linearly independent and spanning. Consequently, every vector in the space has exactly one coordinate representation in that basis. The dimension of a finite-dimensional vector space is the number of vectors in any basis.

For a matrix (A), rank is the dimension of its column space (and also its row space). Full column rank means its columns are linearly independent.

### Ambient and intrinsic dimension

- **Ambient dimension:** number of coordinates used to store a point.
- **Intrinsic dimension:** number of independent directions needed to describe its meaningful variation, exactly or approximately.

Data stored in (mathbb{R}^{768}) may concentrate near a much lower-dimensional structure. Effective dimension is numerical and data-dependent; it should not be inferred from array shape alone.

## 4. Coordinates and Change of Basis

Let the columns of (B) be basis vectors and let (c) contain the coordinates of (x) in that basis:

\[
x = Bc.
\]

For a square invertible basis matrix,

\[
c=B^{-1}x.
\]

In computation, solve (Bc=x) rather than explicitly forming (B^{-1}). If the basis is orthonormal, (B^\top B=I), so

\[
c=B^\top x.
\]

Orthonormal bases make coordinates simple dot products and avoid amplification caused by poorly conditioned bases.

## 5. Orthogonality and Gram-Schmidt

Vectors (u) and (v) are orthogonal under the Euclidean inner product when

\[
u^\top v=0.
\]

A matrix (Q) has orthonormal columns when

\[
Q^\top Q=I.
\]

Classical Gram-Schmidt converts independent vectors into an orthonormal basis for the same span. At each step, it removes from a candidate vector all components aligned with previously accepted basis vectors, then normalizes the remainder:

\[
u_j=v_j-\sum_{i=1}^{j-1}(q_i^\top v_j)q_i,
\qquad
q_j=\frac{u_j}{\lVert u_j\rVert_2}.
\]

This is excellent for learning but can lose orthogonality for nearly dependent inputs because of floating-point cancellation. Modified Gram-Schmidt, Householder QR, and SVD are more robust choices for numerical software.

## 6. Orthogonal Projection

### Onto one direction

For a nonzero direction (u), the projection of (x) must have the form (alpha u). Requiring the residual (x-\alpha u) to be perpendicular to (u) gives

\[
u^\top(x-\alpha u)=0
\quad\Longrightarrow\quad
\alpha=\frac{u^\top x}{u^\top u}.
\]

Therefore,

\[
\operatorname{proj}_u(x)
=\frac{u^\top x}{u^\top u}u.
\]

For unit (u), this simplifies to ((u^\top x)u).

### Onto a subspace

Let the full-column-rank matrix (A \in \mathbb{R}^{n\times k}) span the target subspace. The closest point has the form (A\beta), where

\[
\beta^\star=\arg\min_\beta \lVert x-A\beta\rVert_2^2.
\]

At the optimum, the residual is orthogonal to every column of (A):

\[
A^\top(x-A\beta)=0.
\]

The normal equations are

\[
A^\top A\beta=A^\top x.
\]

The algebraic projection matrix is

\[
P=A(A^\top A)^{-1}A^\top.
\]

This formula is useful for derivation, but production code should not explicitly invert (A^\top A). Forming it squares the 2-norm condition number: (kappa_2(A^\top A)=\kappa_2(A)^2). Prefer least-squares solvers based on QR or SVD.

If (Q) is an orthonormal basis for the same column space,

\[
P=QQ^\top,
\qquad
\hat{x}=QQ^\top x.
\]

The operation first computes subspace coordinates (Q^\top x), then reconstructs (Q(Q^\top x)).

### Defining properties

An orthogonal projection matrix is:

- symmetric: (P^\top=P);
- idempotent: (P^2=P).

Every vector decomposes as

\[
x=\hat{x}+r,
\]

where (hat{x}) is in the subspace and (r=x-\hat{x}) is in its orthogonal complement. Thus (Q^\top r=0) and

\[
\lVert x\rVert_2^2
=\lVert\hat{x}\rVert_2^2+\lVert r\rVert_2^2.
\]

The retained component is the best Euclidean approximation within the subspace; the residual is the discarded information.

## 7. Connections to Machine Learning

### Least squares

Ordinary least squares solves

\[
\hat\beta=\arg\min_\beta\lVert y-X\beta\rVert_2^2.
\]

The fitted vector (\hat y=X\hat\beta) is the orthogonal projection of (y) onto the column space of (X), and the residual satisfies

\[
X^\top(y-\hat y)=0.
\]

Rank deficiency makes coefficient representations non-unique even though a least-norm solution and fitted projection can still be computed with SVD-based methods.

### PCA

For a centered data matrix (X), let (W_k) contain the first (k) orthonormal principal directions. The compressed coordinates and reconstruction are

\[
Z=XW_k,
\qquad
\hat X=ZW_k^\top=XW_kW_k^\top.
\]

PCA chooses the (k)-dimensional linear subspace that minimizes squared reconstruction error. Its assumptions and trade-offs include:

- observations must be centered;
- scale-sensitive features may require standardization;
- linear structure and variance are treated as important;
- components can mix original features and reduce interpretability;
- high retained variance does not guarantee preservation of predictive or retrieval signal;
- the transformation must be fitted only on training data to prevent leakage.

### Embeddings

An embedding model maps an object into (mathbb{R}^d):

\[
f(x)\in\mathbb{R}^d.
\]

The output belongs to a vector space mathematically, but semantic usefulness is learned. The training objective organizes distances, angles, inner products, or neighborhoods so that they support a task. Important qualifications are:

- one coordinate rarely has a stable human interpretation;
- semantic information is usually distributed across directions;
- same-sized vectors from different models are not automatically comparable;
- local neighborhoods may be useful even when global linear analogies fail;
- cosine similarity, dot product, and Euclidean distance are not interchangeable unless normalization and model training make them so.

For unit vectors (x) and (y),

\[
\lVert x-y\rVert_2^2=2-2x^\top y,
\]

so Euclidean distance, cosine similarity, and dot product induce equivalent rankings. Without normalization, magnitude can change those rankings.

## 8. Applications and Trade-offs

### Embedding compression

Projecting 768-dimensional embeddings to 256 dimensions can reduce storage, memory transfer, and distance-computation cost. It can also lower recall, alter similarity thresholds, and change nearest-neighbor rankings.

Evaluate both system and task outcomes:

- index size, memory, latency, ingestion time, and cost;
- recall@k, MRR, nDCG, relevant-context coverage, and downstream answer quality.

Fit the projection on representative training data and version the embedding model, preprocessing, normalization, projection, and index together. Queries and indexed documents must receive the same compatible transformation.

### Redundancy and multicollinearity

Dependent or nearly dependent feature columns can make coefficient estimates unstable and sensitive to small perturbations. Remedies include domain-driven feature removal, regularization, QR or SVD solvers, and dimensionality reduction. PCA may improve conditioning but sacrifices original feature semantics.

### Removing a direction

For a unit direction (u), remove its linear component from (x) with

\[
x_{\text{clean}}=x-(u^\top x)u.
\]

This can probe or suppress a nuisance direction, but it does not prove a semantic concept has been removed. Information may be distributed, nonlinear, or recoverable from other directions.

### When linear projection is insufficient

Linear projection may be a poor model when the data lies on a nonlinear manifold, task signal occupies low-variance directions, distributions shift, or original feature meanings must be preserved. Alternatives include feature selection, random projection, kernel methods, autoencoders, learned projection heads, and manifold-learning methods. Each optimizes a different objective.

## 9. Common Mistakes

- Confusing a vector with its coordinates in one basis.
- Calling a redundant spanning set a basis.
- Equating the number of stored features with intrinsic dimension.
- Treating an exact rank computed in floating point as an unquestionable fact; inspect singular values and tolerances.
- Explicitly calculating ((A^\top A)^{-1}) instead of using a stable solver.
- Assuming classical Gram-Schmidt is robust for nearly dependent vectors.
- Forgetting to center data before PCA or fitting PCA before the train/test split.
- Treating PCA as feature selection rather than construction of new linear combinations.
- Evaluating compression only with reconstruction error or explained variance.
- Mixing query and document embeddings from incompatible models or transformation versions.
- Assuming every embedding dimension corresponds to one interpretable concept.
- Assuming a linear projection preserves nonlinear semantic structure.

## 10. Suggested Experiments

1. Append a column equal to the sum of two basis columns and inspect matrix rank and singular values.
2. Make two columns nearly dependent and compare the behavior of normal equations, `numpy.linalg.lstsq`, QR, and SVD.
3. Compare dot-product, cosine, and Euclidean rankings before and after L2 normalization.
4. Compress explicitly labeled synthetic embeddings with PCA and compare nearest-neighbor overlap at several target dimensions.
5. Remove a chosen unit direction from synthetic vectors and verify that the cleaned vectors have near-zero dot product with it.

## 11. Interview-ready Summary

A basis is a minimal set of linearly independent directions that spans a vector space, and dimension is the number of vectors in a basis. Projection finds the closest representation of a vector inside a subspace. With orthonormal basis (Q), the projection is (QQ^\top x), and the residual is orthogonal to the subspace. This geometry underlies least squares and PCA. Embeddings are points in learned vector spaces, but their metrics and dimensions only have meaning relative to the model and training objective. Any compression or projection must therefore be versioned consistently and evaluated with downstream retrieval or prediction metrics.
