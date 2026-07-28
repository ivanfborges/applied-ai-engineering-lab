# Technical Notes: Vectors, Matrices, and Operations

## Intuition and Representations

A vector is an ordered numerical representation. Depending on context, it may be interpreted as a point, a direction, a feature row, an embedding, a gradient, or a parameter set. A matrix has two complementary interpretations:

1. a collection of vectors, such as a dataset with observations in rows;
2. a mapping that transforms vectors from one coordinate space to another.

For a tabular dataset,

$$
X \in \mathbb{R}^{n \times d},
$$

where $n$ is the number of observations and $d$ is the number of features. Shape is part of the meaning: changing an axis can change an operation even if the same numbers remain present.

In NumPy, an array with shape `(d,)` is neither a two-dimensional row matrix `(1, d)` nor a column matrix `(d, 1)`. This distinction matters for transposition and broadcasting.

## Core Operations

### Addition, Scaling, and Transpose

Vectors of equal dimension add component by component. Scalar multiplication changes magnitude and reverses direction when the scalar is negative. Matrix transposition exchanges axes:

$$
A \in \mathbb{R}^{m \times n} \Rightarrow A^T \in \mathbb{R}^{n \times m}.
$$

These operations appear in bias addition, residual connections, gradient updates, and alignment of operands.

### Dot Product

For $x,y \in \mathbb{R}^d$,

$$
x \cdot y = x^T y = \sum_{i=1}^{d} x_i y_i
= \lVert x \rVert_2 \lVert y \rVert_2 \cos(\theta).
$$

The dot product is positive for acute angles, zero for orthogonal vectors, and negative for opposing directions. Its magnitude also grows with vector length, so a large score does not imply directional similarity alone.

The outer product $xy^T$ produces a matrix whose $(i,j)$ entry is $x_i y_j$. It appears in covariance, gradient derivations, and low-rank constructions.

### Norms and Distances

The general vector norm is

$$
\lVert x \rVert_p = \left(\sum_{i=1}^{d}|x_i|^p\right)^{1/p}.
$$

- L1 norm: $\lVert x \rVert_1 = \sum_i |x_i|$. It is associated with Manhattan geometry and sparsity-inducing regularization.
- L2 norm: $\lVert x \rVert_2 = \sqrt{\sum_i x_i^2}$. It is the Euclidean length.
- Infinity norm: $\lVert x \rVert_\infty = \max_i |x_i|$.

The Frobenius norm treats a matrix as one long vector:

$$
\lVert A \rVert_F = \sqrt{\sum_i\sum_j a_{ij}^2}.
$$

Euclidean and Manhattan distances apply a norm to a difference:

$$
d_2(x,y)=\lVert x-y \rVert_2, \qquad
d_1(x,y)=\lVert x-y \rVert_1.
$$

Both depend on coordinate scale. If one feature spans millions and another spans units, the first can dominate regardless of domain importance.

### Cosine Similarity and Normalization

Cosine similarity measures angular alignment:

$$
\operatorname{cos}(x,y)=
\frac{x \cdot y}{\lVert x \rVert_2\lVert y \rVert_2}.
$$

It is undefined if either vector is zero. L2 normalization produces a unit vector:

$$
\hat{x}=\frac{x}{\lVert x \rVert_2}.
$$

For unit vectors, dot product equals cosine similarity. In addition,

$$
\lVert x-y \rVert_2^2 = 2 - 2(x \cdot y),
$$

so cosine similarity, dot-product similarity, and Euclidean distance produce equivalent rankings when all compared vectors are unit-normalized. This equivalence does not hold for arbitrary vector magnitudes.

### Projection

The projection of $x$ onto nonzero $y$ is

$$
\operatorname{proj}_y(x)=\frac{x \cdot y}{y \cdot y}y.
$$

Projection measures the component of one vector along another. It is central to least squares, regression geometry, PCA, and vector decomposition.

## Matrix Multiplication and Transformations

If

$$
A \in \mathbb{R}^{m \times n}, \qquad
B \in \mathbb{R}^{n \times p},
$$

then $C=AB \in \mathbb{R}^{m \times p}$ and

$$
c_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}.
$$

Every output entry is a row-column dot product. The matching inner dimension is therefore a mathematical requirement, not a library convention.

Matrix multiplication differs from the Hadamard product. In NumPy, `A @ B` performs matrix multiplication and `A * B` performs elementwise multiplication, potentially with broadcasting.

A matrix defines a linear map $T(x)=Ax$ that preserves addition and scalar multiplication and always maps zero to zero. Adding a bias creates an affine map:

$$
y=Ax+b.
$$

Most neural-network “linear” layers are therefore affine. If $y=Bx$ and $z=Ay$, then $z=(AB)x$: $B$ acts first and $A$ second. Since transformation order changes the geometry, $AB$ generally differs from $BA$.

For observations stored as rows, applying the column-vector transformation $y=Ax$ to every observation usually requires `X @ A.T`.

## Assumptions and Trade-offs

### Metric Choice

- Euclidean distance assumes absolute location and magnitude are meaningful and dimensions are comparably scaled.
- Cosine similarity assumes direction is more informative than magnitude.
- Dot product is suitable when magnitude carries signal or when model training explicitly optimizes it.
- L1 distance can reduce the influence of a few large coordinate deviations but is not automatically robust to all outliers.

The correct metric should follow the model objective, data geometry, and application evaluation—not convention alone.

### Dense and Sparse Data

Dense arrays work well for embeddings and accelerator kernels but store every entry. Sparse formats are appropriate for TF-IDF, one-hot features, graphs, and recommender matrices with many zeros. Densifying a large sparse matrix can exhaust memory, while not every sparse operation has an efficient hardware implementation.

### Exactness, Scale, and Precision

Standard dense multiplication of $(m \times n)$ and $(n \times p)$ matrices costs approximately $O(mnp)$. Large systems therefore use batching, optimized kernels, accelerators, sparsity, low-rank approximations, or reduced precision. These techniques trade memory and latency against approximation error or numerical stability.

Exact comparison of a query against every embedding may be acceptable at small scale. At large scale, approximate nearest-neighbor indexing trades some recall for latency and throughput. Recall and end-task quality should be evaluated rather than assumed.

## Applications

- **Tabular ML:** feature matrices and weighted combinations such as $\hat{y}=Xw+b$.
- **Semantic retrieval:** score a query $q \in \mathbb{R}^d$ against document matrix $D \in \mathbb{R}^{n \times d}$ using $Dq$.
- **Recommendation:** user-item affinity often contains the dot product $p_u^Tq_i$.
- **Neural networks:** batched affine layers compute $H=XW+b$.
- **Transformers:** $QK^T$ contains pairwise query-key dot products before scaling and softmax.
- **Graphs:** adjacency matrices encode connectivity and support neighborhood propagation.
- **Computer vision:** arrays and matrix maps support color, geometric, projection, and convolution-related operations.

Linear maps alone cannot express arbitrary nonlinear relationships. Nonlinear features, kernels, trees, or activation functions are required for curved decision boundaries and richer functions.

## Numerical Reliability

Avoid computing $A^{-1}b$ explicitly when solving $Ax=b$. `numpy.linalg.solve(A, b)` is generally more efficient and stable; least-squares problems should use `numpy.linalg.lstsq` or an appropriate decomposition.

A singular matrix has no inverse. An ill-conditioned matrix may be invertible but amplifies small input or floating-point perturbations. Common causes include strongly correlated features, very different scales, and nearly dependent columns. Scaling, regularization, feature removal, or stable decompositions can help.

## Common Mistakes

- Confusing `*` with `@`.
- Skipping shape checks and relying on accidental broadcasting.
- Assuming `AB == BA`.
- Treating a one-dimensional NumPy array as an explicit row or column matrix.
- Fitting a scaler before the train/validation/test split and leaking information.
- Treating dot product and cosine similarity as equivalent without normalization.
- Failing to define zero-vector behavior for cosine similarity.
- Assuming embedding magnitude always represents semantic relevance.
- Using Euclidean distance on unscaled heterogeneous features.
- Explicitly inverting matrices instead of solving the system.
- Densifying a large sparse matrix.
- Ignoring distance concentration and overfitting risk in high-dimensional spaces.

## Suggested Experiments

1. Multiply one document vector in `example.py` by ten and compare all three rankings.
2. Scale one coordinate by 1,000 and observe its effect on Euclidean distance.
3. Verify that normalized dot-product scores equal cosine scores.
4. Reverse scaling and rotation order and compare the resulting coordinates.
5. Generate synthetic vectors in 2, 10, 100, and 1,000 dimensions and compare nearest-to-farthest distance ratios.

## Visual Explorer

The Streamlit application in `visualizations/app.py` connects these formulas to interactive geometry. Its numerical operations live in `visualizations/math_utils.py`, independent of the interface, and are covered by unit tests.

The applied examples are deliberately synthetic:

- feature scaling shows how a large-range coordinate can determine the nearest neighbor;
- handcrafted embedding vectors show why cosine similarity and dot product may rank candidates differently;
- seeded Gaussian vectors illustrate distance concentration without claiming a universal benchmark.

Run it from the repository root:

```bash
python -m streamlit run 00-foundations/02-linear-algebra-vectors-matrices/visualizations/app.py
```
