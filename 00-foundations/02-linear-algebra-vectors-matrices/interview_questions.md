# Senior Interview Questions and Answers

## 1. What is the geometric interpretation of the dot product?

For $x,y \in \mathbb{R}^d$, $x \cdot y=\lVert x\rVert_2\lVert y\rVert_2\cos\theta$. It measures alignment while retaining magnitude. A positive value indicates broadly aligned directions, zero indicates orthogonality, and a negative value indicates opposing directions. It also determines the scalar component used when projecting one vector onto another.

## 2. How do dot product and cosine similarity differ?

The dot product depends on direction and magnitude. Cosine similarity divides by both L2 norms and retains only angular alignment. They are equal for unit-normalized vectors. Metric choice should reflect how the representation was trained and whether magnitude has application meaning.

## 3. Why must inner dimensions match in matrix multiplication?

Each output element is a dot product between a row of the first matrix and a column of the second. Those vectors need equal length. Thus $(m \times n)(n \times p)$ produces $(m \times p)$.

## 4. Why is matrix multiplication not commutative?

Matrices can encode transformations, and applying transformation $B$ followed by $A$ produces $AB$, which generally differs from applying $A$ followed by $B$. In addition, $AB$ and $BA$ can have different shapes, or one can be undefined.

## 5. What distinguishes a linear transformation from an affine transformation?

A linear map $y=Ax$ preserves addition and scalar multiplication and maps the origin to the origin. An affine map $y=Ax+b$ adds translation. Neural-network layers commonly called linear layers are technically affine when they contain a bias.

## 6. Why does feature scaling matter for distance-based methods?

Distance is computed directly from coordinates. A large-range feature can dominate Euclidean or Manhattan distance even if it is not more important. Fit scaling parameters on training data only, then apply them unchanged to validation, test, and serving data to avoid leakage and skew.

## 7. When would you prefer cosine similarity to Euclidean distance?

Cosine is useful when direction captures meaning and magnitude should be ignored, as often occurs with text vectors and some embeddings. Euclidean distance is appropriate when absolute position and scale are meaningful. For unit-normalized vectors they induce equivalent rankings, so operational support and model training may decide the choice.

## 8. What happens to distances in high-dimensional spaces?

Under many data distributions, pairwise distances concentrate: nearest and farthest neighbors become less distinguishable. This can weaken neighborhood methods. Possible responses include feature selection, dimensionality reduction, a better-matched metric, learned representations, and empirical evaluation of retrieval recall.

## 9. How does matrix multiplication appear in neural networks?

For input batch $X \in \mathbb{R}^{n \times d}$ and weights $W \in \mathbb{R}^{d \times k}$, a dense layer computes $H=XW+b$, producing $H \in \mathbb{R}^{n \times k}$. Each output unit is a weighted combination of input features; nonlinear activations between affine layers provide nonlinear capacity.

## 10. How does the dot product appear in Transformer attention?

$QK^T$ creates a matrix of query-key compatibility scores. Each entry is one dot product. Scores are scaled by $\sqrt{d_k}$ before softmax because their variance tends to grow with dimension; unscaled large logits can saturate softmax and reduce useful gradients.

## 11. Why should you avoid explicitly computing a matrix inverse?

Explicit inversion is generally more expensive and less numerically stable than solving the system. Use `numpy.linalg.solve(A, b)` for a square system and a least-squares solver or suitable decomposition for over- or underdetermined systems.

## 12. What is an ill-conditioned matrix?

It is a matrix for which small perturbations in inputs or floating-point arithmetic can cause large output changes. Causes include poorly scaled, highly correlated, or nearly dependent columns. Scaling, regularization, feature removal, and stable factorizations can reduce the problem.

## 13. What is the computational cost of standard dense matrix multiplication?

Multiplying $(m \times n)$ by $(n \times p)$ has conventional complexity $O(mnp)$. Real runtime also depends on memory layout, bandwidth, batching, hardware utilization, precision, sparsity, and optimized kernels.

## 14. How would you search one million document embeddings?

I would first define the metric based on the embedding model and offline task quality, then apply consistent normalization and model versioning. At that scale I would usually evaluate an approximate nearest-neighbor index with metadata filtering, retrieve top-$k$ candidates, and optionally rerank them. Selection among FAISS, pgvector, or a managed vector service depends on latency, recall, update frequency, filtering, cost, and operational constraints.

## 15. A similarity system performs well offline but poorly in production. What would you investigate?

Check embedding-model version skew, normalization consistency, query/document shape and dtype, zero or invalid vectors, index metric configuration, approximate-search recall, filtering, reduced-precision effects, and distribution drift. Also audit the offline set for leakage, duplicates, unrealistic queries, and a mismatch between similarity metrics and user outcomes.

## Interview-Ready Summary

Vectors numerically represent observations, documents, users, tokens, gradients, and parameters. Matrices either collect these vectors or transform them. Dot products measure alignment and magnitude, norms measure size, distances measure separation, and matrix multiplication applies or composes transformations. In production, these choices affect retrieval quality, tensor correctness, memory, latency, and numerical stability, so I choose metrics and representations based on model training, data geometry, and measured system requirements.
