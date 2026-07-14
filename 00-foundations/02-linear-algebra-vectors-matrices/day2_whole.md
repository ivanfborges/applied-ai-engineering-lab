# Day 2 — Linear Algebra I: Vectors, Matrices and Operations

## 1. Executive overview

Linear algebra is the computational language behind most Data Science, Machine Learning, Deep Learning, and Generative AI systems.

Models do not directly process concepts such as customers, documents, images, or tokens. These objects are converted into numerical structures:

* a customer becomes a feature vector;
* a dataset becomes a matrix;
* an image becomes a multidimensional array;
* a document becomes an embedding vector;
* a neural-network layer becomes a matrix transformation;
* transformer attention uses matrix multiplication;
* semantic retrieval uses dot products, cosine similarity, or distances.

A tabular dataset with (n) observations and (d) features is commonly represented as:

[
X \in \mathbb{R}^{n \times d}
]

where:

* (X) is the data matrix;
* (n) is the number of observations;
* (d) is the number of features;
* each row represents one observation;
* each column represents one feature.

A linear model computes:

[
\hat{y} = Xw + b
]

where:

* (w \in \mathbb{R}^{d}) is the weight vector;
* (b) is the bias term;
* (\hat{y}) is the prediction vector.

The same basic operations appear in transformer attention:

[
\operatorname{Attention}(Q,K,V)
===============================

\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
]

Although the expression looks advanced, it is fundamentally built from:

* vectors;
* matrices;
* dot products;
* transposition;
* matrix multiplication;
* normalization.

A senior practitioner should understand linear algebra at three levels:

1. **Computationally:** how to manipulate arrays correctly and efficiently.
2. **Geometrically:** how vectors encode direction, magnitude, distance, and similarity.
3. **Operationally:** how these choices affect latency, memory, numerical stability, retrieval quality, and system design.

---

## 2. Core intuition

A vector can represent:

* a point in space;
* a direction;
* a collection of measurements;
* an embedding of an object;
* a model parameter;
* a gradient.

For example:

[
x =
\begin{bmatrix}
30 \
8000 \
5
\end{bmatrix}
]

could represent:

* age: 30;
* monthly income: 8,000;
* years of experience: 5.

The same vector can also be interpreted geometrically as a point in three-dimensional space.

A matrix has two useful interpretations.

### Matrix as a collection of vectors

A dataset can be represented as:

[
X =
\begin{bmatrix}
30 & 8000 & 5 \
45 & 12000 & 15 \
25 & 5000 & 2
\end{bmatrix}
]

Each row is one observation vector.

### Matrix as a transformation

A matrix can transform a vector:

[
y = Ax
]

Depending on (A), the transformation can:

* rotate;
* scale;
* reflect;
* project;
* compress;
* expand;
* combine dimensions.

The main operations answer different geometric questions:

* **Dot product:** How aligned are two vectors?
* **Norm:** How large is a vector?
* **Distance:** How far apart are two vectors?
* **Matrix-vector multiplication:** How does a transformation affect one vector?
* **Matrix-matrix multiplication:** How are transformations composed or applied in batches?

A practical mental model is:

> Vectors represent information. Matrices organize information or transform it. Linear algebra defines how those representations interact.

---

## 3. Theoretical foundations

### 3.1 Scalars

A scalar is a single numerical value:

[
a \in \mathbb{R}
]

Examples:

* learning rate;
* probability;
* model weight;
* loss value;
* regularization coefficient;
* attention temperature.

---

### 3.2 Vectors

A vector is an ordered sequence of numbers:

[
x =
\begin{bmatrix}
x_1 \
x_2 \
\vdots \
x_d
\end{bmatrix}
\in \mathbb{R}^{d}
]

where:

* (d) is the vector dimension;
* (x_i) is the value in dimension (i).

Common AI examples include:

* feature vectors;
* word or document embeddings;
* neural-network parameters;
* gradients;
* probability distributions;
* sensor measurements.

A column vector has shape:

[
x \in \mathbb{R}^{d \times 1}
]

Its transpose is a row vector:

[
x^T \in \mathbb{R}^{1 \times d}
]

In NumPy, an array with shape `(d,)` is technically neither a two-dimensional row vector nor a two-dimensional column vector. This distinction matters when reasoning about matrix multiplication and broadcasting.

---

### 3.3 Matrices

A matrix is a rectangular arrangement of numbers:

[
A \in \mathbb{R}^{m \times n}
]

where:

* (m) is the number of rows;
* (n) is the number of columns;
* (a_{ij}) is the value at row (i), column (j).

Matrices can represent:

* datasets;
* neural-network weights;
* pairwise distances;
* embedding collections;
* graph adjacency;
* image data;
* linear transformations.

---

### 3.4 Vector addition

Two vectors of the same dimension can be added:

[
x+y =
\begin{bmatrix}
x_1+y_1 \
x_2+y_2 \
\vdots \
x_d+y_d
\end{bmatrix}
]

Geometrically, vector addition combines directions and magnitudes.

In machine learning, vector addition appears in:

* residual connections;
* gradient updates;
* bias addition;
* embedding composition;
* feature aggregation.

---

### 3.5 Scalar multiplication

Multiplying a vector by a scalar changes its magnitude:

[
\alpha x =
\begin{bmatrix}
\alpha x_1 \
\alpha x_2 \
\vdots \
\alpha x_d
\end{bmatrix}
]

where (\alpha) is a scalar.

Interpretation:

* (\alpha > 1): expands the vector;
* (0 < \alpha < 1): contracts the vector;
* (\alpha < 0): reverses its direction.

---

### 3.6 Transpose

The transpose exchanges rows and columns.

Given:

[
A =
\begin{bmatrix}
1 & 2 & 3 \
4 & 5 & 6
\end{bmatrix}
]

then:

[
A^T =
\begin{bmatrix}
1 & 4 \
2 & 5 \
3 & 6
\end{bmatrix}
]

If:

[
A \in \mathbb{R}^{m \times n}
]

then:

[
A^T \in \mathbb{R}^{n \times m}
]

Transposition is often required to align dimensions for matrix multiplication.

---

### 3.7 Elementwise multiplication

Elementwise multiplication multiplies corresponding positions:

[
A \odot B
]

The symbol (\odot) represents the Hadamard product.

Given:

[
A =
\begin{bmatrix}
1 & 2 \
3 & 4
\end{bmatrix}
\qquad
B =
\begin{bmatrix}
5 & 6 \
7 & 8
\end{bmatrix}
]

then:

[
A \odot B =
\begin{bmatrix}
5 & 12 \
21 & 32
\end{bmatrix}
]

In NumPy:

```python
A * B
```

This is not matrix multiplication.

---

### 3.8 Dot product

For two vectors (x,y \in \mathbb{R}^{d}):

[
x \cdot y
=========

# x^T y

\sum_{i=1}^{d}x_i y_i
]

The dot product returns a scalar.

Its geometric interpretation is:

[
x \cdot y
=========

|x|_2
|y|_2
\cos(\theta)
]

where:

* (|x|_2) is the Euclidean norm of (x);
* (|y|_2) is the Euclidean norm of (y);
* (\theta) is the angle between the vectors.

Interpretation:

* (x \cdot y > 0): similar general direction;
* (x \cdot y = 0): orthogonal vectors;
* (x \cdot y < 0): opposing directions.

The dot product incorporates both direction and magnitude.

---

### 3.9 Outer product

The outer product of two vectors produces a matrix:

[
xy^T
]

For:

[
x \in \mathbb{R}^{m}
\qquad
y \in \mathbb{R}^{n}
]

the result has shape:

[
xy^T \in \mathbb{R}^{m \times n}
]

Each element is:

[
(xy^T)_{ij}=x_i y_j
]

Outer products appear in:

* covariance matrices;
* gradient derivations;
* low-rank approximations;
* attention-related operations;
* matrix factorization.

---

### 3.10 Matrix multiplication

Suppose:

[
A \in \mathbb{R}^{m \times n}
]

and:

[
B \in \mathbb{R}^{n \times p}
]

Then:

[
C=AB
]

has shape:

[
C \in \mathbb{R}^{m \times p}
]

Each element is:

[
c_{ij}
======

\sum_{k=1}^{n}a_{ik}b_{kj}
]

Each output element is the dot product between:

* row (i) of (A);
* column (j) of (B).

Shape rule:

[
(m \times n)(n \times p)
\rightarrow
(m \times p)
]

Matrix multiplication is:

* associative:

[
(AB)C=A(BC)
]

* distributive:

[
A(B+C)=AB+AC
]

* generally not commutative:

[
AB \neq BA
]

---

### 3.11 Linear transformations

A matrix defines a linear transformation:

[
T(x)=Ax
]

A transformation is linear when:

[
T(x+y)=T(x)+T(y)
]

and:

[
T(\alpha x)=\alpha T(x)
]

A linear transformation always maps the origin to the origin:

[
T(0)=0
]

Neural-network layers commonly compute:

[
y=Ax+b
]

Because of the bias (b), this is technically an affine transformation rather than a strictly linear transformation.

---

### 3.12 Dense and sparse matrices

A dense matrix stores every value, including zeros.

A sparse matrix stores primarily:

* nonzero values;
* their row positions;
* their column positions.

Sparse representations are useful for:

* TF-IDF;
* bag-of-words;
* graph adjacency matrices;
* recommender systems;
* large one-hot encoded datasets.

A major production mistake is converting a very large sparse matrix to a dense representation without checking memory requirements.

---

## 4. Mathematical foundations

### 4.1 General (L_p) norm

A norm measures the size or magnitude of a vector.

The general (L_p) norm is:

[
|x|_p
=====

\left(
\sum_{i=1}^{d}|x_i|^p
\right)^{1/p}
]

where:

* (x) is a vector;
* (d) is the number of dimensions;
* (x_i) is component (i);
* (p) determines the type of norm.

---

### 4.2 L1 norm

[
|x|_1
=====

\sum_{i=1}^{d}|x_i|
]

The L1 norm is the sum of absolute values.

It is associated with:

* Manhattan distance;
* Lasso regularization;
* sparse solutions;
* robustness to large individual deviations compared with squared penalties.

---

### 4.3 L2 norm

[
|x|_2
=====

\sqrt{
\sum_{i=1}^{d}x_i^2
}
]

The L2 norm is the Euclidean length of a vector.

For a two-dimensional vector:

[
x=
\begin{bmatrix}
x_1 \
x_2
\end{bmatrix}
]

the length follows the Pythagorean theorem:

[
|x|_2
=====

\sqrt{x_1^2+x_2^2}
]

---

### 4.4 Infinity norm

[
|x|_{\infty}
============

\max_i |x_i|
]

The infinity norm is the largest absolute component of the vector.

It is useful when the maximum deviation matters more than aggregate deviation.

---

### 4.5 Frobenius norm

A common matrix norm is the Frobenius norm:

[
|A|_F
=====

\sqrt{
\sum_{i=1}^{m}
\sum_{j=1}^{n}
a_{ij}^2
}
]

It treats all matrix elements as if they belonged to one long vector.

It is commonly used for:

* measuring parameter magnitude;
* matrix reconstruction error;
* regularization;
* comparing weight matrices.

---

### 4.6 Euclidean distance

The Euclidean distance between two vectors is:

[
d(x,y)=|x-y|_2
]

Expanded:

[
d(x,y)
======

\sqrt{
\sum_{i=1}^{d}(x_i-y_i)^2
}
]

It measures straight-line distance.

Euclidean distance is sensitive to:

* feature scale;
* outliers;
* irrelevant dimensions;
* high dimensionality.

---

### 4.7 Manhattan distance

The Manhattan distance is:

[
d_1(x,y)
========

|x-y|_1
]

Expanded:

[
d_1(x,y)
========

\sum_{i=1}^{d}|x_i-y_i|
]

It measures the total coordinate-by-coordinate displacement.

It is called Manhattan distance because it resembles movement along a city grid rather than a direct diagonal path.

---

### 4.8 Cosine similarity

Cosine similarity measures angular alignment:

[
\operatorname{cosine}(x,y)
==========================

\frac{x \cdot y}
{|x|_2|y|_2}
]

Its theoretical range is:

[
[-1,1]
]

Interpretation:

* (1): same direction;
* (0): orthogonal;
* (-1): opposite directions.

For many embedding systems, vector direction carries more semantic meaning than vector magnitude. This is why cosine similarity is widely used in semantic search and RAG.

Cosine similarity is undefined if either vector is the zero vector.

---

### 4.9 Dot product and cosine similarity

Starting from:

[
x \cdot y
=========

|x|_2|y|_2\cos(\theta)
]

we isolate the cosine:

[
\cos(\theta)
============

\frac{x \cdot y}
{|x|_2|y|_2}
]

If both vectors are L2-normalized:

[
|x|_2=1
\qquad
|y|_2=1
]

then:

[
x \cdot y=\cos(\theta)
]

This means that dot-product search over unit-normalized embeddings is equivalent to cosine-similarity search.

---

### 4.10 Vector normalization

L2 normalization transforms a vector into a unit vector:

[
\hat{x}
=======

\frac{x}{|x|_2}
]

After normalization:

[
|\hat{x}|_2=1
]

Normalization:

* preserves direction;
* removes magnitude;
* makes cosine similarity equivalent to dot product.

This is often performed before indexing embeddings, depending on the vector database and model recommendations.

---

### 4.11 Projection

The scalar projection of (x) onto (y) is:

[
\operatorname{comp}_y(x)
========================

\frac{x \cdot y}{|y|_2}
]

The vector projection is:

[
\operatorname{proj}_y(x)
========================

\frac{x \cdot y}{y \cdot y}y
]

This represents the component of (x) that lies in the direction of (y).

Projection is fundamental to:

* least squares;
* linear regression;
* PCA;
* vector decomposition;
* recommender systems.

---

### 4.12 Matrix-vector multiplication

For:

[
A \in \mathbb{R}^{m \times n}
]

and:

[
x \in \mathbb{R}^{n}
]

the result is:

[
y=Ax
]

with:

[
y \in \mathbb{R}^{m}
]

Each component is:

[
y_i
===

\sum_{j=1}^{n}a_{ij}x_j
]

Each row of (A) computes a weighted combination of the input vector.

A neural-network neuron performs this same basic operation.

---

### 4.13 Batched computation

Suppose:

[
X \in \mathbb{R}^{n \times d}
]

contains (n) observations and (d) features.

A linear model has:

[
w \in \mathbb{R}^{d}
]

All predictions can be computed at once:

[
\hat{y}=Xw
]

The shapes are:

[
(n \times d)(d)
\rightarrow
(n)
]

This avoids Python-level loops and allows optimized numerical libraries to use:

* vectorized CPU instructions;
* multithreading;
* GPUs;
* TPUs.

---

### 4.14 Composition of transformations

Suppose:

[
y=Bx
]

and:

[
z=Ay
]

Substituting:

[
z=A(Bx)
]

By associativity:

[
z=(AB)x
]

The matrix (B) acts first, followed by (A).

This is an important interview point:

> In the expression (ABx), the transformation closest to the vector is applied first.

---

### 4.15 Computational complexity

For:

[
A \in \mathbb{R}^{m \times n}
]

and:

[
B \in \mathbb{R}^{n \times p}
]

standard dense matrix multiplication has approximate complexity:

[
O(mnp)
]

Large neural networks spend a major portion of training and inference time performing matrix multiplications.

This explains the importance of:

* GPU and TPU acceleration;
* quantization;
* batching;
* lower precision;
* optimized kernels;
* sparsity;
* low-rank factorization.

---

## 5. Practical applicability

### Tabular machine learning

The feature matrix is:

[
X \in \mathbb{R}^{n \times d}
]

Linear regression, logistic regression, and neural networks compute weighted combinations of features using matrix multiplication.

Distance-based algorithms such as KNN and K-Means operate directly on vectors and distances.

---

### Embeddings and semantic retrieval

Suppose document embeddings are stored as:

[
D \in \mathbb{R}^{n \times d}
]

and the query embedding is:

[
q \in \mathbb{R}^{d}
]

Dot-product scores for every document can be calculated as:

[
s=Dq
]

where:

[
s \in \mathbb{R}^{n}
]

At small scale, this exact calculation may be acceptable.

At large scale, production systems commonly use approximate nearest-neighbor indexes such as:

* FAISS;
* pgvector indexes;
* Vertex AI Vector Search;
* dedicated vector databases.

---

### Neural networks

A dense layer computes:

[
H=XW+b
]

where:

* (X) is the input batch;
* (W) is the weight matrix;
* (b) is the bias;
* (H) is the layer output.

The matrix operation itself is linear or affine. Nonlinear modeling emerges through activation functions and multiple layers.

---

### Transformer attention

Transformer attention computes:

[
QK^T
]

Each element in this matrix is a dot product between:

* one query vector;
* one key vector.

The result is a matrix of compatibility scores.

After scaling and softmax, the weights are applied to the value matrix (V).

---

### Recommendation systems

Users and items can be represented as latent vectors:

[
p_u \in \mathbb{R}^{d}
]

[
q_i \in \mathbb{R}^{d}
]

A preference score can be estimated as:

[
\hat{r}_{ui}=p_u^Tq_i
]

where:

* (p_u) is the user embedding;
* (q_i) is the item embedding;
* (\hat{r}_{ui}) is the predicted affinity.

---

### Computer vision

Images are matrices or higher-dimensional tensors.

Linear algebra appears in:

* geometric transformations;
* color transformations;
* convolution implementations;
* projections;
* image compression;
* dimensionality reduction.

---

### Graph systems

A graph can be represented by an adjacency matrix:

[
A_{ij}
======

\begin{cases}
1, & \text{if nodes } i \text{ and } j \text{ are connected} \
0, & \text{otherwise}
\end{cases}
]

Matrix multiplication can propagate information across graph neighborhoods.

---

### When linear operations are insufficient

A single linear or affine transformation:

[
y=Wx+b
]

cannot model arbitrary nonlinear relationships.

For more complex patterns, we need:

* nonlinear activation functions;
* kernels;
* decision trees;
* engineered nonlinear features;
* multilayer neural networks.

Linear algebra remains the computational foundation, even when the model itself is nonlinear.

---

### Main trade-offs

* Dense matrices are accelerator-friendly but memory-intensive.
* Sparse matrices save memory but are not equally efficient for every operation.
* Cosine similarity removes magnitude information.
* Dot product allows vector magnitude to influence ranking.
* Higher dimensions can capture more information but increase cost and noise.
* Reduced precision improves throughput but may affect numerical stability.
* Exact similarity search maximizes accuracy but may not satisfy latency requirements.
* Approximate search improves speed but can reduce recall.

---

## 6. Common pitfalls and mistakes

### Confusing elementwise and matrix multiplication

In NumPy:

```python
A * B
```

performs elementwise multiplication.

```python
A @ B
```

performs matrix multiplication.

Both can produce valid outputs, which makes this mistake especially dangerous.

---

### Ignoring shapes

Before multiplying matrices, reason about the dimensions:

[
(n \times d)(d \times k)
========================

(n \times k)
]

Many machine-learning errors are shape errors rather than mathematical errors.

A senior candidate should be able to explain the shape of every intermediate tensor.

---

### Assuming matrix multiplication is commutative

In general:

[
AB \neq BA
]

The operations may:

* produce different results;
* produce different shapes;
* be invalid in one order.

---

### Mixing row-vector and column-vector conventions

Mathematical derivations often use column vectors:

[
y=Ax
]

Datasets usually store observations as rows.

If rows are observations and (A) follows the column-vector convention, the transformation is commonly applied as:

```python
X_transformed = X @ A.T
```

---

### Using Euclidean distance without feature scaling

Suppose one feature ranges from 0 to 1 and another from 0 to 1,000,000.

The second feature will dominate the distance calculation.

This can distort:

* KNN;
* K-Means;
* anomaly detection;
* similarity search.

---

### Data leakage during scaling

Incorrect:

```python
scaler.fit(X)
X_train, X_test = train_test_split(X)
```

Correct:

```python
X_train, X_test = train_test_split(X)

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler must be fitted only on training data.

---

### Treating cosine similarity and dot product as automatically equivalent

They are equivalent only when vectors are unit-normalized.

Without normalization, dot product incorporates magnitude.

---

### Ignoring zero vectors

Cosine similarity divides by vector norms.

For a zero vector:

[
|x|_2=0
]

The calculation becomes undefined.

Production code should explicitly handle zero vectors.

---

### Computing matrix inverses unnecessarily

To solve:

[
Ax=b
]

avoid:

```python
x = np.linalg.inv(A) @ b
```

Prefer:

```python
x = np.linalg.solve(A, b)
```

Direct solvers are generally more stable and efficient.

---

### Ignoring singular or ill-conditioned matrices

A singular matrix has no inverse.

An ill-conditioned matrix may be technically invertible but highly sensitive to small perturbations.

Typical causes include:

* strongly correlated features;
* duplicated columns;
* features with very different scales;
* near-linear dependence.

---

### Accidental broadcasting

NumPy and PyTorch may expand arrays automatically.

For example:

```python
matrix + vector
```

may add the vector to every row.

Broadcasting is useful, but it can silently hide shape mistakes.

---

### Densifying large sparse matrices

This can be dangerous:

```python
dense = sparse_matrix.toarray()
```

A sparse matrix that occupies hundreds of megabytes may require many gigabytes when converted to dense form.

---

### Ignoring high-dimensional distance concentration

In high-dimensional spaces, distances between points can become increasingly similar.

This can weaken:

* nearest-neighbor methods;
* Euclidean clustering;
* outlier detection.

Potential responses include:

* feature selection;
* dimensionality reduction;
* cosine similarity;
* learned metrics;
* domain-specific representations.

---

### Assuming vector magnitude always has semantic meaning

Embedding magnitude may be:

* meaningful;
* a training artifact;
* model-specific;
* irrelevant.

The similarity metric should match the embedding model’s training objective and be validated empirically.

---

## 7. Important comparisons

### Dot product versus cosine similarity

| Aspect         | Dot product                                      | Cosine similarity                  |
| -------------- | ------------------------------------------------ | ---------------------------------- |
| Formula        | (x \cdot y)                                      | (\frac{x \cdot y}{|x||y|})         |
| Uses magnitude | Yes                                              | No                                 |
| Uses direction | Yes                                              | Yes                                |
| Range          | Unbounded                                        | Usually ([-1,1])                   |
| Common uses    | Attention, recommendations, normalized retrieval | Semantic similarity                |
| Main risk      | Large-norm vectors may dominate                  | Magnitude information is discarded |

Use dot product when:

* magnitude is meaningful;
* the model was trained with dot-product similarity;
* normalized vectors are already used.

Use cosine similarity when:

* direction matters more than magnitude;
* comparing text or semantic embeddings;
* vector scales vary but should not affect ranking.

---

### Euclidean distance versus cosine similarity

| Aspect                 | Euclidean distance              | Cosine similarity            |
| ---------------------- | ------------------------------- | ---------------------------- |
| Measures               | Absolute spatial separation     | Angular alignment            |
| Sensitive to magnitude | Yes                             | No                           |
| Sensitive to scale     | Strongly                        | Reduced after normalization  |
| Better value           | Smaller                         | Larger                       |
| Common use             | KNN, clustering, geometric data | Text and semantic embeddings |

For normalized vectors:

[
|x-y|_2^2
=========

2-2(x \cdot y)
]

Therefore, cosine similarity and Euclidean distance produce equivalent rankings when all vectors have unit norm.

---

### L1 norm versus L2 norm

| Aspect                      | L1                                | L2                              |
| --------------------------- | --------------------------------- | ------------------------------- |
| Definition                  | Sum of absolute values            | Square root of sum of squares   |
| Sensitivity to large values | Lower                             | Higher                          |
| Regularization effect       | Encourages sparsity               | Smoothly shrinks weights        |
| Geometry                    | Diamond-shaped unit ball          | Circular or spherical unit ball |
| Typical use                 | Sparse models, Manhattan distance | Euclidean geometry, Ridge       |

---

### Elementwise multiplication versus matrix multiplication

| Elementwise multiplication    | Matrix multiplication            |
| ----------------------------- | -------------------------------- |
| Multiplies matching positions | Computes row-column dot products |
| NumPy operator: `*`           | NumPy operator: `@`              |
| Often preserves shape         | Produces a composed output shape |
| Used in masking and gates     | Used in linear transformations   |

---

### Dense versus sparse representations

| Dense                             | Sparse                           |
| --------------------------------- | -------------------------------- |
| Stores all values                 | Stores mainly nonzero values     |
| Efficient for embeddings and GPUs | Efficient for highly sparse data |
| Higher memory usage               | Lower memory usage               |
| Common in neural networks         | Common in TF-IDF and graphs      |

---

### NumPy versus PyTorch

Use NumPy for:

* CPU numerical computing;
* educational examples;
* linear algebra experimentation;
* general scientific computing.

Use PyTorch for:

* automatic differentiation;
* GPU execution;
* neural-network training;
* tensor-based production pipelines.

The underlying operations remain the same.

---

## 8. Practical Python example

This example demonstrates:

* vectors and matrices;
* dot-product ranking;
* cosine similarity;
* Euclidean distance;
* normalization;
* matrix transformations;
* transformation order.

Install the dependencies:

```bash
pip install numpy matplotlib
```

Create `example.py`:

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def cosine_similarity(
    matrix: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between every row of a matrix
    and a single vector.
    """
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)

    if vector_norm == 0:
        raise ValueError("The query vector must not be zero.")

    if np.any(matrix_norms == 0):
        raise ValueError(
            "The matrix must not contain zero vectors."
        )

    dot_products = matrix @ vector

    return dot_products / (matrix_norms * vector_norm)


def main() -> None:
    # Each row represents a simplified document embedding.
    documents = np.array(
        [
            [0.9, 0.8],
            [0.2, 1.0],
            [-0.8, -0.6],
            [1.5, 1.2],
        ],
        dtype=float,
    )

    query = np.array([1.0, 0.9], dtype=float)

    print("Document matrix shape:", documents.shape)
    print("Query shape:", query.shape)

    # One dot product is computed for every document.
    dot_scores = documents @ query

    cosine_scores = cosine_similarity(
        documents,
        query,
    )

    # Broadcasting subtracts the query from every row.
    euclidean_distances = np.linalg.norm(
        documents - query,
        axis=1,
    )

    print("\nDot-product scores:")
    print(dot_scores)

    print("\nCosine similarities:")
    print(cosine_scores)

    print("\nEuclidean distances:")
    print(euclidean_distances)

    print("\nRanking by dot product:")
    print(np.argsort(-dot_scores))

    print("\nRanking by cosine similarity:")
    print(np.argsort(-cosine_scores))

    print("\nRanking by Euclidean distance:")
    print(np.argsort(euclidean_distances))

    # Normalize each vector to unit length.
    normalized_documents = documents / np.linalg.norm(
        documents,
        axis=1,
        keepdims=True,
    )

    normalized_query = query / np.linalg.norm(query)

    normalized_dot_scores = (
        normalized_documents @ normalized_query
    )

    print("\nDot product after normalization:")
    print(normalized_dot_scores)

    print("\nNormalized dot equals cosine:")
    print(
        np.allclose(
            normalized_dot_scores,
            cosine_scores,
        )
    )

    # Scaling transformation.
    scaling = np.array(
        [
            [2.0, 0.0],
            [0.0, 0.5],
        ]
    )

    # Rotation transformation.
    angle = np.deg2rad(30)

    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    # In A @ B @ x, B acts first.
    scale_then_rotate = rotation @ scaling
    rotate_then_scale = scaling @ rotation

    # Rows contain observations, so we apply the transpose.
    transformed_a = documents @ scale_then_rotate.T
    transformed_b = documents @ rotate_then_scale.T

    print("\nAre the transformation orders equivalent?")
    print(np.allclose(transformed_a, transformed_b))

    plt.scatter(
        documents[:, 0],
        documents[:, 1],
        label="Original",
    )

    plt.scatter(
        transformed_a[:, 0],
        transformed_a[:, 1],
        marker="x",
        label="Scale then rotate",
    )

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Matrix as a Linear Transformation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

Run:

```bash
python example.py
```

### Shape interpretation

The operation:

```python
documents @ query
```

has input shapes:

```text
documents: (4, 2)
query:     (2,)
```

and output shape:

```text
(4,)
```

Each output value is the dot product between one document vector and the query.

### What to observe

* Dot product and cosine similarity may rank vectors differently.
* Vector magnitude strongly affects dot product.
* L2-normalized dot product equals cosine similarity.
* Euclidean distance depends on both direction and magnitude.
* Scaling followed by rotation differs from rotation followed by scaling.

---

## 9. From-scratch implementation

Create `from_scratch.py`:

```python
from __future__ import annotations

from math import sqrt
from typing import Sequence


Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


def dot_product(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError(
            "Vectors must have the same dimension."
        )

    return sum(
        x * y
        for x, y in zip(a, b)
    )


def l1_norm(vector: Vector) -> float:
    return sum(abs(value) for value in vector)


def l2_norm(vector: Vector) -> float:
    return sqrt(dot_product(vector, vector))


def euclidean_distance(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError(
            "Vectors must have the same dimension."
        )

    difference = [
        x - y
        for x, y in zip(a, b)
    ]

    return l2_norm(difference)


def cosine_similarity(a: Vector, b: Vector) -> float:
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError(
            "Cosine similarity is undefined "
            "for zero vectors."
        )

    return dot_product(a, b) / (norm_a * norm_b)


def transpose(matrix: Matrix) -> list[list[float]]:
    if not matrix or not matrix[0]:
        raise ValueError("Matrix must not be empty.")

    number_of_columns = len(matrix[0])

    if any(
        len(row) != number_of_columns
        for row in matrix
    ):
        raise ValueError(
            "All matrix rows must have the same length."
        )

    return [
        [
            matrix[row_index][column_index]
            for row_index in range(len(matrix))
        ]
        for column_index in range(number_of_columns)
    ]


def matrix_multiply(
    a: Matrix,
    b: Matrix,
) -> list[list[float]]:
    if not a or not b:
        raise ValueError("Matrices must not be empty.")

    a_columns = len(a[0])
    b_rows = len(b)
    b_columns = len(b[0])

    if any(len(row) != a_columns for row in a):
        raise ValueError(
            "Matrix A has inconsistent row lengths."
        )

    if any(len(row) != b_columns for row in b):
        raise ValueError(
            "Matrix B has inconsistent row lengths."
        )

    if a_columns != b_rows:
        raise ValueError(
            "The number of columns in A must equal "
            "the number of rows in B."
        )

    b_transposed = transpose(b)

    return [
        [
            dot_product(a_row, b_column)
            for b_column in b_transposed
        ]
        for a_row in a
    ]


def main() -> None:
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]

    print("Dot product:", dot_product(x, y))
    print("L1 norm of x:", l1_norm(x))
    print("L2 norm of x:", l2_norm(x))
    print(
        "Euclidean distance:",
        euclidean_distance(x, y),
    )
    print(
        "Cosine similarity:",
        cosine_similarity(x, y),
    )

    a = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]

    b = [
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ]

    print("Matrix multiplication:")
    print(matrix_multiply(a, b))


if __name__ == "__main__":
    main()
```

### How the implementation works

The matrix multiplication function:

1. validates the dimensions;
2. transposes matrix (B);
3. converts every column of (B) into an iterable vector;
4. computes the dot product between every row of (A) and every column of (B);
5. constructs the output matrix.

This implementation is educational.

Production libraries use optimized low-level implementations such as:

* BLAS;
* SIMD instructions;
* multithreading;
* GPU kernels;
* memory-layout optimizations.

---

## 10. Suggested experiments

### Experiment 1 — Change magnitude without changing direction

Multiply one document by 10:

```python
documents[0] *= 10
```

Compare:

* dot product;
* cosine similarity;
* Euclidean distance.

Expected insight:

* cosine similarity remains approximately unchanged;
* dot product increases significantly;
* Euclidean distance changes because magnitude changed.

---

### Experiment 2 — Normalize all embeddings

Normalize documents and query:

```python
normalized_documents = documents / np.linalg.norm(
    documents,
    axis=1,
    keepdims=True,
)

normalized_query = query / np.linalg.norm(query)
```

Verify:

```python
normalized_documents @ normalized_query
```

matches cosine similarity.

---

### Experiment 3 — Introduce feature-scale imbalance

Multiply the first dimension by 1,000:

```python
documents[:, 0] *= 1000
query[0] *= 1000
```

Observe how Euclidean distance becomes dominated by the first feature.

Connect this behavior to:

* KNN;
* K-Means;
* anomaly detection;
* tabular feature scaling.

---

### Experiment 4 — Reverse transformation order

Compare:

```python
rotation @ scaling
```

with:

```python
scaling @ rotation
```

Plot both results.

Explain geometrically why:

[
RS \neq SR
]

---

### Experiment 5 — Explore high dimensionality

Generate random vectors with dimensions:

```text
2
10
100
1000
```

For each dimensionality, calculate:

* nearest distance;
* farthest distance;
* ratio between nearest and farthest distances.

Observe whether the distances become less distinguishable.

---

## 11. Senior interview questions

### 1. What is the geometric interpretation of the dot product?

The dot product measures alignment between two vectors while also incorporating their magnitudes:

[
x \cdot y
=========

|x|_2
|y|_2
\cos(\theta)
]

A large positive value indicates strong alignment, zero indicates orthogonality, and a negative value indicates opposing directions.

It can also be interpreted as measuring how much one vector projects onto another.

---

### 2. What is the difference between dot product and cosine similarity?

Dot product depends on both magnitude and direction.

Cosine similarity divides the dot product by the product of the vector norms, removing the effect of magnitude.

For normalized vectors:

[
x \cdot y
=========

\operatorname{cosine}(x,y)
]

The appropriate metric depends on whether magnitude is meaningful and how the representation model was trained.

---

### 3. Why must the inner dimensions match during matrix multiplication?

Every output element is a dot product between a row of the first matrix and a column of the second.

Those two vectors must have the same number of components.

Therefore:

[
(m \times n)(n \times p)
]

is valid because the inner dimensions are both (n).

---

### 4. Why is matrix multiplication not commutative?

Matrices often represent transformations, and transformation order matters.

Rotating and then scaling usually produces a different result from scaling and then rotating.

Additionally:

* (AB) and (BA) may have different shapes;
* one operation may be valid while the other is not.

---

### 5. What is the difference between a linear and an affine transformation?

A linear transformation has the form:

[
y=Ax
]

and maps the origin to the origin.

An affine transformation has the form:

[
y=Ax+b
]

The bias (b) allows translation.

Most neural-network “linear layers” are technically affine layers.

---

### 6. Why is feature scaling important for distance-based models?

Distance is calculated directly from numerical coordinates.

Features with larger scales contribute more strongly, even if they are not more important.

Scaling must be fitted only on training data to avoid data leakage.

---

### 7. When would you prefer cosine similarity over Euclidean distance?

Cosine similarity is useful when vector direction matters more than magnitude.

Common examples include:

* text embeddings;
* document retrieval;
* semantic clustering;
* sparse text representations.

However, the metric should match the embedding model’s training objective and be validated against the business task.

---

### 8. What happens to distance metrics in high-dimensional spaces?

Distances often become less distinguishable as dimensionality increases.

The nearest and farthest points may have increasingly similar distances.

This can reduce the effectiveness of neighborhood-based methods.

Possible responses include:

* dimensionality reduction;
* feature selection;
* cosine similarity;
* learned metrics;
* domain-specific embeddings.

---

### 9. How is matrix multiplication used in neural networks?

For:

[
X \in \mathbb{R}^{n \times d}
]

and:

[
W \in \mathbb{R}^{d \times k}
]

a dense layer computes:

[
H=XW+b
]

The output shape is:

[
H \in \mathbb{R}^{n \times k}
]

Each output unit computes a weighted combination of the input features.

---

### 10. How does dot product appear in transformer attention?

Queries and keys are compared using:

[
QK^T
]

Each element is the dot product between one query and one key.

This produces pairwise compatibility scores before scaling and softmax.

The resulting attention weights are then applied to (V).

---

### 11. Why is attention divided by (\sqrt{d_k})?

As the key dimension (d_k) increases, dot-product variance tends to increase.

Large attention logits can push softmax into saturated regions, producing very small gradients.

Scaling by:

[
\sqrt{d_k}
]

helps keep the score distribution numerically stable.

---

### 12. Why should you avoid explicitly computing a matrix inverse?

Explicit inversion is generally less stable and more computationally expensive than solving the system directly.

Instead of:

```python
x = np.linalg.inv(A) @ b
```

prefer:

```python
x = np.linalg.solve(A, b)
```

For least-squares problems, use:

```python
np.linalg.lstsq(A, b, rcond=None)
```

---

### 13. What is an ill-conditioned matrix?

An ill-conditioned matrix is highly sensitive to small perturbations.

Small changes in input data or floating-point calculations can cause large changes in the result.

Common causes include:

* correlated features;
* highly different scales;
* duplicate features;
* nearly linearly dependent columns.

Potential mitigations include:

* standardization;
* regularization;
* feature removal;
* numerically stable decompositions.

---

### 14. How would you search one million document embeddings?

I would usually avoid exact comparison against all one million embeddings for every request unless latency and infrastructure allowed it.

A production design would include:

1. consistent embedding generation;
2. normalization when required;
3. an approximate nearest-neighbor index;
4. metadata filters;
5. top-(k) candidate retrieval;
6. optional reranking;
7. offline recall, quality, cost, and latency evaluation.

Possible tools include FAISS, pgvector, or Vertex AI Vector Search.

The final choice depends on:

* corpus size;
* update frequency;
* filter complexity;
* latency target;
* recall target;
* operational cost.

---

### 15. What is the computational cost of dense matrix multiplication?

For:

[
A \in \mathbb{R}^{m \times n}
]

and:

[
B \in \mathbb{R}^{n \times p}
]

the standard complexity is approximately:

[
O(mnp)
]

Real performance also depends on:

* hardware;
* matrix layout;
* memory bandwidth;
* batch size;
* numerical precision;
* sparsity;
* optimized kernels.

---

### 16. A retrieval system performs well offline but poorly in production. What linear-algebra-related issues would you investigate?

I would investigate:

* inconsistent embedding-model versions;
* inconsistent normalization;
* mismatch between index metric and evaluation metric;
* incorrect tensor shapes;
* zero or invalid vectors;
* reduced approximate-search recall;
* quantization effects;
* precision changes;
* embedding distribution drift;
* metadata filters removing relevant candidates;
* offline leakage or duplicated documents.

---

## 12. Interview-ready explanation

Vectors are numerical representations of objects such as observations, documents, users, tokens, model parameters, or gradients. Matrices organize collections of vectors or represent transformations between vector spaces.

The main operations have direct geometric interpretations. The dot product measures vector alignment while also considering magnitude, norms measure vector size, distances measure separation, and matrix multiplication applies or composes transformations.

These concepts appear in linear models, embedding retrieval, neural-network layers, recommendation systems, PCA, and transformer attention.

In a real project, I use them when preparing feature matrices, choosing a similarity metric for RAG, validating tensor shapes, implementing batched inference, diagnosing scaling problems, and evaluating memory or numerical-stability trade-offs. The correct choice depends on whether magnitude matters, whether the data is dense or sparse, the dimensionality, the training objective, and the system’s latency and accuracy requirements.

---

## 13. GitHub file structure

```text
02-linear-algebra-vectors-matrices/
├── README.md
├── notes.md
├── notebook.ipynb
├── example.py
├── from_scratch.py
├── interview_questions.md
└── references.md
```

### `README.md`

Short overview, execution instructions, concepts, and takeaways.

### `notes.md`

Detailed theory, formulas, geometric interpretations, comparisons, and production considerations.

### `notebook.ipynb`

Interactive experiments and visualizations.

### `example.py`

Clean executable NumPy example.

### `from_scratch.py`

Educational implementations of:

* dot product;
* norms;
* distance;
* cosine similarity;
* transpose;
* matrix multiplication.

### `interview_questions.md`

Conceptual, mathematical, practical, and system-design questions.

### `references.md`

Books, official documentation, courses, and relevant papers.

---

## 14. Suggested `README.md` content

````markdown
# Linear Algebra I: Vectors, Matrices and Operations

## Objective

This module reviews the linear algebra foundations used throughout
Data Science, Machine Learning, Deep Learning and Applied AI systems.

The goal is to connect mathematical definitions with practical
operations used in feature matrices, embedding retrieval, neural
networks and production AI systems.

## Concepts Covered

- Vectors and matrices
- Shapes and transposition
- Vector addition and scalar multiplication
- Dot product and geometric alignment
- L1 and L2 norms
- Euclidean and Manhattan distances
- Cosine similarity
- Matrix multiplication
- Linear and affine transformations
- Dense and sparse representations
- Numerical and production considerations

## Practical Examples

The examples demonstrate:

- Ranking simplified document embeddings
- Comparing dot product, cosine similarity and Euclidean distance
- L2-normalizing vectors
- Applying scaling and rotation matrices
- Understanding transformation order
- Implementing core operations from scratch

## Project Structure

```text
.
├── README.md
├── notes.md
├── notebook.ipynb
├── example.py
├── from_scratch.py
├── interview_questions.md
└── references.md
````

## Requirements

* Python 3.10+
* NumPy
* Matplotlib

Install the dependencies:

```bash
pip install numpy matplotlib
```

## Running the Examples

Run the practical example:

```bash
python example.py
```

Run the educational implementation:

```bash
python from_scratch.py
```

## Key Takeaways

* Vectors represent observations, embeddings and model parameters.
* Matrices organize data and implement linear transformations.
* Dot product combines vector alignment and magnitude.
* Cosine similarity removes the effect of vector magnitude.
* Distance metrics must be selected according to data geometry.
* Matrix multiplication depends on shape and is not commutative.
* Scaling and normalization can materially change model behavior.
* Linear algebra choices affect retrieval quality, memory, latency
  and numerical stability.

## Applied AI Connections

These concepts appear directly in:

* semantic search and RAG;
* transformer attention;
* recommendation systems;
* neural-network layers;
* vector databases;
* dimensionality reduction;
* batched model inference.

```

---

## 15. LinkedIn post idea

Nos últimos estudos, revisitei um fundamento que aparece em praticamente todo sistema moderno de dados e IA: vetores e matrizes.

Na prática, uma observação de uma base, um documento convertido em embedding, os parâmetros de um modelo e até os tokens processados por um transformer acabam representados numericamente.

O ponto mais interessante é perceber que operações aparentemente básicas têm impacto direto em decisões de arquitetura:

- produto escalar mede alinhamento, mas também considera magnitude;
- similaridade de cosseno compara principalmente a direção dos vetores;
- distância euclidiana pode ser dominada pela escala das variáveis;
- multiplicação de matrizes permite aplicar transformações e processar grandes lotes de dados de forma eficiente.

Essas diferenças não são apenas matemáticas. Elas influenciam a qualidade de um sistema de busca semântica, o comportamento de um modelo, o consumo de memória e a latência em produção.

Documentei no GitHub a parte teórica, exemplos com NumPy, uma implementação simplificada do zero e algumas perguntas comuns de entrevistas técnicas.

#DataScience #MachineLearning #AIEngineering #LinearAlgebra #AppliedAI

---

## 16. 30–60 minute checklist

### 30-minute essential path

- [ ] **5 minutes:** Read the executive overview and core intuition.
- [ ] **7 minutes:** Review dot product, norms, distances, cosine similarity, and matrix multiplication.
- [ ] **8 minutes:** Run `example.py` and inspect the shapes and rankings.
- [ ] **5 minutes:** Explain aloud the difference between dot product, cosine similarity, and Euclidean distance.
- [ ] **5 minutes:** Answer interview questions 1, 2, 3, 4, and 14 without reading the answers.

### 45-minute recommended path

- [ ] Complete the 30-minute path.
- [ ] **5 minutes:** Run `from_scratch.py`.
- [ ] **5 minutes:** Change the magnitude of one embedding and compare all metrics.
- [ ] **5 minutes:** Rewrite the interview-ready explanation in your own words.

### 60-minute complete path

- [ ] Complete the 45-minute path.
- [ ] **5 minutes:** Compare scale-then-rotate with rotate-then-scale.
- [ ] **5 minutes:** Create the GitHub folder and initial files.
- [ ] **5 minutes:** Add two examples from your own production experience to `notes.md`, such as:
  - embedding similarity in a RAG pipeline;
  - matrix shapes in transformer attention or batched inference.

### Completion criteria

By the end of the session, you should be able to:

- [ ] explain vectors and matrices geometrically and computationally;
- [ ] calculate and interpret a dot product;
- [ ] distinguish L1 norm, L2 norm, Euclidean distance, and cosine similarity;
- [ ] validate matrix multiplication using shapes;
- [ ] explain why transformation order matters;
- [ ] connect linear algebra to embeddings, RAG, neural networks, and attention;
- [ ] identify scaling, normalization, broadcasting, and numerical-stability risks;
- [ ] implement the main operations using NumPy and simplified Python;
- [ ] give a mature two-minute interview answer about the topic.
```
