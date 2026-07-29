# Day 4 — Linear Algebra III: Eigenvalues, Eigenvectors, PCA and SVD

## 1. Executive overview

Eigenvalues, eigenvectors, Principal Component Analysis, and Singular Value Decomposition are different parts of the same central idea:

> Finding directions that reveal the most important structure in a linear transformation or dataset.

They appear throughout Data Science and AI Engineering:

* PCA for dimensionality reduction, visualization, denoising, compression, and preprocessing.
* SVD for low-rank approximation, recommendation systems, information retrieval, latent semantic analysis, image compression, and matrix factorization.
* Eigenvalues for understanding stability, covariance structure, optimization curvature, graph algorithms, Markov chains, and dynamical systems.
* Eigenvectors for identifying directions preserved by a linear transformation.
* Truncated and randomized SVD for processing large sparse matrices.
* Low-rank approximations for compressing models, embeddings, features, and attention-related matrices.

For senior-level understanding, the most important connection is:

[
X_c = U\Sigma V^\top
]

where (X_c) is a centered dataset.

The principal directions of PCA are the columns of (V), while the variance captured by each component is related to the squared singular values:

[
\lambda_i = \frac{\sigma_i^2}{n-1}
]

Here:

* (\lambda_i) is the (i)-th eigenvalue of the covariance matrix;
* (\sigma_i) is the (i)-th singular value of the centered data matrix;
* (n) is the number of observations.

This means PCA can be computed either through eigendecomposition of the covariance matrix or directly through SVD of the centered data.

In production, SVD is usually the preferred numerical route.

---

## 2. Core intuition

Imagine a cloud of points in a high-dimensional space.

The original coordinate axes may not align with the true structure of the data. For example, two variables may move together, producing an elongated diagonal cloud.

PCA asks:

> Along which direction does this cloud vary the most?

The first principal component is the direction with the greatest variance.

The second principal component is the direction with the next greatest variance, under the constraint that it must be orthogonal to the first.

This continues until all dimensions have been represented.

### Geometric interpretation

PCA rotates the coordinate system so that:

* the first axis follows the longest direction of the data;
* the second follows the next longest orthogonal direction;
* later axes describe progressively smaller variation.

Dimensionality reduction happens by discarding the directions with low variance.

This is not merely deleting columns. PCA creates new features as linear combinations of the original ones:

[
z_1 = w_{11}x_1 + w_{21}x_2 + \cdots + w_{d1}x_d
]

where:

* (z_1) is the first principal component score;
* (x_j) is the (j)-th original feature;
* (w_{j1}) is the contribution of feature (j) to the first component;
* (d) is the number of original features.

### Eigenvector intuition

An eigenvector is a direction that a linear transformation does not rotate away from itself.

It may be stretched, compressed, or reversed, but its direction remains aligned.

For a matrix (A):

[
A v = \lambda v
]

where:

* (A) is a square matrix representing a linear transformation;
* (v) is an eigenvector;
* (\lambda) is the associated eigenvalue.

The eigenvalue tells us how strongly the transformation stretches or compresses that direction.

### SVD intuition

SVD generalizes this idea to any rectangular matrix.

It decomposes a transformation into three operations:

1. Rotate or reorient the input space using (V^\top).
2. Scale the resulting directions using (\Sigma).
3. Rotate or reorient the output space using (U).

[
X = U\Sigma V^\top
]

A useful mental model is:

> SVD finds the most important input directions, quantifies their strength, and maps them to corresponding output directions.

---

## 3. Theoretical foundations

### 3.1 Eigenvalues and eigenvectors

For a square matrix (A \in \mathbb{R}^{d \times d}), a nonzero vector (v) is an eigenvector when:

[
A v = \lambda v
]

Rearranging:

[
(A-\lambda I)v=0
]

For this system to have a nonzero solution, the matrix (A-\lambda I) must be singular:

[
\det(A-\lambda I)=0
]

This equation is called the characteristic equation.

Its solutions are the eigenvalues of (A).

### 3.2 Eigendecomposition

When a matrix has enough linearly independent eigenvectors, it can be decomposed as:

[
A = Q\Lambda Q^{-1}
]

where:

* (Q) contains the eigenvectors as columns;
* (\Lambda) is a diagonal matrix containing the eigenvalues;
* (Q^{-1}) transforms from the original basis into the eigenvector basis.

For a real symmetric matrix:

[
A=A^\top
]

the eigendecomposition becomes:

[
A = Q\Lambda Q^\top
]

because the eigenvectors can be chosen to be orthonormal:

[
Q^\top Q=I
]

Covariance matrices are symmetric and positive semidefinite. Therefore:

* their eigenvalues are real;
* their eigenvalues are nonnegative;
* their eigenvectors can be chosen as an orthonormal basis.

These properties make eigendecomposition especially useful for PCA.

---

### 3.3 Covariance matrix

Suppose the centered data matrix is:

[
X_c \in \mathbb{R}^{n \times d}
]

where:

* (n) is the number of observations;
* (d) is the number of features;
* each feature has mean zero.

The sample covariance matrix is:

[
C = \frac{1}{n-1}X_c^\top X_c
]

The entry (C_{ij}) represents the sample covariance between features (i) and (j).

The diagonal entries are feature variances:

[
C_{ii} = \operatorname{Var}(X_i)
]

PCA performs eigendecomposition of (C):

[
C = V\Lambda V^\top
]

where:

* columns of (V) are the principal directions;
* diagonal entries of (\Lambda) are the variances along those directions.

---

### 3.4 Principal Component Analysis

PCA constructs a sequence of orthogonal directions:

[
v_1,v_2,\ldots,v_d
]

The first component solves:

[
v_1 =
\arg\max_{\lVert v\rVert_2=1}
\operatorname{Var}(X_c v)
]

Because:

[
\operatorname{Var}(X_c v)
=========================

v^\top C v
]

the optimization becomes:

[
v_1 =
\arg\max_{\lVert v\rVert_2=1}
v^\top C v
]

The solution is the eigenvector of (C) associated with its largest eigenvalue.

The second principal component maximizes the same objective while remaining orthogonal to the first:

[
v_2 =
\arg\max_{\lVert v\rVert_2=1,;v^\top v_1=0}
v^\top C v
]

The remaining components follow the same pattern.

---

### 3.5 PCA transformation

Let:

[
W_k =
\begin{bmatrix}
v_1 & v_2 & \cdots & v_k
\end{bmatrix}
]

where (W_k \in \mathbb{R}^{d \times k}).

The lower-dimensional representation is:

[
Z = X_c W_k
]

where:

[
Z \in \mathbb{R}^{n \times k}
]

Each row of (Z) represents an observation in principal-component space.

The reconstruction is:

[
\hat{X}_c = ZW_k^\top
]

or, including the original feature means:

[
\hat{X} = ZW_k^\top + \mu
]

where (\mu) is the vector of feature means.

---

### 3.6 Explained variance

The explained variance ratio of component (i) is:

[
r_i =
\frac{\lambda_i}
{\sum_{j=1}^{d}\lambda_j}
]

where:

* (\lambda_i) is the variance captured by component (i);
* the denominator is the total variance in the dataset.

The cumulative explained variance of the first (k) components is:

[
R_k =
\sum_{i=1}^{k}r_i
]

This is commonly used to select the number of components, but it should not be treated as the only selection criterion.

A component count preserving 95% of variance does not necessarily preserve 95% of predictive or retrieval performance.

---

### 3.7 Singular Value Decomposition

For any matrix:

[
X \in \mathbb{R}^{n \times d}
]

the SVD is:

[
X = U\Sigma V^\top
]

where:

* (U \in \mathbb{R}^{n \times r}) contains left singular vectors;
* (\Sigma \in \mathbb{R}^{r \times r}) contains singular values;
* (V \in \mathbb{R}^{d \times r}) contains right singular vectors;
* (r=\operatorname{rank}(X)).

Using the full SVD, the matrices may contain additional columns associated with zero singular values.

The singular values satisfy:

[
\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0
]

The right singular vectors are eigenvectors of:

[
X^\top X
]

because:

[
X^\top X
========

# V\Sigma^\top U^\top U\Sigma V^\top

V\Sigma^2V^\top
]

Similarly, the left singular vectors are eigenvectors of:

[
XX^\top
]

because:

[
XX^\top
=======

U\Sigma^2U^\top
]

---

### 3.8 Relationship between PCA and SVD

For centered data:

[
X_c=U\Sigma V^\top
]

The covariance matrix is:

[
C
=

\frac{1}{n-1}X_c^\top X_c
]

Substituting the SVD:

[
C
=

\frac{1}{n-1}
V\Sigma^2V^\top
]

Therefore:

[
\Lambda = \frac{\Sigma^2}{n-1}
]

and:

[
\lambda_i = \frac{\sigma_i^2}{n-1}
]

The PCA directions are the right singular vectors:

[
W_k=V_k
]

The PCA-transformed observations are:

[
Z=X_cV_k
]

Because:

[
X_cV_k=U_k\Sigma_k
]

the PCA scores can also be obtained as:

[
Z=U_k\Sigma_k
]

---

### 3.9 Truncated SVD and low-rank approximation

Keeping only the first (k) singular values and vectors produces:

[
X_k=U_k\Sigma_kV_k^\top
]

The Eckart–Young–Mirsky theorem states that (X_k) is the best rank-(k) approximation to (X) under the Frobenius norm and spectral norm.

For the Frobenius norm:

[
\lVert X-X_k\rVert_F^2
======================

\sum_{i=k+1}^{r}\sigma_i^2
]

This gives a strong theoretical justification for dimensionality reduction and compression through SVD.

---

### 3.10 Whitening

Standard PCA decorrelates the transformed dimensions:

[
\operatorname{Cov}(Z)=\Lambda
]

Whitening additionally rescales each component to unit variance:

[
Z_{\text{white}}
================

X_cV_k\Lambda_k^{-1/2}
]

After whitening:

[
\operatorname{Cov}(Z_{\text{white}})\approx I
]

Whitening can help algorithms that assume similarly scaled, uncorrelated inputs, but it also amplifies low-variance directions and may magnify noise.

---

### 3.11 Important assumptions and interpretations

PCA implicitly assumes that:

* linear combinations are sufficient to represent the important structure;
* variance is a meaningful proxy for information;
* Euclidean distance and orthogonality are meaningful for the problem;
* the dataset used to fit PCA represents future data reasonably well;
* large variance is not dominated by irrelevant scale or outliers.

PCA does not require normally distributed data. However, normality may matter for some inferential interpretations around principal components.

---

## 4. Mathematical, statistical or logical foundations

### 4.1 Deriving the first principal component

We want to maximize:

[
v^\top C v
]

subject to:

[
v^\top v=1
]

Using a Lagrange multiplier (\lambda):

[
\mathcal{L}(v,\lambda)
======================

v^\top Cv-\lambda(v^\top v-1)
]

Taking the derivative with respect to (v):

[
\frac{\partial\mathcal{L}}{\partial v}
======================================

2Cv-2\lambda v
]

Setting it to zero:

[
Cv=\lambda v
]

Therefore, any stationary solution is an eigenvector of (C).

The objective value is:

[
v^\top Cv
=========

# v^\top \lambda v

# \lambda v^\top v

\lambda
]

because (v^\top v=1).

So maximizing variance is equivalent to selecting the largest eigenvalue and its eigenvector.

---

### 4.2 PCA as reconstruction-error minimization

PCA can also be defined as the orthogonal projection that minimizes reconstruction error.

For a rank-(k) projection:

[
\hat{X}_c=X_cW_kW_k^\top
]

PCA solves:

[
W_k^*
=====

\arg\min_{W_k^\top W_k=I}
\left\lVert
X_c-X_cW_kW_k^\top
\right\rVert_F^2
]

Thus PCA has two equivalent interpretations:

* maximize retained variance;
* minimize squared reconstruction error.

This equivalence depends on using an orthogonal linear projection and squared Euclidean reconstruction error.

---

### 4.3 Total variance

The total variance of the dataset is the trace of the covariance matrix:

[
\operatorname{tr}(C)
====================

# \sum_{j=1}^{d}C_{jj}

\sum_{i=1}^{d}\lambda_i
]

The trace is invariant under orthogonal changes of basis, so PCA redistributes variance among new axes without changing the total variance.

---

### 4.4 Reconstruction error and discarded variance

For centered data, the squared reconstruction error after retaining (k) components is:

[
\left\lVert X_c-X_k\right\rVert_F^2
===================================

\sum_{i=k+1}^{r}\sigma_i^2
]

Using the PCA eigenvalues:

[
\left\lVert X_c-X_k\right\rVert_F^2
===================================

(n-1)
\sum_{i=k+1}^{r}\lambda_i
]

Therefore, discarded eigenvalues quantify the lost variance and reconstruction information.

---

### 4.5 Why scaling changes PCA

Suppose one feature is measured in dollars and another in percentages.

If the dollar feature has a much larger numerical variance, it can dominate the covariance matrix even when it is not more important.

Using standardized features:

[
x'_{ij}
=======

\frac{x_{ij}-\mu_j}{s_j}
]

where:

* (\mu_j) is the mean of feature (j);
* (s_j) is its standard deviation.

PCA on standardized data is equivalent to PCA based on the correlation matrix rather than the raw covariance matrix.

Scaling is not automatically correct. It is a modeling decision.

For example, if all features use the same physically meaningful unit, their absolute variance may be relevant and standardization may remove useful information.

---

### 4.6 Sign indeterminacy

If (v) is an eigenvector, then (-v) is also an eigenvector:

[
A(-v)=-Av=-\lambda v=\lambda(-v)
]

The same applies to singular vectors.

Therefore, two implementations may produce principal directions with opposite signs while representing the same solution.

A sign difference is not an error.

---

## 5. Practical applicability

### Good use cases

#### Visualization

Projecting data from many dimensions into two or three components can reveal:

* clusters;
* outliers;
* domain shifts;
* batch effects;
* class overlap;
* duplicated or anomalous observations.

However, the projection is optimized for variance, not necessarily class separation.

#### Compression

PCA and SVD can reduce:

* feature matrices;
* images;
* embeddings;
* document-term matrices;
* intermediate model representations.

Compression may reduce:

* storage;
* network transfer;
* vector-index memory;
* training time;
* inference latency.

#### Denoising

Low-variance components may represent noise.

Reconstructing data from the dominant components can remove some noise:

[
\hat{X}=U_k\Sigma_kV_k^\top
]

This only works when the noise is concentrated in low-energy directions.

#### Multicollinearity reduction

Principal components are orthogonal, which can stabilize some linear models when features are strongly correlated.

The trade-off is reduced interpretability.

#### Preprocessing for downstream models

PCA may help:

* linear regression;
* logistic regression;
* clustering;
* nearest-neighbor search;
* anomaly detection;
* classical computer vision pipelines.

It is often less useful for tree-based models because trees are generally less affected by correlated features and do not require orthogonal inputs.

#### Embedding dimensionality reduction

For RAG and vector-search systems, PCA may reduce embedding dimensions before indexing.

Potential advantages:

* smaller vector indexes;
* faster similarity calculations;
* lower storage and memory usage;
* faster network serialization.

Potential risks:

* reduced recall;
* degraded semantic separation;
* loss of rare but relevant directions;
* mismatch between PCA training corpus and production queries.

The decision should be based on retrieval evaluation, such as:

* Recall@k;
* MRR;
* nDCG;
* end-to-end answer quality;
* index latency;
* memory consumption.

Explained variance alone is insufficient.

#### Latent semantic analysis

Applying truncated SVD to a term-document or TF-IDF matrix produces latent semantic dimensions.

Unlike conventional PCA implementations, `TruncatedSVD` does not center the input, which is important for sparse matrices because centering would make them dense.

#### Recommendation systems

A user-item interaction matrix can be approximated as:

[
R\approx U_k\Sigma_kV_k^\top
]

The factors can represent latent user and item characteristics.

Basic SVD does not naturally handle missing-not-at-random observations, temporal effects, biases, or implicit feedback. Production systems normally use specialized matrix-factorization objectives.

---

### When PCA may not make sense

PCA may be inappropriate when:

* the structure is strongly nonlinear;
* low-variance directions contain the prediction target;
* interpretability of original features is essential;
* the data has severe outliers;
* observations combine incompatible units without thoughtful scaling;
* categorical encodings produce misleading Euclidean geometry;
* the transformation must preserve local neighborhoods rather than global variance;
* features are already compact and computational cost is not a concern;
* the downstream model handles redundant dimensions naturally.

---

### Production trade-offs

| Decision                | Benefit                                 | Risk                                     |
| ----------------------- | --------------------------------------- | ---------------------------------------- |
| Retain fewer components | Lower latency and storage               | Information loss                         |
| Standardize features    | Prevent large-scale features dominating | Removes meaningful absolute scale        |
| Whiten components       | Equalized variance                      | Noise amplification                      |
| Fit PCA globally        | Consistent representation               | Leakage and population bias              |
| Incremental PCA         | Supports large datasets                 | Approximation and operational complexity |
| Randomized SVD          | Faster for large low-rank matrices      | Approximate results                      |
| PCA on embeddings       | Smaller vector indexes                  | Possible retrieval degradation           |
| Refit PCA over time     | Adapts to drift                         | Changes feature semantics                |

A critical production consideration is component stability. If PCA is refitted, the principal axes may change, making old and new transformed vectors incompatible.

For a vector-search index, refitting PCA usually requires re-encoding and reindexing the entire corpus.

---

## 6. Common pitfalls and mistakes

### 6.1 Fitting PCA before the train-test split

Incorrect:

```python
X_pca = PCA(n_components=10).fit_transform(X)
X_train, X_test = train_test_split(X_pca)
```

This allows test-set statistics to influence the PCA directions.

Correct:

```python
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=10)),
        ("model", LogisticRegression()),
    ]
)
```

Fit the complete pipeline only on the training data.

---

### 6.2 Forgetting centering

Classical PCA assumes centered data.

Without centering, the first component may point toward the mean rather than the dominant variation around the mean.

`sklearn.decomposition.PCA` centers automatically but does not automatically standardize.

---

### 6.3 Standardizing without domain reasoning

Standardization is often useful, but not universally correct.

It changes the question from:

> Which directions have the greatest original variance?

to:

> Which directions explain the greatest standardized variance?

That is a modeling decision, not a mandatory preprocessing ritual.

---

### 6.4 Interpreting PCA components causally

A component is a mathematical direction, not necessarily a real latent cause.

High loading does not imply:

* causal influence;
* business importance;
* predictive importance;
* controllability.

---

### 6.5 Assuming high explained variance preserves target information

PCA is unsupervised.

It does not use labels when choosing components.

A low-variance direction may contain most of the information necessary to predict the target.

Always evaluate the downstream task.

---

### 6.6 Using ordinary PCA on sparse matrices

Centering a sparse term-document matrix makes it dense.

For text data, use approaches such as:

* `TruncatedSVD`;
* randomized SVD;
* specialized sparse matrix factorization.

Remember that truncated SVD on uncentered data is not mathematically identical to PCA.

---

### 6.7 Forming a huge covariance matrix unnecessarily

If (d) is very large, constructing:

[
X^\top X \in \mathbb{R}^{d\times d}
]

may be computationally expensive and can worsen numerical conditioning.

Direct SVD, truncated SVD, or randomized methods are usually preferable.

---

### 6.8 Using `np.linalg.eig` for a symmetric covariance matrix

A covariance matrix is symmetric.

Use:

```python
np.linalg.eigh(covariance_matrix)
```

rather than:

```python
np.linalg.eig(covariance_matrix)
```

`eigh` exploits symmetry and returns real eigenvalues with better numerical behavior.

---

### 6.9 Comparing eigenvector signs directly

Two PCA implementations may return opposite vector signs.

Compare subspaces, absolute correlations, or reconstructions rather than exact signed vectors.

---

### 6.10 Ignoring outliers

PCA is based on squared distances and variance, making it sensitive to outliers.

A few extreme points can dominate the principal directions.

Possible alternatives include:

* robust scaling;
* outlier treatment;
* Robust PCA formulations;
* covariance estimators with improved robustness.

---

### 6.11 Choosing components only through a fixed threshold

“Keep 95% variance” is a heuristic.

Component selection should also consider:

* downstream validation performance;
* latency;
* memory;
* reconstruction quality;
* interpretability;
* retrieval quality;
* stability across datasets.

---

### 6.12 Treating PCA as feature selection

PCA creates new features.

It does not choose a subset of original columns.

That distinction matters when interpretability, governance, lineage, or feature-level actionability is required.

---

## 7. Important comparisons

### PCA vs eigendecomposition

| PCA                                                   | Eigendecomposition                                  |
| ----------------------------------------------------- | --------------------------------------------------- |
| Statistical dimensionality-reduction method           | General matrix decomposition                        |
| Usually applied to a covariance or correlation matrix | Applies to square matrices                          |
| Components represent maximum-variance directions      | Eigenvectors represent invariant directions         |
| Uses ordered eigenvalues for variance ranking         | Eigenvalues may represent many different properties |

PCA uses eigendecomposition, but eigendecomposition is much broader than PCA.

---

### PCA vs SVD

| PCA                                                   | SVD                                                 |
| ----------------------------------------------------- | --------------------------------------------------- |
| Method for identifying variance-maximizing directions | General matrix factorization                        |
| Usually assumes centered data                         | Does not inherently center data                     |
| Can be computed from covariance eigendecomposition    | Can compute PCA directly from centered data         |
| Components correspond to covariance eigenvectors      | Right singular vectors correspond to PCA directions |
| Usually discussed statistically                       | Often discussed algebraically and computationally   |

In practice, many PCA implementations internally use SVD.

---

### PCA vs Truncated SVD

| PCA                                     | Truncated SVD                                    |
| --------------------------------------- | ------------------------------------------------ |
| Centers the input                       | Usually does not center                          |
| Works naturally with dense numeric data | Suitable for large sparse matrices               |
| Components represent centered variance  | Components capture dominant uncentered structure |
| Common for tabular numeric features     | Common for TF-IDF and term-document matrices     |

---

### PCA vs feature selection

| PCA                                             | Feature selection                                |
| ----------------------------------------------- | ------------------------------------------------ |
| Creates linear combinations                     | Retains original variables                       |
| Can compress correlated information efficiently | Preserves interpretability                       |
| Components may be difficult to explain          | Easier governance and business interpretation    |
| Every component may depend on every feature     | Selected subset may miss distributed information |

---

### PCA vs t-SNE

| PCA                                                  | t-SNE                                            |
| ---------------------------------------------------- | ------------------------------------------------ |
| Linear                                               | Nonlinear                                        |
| Preserves global variance structure                  | Emphasizes local neighborhood similarity         |
| Supports deterministic transformation of new samples | Standard t-SNE has limited out-of-sample support |
| Useful for preprocessing and compression             | Mainly useful for visualization                  |
| Axes have linear loading interpretations             | Axes do not have direct semantic meaning         |

t-SNE should generally not be used as a production preprocessing transformation without careful justification.

---

### PCA vs UMAP

| PCA                                           | UMAP                                         |
| --------------------------------------------- | -------------------------------------------- |
| Linear and globally oriented                  | Nonlinear and neighborhood-oriented          |
| Simple and comparatively stable               | More sensitive to hyperparameters            |
| Easy inverse approximation through projection | Inverse transformation is more complicated   |
| Strong mathematical low-rank interpretation   | Better at many nonlinear visualization tasks |
| Fast and easy to deploy                       | Often better for exploratory visualization   |

---

### PCA vs autoencoder

| PCA                                       | Autoencoder                                  |
| ----------------------------------------- | -------------------------------------------- |
| Linear                                    | Potentially nonlinear                        |
| Closed-form or deterministic optimization | Learned through iterative optimization       |
| Easy to inspect and reproduce             | More flexible but more complex               |
| Limited representation capacity           | Can learn nonlinear manifolds                |
| Lower operational cost                    | Requires architecture and training decisions |

A linear autoencoder with squared reconstruction loss and suitable constraints learns the same principal subspace as PCA, although its basis may differ by a rotation.

---

### PCA vs Kernel PCA

Kernel PCA performs PCA implicitly in a nonlinear feature space through a kernel matrix.

Advantages:

* captures nonlinear relationships;
* can separate structures that linear PCA cannot.

Disadvantages:

* difficult to scale;
* harder to transform new observations efficiently;
* harder to interpret;
* sensitive to kernel and hyperparameter choices.

---

## 8. Practical Python example

This example creates a correlated three-dimensional dataset, applies PCA, verifies the relationship with SVD, and visualizes the two-dimensional representation.

```python
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def generate_data(
    n_samples: int = 500,
    random_state: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)

    latent_1 = rng.normal(size=n_samples)
    latent_2 = rng.normal(scale=0.5, size=n_samples)
    noise = rng.normal(scale=0.15, size=(n_samples, 3))

    x1 = 2.0 * latent_1 + 0.2 * latent_2
    x2 = 1.7 * latent_1 + 0.5 * latent_2
    x3 = -0.6 * latent_1 + 1.2 * latent_2

    X = np.column_stack([x1, x2, x3])
    return X + noise


def main() -> None:
    X = generate_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print("Principal directions:")
    print(pca.components_)

    print("\nExplained variance:")
    print(pca.explained_variance_)

    print("\nExplained variance ratio:")
    print(pca.explained_variance_ratio_)

    print(
        "\nCumulative explained variance:",
        pca.explained_variance_ratio_.sum(),
    )

    # Verify PCA-SVD relationship.
    U, singular_values, Vt = np.linalg.svd(
        X_scaled,
        full_matrices=False,
    )

    variance_from_svd = singular_values**2 / (X_scaled.shape[0] - 1)

    print("\nVariance recovered from singular values:")
    print(variance_from_svd[:2])

    print("\nPCA singular values:")
    print(pca.singular_values_)

    # Reconstruct the standardized data using two components.
    X_reconstructed = pca.inverse_transform(X_pca)

    reconstruction_mse = np.mean(
        (X_scaled - X_reconstructed) ** 2
    )

    print("\nReconstruction MSE:")
    print(reconstruction_mse)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        alpha=0.65,
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Synthetic data projected with PCA")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

Install the dependencies:

```bash
pip install numpy scikit-learn matplotlib
```

Run:

```bash
python example.py
```

### What to observe

The first two singular-value-derived variances should match:

```python
pca.explained_variance_
```

The corresponding principal directions should match the rows of:

```python
Vt[:2]
```

possibly with opposite signs.

The first component should capture the dominant shared variation between the original features.

---

## 9. From-scratch implementation when useful

The following implementation computes PCA through eigendecomposition of the covariance matrix.

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PCAFromScratch:
    def __init__(self, n_components: int) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be greater than zero.")

        self.n_components = n_components
        self.mean_: NDArray[np.float64] | None = None
        self.components_: NDArray[np.float64] | None = None
        self.explained_variance_: NDArray[np.float64] | None = None
        self.explained_variance_ratio_: NDArray[np.float64] | None = None

    def fit(self, X: NDArray[np.float64]) -> "PCAFromScratch":
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix.")

        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError("At least two observations are required.")

        if self.n_components > n_features:
            raise ValueError(
                "n_components cannot exceed the number of features."
            )

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        covariance_matrix = (
            X_centered.T @ X_centered
        ) / (n_samples - 1)

        # eigh is designed for real symmetric matrices.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # np.linalg.eigh returns eigenvalues in ascending order.
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.components_ = eigenvectors[:, : self.n_components].T
        self.explained_variance_ = eigenvalues[: self.n_components]

        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_variance
        )

        return self

    def transform(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self._check_is_fitted()

        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_

        return X_centered @ self.components_.T

    def inverse_transform(
        self,
        Z: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self._check_is_fitted()

        Z = np.asarray(Z, dtype=np.float64)
        return Z @ self.components_ + self.mean_

    def fit_transform(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self.fit(X).transform(X)

    def _check_is_fitted(self) -> None:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("The PCA instance has not been fitted.")


def main() -> None:
    rng = np.random.default_rng(42)

    base = rng.normal(size=300)

    X = np.column_stack(
        [
            base + rng.normal(scale=0.1, size=300),
            2.0 * base + rng.normal(scale=0.2, size=300),
            -0.5 * base + rng.normal(scale=0.1, size=300),
        ]
    )

    pca = PCAFromScratch(n_components=2)
    Z = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(Z)

    print("Components:")
    print(pca.components_)

    print("\nExplained variance:")
    print(pca.explained_variance_)

    print("\nExplained variance ratio:")
    print(pca.explained_variance_ratio_)

    print("\nTransformed shape:")
    print(Z.shape)

    print("\nReconstruction MSE:")
    print(np.mean((X - X_reconstructed) ** 2))


if __name__ == "__main__":
    main()
```

### Alternative direct-SVD implementation

The core of PCA can also be written as:

```python
X_centered = X - X.mean(axis=0)

U, singular_values, Vt = np.linalg.svd(
    X_centered,
    full_matrices=False,
)

components = Vt[:n_components]
scores = X_centered @ components.T

explained_variance = (
    singular_values[:n_components] ** 2
) / (X.shape[0] - 1)
```

Direct SVD avoids explicitly constructing the covariance matrix and is generally the preferable implementation strategy.

---

## 10. Suggested experiments

### Experiment 1 — Compare centered and uncentered decomposition

Add a large constant offset to all features:

```python
X_offset = X + np.array([100.0, -50.0, 200.0])
```

Compare:

* SVD directly on `X_offset`;
* SVD on centered `X_offset`;
* `sklearn.PCA`.

Observe how failing to center changes the dominant direction.

---

### Experiment 2 — Compare scaled and unscaled PCA

Multiply one feature by 100:

```python
X_rescaled = X.copy()
X_rescaled[:, 0] *= 100
```

Run PCA:

* directly on `X_rescaled`;
* after `StandardScaler`.

Inspect how the component loadings and explained variance change.

---

### Experiment 3 — Vary the number of components

Test:

```python
n_components = 1
n_components = 2
n_components = 3
```

For each value, calculate:

* cumulative explained variance;
* reconstruction MSE;
* storage reduction.

Plot reconstruction error as a function of component count.

---

### Experiment 4 — Add outliers

Inject a few extreme observations:

```python
X_with_outliers = np.vstack(
    [
        X,
        np.array(
            [
                [15.0, -20.0, 25.0],
                [-18.0, 24.0, -30.0],
            ]
        ),
    ]
)
```

Compare the principal directions before and after adding outliers.

This demonstrates PCA’s sensitivity to squared distances.

---

### Experiment 5 — Evaluate PCA for retrieval

Generate or use a small set of embeddings, then:

1. fit PCA on the corpus embeddings;
2. reduce the dimensions;
3. normalize the transformed embeddings;
4. compare nearest neighbors before and after PCA;
5. calculate Recall@k.

This connects the theory directly to RAG and vector-search systems.

---

## 11. Senior interview questions

### 1. What does an eigenvector represent?

An eigenvector represents a direction preserved by a linear transformation. Applying the matrix changes its magnitude, and possibly its orientation through a negative sign, but does not rotate it into a different line. The corresponding eigenvalue quantifies the scaling along that direction.

---

### 2. Why are covariance-matrix eigenvalues always nonnegative?

The covariance matrix is positive semidefinite.

For any vector (v):

[
v^\top Cv
=========

\frac{1}{n-1}
v^\top X_c^\top X_cv
====================

\frac{1}{n-1}
\lVert X_cv\rVert_2^2
\geq 0
]

If (v) is an eigenvector:

[
v^\top Cv=\lambda v^\top v
]

Since (v^\top v>0), it follows that:

[
\lambda\geq 0
]

Small negative values sometimes returned by numerical software are generally floating-point errors.

---

### 3. Why does PCA select eigenvectors of the covariance matrix?

PCA maximizes the variance of the projected data:

[
\operatorname{Var}(X_cv)=v^\top Cv
]

subject to unit length:

[
v^\top v=1
]

Using Lagrange multipliers produces:

[
Cv=\lambda v
]

The maximum is achieved by the eigenvector with the largest eigenvalue.

---

### 4. What is the relationship between PCA and SVD?

For centered data:

[
X_c=U\Sigma V^\top
]

the principal directions are the columns of (V), the PCA scores are (U\Sigma), and the covariance eigenvalues are:

[
\lambda_i=\frac{\sigma_i^2}{n-1}
]

Therefore, PCA can be calculated directly from the SVD of the centered data matrix.

---

### 5. Why is SVD usually preferable to covariance eigendecomposition?

Constructing the covariance matrix can:

* square the condition number;
* require a large (d\times d) matrix;
* introduce additional numerical error;
* consume unnecessary memory.

Direct, truncated, or randomized SVD is usually more numerically stable and scalable.

---

### 6. Does PCA always require standardization?

No.

PCA always requires thoughtful handling of location, usually through centering. Standardization is optional and depends on whether the original feature scales are meaningful.

Standardize when differences in units or arbitrary scales would dominate the covariance structure. Avoid automatic standardization when absolute scale carries domain meaning.

---

### 7. Can PCA reduce overfitting?

Potentially.

Reducing dimensions may remove noise and reduce model complexity. However, because PCA is unsupervised, it may also remove low-variance directions that are highly predictive.

Its effect should be measured through validation rather than assumed.

---

### 8. Why might PCA hurt a classification model?

PCA maximizes overall variance, not class separability.

A low-variance direction may distinguish classes, while a high-variance direction may reflect irrelevant variation.

Supervised alternatives include:

* feature selection based on validation;
* Linear Discriminant Analysis;
* Partial Least Squares;
* learned supervised embeddings.

---

### 9. What is the difference between PCA and TruncatedSVD in scikit-learn?

`PCA` centers the data before decomposition.

`TruncatedSVD` generally operates on the uncentered input and supports sparse matrices efficiently.

For sparse TF-IDF matrices, centering would destroy sparsity, so `TruncatedSVD` is commonly used.

---

### 10. How would you choose the number of PCA components?

I would combine:

* cumulative explained variance;
* scree-plot inspection;
* reconstruction error;
* downstream validation metrics;
* latency and storage constraints;
* component stability;
* interpretability requirements.

For a retrieval system, I would evaluate Recall@k and end-to-end answer quality rather than selecting components from explained variance alone.

---

### 11. How would you deploy PCA safely in a production ML pipeline?

I would:

1. split the data before fitting preprocessing;
2. fit scaling and PCA only on training data;
3. store them as a single versioned pipeline;
4. use the same transformation for training and inference;
5. validate input schema and feature ordering;
6. monitor input-distribution drift;
7. track downstream performance;
8. avoid silently refitting components;
9. version and migrate transformed datasets when PCA changes.

---

### 12. How would PCA affect a vector-search system?

PCA can reduce vector dimensions and improve index memory, network cost, and search latency.

However, it may degrade semantic recall. I would fit PCA on a representative corpus, apply the same transformation to both documents and queries, preserve normalization requirements, and evaluate the dimensionality-latency-recall trade-off.

Refitting PCA would require regenerating corpus vectors and rebuilding the index.

---

### 13. Why can PCA components change when retrained?

The covariance structure of the fitting data may change because of:

* population drift;
* different sampling;
* outliers;
* changes in preprocessing;
* nearly equal eigenvalues.

When eigenvalues are close, the corresponding individual eigenvectors may be unstable even though their combined subspace remains relatively stable.

---

### 14. What happens when two eigenvalues are equal?

The individual eigenvectors within that eigenspace are not unique.

Any orthonormal basis spanning the same eigenspace is valid.

This is important when comparing PCA implementations: the subspace can be equivalent even when individual component vectors differ.

---

### 15. What is randomized SVD?

Randomized SVD uses random projections to approximate the dominant subspace before performing a smaller decomposition.

It is useful when:

* the matrix is very large;
* only a small number of components is required;
* the spectrum decays sufficiently;
* a small approximation error is acceptable.

It reduces computational cost compared with a complete exact SVD.

---

### 16. How is PCA related to a linear autoencoder?

A linear autoencoder trained with squared reconstruction loss learns the same optimal principal subspace as PCA under appropriate conditions.

However, the encoder weights do not necessarily equal the PCA eigenvectors because the learned latent basis may be rotated within the same subspace.

---

## 12. Interview-ready explanation

PCA is a linear dimensionality-reduction method that identifies orthogonal directions of maximum variance in centered data. Mathematically, these directions are the eigenvectors of the covariance matrix, ordered by their eigenvalues, which represent the variance captured by each component.

In practice, I usually think of PCA through SVD. If the centered data matrix is decomposed as (X=U\Sigma V^\top), the right singular vectors in (V) are the principal directions, the transformed observations are (U\Sigma), and the explained variances are the squared singular values divided by (n-1).

I would use PCA for visualization, compression, denoising, multicollinearity reduction, or reducing the dimensionality of features and embeddings. I would not assume that preserving variance automatically preserves predictive or retrieval quality, because PCA is unsupervised. In a real project, I would fit it only on training data, evaluate scaling carefully, version the transformation, and select the number of components using downstream metrics alongside explained variance, latency, memory, and reconstruction error.

---

## 13. GitHub file structure

```text
day-04-eigenvalues-pca-svd/
├── README.md
├── notes.md
├── notebook.ipynb
├── example.py
├── from_scratch.py
├── interview_questions.md
├── references.md
├── requirements.txt
└── outputs/
    ├── pca_projection.png
    ├── explained_variance.png
    └── reconstruction_error.png
```

### File responsibilities

**`README.md`**

High-level professional overview, execution instructions, practical results, and key takeaways.

**`notes.md`**

Detailed theoretical notes, derivations, production trade-offs, and comparisons.

**`notebook.ipynb`**

Interactive exploration containing visualizations and experiments.

**`example.py`**

Small executable example using scikit-learn.

**`from_scratch.py`**

Educational NumPy implementation of PCA.

**`interview_questions.md`**

Conceptual, mathematical, practical, and system-design questions.

**`references.md`**

Books, papers, documentation, lectures, and useful external material.

**`requirements.txt`**

```text
numpy
matplotlib
scikit-learn
jupyter
```

**`outputs/`**

Generated charts that can be referenced from the README.

---

## 14. Suggested README.md content

Here is the finished README content:

# Eigenvalues, Eigenvectors, PCA and SVD

This project explores the linear algebra foundations behind Principal Component Analysis and Singular Value Decomposition, with emphasis on dimensionality reduction, low-rank approximation, and practical AI engineering applications.

## Objective

The objective is to understand how eigenvalues, eigenvectors, PCA, and SVD are connected and how these concepts can be applied to real machine learning and AI systems.

The project covers both mathematical intuition and executable implementations.

## Concepts covered

* Eigenvalues and eigenvectors
* Eigendecomposition of symmetric matrices
* Covariance matrices
* Principal Component Analysis
* Explained variance
* Singular Value Decomposition
* Low-rank matrix approximation
* Reconstruction error
* PCA through covariance eigendecomposition
* PCA through direct SVD
* Scaling, centering, and whitening
* Production and evaluation trade-offs

## Project structure

```text
.
├── README.md
├── notes.md
├── notebook.ipynb
├── example.py
├── from_scratch.py
├── interview_questions.md
├── references.md
├── requirements.txt
└── outputs/
```

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the examples

Run the scikit-learn example:

```bash
python example.py
```

Run the educational NumPy implementation:

```bash
python from_scratch.py
```

Start the notebook:

```bash
jupyter notebook notebook.ipynb
```

## Key mathematical relationship

For a centered data matrix:

```text
X = U Σ Vᵀ
```

The columns of `V` are the principal directions, the transformed observations are given by `UΣ`, and the explained variance of each component is obtained from the squared singular values divided by `n - 1`.

## Key takeaways

* PCA identifies orthogonal directions that maximize retained variance.
* PCA can be interpreted as both variance maximization and reconstruction-error minimization.
* SVD provides a numerically stable way to compute PCA without explicitly constructing the covariance matrix.
* Explained variance does not necessarily represent predictive or retrieval importance.
* Scaling, outliers, data leakage, and population drift can significantly affect the resulting components.
* The number of components should be selected using downstream metrics and operational constraints, not only a fixed explained-variance threshold.

## Practical applications

These techniques can support:

* dimensionality reduction;
* data visualization;
* denoising;
* image and matrix compression;
* latent semantic analysis;
* recommendation systems;
* embedding compression;
* vector-search optimization;
* multicollinearity reduction.

## Limitations

PCA is a linear and unsupervised method. It may fail to preserve nonlinear structure or low-variance information that is relevant to the downstream task.

Any production use should be validated against task-specific quality, latency, memory, and stability requirements.

And the LinkedIn-ready version:

Nem toda dimensão de um dataset carrega a mesma quantidade de informação.

Ao estudar PCA e SVD com mais profundidade, um dos pontos que mais chama atenção é que redução de dimensionalidade não significa simplesmente excluir colunas. O objetivo é encontrar novas direções que concentrem a estrutura mais relevante dos dados.

Na prática, isso pode ajudar em visualização, compressão, redução de ruído e até na diminuição do tamanho de embeddings utilizados em sistemas de busca vetorial.

Mas existe um cuidado importante: preservar variância não significa necessariamente preservar aquilo que é relevante para uma classificação, previsão ou recuperação de documentos.

Por isso, em um sistema real, a escolha da dimensionalidade precisa considerar não apenas a variância explicada, mas também métricas do problema, custo computacional, latência e estabilidade da transformação.

Documentei a parte teórica, os principais trade-offs e implementações com NumPy e scikit-learn no meu repositório de Applied AI Engineering.

O post pode ser publicado junto a uma imagem do gráfico de variância explicada ou da projeção PCA, sem transformar o conteúdo em um diário diário de estudos.

---

## 15. LinkedIn post idea

The post above focuses on the strongest professional insight:

> Dimensionality reduction is an engineering trade-off, not merely a mathematical compression step.

A good visual for the post would contain:

* the original correlated three-dimensional dataset;
* its two-dimensional PCA projection;
* the cumulative explained variance;
* a small comparison of original versus reduced dimensions.

Avoid presenting PCA as universally beneficial. The production-oriented warning about downstream evaluation differentiates the post from basic educational content.

---

## 16. 30–60 minute checklist

### 30-minute essential path

* [ ] Read the intuition behind invariant directions.
* [ ] Understand (Av=\lambda v).
* [ ] Understand why the covariance matrix is symmetric and positive semidefinite.
* [ ] Review the PCA objective (v^\top Cv).
* [ ] Understand the relationship (X_c=U\Sigma V^\top).
* [ ] Memorize (\lambda_i=\sigma_i^2/(n-1)).
* [ ] Run `example.py`.
* [ ] Explain PCA aloud in two minutes.
* [ ] Commit the code and README.

### 45-minute recommended path

* [ ] Complete the 30-minute path.
* [ ] Run `from_scratch.py`.
* [ ] Compare scaled and unscaled PCA.
* [ ] Test one, two, and three components.
* [ ] Calculate reconstruction error.
* [ ] Inspect component loadings.
* [ ] Answer five interview questions without reading.
* [ ] Save one visualization in `outputs/`.

### 60-minute complete path

* [ ] Complete the 45-minute path.
* [ ] Add outliers and observe component movement.
* [ ] Compare covariance eigendecomposition with direct SVD.
* [ ] Verify that singular-vector signs may differ.
* [ ] Document PCA versus TruncatedSVD.
* [ ] Add one paragraph about embedding compression and RAG.
* [ ] Record the latency-quality trade-off you would evaluate in production.
* [ ] Update `interview_questions.md`.
* [ ] Create a clean Git commit.

Suggested commit sequence:

```bash
git status
git add day-04-eigenvalues-pca-svd
git commit -m "Add Day 4 study on PCA, eigenvalues and SVD"
git push
```

### Final self-test

By the end of the session, you should be able to explain:

1. Why PCA directions are covariance eigenvectors.
2. Why eigenvalues represent variance.
3. How PCA is computed through SVD.
4. Why singular values are squared when converted to explained variance.
5. Why centering is mandatory but standardization is contextual.
6. Why explained variance alone is insufficient.
7. How PCA can affect a production RAG or vector-search system.
8. Why refitting PCA can require rebuilding transformed datasets and indexes.
