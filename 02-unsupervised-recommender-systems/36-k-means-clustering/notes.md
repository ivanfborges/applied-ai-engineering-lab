# K-Means: objective, representation and limits

[Executable record and results](README.md) · [Sources](references.md)

## Why the centroid is the mean

For observations $x_i\in\mathbb{R}^d$, assignments $c_i\in\{1,\ldots,K\}$
and centers $\mu_j$, minimize

$$
J(c,\mu)=\sum_{j=1}^{K}\sum_{i:c_i=j}\lVert x_i-\mu_j\rVert_2^2.
$$

With centers fixed, each assignment independently chooses the nearest center.
With a nonempty cluster $C_j$ fixed, differentiate its contribution:

$$
\nabla_{\mu_j}J=2|C_j|\mu_j-2\sum_{i\in C_j}x_i=0,
\qquad
\mu_j=\frac{1}{|C_j|}\sum_{i\in C_j}x_i.
$$

The Hessian is $2|C_j|I_d$, positive definite when the cluster is nonempty.
Equivalently, for its mean $\bar{x}$,

$$
\sum_i\lVert x_i-\mu\rVert^2
=\sum_i\lVert x_i-\bar{x}\rVert^2+|C_j|\lVert\mu-\bar{x}\rVert^2.
$$

The extra term is minimized at the mean. A centroid need not be an actual
observation. Coordinatewise medians minimize coordinatewise absolute loss;
K-Medoids restricts representatives to observed points. Changing the distance
while keeping the mean update generally loses the minimization argument.

## Alternating optimization and termination

Lloyd's algorithm alternates nearest-center assignment and nonempty-cluster mean
updates. Each exact step cannot increase $J$. With consistent ties and empty
cluster handling, assignments can stabilize; this says nothing about the global
optimum. Equidistant points need a tie policy. A center with no members has no
defined mean.

The [educational implementation](from_scratch.py) samples distinct row indices
once, or accepts explicit centers. Distinct rows can still have identical values.
Ties go to the lowest index. Empty centers retain their previous positions,
so fewer than $K$ occupied clusters are possible. This keeps the update simple
and non-increasing but does not repair a wasted center. It does not implement
scikit-learn's empty-cluster behavior.

Its history records the nearest-center objective at initialization and after
each mean update followed by reassignment. `n_iter` counts mean updates.
It stops for unchanged assignments, absolute Frobenius center movement at most
`tol`, or `max_iter`. `converged` means one of the first two stopping criteria
was reached. It does not certify global optimality or $K$ occupied clusters.
At a movement or iteration stop, returned centers need not be means of the final
labels. Labels and inertia are always recalculated against the returned centers;
the tests deliberately exercise this distinction. Stable assignments after a
mean update do give mean-consistent nonempty centers.

The scratch movement tolerance is absolute and differs from scikit-learn's
documented relative tolerance. The matched comparison uses `tol=0` to avoid
equating different tolerance conventions. Finite inputs, parameter ranges,
initial-center shape and feature counts are validated; overflowing squared
distances raise a rescaling error. This is still a teaching implementation for
moderate magnitudes, without production numerical safeguards.

## Initialization and computation

K-Means++ selects centers with a preference for points far from existing centers
using squared distance. It improves the starting spread but does not guarantee
the global minimum of the subsequent Lloyd fit. scikit-learn uses a greedy
variant. Set `n_init` explicitly: it is the restart budget, not the iteration
count. Each run iterates separately and the lowest-inertia result is retained.
The example uses ten runs per K-Means++ fit instead of relying on `auto`.
See the [KMeans API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
for initialization and stopping details.

A straightforward implementation costs about $O(rnKdT)$ for $r$ restarts,
$n$ rows, $K$ centers, $d$ dimensions and $T$ iterations. The scratch broadcast
allocates an $n\times K\times d$ temporary as well as distances. It is unsuitable
for large embedding corpora. Library implementations optimize memory and
computation; MiniBatchKMeans trades approximate updates for lower fitting cost.
Full Euclidean silhouette also entails pairwise-distance work, roughly
$O(n^2d)$, which can dominate at larger sample sizes. Sampled diagnostics need
their own documented seed and population.

## Choosing K without inventing a true answer

An optimal $K$-center solution remains feasible when another center is allowed,
so

$$
J^*_{K+1}\leq J^*_K.
$$

One center per distinct point can give zero inertia. This explains why raw
minimization of inertia selects complexity. The elbow heuristic looks for
diminishing reductions within one representation, population and distance
objective. A smooth curve may have no defensible elbow. Finite restart budgets
can also yield nonmonotone fitted values even though global optima are monotone.
The [executed table](README.md#scaling-and-cluster-count) permits inspecting
successive reductions without implying an automatic elbow detector.

For a point $i$, let $a_i$ be its average Euclidean distance to other members
of its cluster, and $b_i$ the smallest average distance to any other cluster:

$$
s_i=\frac{b_i-a_i}{\max(a_i,b_i)}.
$$

Silhouette averages these coefficients. It uses pairwise Euclidean distances
here, not K-Means' sum of **squared** point-to-center distances. Values near
one indicate separation, near zero suggest boundaries, and negative values
indicate another cluster has a smaller average distance. scikit-learn requires
between two and $n-1$ occupied clusters and gives singleton members a coefficient
of zero. It favors compact separated groups and does not validate domain meaning.
See the [metric definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

Inspect sizes and profiles, compare partitions across seeds and samples, and
check downstream utility. ARI is invariant to label permutation; it can compare
two runs without known truth. In this example it also compares to synthetic
generating labels used only after fitting. Real unlabeled data generally do
not supply such a reference. Selecting K on internal scores and reporting those
same scores is exploratory evidence, not held-out performance.

## Scaling is a choice of metric

For fitted marginal means $m_p$ and positive standard deviations $s_p$,
standardization maps $x_{ip}$ to $(x_{ip}-m_p)/s_p$. Hence

$$
\lVert z_i-z_r\rVert_2^2
=\sum_{p=1}^{d}\frac{(x_{ip}-x_{rp})^2}{s_p^2}.
$$

Centering alone does not alter pairwise distances. Scaling changes feature
weights by inverse variance. Multiplying every feature by the same positive
constant preserves exact nearest-center geometry and multiplies inertia by its
square; scaling just one feature can change assignments. Inertia values across
different transformations or sample sizes are not comparable as quality scores.

StandardScaler uses population variance (`ddof=0`); a zero-variance feature gets
a scale factor of one. Near-constant noise can be amplified. Correlated features
can count a signal repeatedly even after standardization. Scaling removes neither
correlation nor outlier influence; the scaler itself is sensitive to outliers.
Feature selection, justified weights, robust transforms or PCA may help, but each
changes the representation. PCA preserves variance, not necessarily separation.
See the [scaler contract](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

Inverse-transforming centers helps profile clusters in original units. It does
not make a raw Euclidean prediction equivalent to a scaled prediction: new
observations must still pass through the fitted transformation.

## Geometry, applications and failure modes

Nearest-center regions are convex Voronoi cells. K-Means can run on arbitrary
finite numeric data, but its compactness objective can divide nonconvex groups,
merge nearby unequal-density groups or allocate centers to large dispersed
groups. Gaussianity is not a requirement for execution. The method is especially
suitable when relatively compact groups in Euclidean space reflect the intended
segmentation. Squared distances give distant observations substantial influence
over centroids. A high-dimensional cloud can have weak distance contrast even
when optimization succeeds.

| Need or concern | Alternative to investigate | Trade-off |
|---|---|---|
| Elliptical groups or soft membership | Gaussian mixture | Covariance estimation and distributional assumptions |
| Irregular groups and explicit noise | DBSCAN | Density parameters and varying-density difficulties |
| Nested group structure | Agglomerative clustering | Linkage choice and scaling costs |
| Representatives that are real records | K-Medoids | More expensive optimization; robustness depends on the dissimilarity |
| Large corpus | MiniBatchKMeans | Approximate updates and additional quality checks |

Customer segmentation requires domain review of profiles; cluster zero is not
automatically a customer type. Vector quantization replaces points by centers
with a measurable squared reconstruction loss. Coarse recommendation candidates
still need personalized ranking and relevance evaluation. No application above
is validated by the synthetic experiment. The official
[assumptions demonstration](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html)
provides additional failure cases; this study has not run those cases.

## Embeddings and operational boundaries

For **two unit vectors**, $u$ and $v$,

$$
\lVert u-v\rVert_2^2=2-2\cos(u,v).
$$

This equates pairwise rankings for normalized vectors. Ordinary K-Means learns
arithmetic centers whose norms can differ, so normalizing input embeddings does
not generally turn ordinary K-Means into spherical K-Means. Inspect embedding
quality, similarity choice and dimensionality before clustering. Semantic labels
and retrieval gains require reviewed examples and representative queries.

For exploratory clustering, fitting the whole permitted corpus is legitimate
when the claim concerns that corpus. For future inference, temporal evaluation
or clustering as features in a supervised model, fit the scaler and clusterer
on the training population only, within folds where appropriate. No labels are
needed for information from future observations to leak into a representation.

Version the feature schema, preprocessing and centers together. Cluster IDs
may permute across retraining; compare matched profiles or partitions instead
of assigning stable meaning to numeric IDs. Monitor input quality, cluster sizes,
distance distributions and downstream outcomes. A size change alone need not
indicate failure. A far-away point still gets its nearest center; distances
are not calibrated membership probabilities, and any reject policy needs a
separate validation design. This study deploys no service and estimates no
business impact.
