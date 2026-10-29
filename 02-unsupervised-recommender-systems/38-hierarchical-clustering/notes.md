# Hierarchical clustering: merge criteria, cuts and evidence

## From distances to nested groups

Agglomerative clustering begins with $n$ singleton clusters. At each step it
merges the pair minimizing the linkage criterion, replacing that pair with their
union. A complete binary tree has $n-1$ merges. Divisive methods proceed from
large groups to smaller groups; they are outside this implementation.

Every coarser partition of one merge tree is a union of finer clusters. Early
merges cannot be undone. Independently fitted K-Means partitions need not nest;
Ward is not a sequence of globally optimal K-Means solutions.

For clusters $A,B$ and point dissimilarity $d$, define:

$$
D_{\mathrm{single}}(A,B)=\min_{x\in A,y\in B}d(x,y),
\qquad D_{\mathrm{complete}}(A,B)=\max_{x\in A,y\in B}d(x,y).
$$

$$
D_{\mathrm{average}}(A,B)=\frac{1}{|A||B|}
\sum_{x\in A}\sum_{y\in B}d(x,y).
$$

Single linkage follows connectivity and can join remote groups through a chain.
Complete linkage limits the largest cross-pair distance at each merge and is
sensitive to extremes. Average linkage uses all cross-pairs; it is neither a
centroid distance nor an unweighted average of previous cluster distances.
Its update after joining $A,B$ weights their contributions by cluster sizes.

For Euclidean coordinates, $d(x,y)=\|x-y\|_2$. Feature units change the geometry;
standardization is a modeling choice, not a universally correct remedy. Ordinal,
categorical or mixed data require an explicitly justified representation.
For spatial coordinates, longitude/latitude degrees are not Euclidean meters:
choose an appropriate projected CRS for a local metric analysis, or an appropriate
geodesic dissimilarity with a compatible linkage. Ward needs Euclidean coordinates.

## Ward: merge cost and height are different quantities

Let $SSE(C)=\sum_{x\in C}\|x-\mu_C\|_2^2$. The merged centroid is
$\mu_{A\cup B}=(|A|\mu_A+|B|\mu_B)/(|A|+|B|)$.
Expanding deviations around each child centroid makes the cross terms vanish:

$$
SSE(A\cup B)=SSE(A)+SSE(B)
+|A|\|\mu_A-\mu_{A\cup B}\|_2^2
+|B|\|\mu_B-\mu_{A\cup B}\|_2^2.
$$

Substituting the merged centroid gives the greedy merge cost:

$$
\Delta SSE(A,B)=\frac{|A||B|}{|A|+|B|}\|\mu_A-\mu_B\|_2^2.
$$

The SciPy/scikit-learn height convention used by this study is
$h=\sqrt{2\Delta SSE}$. For singleton clusters, this equals ordinary Euclidean
distance. For larger groups it also incorporates cluster sizes. The tests
reconstruct each merge's actual SSE change to check the convention. Height
retains distance units, while SSE has squared units. Do not read a height of 5
as an SSE increase of 5, or compare heights across methods as a quality score.

Ward assumes Euclidean geometry. Passing cosine or squared Euclidean distances
as if they were ordinary Euclidean distances does not preserve this objective.
The library interface rejects Ward with `metric="cosine"`.

For nonzero vectors, cosine dissimilarity is
$d_{\cos}(x,y)=1-x^Ty/(\|x\|_2\|y\|_2)$.
For unit vectors, $\|x-y\|_2^2=2d_{\cos}(x,y)$. This proportionality does not
make average linkage on cosine equivalent to average linkage on ordinary
Euclidean distances: taking square roots and taking means do not commute.
Cosine is a dissimilarity, not generally a metric satisfying triangle inequality.

## Reading the dendrogram without drawing it

The linkage matrix has one row per merge: `[left, right, height, count]`.
Leaves have IDs `0..n-1`; row `i` creates node `n+i`. Count is the number of
original observations under that node, not just its two direct children.
`model_to_linkage` preserves the full fitted scikit-learn tree; it does not
refit a separate SciPy hierarchy. It requires recorded distances and all merges.

In a dendrogram, each internal branch sits at its merge height. Rotating two
child branches changes leaf ordering without changing the hierarchy. In a
truncated view, a displayed leaf can represent an entire contracted cluster.
Neither horizontal proximity nor branch color is an independent statistical
result. The tree is not a probability model or a validated semantic taxonomy.

For nondecreasing heights and $2\leq K<n$, exactly $n-K$ merges have occurred
at an exact-K cut. A strictly interior horizontal threshold must satisfy:

$$
Z_{n-K-1,2}<t<Z_{n-K,2},
$$

using zero-based rows. `cut_height_for_k` returns the midpoint when such an
interior floating-point value exists, and `None` otherwise. It deliberately
handles internal cuts only; K=1 and K=n are endpoint partitions.
`cut_tree(..., n_clusters=[K])` instead follows a chosen merge order.

For observations `[1,2,8,9]`, average linkage gives heights `[1,1,7]` and
counts `[2,2,4]`. At threshold 4 there are two groups. Below height 1 there
are four; at height 1 both tied merges have happened and there are two.
There is no horizontal height giving three groups. A merge-order cut can stop
after the first merge and return three, with the particular solution depending
on tie handling. `fcluster(..., criterion="maxclust")` requests at most K
groups and should not be treated as an exact-K guarantee in tied cases.

## Implementation and engineering limits

`from_scratch.py` stores pairwise Euclidean distances, enumerates active cluster
pairs and recomputes each mean over its original cross-pairs. Ties use
`(height, smaller ID, larger ID)` order. Reordering input can change a tied
solution; this rule is reproducibility for fixed rows, not mathematical uniqueness.

For fixed feature dimension, recomputing cross-pair means at every iteration is
up to $O(n^3)$ time. The stored distance matrix is $O(n^2)$; initial broadcasting
also uses $O(n^2p)$ temporary storage. This is an educational implementation,
not a production library replacement. Its input contract requires at least two
finite real rows and one feature; overflow explicitly requests rescaling.

SciPy documents $O(n^2)$ time and memory for its unstructured single, complete,
average and Ward implementations. A condensed float64 vector alone holds
$n(n-1)/2$ entries: at 50,000 observations it uses 9,999,800,000 bytes, before
workspace and features. This is a storage calculation, not a measured benchmark.
Sparse connectivity constraints, sampling and coarse-to-fine analysis can reduce
cost but alter the problem or approximate the full hierarchy. Disconnected
connectivity graphs also require care with library completion behavior.

Standard agglomerative clustering provides no native scikit-learn `predict`.
Assigning future observations to a centroid, medoid or nearest neighbor is an
additional policy, not the original algorithm. Specify representation versions,
rebuild cadence, assignment checks and cluster identity changes. For document
taxonomies, inspect representative and ambiguous documents with domain reviewers;
evaluate retrieval effects separately before using cluster metadata operationally.

Unlike DBSCAN, the fitted flat partition assigns all observations, without a
noise label. A singleton is not automatically an anomaly. Silhouette evaluates
separation/cohesion in the chosen geometry, favors some shapes and is defined
only for $2\leq K\leq n-1$. It cannot establish semantic or business validity.
Stability checks should compare shared observations under perturbation or
resampling and distinguish label permutations from membership changes.

## Executed experiment — 2026-10-29

Executed by the AI assistant in this implementation phase. This is a new record;
results from earlier topics are not reused or overwritten.

**Hypothesis:** on separated, standardized synthetic blobs, different linkage
criteria may agree at K=3; changing the cut on one Ward tree changes granularity
without refitting. Equal merge heights can block an exact-K horizontal cut.

**Configuration:** `example.py`; `make_blobs(n_samples=90, centers=[[-3,-2],
[0,3],[3,-1]], cluster_std=[0.65,0.9,0.75], random_state=42)`;
`StandardScaler` fitted on all 90 rows for exploratory analysis; Euclidean metric,
no connectivity constraints, complete trees with distances. Four linkages at
K=3; Ward cuts K=2,3,4,5. K=3 comes from generator knowledge, not model selection.
No train/test split or claim of out-of-sample evaluation. Generator labels are
used only for descriptive ARI. Native thread pools limited to one thread.

**Environment:** Windows x64; Python 3.11.14; NumPy 2.4.6; SciPy 1.17.1;
scikit-learn 1.9.1. Focused tests use pytest 9.1.1.

**Observed linkage comparison:**

| Method | Sorted sizes at K=3 | Silhouette | Generator ARI |
|---|---|---:|---:|
| Single | 30, 30, 30 | 0.745478 | 1.000000 |
| Complete | 30, 30, 30 | 0.745478 | 1.000000 |
| Average | 30, 30, 30 | 0.745478 | 1.000000 |
| Ward | 30, 30, 30 | 0.745478 | 1.000000 |

**Observed cuts of the same Ward tree:**

| K | Sorted sizes | Silhouette | Interior threshold |
|---:|---|---:|---:|
| 2 | 30, 60 | 0.575175 | 12.716522 |
| 3 | 30, 30, 30 | 0.745478 | 6.789790 |
| 4 | 11, 19, 30, 30 | 0.619641 | 2.211375 |
| 5 | 11, 13, 17, 19, 30 | 0.490539 | 1.809450 |

The last four Ward merge heights were 2.063060, 2.359690, 11.219889 and
14.213154; their counts were 30, 30, 60 and 90. Converted-tree K=3 labels
matched the fitted Ward partition (ARI=1). On `[1,2,8,9]`, NumPy/SciPy average
matrices agreed with heights `[1,1,7]`; the K=3 interior threshold was `None`.
The separately run `from_scratch.py` printed that same hand-computed matrix.

**Interpretation candidates — author review required:** this easy geometry does
not separate the four linkage choices at the selected K. Among the four inspected
Ward cuts, K=3 has the highest silhouette and coincides with generator labels.
The large top merge heights are consistent with separated blobs in this design.
The four-point control illustrates why merge order and horizontal cuts differ
under ties.

**Limitations:** one seed, two continuous features, well-separated synthetic
groups, known generator count, no noise stress test or stability study. Agreement
at K=3 does not imply identical full hierarchies or a generally preferred method.
ARI against generator labels does not validate real categories; inspecting four
cuts is not a benchmark, a significance test or independent cluster-count
validation. No visual rendering or semantic/downstream evaluation was performed.

## Study-phase suggestions and follow-up work

These suggestions originated in the study phase. Separate visual controls now
illustrate a fixed bridge, a coordinate-scaling path and normalized synthetic
vectors; their configurations and validation are recorded in
[the visual guide](VISUAL_LAB.md). The historical experiment above remains
unchanged. Broader sensitivity, multi-seed stability and real-document evaluation
remain proposed rather than executed.

- Add bridge points between two groups and compare single/complete/average
  linkage while keeping input coordinates and cut count fixed.
- Change measurement units, then compare raw and standardized hierarchies.
- Perturb observations across several seeds; compare shared-observation
  partitions and representative membership rather than raw numeric labels.
- Normalize nonzero synthetic vectors; compare average cosine, Euclidean and
  squared-Euclidean hierarchies before attempting a real embedding taxonomy.

## Interview checks

**Why a hierarchy instead of independent K-Means fits?** One hierarchy gives
nested partitions across resolutions. K-Means can revisit assignments and is
often more scalable, but independent values of K need not nest.

**What does Ward optimize?** The next merge's SSE increase, including cluster
size effects. It does not find the globally minimum SSE partition for every K.

**Can Ward use cosine?** Its variance objective requires Euclidean geometry.
Average or complete linkage can use cosine dissimilarity, with zero vectors
handled explicitly and representation validity evaluated separately.

**How do you choose a dendrogram cut?** Treat height gaps and internal scores
as diagnostics; check stability and whether finer/coarser groups answer the
application question. Equal-height boundaries may require an explicit tie policy.

**What changes for millions of documents?** Avoid dense all-pairs storage;
consider sampling or coarse grouping and acknowledge the approximate/local
hierarchy. Validate domain labels, new-document assignment and retrieval utility
as separate decisions before deployment.
