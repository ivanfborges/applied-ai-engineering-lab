# Day 36 — K-Means Clustering

**Module 2: Unsupervised Learning and Recommender Systems** ·
[Module index](../README.md) · [Roadmap](../../ROADMAP.md)

How much of a clustering comes from the data, and how much comes from the
distance representation? K-Means gives every observation a nearest centroid
and minimizes within-cluster squared Euclidean distance. It is useful for
segmentation, vector quantization and coarse candidate grouping, provided that
compactness in the chosen feature space serves the task.

This study connects centroids, Lloyd updates, initialization, inertia, the elbow
heuristic and silhouette to a controlled scaling experiment. It uses only
synthetic data. Compact groups do not establish meaningful customer segments,
semantic topics, personalized recommendations or better retrieval.

The objective is

$$
J = \sum_{i=1}^{n}\lVert x_i-\mu_{c_i}\rVert_2^2.
$$

Assign each point to its nearest center, then replace each nonempty center by
the mean of its members. Exact steps cannot increase this objective, but
initialization can lead to different local solutions. Increasing $K$ lowers or
maintains the **globally optimal** objective; independently fitted local solutions
need not be monotone. Inertia alone therefore cannot choose $K$.

## Read and run

| File | Purpose |
|---|---|
| [notes.md](notes.md) | Mean derivation, geometry, elbow reasoning, scaling, assumptions and deployment boundaries |
| [example.py](example.py) | Raw/standardized K sweeps, shared-space partition diagnostics, matched initialization and frozen-pipeline inference |
| [from_scratch.py](from_scratch.py) | Educational NumPy Lloyd loop with inertia history and explicit empty-cluster/stopping behavior |
| [tests/test_kmeans.py](tests/test_kmeans.py) | Hand-computed cases, monotonicity, stopping, invalid inputs, library agreement and frozen preprocessing |
| [interview_questions.md](interview_questions.md) | Questions about objective, validation, stability and embedding use |
| [references.md](references.md) | Official algorithm, metric and synthetic-generator sources |

From the repository root, with stable Python 3.11+ and the canonical
[dependencies](../../pyproject.toml):

```bash
python -m pip install -e ".[dev]"
python 02-unsupervised-recommender-systems/36-k-means-clustering/example.py
python 02-unsupervised-recommender-systems/36-k-means-clustering/from_scratch.py
python -m pytest -q 02-unsupervised-recommender-systems/36-k-means-clustering/tests/test_kmeans.py
```

Both scripts print to the terminal and write no data, plots or model artifacts.
No notebook or separate visual phase is included. Generated caches remain
ignored; no preview assets are selected for publication.

## Executed study record — October 29, 2026

Executed by Codex on Windows x64 with Python 3.11.14, NumPy 2.4.6,
scikit-learn 1.9.1, SciPy 1.17.1, threadpoolctl 3.7.0 and pytest 9.1.1.
The numerical example limits numerical libraries to one thread. These are
environment observations, not a lock or a runtime benchmark. Both scripts
completed; all 37 focused tests passed without skips or warnings.

### Scaling and cluster count

**Hypotheses:** changing one feature's units can change the partition; within
each fixed representation, inertia can keep improving after silhouette peaks.

**Configuration:** `make_blobs`, 600 observations, centers `(-4, -1)`, `(0, 4)`,
`(4, -1)`, standard deviations `0.8`, `1.1`, `0.7`, seed 42. Multiply the
second feature by 50 before either fit. Compare raw data with a pipeline using
`StandardScaler` fitted on this same exploratory corpus. Sweep $K=1,\ldots,8$;
use K-Means++, `n_init=10`, `max_iter=300`, `tol=1e-4`, seed 42 and Lloyd's
algorithm for each fit. Generating labels never enter fitting or the sweep.

**Results:** each inertia and silhouette below belongs to its column's fit
space. Their absolute values must not be ranked across these two spaces.

| K | Raw inertia | Raw silhouette | Standardized inertia | Standardized silhouette |
|---:|---:|---:|---:|---:|
| 1 | 9336518.1436 | undefined | 1200.0000 | undefined |
| 2 | 1169890.2074 | 0.788463 | 638.7874 | 0.526907 |
| 3 | 756469.9022 | 0.699833 | 113.5533 | 0.756527 |
| 4 | 398201.7034 | 0.554276 | 87.0581 | 0.644056 |
| 5 | 289046.3384 | 0.554097 | 75.0782 | 0.513935 |
| 6 | 206587.6585 | 0.507788 | 63.5291 | 0.384767 |
| 7 | 159424.8340 | 0.524056 | 54.1562 | 0.372418 |
| 8 | 121994.1383 | 0.526976 | 48.9988 | 0.371379 |

At a fixed illustrative $K=3$, independent of metric selection:

| Fit | Cluster sizes, sorted | ARI against synthetic generating labels | Silhouette in shared standardized space |
|---|---|---:|---:|
| Raw | 98, 104, 398 | 0.440059 | 0.314778 |
| Standardized | 200, 200, 200 | 1.000000 | 0.756527 |

The ARI between the two fitted partitions was 0.440059. ARI compares
partitions independently of arbitrary numeric cluster IDs. All 16 fitted
inertias agreed with a separate residual-based calculation in the code.

**Interpretation candidates — pending author review:** in this deliberately
distorted generator, standardization recovered the generating partition at
$K=3$, while raw distances produced a different partition. The standardized
inertia reductions suggest an elbow near three; standardized silhouette also
peaked at three within the evaluated range. Raw silhouette peaked at two.
These diagnostics support inspecting the representation before choosing $K$.
The continued inertia reduction past three does not establish better recovery.

**Limits:** one favorable synthetic corpus and one estimator seed, with ten
restarts per fit. The shared evaluation space itself encodes a choice of
geometry. Neither the elbow nor ARI demonstrates that real data have three
useful groups. This is exploratory fitting on all rows, without held-out
generalization, resampling stability or domain validation.

### Matched implementation and inference checks

**Hypothesis:** with identical initial centers and a stable partition, the
educational loop should match scikit-learn's objective and partition.

**Configuration:** use the same standardized 600-point corpus, three initial
centers from `kmeans_plusplus(random_state=42)`, one run for each implementation,
`tol=0`, `max_iter=300`. This is separate from the ten-restart sweep above.

**Result:** both inertias were 113.553274 to six decimals; partition ARI was
1.000000. NumPy stopped after one mean update with unchanged assignments.
The separate six-point scratch demonstration produced centers
$(2/3,2/3)$ and $(32/3,32/3)$, inertia changing from 16 to
10.666666666666668 in one update, and labels `[0, 0, 0, 1, 1, 1]`.

**Interpretation candidate — pending author review:** this supports agreement
of the assignment/update logic on these controlled inputs.
**Limit:** matched initialization on well-separated data does not establish
equivalence on ties, empty clusters or other numerical environments. The scratch
implementation omits optimized distances, K-Means++ and multiple restarts.

The standardized pipeline also assigned three new synthetic raw-unit points
`[-4, -50]`, `[0, 200]`, `[4, -50]` to `[1, 2, 0]` with its already fitted
scaler and centers. This checks executable inference; IDs have no permanent
meaning and the three constructed points do not measure predictive quality.

## Boundaries and next investigations

Scaling defines feature weights; standardization is a modeling decision.
Correlations, outliers, nonconvex groups, unequal densities and irrelevant
dimensions can still make K-Means unsuitable. Every new point receives an
assignment, even far from all centers. Distances are not membership or anomaly
probabilities. Keep the scaler, feature schema and centroids together at inference.

The following remain **suggestions, not executed experiments**: compare random
versus K-Means++ initialization over multiple seeds; resample rows and assess
partition stability; add outliers; test elongated groups or moons; compare
MiniBatchKMeans or a density-based method. Evaluate downstream usefulness before
assigning semantic names or using clusters in recommendation or retrieval.

Implementation and automated validation are complete for this scope.
Interpretation candidates above still require the author's explicit review.
