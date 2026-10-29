# DBSCAN: local density, global connectivity

## Neighborhoods and expansion

For finite real observations $X=\{x_1,\ldots,x_n\}$ and a distance $d$,

$$
N_\varepsilon(x_i)=\{x_j\in X:d(x_i,x_j)\leq\varepsilon\},
\qquad \mathrm{core}(x_i)\iff |N_\varepsilon(x_i)|\geq m.
$$

Here $\varepsilon>0$ is `eps` and the positive integer $m$ is `min_samples`.
The count includes the observation itself and repeated observations at the
same coordinates. The implementation covers unweighted data; scikit-learn
also supports sample weights, which change the count rule.

A non-core point is border if it lies within epsilon of a core point; it is
noise otherwise. A border point can be encountered before its component's
core and must still be eligible for assignment later.

Point $p$ is directly density-reachable from $q$ when $q$ is core and
$p\in N_\varepsilon(q)$. Density reachability follows chains of such steps.
It is generally asymmetric: a border point can be reached from core but
cannot propagate expansion. Density connectivity means reachability from
a common origin. Core points within epsilon form a graph; its connected
components determine cluster cores, with adjacent border points attached.
Long chains explain why curved clusters need no shared centroid.

Two core components may share a non-core neighbor. Assigning that border point
to the first discovered component is a valid DBSCAN convention; it must not
join the components. Reordering can change that assignment and arbitrary
numeric IDs while preserving core membership and core-component connectivity.
The [tests](tests/test_dbscan.py) use a symmetric one-dimensional construction
to separate these effects. Compare partitions on core points rather than
requiring literal label equality across implementations.

## What the density threshold assumes

For Euclidean distance in $D$ continuous dimensions, an intuitive fixed-radius
empirical density is

$$
\widehat f_\varepsilon(x_i)=\frac{|N_\varepsilon(x_i)|}{nV_D\varepsilon^D},
\qquad
\mathrm{core}(x_i)\iff \widehat f_\varepsilon(x_i)\geq
\frac{m}{nV_D\varepsilon^D}.
$$

$V_D$ is the volume of the unit Euclidean ball ($V_2=\pi$). This interpretation
assumes ordinary volume and ignores boundary bias; DBSCAN does not fit a
probability-density model. The self-count, sampling design, duplicates and
observation window affect the empirical density. At fixed radius and count,
adding data can turn noise into core points or create bridges.

At fixed $m$, increasing epsilon cannot remove core status, but total cluster
count need not be monotone: new components can appear and existing components
can merge. Raising $m$ at fixed epsilon cannot create core points; it can remove
or fragment components. One global pair of parameters can fail when groups
have different densities. Neither an automatic cluster count nor a noise label
eliminates these assumptions.

## Distances and parameter diagnostics

In this study the two axes share synthetic coordinate units, so all fits use
raw Euclidean geometry. Scaling is a modeling decision. Standardizing a
meaningful isotropic spatial geometry can distort distances; leaving unrelated
tabular units unscaled can let one feature dominate. Fit any learned transform
on the allowed historical/development population and keep it fixed when
evaluating a future assignment policy.

For GPS, use a suitable projected coordinate system with Euclidean distances
in meters, or haversine on `[latitude, longitude]` in radians with epsilon as
an angular radius (`distance / chosen Earth radius`). Do not feed geographic
degrees into a meter-based epsilon. This is domain guidance, not an executed
geospatial experiment; see the [haversine API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html).

The $m$-distance diagnostic uses each point's distance to its $m$-th observation
**including self**, then sorts these radii. With scikit-learn,
`NearestNeighbors(n_neighbors=m).fit(X).kneighbors(X)` includes self;
calling `kneighbors()` without `X` excludes it. That off-by-one changes the
diagnostic. For $m>n$, DBSCAN can legitimately return all noise, whereas this
nearest-neighbor diagnostic is undefined and explicitly raises an error.

A change in the sorted curve's slope can suggest candidates; it neither
guarantees an elbow nor proves an optimum. The example reports quantiles
without claiming to have inspected a curve. Check several nearby settings,
noise/coverage, population sizes and domain meaning before choosing parameters.

## Evaluation and operational trade-offs

| Approach | Useful question | Boundary |
|---|---|---|
| Cluster/core/border/noise counts | How much data is retained, and what supports expansion? | Counts alone do not establish useful groups |
| ARI against known labels | Does the partition match a synthetic generator or external taxonomy? | Taxonomies can disagree with density; all `-1` points form one label |
| Silhouette | How separated and compact is a partition in a specified metric? | Can penalize curved clusters; including noise treats it as a cluster, dropping noise changes coverage/population |
| Reordering, perturbation, resampling | Are core components and membership stable? | IDs must be matched; changed samples can change density legitimately |
| Domain review and downstream checks | Are clusters useful for an intended decision? | No business or downstream validation is performed here |

K-Means minimizes squared distances to centroids and assigns every point;
DBSCAN finds density-connected components and can abstain. Their objectives
differ, so inertia is not a common comparison metric. OPTICS provides a
reachability ordering across scales; HDBSCAN extracts clusters from a density
hierarchy. They are alternatives to investigate for varied density, not
guaranteed repairs. High-dimensional embeddings require distance and
representation checks; density clustering is not a replacement for retrieval.

Scikit-learn DBSCAN has `fit`/`fit_predict`, but no native `predict` for unseen
points. A separate frozen policy could assign new points within epsilon of a
stored core, with explicit handling of conflicts and abstention. Such a policy
does not update core counts and is not equivalent to refitting on the expanded
dataset. Store preprocessing, metric, parameters, core samples and a label
mapping; validate temporal coverage and usefulness before deployment.

The educational implementation calculates distances row by row in
$O(n^2D)$ arithmetic and stores an $O(n^2)$ distance matrix and, in the worst
case, $O(n^2)$ neighborhood indices. Queue expansion visits each assigned core
once and scans its neighbors. Scikit-learn can use indexed searches, but its
bulk neighborhoods still have worst-case quadratic memory. Neither implementation
is benchmarked here. Sparse radius graphs, duplicate aggregation with weights,
or a different algorithm can help at scale; validate their metric semantics.

## Executed experiment — 2026-10-29

Executed by Codex. Observations and candidate interpretations are separate;
**author review required** for every interpretation below. This is a new Day 37
record, not a replacement for Day 36 or for any historical experiment.

**Hypotheses.** At the source's proposed starting configuration, connectivity
may follow two curved moon populations more closely than K-Means. Changing
epsilon or `min_samples` may change coverage and connectivity. Injected
background is not equivalent to algorithmic noise.

**Configuration.** `make_moons(n_samples=450, noise=0.06, random_state=42)` plus
35 independent NumPy `default_rng(42)` uniform points between `[-1.4, -0.9]`
and `[2.4, 1.4]`; 485 total observations. All models fit the same complete raw
Euclidean corpus. Epsilon sweep `[0.08, 0.12, 0.18, 0.30, 0.45]` holds $m=7$;
density sweep `[3, 5, 7, 12, 20]` holds epsilon at 0.18 (shared default row
printed once). K-Means uses `K=2`, `n_init=10`, `algorithm="lloyd"`, seed 42;
one numerical thread. Generator labels are used only to diagnose the first
450 observations, with no search against those labels. DBSCAN is unweighted.

**Environment and commands.** Windows x64; Python 3.11.14, NumPy 2.4.6,
SciPy 1.17.1 and scikit-learn 1.9.1. From the root, Codex ran
`.local/environments/s08-reference-2026-10-08/Scripts/python.exe` with
`02-unsupervised-recommender-systems/37-dbscan-density-based-clustering/example.py`
and then with the sibling `from_scratch.py`. This is an existing stable local
environment, not a claim that the repository's default Python has those versions.
The portable commands appear in the [README](README.md).

**Observed results.** Cluster counts exclude noise. Core/border/noise counts
refer to all 485 observations; moon/background noise columns refer to 450/35
points respectively. ARI refers only to moons, retaining any `-1` assignments
as a single label, and is rounded to six decimals.

| eps | m | Clusters | Core | Border | Noise | Noise fraction | Moon noise | Background noise | Moon ARI |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.08 | 7 | 16 | 235 | 157 | 93 | 0.191753 | 66 | 27 | 0.129834 |
| 0.12 | 7 | 2 | 421 | 38 | 26 | 0.053608 | 3 | 23 | 0.986696 |
| 0.18 | 7 | 2 | 463 | 6 | 16 | 0.032990 | 0 | 16 | 1.000000 |
| 0.30 | 7 | 1 | 474 | 3 | 8 | 0.016495 | 0 | 8 | 0.000000 |
| 0.45 | 7 | 1 | 478 | 3 | 4 | 0.008247 | 0 | 4 | 0.000000 |
| 0.18 | 3 | 2 | 466 | 4 | 15 | 0.030928 | 0 | 15 | 1.000000 |
| 0.18 | 5 | 2 | 463 | 6 | 16 | 0.032990 | 0 | 16 | 1.000000 |
| 0.18 | 12 | 2 | 451 | 16 | 18 | 0.037113 | 0 | 18 | 1.000000 |
| 0.18 | 20 | 3 | 346 | 118 | 21 | 0.043299 | 0 | 21 | 0.768967 |

K-Means moon-only ARI was **0.228682**, with all 485 points assigned. At
epsilon 0.18 and $m=7$, NumPy and scikit-learn agreed on core and noise masks;
core-only and full partition ARIs between implementations were both **1.0**.
Sorted 7-distance quantiles at `[0.5, 0.9, 0.95, 0.99, 1]` were
`[0.080603, 0.131402, 0.163476, 0.517188, 0.962487]` in raw coordinate units.
These summaries do not identify a validated elbow.

The code-defined 1D demonstration, with `[0, 0.1, 0.2, 0.3, 0.55, 2]`,
epsilon 0.3 and $m=4$, printed labels `[0, 0, 0, 0, 0, -1]` and core mask
`[True, True, True, True, False, False]`.

**Interpretation candidates — author review required.** The proposed starting
parameters recover the moon generator partition on this corpus while leaving
16 background points unassigned. The remaining 19 background points belong
to clusters: generation provenance is not equivalent to density-based noise.
At epsilon 0.30 the moons lie in one cluster; at 0.08 the corpus fragments.
Raising $m$ to 20 at epsilon 0.18 fragments the result into three clusters.
These runs illustrate sensitivity and the different inductive biases of
centroid and density clustering.

**Limitations.** One deliberately favorable 2D generator and one seed, no
held-out data, no validated anomaly truth and no business evaluation. The
K-Means baseline uses generator knowledge of K. ARI excludes background and
does not measure noise quality; no setting was selected as optimal. No
runtime, memory, broad stability or high-dimensional comparison was measured.

**Focused validation.** Both scripts completed. The final focused run used
pytest 9.1.1 and passed 33 tests with no skips or warnings. An initial run had
32 passes and one failure because the proposed border-chain fixture actually
made its intended border point core at the inclusive epsilon boundary. Moving
that fixture point from 0.9 to 0.95 (and its neighbor from 1.8 to 1.9) corrected
the test geometry; the algorithm did not need a change. This fixture correction
does not change the synthetic experiment corpus or its results.

## Visual companion — separate execution phase

The [visual lab](VISUAL_LAB.md) adds exact intermediate states to the same NumPy
engine, ten geometric scenes and a supplementary distance-
concentration control. The original study-script record above remains unchanged;
a fresh execution after the trace refactor reproduced its printed numerical
results. Visual configurations and new observations have their own dated record
in the guide. Implementation, rendering and author interpretation review remain
separate states.

## Original follow-ups — status after the visual phase

- Multiply one axis by ten and compare justified preprocessing choices, with
  epsilon interpreted in each resulting geometry. This exact x10 proposal is
  unrun; the visual phase implements a separately documented x30 control.
- Generate populations of unequal density and compare OPTICS/HDBSCAN with a
  predefined evaluation policy. The visual phase adds a DBSCAN-only unequal-
  density control; the OPTICS/HDBSCAN comparison remains unrun.
- Perturb, resample or append observations and match core components across
  fits. The deterministic border-reordering unit test is not a stability study.
- Evaluate geographic hotspots with correct distance units and independent
  domain labels. Noise must remain an investigation candidate.

## Interview checks

**Does DBSCAN avoid choosing model complexity?** No. It avoids an explicit K,
but epsilon, density threshold, metric and representation control the partition.

**Why can a shared border point not join two clusters?** It lacks sufficient
neighbors to propagate expansion. Only core-to-core paths connect components.

**How would you choose epsilon without labels?** Start with domain units and
self-consistent neighbor radii, then assess parameter sensitivity, coverage,
component stability and usefulness. A silhouette maximum alone is insufficient.

**Would you serve these cluster IDs for new requests?** Only after designing
and validating an explicit assignment/abstention policy. DBSCAN itself does
not provide that prediction behavior, and refitting can change the partition.

**Is `-1` an anomaly verdict?** It is a density-dependent absence of cluster
membership. Sparse legitimate populations, bad units or incomplete sampling
can all produce that label.
