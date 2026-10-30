# Neighborhoods, objectives and interpretation

## What the map is trying to retain

For $X\in\mathbb{R}^{n\times D}$, seek coordinates
$Y\in\mathbb{R}^{n\times 2}$. This generally cannot preserve all distances.
The choice of objective determines which distortions receive larger penalties.
PCA maximizes retained linear variance; t-SNE compares pair similarities; UMAP
constructs and embeds a weighted neighborhood graph. None automatically knows
whether a distance is meaningful for the domain.

For pixel counts with a common measurement scale, dividing every feature by 16
changes units but preserves Euclidean neighbor ranks. Standardizing each pixel
would give low-variance locations more relative weight and change those ranks.
For normalized document embeddings, choose a metric consistent with retrieval
and evaluate against that same original representation. A preprocessing PCA can
reduce cost on very wide inputs, but changes the representation being embedded;
record it and distinguish diagnostics against pre-PCA and post-PCA spaces.

## t-SNE: local probabilities and a heavy-tailed map

For $j\ne i$, Gaussian affinities and their symmetrization are

$$
p_{j\mid i}=
\frac{\exp(-\lVert x_i-x_j\rVert^2/(2\sigma_i^2))}
{\sum_{k\ne i}\exp(-\lVert x_i-x_k\rVert^2/(2\sigma_i^2))},
\qquad p_{ij}=\frac{p_{j\mid i}+p_{i\mid j}}{2n}.
$$

Self-affinities are zero. Bandwidth search targets
$\operatorname{Perp}(P_i)=2^{-\sum_j p_{j\mid i}\log_2 p_{j\mid i}}$.
This is an effective neighbor count derived from entropy, rather than a hard
cutoff. At equal distances, some perplexities cannot be attained exactly.

In two dimensions, define

$$
w_{ij}=(1+\lVert y_i-y_j\rVert^2)^{-1},\qquad
q_{ij}=\frac{w_{ij}}{\sum_{k\ne l}w_{kl}},\qquad
C=\sum_{i\ne j}p_{ij}\log\frac{p_{ij}}{q_{ij}}.
$$

The sum counts ordered off-diagonal pairs. The Student-t kernel's heavy tails
help address crowding. Minimizing $KL(P\Vert Q)$ penalizes separation of pairs
with large original affinity; it does not optimize pairwise-distance error.
The gradient is

$$
\frac{\partial C}{\partial y_i}
=4\sum_{j\ne i}(p_{ij}-q_{ij})w_{ij}(y_i-y_j).
$$

Positive $(p_{ij}-q_{ij})$ pulls a pair together under gradient descent;
negative values push it apart. The nonconvex objective depends on initialization
and optimization settings. See the [original paper](https://jmlr.org/papers/v9/vandermaaten08a.html).

`from_scratch.py` fixes every $\sigma_i=2$ rather than matching perplexity.
It evaluates dense pairs, masks zero-probability KL terms, computes log Q
without clipping, and takes plain gradient steps. Row-wise log-weight shifting
prevents all-zero Gaussian normalization under ordinary sharp bandwidths.
Overflowing distances raise explicit errors. Storage is $O(n^2)$ for fixed
dimension, and work is $O(n^2D)$ for affinities plus $O(n^2d)$ per gradient step.
No approximation, momentum, early exaggeration or convergence guarantee is
claimed. Finite differences verify the implemented loss/gradient pair, not
equivalence to scikit-learn's full algorithm.

## UMAP: graph membership and packing

With local offset $\rho_i$ and scale $\sigma_i$, a schematic directed membership
for an edge in the neighborhood graph is

$$
\mu_{i\to j}=\exp\left[-\frac{\max(0,d(x_i,x_j)-\rho_i)}{\sigma_i}\right],
\qquad
\mu_{ij}=\mu_{i\to j}+\mu_{j\to i}-\mu_{i\to j}\mu_{j\to i}.
$$

Absent edges have zero membership. The fuzzy union above is the usual union
setting; other set-operation mixes exist. The embedding curve
$\nu_{ij}=(1+a\lVert y_i-y_j\rVert^{2b})^{-1}$ uses $a,b$ calibrated from
`min_dist` and `spread`. A conceptual cross-entropy, omitting constants, is

$$
-\sum_{i<j}\left[\mu_{ij}\log\nu_{ij}
+(1-\mu_{ij})\log(1-\nu_{ij})\right].
$$

The practical stochastic optimizer samples edges and negative pairs. It should
not be described as computing this full dense loss exactly; sampling and
weighting affect effective attraction/repulsion. Manifold and local metric
assumptions motivate the method, but arbitrary data need not satisfy them.
[Original paper](https://arxiv.org/abs/1802.03426) and
[implementation intuition](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html).

`n_neighbors` sets the graph's neighborhood scale. `min_dist` changes the
preferred packing curve, rather than imposing a guaranteed minimum pairwise
distance. More spread or compactness is not automatically more faithful geometry.
[Official parameter guide](https://umap-learn.readthedocs.io/en/latest/parameters.html).

## Measuring local fidelity and seed stability

Let $N_X^k(i)$ and $N_Y^k(i)$ denote original and embedded kNN sets, excluding
self. The example reports

$$
R(k)=\frac{1}{n}\sum_i\frac{|N_X^k(i)\cap N_Y^k(i)|}{k}.
$$

This is original-neighbor recall; at equal k its numeric value is also neighbor
precision. It counts every lost neighbor equally. In contrast, trustworthiness
weights introduced neighbors by their rank in X:

$$
T(k)=1-\frac{2}{nk(2n-3k-1)}
\sum_i\sum_{j\in N_Y^k(i)\setminus N_X^k(i)}(r_X(i,j)-k).
$$

Rank starts at one among non-self points; require $k<n/2$ for the implementation's
normalization. An introduced neighbor ranked just beyond k costs relatively
little, so high T can coexist with moderate R. Neither establishes global or
semantic fidelity. See the [metric API](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html).

For seed comparisons, replace X and Y neighbor sets in R with the two embedded
neighbor sets. This measures neighborhood agreement without aligning coordinates;
it says nothing about whether both runs preserve the *right* neighbors. Pure
rotation, reflection and translation leave Euclidean neighborhoods unchanged.
The helper breaks exact ties by input-row order; scikit-learn's trustworthiness
can use a different ordering at ties. Ties and near ties limit precision of
individual-rank interpretations.

There are three different scales: t-SNE perplexity, UMAP `n_neighbors` and
diagnostic k. Values of 30, 20 and 10 need not describe the same neighborhood.
Never compare objectives across the two libraries as if their units matched.
Even t-SNE KL values change their target P when perplexity changes.

## Common mistakes in applied work

| Tempting inference | What to check instead |
|---|---|
| Large 2D gaps prove semantic separation | Query original-space neighbors and evaluate labeled relevance |
| Group area or density estimates original prevalence/density | Inspect sampling, original-space density and population counts separately |
| Smaller `min_dist` means better clusters | Measure stability and validate clusters in the representation used for decisions |
| One isolated dot proves an anomaly | Check original features, source quality and an independent anomaly score |
| Independent fits have comparable axes or absolute coordinates | Compare neighbors; alignment alone does not repair distortions |
| Highest trustworthiness identifies the best retrieval model | Evaluate held-out query relevance with Recall@k, MRR or nDCG |
| Fitting on all rows is always leakage | Full-data exploratory mapping is transductive; a predictive evaluation requires a training-only learned transform |

UMAP can transform held-out points approximately. Fit it and any learned scaling
inside each training fold when evaluating a classifier; never refit on held-out
rows to improve their placement. scikit-learn t-SNE lacks `transform()`.
Distribution shift can invalidate a previously useful UMAP mapping.
[UMAP transform documentation](https://umap-learn.readthedocs.io/en/latest/transform.html).

Seeds help reproduce a specified software environment, but are not a portability
guarantee across packages or hardware. These runs set UMAP `n_jobs=1` and constrain
BLAS/OpenMP pools to one while fitting. Numba warmup is not a fair steady-state
runtime comparison. [Reproducibility guide](https://umap-learn.readthedocs.io/en/latest/reproducibility.html).

## Interview questions

**Why does a high trustworthiness score not prove the map preserves all neighbors?**
It rank-weights introduced neighbors, rather than directly counting every lost
original neighbor. Report neighbor recall at several k and inspect original
distances; neither metric covers all global relationships.

**Can UMAP be used before clustering or classification?**
Yes, with explicit validation of dimensionality, stability and task outcomes.
Two components selected for display may discard task-relevant structure.
For prediction, evaluate training-only fitting and held-out transforms.

**Why not implement a full t-SNE optimizer here?**
The exact probability model and gradient expose the learning mechanism. Adaptive
perplexities and scalable optimization are substantial additional engineering;
omitting them is acceptable only when the simplified model is named honestly.

**How would you inspect a suspicious RAG map?**
Verify model version, normalization, metric, duplicates and representative
sampling. Inspect retrieved document pairs in original space, then score held-out
query relevance. The display supplies investigation candidates rather than a
retrieval-quality conclusion.

## Executed experiments — 2026-10-30

Executed by Codex on Windows x64 in an isolated stable Python 3.11.14
environment. Package versions: NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1,
umap-learn 0.5.12, Numba 0.68.1, PyNNDescent 0.7.0, threadpoolctl 3.7.0 and
pytest 9.2.0. These are run metadata, not dependency pins. Both scripts were
executed from the repository root with that environment's Python executable.

### Digits fidelity and controlled variations

**Hypothesis:** different embedding objectives and neighborhood/packing settings
can preserve different neighbor sets. Trustworthiness alone may conceal loss
of exact original neighbors. Changing a seed is useful only if it actually
perturbs the computation.

**Configuration:** `python 02-unsupervised-recommender-systems/40-t-sne-umap/example.py`.
800 of the bundled 1,797 real digits, selected without replacement using
`default_rng(42).choice`; preserve generated row order. Sample-index SHA-256,
computed from little-endian int64 bytes:
`21c86357cdb1bceb839ee0daaacedda29be739863ac4e8eafdf23132703c9b38`.
64 count features divided by 16, Euclidean metric, no fitted standardization,
no preliminary PCA, no labels used. Every output has two dimensions. Evaluate
the same rows at k=10 and k=30. BLAS/OpenMP pools limited to one during fitting;
library `n_jobs=1` where supported.

- PCA: full SVD, no whitening.
- t-SNE: PCA initialization, Barnes-Hut with angle=0.5, learning rate `auto`
  (50 here), early exaggeration=12, max_iter=1,000; other defaults include
  min_grad_norm=1e-7 and n_iter_without_progress=300. Baseline perplexity=30,
  seed=42; change only perplexity to 50 or only seed to 21.
- UMAP: spectral initialization, n_neighbors=20, spread=1, n_epochs=500,
  default learning rate=1 and negative_sample_rate=5. Baseline min_dist=0.1,
  seed=42; change only min_dist to 0.8 or seed to 21. `transform_seed` matches
  the embedding seed, though no held-out transform is invoked.

**Results:** run completed at 2026-10-30 14:13:46 UTC (11:13:46 São Paulo).
The dated stdout JSON, including all estimator parameters, remains ignored
under `outputs/`; only the following numerical record is public. All seven
embeddings had finite coordinates. Rounded diagnostics:

| Configuration | T(10) | R(10) | T(30) | R(30) |
|---|---:|---:|---:|---:|
| PCA | 0.826850 | 0.185000 | 0.826295 | 0.327208 |
| t-SNE p=30, seed=42 | 0.988967 | 0.643375 | 0.976052 | 0.678583 |
| t-SNE p=50, seed=42 | 0.987579 | 0.640125 | 0.973534 | 0.679500 |
| t-SNE p=30, seed=21 | 0.988967 | 0.643375 | 0.976052 | 0.678583 |
| UMAP min_dist=0.1, seed=42 | 0.987360 | 0.578625 | 0.978028 | 0.668292 |
| UMAP min_dist=0.8, seed=42 | 0.979870 | 0.545375 | 0.973359 | 0.674083 |
| UMAP min_dist=0.1, seed=21 | 0.985411 | 0.573125 | 0.975890 | 0.665292 |

PCA's two-component explained variance ratio summed to 0.280564.
Across seeds, embedded neighbor overlap was 1.000000 at both k for t-SNE,
and 0.753250 / 0.875500 at k=10 / k=30 for UMAP.
A diagnostic PCA fit on the identical input confirmed the installed automatic
solver choice as `covariance_eigh`, which does not randomize its decomposition.

**Interpretation candidates — author review required:**

- On this sample the nonlinear methods retained more original neighbors than
  two-component PCA. High T did not imply complete neighbor retention: the
  baseline t-SNE T(10) was about 0.989 while R(10) was about 0.643.
- Changing perplexity or min_dist had different effects at k=10 and k=30;
  these diagnostics do not support a universal parameter winner. No inference
  about visual compactness was made because no map was rendered or inspected.
- The t-SNE seed comparison introduced no change in neighbor sets. In the
  installed implementation its PCA initialization used the deterministic
  covariance eigensolver. Identical neighbors
  here cannot establish robustness to a different initialization. UMAP's
  corresponding seed change altered neighbor sets despite similar T values.

**Limitations:** a single sample, two embedding seeds and a few prescribed
settings; no uncertainty interval, original-space cluster validation, timing
benchmark, held-out transformation or downstream retrieval/classification test.
PCA's variance objective differs from the nonlinear objectives. Pixel distances
can tie, and diagnostics are conditional on this sample, metric and software.
No fitted representation was selected as a production model.

### Synthetic fixed-bandwidth gradient descent

**Hypothesis:** a correctly differentiated, fixed-bandwidth objective can decrease
under a specified small demonstration; this does not reproduce full t-SNE.

**Configuration:**
`python 02-unsupervised-recommender-systems/40-t-sne-umap/from_scratch.py`.
Two synthetic four-dimensional Gaussian groups, 10 rows each, means -1.5 and
+1.5, standard deviation=0.5, data seed=42. Shared sigma=2, initialization
seed=42, centered Gaussian coordinates with pre-centering std parameter=0.01,
300 fixed gradient steps at learning_rate=10. Same Python/NumPy versions above.

**Result:** initial KL=0.669385208657154; final KL=0.005246165131381776;
all final coordinates finite. No plot or group-separation score was evaluated.

**Interpretation candidate — author review required:** the chosen steps reduced
the implemented loss on this toy input. Finite-difference tests support that
the analytical gradient differentiates that same objective.

**Limitation:** no adaptive perplexity, production optimizer, global-minimum
guarantee or stability claim for other learning rates and data. A small KL is
not proof of useful original-space clusters or semantic geometry.

### Validation scope

Focused tests cover probability normalization/symmetry, hand-calculated
affinities, sharp bandwidths and duplicate points, finite-difference gradients,
translation/rotation invariance, repeatable toy descent, invalid inputs and
neighbor tie/self handling. The integration smoke executes PCA, t-SNE and UMAP
on a small synthetic matrix; it does not assert which algorithm should win.
It shortens only UMAP's epoch count for the smoke; the digits run above uses
500 epochs. No test uses optional-dependency skipping. Repository-wide syntax
and local-link checks are separate from scientific or browser-render review.

## Visual-phase follow-up — 2026-10-30

The separately authorized [visual laboratory](VISUAL_GUIDE.md) now implements
Swiss Roll/real 3D exploration, adaptive probability search, actual NumPy t-SNE
trajectory frames, UMAP graph construction, parameter sweeps, deliberately random
t-SNE initialization and interpretation counterexamples. Its sample sizes,
optimizer and run records differ from the numerical study above. Historical
measurements and their pending interpretations are preserved.

## Suggested follow-ups — not executed

- Repeat over more sampling and embedding seeds with paired summaries; two
  embedding seeds are not a stability distribution.
- Vary UMAP `n_neighbors`, metric and per-feature scaling separately. Evaluate
  each against the appropriate original geometry.
- Add a Gaussian-noise negative control; without an inspected map, do not claim
  to have observed false visual clusters.
- Test training-only UMAP transforms under a controlled distribution shift and
  measure an independently selected downstream task.
