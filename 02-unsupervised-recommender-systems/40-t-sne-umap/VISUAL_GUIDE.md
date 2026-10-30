# Day 40 visual laboratory

Follow a neighborhood from its original geometry through probabilities, graph
connections and fitted coordinates. Then investigate which relationships were
lost. This is a CPU-only offline teaching laboratory, with measured diagnostics
and explicitly limited interpretation candidates.

## Install and run

Use an activated **stable Python 3.11+** environment. Dependencies are centralized
in [pyproject.toml](../../pyproject.toml); no separate requirements file is needed.
From the repository root:

```powershell
python -m pip install -e ".[dev,manifold]"
cd 02-unsupervised-recommender-systems/40-t-sne-umap
python visualize.py --quick --output-dir outputs/quick
python visualize.py --interactive
python visualize.py --animations
python visualize.py --experiments
python visualize.py --all
```

Open `outputs/index.html` directly in a browser after a standard command, or
`outputs/quick/index.html` after the quick command shown above. That first command
generates the four core modules with a smaller profile in a separate directory.
A complete existing
module is skipped when source/profile match. Changing source or profile in the
same directory requires explicit `--force`; alternatively use a different
`--output-dir`. `--force` replaces visual assets while retaining reusable fit
caches and earlier dated experiment records.

| Command | Modules |
|---|---|
| `--quick` | Swiss Roll, real digits 3D, t-SNE optimization, parameter comparisons; small profile |
| `--interactive` | All five Plotly explorers: Swiss Roll, digits 3D, UMAP graph, parameters, neighbor errors |
| `--animations` | Conditional-probability search, actual t-SNE optimization and separate parameter-fit GIFs |
| `--experiments` | Parameter fidelity/timing, distance/density traps, random initialization, neighbor errors |
| `--all` | All ten modules in learning order |
| `--demo graph` | One independent module; `--list` shows groups and supported module names |
| `--all --fast` | All modules using the smaller profile |

`--demo` names: `swiss`, `digits-3d`, `probabilities`, `optimization`, `kernels`,
`graph`, `parameters`, `geometry-traps`, `stability`, `neighbors`.
Groups also generate their associated explanatory PNGs/GIFs. `--interactive`
therefore includes the parameter grid and its GIF rather than discarding useful
outputs of the same computed fits.

| Setting | Quick / `--fast` | Standard |
|---|---:|---:|
| Public digits sample | 180 | 400 |
| Synthetic Swiss Roll | 220 | 500 |
| Library t-SNE maximum iterations | 500 | 1,000 |
| UMAP epochs | 150 | 400 |
| Educational optimization steps | 180 | 400 |
| Educational optimizer observations | 60 | 60 |

Numba can compile on the first UMAP fit. Fits and visual exports are bounded;
the NumPy probability/optimization implementation rejects inputs above 100 rows,
and the library helper caps inputs at 2,000 rows and 256 features. This lab is
not an interface for loading arbitrary massive embedding collections.

## Learning path

### 1. Swiss Roll — `interactive/01_swiss_roll.html`

- **Concept:** compare exact original 3D geometry with actual two-dimensional PCA,
  t-SNE and UMAP fits, using the same roll-position color throughout.
- **Mathematical intuition:** a linear projection cannot generally unfold a
  curved surface; nonlinear neighborhood objectives optimize different losses.
- **Observe:** rotate the roll, inspect rows, follow color continuity, compare
  calculated trustworthiness/recall. No method is required to produce a rectangle.
- **Common mistake:** interpreting Euclidean shortcuts between roll layers as
  intrinsic surface distances, or assuming a toy in 3D represents all high-D data.
- **Production relevance:** explore nonlinear image or multimodal representations;
  keep preprocessing and the original metric explicit.
- **Interview insight:** explain what linear variance retention cannot establish
  about nonlinear neighborhoods or semantic relevance.

### 2. Digits 3D — `interactive/02_digits_3d.html`

- **Concept:** genuine `n_components=3` PCA, t-SNE and UMAP outputs, switched with
  a dropdown; camera rotation, wheel zoom and class/row hover are available.
- **Mathematical intuition:** every model fits three coordinates directly. The
  scikit-learn implementation handles its dimension-dependent t-SNE kernel.
- **Observe:** the same digit colors from different camera angles and methods.
- **Common mistake:** treating nonlinear axes as independent semantic features
  or comparing absolute coordinate values across independent fits.
- **Production relevance:** inspect representation anomalies and possible class
  overlap before checking original features and task-specific outcomes.
- **Interview insight:** distinguish a real 3D fit from an invented noise axis.

### 3. Conditional probabilities — `animations/03_neighborhood_probabilities.gif`

- **Concept:** a reference point, original distances, Gaussian weights, normalized
  conditional probabilities and actual bandwidth-search iterations.
- **Mathematical intuition:** $p_{j\mid i}\propto\exp(-d_{ij}^2/(2\sigma_i^2))$;
  perplexity is $2^{H(P_i)}$, with entropy in bits. Self is excluded.
- **Observe:** targets 5, 15 and 30 on 72 synthetic 2D points. The dashed circle
  denotes sigma, not a boundary. A PNG places the three solved settings together.
- **Common mistake:** equating perplexity with an exact number of included points.
- **Production relevance:** neighborhood scale changes which relationships a map
  emphasizes when diagnosing document or image embeddings.
- **Interview insight:** explain per-observation adaptive bandwidths and why
  duplicate/tied distances can make certain target perplexities unattainable.

### 4. Optimization — `animations/04_tsne_optimization.gif`

- **Concept:** genuine saved optimization positions and a synchronized KL curve.
- **Mathematical intuition:** $\nabla_{y_i}C=4\sum_j(p_{ij}-q_{ij})w_{ij}(y_i-y_j)$,
  where $w_{ij}=(1+\lVert y_i-y_j\rVert^2)^{-1}$ and Q is globally normalized.
- **Observe:** the initial random state, later positions and measured KL. Arrows
  at row 0 are exact negative-gradient attraction/repulsion contributions with a
  common **display** scale; numerical updates sum the unscaled vectors and use
  backtracking. Frame labels contain actual step numbers.
  A persistent zoom inset makes the tiny initial random coordinates inspectable
  while the main embedding uses fixed limits throughout the animation.
- **Common mistake:** treating the arrow lengths as literal update displacements,
  the known synthetic labels as optimization inputs, or small KL as task quality.
- **Production relevance:** optimizer diagnostics differ from semantic validation.
- **Interview insight:** name the omitted momentum, early exaggeration, scalable
  approximations and advanced stopping controls. This visual optimizer matches
  perplexities; the original `from_scratch.py` study remains a separate, fixed-sigma
  experiment with its historical record preserved.

### 5. Crowding — `figures/05_crowding_kernels.png`

- **Concept:** Gaussian versus Student-t tails on linear and logarithmic scales.
- **Mathematical intuition:** Gaussian weights decay exponentially in squared
  distance; the Student-t kernel retains appreciable moderate-distance weights.
- **Observe:** equal peaks and different tails, rather than estimated densities.
- **Common mistake:** treating this analytical kernel illustration as a measured
  experiment proving that all crowding has disappeared.
- **Production relevance:** geometry must be checked against the representation's
  intended relationships, even when a familiar kernel is used.
- **Interview insight:** explain limited low-dimensional room for many neighbors.

### 6. UMAP graph — `interactive/06_umap_graph.html`

- **Concept:** original moons, kNN connectivity, row 0's directed local strengths,
  the complete fuzzy union and a final actual UMAP fit, for k=5,15,30.
- **Mathematical intuition:** $\mu_{ij}=\mu_{i\to j}+\mu_{j\to i}-\mu_{i\to j}\mu_{j\to i}$;
  offsets rho and scales sigma come from the fitted library model.
- **Observe:** stronger edges are thicker/more opaque; weights are binned for
  readability. Exact supplied neighbors include self first, as UMAP expects;
  self edges are then removed. Original geometry is genuinely two-dimensional.
- **Common mistake:** calling the stages optimization snapshots. The union is
  captured before finite-epoch scheduling can prune weak edges. No invented UMAP
  trajectory is shown. Edge arrowheads are omitted in the connectivity panel.
- **Production relevance:** graph scale changes the objective imposed on an
  embedding; locally scaled memberships do not estimate original density.
- **Interview insight:** separate graph construction, fuzzy union, sampled
  optimization and the final coordinate representation.

The graph audit uses installed UMAP `_sigmas`, `_rhos`, `_fit_embed_data` and
`graph_`, guarded by a numerical graph-equality test. These implementation-level
hooks may require adjustment after a library update. Supplied exact neighbors
have no search index, so UMAP warns that this demonstration cannot transform
new rows; that path is not used. Disconnected-graph warnings, if emitted, are
retained in its run record rather than hidden as evidence of quality.

### 7. Parameters and metrics — `interactive/07_parameters.html`

- **Concept:** twelve prescribed configurations, a side-by-side grid, a discrete
  t-SNE sweep GIF and numerical fidelity/timing plots.
- **Mathematical intuition:** compare trustworthiness and original-neighbor
  recall at k=5,10,30 while changing just one parameter within each family.
- **Observe:** t-SNE perplexity `[5,15,30,50]`; UMAP n_neighbors `[5,15,50,100]`
  at min_dist=0.1; UMAP min_dist `[0,0.1,0.5,0.9]` at n_neighbors=15.
- **Common mistake:** reading the sweep as a continuous trajectory. Each state
  is a separate fit; GIF transitions do not interpolate coordinates.
- **Production relevance:** inspect sensitivity, then evaluate held-out task
  outcomes instead of selecting a model by how separated its map appears.
- **Interview insight:** perplexity, graph neighbor count and evaluation k are
  different concepts. A parameter change can help one k and hurt another.

Timing is a single `perf_counter` measurement of `fit_transform` with one-thread
BLAS/OpenMP limits and `n_jobs=1` where supported. It includes possible JIT/warmup,
but excludes cache loading, metric evaluation and rendering. Cached fits retain
their original measured time; it is never replaced with the faster lookup time.
These observations are not a controlled speed benchmark or confidence interval.

### 8. Gaps and density — `figures/09_geometry_traps.png`

- **Concept:** compare three synthetic Gaussian populations in exact original
  2D, t-SNE and UMAP. All groups contain 60 observations.
- **Mathematical intuition:** measure centroid-pair distances relative to the
  largest pair and within-group median fifth-neighbor radius relative to group 0.
- **Observe:** the measured ratios under each plot; original std values are
  0.15,0.5,1.1 and center x coordinates are 0,4,12.
- **Common mistake:** inferring original separation, density or population size
  from the displayed gap, compactness or area. Radii are a diagnostic proxy,
  not a calibrated population-density estimate.
- **Production relevance:** verify customer or embedding segmentation in the
  feature space used to make decisions.
- **Interview insight:** explain why ratios can change even without a change in
  sample counts or original observations. This is a constructed case, not a
  proof that every map fails in the same way.

### 9. Stability — `figures/10_seed_stability.png`

- **Concept:** seed=0,21,42 with identical digits, metrics and preprocessing.
- **Mathematical intuition:** compare embedded kNN overlap against seed 42;
  rotations and reflections do not require coordinate alignment for this metric.
- **Observe:** t-SNE deliberately uses `init="random"` at perplexity=30; UMAP
  uses spectral initialization at n_neighbors=15, min_dist=0.1.
- **Common mistake:** inferring robustness from a seed parameter that does not
  actually perturb a deterministic PCA initialization, as in the earlier study.
- **Production relevance:** verify representation conclusions across realistic
  sources of variation, including sample composition and model versions.
- **Interview insight:** aggregate trustworthiness and run-to-run neighborhood
  agreement answer different questions. Three seeds do not establish a distribution.

### 10. Incorrect neighbors — `interactive/11_neighbor_errors.html`

- **Concept:** inspect lost, introduced and shared neighbors for selected digits.
- **Mathematical intuition:** original neighbors are calculated in **64D** using
  Euclidean distance; embedding neighbors use the real 2D UMAP coordinates.
- **Observe:** switch reference row and hover for exact source IDs. The left
  PCA panel is solely a display of the original rows, not the original metric.
- **Common mistake:** recomputing "true" neighbors in the PCA display or treating
  the deliberately selected lowest-recall row as typical of all observations.
- **Production relevance:** inspect document pairs, source quality, anomalies and
  relevance judgments alongside global embedding summaries.
- **Interview insight:** high average fidelity does not certify an individual
  retrieved result, a semantic distinction, or held-out Recall@k/nDCG.

## Files, outputs and reproducibility

- [visualize.py](visualize.py): CLI, bounded profiles, dated run records and index.
- [manifold_visual_core.py](manifold_visual_core.py): numerical mechanisms,
  exact recorded updates, fitted graphs, library fits, cache and measurements.
- [manifold_visualizations.py](manifold_visualizations.py): coherent explanatory
  HTML scenes, Matplotlib figures and Pillow animations.
- [tests/test_visual_lab.py](tests/test_visual_lab.py): mathematical invariants,
  finite differences, cache identity, real 3D fits and temporary export checks.

The HTML files share a **local** `outputs/plotly.min.js`; keep it beside the
index when moving the lab. No CDN, web server, credentials, external API, GPU
or runtime dataset download is needed. Interactive graphs support hover and
zoom; genuine 3D graphs also support rotation. GIFs are prerecorded and do not
respond to controls. The center page links every generated module.

Fit-cache keys include input bytes/order, shape, full model parameters, runtime
versions and numerical-source hash. Cache files contain arrays and JSON, never
pickled estimators. The seed alone is not a guarantee across versions/hardware.
The public digits source and license are in [references.md](references.md).

Generated HTML, figures, GIFs, fit caches and dated `run_*.json` stay in ignored
`outputs/` unless deliberately selected as documentation previews. Each run
record contains its hypothesis where applicable, configuration, versions,
measured results and interpretation status. Earlier nonvisual measurements in
[notes.md](notes.md#executed-experiments--2026-10-30) use 800 digits and different
settings; do not silently substitute this visual phase's 180/400-row results.

## Visual-phase execution record — 2026-10-30

Executed by Codex on Windows x64 with stable Python 3.11.14, NumPy 2.4.6,
SciPy 1.17.1, scikit-learn 1.9.1, umap-learn 0.5.12, Numba 0.68.1,
PyNNDescent 0.7.0, Matplotlib 3.11.2, Plotly 6.9.0, Pillow 12.3.0 and
threadpoolctl 3.7.0. pytest was 9.2.0. Versions are measurements, not a lock.

**Runs and configuration:** `--quick` completed from the topic directory into
an ignored local review directory, then `--all` generated all ten standard-profile
modules. Explicit `--force` runs regenerated visuals after inspection refinements;
input/version/parameter-matched caches reused the same real model fits and their
original timings. Earlier dated JSON records remain intact. The following table
uses only the standard profile, not quick-mode or original 800-digit measurements.

**Hypotheses:** neighborhood-focused fits can reveal curved structure; entropy
search can match prescribed perplexities; exact KL updates can reduce the toy
loss; changing parameters or initialization can change neighbors; 2D gaps and
local radii need not preserve their original-space ratios. Individual modules'
dated records retain their own hypothesis, settings, versions and measurements.

**Results:**

| Standard-profile representation | T(10) | Original-neighbor R(10) |
|---|---:|---:|
| Swiss Roll, PCA 2D | 0.940869 | 0.580200 |
| Swiss Roll, t-SNE 2D | 0.998712 | 0.866000 |
| Swiss Roll, UMAP 2D | 0.997481 | 0.795600 |
| Digits, PCA 3D | 0.895914 | 0.398000 |
| Digits, t-SNE 3D | 0.990685 | 0.753250 |
| Digits, UMAP 3D | 0.983441 | 0.666500 |

The NumPy trajectory used 60 synthetic 6D blob observations (three centers from
`make_blobs`, cluster_std=1.3), perplexity=10,
data/initialization seeds=42, 400 steps and initial learning_rate=100 with
backtracking. Initial coordinates were centered Gaussian draws with std=0.01.
KL decreased from **1.6816610851** to **0.1931547791**. Accepted
rates ranged from 50 to 100. Maximum achieved-perplexity error was below 1e-6.
This is a different model/data configuration from the historical fixed-sigma
20-point study. Its GIF has 41 decoded states; repeated end frames can be merged
by Pillow into longer frame durations.

The three UMAP graph reconstructions matched the library's captured pre-scheduling
union to within 1.2e-7 maximum absolute error. Finite-epoch scheduling changed
some very weak edges afterward; this distinction is retained in the run records.
Each exact-neighbor graph fit emitted the expected missing-search-index warning
about unavailable held-out transforms; no such transform is attempted.

| Controlled parameter | Value | T(10) | R(10) | T(30) | R(30) | Original fit seconds |
|---|---:|---:|---:|---:|---:|---:|
| t-SNE perplexity | 5 | 0.975124 | 0.635000 | 0.936002 | 0.641333 | 1.238 |
| t-SNE perplexity | 15 | 0.983432 | 0.694750 | 0.945098 | 0.668083 | 1.339 |
| t-SNE perplexity | 30 | 0.982371 | 0.698750 | 0.946811 | 0.680417 | 1.858 |
| t-SNE perplexity | 50 | 0.977960 | 0.690250 | 0.939307 | 0.672667 | 1.916 |
| UMAP n_neighbors | 5 | 0.972500 | 0.627750 | 0.924056 | 0.626167 | 0.538 |
| UMAP n_neighbors | 15 | 0.980699 | 0.644250 | 0.944665 | 0.664333 | 0.697 |
| UMAP n_neighbors | 50 | 0.975869 | 0.598000 | 0.950120 | 0.675500 | 1.129 |
| UMAP n_neighbors | 100 | 0.972155 | 0.597000 | 0.943103 | 0.669250 | 1.040 |
| UMAP min_dist | 0.0 | 0.981245 | 0.631750 | 0.936209 | 0.654000 | 0.611 |
| UMAP min_dist | 0.1 | 0.980699 | 0.644250 | 0.944665 | 0.664333 | 0.697 |
| UMAP min_dist | 0.5 | 0.978151 | 0.644000 | 0.951522 | 0.680417 | 0.751 |
| UMAP min_dist | 0.9 | 0.971764 | 0.630000 | 0.947483 | 0.681167 | 0.632 |

Within the synthetic three-group counterexample, the relative centroid distance
for groups 0–1 changed from 0.324132 in original space to 1.0 in both maps.
Relative within-group fifth-neighbor radius for group 2 versus group 0 changed
from 9.529958 to 0.988041 in t-SNE and 1.000300 in UMAP. Ratios are normalized
separately per representation; coordinate units are not directly comparable.

With actually randomized t-SNE initialization, neighbor overlap@10 against seed
42 was 0.929250 for seed 0 and 0.957000 for seed 21. UMAP's corresponding
overlaps were 0.783750 and 0.807750. The diagnostic extreme in the neighbor-error
module was sample row 25 (source row 206), with recall@10=0.2; rows 0 and 1
had recall@10=0.7 and 0.9. The extreme was selected using this same sample.

**Interpretation candidates — author review required:**

- These nonlinear fits retained more original neighbors than the specified PCA
  baselines. This does not show that every curved dataset will unfold correctly
  or establish meaningful semantic groups.
- Parameter rankings changed with the diagnostic and k. High T can coexist with
  moderate exact-neighbor recall or a poor individual row; attractive gaps are
  insufficient evidence for choosing a model.
- The synthetic maps changed relative gaps and local radii despite equal counts.
  This illustrates specific possible distortions, not a universal failure claim.
- Actual initialization changes affected neighbor sets. Similar aggregate
  fidelity and a shared seed parameter do not establish stable relationships.

**Validation and inspection:** 67 focused topic tests passed with no skips or
warnings, including the original 41 tests plus 26 visual tests. All eight PNGs
were decoded; all frames of the three GIFs were decoded. The GIFs contained
30 probability-search states, 41 optimization states and four discrete sweep
states. Preview candidates and browser captures were inspected visually.
Offline Chrome testing opened the index and all five explorers, verified 21
dropdown states, an actual dropdown click, actual mouse rotation and wheel
zoom, with zero JavaScript exceptions and zero remote requests. Headless local
inspection does not certify every browser, GitHub rendering or scientific review.

**Limitations:** one sample per profile, prescribed rather than exhaustive
parameters, three embedding seeds, Euclidean geometry, in-sample diagnostics,
small synthetic mechanisms and implementation-specific UMAP audit hooks.
There is no formal clustering validation, held-out classifier/retrieval result,
confidence interval, scalability study or controlled performance comparison.

## Selected public previews

Only these three inspected assets are deliberately unignored and referenced in
the [topic README](README.md):

- [Conditional-probability search](outputs/animations/03_neighborhood_probabilities.gif).
- [Actual t-SNE optimization](outputs/animations/04_tsne_optimization.gif).
- [Misleading gaps and local-radius ratios](outputs/figures/09_geometry_traps.png).

They total approximately 1.5 MiB. All HTML, the shared bundle, other PNGs/GIFs,
fit caches, saved optimizer arrays and dated run records remain ignored.

## Further experiments — not executed

- Repeat with new sampling seeds and distributions, rather than only changing
  an embedding seed on one fixed sample.
- Add a continuous or Gaussian-noise negative control and inspect it before
  making any claim about artificial visual separation.
- Compare cosine/normalized embeddings against the appropriate original metric.
- Evaluate training-only UMAP transforms under shift, held-out clustering or
  retrieval relevance. This lab performs exploratory in-sample analysis.
