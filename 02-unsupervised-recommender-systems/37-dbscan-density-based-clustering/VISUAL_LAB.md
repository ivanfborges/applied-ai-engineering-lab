# DBSCAN: observe the geometry

[Study overview](README.md) · [Generator](generate_visualizations.py) ·
[Numerical helpers](dbscan_visual_core.py) · [Exact engine](from_scratch.py) ·
[Visual tests](tests/test_visual_dbscan.py)

This local learning lab connects epsilon balls, density counts and graph
traversal to observable behavior. Every dataset is synthetic or defined in
code. Colors name clusters within a fit; IDs and colors are not persistent
identities across fits. Circles identify core points, squares identify border
points and grey crosses identify noise. The classification-only scene colors
roles directly: blue core, amber border and grey noise.

## Generate and explore

Use stable Python 3.11+ and the repository's canonical dependencies:

```bash
python -m pip install -e ".[dev]"
cd 02-unsupervised-recommender-systems/37-dbscan-density-based-clustering
python generate_visualizations.py --list
python generate_visualizations.py --all
python generate_visualizations.py --visualization 3
python generate_visualizations.py --visualization 5 --output-dir outputs/another-run
```

The default output directory is beside the script, independent of the current
working directory. A relative `--output-dir` is resolved from the caller's
directory. Existing names in that directory are replaced; use a distinct
directory to preserve a historical export. The scripts never open GUI windows
or start a server. Matplotlib uses Agg, animations use FuncAnimation and
PillowWriter, and Plotly HTML embeds its JavaScript. No FFmpeg, Kaleido, cloud,
external dataset or internet connection is required for generation/viewing.

Open generated `outputs/index.html` in a local browser. It links available
artifacts and displays image exports; open standalone HTML files for hover,
rotation and parameter controls. GitHub does not display interactive HTML
inline. 3D interaction requires browser WebGL support. The HTML files are
larger than images because each bundles Plotly.js.

Each command writes an ignored `visual_report_NN.json` (or `visual_report_all.json`)
with UTC time, versions, numerical observations, configuration and artifact
sizes. These are reproducibility records, not scientific approval. The seed is
fixed at 42 where randomness is used. Exact algorithm controls are code-defined.
Failures stop the command and identify the scene; they are not silently skipped.

## The mechanism in symbols

$$
N_\varepsilon(x)=\{y\in X:d(x,y)\leq\varepsilon\},
\qquad |N_\varepsilon(x)|\geq m \Longleftrightarrow x\text{ is core}.
$$

$X$ is the observed sample, $x$ is the reference observation, $y$ is a candidate
neighbor, $\varepsilon$ is the radius, and $m$ is `min_samples`, including self.
For the real feature coordinates used here,

$$
d(x,y)=\sqrt{\sum_{j=1}^{D}(x_j-y_j)^2},
$$

where $j$ indexes the $D$ input features. Changing feature units changes this
distance. Border points have fewer than $m$ neighbors but neighbor a core;
noise has no core neighbor. The metric and sampling window are model choices.

## 01 — What does a larger epsilon include?

Artifact: `01_epsilon_neighborhood.gif`.

**Concept:** the epsilon neighborhood of one fixed observation. **Mechanism:**
the displayed circle is the exact Euclidean radius and its count includes the
reference star. **Observe:** the count rises as the radius grows and falls as
it shrinks, while `min_samples=7` remains fixed; core status switches at the
threshold. **Takeaway:** epsilon has units and defines local density, not a
maximum diameter for an entire cluster. This is one neighborhood, not a fit.

## 02 — Why is non-core different from noise?

Artifacts: `02_core_border_noise.png`, `02_core_border_noise.gif`.

**Concept:** core, border and noise are distinct roles. **Mechanism:** core
meets the count rule; a border point fails it but is within epsilon of core.
Dashed circles show one core and one border neighborhood. **Observe:** the
border's own count is too small, yet its position allows attachment. The GIF
inspects/reveals roles from a fixed completed fit; it is not the execution
trace. **Takeaway:** unassigned density membership is not an anomaly verdict.

## 03 — Which points propagate a cluster?

Artifacts: `03_dbscan_expansion.gif`, `03_dbscan_final.png`.

**Concept:** cluster membership propagates through core connectivity.
**Mechanism:** the engine records exact selection, count inspection, cluster
creation, queue expansion, core addition, border attachment, provisional noise
and component completion. **Observe:** point 0 is first marked noise and later
becomes border; square border points never enter the expansion queue. Dark
rings mark expanded core points, a gold star marks the current point, and
gold rings mark its neighborhood. Completed clusters retain their colors.
**Takeaway:** a shared border can attach to the first component encountered
without connecting the cores. The symmetric order-ambiguity construction is
tested separately; not every concept is forced into the small animated cloud.

The trace and `dbscan_numpy` use one implementation. Pending labels are `-2`,
provisional/final noise is `-1`, and nonnegative labels denote cluster IDs.
Counts are precomputed in a quadratic distance matrix; captions show when
the algorithm consults them. Snapshots are exact and paused, never interpolated.
Core roles are fixed for this input and parameter pair. Snapshot generation
is limited to 200 points; the selected control uses only 27. Trace snapshots
add memory beyond the $O(n^2)$ distances/neighborhoods. This is educational
code, not a production library replacement or performance benchmark.

## 04 — Why can two methods disagree on the same coordinates?

Artifact: `04_kmeans_vs_dbscan.png`.

**Concept:** centroid proximity versus density connectivity. **Mechanism:**
the K-Means panel includes nearest-centroid background regions, their bisector
and fitted centroids; DBSCAN has no centroid objective. **Observe:** the same
curved moon sample can be split by centroid regions and followed by chains of
core points. K-Means assigns all observations, DBSCAN can abstain. **Takeaway:**
choose the grouping rule for the problem; DBSCAN is not a universal winner.
The known generator K=2 is supplied to the baseline. Background generation
does not define noise truth. This scene reuses the 485-point study corpus.

## 05 — How do radius and density requirement interact?

Artifacts: `05_parameter_sensitivity.html`, `05_cluster_count_heatmap.png`,
`05_noise_fraction_heatmap.png`.

**Concept:** joint sensitivity to epsilon and `min_samples`. **Mechanism:**
precomputed fits for six radii and five counts update the point roles and
summary statistics; heatmaps summarize all 30 settings. **Observe:** small
radii can fragment populations or leave noise; larger radii can create core
points and bridges. Increasing `min_samples` raises the local count requirement.
**Takeaway:** cluster count need not vary monotonically. These are sensitivity
diagnostics, not automatic parameter selection or evidence of stable segments.

## 06 — Does the k-distance curve determine epsilon?

Artifact: `06_k_distance_plot.png`.

**Concept:** a local-radius diagnostic. **Mechanism:** sort the distances to
neighbor $m=7$, including self, by explicitly passing the indexed sample to
`kneighbors(X)`. **Observe:** candidate horizontal lines cut different parts
of the same empirical curve. **Takeaway:** a slope change suggests candidates,
not a uniquely optimal epsilon. Variable densities may make the elbow unclear.

## 07 — What does clustering three actual features look like?

Artifact: `07_dbscan_3d_interactive.html`.

**Concept:** density connectivity in genuine 3D vectors. **Mechanism:** fit
Euclidean DBSCAN to all x/y/z coordinates of two separated elongated, curved
groups plus background. This is not a 2D embedding displayed with an invented
height. **Observe:** rotate/zoom, hover for index, label, role and all coordinates,
and inspect noise. **Takeaway:** view occlusion is not numerical connectivity;
rotate before interpreting a projection. Data aspect and orthographic camera
reduce display distortion. Three readable features do not validate embeddings.

## 08 — Is a density surface the DBSCAN objective?

Artifact: `08_density_landscape_3d.html`.

**Concept:** the geometric meaning of a fixed-radius local count.
**Mechanism:** evaluate counts on a 2D grid and divide by $n\pi\varepsilon^2$;
height is an empirical density illustration. The threshold plane is
$m/(n\pi\varepsilon^2)$, and observed points show their actual local heights
with DBSCAN labels/roles. **Observe:** peaks, valleys and a density threshold;
rotate to relate nearby dense regions. **Takeaway:** DBSCAN itself uses discrete
neighborhood counts and core connectivity; it does not calculate or optimize
this plotted continuous-looking surface. Grid interpolation, boundary bias
and self-counts are limitations of the illustration, not new model outputs.

## 09 — What if groups have different densities?

Artifact: `09_variable_density_limitation.png`.

**Concept:** a shared radius must serve unlike local densities. **Mechanism:**
hold one compact Gaussian population, one broader Gaussian population and
background fixed while changing epsilon. **Observe:** sparse coverage,
fragmentation and possible bridges change together. **Takeaway:** one setting
can work on a particular sample, but a global threshold has no guarantee of
recovering every population. OPTICS/HDBSCAN are possible follow-ups, not
guaranteed solutions; no optional framework is required or silently skipped.

## 10 — What changes when feature units change?

Artifact: `10_feature_scaling.png`.

**Concept:** distances depend on representation. **Mechanism:** compare the
same points in comparable synthetic units, with one axis multiplied by 30,
and after fitting StandardScaler on the distorted exploratory corpus.
**Observe:** stretched numeric distances can destroy neighborhoods even when
an auto-scaled plot appears similar; epsilon values belong to the displayed
feature space. **Takeaway:** standardization is a modeling decision, not a
universal correction. Freeze learned transforms for future evaluation. Plot
aspect ratios are annotated where equal-unit display would conceal structure.

## 11 — Why can high-dimensional distances lose contrast?

Supplementary artifact: `11_high_dimensional_distances.png`.

**Concept:** distance concentration in an isotropic noise control.
**Mechanism:** generate independent Gaussian coordinates in increasing
dimensions and inspect pair distances relative to $\sqrt{2D}$, plus the
nearest/farthest distance ratio. **Observe:** relative dispersion and contrast
change. **Takeaway:** a usable neighborhood scale requires justified features;
this control does not prove that real embeddings or all high-dimensional
distributions are unclusterable. No semantic clusters are planted here.

## Executed visual record — 2026-10-29

Codex generated all eleven scene selectors on Windows x64 using Python 3.11.14,
NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, Matplotlib 3.11.2,
Plotly 6.9.0 and Pillow 12.3.0. Numerical libraries used one thread.
The executable was the existing local
`.local/environments/s08-reference-2026-10-08/Scripts/python.exe`; portable
commands appear above. Real media were generated through individual scene
selectors; a separate orchestration test also checks `--all` dispatch.
Scene 5 was re-exported to move its footer away from slider labels, without
changing numerical configurations or observations; its initial report is retained
as ignored `visual_report_05_initial.json`.

This record is separate from the earlier [study experiment](notes.md#executed-experiment--2026-10-29).
Scenes 4–6 and 8 intentionally reuse that 485-point corpus. The expansion,
3D, unequal-density, scaling and high-dimensional scenes use different controls;
their counts must not be compared as if they were repeated measurements on
one dataset. All interpretations below are **candidates: author review required**.

| Hypothesis / question | Configuration | Observation | Interpretation candidate and limitation |
|---|---|---|---|
| A larger ball can cross the core threshold | 31 points; uniform box plus origin; seed 42; m=7; epsilon 0.12→1.25→0.12; 43 frames at 5 fps | Reference count ranges from 1 to 26 including self | Core status follows the threshold; one local neighborhood is not a clustering result. |
| Only core points propagate | 27 code-defined points on two short curves plus isolated points; eps=0.64, m=4 | 2 clusters, 19 core, 4 border, 4 noise; 68 exact states at 2 fps; points 0 and 12 change from noise to border | Core connectivity drives expansion. A small hand-built example does not establish robustness; border ambiguity is tested on a separate symmetric control. |
| Centroid and density rules can partition differently | 450 moons (noise=0.06) + 35 uniform background, seed 42; K=2/n_init=10 versus eps=0.18/m=7 | K-Means assigns all 485 points; DBSCAN finds 2 clusters and 16 noise | The curved geometry favors density connectivity. This is not an equal-budget model selection benchmark or anomaly validation. |
| Two density parameters interact | Same 485 points; eps=[0.08,0.12,0.18,0.25,0.35,0.50], m=[3,5,7,10,15] | Cluster counts range 0–16; for m=15, eps 0.08 gives 0 clusters and eps 0.12 gives 14; final browser control state eps=0.50/m=15 gives 1 cluster and 5 noise | Birth and merging make counts nonmonotone. The grid does not establish a useful or optimal partition. All 30 per-fit counts remain in the generated report. |
| Neighbor radii can suggest candidates | Same corpus; self-inclusive 7-distance; candidate radii 0.12/0.18/0.25 | Quantiles 0.5/0.9/0.95/0.99/1: 0.080603/0.131402/0.163476/0.517188/0.962487 | The upper tail changes slope; no elbow detection or optimality claim is made. |
| Curved connectivity extends to actual 3D | 2 tubes × 140 points; t in [-1.6,1.6], sinusoidal y/z; independent jitter std 0.045; 24 box-background points; seed 42; eps=0.26/m=7 | 304 3-feature rows; 2 clusters, 280 core, 4 border, 20 noise | Euclidean proximity works on this separated construction. A 3D rendering is still a projection and is not evidence about semantic embeddings. |
| Counts admit a density-height illustration | Same 485-point 2D corpus; eps=0.18/m=7; 65×75 query grid over x=[-1.6,2.6], y=[-1.1,1.6] | Observed-point counts 1–30, grid counts 0–30; illustrative threshold 0.141795 | Peaks and valleys relate to counts; interpolated grid height is not an estimated DBSCAN feature or optimization objective. |
| One radius can treat unequal densities differently | 120 points at (-0.65,0), std=0.055; 120 at (0.65,0), std=0.32; 15 background; seed 42; m=7; eps=0.08/0.18/0.36 | Clusters/noise: 3/119, 2/33, 2/10. Sparse-source assigned: 18/120, 99/120, 119/120. Background assigned: 0/15, 3/15, 6/15 | Coverage trades off against background absorption. The larger setting recovers two groups here: this sample does not prove that every global epsilon must fail. No HDBSCAN comparison was run. |
| Changing units changes neighbor counts | 300 Gaussian points at (-2,-1)/(2,1), std=0.27; seed 42; x multiplier 30; StandardScaler on the distorted corpus; eps=0.30/m=7 in each representation's own units | Comparable/distorted/standardized clusters/noise: 2/3, 4/268, 2/1 | The constructed scale distortion changes connectivity. No universal recommendation to standardize spatial data follows. |
| Independent noise dimensions reduce relative distance contrast | 180 standard-Gaussian rows at D=2/10/50/100; sequential default_rng(42) draws; pairwise Euclidean distances | CV: 0.532310/0.221631/0.098377/0.066434; median nearest/farthest: 0.033375/0.328792/0.621970/0.728493 | Relative contrast declines in this isotropic control. Pair distances are dependent, only one seed is used, and no real-embedding or clustering impossibility claim follows. |

## Generated artifact inventory

All **15 scene exports succeeded**. Paths below are relative
to this topic folder; backticked ignored paths are filenames, not broken public
Markdown links. `outputs/index.html` and the per-scene JSON reports also exist
and remain ignored. PNGs were decoded and inspected; the three GIFs were
decoded and representative exact frames were inspected, including role changes.

| Relative path | Status / purpose | Boundary |
|---|---|---|
| `outputs/01_epsilon_neighborhood.gif` | Generated; selected preview candidate; self-inclusive radius/count animation | One reference neighborhood |
| `outputs/02_core_border_noise.png` | Generated; role classification and two example neighborhoods | Completed-fit classification |
| `outputs/02_core_border_noise.gif` | Generated; sequential point/role inspection | Reveals final roles, not algorithm execution |
| `outputs/03_dbscan_expansion.gif` | Generated; selected preview candidate; exact queue trace | Small educational input; about 36 seconds including final hold |
| `outputs/03_dbscan_final.png` | Generated; final components and roles | Cluster IDs are arbitrary |
| `outputs/04_kmeans_vs_dbscan.png` | Generated; selected preview candidate; shared-axis comparison | Synthetic favorable moons, known K |
| `outputs/05_parameter_sensitivity.html` | Generated; offline radius slider, min_samples dropdown, role scatter and two heatmaps | 30 precomputed configurations, not live arbitrary fitting |
| `outputs/05_cluster_count_heatmap.png` | Generated; counts over the full parameter grid | Counts do not establish quality |
| `outputs/05_noise_fraction_heatmap.png` | Generated; abstention over the same grid | Noise is not anomaly truth |
| `outputs/06_k_distance_plot.png` | Generated; self-inclusive neighbor-radius curve and candidate thresholds | No optimal-epsilon claim |
| `outputs/07_dbscan_3d_interactive.html` | Generated; offline genuine 3-feature hover/rotation/zoom | WebGL; viewpoint can occlude points |
| `outputs/08_density_landscape_3d.html` | Generated; offline rotatable count-density surface | Height is illustrative; DBSCAN fits only x/y |
| `outputs/09_variable_density_limitation.png` | Generated; compact/broad populations under three radii | Wider radius works substantially better on this sample; no universal failure claim |
| `outputs/10_feature_scaling.png` | Generated; original, distorted and standardized feature spaces | Numeric epsilon units differ across panels |
| `outputs/11_high_dimensional_distances.png` | Generated supplement; Gaussian distance concentration | No semantic signal or embedding validation |

## Validation and preview policy

The focused suite passed **55 tests**, with no skips or warnings, using pytest
9.1.1. It covers the original core/noise semantics, trace snapshots, provisional
noise becoming border, non-propagation through border, label-invariant core
comparison and ambiguous borders, all 30 parameter frame counts, exact 2D
count/area heights, actual 3D coordinates/tooltips, scaling identities,
Pillow export, standalone HTML embedding and temporary CLI/gallery exports.
Tests write into temporary directories, not the selected previews.

```bash
python -m pytest -q 02-unsupervised-recommender-systems/37-dbscan-density-based-clustering/tests/test_dbscan.py 02-unsupervised-recommender-systems/37-dbscan-density-based-clustering/tests/test_visual_dbscan.py
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

Repository checks passed for **243 Python source files** and **734 internal
links in 172 Markdown files**; the selected environment's dependency check and
`git diff --check` also passed. Git emitted LF-to-CRLF normalization notices
for `.gitignore` and the module index. These focused checks are not the full
repository test/app suite, Linux CI or scientific interpretation review.

The three standalone HTML files were opened as local `file:` URLs in isolated
headless Chrome 156 with networking disabled. A visible epsilon slider click
and min_samples dropdown selection changed the initial 0.18/7 state (2 clusters,
16 noise) to 0.50/15 (1 cluster, 5 noise), retained all 485 points and moved both
heatmap selection markers to the correct cell. Both 3D scenes rendered WebGL
canvases using software rendering and responded to camera changes. The local
gallery loaded all nine image previews. No remote HTTP requests or JavaScript
exceptions were recorded in the successful runs. An initial browser connection
attempt timed out; the isolated rerun with the environment's working headless
launch flags succeeded. Browser screenshots/reports remain ignored. This is not
exhaustive hover, camera, accessibility, multi-browser or GitHub rendering review.

The inspected epsilon GIF, expansion GIF and K-Means comparison PNG are the
three public README **preview candidates**, deliberately unignored after checking
their Git candidate status. Together they are about **2.84 MiB**. HTML files
are roughly 4.7–6.2 MiB each because they embed Plotly.js; they stay ignored
with the other images, reports, gallery and inspection intermediates.
No files were committed, pushed or published. The root EN/PT READMEs were
preserved during this phase; only the topic and existing module index evidence
descriptions were reconciled. Interpretation candidates await author review.
