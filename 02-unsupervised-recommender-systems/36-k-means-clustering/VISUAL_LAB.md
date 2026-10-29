# K-Means visual lab

[Topic overview](README.md) · [Generator](visualize_kmeans.py) ·
[Exact numerical trace](kmeans_visual_core.py) · [Tests](tests/test_visual_kmeans.py)

Follow one connected argument: inspect exact Lloyd updates, question the feature
geometry, evaluate K, then challenge the result with geometry and initialization
controls. All data are synthetic. Feature names are illustrative numerical
proxies, not measurements from a business dataset. Generator labels are used
only for the scaling diagnostic, never as fitting targets.

## Install and explore

Use stable Python 3.11+ and the repository's canonical dependencies. No new
dependency file, web framework, server, GPU library or external service is needed.
Matplotlib/Pillow export the GIF; Plotly supplies the locally bundled 3D viewer.

From the repository root:

```bash
python -m pip install -e ".[dev]"
cd 02-unsupervised-recommender-systems/36-k-means-clustering
python visualize_kmeans.py --all --save --no-show
```

Open the generated `outputs/index.html` in a browser. It provides the teaching
sequence and links to all seven exports, including the standalone
`outputs/06_interactive_kmeans_3d.html`. No Python process needs to keep running.
The gallery uses relative local file links and requires no network connection.
GitHub does not render interactive HTML inline; the 3D file must be generated
and opened locally, or deliberately distributed as an offline artifact.

Generate a single scene, from the topic directory:

```bash
python visualize_kmeans.py --scene convergence --no-show
python visualize_kmeans.py --scene scaling --no-show
python visualize_kmeans.py --scene elbow --k-min 2 --k-max 10 --no-show
python visualize_kmeans.py --scene boundaries --no-show
python visualize_kmeans.py --scene limitations --no-show
python visualize_kmeans.py --scene 3d --no-show
python visualize_kmeans.py --scene initialization --no-show
```

Exports are always saved; `--save` makes that intention explicit. No arguments
exports the convergence GIF. Without `--no-show`, the exported file opens in
the default browser; `--all` opens the local gallery. `--no-show` selects Agg
and opens no windows. The default output path is beside the script, independent
of the working directory. `--output-dir PATH` overrides it; a relative override
is resolved from the caller's working directory. `--seed` defaults to 42.
The elbow scene accepts a valid inclusive K range with $2\leq K<360$;
keep the range small for readable plots and bounded runtime.

Each run also writes an ignored JSON report with configuration, versions,
measurements and artifact sizes. All-scene and single-scene reports have
different filenames. Repeating the same command replaces its outputs and report;
use a different `--output-dir` to preserve a run. Tests export into temporary
directories and do not regenerate the public previews.

## Exact states and shared conventions

An assignment state chooses the closest current center using squared Euclidean
distance. An update state holds those assignments fixed and moves every nonempty
center to its arithmetic member mean. Inertia is calculated from the displayed
labels and center positions, including the intermediate update state. Initial
points are grey, while their nearest-center objective is shown as a baseline.
The next assignment state can reduce inertia further.

No frame interpolates an optimization step. The animation pauses on exact states;
Pillow may combine identical hold frames into longer frame durations. Colors
follow fixed centroid IDs within a run. IDs and colors have no guaranteed
correspondence between independent fits. Ties select the lowest ID; empty centers
retain their previous coordinates. Fewer occupied clusters than K are therefore
possible. The trace stops on unchanged assignments or an 80-update cap, and
reports whether it stabilized. The 3D slider shows completed assignment states,
including iteration zero, with zero-duration transitions and explicit redraws.

## 01 — How do assignments and means interact?

Artifact: `01_kmeans_convergence.gif`.

**What to observe:** the initial centers, grey-to-colored first assignment,
alternating ASSIGN/UPDATE captions, centroid arrows, current inertia and final
unchanged-assignment state. In an update frame, a point may temporarily be
colored for a center that is no longer its nearest center: reassignment follows.

**Why it happens:** each block minimizes the squared objective with the other
block fixed. The mean is the minimizer for a nonempty cluster; exact steps cannot
increase the displayed objective. Trails show actual successive means.

**Production implication:** a stable partition is a termination condition, not
evidence of a global solution or business usefulness.

## 02 — What does feature scaling change?

Artifact: `02_feature_scaling_comparison.png`.

**What to observe:** raw and StandardScaler-fitted partitions on identical
original-coordinate axes. Standardized centers are inverse-transformed. The
inset breaks down squared distance for the same difference of 60 processing
minutes and one interaction proxy unit; it shows percentages, not comparable
absolute objectives.

**Why it happens:** standardization divides each squared difference by its
fitted marginal variance. The bar uses those exact fitted scales. Translation
does not alter distances; scaling one feature can change assignments.

**Production implication:** preprocessing defines feature weights. Preserve the
fitted scaler for inference and choose weights from the task, rather than
assuming standardization is always beneficial. Interaction counts are modeled
as continuous synthetic proxies, not literal event-count observations.

## 03 — What evidence supports a candidate K?

Artifact: `03_k_selection.png`.

**What to observe:** the inertia reductions, silhouette maximum within the
evaluated range, and cluster-size composition at the range endpoints and candidate.
Size segments are sorted within each fit and do not imply cross-fit identities.

**Why it happens:** additional centers can improve optimal compactness. Silhouette
compares mean within-cluster distance $a$ with the nearest other cluster's mean
distance $b$, using $(b-a)/\max(a,b)$. It uses Euclidean pairwise distances here,
whereas inertia uses squared point-to-center distances. Undefined silhouette
configurations return no score.

**Production implication:** inspect the representation, stability, profiles and
downstream utility. The marked K is a silhouette candidate; there is no automatic
elbow detector or guarantee of a true cluster count.

## 04 — Why are the regions convex?

Artifact: `04_decision_boundaries.png`.

**What to observe:** dense-grid nearest-center regions before and after moving
C0 by $(2,1.5)$, with the same coordinates and fixed center identities. The
right panel is a controlled edit, not another optimized fit.

**Why it happens:** equal squared distances to two centers define a linear
bisector; intersecting the half-spaces produces a convex Voronoi region. The
260-by-240 grid approximates those boundaries, so edges have slight raster steps.

**Production implication:** the model cannot learn arbitrary nonconvex assignment
regions. Moving a center can shift the assignment of points outside its previous
region. Sparse gaps can move substantially while few sampled points change IDs.

## 05 — Which distributions challenge the grouping rule?

Artifact: `05_kmeans_limitations.png`.

**What to observe:** five matched-axis rows: compact, elongated, moons,
unequal-density and distant-observation controls. Compare means on the left
with density connectivity on the right; grey DBSCAN observations are noise.
Some controls can work for both methods. Elongation alone does not guarantee
that K-Means fails on every dataset.

**Why it happens:** squared distances can allocate a center to distant records;
Voronoi cells split moons; a single DBSCAN radius can fragment a low-density
group while preserving a dense group. DBSCAN has no centroid markers.

**Production implication:** choose the grouping rule and parameters for the
distribution and use case. Per-dataset DBSCAN parameters here are illustrative
choices, not an equal-budget tuning study or an algorithm ranking.

## 06 — What changes in three dimensions?

Artifact: `06_interactive_kmeans_3d.html` (standalone, ignored by default).

**What to observe:** rotate, pan and zoom the three-feature cloud, hover over
observations and distinctive diamond centers, then move the iteration slider.
Coordinates are displayed in original feature units; the title reports the
iteration and inertia in standardized fitting space. The camera is retained
when changing frames. Legend entries distinguish observations from centers.

**Why it happens:** this is a genuine full-rank three-feature Gaussian generator,
not a 2D projection with an invented height. The same nearest-center/mean loop
operates in three dimensions. Actual standardized centers are inverse-transformed
only for display. The frame updates assignments, centers, title and metric together.

**Production implication:** intuitions about Euclidean geometry extend to more
features, but interpretable axes in a synthetic cloud do not validate the
semantics of high-dimensional embeddings. A browser supporting WebGL is required;
the verified headless run used software rendering without a GPU requirement.

## 07 — Can different starts settle differently?

Artifact: `07_initialization_comparison.png`.

**What to observe:** initial hollow squares and final X markers for random
sampling and scikit-learn's K-Means++ initializer, each with seeds 42, 43 and 44
at the default seed. Dashed lines show net displacement, not the optimization
trajectory. The corpus has four generating lobes but the optimizer receives K=3.

**Why it happens:** different starts can settle at different partitions of this
constrained quantization problem. Random starts use NumPy's default generator;
the plus-plus starts use scikit-learn's initializer. The final fits all use
those explicit centers, `n_init=1`, `tol=0` and Lloyd's algorithm. Ten-restart
fits are separate scikit-learn runs shown in the footer.

**Production implication:** an initialization method and a restart budget are
different choices. Do not infer a universal winner from six selected runs.
K-Means++ can lose to a random start on a particular corpus.

## Executed visual record — October 29, 2026

Codex executed all seven exports on Windows x64 with Python 3.11.14, NumPy 2.4.6,
scikit-learn 1.9.1, Matplotlib 3.11.2, Plotly 6.9.0 and Pillow 12.3.0. Numerical
libraries used one thread. This record is separate from the historical
[600-point study-script record](README.md#executed-study-record--october-29-2026):
the visual scenes intentionally use smaller, different controls. No cross-phase
metric comparison is implied. Interpretations below remain **pending author review**.

Common reference fits use `random_state=42`, K-Means++, `n_init=10`,
`max_iter=300`, `tol=0`, `algorithm="lloyd"`, except for the explicit single-start
comparisons. All configuration details also live in [the numerical helper](kmeans_visual_core.py).

| Scene and hypothesis | Configuration | Observed result | Interpretation candidate and limitation |
|---|---|---|---|
| Exact updates reduce the objective | 240 blobs, centers (-4,-1), (0,4), (4,-1), std 0.7; fixed starts (-4,-3), (-2,-3), (1,-3) | Three mean updates; inertia 5594.548146 → 223.830727; final assignments stable | The displayed loop decreases squared loss. One manually chosen start on favorable data is not a global convergence result. |
| Scaling changes assignments | 360 blobs at the same generating centers, stds 0.65/0.85/0.65; raw features ((latent y + 3) × 60, latent x + 6), K=3 | ARI vs generator: raw 0.502227; scaled 1.000000. Fixed raw delta: time contributes 99.9722%; standardized delta: time contributes 65.2830% | Geometry changes meaningfully on this deliberately distorted control. The generator and evaluation representation are favorable; no universal scaling advantage follows. |
| Additional K can improve inertia beyond a silhouette peak | 360 blobs, std 0.85, original synthetic coordinates; K=2…10 | J: 2849.468106 at K=2, 501.972169 at K=3, 197.032790 at K=10; silhouette peaks at K=3, 0.757555 | Three is a candidate in this sweep; internal criteria do not validate real-world segments. |
| Moving one center shifts regions | The 240-point corpus; fitted K=3; move C0 by (2,1.5); 260×240 grid | Regions changed and one observed point changed assignment | The region geometry responds directly to centers. A controlled edit does not establish model improvement. |
| Different grouping rules fail differently | 280 rows per case; anisotropic transform [[1,0.7],[-0.4,0.25]]; moons noise 0.06; unequal stds 0.16/0.65/1.3; outlier case adds 12 rows near (14,13), std 1.3 | DBSCAN groups/noise: compact 3/4; elongated 3/6; moons 2/0; unequal densities 5/31; distant observations 3/15 | The moons and outlier panels expose centroid-model constraints; unequal density also challenges DBSCAN. Illustrative radii and no downstream evaluation preclude an algorithm ranking. |
| Exact updates extend to 3D | 240 points; centers (2,3,1), (7,10,4), (11,4,8); stds 0.6/0.9/0.7; StandardScaler; three sampled starts | Slider iterations 0,1,2; standardized inertia 447.488843 → 69.903342 → 36.958607; converged | The slider represents actual multidimensional updates. Three human-readable axes are not a model of embedding semantics. |
| Starts can reach different local solutions | 400 points, four rectangle corners (±4,±2), std 0.55; K=3; initializer seeds 42/43/44 | Random: 1034.873986, 1034.873986, 1081.194119. Plus-plus: 1081.194119 for all three. Separate ten-restart fits both reached 1034.873986 | Restarting can help on this control; K-Means++ is not a per-run guarantee. Six runs do not estimate a general initializer ranking. |

An earlier setup check asked whether three separated blobs exposed local-solution
differences: 420 points with sizes 70/280/70 at (-5,0), (0,0), (5,0), std 0.7,
data seed 42, and the same six initializer seeds and restart controls. Every fit
reached inertia 394.631315. Interpretation candidate: this favorable control did
not expose differing objectives; a negative setup check does not establish that
initialization never matters. The final rectangle control was selected to make
different local outcomes visible. The two datasets are not competing benchmarks.
The final default configurations and observations above are the exported visual
experiment record.

## Publication and validation boundaries

Three inspected previews are deliberately unignored and linked from the topic
README: the convergence GIF, scaling PNG and K-selection PNG. They total about
0.7 MiB. The other four requested scene exports, local gallery, JSON reports,
browser screenshots and inspection intermediates remain ignored regenerable
artifacts. The self-contained 3D HTML is about 4.7 MiB because it embeds Plotly.js;
it should not be committed by default. No generated media has been committed.

Run the focused checks from the repository root:

```bash
python -m pytest -q 02-unsupervised-recommender-systems/36-k-means-clustering/tests/test_kmeans.py 02-unsupervised-recommender-systems/36-k-means-clustering/tests/test_visual_kmeans.py
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

The focused numerical/rendering tests cover independent inertia calculations,
exact phase contracts, mean updates, empty clusters, invalid inputs, shared
scaling, full-rank 3D data, frame contents, Pillow GIF export, embedded Plotly.js,
temporary CLI exports and gallery links. They do not constitute exhaustive
browser-interaction, accessibility or scientific interpretation review.

Executed checks for this phase: **56 focused tests passed with no skips or
warnings**; repository AST parsing passed for 237 Python files; internal links
passed for 704 targets in 168 Markdown files; dependency compatibility and
`git diff --check` passed. Git emitted its Windows LF-to-CRLF normalization
notices. These are scoped checks, not the full repository test/app suite.
All five PNGs were decoded, the GIF was inspected at exact states, and layout
issues found in the scaling, convergence and limitation exports were corrected.

The offline HTML was additionally opened as a local `file:` URL in isolated
headless Chrome 156 with networking disabled and software WebGL rendering. A
visible slider click moved iteration zero (inertia 447.489) to iteration two
(36.959), and a camera update rendered successfully; no remote HTTP requests
or JavaScript exceptions were recorded. Browser inspection exposed Plotly's
default autoplay, which is now explicitly disabled, and prompted layout fixes.
The local gallery also loaded all six image exports offline and resolved its
link to the standalone 3D file; its opening layout was inspected.
The final 3D re-export has its own `visual_report_3d.json`; the all-scene report
retains the earlier full export. The mathematical configurations and results
above are unchanged by those presentation fixes. Browser logs and screenshots
are ignored local verification evidence. Manual hover, every camera angle,
multiple browsers, GitHub rendering and accessibility were not exhaustively tested.

Optional outlier-influence and anomaly-distance animations are not implemented;
the static limitation controls already show outlier effects. Resampling stability,
large-data runtime comparisons, real-data profiles and downstream utility remain
unexecuted extensions. This lab optimizes teaching clarity with small dense data,
not production memory use or a comprehensive clustering evaluation.
