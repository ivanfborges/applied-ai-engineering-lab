# Visual lab: inspect the merge, then read the hierarchy

The question for each scene is what mathematical criterion caused the visible
operation. All coordinates are synthetic and small enough to inspect. This is
an educational visualization project; it measures no model-performance or
runtime benchmark and establishes no real document taxonomy.

## Run locally

Use stable Python 3.11+ on Windows, Linux or macOS. Install from the repository
root using the canonical [dependency configuration](../../pyproject.toml):

```bash
python -m pip install -e ".[dev]"
python 02-unsupervised-recommender-systems/38-hierarchical-clustering/visualize_hierarchical.py --help
python 02-unsupervised-recommender-systems/38-hierarchical-clustering/visualize_hierarchical.py --all
python 02-unsupervised-recommender-systems/38-hierarchical-clustering/visualize_hierarchical.py --scene 3
python 02-unsupervised-recommender-systems/38-hierarchical-clustering/visualize_hierarchical.py --scene 1 --fast --output-dir 02-unsupervised-recommender-systems/38-hierarchical-clustering/outputs/fast
python -m pytest -q 02-unsupervised-recommender-systems/38-hierarchical-clustering/tests/test_visual_hierarchical.py
```

Open `outputs/index.html` next to the generator. The CLI creates this directory
and a local gallery automatically. Each HTML embeds Plotly JavaScript: no server,
network, account or external API is needed. GIFs use Matplotlib `PillowWriter`;
there is no FFmpeg, notebook or GPU requirement. Browser WebGL renders the 3D
scene; software rendering can support machines without dedicated graphics.

`--list` names all outputs. `--scene N` renders only one scene. `--fast` lowers
GIF/PNG resolution and increases GIF playback speed, retaining every merge and
cut state. It does not change the data, linkage or result. Existing complete
scenes are skipped, even if a different quality mode is requested. Use a distinct
`--output-dir` for fast inspection, or explicitly use `--force` after editing the
generator or to replace outputs. Missing members of multi-file scenes are filled
without replacing existing artifacts unless `--force` is used.

Every invocation writes a uniquely timestamped `visual_run_*.json` recording
versions, parameters/results for newly generated scenes, skipped scenes and
artifact sizes. New invocations preserve previous run records. Cached artifacts
must not be interpreted as having been regenerated under the current versions.

## Concept-to-scene map

| Scene | Regenerable output(s) | Question and interaction |
|---:|---|---|
| 1 | `01_agglomerative_process.gif` | How do 15 singleton groups become one? Every average-linkage merge has a selection frame and a union frame; rings, cross-pairs, heights and remaining counts explain the operation. |
| 2 | `02_linkage_methods.gif` | Why can four methods disagree? The same 11 coordinates appear in a synchronized 2×2 grid: nearest pair, farthest pair, all cross-pairs, or centroid/size-dependent Ward cost. |
| 3 | `03_dendrogram_construction.gif` | Where does a branch come from? A 12-point scatter and dendrogram share one average-linkage tree; one actual branch appears per completed merge. |
| 4 | `04_dendrogram_cuts.gif`, `04_dendrogram_explorer.html` | How does one tree support K=2–5? The cut and scatter stay synchronized. The slider also offers a separately labeled four-point average-linkage tie control at K=2 and K=3. |
| 5 | `05_chaining_effect.gif` | How can local links span distant groups? Three methods merge the same 16 deliberately arranged bridge/end points. Every merge is shown. |
| 6 | `06_feature_scaling.gif` | How do units change Euclidean geometry? Raw coordinates stay fixed on the left; the right follows nine linear coordinate states from raw to standardized, with identical row IDs and selected numerical pair distances. |
| 7 | `07_ward_variance.gif` | Why does Ward need more than centroid separation? Three actual active groups with sizes 2, 2 and 8 expose deviations, candidate costs, the selected merge and its distinct height scale. |
| 8 | `08_hierarchical_3d.html` | What changes across five granularities in three original features? Rotate, zoom/pan and choose K=2–6, reusing one Ward tree on 48 observations; no projection is used. |
| 9 | `09_embedding_distances.html` | What changes when length matters versus angle? Select raw Euclidean/cosine or unit Euclidean/cosine/squared Euclidean; inspect vectors, exact pair distances and average-linkage trees. |
| 10 | `10_complexity.png` | What does quadratic storage mean? Exact unique-pair counts and raw float64 byte calculations distinguish condensed storage from a full square matrix, without allocating either. |

## Read the geometry carefully

The seed is 42 wherever random data are generated. Scenes 1, 4 and 6 derive from
the same 15 synthetic blobs with centers `[[-2,-0.5],[0,2],[2,-0.3]]` and standard
deviation 0.36; scene 3 uses the first 12 of these rows. Scene 2 uses 11 selected
rows of the explicit 16-point bridge used by scene 5. A scene's scatter,
dendrogram and cuts reuse its fitted SciPy `linkage` matrix. Scene 6 intentionally
refits at each changed geometry; it is not pretending that different coordinate
systems share one fitted hierarchy.

Point numbers always identify original rows. A merged group inherits the color
of its minimum-index member, so changing numerical cluster-label permutations
does not change the visual grouping. Scene 2 compares selected *memberships*,
since node IDs from diverging trees can refer to different groups. Colors cannot
guarantee global semantic identity across different trees or datasets.

In scenes 1–3 and 5, every merge is shown, including singleton and final states.
Average-linkage cross-pair segments can overlap. In scene 2, synchronization is
by number of completed merges, not by a common distance threshold: heights have
different meanings across linkages. Ward displays both $\Delta SSE$ and
$h=\sqrt{2\Delta SSE}$, and its variance interpretation requires Euclidean
geometry. Scene 7 begins at three preformed groups that actually coexist in the
reference Ward tree; it illustrates a local candidate decision rather than
replaying its earlier within-group merges.

Scene 4 uses exact merge-order partitions and only draws a horizontal cut when a
strict interior height exists. For `[1,2,8,9]`, heights `[1,1,7]` skip directly
from four to two groups at height 1; the K=3 state has no cutting line. Its
membership follows SciPy's chosen tied merge order, which is not unique in
general. Branch positions come from SciPy leaf ordering and child midpoints;
rotating children preserves the hierarchy.

Scene 6 prints distance values because raw and standardized plots fit their
axes independently. Screen segment lengths across these panels are not directly
comparable. The intermediate coordinates are a pedagogical linear interpolation,
not a fitted preprocessing procedure for deployment. Standardization changes
feature weighting; it does not automatically improve meaningful segmentation.

Scene 9 has seven explicitly synthetic nonzero 2D vectors, not real embedding
outputs. Vector rays illustrate direction and norm. The heatmap rescales with
each selected dissimilarity; use the hover values for numerical comparisons.
For unit vectors:

$$
\|x-y\|_2^2=2\left(1-\frac{x^Ty}{\|x\|_2\|y\|_2}\right).
$$

Thus unit squared Euclidean and cosine distances are proportional. Ordinary
Euclidean takes a square root before averaging, which can change average-linkage
merge order. A pointwise monotonic relation does not generally commute with
linkage aggregation. Ward is never used with cosine here. Human-reviewed
taxonomy discovery could use such diagnostics in a RAG workflow, but retrieval
quality, representation stability and domain meaning need separate evidence.

Scene 10 uses $n(n-1)/2$ unordered pairs and eight bytes per float64 value.
Condensed storage omits the mirrored triangle and diagonal. A full square matrix
uses $8n^2$ bytes. GiB means $2^{30}$ bytes. These calculations exclude input
features, algorithm workspace and implementation overhead; they are not measured
RAM peaks or timing results. Sparse constraints and coarse-to-fine processing
change or approximate the original unconstrained problem.

## Executed visual phase — 2026-10-29

Executed by the AI assistant, separately from the historical 90-observation
terminal experiment in [notes.md](notes.md). The visual phase uses the smaller
scene-specific populations above. Counts, costs and comparisons below belong
to those controls, not to the earlier study dataset.

**Hypotheses:** the deliberate bridge can connect distant endpoints through
short single-linkage steps; changing feature units can change a fixed-K
partition; Ward's size weighting can favor a farther small-group pair over a
closer pair involving a large group; tied heights can prevent a horizontal
exact-K separator. These hypotheses concern mechanisms, not model quality.

**Configuration and environment:** the ten-scene generator was run with `--all`
at normal resolution. Only scenes 7 and 9 were regenerated afterward with
`--scene 7 --force` and `--scene 9 --force` to improve per-group SSE labels and
HTML title/control spacing. Windows x64; stable Python 3.11.14; NumPy 2.4.6;
SciPy 1.17.1; Matplotlib 3.11.2; Pillow 12.3.0; Plotly 6.9.0; scikit-learn
1.9.1; pytest 9.1.1. Random generators use seed 42. Native thread pools are
limited to one thread. Each regeneration retained its own timestamped JSON run
record; original run records were preserved.

**Generated artifacts:** all seven requested GIFs, all three requested explorer
HTMLs, the complexity PNG and a local gallery. GIF frame counts for scenes
1–7 were respectively 29, 21, 23, 6, 31, 9 and 6. All GIF frames decoded with
Pillow; the complexity PNG decoded successfully. Initial, middle and final
frames were extracted for inspection, with a contact sheet inspected across
all seven scenes and the PNG. Browser screenshots were inspected for all three
explorers, including changed controls and rotated 3D geometry.

**Observed controls:**

| Scene | Measured result | Interpretation candidate — author review required |
|---|---|---|
| 4 | Ward cuts K=2,3,4,5 gave sorted sizes `[5,10]`, `[5,5,5]`, `[2,3,5,5]`, `[2,2,3,3,5]`; the separate tie-control K=3 had no interior threshold. | One hierarchy supports several granularities; a merge-order partition need not have a horizontal separator. No optimal K is established. |
| 5 | Single/complete/average final heights were 1.000050, 8.800006 and 5.804006; at the explicit threshold 1.1 there were respectively 1, 6 and 6 groups. | The constructed short-link bridge spans both ends under single linkage. Different linkage heights encode different criteria and do not rank method quality. |
| 6 | Raw-versus-standardized K=3 partition ARI was 0.44. Distances for row pairs `(0,1)`, `(0,7)`, `(3,12)` changed from 62.155849, 251.823043, 14.487945 to 0.616977, 2.842724, 2.376313. | Feature weighting changes pair geometry and the hierarchy. ARI here measures partition disagreement, not an improvement from scaling. |
| 7 | Candidate SSE increases were 5.29, 6.40 and 29.584. The selected 2+2 group merge had centroid distance 2.3; a competing 2+8 merge had shorter centroid distance 2.0 but greater cost 6.4. Selected SciPy height was 3.252691, satisfying $h^2/2=5.29$. | Ward's size weighting matters; nearest centroids alone do not identify the chosen merge. |
| 8 | Five cuts of one Ward tree retained all 48 observations in three genuine input dimensions. | Rotating the same coordinates can expose occlusion; it adds no additional clustering evidence. |
| 9 | Raw Euclidean-versus-cosine K=3 partition ARI was -0.258065. Unit squared-Euclidean heights agreed with twice the unit-cosine heights to maximum absolute error $3.51\times10^{-16}$. | Magnitude and direction can yield different groups; the proportionality identity holds numerically. Negative ARI here denotes disagreement, not failed semantic prediction. |
| 10 | At n=50,000: 1,249,975,000 unique pairs; 9,999,800,000 condensed bytes versus 20,000,000,000 full-square bytes. | Quadratic pair storage is a concern before runtime is considered. These are exact arithmetic calculations, not memory measurements. |

**Software and browser checks:** the focused topic suite passed 53 tests,
without skips or warnings. The visual tests include hand-computed merges,
actual state coverage/nesting, SciPy branch geometry, tie/zero-height behavior,
Ward's SSE identity, deterministic bridge and scaling controls, synthetic-vector
identities, three-feature input rank, temporary GIF/HTML/PNG export and CLI cache
preservation. Rendering tests write only to temporary directories.

Repository checks also passed: AST parsing for 248 Python files, 760 local links
across 176 Markdown files, dependency compatibility through `pip --python`, and
`git diff --check` plus whitespace checks on the new text files. A second
`--all` invocation skipped all ten existing scenes; hashes confirmed all 11
scene artifacts were unchanged. A separate `--scene 1 --fast` CLI run produced
29 frames at 715×487 pixels, preserving all 14 merges.

All three standalone HTML explorers were loaded through `file://` in headless
Chrome 156 with networking disabled. All 6 cut states, 5 3D states and 5
embedding states were exercised, including real mouse slider/dropdown actions
and a 3D mouse rotation. Final checks reported zero JavaScript exceptions and
zero HTTP(S) requests. An initial browser-helper assertion was corrected to
synchronize slider active state after programmatic updates; it was a verification
harness issue. Final screenshots, browser records and inspection intermediates
stay ignored. Headless desktop checks do not certify every browser, phone layout,
display environment or GitHub's rendered appearance.

**Preview policy:** three inspected GIF candidates are deliberately unignored:
[four linkage criteria](outputs/02_linkage_methods.gif),
[incremental dendrogram](outputs/03_dendrogram_construction.gif) and
[Ward's size/variance decision](outputs/07_ward_variance.gif). They were verified
as public Git candidates before adding links. Other artifacts remain ignored.
The root EN/PT READMEs were preserved as requested; the topic README and existing
module-index evidence row describe the completed visual implementation. Review
of scientific interpretations remains pending regardless of software validation.

**Limitations:** one fixed seed and a few deliberately controlled, low-dimensional
geometries; no perturbation/stability study, real document embeddings, deployment
or performance benchmark. Feature interpolation is illustrative. GIF playback
and compressed colors can vary by viewer. Plotly HTMLs are roughly 5 MB each
because the JavaScript is embedded; they are regenerable local artifacts.

## Interview use and remaining limits

Explain a selected merge before discussing its resulting partition. Then trace
the corresponding branch and change the cut. Ask whether the metric captures
the intended relationship and whether the hierarchy adds value over K-Means or
DBSCAN. Agglomerative merges are irreversible; all fitted observations are
assigned, and new-observation assignment requires an additional policy.

These are small controlled geometries, not evidence of real-world stability,
semantic correctness, a reliable anomaly policy or scalable deployment. A
dendrogram is not a probability model. Feature relevance, ties, perturbation,
embedding versions, human labels and downstream utility remain consequential.
All visual interpretation candidates require author review.
