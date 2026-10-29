# See PCA working

An executable visual laboratory for understanding variance, basis changes,
projection and reconstruction. Start with the geometry GIF, then rotate the
3D cloud yourself. The figures expose what PCA retains and what its variance
objective cannot establish about a downstream task.

## Run locally

From the repository root with a stable Python 3.11+ environment:

```powershell
python -m pip install -e ".[dev]"
$topic = "02-unsupervised-recommender-systems/39-pca-in-practice"
python "$topic/visualize_pca.py" --list
python "$topic/visualize_pca.py" --all
Start-Process "$topic/outputs/index.html"
```

The installation uses the existing [canonical dependencies](../../pyproject.toml).
No extra packages, credentials, API, server or dashboard framework are required.
Plotly's JavaScript is embedded in the HTML; opening it needs no internet.

Each demonstration also runs independently:

```powershell
python "$topic/visualize_pca.py" --demo geometry
python "$topic/visualize_pca.py" --demo 3d
python "$topic/visualize_pca.py" --demo explained-variance
python "$topic/visualize_pca.py" --demo reconstruction
python "$topic/visualize_pca.py" --demo scaling
python "$topic/visualize_pca.py" --demo predictive-information
python "$topic/visualize_pca.py" --demo outliers
python "$topic/visualize_pca.py" --demo loadings
```

Exports default to `outputs/` beside the script, independently of the working
directory. Complete existing demos are skipped. Use `--force` to explicitly
regenerate them; timestamped run records are preserved. For quicker study:

```powershell
python "$topic/visualize_pca.py" --demo geometry --fast --output-dir "$topic/outputs/fast"
python "$topic/visualize_pca.py" --demo scaling --show
python -m pytest -q "$topic/tests/test_pca_visuals.py"
```

`--fast` lowers image resolution and the number of interpolation frames; it
preserves datasets and exact PCA endpoint states. `--show` additionally opens
Matplotlib figures after export, when an interactive desktop backend exists.
Without it, the CLI uses Agg and opens no windows. Open the HTML yourself;
its controls work independently of every GIF. To inspect a different projection,
drag the cloud, scroll to zoom, hover over observations, or use the k slider.
The legend can hide traces. Camera orientation persists while k changes.

## Eight questions, eight demonstrations

| CLI demo | Export names in `outputs/` | What to inspect |
|---|---|---|
| `geometry` | `01_pca_geometry.gif`, `.png` | Covariance ellipse, mean, rotating basis, variance by direction and projection onto PC1 |
| `3d` | `03_pca_3d_interactive.html` | Exact k=1/2/3 reconstruction in original 3D units, vectors, PC1–PC2 plane and orthogonal residuals |
| `explained-variance` | `02_explained_variance.gif`, `.png` | Individual and cumulative calculated EVR, >95% crossing and selected count |
| `reconstruction` | `04_pca_reconstruction.gif`, `.png` | Original versus reconstructed points after inverse scaling, residual segments and MSE |
| `scaling` | `05_scaling_comparison.png` | Identical rows in raw versus standardized coordinates, PC directions and EVR |
| `predictive-information` | `06_variance_vs_predictive_information.png` | Held-out labels in original/PC coordinates and actual classification scores |
| `outliers` | `07_outlier_sensitivity.gif`, `.png` | Four real additions, refitted means, changed directions and variance ratios |
| `loadings` | `08_component_loadings.png` | Signed Wine direction coefficients, feature names and zero-centered color scale |

The first PNG captures the projection endpoint. The reconstruction PNG captures
the nonzero-error k=2 state, so residuals remain visible. Scree and outlier GIFs
advance through exact discrete states; repeated hold frames may be combined by
Pillow. Geometry and reconstruction include smooth intermediate motion.

## Rotation, projection and reconstruction

In the geometry scene, the original scatter **stays fixed during the basis
change**. The middle panel shows changing coordinates of the same observations
in that rotating basis. PCA fitting itself does not physically rotate samples.
The arbitrary starting basis is then oriented to the fitted principal axes;
this animation illustrates the result, not the internal SVD algorithm.

For unit direction $v$, projected training variance is $v^\top S v$. The curve
in the third panel is calculated from the data. Its maximum is the leading
covariance eigenvector. Perpendicular PC2 has the remaining eigenvalue. The
ellipse has radii $2\sqrt{\lambda_1}$ and $2\sqrt{\lambda_2}$; it is a descriptive
two-standard-deviation contour, not an asserted confidence or coverage region.

With directions stored as columns of $V_k$, scores and reconstruction are:

$$
Z=(X-\mathbf{1}\mu^\top)V_k,\qquad
\widehat X=ZV_k^\top+\mathbf{1}\mu^\top.
$$

Projection discards coordinates. The mean plus one direction is an affine line;
the mean plus PC1 and PC2 spans an affine plane. In the 3D explorer, orange
diamonds are exact reconstructions, and dashed segments are orthogonal
residuals. The translucent plane appears at k=2. Vector display lengths are
two standard deviations along each PC. Equal 3D units and `aspectmode="data"`
preserve the geometry; perspective still changes apparent screen angles.

The separate reconstruction GIF fits PCA **after standardization** and reverses
the scaler before displaying points in original space. Orthogonality then holds
in standardized space, and its displayed residuals need not be perpendicular
in the original metric. It reports both standardized and original-unit MSE.
Fractional frames interpolate between exact integer-rank reconstructions; they
are labeled display interpolation, not fitted PCA with fractional dimensions.

The two 3D demonstrations use the same underlying rows but **different metrics**:
raw covariance for the interactive explorer, standardized covariance for the
reconstruction GIF. Their eigenvalues, bases and losses must not be conflated.

## Reading the other comparisons

EVR measures retained **training input variance**. The scree scene computes it
from the full fitted model and applies sklearn's strict `>0.95` convention.
It does not use class labels or certify a useful prediction policy.

The scale comparison uses equal display units within each panel. Consequently,
the raw cloud is almost horizontal: its features have radically different
numerical scales. The right panel shows standardized coordinates of those same
rows, rather than raw points with standardized arrows superimposed. Arrow
lengths are equal for visibility in this comparison; EVR is stated numerically.
Standardization is a choice about the objective's metric, not a universal rule.

The predictive example deliberately makes a small-variance second feature
determine the label while a large-variance nuisance dominates PC1. PCA fits raw
training rows; both original features and retained scores are scaled before
logistic regression. Test data never fit these transformations. Displayed
scatter axes and PC-score panels use different numerical scales to reveal the
small direction; do not read physical angles from that scatter. Class rows
in the one-dimensional panels are display offsets, not extra features.

Outlier frames refit actual datasets with zero through four added observations;
no eigenvectors are interpolated between fits. The angle change is unsigned
and sign-invariant. Explained variance need not change monotonically as
outliers arrive. This designed demonstration does not quantify typical risk.

The Wine heatmap uses `components_[j, i]`, **direction coefficients**. These are
neither $v_{ij}\sqrt{\lambda_j}$ loadings nor feature-score correlations. For
nonconstant features, the latter additionally depend on feature standard
deviations. The largest absolute weight is made positive purely for display;
flipping a component and its scores leaves its reconstruction unchanged.
Individual directions can be unstable near eigenvalue ties. No chemical,
causal or business-factor interpretation is validated by a heatmap.

## What a Senior ML Engineer Should Understand

- PCA optimizes variance preservation and linear squared-error reconstruction,
  without optimizing classification accuracy or semantic relevance.
- Scale, outliers and the fitted population affect principal directions.
  Uncorrelated training scores do not imply statistical independence.
- More retained dimensions reduce reconstruction loss for nested orthogonal
  projections; that does not guarantee better downstream prediction.
- A smaller vector can reduce storage or computation, but retrieval quality,
  latency and resource trade-offs must be measured. This lab measures no such gain.
- Inference must preserve feature order, fitted centering/scaling and the
  component version. Changing the basis requires consistent downstream artifacts.

## Output and publication policy

Three inspected preview candidates are intentionally unignored and referenced
in the [topic README](README.md): geometry GIF, explained-variance PNG and
predictive-information PNG. Their interpretations still require author review.
The other exports, gallery, timestamped JSON measurements, browser screenshots
and inspection contact sheets remain ignored and regenerate locally.

`pca_visual_core.py` contains data and numerical states;
`pca_visualizations.py` renders them; `visualize_pca.py` selects demos and writes
independent timestamped run records. Rendering tests use temporary directories.
Do not overwrite reviewed previews casually; use a different `--output-dir`
for experiments or deliberately pass `--force` when updating an asset.

The optional camera-orbit GIF, animated scaling transition and biplot are not
implemented. The standalone 3D explorer provides mouse rotation, while the
scaling figure and coefficient heatmap provide simpler inspectable comparisons.

## Visual-phase experiments — 2026-10-29

These are **new visual-phase runs**, not replacements for the [study experiments](notes.md#executed-experiments--2026-10-29).
All eight demos were executed at normal resolution on Windows with Python
3.11.14, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, Matplotlib 3.11.2,
Plotly 6.9.0 and Pillow 12.3.0. Library computation was limited to one thread.
Timestamped UTC JSON records distinguish initial exports from presentation
reruns. Geometry was rerun to fix a missing-font symbol; 3D, reconstruction and
outliers were rerun for camera/layout presentation. Numerical results were unchanged.

**Author review required for every interpretation candidate below.** All data
are synthetic except the Wine heatmap; its real public source and license are
documented in [references.md](references.md). Exact generator arrays and
transforms are inspectable in [pca_visual_core.py](pca_visual_core.py).

| Scene / hypothesis | Configuration | Measured result | Interpretation candidate and limitation |
|---|---|---|---|
| Geometry: the maximum directional variance matches the leading eigenvalue | Seed 42; 100 Gaussian 2D rows; latent SDs 2.5/0.65; basis angle 33°; offset [2,1]; raw PCA | Eigenvalues 5.211923 / 0.300466; fitted PC1 angle 35.3291° | Sample geometry follows its covariance. One small designed Gaussian population, not a universal orientation. |
| 3D: an approximate plane loses little variance at k=2 | Seed 43; 100 Gaussian 3D rows; latent SDs 3/1.5/0.22, QR-orthogonal mixing and offset [2,-1,1]; raw PCA | EVR k=1/2: 0.76579460 / 0.99617475; MSE 0.86621409 / 0.01414776; k=3 MSE below 1e-28 | The designed cloud is nearly planar. No label, denoising or generalization claim. |
| Spectrum: correlated features admit a compact variance representation | Seed 44; 180 rows, 6 features; 3 Gaussian latent signals with SDs 3/1.4/0.7, fixed mixing plus SD 0.18 noise; raw PCA | EVR PC1/PC2 0.83953408 / 0.14534560; k=2 retains 0.98487969 | Two PCs cross 95% here. The generator deliberately makes features redundant; no prediction is evaluated. |
| Reconstruction: adding directions lowers standardized loss | Same seed-43 rows, fitted StandardScaler and full PCA, inverse scaling for display | Standardized MSE k=1/2/3: 0.19188274 / 0.00584752 / below 1e-28; original-unit MSE 0.90556985 / 0.01673363 / below 1e-28 | Nested projections reduce loss in this fitted metric. These are in-sample losses, not the raw-PCA explorer's objective or test errors. |
| Scaling: changing units changes the dominant direction | Seed 45; 130 rows; independent standard normal a,b; features [100a, 0.6a+0.8b]; same rows in both fits | PC1 EVR raw 0.99992792 vs standardized 0.79567069; directions approximately [0.999981,0.006226] vs [0.707107,0.707107] | Scaling changes the geometric question. This does not establish that equal feature weight is preferable for another application. |
| Prediction: nearly all input variance can omit useful label information | Seed 46; 500 rows; nuisance SD20, signal SD0.12; label signal>0; 70/30 stratified split seed46, 350/150 rows; raw PCA, training-only scalers; LR C=1, lbfgs, max_iter=2000 | PC1 EVR 0.99996715; held-out balanced accuracy original 1.00000000, k=1 0.44612376, k=2 1.00000000 | This constructed failure separates variance from task quality. One split, noiseless threshold labels and no tuning; the below-chance value is not a general PCA behavior. |
| Outliers: extreme additions change means and directions | Seed-42 clean geometry, then [3,18], [2,22], [1,25], [-1,28]; refit raw PCA at each actual addition | Mean changes [1.9854,0.9548]→[1.9571,1.8123]; PC1 unsigned change 50.079°; EVR 0.94549263→0.86026869 (nonmonotone path) | A few intentionally extreme rows can dominate covariance. No robust estimator or realistic contamination-rate comparison. |
| Weights: coefficients describe combinations in a declared input metric | Real Wine; training 133 rows from stratified 75/25 split seed42; StandardScaler and full PCA; PCs1–3, 13 named features | PC1 top absolute weight: flavanoids +0.427807; EVR PCs1–3 0.35749925 / 0.19209251 / 0.10845610 | Direction weights expose a fitted linear combination. No causal naming, chemical validation or cross-sample stability assessment. |

The topic suite passed **51 tests** (30 original study tests and 21 visual tests),
without skips or warnings. Tests verify orthonormality, full reconstruction, affine-plane projection,
covariance-ellipse geometry, variance-tail loss, monotone loss, inverse scaling,
training boundaries, sign invariance and strict-threshold selection. Browser
verification disables networking and checks all k states, recalculates MSE from
displayed points, clicks the slider, drags the camera and uses wheel zoom.
GIFs were decoded frame by frame, and PNGs plus representative animation frames
were visually inspected. Matplotlib desktop `--show` was not exercised in the
headless validation. Synthetic examples and screenshots do not certify scientific
interpretations on the author's behalf.

After author review, a root README evidence update could mention the eight-demo
PCA visual lab, offline k-selectable 3D explorer and tested geometric states.
No topic counts change: Day 39 remains the latest study and this module stays
at 4/9. Root READMEs are intentionally left unchanged by this visual phase.

Implementation references: [Plotly offline HTML export](https://plotly.com/python/interactive-html-export/),
[3D scatter](https://plotly.com/python/3d-scatter-plots/),
[3D surfaces](https://plotly.com/python/3d-surface-plots/) and
[Matplotlib PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html).
