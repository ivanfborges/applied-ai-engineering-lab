# Eigenvectors, PCA, and SVD Visual Laboratory

## Objective

This laboratory turns the Day 4 linear-algebra theory into geometric,
reproducible experiments. Every visual answers a specific question: which
directions a matrix preserves, how covariance determines PCA axes, how
projection discards a residual, and why truncated SVD is an optimal linear
low-rank approximation.

All datasets and images are synthetic and generated locally with fixed seeds.
The lab requires no API, external dataset, internet connection, GPU, or
`ffmpeg`.

## Concepts Visualized

- invariant eigenvector directions and eigenvalue scaling;
- covariance ellipses and PCA directions;
- PCA as a rotation to an orthogonal coordinate system;
- orthogonal projection from 3D onto a two-component plane;
- explained variance and reconstruction error;
- SVD as \(V^\top\), then \(\Sigma\), then \(U\);
- truncated SVD and singular-value energy;
- equivalence among covariance eigendecomposition, direct SVD, and
  scikit-learn PCA;
- centering, feature scaling, and outlier sensitivity.

## Structure

```text
visual_lab/
├── __init__.py
├── datasets.py
├── math_utils.py
├── plotting.py
├── animations.py
├── generate_visuals.py
├── interactive_lab.py
├── test_visual_math.py
└── README.md
```

- `datasets.py` contains deterministic synthetic dataset and image factories.
- `math_utils.py` implements the core PCA and SVD mathematics explicitly.
- `plotting.py` builds static Matplotlib and standalone Plotly outputs.
- `animations.py` generates GIFs through Matplotlib's `PillowWriter`.
- `generate_visuals.py` is the cross-platform command-line generator.
- `interactive_lab.py` is the eight-tab Streamlit study environment.
- `test_visual_math.py` validates numerical identities and generated artifacts.

## Installation

Run the following commands from the Day 4 folder:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Generate the Assets

Generate everything:

```bash
python -m visual_lab.generate_visuals --all
```

Generate selected categories:

```bash
python -m visual_lab.generate_visuals --eigenvectors
python -m visual_lab.generate_visuals --pca
python -m visual_lab.generate_visuals --svd
python -m visual_lab.generate_visuals --pitfalls
```

The command creates output directories automatically, reports each generated
path, and returns a nonzero exit code if a task fails.

## Start the Interactive Application

```bash
streamlit run visual_lab/interactive_lab.py
```

The application recomputes all results locally. Controls include matrix
entries, sample count, noise, correlation, random seed, centering,
standardization, component count, SVD rank, scale mismatch, and outlier
intensity.

You can also launch the app from the repository root without changing
directories:

```bash
python -m streamlit run 00-foundations/04-eigenvalues-eigenvectors-pca-svd/visual_lab/interactive_lab.py
```

## Run the Tests

Generate the visual assets first, then run:

```bash
python -m pytest visual_lab/test_visual_math.py
```

Tests cover the complete SVD reconstruction, the PCA-SVD variance identity,
orthonormal components, diagonal score covariance, monotonic reconstruction
error, agreement with scikit-learn, and required output artifacts.

## Generated Visualizations

### 1. Eigenvectors and Eigenvalues

`outputs/animations/eigenvectors_transformation.gif` animates
\(A(t)=(1-t)I+tA\). Arbitrary vectors generally rotate, while eigenvectors stay
on invariant lines. The eigenvalues label their stretching factors.

`outputs/static/eigenvectors_transformation.png` is the final transformation.

**Observe:** direction preservation does not mean unchanged length. The example
uses a symmetric matrix, so its eigenvectors are real and orthogonal.

**Limitation:** linear interpolation from \(I\) to \(A\) is a teaching device;
it is not the only path through the space of transformations.

### 2. Covariance and Principal Components

`outputs/static/covariance_and_principal_components.png` overlays a covariance
ellipse and PCA axes on correlated 2D data. Arrow lengths are proportional to
\(\sqrt{\lambda_i}\).

**Observe:** PC1 follows the ellipse's long axis; PC2 is orthogonal and has a
smaller eigenvalue.

**Limitation:** a 2D Gaussian-like cloud is deliberately clean. Real data may
be multimodal, nonlinear, or heavy-tailed.

### 3. Interactive PCA Axes in 3D

The locally generated `outputs/interactive/pca_3d_axes.html` shows original
axes and all three PCA axes. Rotate, zoom, and hover without a running server.

**Observe:** PCA constructs a new orthogonal coordinate system through the
data mean.

**Limitation:** three dimensions make rotation visible but do not remove the
interpretation challenges present in higher dimensions.

### 4. Projection from 3D to 2D

The locally generated `outputs/interactive/pca_projection_3d_to_2d.html`
animates observations onto the PC1-PC2 plane. A representative subset of
projection paths avoids visual clutter.

`outputs/static/pca_projection_comparison.png` compares original 3D
coordinates with the resulting two PCA scores.

**Observe:** the discarded residual is perpendicular to the retained plane.

**Limitation:** minimizing reconstruction error does not guarantee preserving
labels or nearest neighbors.

### 5. Explained Variance and Reconstruction

`outputs/static/explained_variance_and_reconstruction.png` derives variance
ratios and reconstruction MSE from synthetic eight-feature data.

**Observe:** cumulative explained variance rises while reconstruction error
falls. The 90% marker demonstrates a heuristic component choice.

**Limitation:** explained variance measures input geometry, not downstream
quality.

### 6. SVD Geometry

`outputs/animations/svd_geometric_decomposition.gif` and
`outputs/static/svd_geometric_decomposition.png` show
\(V^\top \rightarrow \Sigma \rightarrow U\).

**Observe:** \(V^\top\) selects orthogonal input directions, \(\Sigma\) sets
their scale, and \(U\) positions them in the output space. The final result
equals direct multiplication by \(X\).

**Limitation:** a 2D circle makes the factorization interpretable but does not
show rectangular mappings between spaces of different dimension.

### 7. Low-Rank Approximation

`outputs/static/svd_low_rank_reconstruction.png` compares ranks 1, 2, 5, 10,
20, and full rank, alongside singular-value decay and error.

The locally generated `outputs/interactive/svd_rank_slider.html` provides a
standalone rank selector.

**Observe:** dominant singular triplets reconstruct broad structure before
fine detail. Energy increases and MSE decreases with rank.

**Limitation:** the displayed compression ratio counts dense values in
\(U_k\), \(\Sigma_k\), and \(V_k^\top\). It is not a file-format benchmark and
does not include metadata, quantization, or entropy coding.

### 8. PCA-SVD Equivalence

`outputs/static/pca_svd_equivalence.png` compares covariance eigenvalues,
SVD-derived variances, directions after sign alignment, and reconstructions.

**Observe:** for centered data,

\[
\lambda_i=\frac{\sigma_i^2}{n-1},\qquad
Z=X_cV=U\Sigma.
\]

**Limitation:** nearly tied eigenvalues can yield different individual bases
within the same valid subspace.

### 9. Common PCA Pitfalls

`outputs/static/pca_common_pitfalls.png` compares correct centering, omitted
centering, incompatible scales, and influential outliers.

**Observe:** preprocessing changes the covariance structure and therefore the
fitted directions.

**Limitation:** the figure diagnoses sensitivity; it does not prescribe
automatic standardization or outlier deletion.
