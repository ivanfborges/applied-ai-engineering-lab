# Vector Spaces, Bases, and Projections — Visual Laboratory

## Project Objective

This executable laboratory turns the geometry of vector spaces into static figures, animations, and interactive 3D views. It is designed for interview revision by experienced Data Scientists and AI Engineers: each output connects a mathematical invariant to an observable geometric result and uses deterministic synthetic data rather than external services.

## Concepts Visualized

- Linear combinations, span, independence, basis, rank, and dimension
- Coordinates, change of basis, orthogonal and non-orthogonal bases
- Gram-Schmidt orthogonalization
- Projection onto a direction or subspace
- Residual orthogonality and projection-matrix invariants
- Ambient versus intrinsic dimension
- PCA as a lower-dimensional projection
- Synthetic embeddings, cosine similarity, and Euclidean distance

## Directory Structure

```text
visualizations/
├── visual_lab.py
├── generate_animations.py
├── interactive_3d.py
├── visualization_utils.py
├── README.md
└── outputs/
    ├── images/
    ├── animations/
    └── interactive/
```

The scripts create output directories automatically. Generated assets are self-contained and require no network access.

## Installation

Python 3.11 or newer is recommended.

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

From the repository root, install the shared dependencies:

```bash
python -m pip install -e ".[dev]"
```

## Generate Static Figures

Generate all 12 high-resolution PNG figures:

```bash
python visual_lab.py
```

Choose a destination or one figure:

```bash
python visual_lab.py --output-dir outputs/images
python visual_lab.py --only projection-vector
python visual_lab.py --list
```

## Generate GIF Animations

Generate all five animations with Pillow, without FFmpeg or ImageMagick:

```bash
python generate_animations.py
```

Adjust playback or generate one animation:

```bash
python generate_animations.py --fps 20
python generate_animations.py --only gram-schmidt
python generate_animations.py --list
```

## Generate Interactive HTML

Generate three self-contained Plotly pages:

```bash
python interactive_3d.py
```

Generate only the embedding view:

```bash
python interactive_3d.py --only embedding-space
python interactive_3d.py --list
```

Open any generated HTML file directly in a browser. No Python server is required.

## Output Guide

### Static images

- `01_vectors_linear_combinations.png`: coefficient changes explore a span.
- `02_span_one_vs_two_vectors.png`: a line subspace compared with the full plane.
- `03_independence_vs_dependence.png`: independent and redundant directions with rank.
- `04_different_bases_same_space.png`: one vector, two coordinate systems.
- `05_orthogonal_vs_nonorthogonal.png`: basis angle and dot-product comparison.
- `06_projection_onto_vector.png`: projection, residual, and right-angle condition.
- `07_projection_onto_plane.png`: a 3D subspace projection.
- `08_orthogonal_decomposition.png`: head-to-tail decomposition and Pythagorean identity.
- `09_ambient_vs_intrinsic_dimension.png`: noisy 3D data concentrated around a plane.
- `10a_pca_projection_3d.png`: principal plane and projected observations.
- `10b_pca_coordinates_2d.png`: the same observations in PCA coordinates.
- `11_synthetic_embedding_space.png`: controlled high-dimensional clusters shown with PCA.
- `12_cosine_vs_euclidean.png`: metrics that rank candidates differently.

Visualization 10 intentionally produces two figures, so the static generator creates 13 PNG files for 12 concepts.

### Animations

- `linear_combinations_span.gif`
- `projection_onto_vector.gif`
- `gram_schmidt.gif`
- `projection_onto_plane.gif`
- `pca_projection.gif`

### Interactive pages

- `projection_plane_3d.html`
- `pca_subspace_3d.html`
- `synthetic_embedding_space_3d.html`

## Applied AI Connections

Embedding models place text, images, users, or products in learned vector spaces. Semantic search and RAG compare those vectors with a metric selected by the model and index configuration. The embedding figures demonstrate neighborhoods and metric behavior without implying that PCA coordinates are semantic features.

PCA can reduce storage and compute, but it optimizes retained variance rather than retrieval relevance. A production compression decision should be validated with retrieval metrics and downstream answer quality. Query and document vectors must use the same compatible embedding model, normalization, and projection version.

## Numerical and Interpretive Limitations

- All datasets are synthetic and are not benchmark results.
- Floating-point zeros are approximate; checks use explicit tolerances.
- Classical Gram-Schmidt can lose orthogonality for nearly dependent inputs.
- PCA models linear variance and can miss nonlinear or low-variance task signal.
- PCA plots distort some high-dimensional distances and are explanatory views, not proof of semantic quality.
- Matplotlib's 3D projection can visually distort angles; the algebraic assertions remain authoritative.
- GIF file size and generation time increase with frame count, resolution, and FPS.

The from-scratch routines are educational. Production numerical code should rely on tested QR-, SVD-, or least-squares implementations, monitor conditioning, and validate behavior on representative data.
