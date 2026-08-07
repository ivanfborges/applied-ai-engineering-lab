# Linear Algebra II: Vector Spaces, Bases, and Projections

## Overview

This module develops the geometric structure behind feature vectors, model parameters, and embeddings. It explains how span and linear independence define a basis, why dimension counts independent directions rather than stored coordinates, and how orthogonal projection finds the closest vector in a subspace.

The practical scripts project a vector onto a non-orthogonal basis, verify the defining properties of a projection matrix, and construct an orthonormal basis with Gram-Schmidt. All inputs are small synthetic arrays created for education; no benchmark or public dataset is used.

## Concepts Covered

- Vector spaces and subspaces
- Linear combinations and span
- Linear independence, basis, dimension, and rank
- Ambient versus intrinsic dimension
- Coordinates and change of basis
- Orthogonal and orthonormal bases
- Gram-Schmidt orthogonalization
- Projection onto a vector or subspace
- Projection matrices and orthogonal residuals
- Connections to least squares, PCA, and embeddings

## Why It Matters

These concepts explain why redundant features cause rank and conditioning problems, why ordinary least squares is a projection, and why PCA can compress data. They also clarify what an embedding vector is: a point in a learned coordinate space whose distances, angles, or inner products are useful because of the model's training objective.

In production semantic search, query and document vectors must use compatible models, preprocessing, normalization, and transformations. A projection can reduce vector-index cost, but it can also change neighborhoods and remove task-relevant information, so retrieval and downstream quality must be evaluated.

## Files

- `notes.md`: intuition, theory, formulas, assumptions, trade-offs, applications, limitations, and common mistakes.
- `example.py`: projection with NumPy least squares and QR decomposition, including numerical checks.
- `from_scratch.py`: educational classical Gram-Schmidt and subspace projection implementation.
- `visualizations/`: static, animated, and interactive geometric laboratory for the core concepts.
- `interview_questions.md`: senior-level conceptual, mathematical, practical, and production Q&A.
- `references.md`: books, course material, and official numerical-computing documentation.

## How to Run

From the repository root, install the shared dependencies if needed:

```bash
python -m pip install -e ".[dev]"
```

Run the NumPy example:

```bash
python 00-foundations/03-vector-spaces-bases-projections/example.py
```

Run the first-principles implementation:

```bash
python 00-foundations/03-vector-spaces-bases-projections/from_scratch.py
```

Generate the complete visualization laboratory:

```bash
cd 00-foundations/03-vector-spaces-bases-projections/visualizations
python visual_lab.py
python generate_animations.py
python interactive_3d.py
```

See `visualizations/README.md` for installation, individual-output CLI options, and the generated asset guide.

The core examples require only NumPy. The visualization laboratory uses its
own minimal requirements file; none of the scripts needs external data,
credentials, or network access at runtime.

## Key Takeaways

- A basis is a linearly independent spanning set, and coordinates are defined relative to that basis.
- Dimension counts independent directions; ambient coordinate count can exceed a dataset's intrinsic dimension.
- Orthogonal projection minimizes Euclidean distance to a subspace, and its residual is perpendicular to that subspace.
- For an orthonormal basis `Q`, the projection matrix is `Q @ Q.T`; it is symmetric and idempotent.
- Least-squares fitted values are projections onto a design matrix's column space, while PCA projects centered data onto a variance-maximizing subspace.
- Embedding geometry is learned and model-specific; equal vector dimensions do not imply compatible representation spaces.
- Compression decisions should be judged with downstream metrics, not explained variance alone.
