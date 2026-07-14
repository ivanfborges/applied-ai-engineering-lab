# Linear Algebra I: Vectors, Matrices, and Operations

## Overview

This module reviews the linear algebra operations that underpin Data Science, Machine Learning, and Applied AI. It connects numerical computation with geometry: vectors represent observations, embeddings, parameters, or directions, while matrices organize vectors and apply transformations.

The practical example uses small synthetic vectors to compare similarity metrics and demonstrate that transformation order matters. The first-principles script exposes the mechanics behind common NumPy operations.

## Concepts Covered

- Scalars, vectors, matrices, shapes, and transposition
- Vector addition and scalar multiplication
- Dot and outer products
- L1, L2, and infinity norms
- Euclidean and Manhattan distances
- Cosine similarity and L2 normalization
- Elementwise versus matrix multiplication
- Linear and affine transformations
- Dense and sparse representations
- Numerical stability and high-dimensional trade-offs

## Why It Matters

Linear algebra is the computational language of feature matrices, linear models, embedding retrieval, recommendation systems, neural-network layers, and Transformer attention. A senior practitioner must reason about both values and shapes, choose metrics that match the representation geometry, and recognize when scaling, sparsity, numerical conditioning, or approximate search changes system behavior.

## Files

- `notes.md`: intuition, theory, formulas, assumptions, applications, trade-offs, and common mistakes.
- `example.py`: NumPy example comparing retrieval metrics and matrix transformations on synthetic data.
- `visualizer.py`: interactive plots for vector addition, projection, distances, and matrix transformations.
- `from_scratch.py`: educational implementations of core vector and matrix operations without NumPy.
- `interview_questions.md`: senior-level conceptual, mathematical, practical, and system-design Q&A.
- `references.md`: authoritative documentation and further reading.

## How to Run

From the repository root, install the shared dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

Run the NumPy example:

```bash
python 00-foundations/02-linear-algebra-vectors-matrices/example.py
```

Run the first-principles implementation:

```bash
python 00-foundations/02-linear-algebra-vectors-matrices/from_scratch.py
```

Open the interactive visualizer:

```bash
python 00-foundations/02-linear-algebra-vectors-matrices/visualizer.py
```

Choose an operation in the terminal, then move the sliders in the Matplotlib window. Close the window to return to the menu. A specific view can also be opened directly; for example:

```bash
python 00-foundations/02-linear-algebra-vectors-matrices/visualizer.py --demo 4
```

The scripts use small synthetic vectors defined in code. Their outputs illustrate mathematical behavior and are not benchmark results.

## Key Takeaways

- The dot product combines alignment and magnitude; cosine similarity removes magnitude.
- For unit-normalized vectors, dot product and cosine similarity are equivalent, and squared Euclidean distance is a monotonic transformation of them.
- Matrix multiplication is shape-dependent and generally not commutative because transformation order matters.
- Feature scaling can dominate distance-based algorithms, while inconsistent embedding normalization can change retrieval rankings.
- Dense, sparse, exact, approximate, and reduced-precision computation introduce different memory, latency, accuracy, and stability trade-offs.
- Solving a linear system directly is usually preferable to explicitly computing a matrix inverse.
