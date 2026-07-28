# Linear Algebra I: Vectors, Matrices, and Operations

## Overview

This module reviews the linear algebra operations that underpin Data Science, Machine Learning, and Applied AI. It connects numerical computation with geometry: vectors represent observations, embeddings, parameters, or directions, while matrices organize vectors and apply transformations.

The practical example uses small synthetic vectors to compare similarity metrics and demonstrate that transformation order matters. The first-principles script exposes the mechanics behind common NumPy operations. The Streamlit visual explorer turns the same ideas into interactive geometric experiments with Plotly charts and locally generated GIFs.

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
- Feature scaling and nearest-neighbor geometry
- Synthetic embedding retrieval and metric selection
- High-dimensional distance concentration

## Why It Matters

Linear algebra is the computational language of feature matrices, linear models, embedding retrieval, recommendation systems, neural-network layers, and Transformer attention. A senior practitioner must reason about both values and shapes, choose metrics that match the representation geometry, and recognize when scaling, sparsity, numerical conditioning, or approximate search changes system behavior.

## Files

- `notes.md`: intuition, theory, formulas, assumptions, applications, trade-offs, and common mistakes.
- `example.py`: NumPy example comparing retrieval metrics and matrix transformations on synthetic data.
- `visualizer.py`: interactive plots for vector addition, projection, distances, and matrix transformations.
- `from_scratch.py`: educational implementations of core vector and matrix operations without NumPy.
- `interview_questions.md`: senior-level conceptual, mathematical, practical, and system-design Q&A.
- `references.md`: authoritative documentation and further reading.
- `requirements.txt`: focused dependencies for this module.
- `visualizations/app.py`: Streamlit entry point for the complete visual explorer.
- `visualizations/components.py`: reusable Plotly figure builders.
- `visualizations/math_utils.py`: tested NumPy operations used by the interface.
- `visualizations/export_gifs.py`: deterministic generator for the local animation assets.
- `visualizations/assets/`: generated GIFs for addition, projection, and operation order.
- `tests/test_math_utils.py`: unit tests for vector, matrix, retrieval, and scaling logic.

## How to Run

From the repository root, install the shared dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

Alternatively, install only the dependencies for this topic:

```bash
python -m pip install -r 00-foundations/02-linear-algebra-vectors-matrices/requirements.txt
```

### Run the visual explorer

```bash
python -m streamlit run 00-foundations/02-linear-algebra-vectors-matrices/visualizations/app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`, and opens it in the default browser. The explorer contains five laboratories:

1. vector addition, subtraction, scalar multiplication, magnitude, and normalization;
2. dot product, angles, orthogonality, cosine similarity, and projection;
3. L1, L2, and infinity norms plus Manhattan and Euclidean distances;
4. scaling, rotation, reflection, shear, matrix multiplication, and composition order;
5. feature scaling, synthetic embedding retrieval, and high-dimensional distance concentration.

All charts are interactive: hover for coordinates, zoom into regions, and change values using the controls.

### Run the numerical examples

Run the NumPy example:

```bash
python 00-foundations/02-linear-algebra-vectors-matrices/example.py
```

Run the first-principles implementation:

```bash
python 00-foundations/02-linear-algebra-vectors-matrices/from_scratch.py
```

The original Matplotlib desktop visualizer remains available as a lightweight alternative:

```bash
python 00-foundations/02-linear-algebra-vectors-matrices/visualizer.py
```

Choose an operation in the terminal, then move the sliders in the Matplotlib window. Close the window to return to the menu. A specific view can also be opened directly; for example:

```bash
python 00-foundations/02-linear-algebra-vectors-matrices/visualizer.py --demo 4
```

### Run the tests

```bash
python -m unittest discover \
  -s 00-foundations/02-linear-algebra-vectors-matrices/tests \
  -p "test_*.py" \
  -v
```

In Windows PowerShell, the same command can be entered on one line.

### Regenerate the GIF assets

```bash
python 00-foundations/02-linear-algebra-vectors-matrices/visualizations/export_gifs.py
```

The scripts use small deterministic or seeded synthetic datasets defined in code. Their outputs illustrate mathematical behavior and are not benchmark results.

## Visual Explorer Architecture

```text
visualizations/
├── app.py              # Streamlit pages and learning narrative
├── components.py       # Plotly figures
├── math_utils.py       # reusable numerical operations
├── export_gifs.py      # reproducible animation generator
└── assets/
    ├── vector_addition.gif
    ├── vector_projection.gif
    └── transformation_order.gif

tests/
└── test_math_utils.py
```

The UI is intentionally separated from numerical logic. This makes the equations independently testable and prevents the visual layer from becoming the source of mathematical truth.

## Suggested Learning Path

- Predict the result before moving a control.
- Change one variable at a time and explain what stayed invariant.
- Find two nonzero orthogonal vectors by driving the dot product to zero.
- Compare normalized and non-normalized retrieval rankings.
- Reverse two matrix transformations and explain why the result changes.
- Increase dimensionality and observe relative distance concentration.

## Key Takeaways

- The dot product combines alignment and magnitude; cosine similarity removes magnitude.
- For unit-normalized vectors, dot product and cosine similarity are equivalent, and squared Euclidean distance is a monotonic transformation of them.
- Matrix multiplication is shape-dependent and generally not commutative because transformation order matters.
- Feature scaling can dominate distance-based algorithms, while inconsistent embedding normalization can change retrieval rankings.
- Dense, sparse, exact, approximate, and reduced-precision computation introduce different memory, latency, accuracy, and stability trade-offs.
- Solving a linear system directly is usually preferable to explicitly computing a matrix inverse.
