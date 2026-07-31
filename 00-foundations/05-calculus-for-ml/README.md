# Calculus for Machine Learning

## Overview

Derivatives measure local sensitivity: how much an output changes when an input
changes slightly. Machine learning extends this idea from one variable to many
parameters. Partial derivatives describe sensitivity to individual parameters,
the gradient collects those sensitivities, and the chain rule propagates them
through composed computations.

The examples use small synthetic datasets generated in code. They are
educational demonstrations, not benchmark studies.

## Concepts Covered

- Derivatives and local linear approximations
- Partial derivatives and gradients
- Directional derivatives
- Jacobians and Hessians
- Scalar and vector forms of the chain rule
- Computational graphs and backpropagation
- Analytical, numerical, and automatic differentiation
- Gradient checking
- Vanishing and exploding gradients
- The relationship between gradients and optimization

## Why It Matters

Training a model usually means minimizing a scalar loss with respect to many
continuous parameters. The gradient indicates how sensitive that loss is to
each parameter, while the chain rule makes it possible to compute those
sensitivities through nested transformations.

This foundation appears in regression, neural networks, embedding and reranker
training, LLM fine-tuning, custom losses, and gradient-based explainability. It
also helps diagnose unstable training, disconnected computation graphs,
incorrect tensor shapes, and objectives that do not reflect the actual product
goal.

## Files

- `notes.md`: intuition, theory, formulas, assumptions, trade-offs,
  applications, limitations, and common mistakes.
- `example.py`: NumPy linear regression using analytical gradients, with a
  centered finite-difference gradient check.
- `from_scratch.py`: manual forward and backward pass through a nonlinear
  scalar neuron.
- `notebook.ipynb`: guided visual tour that reuses the visualization modules.
- `visual_explorer.py`: central CLI for generating all visual artifacts.
- `visualizations/`: focused modules for each mathematical concept.
- `interview_questions.md`: senior-level conceptual, mathematical, practical,
  and Applied AI questions with answers.
- `references.md`: books, course notes, papers, and official documentation.
- `requirements.txt`: minimal dependencies for this topic.

## How to Run

From the repository root, install the shared dependencies if needed:

```bash
python -m pip install -r requirements.txt
```

Run the NumPy example:

```bash
python 00-foundations/05-calculus-for-ml/example.py
```

Run the manual chain-rule example:

```bash
python 00-foundations/05-calculus-for-ml/from_scratch.py
```

Both scripts generate their inputs locally and require no network access,
credentials, or external dataset.

## Visual Exploration

The visual explorer turns the local calculus definitions into observable
geometric and numerical behavior. Every artifact answers a specific question:
What does a derivative measure locally? What is held fixed in a partial
derivative? Why does optimization move against the gradient? How does a loss
signal travel backward through composed operations? Why can repeated
derivatives shrink or grow?

All functions and data are generated locally and deterministically. GIFs use
Pillow and require neither ffmpeg nor ImageMagick.

### Installation

From this topic folder:

```bash
python -m pip install -r requirements.txt
```

Generate a reduced-frame visual tour:

```bash
python visual_explorer.py --quick
```

Generate every full-quality static figure, GIF, and interactive page:

```bash
python visual_explorer.py --all
```

Generate and display one group:

```bash
python visual_explorer.py --only gradient-descent --show
```

Additional controls:

```bash
python visual_explorer.py --only derivative --no-gif
python visual_explorer.py --all --frames 100 --dpi 180
python visual_explorer.py --quick --output-dir custom_outputs
```

`--only` accepts `derivative`, `partials`, `gradient-descent`, `chain-rule`,
`stability`, `numerical`, or `activations`. Running with only `--show` (or with
no mode flag) generates all groups at full quality and displays Matplotlib
figures after saving.

### Generated Outputs

| Visualization | Concept | Main output |
| --- | --- | --- |
| Moving tangent and secants | Derivative and local linearization | PNG and GIF |
| Surface slices | Partial derivatives | PNG and HTML |
| Loss landscape and trajectory | Gradients and gradient descent | PNG, GIF, and HTML |
| Learning-rate comparison | Step-size trade-offs | PNG |
| Computational graph | Chain rule and backpropagation | PNG and GIF |
| Gradient propagation | Vanishing and exploding gradients | PNG and GIF |
| Finite-difference comparison | Numerical differentiation | PNG |
| Activation comparison | Activation derivatives and saturation | PNG |
| Saddle surface | Stationary points | HTML |

Static PNG files are presentation-ready snapshots. GIFs show a short sequence
of mathematical states. Interactive HTML files support rotation, zoom, and
hover inspection; they are self-contained and can be opened directly in a
browser without a server or internet connection.

Artifacts are written under:

```text
outputs/
├── static/
├── animations/
└── interactive/
```

The repository-level `.gitignore` excludes `outputs/`, so generated artifacts
are not committed by default.

### Interview Observations

- A derivative is a local approximation; its tangent need not describe distant
  behavior.
- A partial derivative changes one coordinate while holding the others fixed.
- The gradient points uphill under the Euclidean norm; gradient descent uses
  its negative and still requires an appropriate step size.
- Backpropagation computes gradients by composing local derivatives. It is not
  an optimizer.
- Repeated Jacobian products, not activation functions alone, determine whether
  gradients vanish or explode.
- Finite differences are valuable for gradient checks, but their accuracy
  depends on epsilon and they scale poorly with parameter count.
- A zero gradient can identify a saddle point rather than a minimum.

## Key Takeaways

- A derivative is a local linear approximation, not a global description of a
  function.
- A gradient contains the partial derivatives of a scalar-valued function and
  points toward its steepest local increase under the Euclidean norm.
- Gradient-based minimization moves in the negative-gradient direction, but a
  decrease is not guaranteed for an arbitrary step size.
- The chain rule composes local derivatives; backpropagation applies it
  efficiently in reverse through a computational graph.
- Numerical differentiation is useful for checking gradients, not for training
  large models.
- Correct gradients cannot compensate for leakage, poor data, or a loss that
  is misaligned with the real objective.
