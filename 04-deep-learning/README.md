# Deep Learning

[Repository overview](../README.md) · [Full roadmap](../ROADMAP.md)

Module 4 spans global Days 53–62. **3 of 10 studies are implemented.**
Days 56–62 remain planned; no placeholder topic folders are published.

| Day | Implemented study | Evidence and review status |
|---:|---|---|
| 53 | [Perceptron and MLPs](53-perceptron-mlps/) | Tested perceptron, constructive XOR and NumPy gradients; synthetic comparison and nine-part offline visual lab with real checkpoints, six GIFs and three interactive HTMLs; author interpretation review required |
| 54 | [Backpropagation](54-backpropagation/) | Tested chain rule, finite differences, update semantics and weighted microbatch gradients; synthetic regression and eight-module visual lab with six GIFs and offline 3D loss surface; author interpretation review required |
| 55 | [Activation Functions](55-activation-functions/) | Tested derivatives/autograd, two native slider explorers, four GIFs, offline probability simplex and matched CPU MLP surfaces with an affine control; author interpretation review required |

Day 53 asks why nonlinear hidden representations change what a network can
express. Its [dated analytical record](53-perceptron-mlps/notes.md#executed-study--2026-11-18)
separates a noisy synthetic holdout comparison from exact truth-table fitting.
Universal approximation, optimization and generalization are distinct claims.
**Author review of interpretation candidates is pending.** No real-data or
production model claim follows from the examples or passing tests.

The separate [visual guide](53-perceptron-mlps/VISUAL_LAB.md) documents neuron
controls, hidden representations, a genuine optimization animation and
constructive approximation. Generated exports remain ignored pending preview
selection; visual execution does not approve scientific interpretations.

Day 54 asks how a scalar loss produces gradients throughout a computation.
Its [dated execution record](54-backpropagation/notes.md#executed-study--2026-11-18)
separates gradient-validation identities from training loss reduction on a
synthetic function. The backward pass and parameter update are explicit,
separate operations; no holdout or framework comparison is claimed.
Its [visual guide](54-backpropagation/VISUAL_LAB.md) separates a biased scalar
graph, an actual two-parameter loss landscape and controlled derivative chains.

Day 55 asks how an activation changes both values and local gradients.
Its [dated numerical record](55-activation-functions/notes.md#executed-study--2026-11-18)
separates mathematical derivative checks from model training: no activation
quality ranking or calibration benefit is inferred from the temperature sweep.
The separate [visual guide](55-activation-functions/VISUAL_LAB.md) connects
local slopes, controlled gradient chains, dying neurons, coupled class scores,
attention normalization and actual nonlinear decision surfaces.

Use the topic's execution commands and focused tests. The shared
[validation contract](../docs/validation.md) describes the checks and their
limits; [methodology](../docs/methodology.md) separates implementation,
automated verification and author interpretation review.
