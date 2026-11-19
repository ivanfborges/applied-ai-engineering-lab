# Deep Learning

[Repository overview](../README.md) · [Full roadmap](../ROADMAP.md)

Module 4 spans global Days 53–62. **7 of 10 studies are implemented.**
Days 60–62 remain planned; no placeholder topic folders are published.

| Day | Implemented study | Evidence and review status |
|---:|---|---|
| 53 | [Perceptron and MLPs](53-perceptron-mlps/) | Tested perceptron, constructive XOR and NumPy gradients; synthetic comparison and nine-part offline visual lab with real checkpoints, six GIFs and three interactive HTMLs; author interpretation review required |
| 54 | [Backpropagation](54-backpropagation/) | Tested chain rule, finite differences, update semantics and weighted microbatch gradients; synthetic regression and eight-module visual lab with six GIFs and offline 3D loss surface; author interpretation review required |
| 55 | [Activation Functions](55-activation-functions/) | Tested derivatives/autograd, two native slider explorers, four GIFs, offline probability simplex and matched CPU MLP surfaces with an affine control; author interpretation review required |
| 56 | [Optimizers](56-optimizers/) | Tested NumPy/PyTorch rules; eight-scene offline lab with seven GIFs, interactive 3D step inspection, moment histories and decay decomposition; matched synthetic CPU MLP configurations; author interpretation review required |
| 57 | [Regularization in Deep Learning](57-regularization/) | Tested dropout, BatchNorm, SGD/L2 and full checkpoint restoration; six-page Streamlit lab, offline 3D decay, five GIFs and three-seed 2D CPU comparison, separate from the analytical study; author interpretation review required |
| 58 | [Weight Initialization and Training Stability](58-weight-initialization/) | Tested initialization/gradients; offline dashboard, measured 3D propagation surface, five GIFs, BN/LN mechanisms, multi-seed synthetic training and optional Jacobian SVDs; separate analytical/visual protocols; author interpretation review required |
| 59 | [CNNs](59-cnns/) | Tested operators, exact spatial support and checkpoints; Streamlit lab, six GIFs, offline RGB/tensor 3D and paired synthetic CNN/MLP stress/failure analysis; separate analytical/visual protocols; author interpretation review required |

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

Day 56 asks how gradient history and parameter decay change an update.
Its [dated numerical record](56-optimizers/notes.md#executed-study--2026-11-18)
separates exact update controls from a single-seed synthetic classification
exercise. Rates are illustrative and no general optimizer ranking is claimed.
The separate [visual guide](56-optimizers/VISUAL_LAB.md) connects rotated and
Rosenbrock geometry to actual update vectors, internal states, bias correction,
coupled/decoupled decay, rate sensitivity and controlled synthetic noise.

Day 57 asks how activation noise, decay, normalization and stopping change
training and selection. Its [dated analytical record](57-regularization/notes.md#executed-study--2026-11-18)
separates mechanism checks from a single-seed synthetic comparison. Different
checkpoint policies and update budgets prevent a general ranking of techniques;
the final test is reserved for the configuration selected on validation.
The separate [visual guide](57-regularization/VISUAL_LAB.md) covers forward
masks, coupled/decoupled decay, BatchNorm running statistics, stopping-rule
replay and three-seed 2D training; it preserves the analytical study's distinct
20-feature configuration and evidence boundaries.

Day 58 asks how initialization scale changes forward magnitude and backward
sensitivity before learning. Its [dated analytical record](58-weight-initialization/notes.md#executed-study--2026-11-18)
separates squared-output loss gradients from a shared output-adjoint probe.
Finite-width second moments do not establish Jacobian conditioning, convergence
or generalization. Its separate [visual lab](58-weight-initialization/VISUAL_LAB.md)
executes paired multi-seed propagation, normalization mechanisms and synthetic
training; those measurements retain a distinct configuration and review status.

Day 59 asks how local shared filters and spatial aggregation support image
classification. Its [dated analytical record](59-cnns/notes.md#executed-study--2026-11-18)
separates single-channel operator checks from learned filters on synthetic
bars. Pooling changes spatial resolution, while global averaging does not
guarantee translation invariance. The same-generator test result establishes
neither realistic image performance nor robustness under distribution shift.
The separate [visual lab](59-cnns/VISUAL_LAB.md) makes weighted sums, channel
mixing, pooling and receptive fields inspectable. It adds two-seed 32×32
CNN/MLP measurements, real filter checkpoints and translation/noise diagnostics
without pooling them with the earlier 16×16 analytical evidence.

Use the topic's execution commands and focused tests. The shared
[validation contract](../docs/validation.md) describes the checks and their
limits; [methodology](../docs/methodology.md) separates implementation,
automated verification and author interpretation review.
