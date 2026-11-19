# Day 53 — Perceptron and Multilayer Perceptrons

[Deep Learning module](../README.md) · [Roadmap](../../ROADMAP.md)

**Why does depth need nonlinearity?** A perceptron draws one affine decision
boundary. An MLP composes learned affine transformations with nonlinear
activations, creating intermediate features that can separate patterns such
as XOR. Stacking affine layers alone leaves the final logit affine.

This study connects neurons, weights, bias, layer shapes, activation choices
and universal approximation intuition to executable checks. It separates
representational capacity, optimization success and generalization. These
distinctions matter when choosing a prediction head, combining embeddings
with structured features or deciding whether a neural model adds value over
a simpler baseline.

## What to inspect

| File | Purpose |
|---|---|
| [example.py](example.py) | Same synthetic noisy-XOR split for logistic regression, identity MLP and ReLU MLP; training/holdout metrics and fit warnings |
| [from_scratch.py](from_scratch.py) | Mistake-driven perceptron, exact two-ReLU XOR construction and educational NumPy tanh MLP with explicit gradients |
| [notes.md](notes.md) | Geometry, shapes, equations, assumptions, dated measured results and proposed extensions |
| [tests](tests/test_perceptron_mlps.py) | AND/XOR behavior, affine composition, every parameter's finite-difference gradient, numerical stability and library integration |
| [interview_questions.md](interview_questions.md) | Questions about capacity, bias, output semantics and evaluation |
| [references.md](references.md) | Primary approximation paper and official implementation guidance |
| [visualize_day53.py](visualize_day53.py) | Offline visual lab entry point; nine modes with PNG, Pillow GIF and standalone Plotly HTML exports |
| [VISUAL_LAB.md](VISUAL_LAB.md) | Visual questions, commands, mathematical contracts, measured runs and preview recommendations |
| [visual tests](tests/test_visual_math.py) | Checkpoint replay, activation derivatives, feature geometry, loss slices and temporary-directory rendering |

## Run from the repository root

Use stable Python 3.11+ in an isolated environment. Dependencies and bounds
come from [pyproject.toml](../../pyproject.toml); no deep-learning framework or
additional optional extra is needed for this topic.

The local [requirements.txt](requirements.txt) is a minimal runtime list only
for standalone copying of this lab. Use the root configuration and `dev` for
repository tests; the standalone list does not freeze versions or supply pytest.

```powershell
python -m pip install -e ".[dev]"
python 04-deep-learning/53-perceptron-mlps/example.py
python 04-deep-learning/53-perceptron-mlps/from_scratch.py
python scripts/validate_repo.py tests --scope 04-deep-learning/53-perceptron-mlps
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
```

The two analytical scripts print JSON to standard output and create no figures or model
artifacts. The comparison uses 800 synthetic observations around binary XOR
corners, Gaussian feature noise with standard deviation 0.12, and a stratified
600/200 split. Each model's scaler is fitted only on training observations.
All model settings are fixed before evaluating the holdout.

The NumPy demonstration uses all four exact binary corners for fitting and
evaluation. Its accuracy is a training diagnostic, with no independent test
population. Its `2 → 8 → 1` architecture has 33 trainable parameters. The
fixed two-ReLU construction represents XOR without learning any parameters.

## Executed evidence and boundaries

In the **2026-11-18 local run**, holdout accuracy was 0.575 for logistic
regression and the identity MLP, and 1.000 for the ReLU MLP. The separate
NumPy MLP reduced training BCE from 0.713177 to 0.000259 and predicted all
four truth-table labels. Full configuration, log-loss values, warnings,
interpretation candidates and limitations are in the
[dated experiment record](notes.md#executed-study--2026-11-18).

**Author review required:** the measured results are synthetic observations;
their interpretation remains pending. One seed and split do not establish
comparative reliability, calibrated probabilities or performance on real
tabular data. Matching gradients checks numerical implementation, not
generalization. This code is an educational implementation rather than a
production library replacement.

## Explore the visual lab

```powershell
python 04-deep-learning/53-perceptron-mlps/visualize_day53.py --mode all --quick
python 04-deep-learning/53-perceptron-mlps/visualize_day53.py --mode all
```

The [visual guide](VISUAL_LAB.md) maps nine demonstrations to their generated
files: neuron controls, real perceptron updates, XOR geometry, activation
derivatives, hidden features and 3D lifting, forward propagation, genuine
NumPy training checkpoints, a restricted loss surface and constructive ReLU
approximation. Open the three generated HTML files directly in a browser;
no Python server is needed. GIFs use Pillow.

Outputs remain ignored; no preview is automatically selected for Git.
The visual run record is separate from the analytical study above, with
quick/full configurations and interpretation candidates pending author review.

## Takeaways

- Bias shifts a neuron's response; weights and activation determine how inputs
  become features. Raw network weights are not global feature importance.
- A hard-threshold perceptron and a sigmoid classifier have different learning
  rules, despite both having affine decision boundaries in their inputs.
- Nonlinear hidden features make XOR representable; an optimizer must still
  find useful parameters.
- Universal approximation concerns the existence of approximating networks
  under stated assumptions. It gives no finite-data training guarantee.
- Input scaling, validation design, calibration, serving cost and simpler
  baselines remain separate engineering questions.
