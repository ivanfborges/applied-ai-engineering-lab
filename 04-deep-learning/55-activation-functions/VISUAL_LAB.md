# Day 55 — Visual Learning Laboratory

[Topic](README.md) · [Analytical derivations and historical run](notes.md)

Turn a local slope, a Jacobian entry or a logit into something inspectable.
The laboratory has ten teaching modules, two native slider windows, four
Pillow GIFs and three self-contained Plotly HTMLs. Every dataset and attention
matrix is synthetic. There are no downloads at runtime, pretrained models,
GPU requirements, external APIs or running web servers.

## Install and run

Keep the repository's Python **3.11+** requirement. The validated local
interpreter is Python 3.11.14; Python 3.10 is outside this repository's
supported dependency contract. NumPy, SciPy, Matplotlib, Pillow, scikit-learn
and Plotly are shared dependencies; PyTorch belongs to the canonical
`deep-learning` extra in [pyproject.toml](../../pyproject.toml).

From the repository root, in an isolated environment:

```powershell
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,deep-learning]"

# Seven static PNGs and analytical / CPU autograd checks; no MLP training.
python 04-deep-learning/55-activation-functions/visual_lab.py --mode quick

# All eight PNGs, four GIFs and three HTMLs; four matched CPU models.
python 04-deep-learning/55-activation-functions/visual_lab.py --mode full

# A lighter full run: all modules, fewer frames, updates and grid points.
python 04-deep-learning/55-activation-functions/visual_lab.py --mode full --frames 16 --epochs 150 --grid 31 --output 04-deep-learning/55-activation-functions/outputs/light

# One native Matplotlib window at a time; close to return to the terminal.
python 04-deep-learning/55-activation-functions/visual_lab.py --mode interactive --explorer activations
python 04-deep-learning/55-activation-functions/visual_lab.py --mode interactive --explorer softmax

# Three HTMLs and a complementary boundary PNG; same defaults as full.
python 04-deep-learning/55-activation-functions/interactive_3d.py
# Equivalent: visual_lab.py --mode 3d

python scripts/validate_repo.py tests --scope 04-deep-learning/55-activation-functions
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
```

Batch modes select the Agg backend and never open windows. Interactive mode
needs a working desktop backend; when the environment defaults to Agg, the
CLI selects TkAgg. Install Python with Tcl/Tk support for that route. A
headless shell cannot provide native slider interaction. Core analytical
examples do not need PyTorch; the visual batch checks do. Missing optional
libraries produce an installation message rather than silently omitting an
experiment. No FFmpeg, ImageMagick, Plotly account or Kaleido is required.

`--frames` accepts 12–80 frames per GIF; `--epochs` 1–2000 full-batch updates;
`--grid` 15–121 points per grid axis. Defaults are 32, 400 and 61. Every batch
prints generated filenames, decodes exported images and saves a uniquely dated
`run_<mode>_<UTC timestamp>.json` with versions, hypotheses, configuration,
observations, file hashes and limitations. Regenerating figures can replace
same-name exports in the chosen directory; dated JSON records remain separate.
Choose `--output` for a distinct run. Time and timestamps are observations,
not deterministic outputs; generated HTML element IDs can also differ.

## What to look for

All names below are generated inside ignored `outputs/`, unless a curated
copy is explicitly selected as a public preview. Open HTMLs directly in a
browser from disk. The runtime is embedded, so the HTMLs work offline.

| Module and output | What you see / mechanism | Observation to inspect | What it does not establish |
|---|---|---|---|
| A: `01_activation_derivatives.png`, activation explorer | Five forward functions, slopes and a shared input marker | Move into sigmoid/tanh tails; change the Leaky ReLU slope; compare local sensitivity | A local slope is not a whole-network gradient or training result |
| B: `02_signal_transformation.gif` | Arrows connect a selected input on one rail to its actual output on another, in four coordinated panels | Compression toward sigmoid/tanh bounds, negative ReLU suppression, smooth GELU modulation | Arrow length is not a gradient; the slope is displayed separately |
| C: `03_gradient_chain.png` / `.gif` | Layer-by-layer products of activation derivatives AND scalar weights, log magnitude axis and selected-depth table | Compare saturation, positive ReLU with unit weights and ReLU with weights 0.5 / 1.5 | These chosen scalar chains do not benchmark real deep-network training |
| D: `04_dying_relu.png` / `.gif` | Preactivation histograms, supervised outputs and actual gradient-descent loss history | ReLU stays inactive; Leaky ReLU follows a nonzero negative-side path in this construction | Leaky ReLU does not guarantee recovery for arbitrary inputs or settings |
| E: `05_gelu_geometry.png` | Exact GELU/ReLU/Leaky forward functions, derivatives and an origin zoom | GELU is smooth, slightly nonmonotonic and can have slopes above one | GELU is not a probability function or a universally better activation |
| F: `06_softmax_jacobian.png`, `.gif` temperature animation and softmax explorer | Logit bars, probability bars, entropy, Jacobian and change from the initial Jacobian | Lower T concentrates; a common offset preserves probabilities up to rounding | Normalization and entropy do not prove calibration or factual correctness |
| G: `07_softmax_simplex.html` | Probability vectors on the triangle $p_1+p_2+p_3=1$; fixed-logit trajectory colored by T | Rotate to see the plane, then move T to shift points toward vertices or the center | The triangle is a 2D manifold embedded in 3D, not a probability volume |
| H: `08_jacobian_change.png` | Initial Jacobian, Jacobian after raising the second logit and their difference, with a finite-difference error | Positive diagonal terms and negative cross-class effects; changing competition changes sensitivity | A difference of Jacobians is not a difference of probabilities |
| I: `09_decision_boundaries.png`, `09_probability_surfaces.html`, `09_raw_logit_surfaces.html` | Matched real CPU MLPs, training observations and seven real probability checkpoints; separate final logit surfaces | Rotate / zoom, select an epoch, compare $p=0.5$ contours to an identity-hidden-layer control | Single-split observations do not rank activations; a curved sigmoid surface alone does not prove a curved boundary |
| J: `10_attention.png` | Synthetic signed $QK^\top/\sqrt{d_k}$ scores and three temperature-normalized attention maps | Every query row sums to one across keys; temperature redistributes existing scores | Weights are not direct semantic importance or factual correctness; no full Transformer is trained |

## Interactive controls

**Activation explorer:** `Input x` moves a marker across both value and slope
panels; `Leaky slope` updates both the curve and derivative. The readout lists
every activation's current value and selected slope. At zero, ReLU and Leaky
ReLU have no ordinary derivative; the displayed 0 and alpha are backward
conventions. The numerical/autograd comparisons avoid these kinks.

**Softmax explorer:** three logit sliders are independent. A positive
`Temperature` slider updates probabilities, entropy, the current Jacobian
and its difference from the initial `[2,1,-1], T=1` Jacobian. `Offset`
adds the same constant to all logits. The bars show logits before that offset
so their vertical scale stays readable; the status reports its value and the
observed probability change. Exact tied maxima are displayed together. The
Jacobian formula is $(\operatorname{diag}(p)-pp^\top)/T$; the $1/T$ factor
must remain present when differentiating the original logits.

**3D controls:** drag to rotate, scroll to zoom and hover for values. The
simplex slider changes both the cloud and highlighted point; the temperature
trajectory remains as a reference. The probability-surface slider selects
actual recorded update counts. It starts at update 0; move it to 400 to see
the final default model. Black-edged observations at height 0/1 are training
labels, not model predictions. Logit surfaces are separate final-state HTMLs;
their z-axis is unbounded and is not a probability scale. Each panel rotates
independently and starts from the same camera orientation.
Vertical ranges and color limits are shared across all four panels within
each HTML, including the raw-logit comparison; the identity control's smaller
logit magnitude is therefore shown on the same scale as the nonlinear models.

## Mathematical contracts

Numeric readouts are rounded for readability: a displayed slope of `0.000`
can be a small nonzero value. Curves, products, losses and numerical checks
use the unrounded arrays.

For controlled chains, upstream sensitivity is one and local preactivations
are held at 4 for sigmoid/tanh and 1 for ReLU. These are realizable scalar
chains by choosing biases $b_l=z_l-w_l f(z_{l-1})$; at the first layer choose
the corresponding input. Sensitivity after $L$ steps is
$\prod_{l=1}^L |w_l f'(z_l)|$. It is not computed by repeatedly passing a
constant input through unbiased layers. The table and GIF use these same
derivative products.

The dying-neuron comparison uses 41 evenly spaced inputs in `[-1,1]`, target
$0.6+0.2x$, initial weight 0.2 and bias -1.2 for both activations. All initial
preactivations are negative. Both models minimize mean half-squared error
with 80 simultaneous full-batch gradient-descent updates at learning rate
0.5; Leaky ReLU has alpha 0.1. The loss derivative multiplies the residual,
the actual local activation slope and the input for the weight gradient.
Different initial outputs imply different initial losses despite identical
parameters. This comparison does not secretly reset a stalled neuron.

Exact GELU is $x\Phi(x)$ and its slope is $\Phi(x)+x\phi(x)$. NumPy/SciPy
formulas are checked against PyTorch `GELU(approximate="none")` in float64;
the tanh approximation is not used. The probability and attention calculations
reuse the stable NumPy softmax. Attention uses 3 synthetic queries, 4 keys
and key dimension 2, without masking. Sigmoid/tanh also appear in recurrent
gates/state updates, ReLU in many feed-forward/convolutional networks, GELU
in many Transformer feed-forward blocks and softmax in attention and token
normalization. These are architectural examples, not universal requirements:
modern Transformers can instead use SiLU, SwiGLU or other gated variants.

## Matched MLP experiment

**Hypothesis:** nonlinear hidden transformations permit curved boundaries
on noisy synthetic moons; a matched identity-hidden-layer model is an affine
logit control. A fair configuration does not imply equal optimization
difficulty or identify the best activation.

Generate 400 `make_moons` observations with noise 0.18 and seed 55. A fixed
stratified 75/25 split gives 300 training and 100 validation rows. Fit the
standardizer only on training rows. Each `2 -> 16 -> 16 -> 1` model uses
identical initial tensors copied from PyTorch Linear defaults, with no
activation-specific gain. The report records the common parameter SHA-256.
ReLU, Leaky ReLU (alpha 0.1), exact GELU and identity-hidden-layer models use
Adam, learning rate 0.02, 400 full-batch updates and CPU float32. The output
is a raw logit; `BCEWithLogitsLoss` receives that logit. Sigmoid is applied
only for plotted probabilities and a zero-logit threshold defines accuracy.

Models are evaluated at the final epoch, without validation-based selection
or early stopping. Seven grid checkpoints include the initial state and final
model. Measured training-loop time includes checkpoint grid evaluation but
excludes imports, setup and initial-state evaluation. It is neither pure
optimizer timing nor an inference benchmark. Runs use one numerical thread;
the helper restores caller RNG, thread and deterministic-algorithm settings.
Repeatability tests compare predictions, not wall-clock times or HTML bytes.

The plotting grid spans the observed feature extent for display, including
validation feature locations; this does not refit the training standardizer
or enter the optimizer. There is no independent test set, multi-seed uncertainty,
learning-rate search, noise sweep or real-data assessment. Threshold 0.5 is
fixed. Claims about calibration, throughput or generalization need separate
experiments. Interpretation candidates remain **author review required**.

## Primary references

- [PyTorch GELU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html): exact and approximate definitions.
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html): stable binary logit objective.
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html): seeds do not guarantee cross-platform/version identity.
- [Matplotlib Slider](https://matplotlib.org/stable/gallery/widgets/slider_demo.html): native interactive controls.
- [Plotly standalone HTML export](https://plotly.com/python/interactive-html-export/): embedded runtime for offline files.

Exact GELU and softmax derivations also appear in the [analytical notes](notes.md).

## Executed visual study — 2026-11-18

**Executor:** Codex, Windows CPU, Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1,
Matplotlib 3.11.2, Pillow 12.3.0, Plotly 6.9.0, scikit-learn 1.9.1,
PyTorch 2.15.0+cpu and pytest 9.2.0. **Author review required.** These are
visual-phase executions, separate from the historical analytical record.

Executed commands used the explicit stable interpreter described in the local
workflow, without changing global PATH or the old environment. PyTorch CPU
and its missing dependencies were added to that environment with `uv pip`
using the official CPU wheel index; `uv pip check --python` passed for all
78 installed packages. `pip` itself remains absent in this local environment.
The public installation commands above use standard pip in the reader's
isolated environment.

The `quick` run generated seven PNGs. The `full` run generated all 15
artifacts with default **32 requested frames, 400 updates/model, 61x61 grids**.
The dated manifest is `outputs/run_full_20261118T150304442621Z.json`, ignored
and intentionally not linked as a public asset. The subsequent standalone
`interactive_3d.py` command also executed after disabling Plotly autoplay;
the simplex export was then regenerated to improve control spacing.
Existing dated run manifests were retained. Rerunning batch commands replaces
regenerable named exports and adds a new timestamped manifest; selected
`assets/` previews are not overwritten by the generators.

**Observed numerical results:** the largest float64 analytical/autograd
derivative discrepancy was $2.22\times10^{-16}$. The plotted softmax
Jacobian disagreed with central differences by at most $2.19\times10^{-12}$.
After 20 controlled layers, saturated sigmoid and tanh chains had gradient
magnitudes $8.73\times10^{-36}$ and $3.53\times10^{-58}$. Positive ReLU
with unit weights retained magnitude 1; weights 0.5 and 1.5 produced
$9.54\times10^{-7}$ and $3.33\times10^3$. These are local derivative
products under the stated fixed-preactivation construction.

In the dying-neuron demonstration, ReLU's initial gradient norm was zero
and half-MSE stayed at 0.187. Leaky ReLU's initial gradient norm was
0.0722751, and its half-MSE changed from 0.264870 to
$4.00\times10^{-10}$ after 80 updates. **Interpretation candidate:** the
retained negative-side derivative supplied a parameter-update path in this
deliberately constructed example. It is not a general recovery guarantee.

The first full-run MLP observations were:

| Hidden activation | Initial training BCE | Final training BCE | Validation BCE | Validation accuracy | Training-loop seconds |
|---|---:|---:|---:|---:|---:|
| ReLU | 0.734236 | 0.004047 | 0.270106 | 0.97 | 0.379 |
| Leaky ReLU, alpha=0.1 | 0.739556 | 0.005642 | 0.273976 | 0.98 | 0.370 |
| Exact GELU | 0.727365 | 0.002536 | 0.401959 | 0.95 | 0.476 |
| Identity control | 0.781477 | 0.259009 | 0.227115 | 0.91 | 0.354 |

All four models shared initial parameter SHA-256
`f99c34e38a1e32f215a8cd86e3a97cefa284e06bd0e25db291aa9c71ab452f11`.
The recorded checkpoints were updates 0, 66, 133, 200, 266, 333 and 400.
Times are individual local observations with the timing scope above; import
and optimizer setup overhead is excluded. They are not reproducible constants.

**Interpretation candidates:** the nonlinear models fitted curved moons
boundaries, whereas the identity model retained an affine logit and straight
$p=0.5$ boundary. The lowest validation BCE belonged to the identity control,
despite its lower accuracy. Low training loss, probability confidence and
threshold accuracy tell different stories. These observations do not identify
a universally best activation, and the 100 validation rows provide limited
evidence. No independent test set, calibration fit or multi-seed experiment
was executed.

All exported PNGs/GIFs decoded. The four GIFs contained respectively 32
(signal transformation), 21 (integer-depth chain), 32 (dying neuron) and 31
(temperature sweep) decoded frames: Pillow coalesces identical adjacent frames
without removing their playback time. All loop indefinitely and use Pillow
only. Representative first/intermediate/final frames and the eight PNGs were
visually inspected. The same batch is not a GitHub rendering check.

The two native TkAgg windows were opened and closed locally. Automated mouse
press/motion/release events exercised all seven sliders, including input zero,
Leaky slope 0.2, modified logits, T=2 and an offset of 5,000. The displayed
probabilities matched the numerical calculation; native-window screenshots
were inspected after increasing the activation window height. This verifies
the tested event paths, not every human interaction, display scaling or OS.

**Browser verification:** all three HTMLs opened as local files in offline
Chrome 157.0.8103.79 at 1440x1100. Twelve checks passed: simplex geometry,
initial states without autoplay, actual slider clicks, real 3D rotation,
modebar Zoom with mouse drag, probability checkpoints, all 14,884 final
probability/logit cell correspondences and the affine-logit control. Maximum
probability-versus-sigmoid disagreement was $8.76\times10^{-8}$; identity
grid second differences were at most $3.82\times10^{-6}$. There were zero
page/console errors and zero HTTP(S) requests, with three Canvas2D readback
performance warnings. Screenshots of the three pages were saved locally.
This is Chrome-only desktop evidence, not cross-browser, mobile, accessibility
or GitHub rendering validation. Mouse-wheel zoom was not confirmed by the
initial harness; the verified zoom path is the Plotly Zoom control plus drag.
Earlier harness attempts read an asynchronous frame too early and waited
on an undocumented transition field. Those validation attempts were corrected
without treating a timeout as success.

## Preview selection and local artifacts

**Final validation:** 86 focused tests passed (68 analytical and 18 visual),
without skips or warnings. Following the final 3D scale/layout adjustment,
its export/checkpoint regression was rerun: 1 passed, 17 deselected. The
complete focused run included actual GIF exports in temporary directories;
the additional selection does not represent another 18-test run. Repository
syntax passed for 377 Python files and internal links passed for 1,307
targets in 250 Markdown files. `git diff --check` passed, with Git's LF/CRLF
conversion notices. The CPU dependency compatibility check passed.
CI configuration now installs CPU PyTorch and the canonical extra, but no
GitHub Actions execution, Linux run or repository-wide test suite was claimed
for this change.

Three public Git candidates were selected after inspection:

| Preview | Bytes | Learning question |
|---|---:|---|
| `assets/01_activation_derivatives.png` | 189,379 | How do signal transformations and local slopes differ? |
| `assets/04_dying_relu.gif` | 478,459 | How can a negative-side update path change a stalled neuron? |
| `assets/09_decision_boundaries.png` | 377,547 | What changes when an otherwise matched MLP has nonlinear hidden layers? |

Total: **1,045,385 bytes**, within the repository's preview budgets. Files
are exact copies of inspected exports; only these paths are intentionally
unignored. All other exports and execution/browser records remain ignored.
The default standalone HTMLs are around 5–6 MiB each because they embed
Plotly's runtime and actual surface data. They are local exports, not public
README links. No frame dumps, model checkpoints or pretrained weights are
selected for publication.
