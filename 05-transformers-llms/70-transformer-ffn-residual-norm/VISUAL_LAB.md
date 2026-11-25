# Day 70 visual laboratory

[Topic overview](README.md) · [Generator](visualizations.py) · [Numerical core](visual_core.py)

This lab makes feature transformations, normalization and residual paths
observable using **synthetic arrays and untrained networks**. Every empirical
plot is computed from actual tensors. Scalar products and architecture/storage
estimates are labeled analytical or schematic. Initialization diagnostics do
not establish convergence, semantic quality or production readiness.

## Install and execute

Use Python 3.11+ in an isolated compatible environment. Dependencies and ranges
are centralized in [pyproject.toml](../../pyproject.toml):

```sh
python -m pip install -e ".[dev,deep-learning]"
python 05-transformers-llms/70-transformer-ffn-residual-norm/visualizations.py --list
python 05-transformers-llms/70-transformer-ffn-residual-norm/visualizations.py --validate-only
python 05-transformers-llms/70-transformer-ffn-residual-norm/visualizations.py --all --quick
python 05-transformers-llms/70-transformer-ffn-residual-norm/visualizations.py --topic normalization --quick --show
python 05-transformers-llms/70-transformer-ffn-residual-norm/visualizations.py --topic gradients pre-post-norm --depth 48 --d-model 64 --expansion 4 --seed 270
python 05-transformers-llms/70-transformer-ffn-residual-norm/visualizations.py --topic complexity --html-only
```

Each invocation creates a unique timestamped directory under `outputs/`.
Open its `index.html` to navigate the generated files. Each interactive HTML
contains its Plotly runtime, data and controls, so it can also be opened or
copied individually without an internet connection. No web server is required.
`--show` opens only the index, never dozens of tabs. Default plotting uses Agg
and saves files without a desktop. No pretrained model or external data is
downloaded. Dependencies must already be installed.

Use `--output-dir` for a different output root. Keep it in an ignored directory
for bulk exports. Default CPU execution is sufficient; optional `--device cuda`
requires an available CUDA installation and is not automatically selected.
Pure NumPy and small explanatory fixtures remain CPU operations.

With no selection the script generates the FFN animation. `--quick` without
`--topic` selects all groups. `--all` without quick mode uses the full profile.
`--html-only` skips raster rendering but still computes selected experiments and
writes their CSVs. Raster groups export their related PNGs and GIFs together.

| Setting | Quick | Full default |
|---|---|---|
| Synthetic seed | 270 | 270 |
| Diagnostic seeds | 270, 271 | 270, 271, 272 |
| Depths | 6, 12, requested depth (default 24), deduplicated | 6, 12, 24, 48, requested depth, deduplicated |
| Model/FFN width | 64/256 unless configured | 64/256 unless configured |
| Transformer heads | 4 | 4 |
| Batch/tokens | 2/8 | 2/16 |
| Numeric precision/threads | float64 / one PyTorch thread | float64 / one PyTorch thread |
| Geometry samples | 48 | 96 |
| Distribution samples | 512 | 2,048 |

The small FFN, SwiGLU, 3D geometry and complete-block fixtures retain their
explicit teaching dimensions independently of the configurable diagnostic
model. Parameter bounds reject accidental unbounded runs: model width must be
a multiple of four in 4–256; requested depth 1–96; expansion 0.25–8.

## Visualization index

The generated index repeats each question, mechanism, observation, engineering
connection and limitation next to its outputs. Names below are **local output
labels**, not links to versioned assets.

| Topic selection | Main artifacts | What to inspect and why |
|---|---|---|
| `ffn` | `ffn_information_flow.gif` | Actual 4×8 → 4×32 → 4×8 matrices, shared weights, GELU and one highlighted token. Read rounded values; changing a token affects only that row. FFN feature processing does not directly mix tokens. |
| `geometry` | `ffn_geometric_transformations.html`, `linear_vs_nonlinear_ffn.gif` | Same 3→12→3 projections with identity/ReLU/GELU. Colors and sample IDs persist. Linear composition equals one affine map. Random nonlinear features are not assumed to improve clusters. |
| `activations` | `activation_functions.png`, `activation_derivatives.png`, `activation_distribution_comparison.html` | Curves, autograd slopes and outputs under the same shifted/scaled distribution. ReLU derivative at zero is 0; GELU uses the tanh approximation. Input distribution affects suppression and local gradients. |
| `residual` | `residual_vector_addition.html`, `residual_stream_trajectory.html`, `residual_information_flow.gif` | Original-coordinate addition and translated branch direction, alpha 0–2, and repeated updates versus replacement with the same transformations. State-dependent branch inputs diverge. No information-preservation guarantee. |
| `gradients` | `analytical_gradient_propagation.html`, `gradient_propagation_experiment.png`, `gradient_propagation_depth.gif`, `dense_gradient_measurements.csv` | Separate scalar formulas from actual tanh-network autograd. Compare L2, RMS, relative gradients, activation scale and sampled signed distributions. Exact zeros and float64 overflow/underflow remain explicit. |
| `normalization` | `layer_normalization_step_by_step.gif`, `layer_normalization_geometry.html`, `layer_normalization_distributions.png` | Mean, centering, variance, stabilized scale and affine parameters. Explore zero-mean plane, reference circle, input mean/scale and near-constant vectors. Moments do not imply a Gaussian distribution. |
| `normalization-comparison` | `normalization_axes.gif`, `normalization_methods_comparison.html`, `layernorm_vs_rmsnorm_geometry.html` | Which values share statistics? Compare scale/offset, adding/removing a batch member and modifying a token. Training BatchNorm couples tokens and samples; LN/RMS and fixed-stat eval BN do not. |
| `pre-post-norm` | `pre_norm_vs_post_norm_gradients.png`, `pre_norm_vs_post_norm_activations.png`, `residual_update_ratio.png`, `transformer_gradient_heatmap.png`, `normalization_depth_comparison.html`, `transformer_measurements.csv` | Genuine attention blocks and matched initial parameters. Profile absolute/relative gradients, stream and branch RMS, update ratios, consecutive-state cosine and parameter gradients. Explore measured depth surfaces and readout sensitivity; no trained-model ranking. |
| `swiglu` | `swiglu_information_flow.gif`, `gated_feature_transformations.html`, `ffn_parameter_comparison.png` | Independent gate/up projections, SiLU and signed products. Manual feature/gate controls show amplification, suppression and sign changes. Counts include biases and use the nearest integer gated width to match a dense budget. |
| `complexity` | `transformer_parameter_explorer.html`, `attention_vs_ffn_complexity.html`, `transformer_parameter_distribution.png` | Model width, expansion, layers, sequence length and dense/gated FFN. Separate parameter/storage accounting from full-sequence or cached one-token MACs. These are theoretical estimates, not memory/latency benchmarks. |
| `transformer-block` | `transformer_block_forward_pass.gif`, `transformer_block_backward_pass.gif`, `transformer_block_architecture.png` | Actual 1×4×8 Pre-LN activations, two heads and 32 FFN features. Shapes and colors identify operations. Backward contributions use actual VJPs through attention and FFN as well as identity paths; vector norms need not add. |

## Coordinate, normalization and gradient contracts

**Geometry:** the 3→12→3 FFN displays original input/output coordinates and
selected hidden coordinates 0–2. There is no independently refitted PCA.
Stage transitions cross feature spaces; interpolated animation frames are
display transitions, not the full forward computation or a physical trajectory.
Camera and axis bounds are stable across activation comparisons. Point IDs
remain available on hover in the HTML.
Residual pages share equal coordinate ranges across all displayed paths and
adapt those ranges when alpha changes, keeping states visible even at alpha=2.
Camera orientation persists; compare numerical axis ticks and vector readouts
when assessing magnitude across different alpha values.

For LayerNorm in three dimensions, centering is the orthogonal projection
onto $x_1+x_2+x_3=0$. Before affine transformation:

$$
\|\hat{x}\|_2^2=\frac{3v}{v+\epsilon}.
$$

It approaches the radius-$\sqrt{3}$ reference circle for nonconstant inputs
when epsilon is negligible. Constant vectors map to zero before beta;
near-constant inputs shrink when epsilon dominates. Scalar gamma/beta controls
are a teaching simplification; the step animation uses feature-wise affine
parameters. Neither mechanism imposes Gaussian output distributions.

**BatchNorm layout:** input is $(B,T,D)$, transposed to $(B,D,T)$ for
`BatchNorm1d(D)`. Training normalization uses population variance over $(B,T)$
per feature channel. The base tensor calibrates actual PyTorch running moments
with momentum 0.1, mean initially zero and variance initially one. The running
variance update uses the unbiased estimate. Evaluation uses those frozen
moments. Browser interventions do not recalibrate them.

**Empirical diagnostics:** independent Gaussian inputs and targets, mean MSE,
untrained float64 modules, zero optimizer updates, no dropout or masks. Dense
experiments use tanh and no normalization. Transformer experiments use genuine
bidirectional four-head attention, tanh-approximate GELU and two affine
LayerNorms. Layers have separate parameters; depth comparisons reuse the same
initialized prefix. Pre/Post models copy corresponding weights. Conditions
are raw-readout Pre-LN, raw-readout Post-LN, and Pre-LN with a non-affine final
LayerNorm sensitivity readout. This last comparison is separate from the
earlier analytical run's shared readout intervention.

For each seed and depth, relative gradients divide by the raw final-block
state's gradient L2 norm. L2/RMS aggregate all batch/token/feature elements.
The update ratio uses branch RMS divided by the stream immediately before
addition, before any subsequent Post-LN. Consecutive-state cosine flattens
the whole tensor. Lines are seed means; shaded bands are seed min/max, not
confidence intervals. Missing depth/layer combinations are blank, not zeros.
Gradients are retained at all block boundaries; distributions sample at most
64 input-gradient coordinates per seed. Both CSVs contain every boundary.

**Numerical extremes:** intentionally unstable scalar cases are evaluated in
log space with explicit exact-zero, float64 overflow and float64 underflow
statuses. Zero values are omitted from the logarithmic plot and counted in its
readout; they are not replaced by an arbitrary positive constant. Empirical
networks raise on nonfinite states/gradients; they do not silently clip a failed
run into a successful visualization.

## Accounting contract

Conventional full-width MHA has $4d^2+4d$ parameters. Two affine norms add
$4d$. Dense FFN has $2dh+h+d$; SwiGLU with biases has $3dh+2h+d$.
The small default budget match is 8→32→8 dense versus 8→21→8 gated:
552 versus 554 parameters. Matching ignores no biases and does not establish
equal quality, memory traffic or latency.

Matrix arithmetic uses MACs, with **one MAC = two FLOPs** by convention.
Full-sequence MHA is $4Td^2+2T^2d$, and cached one-token MHA is
$4d^2+2Td$, where $T$ includes the new token. FFN uses $(2\text{ or }3)Tdh$
for full sequence or $(2\text{ or }3)dh$ for a cached new token. Bias addition,
nonlinearities, norms, softmax, cache reads and other memory movement are
excluded. Weight byte estimates omit embeddings, task readout, optimizer,
activations, gradients and KV cache. Ideal INT8 omits scales/metadata.

## Source and validation map

- [visualizations.py](visualizations.py): CLI, unique runs, manifest, CSV and offline gallery checks.
- [visual_core.py](visual_core.py): reusable calculations and actual PyTorch experiments.
- [visual_figures.py](visual_figures.py): Matplotlib PNG/GIF rendering with PillowWriter.
- [visual_html.py](visual_html.py) and [visual_controls.js](visual_controls.js): self-contained HTML and live controls.
- [Numerical tests](tests/test_visual_core.py) and [real-export tests](tests/test_visual_exports.py): equations, axes, counters, actual autograd and temporary-directory exports.

```sh
python scripts/validate_repo.py tests --scope 05-transformers-llms/70-transformer-ffn-residual-norm
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

Raster validation decodes all composited GIF frames and verifies looping;
HTML resource validation checks embedded runtime and local gallery links.
Neither substitutes for browser interaction checks or scientific review.

## Preview candidates and human decisions

All bulk exports remain ignored. No asset has been automatically selected,
unignored, committed or published. Recommended candidates to inspect are:

1. `transformer_block_forward_pass.gif`: complete actual operation flow.
2. `layer_normalization_geometry.html`: exact 3D plane/circle and epsilon controls;
   for GitHub, prefer a reviewed screenshot over publishing every large HTML.
3. `pre_norm_vs_post_norm_gradients.png`: measured profiles with seed variability.
4. `linear_vs_nonlinear_ffn.gif`: fixed-coordinate nonlinear transformations.

Final interpretation, asset selection and publication remain the author's
decisions. Full-profile CPU runs, CUDA, training, mixed precision and actual
latency comparisons require separate execution evidence before claiming results.

## Execution record — 2026-11-25

**Executor:** Codex, Windows CPU. **Status:** author interpretation and public
asset selection required. This visual phase is separate from the analytical
seed-70 records in [notes.md](notes.md#execution-record--2026-11-25).

**Executed configuration:** `--all --quick`, seed family 270/271, depths
6/12/24, batch 2, eight tokens, width 64, four heads, FFN width 256, float64,
one PyTorch thread. Python 3.11.14, NumPy 2.4.6, PyTorch 2.15.0+cpu,
Matplotlib 3.11.2, Plotly 6.9.0 and Pillow 12.3.0. No dependency was installed
or upgraded. The first fully completed run was
`visual-run-20261125T183743533299Z`; the earlier partial directory was retained
after an index-description lookup failure, fixed before the successful run.
Each completed run includes its own source SHA-256 map and artifact manifest.

The complete presentation-corrected raster run is
`visual-run-20261125T185010987198Z`. The final assembled gallery is
`visual-run-20261125T185707356399Z`: its HTMLs were re-exported to correct
residual axis bounds at alpha=2; twenty unchanged, inspected PNG/GIF files were
copied losslessly from the raster run. The manifest records that reuse, verifies
the raster hashes and numerical/renderer source hashes, and confirms both CSVs
are byte-identical. This is a presentation revision, not a new training or
quality experiment. A normal `--all --quick` invocation regenerates every asset
from the final source without needing the earlier directories.

**Hypotheses:** identity paths change measured gradient propagation;
normalization placement changes stream/update scales; final normalization
changes the loss-boundary gradient without changing raw internal states.
The norm-axis interventions should couple an unmodified sample only through
training BatchNorm. These are initialization/mechanism questions, not training
or language-quality comparisons.

The quick run produced 35 validated artifacts: 11 PNGs, nine GIFs, twelve
interactive standalone HTMLs, one gallery and two CSVs. The CSVs contain 180
dense-network and 270 Transformer boundary measurements. Every PNG and every
composited GIF frame was decoded; first/middle/last GIF states and PNG contact
sheets were visually inspected. Presentation corrections use new run
directories so earlier numerical/visual records remain available.

### Actual depth-24 endpoints

Input gradients below are L2 norms; the ratio divides by the raw final-state
gradient L2 norm for the same condition and seed. RMS is measured at the raw
last-block state, before any optional final readout normalization.

| Experiment | Seed | Condition | Output RMS | Input gradient L2 | Relative input gradient |
|---|---:|---|---:|---:|---:|
| Dense tanh | 270 | plain | 0.099853 | 5.929444e-8 | 9.070354e-7 |
| Dense tanh | 270 | residual | 3.385296 | 1.080779 | 4.921176 |
| Dense tanh | 271 | plain | 0.071218 | 7.555431e-8 | 1.146301e-6 |
| Dense tanh | 271 | residual | 3.626736 | 1.103735 | 4.693062 |
| Transformer | 270 | Pre-LN | 1.745684 | 0.473962 | 3.841075 |
| Transformer | 270 | Post-LN | 0.999995 | 0.111348 | 1.269258 |
| Transformer | 270 | Pre-LN + final LN | 1.745684 | 0.077013 | 2.090040 |
| Transformer | 271 | Pre-LN | 1.974122 | 0.591569 | 4.172955 |
| Transformer | 271 | Post-LN | 0.999995 | 0.059728 | 0.663079 |
| Transformer | 271 | Pre-LN + final LN | 1.974122 | 0.058967 | 1.893313 |

Maximum absolute NumPy/PyTorch errors at seed 270 were
$4.441\times10^{-16}$ for LayerNorm forward, $8.882\times10^{-16}$ for its
input VJP and $3.553\times10^{-15}$ for the FFN. The complete block's gradient
vector decomposition reconstructed the input gradient within
$1.388\times10^{-17}$. Modifying one token in the second batch member changed
the unmodified first sample by 0 for LN, RMSNorm and frozen-stat BN, and by
0.903612 at maximum absolute difference for training BN. Actual empirical
states and gradients were finite. Scalar controls separately reported exact
zero, overflow and underflow without substituting positive floors.

The focused topic suite passed 69 tests across three isolated files, without
skips or warnings. Affected numerical and HTML/export checks were rerun after
their final changes. Repository parsing passed for 508 Python files; internal
links passed for 1,881 links in 314 Markdown files. The explicit-interpreter
offline dependency check found 81 compatible installed packages. Node parsed
the final controls script successfully. Artifact/source hashes, all 450 CSV
rows, the ten published endpoint rows, ignored-output status and unchanged
private-source SHA-256 were also checked. `git diff --check` passed with only
Git's informational LF-to-CRLF warning for the module index.

**Candidate interpretations — author review required:** the dense example
shows strong attenuation in its plain-network input gradient for these two
seeds, while residual addition also increases stream magnitude. Post-LN's
near-unit RMS follows its final norm at initialization, rather than proving
better representations. The Pre-LN readout intervention substantially changes
input gradients at unchanged raw activation RMS. None of these norms, ratios
or random-target losses establishes convergence or final model quality.

### Browser evidence and limits

Twelve final-revision HTMLs passed 113 local interaction/numerical assertions
in offline Headless Chrome 158 on Windows, using software WebGL through SwiftShader.
Checks exercised keyboard sliders, selectors, play/pause, reset, actual 3D
camera drags, numerical identities, BatchNorm coupling, theoretical counts and
the alpha=2 residual axis-bound regression.
There were no JavaScript/console errors and no external HTTP(S) requests.
Canvas readback/`ReadPixels` performance warnings and one SwiftShader/ANGLE
framebuffer diagnostic occurred; no JavaScript exception was reported. These
browser-driver messages are not inference timings. CLI daemon startup could
not retain a session in this environment, so the audit used the existing cached Playwright package
directly with an explicit installed Chrome path. No browser was downloaded.

This establishes only the inspected local desktop flows. It does not establish
native-GPU behavior, cross-browser/mobile/accessibility coverage, GitHub math
rendering or scientific validity. CUDA, the full three-seed/16-token/48-layer
profile, training, mixed precision and latency benchmarks were not executed.
