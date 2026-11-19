# Day 58 — Signal propagation and stability lab

This is a separate, executable visual phase. It preserves the historical
[analytical experiment](notes.md#executed-study--2026-11-18), its configuration
and its interpretation-review status. All data are synthetic. **Author review
of interpretation candidates is required.**

## Run from the repository root

Use stable Python 3.11+ in an isolated environment, with the canonical extras:

```powershell
python -m pip install -e ".[dev,deep-learning]"
python 04-deep-learning/58-weight-initialization/main.py --mode quick --no-display
python 04-deep-learning/58-weight-initialization/main.py --mode full --no-display
python 04-deep-learning/58-weight-initialization/main.py --mode interactive --open
```

For CPU-only PyTorch, follow [the shared instructions](../../docs/validation.md).
NumPy, Matplotlib, Plotly and Pillow are already canonical dependencies. No new
framework, notebook kernel, CDN, network connection, server or GPU is required.
In Ivan's workspace, use the stable interpreter selected by the local workflow;
the global Python and old `.venv` are not the execution reference.

```powershell
python 04-deep-learning/58-weight-initialization/main.py --mode quick --seed 61 --depth 40 --width 96 --output-dir outputs/day58/custom --no-display
python scripts/validate_repo.py tests --scope 04-deep-learning/58-weight-initialization
```

The default is no display windows. `--open` opens the finished dashboard in a
browser; `--no-display` makes the default explicit. Depth is bounded to 1–50,
width to 2–128. Very shallow depth choices can have fewer distinct early/middle/
late histograms; depth 20–50 is the intended propagation demonstration.

The default output root is ignored `outputs/day58/visual-lab/`. Each invocation
creates a unique timestamped run directory with `interactive/`, `static/`,
`gifs/` and `run.json`. The printed dashboard path opens directly as a local file.
Its Plotly runtime, measurements, styles and controls are embedded in one HTML.
There is no call to Python when a selector changes.

The sibling `measurements-<fingerprint>.json` is a reusable cache keyed by the
experiment source, configuration, mode and numerical-library versions. Quick
and interactive use the same measurement cache. Rendering changes do not
require recomputing measurements. The run manifest records cache reuse, elapsed
time, hashes and artifact sizes. Keep that JSON and the measurement file together
when retaining evidence. New records do not replace dated analytical runs.

## Modes and numerical protocol

| Mode | Configuration and scope |
|---|---|
| `quick` | Depth 30, width 64, Gaussian batch 64; 3 paired seeds; selectable grid; measured surface; normalization; 27 training trajectories including clipping; five GIFs, five PNGs and two HTMLs |
| `interactive` | Same quick measurements, reused when cached; emits only the two HTML documents |
| `full` | Default depth 50, width 64; 5 paired seeds; 45 training trajectories; 100 epochs; adds actual 12×12 Jacobian/SVD diagnostics and a sixth PNG |

All experiments use CPU float64 and the CLI sets one PyTorch numeric thread.
The numerical runner additionally limits common numerical thread pools for tests.
Workload grows with depth, batch and approximately the square of width. GIF
rendering has a separate cost that grows with the number of frames. The manifest's
elapsed time includes measurement and export, or just cached loading and export;
it is not a model-performance benchmark. Concrete local timings belong in the
dated execution record below.

### Propagation: identical draws, different scales

The visual phase uses small weights with standard deviation **0.25 times He**
and oversized weights with standard deviation **1.6 times He**. These are
width-aware scale multipliers, distinct from the analytical script's fixed
0.06/0.18 standard deviations. Square layers use unit-gain Xavier standard
deviation $1/\sqrt{n}$ and He/ReLU fan-in standard deviation $\sqrt{2/n}$.
The He-plus-LayerNorm baseline reuses exactly the He weights.

Weights share Gaussian samples across strategies; biases start at zero. The
dashboard's precomputed grid crosses two widths (requested width and half),
two depths (requested depth and 10, bounded by the requested depth), ReLU/tanh,
none/LayerNorm/BatchNorm and the first two seeds. Each initializer is actually
measured for every listed grid point. Duplicate widths/depths collapse. Controls
select saved results and never continuously recompute a network. The strategy
selector chooses which matched curves are visible.

Tanh deliberately receives the same scales as the ReLU comparison; this is a
controlled activation mismatch, not a recommendation for tanh initialization.
BN and LN are before the nonlinearity, with epsilon $10^{-5}$ and initial affine
scale 1/shift 0. BN diagnostic traces use training mode. Width changes and
normalization invalidate parts of the simple independent-layer approximation.

Second moment $q=\operatorname{mean}(a^2)$, centered variance and mean are
recorded separately. ReLU's nonzero mean means they cannot share a label.
Layer 0 is the input. Histograms use at most 1,024 deterministic samples in
baseline exports and 128 in selectable grid entries; annotations use the full
batch. The synthetic paired-symmetry ReLU sample contains 100,000 values.

### Backward: two objectives and separate parameter gradients

The fixed output adjoint is a shared Gaussian tensor normalized to RMS 1.
Autograd computes $J_{D:l}^{\top}u$, equivalently the derivative of
$S=\sum_{b,j}u_{bj}a_{D,bj}$ with fixed $u$. Its terminal RMS is 1 independently
of activation scale. The second objective is $L=\operatorname{mean}(a_D^2)$;
its terminal gradient already depends on output scale. The dashboard labels
them separately and never pools their absolute gradient magnitudes.

Parameter-gradient RMS always refers to the **squared-output loss**, including
when the activation plot shows the fixed probe. Only linear-weight matrices
are included in that parameter-gradient chart; biases and normalization affine
parameters are not part of it. Training's global norms later include all model
parameters and concern a different task/objective.

Zeros and non-finite metrics are gaps on log plots, with explicit status counts.
Positive subnormal values are retained, not replaced by a floor. Raw JSON stores
non-finite values as `null` alongside a non-finite status; zero metrics are
explicitly classified as exact-zero-or-underflow. A zero alone cannot determine
whether ReLU gating or numerical underflow caused it. Scaled RMS computation
reduces premature overflow in the statistic, but it cannot repair non-finite
network values or loss computation.

### Measured surface

The 3D surface fixes ReLU/no normalization, batch 16, one seed and width capped
at 48. It measures a finite grid of network depths and 19 weight-scale
multipliers from 0.2 to 2.0, explicitly including $1/\sqrt2$ (Xavier) and 1 (He).
Each backward cell comes from a real vector-Jacobian product for that depth.
The output probe is common across scale choices at a fixed shape. The surface
is cached before interaction. Triangulation connects neighboring measured
cells; intermediate positions are not additional network measurements.

He scale 1 marks an approximate balance under constant-width ReLU assumptions;
it is not a learned optimum or a guaranteed stable boundary. This surface is
not a loss landscape. The forward z-axis is log10(second moment), whereas the
backward view is log10(input fixed-adjoint RMS), with distinct units.

### Normalization and inference

Synthetic features have means $[-3,-1,2,6]$ and scales $[0.5,1,2,3]$.
Heatmaps use rows as examples and columns as features. Training BN shares
statistics down a column; LN shares them along a row. These mechanism
demonstrations disable affine parameters to isolate normalization axes.

Batch-size sensitivity uses 80 independent batches at sizes 2, 8, 32 and 128.
The displayed standard deviations describe variability of estimated feature
means; they are not training accuracies. The anchor demonstration holds one
sample fixed while replacing its companions with a shifted group.

Before the inference demonstration, BN runs **100 actual training-mode forward
batches of size 32** from the reference population. The same held batch is then
passed once in training mode and once in evaluation mode; stored buffers after
the 101st pass are used for evaluation. This ordering is disclosed in the plot
and recorded buffers. No optimizer or model fitting occurs here. Training
variance uses the population convention, while PyTorch's running variance uses
its unbiased batch estimator. BN/LN do not guarantee a conditioned Jacobian.

### Actual supervised learning

The visual training experiment uses
$y=\sin(1.5x_0)+0.4x_1+\epsilon$, with Gaussian inputs and Gaussian noise of
standard deviation 0.08. A fixed data generator/split seed **5800** gives 288
training and 96 validation examples. All configurations and model seeds reuse
these exact samples. There is no test set or validation-driven winner selection.

Each model has six hidden ReLU layers, width capped at 32, no normalization,
zero initial biases and an identical initialized linear scalar-output head.
Hidden Gaussian draws are paired. Xavier uses both actual fans in the
rectangular first layer. Small/large strategies multiply the appropriate
fan-in He scale, so they are not just constant rescalings of Xavier's first layer.

Full-batch SGD has no momentum or weight decay. The two illustrative learning
rates are 0.01 and 0.15; quick uses 60 updates and full uses 100. Every checkpoint
records training/validation MSE. Each update records pre-clipping global L2
gradient norm, post-clipping norm and the L2 norm of actual parameter changes.
Epoch 0 is before learning. A separate matched oversized/lr=0.15 configuration
uses global clipping at 1. Clipping controls the gradient input to SGD; it does
not restore vanished directional information.

The first non-finite loss, gradient or parameter update is recorded with its
epoch/stage and terminates the affected run. Missing post-failure metrics are
not fabricated or replaced with a plateau. Extremely large finite values remain
visible. Training diagrams show all seeds for losses and the first seed for
the lower norm panels; the dashboard lets the reader select every training seed.
There is no hyperparameter tuning, convergence guarantee or general ranking.

## Figure map: axes, evidence and limits

| Artifact, under the run directory | Measurement and axes | Inspect this pattern; do not infer this conclusion |
|---|---|---|
| `gifs/forward_propagation.gif` | A1: nodes are successive layers; node color is measured log10(q), fixed across strategies/frames; accompanying chart is depth versus log10(q) | Follow shrinking/growing magnitudes; nodes summarize layers, not a complete wiring diagram or accuracy |
| `interactive/dashboard.html`, Signals | A2/C2: depth versus measured log10(q), activation-gradient RMS and linear-weight loss-gradient RMS; meaningful grid selectors | Compare matched scales and objectives; stable scalar summaries do not establish directional information |
| `static/activation_distributions.png` | A3: activation value versus count at early/middle/final layers; full-batch q, variance and mean annotated | Inspect zero mass and scale drift; panel-specific x ranges are disclosed and cannot be compared by bar width alone |
| `static/initialization_moments.png` | B1: depth versus log10(q/q0), all seeds plus median; dashed idealizations. B3: measured q and variance before/after ReLU | Separate theoretical approximation and finite networks; no trainability claim follows |
| `gifs/xavier_vs_he.gif` | B2: parallel paired Xavier/He networks and depth-versus-log10(q), one shared color scale | Compare the same Gaussian weight realization at two scales; one seed is not an expected-effect estimate |
| `gifs/relu_gate.gif` | B3: value-versus-count histogram and input-versus-output transfer function | Negative values move to zero through an explicit leaky-rectifier interpolation; intermediate frames are not ordinary ReLU. The endpoint halves symmetric second moment, not variance |
| `gifs/backward_gradients.gif` | C1: layer order traversed backward; measured fixed-adjoint RMS and depth profile, shared log color scale | Compare attenuation/amplification from the same terminal RMS; one adjoint is not a Jacobian spectrum |
| `interactive/stability_surface.html` | D: network depth × weight std/He std × log10(q), switchable to log10(input probe RMS) | Rotate/hover near reference scales; this measured propagation surface is not optimization loss or a universal stability frontier |
| `static/normalization_axes.png`, dashboard Normalization | E1–E3: example×feature heatmaps, value/count histograms, batch size versus SD of mean estimates, feature versus normalized anchor | Inspect axes and batch-companion dependence; normalization of a batch is not proof of better training |
| `static/batchnorm_modes.png`, dashboard mode selector | E4: same example×feature batch in train/eval, populated running statistics | Inspect the changed statistics source; buffers are populated, but no trained predictor or deployment performance is established |
| `static/training_comparison.png`, `gifs/training_trajectories.gif`, dashboard Training | F: completed updates versus log10(train/validation MSE); update index versus log10(gradient/update norms), with recorded failures | Compare seed variability, scale×rate interactions and clipping. A short synthetic run cannot rank initializers universally or establish generalization |
| `static/jacobian_spectrum.png`, full dashboard Conditioning | G: descending singular-value rank versus measured log10(singular value), grouped by depth/initializer; linear-map analytical control | Near-zero directions can coexist with a reasonable moment; finite precision and one input prevent a global conditioning guarantee |

The quick dashboard's conditioning panel uses an explicitly analytical control:
identity and rank-one maps with equal mean squared singular value of 1. Full mode
adds actual 12-dimensional neural Jacobians at depths 1/5/12, computed with
autograd followed by SVD. Numerical rank uses width × machine epsilon × largest
singular value. Zero singular values remain gaps; their omission from a log
plot is not evidence of full rank. JSON retains every singular value.

## Source and validation

- [experiments.py](experiments.py): model construction, diagnostics, repeated
  mechanisms, matched supervised learning, Jacobian SVD and measurement cache.
- [visualizations.py](visualizations.py): Matplotlib/Pillow exports, offline
  surface and HTML assembly; uses the bundled Plotly runtime.
- [dashboard.js](dashboard.js): saved-grid selectors and rendering, with no model
  recomputation or network dependencies.
- [main.py](main.py): bounded CLI, timestamped output and hashed manifests.
- [tests/test_visual_lab.py](tests/test_visual_lab.py): formula tolerances,
  dimensions, ReLU moments, normalization axes, populated BN buffers, gradient/
  update semantics, explicit failures, Jacobian oracle, cache and real exports
  in temporary directories. Existing analytical tests remain separate.

Primary theory and version-matched initialization/normalization references are
in [the analytical notes](notes.md#primary-references). Relevant implementation
references: [PyTorch Jacobian](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.functional.jacobian.html),
[BatchNorm](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BatchNorm1d.html),
[gradient clipping](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.clip_grad_norm_.html)
and [Plotly standalone export](https://plotly.com/python/interactive-html-export/).

## Evidence and preview policy

Full HTMLs, JSON measurements, GIF collections and alternate runs remain ignored.
Only a small inspected preview set is deliberately selected for public docs;
the selected files are:

| Public preview | Bytes | Source and learning value |
|---|---:|---|
| [initialization_moments.png](assets/initialization_moments.png) | 154,740 | Full run: depth 50, five seeds; measured moments versus labeled approximation and ReLU moment/variance distinction |
| [normalization_axes.png](assets/normalization_axes.png) | 197,656 | Corrected full-run layout; shared heatmap scale, normalization axes, mean-estimate variability and fixed-anchor dependence |
| [xavier_vs_he.gif](assets/xavier_vs_he.gif) | 1,141,298 | Quick run: seed 58, depth 30; all 31 layer states with paired weights, fixed color scale and synchronized depth chart |

The GIF is an intentional 92,722-byte exception to the 1 MiB review target:
31 layer states preserve the sequential mechanism, and readable labels explain
the common metric/color scale. It is linked rather than inline. Total selected
previews are 1,493,694 bytes, with only the 154,740-byte moments PNG inline in
the topic README. Larger full-run GIFs and the 9–11 MB dashboard HTMLs remain
ignored; the latter embed runtime and measurement grid for offline interaction.
The standalone surface is about 4.9 MB. No CDN substitution or license stripping
was used to reduce size. See the shared [preview policy](../../docs/visual-assets.md).

Author interpretation review remains distinct from numerical tests, export
checks, browser interaction and any future GitHub publication.

## Executed visual study — 2026-11-18

**Executor: Codex. Interpretation candidates: author review required.**
This record concerns the visual protocol above, not the earlier analytical table.
Windows x64; Python 3.11.14; NumPy 2.4.6; PyTorch 2.15.0+cpu;
Matplotlib 3.11.2; Plotly 6.9.0; Pillow 12.3.0. CPU float64, one PyTorch thread.

| Executed command suffix for `main.py` | Measured configuration / outputs | Recorded duration |
|---|---|---:|
| `--mode quick --no-display` | New measurements: depth 30, width 64, seeds 58–60; 12 artifacts | 154.2 s |
| `--mode full --no-display` | New measurements: depth 50, width 64, seeds 58–62; 13 artifacts including measured Jacobian spectra | 242.6 s |
| `--mode quick --no-display` (later export) | Reused the original quick measurements; improved distribution-axis formatting and training detail insets; 12 artifacts | 143.2 s |
| `--mode interactive --no-display` (final export) | Reused quick measurements; final standalone layout and explicit status text; 2 HTMLs | 1.1 s |

These durations are the CLI manifest's measured/export interval, excluding
interpreter/import startup. They are local execution observations, not speed
benchmarks. The cache reuse rows contain no new stochastic measurements.
Timestamped runs and hashes remain in ignored output manifests. The latest quick
export is `quick-20261118T232409706823Z`; the final interactive export is
`interactive-20261118T232751872524Z`. The full run is
`full-20261118T213440703317Z`. Earlier exports remain preserved, including their
original layouts; renderer fixes do not silently revise those manifests.
The final full-measurement HTML-only refresh is
`full-interactive-20261118T233353737231Z`: it reuses exactly the recorded five-seed
full cache, includes the latest renderer/text fixes and passed the same eight
full-mode browser assertions. It does not add or repeat training measurements.

### Hypothesis 1: repeated scale changes alter both signal and sensitivity

The full experiment's first seed (58), depth 50 and width 64 produced:

| Initialization | Final/input second moment | Input fixed-adjoint RMS |
|---|---:|---:|
| Small, 0.25×He std | 8.703310e-62 | 3.881513e-31 |
| Xavier, gain 1 | 1.242178e-16 | 1.466394e-08 |
| He, fan-in/ReLU | 1.398568e-01 | 4.920402e-01 |
| He + LayerNorm | 4.553372e-01 | 1.088428e+00 |
| Large, 1.6×He std | 3.611452e+19 | 7.906782e+09 |

**Interpretation candidate:** the paired realization shows attenuation and
amplification consistent with repeated scale multiplication; He does not force
an exactly constant moment, and LayerNorm changes the forward and backward
measurements differently. The companion B1 plot contains all five seeds rather
than treating this first seed as an expectation. **Limitation:** no learning
occurred in these diagnostics; one output probe does not characterize every
direction, and these endpoints must not be compared as supervised task scores.

### Hypothesis 2: normalization axes change batch dependence

Across 80 repeated batches, feature 0's estimated-mean standard deviation was
0.32723, 0.18067, 0.09204 and 0.03854 for batch sizes 2, 8, 32 and 128.
Changing the anchor's companions changed its BN output; the maximum difference
between the two LN anchor outputs was 0. The inference view used buffers after
101 actual training-mode passes, as specified above.

**Interpretation candidate:** these controlled measurements expose the selected
axes and batch dependence. **Limitation:** they concern synthetic distribution
statistics, not a trained model or an empirical BN/LN quality ranking.

### Hypothesis 3: initialization and learning rate interact during actual learning

Full mode ran five model seeds per cell, 100 full-batch SGD updates on the fixed
288/96 synthetic split. The number of runs stopped for a recorded non-finite
loss, gradient or update was:

| Hidden initialization | SGD 0.01 | SGD 0.15 |
|---|---:|---:|
| Small | 0/5 | 0/5 |
| Xavier | 0/5 | 0/5 |
| He | 0/5 | 4/5 |
| Large | 5/5 | 4/5 |
| Large + global clip 1 | Not run | 0/5 |

All five He/0.01 final validation MSE values lay between 0.11190 and 0.15576;
the single finite large/0.15 run ended at approximately $2.81257\times10^{36}$.
Thus remaining finite is itself insufficient evidence of useful fitting.
These are observed values, not tuned performance estimates. Failure counts,
completed budgets and survivor metrics are different summaries; do not average
only survivors and report that as performance across all seeds.

**Interpretation candidate:** the fixed optimizer does not erase initialization
effects, and the same He construction can behave differently under the two
illustrative rates. Clipping bounded the gradient input and kept the matched
large/rate-0.15 runs finite in this experiment. **Limitation:** this does not show
that clipping recovers lost directions, guarantees convergence, improves real
validation behavior or makes a particular initializer universally preferable.
There was no test set, hyperparameter search or fairness claim across architectures.

### Advanced directionality check

The measured He Jacobian at width 12/depth 12 had output second moment about
0.37659 and numerical rank 2/12 for the selected input and recorded tolerance.
The analytical identity/rank-one control had equal mean squared singular value
but radically different directional retention.

**Interpretation candidate:** a reasonable scalar moment can coexist with
attenuated directions. **Limitation:** ReLU masks create structural rank loss;
full input-output invertibility is not a requirement for every prediction task.
This local rank calculation does not prove poor optimization or generalization.

### Verification and remaining boundaries

Real export checks verified PNG decoding, GIF multiple frames, artifact hashes
and bundled HTML runtimes. The full GIFs contain 51/51/51 propagation frames,
21 rectifier frames and 26 training checkpoints. Representative early/middle/
late frames and the selected previews were inspected visually. The three public
assets were verified as non-ignored Git candidates before Markdown linking.

Chrome 157.0.8103.79 was checked offline at 1440×1000 and selected mobile layout
at 430×900. Thirty final quick/dashboard/standalone assertions and eight full-mode
assertions passed: selectors changed actual measurements, hover exposed values,
3D rotation and zoom worked, normalization modes and clipping changed their data,
SVD curves matched saved values, and the final standalone title/menu/legend
regions did not overlap. Both completed browser audits had zero JavaScript/
console errors and zero external HTTP(S) requests; each observed two Canvas2D
readback-performance warnings. This is Chrome-only local evidence, not a native
GPU benchmark, full accessibility audit, GitHub rendering check or Linux CI.

An initial statistical test assumed an arbitrary narrow oversized network must
overflow; it was corrected to use an actually observed fixed failure case.
Browser harness corrections handled Plotly's event-catching SVG layer, SVG text
access, asynchronous camera updates and the public `plotly_hover` event rather
than assuming a private 2D field exists for 3D. Layout corrections and the final
exports have distinct records; failed checks were not counted as passing.

Final local commands also passed:

```powershell
python scripts/validate_repo.py tests --scope 04-deep-learning/58-weight-initialization
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
node --check 04-deep-learning/58-weight-initialization/dashboard.js
uv pip check --python .local/environments/n09-full-2026-11-13/Scripts/python.exe --offline --no-cache
git diff --check
```

The scoped suite passed **56 tests** (32 analytical + 24 visual) in two isolated
files, with no skips or warnings. Public syntax parsing covered 400 Python files;
internal-link checking covered 1,412 targets in 259 Markdown files. The offline
dependency check reported all 78 installed packages compatible. Tests wrote
real exports only to temporary directories and did not overwrite public previews.
The suite was scoped to Day 58; no full-curriculum, CI or production validation
is claimed. Raw private source SHA-256 remained unchanged, and LinkedIn drafts
were not modified. Root EN/PT and module navigation remain synchronized at
58 implemented studies, 6/10 in Deep Learning, with interpretation review pending.
