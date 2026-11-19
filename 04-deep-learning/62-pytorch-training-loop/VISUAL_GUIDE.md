# Visual learning lab: from a batch to a selected model

The lab records real CPU PyTorch computations, then renders them with
Matplotlib, FuncAnimation/Pillow and standalone Plotly HTML. It requires no
server, downloads, notebook, CUDA or credentials. All datasets are synthetic.
HTML controls replay measured states; they do not execute Python or retrain
the model in the browser.

## Install and run

From the repository root, with stable Python 3.11+:

```bash
python -m pip install -e ".[dev,deep-learning]"
python 04-deep-learning/62-pytorch-training-loop/visualize_training_loop.py --quick
python 04-deep-learning/62-pytorch-training-loop/visualize_training_loop.py --all
python 04-deep-learning/62-pytorch-training-loop/visualize_training_loop.py --interactive
```

Dependency ranges remain in [pyproject.toml](../../pyproject.toml). NumPy,
Matplotlib, Pillow and Plotly are base dependencies; PyTorch is supplied by
`deep-learning`. No separate dependency list or external image renderer is used.

`--quick` generates the dataset GIF plus dataset, BCE, training/validation and
dashboard PNGs. Defaults: 240 observations, 16 epochs, 12 requested frames and
a 25×25 boundary grid. `--all` generates all ten demonstrations: nine PNGs,
four GIFs and three HTMLs. Full/interactive defaults: 600 observations,
40 epochs, 20 requested frames and a 35×35 boundary grid. `--interactive`
alone generates only the three HTMLs; it can accompany `--quick` to add them
to the reduced run. No flags means quick mode.

The CLI prints a fresh `assets/run-<UTC timestamp>/` directory. Each invocation
creates a new directory, leaving previous records intact. Open its HTML files
directly in a browser. Rotate the 3D scene by dragging; select **Zoom** in its
toolbar and drag to change camera distance. Mouse-wheel zoom is enabled in
the export configuration, but its behavior depends on browser/input handling.
Adjust the experiment or destination explicitly:

```bash
python 04-deep-learning/62-pytorch-training-loop/visualize_training_loop.py --all --epochs 30 --samples 400 --frames 16 --batch-size 16 --output-dir 04-deep-learning/62-pytorch-training-loop/outputs/visual-runs
```

Other controls: `--seed`, `--learning-rate` and `--grid-size`. Input limits
reject excessive requests: 80–1,500 observations, 2–120 epochs, batch sizes
4–256, 4–40 requested replay frames and boundary grid sides 12–45. These are
bounded teaching runs, not resource benchmarks. GIFs run at 1–2 frames/second;
the forward/backward diagrams repeat stages to allow reading. Some GIF frame
counts are lower than the requested count because duplicate sampled indices
are removed; Pillow also merges repeated identical diagram frames while
preserving their durations. HTML export rejects files over 12 MiB.

## Follow the learning sequence

Paths below are names within a generated run directory, not public asset links.

| # | Generated files | Question and PyTorch connection |
|---:|---|---|
| 1 | `01_dataset_batches.gif`, `01_dataset_overview.png` | Which observations enter this update? `TensorDataset → DataLoader(shuffle=True) → for X, y, ids in loader`. Read the actual row IDs; only training IDs become active. The tile window follows the batch, rather than squeezing every ID into unreadable text. |
| 2 | `02_forward_pass.gif` | How do two features become one logit? Actual `Linear → ReLU → Linear` tensors from a recorded pre-update sample. Edge sign/thickness encodes measured weights; the matrix lists W₁ values. Highlighting denotes tensor operations, not moving particles. |
| 3 | `03_binary_cross_entropy.png` | Why does confident error cost more? PyTorch evaluates raw-logit BCE for both targets and three labeled predictions; sigmoid probabilities are for interpretation. Gradients in this panel are per-example derivatives. |
| 4 | `04_backward_pass.gif`, `04_gradient_norms.png` | Which operation creates gradients, and which changes weights? `loss.backward()` produces actual parameter gradients and retained activation adjoints. Arrows illustrate chain-rule direction. `optimizer.step()` produces the separately measured parameter delta. |
| 5 | `05_loss_landscape.html`, `05_loss_contour.png` | How does an update move on the actual objective? A one-feature logistic model has exactly two trainable parameters, w and b. Both the surface and trajectory heights evaluate the same fixed 80-row mean BCE. Rotate, zoom, scrub updates and hover for Δw, Δb and update norm. |
| 6 | `06_optimizer_comparison.png`, `06_optimizer_controls.html` | What changes with optimizer, rate and batch size? Choose eight measured SGD/Adam configurations, each starting at w=−1.5, b=1. Plots retain all step lengths and full-objective increases. Equal update counts process different numbers of examples. |
| 7 | `07_boundary_evolution.gif`, `07_final_comparison.png` | How does the represented function change? Every background is `sigmoid(model(fixed_grid))` at an actual checkpoint. The animation includes initialization; the final comparison restores each candidate's minimum-validation-loss epoch. |
| 8 | `08_training_validation.png` | Which checkpoint is selected, and what does a loss gap mean? Separate online training averages, frozen end-of-epoch training BCE and validation BCE. The signed gap uses two frozen-state evaluations. |
| 9 | `09_train_eval_modes.png` | Do eval and inference mode do the same thing? Four combinations on a seeded untrained network containing Dropout: train/eval × normal-grad/inference context. Check output variation and `requires_grad` independently. |
| 10 | `10_training_dashboard.html`, `10_dashboard_overview.png` | Can the panels describe one real update? Scrub the recorded optimizer-step slider. The active batch, pre-update predictions/gradients, parameter delta and post-update boundary/evaluations belong to that same actual event. |

Colors are consistent: blue/coral identify binary classes; blue also denotes
training data/loss, purple denotes gradients or validation, and teal denotes
parameter updates. Axes and legends identify the context. Losses are mean BCE
in nats per example; feature axes use training standard deviations. Surfaces
outside the observed feature support are extrapolations of the model.

## What is recorded, and what is conceptual?

[visual_core.py](visual_core.py) implements the experiments.
[visual_figures.py](visual_figures.py) renders those records;
[visualize_training_loop.py](visualize_training_loop.py) selects the scope and
writes outputs. Rendering does not invent or interpolate model states.

The main task uses the nonlinear generator from the analytical study's
question, with a separate NumPy data protocol: independent Gaussian features,
labels from `x1*x2 + 0.4*x1 - 0.2*x2 + Gaussian noise(scale=0.3) > 0`.
Seed 62 generates data, 63 splits it 60/20/20%, 64 controls training shuffle
and 65 initializes each architecture. Training-only means and population
standard deviations scale all partitions. Batch size 32, Adam rate 0.02 and
no weight decay are fixed defaults. The MLP has 33 parameters (2→8→1 ReLU);
the linear classifier has 3. They share data and batch order, but their initial
functions and capacities differ. The two training examples are not a general
architecture comparison.

Each epoch records the actual shuffled observation IDs, sample-weighted online
loss, frozen train/validation metrics, grid predictions and parameter snapshot.
Gradient norms are recorded before every update. A bounded subset of updates
also records the entire active batch, parameter states before/after, parameter
gradients, update deltas, and first-sample forward/adjoint tensors.
The first sample's logit adjoint is `(p−y)/actual_batch_size`, because backward
uses the batch-mean loss. It differs from the per-example derivative in panel 3.

The dashboard identifies timing explicitly:

- **Before step:** current batch logits/probabilities, batch loss and gradients.
- **After the same step:** new batch predictions/loss, parameter delta, boundary,
  frozen full-training loss and inference-only validation loss.
- **Epoch chart:** the separate aggregate online train loss spans changing
  parameter states. Its comparison with validation needs that qualification.

The replay skips unrecorded steps without smoothing numerical values. Loss
lines connect recorded points as a visual guide, not measurements between
them. The 3D slider moves a marker over the complete prerecorded trajectory;
the complete path remains visible for context. MLP selection uses epochs
1 through E, excluding initialization; strict ties retain the earlier epoch.
Validation selects the architecture and epoch, followed by one selected-model
test evaluation in the manifest. Test data never enters gradient updates,
preprocessing fitting or model selection.

The landscape/optimizer experiment is **separate from the MLP**: 80 one-feature
Gaussian training observations, noisy linear labels, float64 logistic model,
64 optimizer updates and a fixed empirical objective. SGD rates 0.05/0.8 and
Adam rates 0.03/0.3 each use batch sizes 8/80, identical initial parameters and
seeded data ordering (data seed 92, loader seed 93 at the default seed).
The interactive comparison keeps axis ranges fixed across configurations.
`logaddexp` evaluates the surface stably; tests compare
the same objective with actual PyTorch BCE at trajectory coordinates.
Mini-batch steps that increase full-dataset loss are counted and retained.

The forward/backward animations use conceptual highlights/arrows to show
operation order. All displayed activations, adjoints, losses, weight gradients
and parameter deltas come from a real recorded batch. ReLU-zero adjoints are
valid measured zeros. No neural-network loss is projected onto the logistic
surface. The mode demonstration uses a separate untrained Dropout network and
establishes mechanism behavior, not Dropout's generalization benefit.

## Records, preview policy and limitations

Each run contains `manifest.json` (configuration, versions, actual summary
metrics, hypotheses, limitations and output sizes/hashes) and
`numerical_records.json` (data, observation IDs, checkpoint states/predictions,
events, gradient history, logistic paths/surface and mode-probe values).
These are local reproducibility artifacts, not serving models or resumable
optimizer checkpoints. The exporter never reloads untrusted pickled objects.

Run directories remain ignored. Good preview candidates answer complementary
questions: the dataset GIF (batch selection), final boundary comparison PNG
(function capacity), and training/validation PNG (measurement and selection).
Large HTMLs and complete galleries should be regenerated locally. A selected
preview must be inspected and intentionally unignored before README linking.

Known limits: one synthetic seed, small models, no uncertainty analysis,
no real data, no GPU/distributed/mixed-precision experiments, no throughput
benchmark, no live retraining in HTML and no universal optimizer ranking.
Offline Plotly bundles its runtime, so HTML size includes several MiB of code.
Cross-release/device identity is not guaranteed by seeds. A positive loss gap
alone does not diagnose overfitting; no optional overfitting experiment is
claimed here. **Author review required** for all interpretation candidates.

The historical [analytical record](notes.md#executed-study--2026-11-19) uses
1,800 cases, a 2→16→1 MLP, a different data RNG and Adam rate 0.001. Its numbers
must not be pooled with any visual run, including quick versus full runs.

## Validate the implementation

```bash
python scripts/validate_repo.py tests --scope 04-deep-learning/62-pytorch-training-loop
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

[Visual tests](tests/test_visual_lab.py) replay forward/backward tensors,
checkpoint grids, actual SGD deltas and the shared logistic objective. They
check split/order/scaling invariants, validation isolation, uneven batches,
Dropout/grad-mode independence, CLI limits and real PNG/GIF/HTML export into
temporary directories. HTML parser checks reject external resource attributes;
real browser control validation is separate evidence.

## Execution evidence — 2026-11-19

Executed by Codex on Windows x64 with Python 3.11.14, PyTorch 2.15.0+cpu,
NumPy 2.4.6, Matplotlib 3.11.2, Pillow 12.3.0 and Plotly 6.9.0. The CLI used
one Torch numerical thread and deterministic algorithms. No package was
installed or upgraded. **Author review required** for interpretation candidates.

**Prespecified hypothesis:** measured updates change the decision function;
optimizer, learning rate and batch size change the path on a fixed objective.
Rates, dataset, split and budgets were set before inspecting scores.

The default quick run generated four PNGs and one GIF under
`assets/run-20261119T105619377239Z/`. Its 240-row, 16-epoch metrics remain a
separate reduced protocol, not extra observations for the full experiment.
The tile renderer was subsequently enlarged for readability and exercised by
the full run and temporary-directory export tests.

The first default full run generated **nine PNGs, four GIFs and three HTMLs** under
`assets/run-20261119T110031822503Z/`. It used 600 rows (360/120/120), seed 62,
40 epochs, batch size 32, rate 0.02 and the 2→8→1 ReLU MLP. Each model received
480 updates; each training epoch ended with an eight-example tail batch.
Positive counts were 190/63/55 across train/validation/test. No optional
small-training-set overfitting run was performed.

| Candidate | Parameters | Selected epoch | Frozen train BCE | Validation BCE | Validation accuracy |
|---|---:|---:|---:|---:|---:|
| Linear | 3 | 13 | 0.656434 | 0.649121 | 70.00% |
| ReLU MLP | 33 | 35 | 0.335365 | 0.355507 | 84.17% |

Validation selected the MLP; the restored epoch-35 model had test BCE
**0.341427** and accuracy **85.83%** on 120 synthetic cases. Only that candidate
received a final test evaluation. The epoch-40 MLP instead had train BCE
0.332842 and validation BCE 0.361922. The boundary animation and dashboard
retain the chronological path through epoch 40; the final comparison and test
use the selected epoch 35. Model selection does not rewrite training history.

**Interpretation candidate — author review required:** the nonlinear model
fitted this generator better under the specified budget, and the selected
checkpoint differs from the final trained state. This exercise does not identify
the cause of every validation fluctuation or establish a general model ranking.

The separate logistic surface started at mean BCE 1.554043. SGD with rate 0.8
and batch 8 ended at 0.454716, with **19 of 64 updates increasing full-dataset
loss**. Full-batch SGD at the same rate ended at 0.454834 with zero increases.
Adam at rate 0.3/batch 8 had 27 increases; full-batch Adam at that rate had 12.
All these movements are retained. **Interpretation candidate — author review
required:** batch sampling and optimizer dynamics both matter; full-batch
updates are not a general guarantee of monotone descent for every optimizer.
Equal update budgets here process unequal numbers of examples. No rate or
optimizer was declared universally preferable.

Separate `--interactive` runs reproduced byte-identical numerical records to
the full run while refining only the HTML presentation. The final inspected
HTMLs are under `assets/run-20261119T134556939666Z/`. Previous runs remain
ignored and intact. The revisions enabled scroll zoom explicitly, separated
the 3D legend from the colorbar, retained titles during optimizer-menu updates,
fixed comparison axis ranges and moved the optimizer menu away from the title.
Replay controls own their animation promises, handle expected interruption
locally and advance the slider label from actual frame events. Real replay
errors remain visible; no global rejection filter is installed.
The full exporter was also rerun under `assets/run-20261119T112509398484Z/`,
including matching positive/negative weight colors in the forward diagram.
Its numerical record is identical to the initial full run.

All nine full-run PNGs were decoded and visually inspected; representative
initial/middle/final GIF frames were inspected, and real-export tests decode
every frame of their temporary animations. Three intentionally unignored
previews were copied byte-for-byte from the full run:

| Public preview | Bytes | Distinct teaching purpose |
|---|---:|---|
| [Dataset batches](assets/01_dataset_batches.gif) | 641,902 | Actual shuffled IDs and active batches, including the short tail |
| [Final boundary comparison](assets/07_final_comparison.png) | 220,493 | Linear versus nonlinear functions at their selected checkpoints |
| [Training and validation](assets/08_training_validation.png) | 153,429 | Online/frozen loss semantics and validation checkpoint selection |

The selection totals **1,015,824 bytes** (about 0.97 MiB), within the repository
preview targets; all three were confirmed as public Git candidates before
linking. Remaining PNG/GIF/HTML exports and numerical records stay ignored.

The numerical observations above are synthetic, single-seed results. Repeating
this exact configuration in the same environment tests reproducibility rather
than uncertainty across seeds. No production, real-data, GPU, timing-benchmark,
cross-platform or scientific-author-approval claim follows.

Final local verification on the current source passed **26 analytical tests and
17 visual-lab tests**, with no skips or warnings, in two isolated test files.
Repository checks parsed 430 Python files and validated 1,539 internal targets
in 273 Markdown files. Dependency compatibility passed for 78 packages using
the existing interpreter and the offline `uv pip check`; `git diff --check`
also passed. These checks do not constitute a full-curriculum test run or CI.

Chrome 157.0.8103.79, offline at 1560 × 1100, passed **21 browser checks** on
the final HTML exports: real slider/menu/replay controls, camera rotation and
toolbar zoom, all eight optimizer paths, and synchronized dashboard values.
Browser surface and probability-grid values matched the saved numerical records
with maximum absolute error zero. No JavaScript/console errors or HTTP(S)
requests occurred. One Canvas2D readback performance warning remained. Mouse-wheel
zoom was not confirmed; toolbar zoom was exercised. This is one desktop browser,
not mobile, accessibility or cross-browser validation.

The latest full gallery decoded successfully: nine PNGs, all 43 composited
frames across four GIFs, and three HTML files. Export hashes matched their
manifests, and the selected previews matched the full-run files byte-for-byte.
The prior analytical implementation, its dated result, the private raw studies
and the unrelated Day 61 files retained their recorded hashes. Interpretations
remain pending author review; no commit or push was performed.
