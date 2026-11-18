# Interactive regularization lab

The lab connects activation noise, parameter updates, batch statistics and
validation checkpoint selection. It uses deterministic mathematical simulations
and a separate real PyTorch CPU experiment. Every page identifies which kind of
evidence it displays. Interpretation candidates require author review.

## Run locally

From the repository root, in an isolated stable Python 3.11+ environment:

```powershell
python -m pip install -e ".[dev,deep-learning]"
python -m streamlit run 04-deep-learning/57-regularization/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

The dependency list and supported ranges are the shared
[pyproject.toml](../../pyproject.toml): NumPy, scikit-learn, PyTorch,
Matplotlib, Pillow, Plotly, pandas and Streamlit; pytest supplies checks.
There are no topic-specific dependency pins or additional third-party packages.
For CPU wheels, see the shared [validation guidance](../../docs/validation.md).

Open the loopback address printed by Streamlit. Select one of six pages in
**Explore**. Dropout, decay and BatchNorm simulations work immediately. To
populate the other pages, choose a budget and press **Run six training
configurations**. The progress indicator identifies the candidate and epoch.
Form edits do not silently retrain or replace the displayed completed run.
Completed results are cached with at most two parameter combinations.

Default budget: 400 synthetic examples, at most 120 epochs per candidate,
one numeric training thread, grid 41×41. The form caps the budget at 1,000
examples and 200 epochs. Training is serialized across sessions because its
PyTorch RNG/thread controls are process-global. This remains a local educational
tool, not a multi-user serving design.

## Six questions and interactions

| Page | Question | Interaction and evidence |
|---|---|---|
| Generalization and overfitting | When does improved fitting stop helping validation? | Scrub real epochs; compare baseline with a selected candidate, inspect train/validation points, retained boundaries, CE and gap. |
| Dropout | What does a mask change in an actual forward calculation? | Adjust p from 0 to .8 and select a forward pass. A fixed untrained 3-6-5-2 network shows masks, activation values and output probabilities; repeated masks estimate conditional moments. |
| Weight decay and optimization | How do the landscape and update rule differ? | Rotate/zoom a real 3D surface, vary λ from 0 to 1 and inspect 2D paths, norms and reference objective. Four explicitly defined optimizers use the same initial vector. |
| Batch normalization | What changes before/after affine normalization? | Change gamma, beta, epsilon, batch size, feature and observed batch; inspect histogram statistics, running estimates and same-input train/inference outputs. |
| Early stopping | Which checkpoint does the observed validation prefix justify? | Replay a stopping rule on real baseline losses using maximum epochs, patience and min_delta. Scrub the eligible prefix; compare best and last-at-stop boundaries, losses and parameter distance. |
| Integrated comparison | Which declared configuration is selected on validation? | Inspect all train/validation CE and accuracy curves, gaps, retained metrics, effort, norm, and each retained boundary. Test metrics are deliberately absent during dashboard exploration. |

The dashboard uses Plotly for rotation, hover, zoom and panning. The default
bundled favicon avoids a remote icon fetch. No trained state is
invented or interpolated into an animation frame. Boundary colors share a
probability scale from zero to one, and raw feature units are retained on axes.

## Reproduce exports without opening the app

```powershell
python 04-deep-learning/57-regularization/visualizations.py --export
```

This creates a new UTC-stamped subdirectory under the topic's ignored `outputs/`.
For a bounded, explicitly dated run and three seeds:

```powershell
python 04-deep-learning/57-regularization/visualizations.py --export --epochs 120 --n 400 --grid 41 --frames 24 --seeds 157 158 159 --output 04-deep-learning/57-regularization/outputs/visual-2026-11-18
```

Add `--evaluate-test` only for a frozen final selection. It performs one
selected-model test evaluation per seed after validation selection; do not tune
using those outcomes. An existing `manifest.json` prevents accidental overwrite
of a dated run. Choose a new output directory for another execution.

| Artifact | Specific teaching purpose |
|---|---|
| `overfitting_evolution.gif` | Actual baseline/combined boundaries with losses and gap on the same epoch timeline; stopped candidates show the restored state. |
| `decision_boundary_comparison.png` | Retained baseline/combined checkpoints and their complete executed loss/gap histories. |
| `dropout_neurons.gif` | Actual masks and activation values in a fixed untrained network, alongside conditional-moment checks. |
| `dropout_activation_distribution.png` | Noise in a fixed activation versus inference identity; empirical mean and variance against theory. |
| `optimizer_trajectories.gif` | Actual iterative SGD/L2, Adam/L2 and AdamW updates on a declared quadratic. |
| `weight_norm_comparison.png` | Parameter paths, norms and reference J at the final simulated update. |
| `weight_decay_3d.html` | Self-contained rotatable L/J surfaces, analytic minima and paths; λ slider updates the underlying calculations. |
| `batchnorm_transformation.gif` | Raw → normalized → affine activations as batches accumulate, including running statistics and mode differences. |
| `batchnorm_distributions.png` | Final finite-batch distributions and running estimates, with means/variances in legends. |
| `batchnorm_running_statistics.html` | Offline batch-size slider showing stationary population truth, finite-batch variability and running mean/variance. |
| `early_stopping_animation.gif` | Observed baseline losses, current best and patience counter over the eligible prefix. |
| `early_stopping_interactive.html` | Offline epoch scrubber and Play/Pause for the default rule on recorded training; the dashboard provides rule controls. |
| `checkpoint_comparison.png` | Best versus last-at-stop validation checkpoint, real boundaries and linear-parameter distance. |
| `integrated_comparison.png` | Six executed train/validation curves, gaps and retained validation-loss bars. |

Each HTML embeds its Plotly runtime and opens without autoplay. No CDN, external
dataset, pretrained model, paid API or GPU is required. Open HTMLs from the
generated directory in a browser. Python-side dashboard controls require the
local Streamlit process; standalone HTML controls operate within embedded data.

The manifest records versions, configurations, measured results, file sizes,
hashes and decoded image/frame counts. Full histories and snapshots are saved
as `training_seed_<seed>.json`; CSVs contain per-configuration metrics and
mean/sample-standard-deviation of retained validation loss over requested seeds.
With one seed, the CSV standard deviation is undefined (blank), not zero.
Only the first seed supplies the figures and animations. Multiple seeds supply
the variability CSV; the dashboard shows one seed per explicit training request.

## Experimental protocol and interpretation boundaries

The visual study is **not a rerun of the historical 20-input example**. It uses
two moons coordinates directly, with noise .30, no extra noise features,
seed 157 by default, and a 2-64-64-2 ReLU MLP. Train/validation/test proportions
are 60/20/20, stratified. A StandardScaler is fitted on training observations
only. The display grid extent also uses training coordinates only; test points
are omitted from the visualization.

The six candidates reuse identical initial linear parameters and precomputed
batch orders within each seed. AdamW uses lr .003, betas (.9,.999), epsilon
1e-8 and batch size 64. Nonzero decay is .01 on weight matrices only; dropout
is .3 where enabled. BatchNorm uses epsilon 1e-5 and momentum .1, with tracked
statistics. Bias and normalization parameters have zero decay. BatchNorm adds
256 affine parameters. There is no scheduler or clipping.

Both reported train and validation CE use `eval()` and exclude decay penalties.
The non-stopping configurations retain the final epoch. The two stopping
configurations actually stop with patience 15 and min_delta .0001 by default,
and restore the complete best validation state, including BatchNorm buffers.
Strict minima choose the best; significant progress controls patience resets
separately. Snapshot arrays include grid probabilities and linear parameters;
reported norms include linear biases, so they are not the norm of the decay
group alone. Durations include snapshots, evaluation and checkpoint copies.

The replay page uses the baseline's already executed losses and weights; it
cannot know losses after its hypothetical stopping time. Rule changes do not
alter the underlying optimization path. Its checkpoint panels compare the
completed replay rule even while the loss timeline is scrubbed earlier.

The quadratic simulation uses A=[[3,1],[1,1]], c=[1.6,−1] and start [−2,2].
SGD learning rate is .12, with no momentum. Adam/L2 and AdamW both use .08,
betas (.9,.999), epsilon 1e-8, and no AMSGrad. For vanilla SGD, L2 and
multiplicative decay agree under the λ/2 penalty convention. For AdamW, J is
only a reference surface: the adaptive update with decoupled shrinkage does
not implement ordinary gradient descent on J. Surface minima are analytical;
paths are actual iterative updates, independently checked against PyTorch.

BatchNorm simulations use stationary Gaussian features with means [-3,1,5]
and variances [.04,1,6.25]. They initialize running means at zero and variances
at one. The training forward pass uses population-denominator batch variance;
the running update uses its unbiased correction. Inference is evaluated after
the displayed batch's statistics update. Small batches increase variability in
this fixed simulation, not proof of poorer generalization in every model.

No technique is guaranteed to improve held-out performance. Different retained
checkpoint policies and update budgets prevent a general ranking. Even three
seeds do not provide a definitive robustness assessment or evidence about real
document, vision, language or production systems. Repeated validation-guided
exploration can overfit validation; a test split cannot remain a final holdout
once its results guide further choices.

## Executed visual study — 2026-11-18

This is a new Codex execution, distinct from the analytical study in
[notes.md](notes.md#executed-study--2026-11-18). The export completed at
`2026-11-18T19:45:41.302905+00:00` (16:45 in America/Sao_Paulo); subsequent
render-only corrections reused its saved trajectories and checkpoints.

**Hypothesis:** the mechanisms can change decision-boundary evolution and
retained validation loss, without assuming a universal ordering or benefit.

**Configuration:** the protocol above, 120 maximum epochs, 400 observations,
41×41 grid, 24 maximum animation frames, seeds 157/158/159. Each seed used
240 training, 80 validation and 80 test observations. Dependencies were Python
3.11.14, NumPy 2.4.6, scikit-learn 1.9.1, PyTorch 2.15.0+cpu, Matplotlib 3.11.2,
Plotly 6.9.0, Pillow 12.3.0 and Streamlit 1.67.0 on Windows x64.

The actual generation command was:

```powershell
& .local/environments/n09-full-2026-11-13/Scripts/python.exe 04-deep-learning/57-regularization/visualizations.py --export --epochs 120 --n 400 --grid 41 --frames 24 --seeds 157 158 159 --evaluate-test --output 04-deep-learning/57-regularization/outputs/visual-2026-11-18
```

**Results:** selection used retained validation CE only. Each selected model
received one test evaluation, after selection; no configuration was changed
using these test results.

| Seed | Selected configuration | Retained epoch | Epochs run | Validation CE | Final test CE | Final test accuracy |
|---:|---|---:|---:|---:|---:|---:|
| 157 | BatchNorm | 120 | 120 | 0.148540 | 0.150985 | 0.9250 |
| 158 | Early stopping | 54 | 69 | 0.270267 | 0.291607 | 0.8875 |
| 159 | Early stopping | 34 | 49 | 0.146506 | 0.269782 | 0.9125 |

Retained validation CE across the three independently seeded splits and fits:

| Configuration | Mean | Sample standard deviation | Seeds |
|---|---:|---:|---:|
| Baseline | 0.224626 | 0.061266 | 3 |
| Dropout | 0.208887 | 0.080078 | 3 |
| Weight decay | 0.228479 | 0.063507 | 3 |
| BatchNorm | 0.208515 | 0.064607 | 3 |
| Early stopping | 0.188587 | 0.070748 | 3 |
| Combined | 0.192755 | 0.068172 | 3 |

The seed-157 baseline/early-stopping trajectories agree through termination.
Early stopping retained epoch 49 after 64 epochs; combined retained epoch 42
after 57. The static checkpoint comparison uses baseline checkpoints 49 and 64,
with validation CE approximately 0.149 and 0.157 respectively. Linear-parameter
distance includes biases and is approximately 1.5153. The corresponding replay
uses only the eligible prefix; the later full baseline trajectory is excluded
from its selection.

**Interpretation candidates — author review required:** the validation-selected
configuration changed across seeds. Combining the mechanisms was not selected
in these runs. The plotted boundary differences and gaps demonstrate actual
training behavior, but neither boundary smoothness nor three-seed averages
establish a generally superior regularizer.

**Limitations:** three small synthetic splits; illustrative strengths and
learning rate; different checkpoint policies/update budgets; extra BatchNorm
parameters; selection over multiple validation checkpoints. The sample standard
deviation is descriptive, not a confidence interval or a significance test.
Figures show seed 157 only. No real-data, production, calibration or throughput
claim follows. Snapshot JSONs can occupy tens of MiB per seed and remain ignored.

## Validation commands

```powershell
python scripts/validate_repo.py tests --scope 04-deep-learning/57-regularization
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

Tests check NumPy/PyTorch update agreement, complete BatchNorm stream/inference
behavior, conditional dropout moments, stopping-prefix semantics, matched
initialization, real checkpoint probabilities/weights, deterministic repeats,
actual exports in temporary directories and Streamlit interactions.
Rendering tests never overwrite public previews. Browser checks and GitHub
math rendering require separate evidence; tests do not approve interpretations.

Automated checks actually executed with the stable interpreter and pytest 9.2.0:

| Scope | Observed result |
|---|---|
| Initial topic runner | 38 analytical tests + 23 visual tests passed, no skips/warnings. |
| Visual non-rendering recheck after fixed-axis coverage | 23 passed, 1 rendering test deselected; partial check. |
| Targeted dashboard/surface recheck | 2 passed; overlaps the visual file above. |
| Final visual-file runner after export/Play/Pause corrections | 24 passed, no skips/warnings; includes actual exports and AppTest. |
| Full seed-157 saved run | Baseline/ES prefix, all six selected snapshots and byte-for-byte preview copies verified. |
| Dependency consistency | `uv --no-cache pip check --python .local/environments/n09-full-2026-11-13/Scripts/python.exe`: 78 installed packages compatible. |
| Public syntax/local links | 393 Python files parsed; 1,374 links checked in 256 Markdown files. |
| Whitespace | `git diff --check` and a separate scan of new text candidates passed. |

The final file-scoped command was:

```powershell
& .local/environments/n09-full-2026-11-13/Scripts/python.exe scripts/validate_repo.py tests --scope 04-deep-learning/57-regularization/tests/test_visual_lab.py
```

These are overlapping checks, not counts to add together. The final visual file
contains one additional fixed-axis test relative to the initial run. This is
topic-scoped evidence, not a full-curriculum test run or CI result.

### Runtime and browser evidence — 2026-11-18

The Streamlit agent API was checked on a loopback-only test server, using the
running version's OpenAPI schema. Initial load, dropout controls, BatchNorm
controls and decay controls returned `ready` without application exceptions.
A fresh final session also confirmed the updated dropout page at p=.8.

The final Playwright-driver run passed **22 checks** in headless Chrome
157.0.8103.79 at 1440×1000 on Windows. It exercised all six dashboard pages,
the explicit training button, actual checkpoint grids, masks, normalization
controls, replay controls, integrated selection and all three standalone HTMLs.
Real mouse drags rotated/zoomed both 3D surfaces. Slider checks verified surface
heights and analytic minima, preserved camera state, batch-size effects, best
checkpoint/patience, and Play/Pause behavior. The completed run had **no page
JavaScript/console errors and no external page HTTP(S) requests**. Two Plotly
Canvas2D readback-performance warnings remained; no functional failure followed.

Browser inspection led to a local favicon instead of a remote Material icon,
wrapped distribution-plot titles and a separate 3D caption below slider ticks.
Plotly's intentional Pause rejected the interrupted animation with `undefined`;
the local controls consume that cancellation while preserving defined error
reasons. Standalone HTMLs were checked again after this correction. Numerical
tests do not substitute for these interaction checks.

Verification used the installed Playwright driver in a single process.
Browser evidence and diagnostics stayed ignored. WebGL used
software rendering (SwiftShader); this is not GPU performance evidence. The
temporary server was stopped after checking. Linux, other browsers, mobile,
GitHub math rendering and full-curriculum CI were not validated. `ty`/`mypy`
were not installed in the selected environment, so no static type-check result is claimed.

Generated exports are regenerable local artifacts by default. Only explicitly
selected, inspected files copied into `assets/` are public preview candidates;
full HTMLs, snapshots, CSVs, manifests and browser screenshots remain ignored.

Selected public candidates were verified as non-ignored Git candidates and
copied byte-for-byte from the inspected outputs:

| Preview | Bytes | Purpose |
|---|---:|---|
| [Decision boundaries](assets/decision_boundary_comparison.png) | 276,105 | Main comparison: actual 2D predictions, losses and gap under explicit checkpoint policies. |
| [Dropout distribution](assets/dropout_activation_distribution.png) | 171,402 | Masks plus the conditional mean/variance mechanism; this is an untrained-network simulation. |
| [Optimizer trajectories](assets/optimizer_trajectories.gif) | 538,046 | Motion distinguishes update paths and weight norms under coupled and decoupled decay. |

Total 985,553 bytes; only the 276,105-byte boundary PNG is embedded in the topic
README. PNGs use 120 DPI; the selected GIF has 24 sampled frames at 5 fps over
4.8 seconds, with loop enabled. All five generated GIFs were decoded and their
first/middle/final states inspected. This does not constitute author approval
of any scientific interpretation.

[Topic overview](README.md) · [Mathematical notes and historical analytical record](notes.md)
