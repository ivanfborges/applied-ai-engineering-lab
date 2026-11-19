# Recurrent Memory Visual Lab

[Topic overview](README.md) · [Numerical study](notes.md)

The lab runs on CPU and exports local files. HTMLs bundle Plotly and their
data; opening them needs neither internet nor an application server. Keep
`lab_controls.js` beside the Python sources when regenerating exports.
An installed browser with WebGL support is needed for the 3D view.

## Run the collection

From the repository root, using stable Python 3.11+ in an isolated environment:

```bash
python -m pip install -e ".[dev,deep-learning]"
python 04-deep-learning/60-rnn-lstm-gru/visualizations.py --quick
python 04-deep-learning/60-rnn-lstm-gru/visualizations.py --full
python 04-deep-learning/60-rnn-lstm-gru/visualizations.py --only lstm
python 04-deep-learning/60-rnn-lstm-gru/visualizations.py --only gradients
python 04-deep-learning/60-rnn-lstm-gru/visualizations.py --only hidden3d
python 04-deep-learning/60-rnn-lstm-gru/visualizations.py --quick --matched-budget --only comparison
```

The default is quick mode. Each invocation prints progress and the new output
directory under ignored `outputs/`. Open the desired `.html` by double-clicking
it, or use the browser's Open File command. PNG/GIF files open in ordinary image
viewers. CSV/JSON files preserve measurements without fabricating missing runs.
Choose a reproducible destination with `--output-dir`; it must be new or empty.

`--only lstm`, `--only gru`, `--only rnn` and `--only attention` require no model
training. Empirical gradients, comparison and 3D share one experiment cache.
Their first invocation trains all configurations for the selected mode; later
invocations reuse it. Cache keys bind numerical source hashes, configuration,
Python/NumPy/PyTorch versions and platform. Checkpoint checksums are verified
before reuse. Altered or incomplete caches fail visibly. Choose a different
`--cache-dir` for an intentionally new run; records are not silently replaced.

## What to explore

| Outputs | Question, mechanism and interpretation boundary |
|---|---|
| `01_rnn_unrolled.gif`, `01_rnn_hidden_state.html` | How does shared recurrence transform history? The GIF highlights each read/update and displays the first coordinate; both hidden coordinates appear in the trajectory. Edit scalar inputs, recurrent strength and timestep in HTML. Fixed, untrained two-dimensional tanh cell; not learned recall. |
| `02_gradient_vanishing_exploding.gif`, `02_gradient_analysis.html` | Why can backward signals shrink or grow? The GIF shows constant-derivative products for 0.8, 1 and 1.2. HTML separates adjustable analytical products from actual autograd loss gradients of trained models. Empirical curves are not assumed to follow a constant scalar derivative. |
| `03_lstm_memory_flow.gif`, `03_lstm_gates_interactive.html` | What do retention, writing and exposure control? The GIF uses an explicit manual gate schedule with initial cell value 0.8: retain, close output without erasing memory, write negative content, reset, then write positive content. HTML offers independent manual controls and a separate fixed-weight NumPy recurrence. Neither mechanism view is trained. |
| `04_gru_memory_flow.gif`, `04_gru_gates_interactive.html` | How does one state combine retention and a reset-conditioned candidate? Fixed, untrained scalar weights use the PyTorch reset-after-affine convention. HTML manipulates gates and contrasts a reset-before candidate at the same previous state. The convention selector is a one-step intervention, not retraining or rerolling a different historical sequence. |
| `05_architecture_comparison.html`, `05_training_curves.png`, `05_experiment_results.csv` | Which delayed-bit configurations actually learn? Inspect test accuracy by length, validation BCE histories, parameter counts, actual loss gradients and fit wall time. All values come from CPU training; validation alone selects epochs. Error bars in multi-seed mode are sample standard deviations, not confidence intervals. |
| `06_hidden_state_3d.html` | How do trained representations evolve? Select architecture, length and seed; reveal timesteps with Play, rotate and hover. Four independent probe sequences (two per label) use one PCA basis per checkpoint, including the zero initial states. The title reports variance for PC1/2/3. Bases are not aligned across models and projected class separation is not a generalization test. |
| `07_recurrence_vs_attention.gif`, `07_attention_matrix.html` | What changes when a query accesses earlier values directly? Hand-defined Q/K/V vectors produce actual scaled scores, causal masking, row-normalized weights and a weighted value vector. This is an untrained attention illustration, not learned attention or an attention benchmark. |

Every HTML includes a technical explanation, mathematical mechanism, numeric
readout and limitations. The slider output describes the quantity, and manual
gate controls disappear in computed mode to avoid confusing interventions with
computed values. Plotly supports hover, zoom, trace toggling and local PNG
export. The 3D camera remains stable when revealing timesteps.

## Protocols and evidence boundaries

| Setting | Quick | Full |
|---|---|---|
| Experiment seeds | 160 | 160, 161, 162 |
| Lengths | 10, 20, 50 | 10, 20, 50 |
| Hidden width in equal-width mode | 12 | 16 |
| Train/validation/test counts | 512/128/256 | 1,024/256/512 |
| Epochs; batch size | 12; 64 | 20; 64 |
| Number of trained models | 9 | 27 |

Both use one unidirectional recurrent layer, a scalar logit head, Adam at 0.005,
clipping norm 1, CPU float32 and one PyTorch thread. Deterministic algorithms
are enabled by the CLI. For experiment seed $s$, seeds $10s$ through $10s+4$
control training data, validation data, test data, initialization and minibatch
orders respectively. Probe seed $10s+5$ is independent. Equal-width models
share datasets, minibatch orders and epoch counts at each length; they have
different parameter counts and different internal initialization draws.
Labels are paired across lengths, but noise prefixes are not identical.

All models complete the epoch budget. The earliest minimum validation BCE
checkpoint is restored for each predefined cell/length/seed before its test
evaluation. There is no test-based architecture selection or rate search.
Hidden states reset for independent sequences. This IID diagnostic has no real
timeline; it does not justify random splitting of real time-series windows.

`--matched-budget` chooses each width closest to the parameter count of that
mode's base LSTM (search widths 3–256). In quick mode, RNN/LSTM/GRU widths are
26/12/14 and counts are 807/781/771. This approximates parameters, not operations,
optimizer state or time. It is a distinct protocol, not pooled with equal-width
results. Full matched mode is available but has not been executed here.

The gradient diagnostic is
$\frac{1}{N}\sum_i\|\partial\mathrm{BCE}_i/\partial x_{it}\|_2$ over 32 probe
examples, including both input features. Summing individual losses for autograd
avoids an accidental extra $1/N$ factor. Norms are taken before averaging, so
signed gradients do not cancel. The constant marker feature's sensitivity can
describe off-support perturbations: this is a local derivative, not proof that
a feature carries predictive information. Values below $10^{-18}$ are floored
only for log-axis display; stored measurements retain the actual values.

Fit time includes initialization, training, validation passes and checkpoint
restoration. It excludes probe calculations and rendering; it is neither
inference latency nor a controlled cross-hardware performance comparison.
Loss saturation can make a successful model's loss gradients small. Small
gradients alone do not diagnose vanishing gradients as the cause of poor recall.

## Executed visual study — 2026-11-19

This record is separate from the earlier analytical study in `notes.md`.
It preserves different seeds and quick/full configurations rather than replacing
the earlier measurements.

**Hypothesis:** gating may help delayed recall, but a fixed optimizer/budget can
fail to realize that advantage, and architecture order may vary across seeds.

**Environment:** Python 3.11.14, NumPy 2.4.6, PyTorch 2.15.0+cpu, Matplotlib
3.11.2, Plotly 6.9.0 and Pillow 12.3.0. Windows 10 build 19045, x86-64,
Intel64 Family 6 Model 158 Stepping 9; CPU float32, one PyTorch thread.
No clean installation, Linux CI or GPU execution is claimed.

The quick collection generated all 14 named outputs. Full mode was initially
executed with `--only comparison`, training 27 models and producing comparison
HTML, PNG, CSV and JSON. The quick matched-budget comparison trained nine
additional models. Re-rendering the quick collection subsequently reported a
cache hit and performed no training.
The complete `--full` collection was then exported from its cache, generating
all 14 named artifacts without retraining. The inspected final collection is
under the local directory `outputs/visual-complete-20261119/`; the default CLI
uses a fresh timestamp rather than reusing that historical export directory.

**Measured test accuracy** (percent; full = mean ± sample SD across three seeds):

| Length | Quick RNN | Quick LSTM | Quick GRU | Full RNN | Full LSTM | Full GRU |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100.00 | 100.00 | 100.00 | 99.67 ± 0.56 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| 20 | 99.22 | 48.83 | 51.95 | 99.67 ± 0.56 | 84.31 ± 27.18 | 83.72 ± 28.19 |
| 50 | 53.12 | 46.09 | 51.95 | 66.08 ± 28.61 | 83.79 ± 28.08 | 66.21 ± 29.33 |

The exact first-step sign control attained 100% throughout. Quick mode's
training-majority test accuracy was 51.56%; its final-step sign controls were
49.61%, 44.92% and 46.09% for lengths 10/20/50. Thus the long quick sequence
does not show successful recall for any cell. Full mode produced a mixture
of successful and unsuccessful length-50 runs, with large variation. These
are measurements, not a promise that more training reliably fixes the task.

The nine quick fits accumulated 6.33 seconds inside the fit timer; the 27 full
fits accumulated 70.68 seconds on this machine. These sums exclude imports,
rendering and probes and do not measure end-to-end generation time. The CSV
retains each individual fit time and checkpoint's validation/test metrics.

**Interpretation candidates — author review required:** short recall was
learned under both protocols. The length-50 quick failures and full variation
illustrate dependence on data, initialization, shuffle and budget. The full
means are descriptive summaries of three seeds, not evidence of general LSTM
superiority. Manual gate interventions isolate equations; they are not complete
explanations of trained decisions. Projected probe paths and real local
sensitivities complement, but do not replace, held-out evaluation.

The executed quick matched-budget run attained RNN/LSTM/GRU test accuracies of
100/100/100% at length 10, 100/48.83/46.48% at length 20 and
69.92/46.09/48.05% at length 50. Its hypothesis was the same delayed-recall
comparison under approximate capacity control, using the quick settings above
and widths 26/12/14. **Interpretation candidate — author review required:** the
wider RNN learned more length-50 recall than the equal-width quick RNN in this
single seed. Changing widths changes initialization, optimization and compute;
these values do not isolate a causal effect of parameter count or justify
test-informed tuning. No additional tuning followed these test results.

**Limitations:** synthetic task, one rate, no equal tuning budget search,
three full seeds, unequal capacity in equal-width mode, only four displayed
trajectories per checkpoint and no real forecasting, distribution-shift,
serving-latency or memory experiment. No Transformer was trained. Actual gates
of the large trained models are not decoded as decision explanations.

## Preview curation and verification

The selected public set is the RNN, scalar-gradient and LSTM GIFs, plus one
standalone LSTM HTML. The GIFs answer distinct questions; the HTML is an
ordinary download/link and bundles Plotly to support offline controls.
The remaining outputs, caches, checkpoints, CSV/JSON and browser screenshots
remain ignored. Regenerate them locally; do not link ignored files from public
Markdown.

| Selected asset | Bytes | Purpose |
|---|---:|---|
| `01_rnn_unrolled.gif` | 183,786 | Reused recurrent transitions and evolving coordinates |
| `02_gradient_vanishing_exploding.gif` | 404,239 | Three analytical gradient-product regimes |
| `03_lstm_memory_flow.gif` | 203,113 | Retention, reset, writing and exposure as separate interventions |
| `03_lstm_gates_interactive.html` | 4,877,806 | Standalone offline manual/computed gate explorer |

The selection totals 5,668,944 bytes (about 5.41 MiB); inline README GIFs total
386,899 bytes. The HTML is linked, not embedded, and is below the repository's
6 MiB offline-HTML review target. Selected copies match the inspected exports
byte for byte. All frames of the selected GIFs were decoded successfully.

Numerical tests cover gate limits, GRU/PyTorch agreement, causal masking, PCA
geometry, actual gradient scaling, parameter budgets and checkpoint reloads.
An integration test verifies that cache reuse does not invoke training and
that altered weights fail checksum verification. A real rendering test creates
GIF/HTML in a temporary directory, never over public previews.

The first browser audit loaded seven HTML files with networking disabled,
checked controls against Python calculations and rotated the 3D view by a real
pointer drag. It reported no JavaScript/console errors or remote requests.
A Canvas2D `willReadFrequently` performance advisory appeared; this does not
certify performance or cross-browser support. Screenshot inspection identified
a grid-resize issue, which was corrected before selecting previews.
The final Chromium offline audit passed 36 checks, including plot sizing,
visible scalar extremes and complete SD bars, seed/architecture selection,
computed gate agreement and the active WebGL camera's response to pointer drag.
There were no JavaScript/console errors or remote requests; the Canvas2D
performance advisory remained. The focused Python run passed 37 analytical
and 20 visual tests, with no skips or warnings. These are local checks, not
clean-install CI, GitHub rendering, cross-browser coverage or author approval.

Run focused validation from the repository root:

```bash
python scripts/validate_repo.py tests --scope 04-deep-learning/60-rnn-lstm-gru
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

Additional repeated seeds, real time series, variable-length packing and
truncated-BPTT studies remain future work; a generated picture does not approve
interpretations on the author's behalf.
