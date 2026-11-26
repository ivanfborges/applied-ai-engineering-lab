# Interactive pretraining laboratory

The laboratory follows one learning loop: construct self-supervised targets,
observe actual optimization, inspect probabilities and representations, then
test generalization and leakage before discussing analytical scale and compute.
It is a separate experiment from the earlier [analytical study](notes.md).
All training text is synthetic; interpretations require author review.

## Run locally

Dependencies remain in [pyproject.toml](../../pyproject.toml). Use stable
Python 3.11+ and the canonical `deep-learning` extra. NumPy, Matplotlib, Pillow,
Plotly and scikit-learn are already base dependencies. No extra framework,
FFmpeg, cloud account, external API or pretrained weights are required.

```powershell
python -m pip install -e ".[deep-learning,dev]"
python 05-transformers-llms/73-llm-pretraining/visualize.py --all --quick
python 05-transformers-llms/73-llm-pretraining/visualize.py --all
```

The terminal prints the run directory. Open its `index.html` locally. GIFs use
Pillow; all interactive pages load one adjacent `plotly.min.js`. Keep that
script beside the HTMLs when copying a gallery. No runtime requests are needed.

Select one or several sections with repeated `--section` flags:

```powershell
python 05-transformers-llms/73-llm-pretraining/visualize.py --section objectives --quick
python 05-transformers-llms/73-llm-pretraining/visualize.py --section training --section embeddings --quick
python 05-transformers-llms/73-llm-pretraining/visualize.py --section scaling --section compute
```

Scaling, compute and softmax sections require no Transformer training.
Other sections train only the experiments they use. Separate measurement and
rendering explicitly when desired:

```powershell
python 05-transformers-llms/73-llm-pretraining/visualize.py --all --train-only --output-dir 05-transformers-llms/73-llm-pretraining/outputs/my-run
python 05-transformers-llms/73-llm-pretraining/visualize.py --all --render-only --reuse-run 05-transformers-llms/73-llm-pretraining/outputs/my-run --output-dir 05-transformers-llms/73-llm-pretraining/outputs/my-render
```

Saved experiment reuse checks configuration, source fingerprint and PyTorch
version when training would otherwise be requested. A mismatch fails with an
instruction to use a new directory. `--render-only` reads the existing records
and never trains; its provenance lists each source file's hash and **actual**
recorded configuration. Requested CLI defaults do not relabel saved runs.
Use a new directory for a new experiment or a changed analytical assumption.
Historical metrics must not be overwritten. Rendering into an existing output
directory intentionally regenerates figures; use a new render directory when
preserving an earlier presentation matters.

## Configuration and measurements

| Setting | Quick | Default full |
|---|---:|---:|
| Base updates | 60 | 240 |
| Width / attention heads | 24 / 2 | 32 / 2 |
| Blocks / FFN width | 1 / 48 | 2 / 64 |
| Batch size | 16 | 32 |
| Saved snapshots, including initialization | 7 | 13 |
| Seed | 73 | 73 |
| AdamW learning rate / weight decay | 0.005 / 0.01 | 0.005 / 0.01 |
| Dropout | 0 | 0 |
| Nominal interior MLM selection probability | 0.25 | 0.25 |

All of these choices except dropout and weight decay have CLI overrides where
relevant: `--seed`, `--steps`, `--width`, `--heads`, `--layers`, `--ffn`,
`--batch-size`, `--snapshots`, `--learning-rate`, `--masking-probability`.
Use `--device cpu` by default; `--device cuda` is optional and fails explicitly
when CUDA is unavailable. CPU reproducibility is checked in the installed
environment; cross-version or CPU/CUDA identity is not promised.

The exposed model implements post-norm self-attention, ReLU FFNs and learned
positions. It is educational and is not a GPT/BERT replica. Per-head attention
comes from the actual layer-0 snapshot forward pass of each trained model,
using `MultiheadAttention(..., average_attn_weights=False)`. Snapshots also
retain contextual hidden states; input-embedding geometry is plotted separately.

Each run writes `experiments/<name>.json` and a final weight-only `.pt` file.
JSON contains configuration, exact corpus IDs/vocabulary, versions, timestamps,
per-update batch loss/gradient norm/exposure/time, fixed train/validation
evaluations, probabilities, embeddings, hidden states and selected attention.
The final weight file contains no optimizer state and is not a full resumable
training checkpoint. Plots use the saved JSON, not simulated intermediate states.

Elapsed times include snapshot evaluation and synchronization. They are
observations on this environment, not performance comparisons. Input tokens and
supervised targets have separate counters; MLM computes unscored representations
too. The final update is evaluated; validation does not select checkpoints.

## Synthetic data protocol

Two explicitly invented domains contain four subjects, four objects and four
verbs each, with two six-token grammatical templates:

```text
<bos> the SUBJECT VERB OBJECT <eos>
<bos> SUBJECT VERB the OBJECT <eos>
```

Animal-domain verbs are `likes`, `eats`, `finds`, `sees`; technical-domain verbs
are `learns`, `analyzes`, `finds`, `uses`. Every permitted domain combination is
constructed. These are symbolic associations, not assertions about real diets,
capabilities or factual truth. Vocabulary and categories come from that declared
grammar; no learned tokenizer uses validation examples.

Within each domain, `(subject_index + verb_index) % 4 == 0` reserves an entire
subject/verb combination, including all objects and both templates. This yields
192 available training and 64 common validation sentences. Splitting precedes
batching; no overlapping windows or cross-document packing are used. The base
pool contains 96 seeded training sentences, and batches sample with replacement.
Validation tokens occur in training, while complete validation sentences do not.

The fixed probes use observed `the cat eats` and held-out `the cat likes`.
`fish` is one admissible continuation, not the only correct answer. Plotted
reference-token NLL and probability therefore do not define a factual-accuracy
metric. The validation split creates a known context shift, not a comprehensive
test of compositional generalization or natural language.

CLM predicts shifted tokens with an upper triangle blocked in attention; BOS
is an input, EOS is scored. MLM samples interior positions independently, then
forces one selection if none were sampled. Consequently its realized selection
rate exceeds the nominal probability. Every selected token becomes `<mask>`;
unselected targets are `-100`. This is different from BERT's original corruption
mixture. Evaluation uses fixed seeded corruptions of training and validation
sentences, separate from fresh training masks, with the same nominal policy.
MLM NLL is not autoregressive perplexity. The two loss curves are labeled as
different objectives and must not be read as a capability ranking.

## Experiments and controls

| Run | What changes relative to CLM base | What stays controlled |
|---|---|---|
| `clm` | Base 96-sentence diverse pool | Reference configuration |
| `mlm` | Objective, bidirectional visibility and target-selection policy | Architecture, initialization seed, update count and training pool; input/supervision counts differ |
| `large_restricted` | Twice the width/FFN, 24 unique training sentences | Seed, batch size, updates and common validation; **two factors change** |
| `longer` | Twice the updates | Architecture and training pool; spends twice the input exposure |
| `more_unique` | All 192 available unique training sentences | Architecture, updates and token exposure |
| `duplicated` | First 24 base sentences repeated four times | 96 rows, architecture, updates and exposure |
| `single_grammar` | Only the first template, 96 unique sentences | Architecture, updates, exposure and common validation |
| `conflicting` | Object replaced by a subject-role token in every fourth base sentence | 96 rows and matched training exposure; intentionally inconsistent symbolic associations |
| `domain_imbalanced` | 86 animal / 10 technical sentences | 96 rows and matched training exposure |
| `unmasked` | Intentionally remove the causal mask for next-token training | Base weights at initialization, data, optimizer, batch sampling and update budget |

The capacity/diversity scenarios are not isolated causal estimates of model
size. The five quality conditions hold architecture and exposure constant but
change corpus composition and coverage. One seed cannot establish an effect
distribution. No experiment is used to fit a foundation-model scaling law.

For leakage, both arms are evaluated twice: full-sequence teacher forcing under
their respective training visibility, then every target with **only its preceding
prefix**. The latter denies the target to both models. Actual greedy completions
from identical four-token prefixes are also recorded. Unmasked CLM is explicitly
incorrect for ordinary causal training; a lower leaky loss can be misleading.

## Visual map

Names below are generated filenames, not links to ignored public assets. The
local gallery links actual exports after generation.

| Section | Generated files | Question and evidence |
|---|---|---|
| `objectives` | `next_token_prediction.gif` | How does text supply shifted labels? Separate measured one-sentence NumPy bigram fit; training prefixes versus unknown generation future |
| `objectives` | `clm_vs_mlm.gif`, `attention_masks.png` | Which input positions may communicate, and where are prediction targets? Exact analytical masks |
| `objectives` | `masked_token_reconstruction.gif` | Which corrupted positions enter the loss? Actual final MLM predictions on fixed held-out examples |
| `probabilities` | `token_probability_evolution.gif`, `prediction_entropy_evolution.png`, `prediction_distributions.html` | How do two fixed-context distributions change? Actual CLM snapshots, stable vocabulary order and probability scale |
| `softmax` | `softmax_gradient_descent.gif`, `cross_entropy_curve.html`, `cross_entropy_gradient_descent_3d.html` | How do probabilities, NLL and gradients interact? Actual NumPy steps on two free logits; the third stays fixed |
| `embeddings` | `embedding_evolution_2d.gif`, `embedding_evolution_3d.html`, `embedding_cosine_similarity.png` | How do static embeddings move? One PCA fitted to pooled snapshots; original-dimensional cosine and neighbor records |
| `attention` | `attention_evolution.gif`, `attention_heads_comparison.png`, `interactive_attention_explorer.html` | How do actual heads change? Selected trained-layer snapshots, consistent 0–1 scales and checkpoint/head/objective controls |
| `training` | `training_dynamics.png`, `training_dynamics_interactive.html`, `perplexity_vs_loss.html` | What do loss, accuracy, gradient norms and time show? Measured curves plus a separately labeled analytical exponential curve |
| `generalization` | `generalization_comparison.png` | How do capacity, repeated optimization and unique data relate in four measured scenarios? Parameter and exposure budgets shown |
| `leakage` | `future_token_leakage.gif`, `causal_mask_ablation.png` | Does a teacher-forced score survive prefix-only evaluation? Actual paired ablation |
| `scaling` | `scaling_laws_3d.html`, `scaling_laws_heatmap.png`, `fixed_compute_tradeoffs.html` | Where do model capacity and token exposure limit the assumed law? Illustrative coefficients and multiple compute contours |
| `compute` | `compute_optimal_allocation.gif`, `compute_optimal_allocation.html`, `chinchilla_comparison.png` | What allocation minimizes the assumed loss? Derived optimum and a separately sourced historical allocation |
| `compute` | `training_compute_simulator.html` | How do assumed accelerator count, peak throughput, utilization and hourly price affect estimates? Local analytical calculation |
| `data` | `data_quality_comparison.png`, `corpus_diversity.html` | Does equal token exposure imply equal useful information? Actual synthetic conditions and frequency/uniqueness controls |

## Mathematical interpretation

Embedding PCA uses a single pooled fit over checkpoint × token rows, then
projects every checkpoint with that same basis and mean. Two-dimensional and
three-dimensional explained variances are reported in the views and
`embedding_geometry.json`. Nearest neighbors and cosines use the original
embedding dimensions. Category colors are predefined; neither a cluster nor
its absence establishes semantic quality. Static embeddings also differ from
contextual hidden representations. Attention weights alone are not causal
feature importance or a complete interpretation of model behavior.

For the three-token optimization slice, the observed target is index zero:

$$
L=-\log p_0,\qquad \nabla_z L=p-y,\qquad z_{0:2}\leftarrow z_{0:2}-\eta\nabla_{z_{0:2}}L.
$$

The third logit remains zero. The surface and trajectory therefore share
exactly the same objective and free coordinates. It is not the Transformer's
high-dimensional loss landscape. The existing NumPy bigram implementation
uses the same categorical gradient, accumulating repeated context rows.

Scaling coefficients here are explicitly illustrative:

$$
L(N,D)=E+\frac{A}{(N/10^9)^\alpha}+\frac{B}{(D/10^9)^\beta}.
$$

Defaults are $E=1$, $A=B=4$, $\alpha=0.34$, $\beta=0.28$. These are **not fitted
paper coefficients**. N and D are normalized to billions for these chosen
coefficients. CLI flags `--E`, `--A`, `--B`, `--alpha`, `--beta` change export
assumptions; HTML controls change them locally. Under $C=6ND$, let
$c=C/(6\times10^{18})$, $n=N/10^9$, $d=D/10^9$. Differentiation gives

$$
n^*=\left(\frac{\alpha A}{\beta B}c^\beta\right)^{1/(\alpha+\beta)},
\qquad d^*=c/n^*.
$$

The minimum is checked against a dense numerical grid. No arbitrary point is
placed at the apparent bottom of a plot. The historical Gopher/Chinchilla panel
uses [Hoffmann et al., Table 1](https://arxiv.org/html/2203.15556v1):
280B / 300B and 70B / 1.4T parameters/training tokens respectively. Those counts
follow the paper's parameter convention; they are not this lab's measurements.
Approximately twenty tokens per parameter is a historical allocation, not a
universal optimum or deployment-cost prescription.

The simulator calculates

$$
C=6ND,\quad F_{\mathrm{effective}}=kF_{\mathrm{peak}}u,\quad
t=C/F_{\mathrm{effective}},\quad H=kt/3600,\quad \mathrm{cost}=Hr.
$$

Peak TFLOP/s is converted to FLOP/s; $u$ is assumed utilization and $r$ is a
hypothetical hourly rate. Constant utilization assumes ideal parallel scaling.
Attention/context computation, communication, activation recomputation,
optimizer work, memory feasibility, checkpointing and downtime need separate
assessment. Sparse models need active-parameter accounting. No hardware,
provider billing or lifetime inference-cost benchmark is performed.

## Public selection and validation

All exports, metrics and weight files stay under ignored `outputs/`. The gallery
and shared JavaScript are regenerable artifacts; weights and full snapshots
are intermediates for inspection. No preview is automatically unignored or
committed. Recommended candidates after author review are probability evolution,
CLM/MLM visibility, embedding evolution and compute allocation. Prefer a few
small inspected files; a local HTML gallery is not rendered interactively by
GitHub Markdown.

```powershell
python scripts/validate_repo.py tests --scope 05-transformers-llms/73-llm-pretraining
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

Tests cover target shifts, actual mask semantics and prefix invariance, MLM
target exclusion, numerical stability and gradients, disjoint data, finite
reproducible CPU runs, pooled PCA and unit-correct budgets/derived optima.
Real GIF/PNG/HTML export tests use temporary directories. No test imposes
monotone training loss or a predetermined validation ranking.

## Execution record — 2026-11-25

These are new visual-phase measurements; the original analytical record in
`notes.md` remains unchanged. **Interpretation candidates: author review required.**

**Hypotheses:** measured self-supervised learning can fit symbolic training
patterns; held-out contexts can remain difficult; full-sequence unmasked CLM
scores may overstate legitimate prefix-only prediction. Data-condition and
capacity scenarios explore their specified trade-offs without fixing a desired
ranking in advance.

Both quick and full suites executed all ten runs on Windows with Python 3.11.14,
NumPy 2.4.6, PyTorch 2.15.0+cpu, Matplotlib 3.11.2, Plotly 6.9.0,
scikit-learn 1.9.1 and Pillow 12.3.0. Training used one CPU thread and the
configuration table above. No CUDA run was performed.

The full experiment records span `2026-11-25T23:09:22.606950+00:00` through
`2026-11-25T23:09:40.315522+00:00`. Their immutable source directory is
`outputs/visual-full-20261125-01/experiments/`. The earlier quick records are
separate under `outputs/visual-quick-20261125-01/experiments/`.

| Full run | Parameters | Unique sentences | Processed input tokens | Train NLL | Validation NLL | Validation token accuracy | Run seconds, incl. diagnostics |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLM base | 19,035 | 96 | 38,400 | 0.982426 | 1.762752 | 0.462500 | 1.695078 |
| MLM | 19,035 | 96 | 46,080 | 0.962898 | 1.650175 | 0.340909 | 1.829056 |
| Larger / restricted | 70,811 | 24 | 38,400 | 0.670101 | 3.018679 | 0.450000 | 1.863403 |
| More optimization | 19,035 | 96 | 76,800 | 0.958689 | 1.897167 | 0.459375 | 3.540842 |
| More unique examples | 19,035 | 192 | 38,400 | 1.087341 | 1.616053 | 0.462500 | 1.730373 |
| Duplicated | 19,035 | 24 | 38,400 | 0.668181 | 2.705015 | 0.443750 | 1.567749 |
| Single grammar | 19,035 | 96 | 38,400 | 0.946963 | 3.659030 | 0.346875 | 1.584934 |
| Conflicting roles | 19,035 | 94 | 38,400 | 1.003110 | 1.904548 | 0.450000 | 1.603023 |
| Domain imbalance | 19,035 | 96 | 38,400 | 0.961698 | 2.184174 | 0.456250 | 1.597827 |
| Incorrect unmasked CLM | 19,035 | 96 | 38,400 | 0.002817 | 0.033057 | 0.990625 | 1.901729 |

The unmasked row uses a deliberately invalid full-sequence next-token scoring
protocol; its NLL is not legitimate autoregressive likelihood. MLM uses fixed
selected-token reconstruction targets. These two rows are explicitly different
prediction problems from ordinary masked CLM. All CLM updates supervised
one target per input position. MLM supervised 10,134 targets across 46,080 input
tokens, with an observed eligible-position selection fraction of
$10134/(240\times32\times4)\approx0.3299$.

| Full ablation evaluation on the same 64 validation sentences | Causal model | Incorrect unmasked model |
|---|---:|---:|
| Full-sequence teacher-forced accuracy | 0.462500 | 0.990625 |
| Prefix-only accuracy | 0.462500 | 0.303125 |
| Prefix-only NLL | 1.762752 | 3.913205 |
| Prefix-only autoregressive perplexity | 5.828455 | 50.059130 |

Actual greedy continuations from the observed prefix were
`<bos> the cat eats seeds <eos>` (causal) and
`<bos> the cat eats fish <eos>` (unmasked); both produced
`<bos> the cat likes fish <eos>` from the held-out prefix. These individual
examples are compatible with the invented grammar but do not overturn the
aggregate prefix-only result or establish reliable factual generation.

The shared PCA projection of the full CLM input-embedding snapshots explained
**12.5195%, 11.0206% and 9.8531%** on its first three components: **23.5402%**
in 2D and **33.3932%** in 3D. In original dimensions, the three nearest input
embedding neighbors of `cat` changed from `fish`, `finds`, `learns` to `fish`,
`finds`, `dog`. The `cat`–`fish` cosine changed from 0.388871 to 0.396872.
These individual observations do not establish category separation or semantic
quality; most embedding variance is outside the displayed coordinates.

The quick ablation separately recorded causal full/prefix NLL
1.577636 / 1.577636 and unmasked full/prefix NLL 0.450068 / 2.674059.
Do not mix quick and full measurements into a single trajectory.

**Interpretation candidates:** in this full configuration, the unmasked model's
very strong teacher-forced scores did not transfer to equivalent prefix-only
evaluation. Additional optimization reduced base training NLL while increasing
validation NLL at the final checkpoint. More unique data produced lower final
validation NLL at the same exposure in this comparison. Restricted, duplicated,
single-grammar and imbalanced conditions each showed their recorded gaps; their
composition and coverage differ, so no universal effect size or isolated causal
attribution follows. Original-dimensional neighbors and pooled PCA must be
inspected as measurements, without assuming meaningful semantic clusters.

**Limits:** single-seed symbolic data, no independent test set, no real-text or
downstream task evaluation, no foundation-model scale sweep, no hardware or
cost benchmark. Test success and successful exports do not approve these
interpretations or publication on the author's behalf.

### Local verification and final presentation

The final inspected gallery is `outputs/visual-reviewed-20261125-01/index.html`.
It reuses the full suite's original experiment JSON; presentation changes
(Plotly axis titles, annotation spacing and cosine labels/color bars) did not
retrain the models or replace scientific measurements. The full initial render
provenance and the subsequent selected-section provenance remain in that
directory. The local gallery contains **9 GIFs, 10 PNGs and 12 interactive HTMLs**.

Actual checks: **75 scoped tests passed** across the three test files, with no
skips or warnings. The two real-export tests were repeated after the last
presentation changes and passed again. Repository parsing covered **536 Python
files**; internal-link checking covered **1,983 links in 329 files**. Installed
dependency compatibility passed for **89 packages** using explicit-interpreter
offline `uv pip check`. `git diff --check` had no errors (Git reported its normal
LF/CRLF conversion notice for the module index).

All GIF frames decoded, PNGs verified and **72 relative gallery/HTML references**
resolved locally. Initial/intermediate/final GIF frames and static-plot contact
sheets were inspected. The final 12 HTMLs passed **94 browser assertions** in
offline Headless Chrome 158 on Windows with SwiftShader: actual controls,
calculated budgets/optima, attention row normalization, meaningful 3D axis
labels, camera drags, nonblank 3D canvases and visible charts. There were zero
JavaScript/console errors and zero HTTP(S) requests. Three Canvas2D readback
performance warnings occurred during inspection. This is local desktop-flow
evidence, not GPU acceleration, cross-browser, mobile, accessibility, GitHub
Markdown rendering or scientific approval.

Recommended preview candidates remain ignored: `token_probability_evolution.gif`
(about 501 KiB), `clm_vs_mlm.gif` (154 KiB), `embedding_evolution_2d.gif`
(349 KiB), and `compute_optimal_allocation.gif` (310 KiB). The interactive scaling
surface is useful in the local gallery; its adjacent JavaScript dependency makes
it a different delivery choice from a small GitHub image. No preview selection,
author interpretation approval, commit or push is inferred from these checks.
