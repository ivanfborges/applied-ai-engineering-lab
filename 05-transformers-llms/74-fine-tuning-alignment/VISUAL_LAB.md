# Fine-tuning and alignment visual learning lab

[Topic overview](README.md) · [Theory and analytical record](notes.md) · [References](references.md)

Fourteen demonstrations separate analytical calculations, actual synthetic
training, theoretical storage estimates and explicitly labeled illustrations.
The laboratory uses local CPU operations. It downloads no model or dataset,
calls no service and requires neither credentials nor a GPU. All interpretations
remain **author review required**.

## Run and inspect

Use stable Python 3.11+ from the repository root. Dependencies are already
declared in [pyproject.toml](../../pyproject.toml); no additional framework or
topic requirements file is needed.

Windows PowerShell:

```powershell
python -m pip install -e ".[deep-learning,dev]"
python 05-transformers-llms/74-fine-tuning-alignment/visualize_fine_tuning.py --list
python 05-transformers-llms/74-fine-tuning-alignment/visualize_fine_tuning.py --all --quick
python 05-transformers-llms/74-fine-tuning-alignment/visualize_fine_tuning.py --demo dpo --quick
python scripts/validate_repo.py tests --scope 05-transformers-llms/74-fine-tuning-alignment
```

To use this checkout's existing stable environment, replace `python` with
`& .local/environments/n09-full-2026-11-13/Scripts/python.exe` in PowerShell.
The installation command describes environment preparation; the visual work
verified the existing installation rather than installing or upgrading packages.

Linux shell:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[deep-learning,dev]'
python 05-transformers-llms/74-fine-tuning-alignment/visualize_fine_tuning.py --all --quick
```

Linux commands are provided for reproducibility; execution evidence here is
Windows only. No desktop GUI is required for generation (`Agg` + Pillow).

The script prints a new run directory under ignored `outputs/`. Open its
`index.html` directly. Keep the adjacent `plotly.min.js` with the HTML files;
the directory is an offline bundle, not a collection of individually standalone
HTMLs. One shared runtime avoids duplicating several megabytes fourteen times.
No CDN or HTTP server is required.

Each interactive page offers configuration selection, recorded/calculated
checkpoint selection, playback and reset. Plotly supplies hover, toolbar,
legend toggles and applicable rotation/zoom/pan controls. Some analytical
configuration views have one frame: playback cannot invent missing dynamics.
Three-dimensional views complement 2D contours, bars, heatmaps and curves.

`--quick` alone and `--all --quick` both select all demos. Repeat `--demo` to
combine a subset; `--seed 74` selects a different reproducible experiment.
Use `--output-dir <new-directory>` for an explicit destination. Existing output
directories are refused to protect historical runs.

To regenerate presentations from unchanged records:

```powershell
python 05-transformers-llms/74-fine-tuning-alignment/visualize_fine_tuning.py --demo dpo --reuse-run 05-transformers-llms/74-fine-tuning-alignment/outputs/visual-quick-20261125-02 --output-dir 05-transformers-llms/74-fine-tuning-alignment/outputs/dpo-new-render
```

The reused configuration and original measurement timestamps are retained;
digests are verified. A new manifest records current rendering sources and the
measurement source separately. Choose a different destination for every rerun.

## Budget and provenance

| Setting | Quick | Default/full |
|---|---:|---:|
| General optimization updates | 80 | 240 |
| Exact categorical policy / population DPO updates | 320 | 960 |
| GIF frames per animation | 10 | 25 |
| Matrix width | 16 | 24 |
| Base sample setting | 128 | 512 |
| Instruction/forgetting seeds | 1 | 3 |

Full mode repeats instruction and forgetting experiments with seed offsets
10,000 and 20,000; the JSON preserves final scores from all repetitions.
Plots and animation trajectories show the primary seed, rather than silently
pooling incompatible coordinates. Other demos retain a single seed in both
modes. Full mode is implemented but was not executed in this handoff.

Models initialize deterministically inside a restored PyTorch RNG context and
use one CPU thread. Data generators have independent local seeds; numerical
reproducibility across versions or hardware is not guaranteed. Relevant
float64 NumPy/matrix calculations and float32 neural training are distinguished
below. Evaluation runs under `no_grad`/inference mode and never selects an
update, learning rate or checkpoint.

`experiments/<demo>.json` preserves configurations, every recorded update,
measured arrays, hypothesis, timestamp, versions and limitations. `manifest.json`
contains source/measurement/artifact SHA256 hashes and rendering metadata.
`--reuse-run` creates no new training measurements. Generated JSON, HTML, PNG,
GIF, intermediate files and the runtime stay ignored. There are no selected
public previews or links to ignored assets in public Markdown.

## Demo and artifact index

Use `--demo <name>` with any name below. Filenames are relative to the printed
run directory; these are generated artifacts, not public Markdown links.

| Demo | Question | Generated files |
|---|---|---|
| `landscape` | Where does adaptation start? | `01_pretraining_vs_fine_tuning.gif`, `01_parameter_landscape_3d.html`, `01_parameter_landscape_contour.png` |
| `sft` | How does demonstration likelihood change probabilities? | `02_sft_probability_evolution.gif`, `02_cross_entropy_landscape.png`, `02_token_loss_masking.png`, `02_sft_probabilities.html` |
| `instruction` | Does specialization transfer across numerical task rules? | `03_instruction_tuning_tasks.gif`, `03_multitask_performance_heatmap.png`, `03_task_training_curves.png`, `03_instruction_generalization.html` |
| `reward` | How do comparisons become reward scores? | `04_reward_model_training.gif`, `04_reward_surface_3d.html`, `04_bradley_terry_probability.png`, `04_preference_noise_comparison.png` |
| `rlhf` | How does a KL coefficient change the finite-policy optimum? | `05_rlhf_policy_evolution.gif`, `05_rlhf_objective_surface_3d.html`, `05_kl_regularization_comparison.png` |
| `dpo` | Which probabilities and margins does DPO optimize? | `06_dpo_training_dynamics.gif`, `06_dpo_loss_surface_3d.html`, `06_dpo_loss_and_gradients.png`, `06_relative_vs_absolute_probability.png` |
| `comparison` | When do reward optimization and DPO share an optimum? | `07_rlhf_vs_dpo.gif`, `07_alignment_metrics_comparison.png`, `07_alignment_comparison.html` |
| `lora` | What can a learned low-rank update reconstruct? | `08_lora_matrix_evolution.gif`, `08_lora_rank_comparison.png`, `08_lora_reconstruction_3d.html`, `08_singular_value_spectrum.png` |
| `geometry` | Which matrix optima does a rank restriction exclude? | `09_full_vs_lora_optimization.gif`, `09_lora_constraint_geometry_3d.html` |
| `efficiency` | What exactly is being counted? | `10_parameter_efficiency.png`, `10_parameter_efficiency_3d.html`, `10_adapter_bottleneck.png` |
| `qlora` | How do low-bit weights differ from their originals? | `11_quantization_animation.gif`, `11_quantization_error.png`, `11_quantization_storage_comparison.png`, `11_quantized_base_with_lora.png`, `11_quantization_explorer.html` |
| `forgetting` | What happens to an earlier conflicting task? | `12_catastrophic_forgetting.gif`, `12_capability_retention_comparison.png`, `12_forgetting_boundaries.html` |
| `hacking` | Can higher proxy reward hide lower intended utility? | `13_reward_hacking.gif`, `13_proxy_vs_true_utility_3d.html`, `13_reward_hacking_metrics.png` |
| `pipeline` | How do objectives, frozen parameters and held-out metrics evolve? | `14_complete_post_training_pipeline.gif`, `14_post_training_dashboard.html`, `14_final_evaluation_summary.png` |

Common semantics: blue denotes a base/reference, green adaptation or a chosen
outcome, red a rejected outcome, orange loss/error and purple preference
optimization. Legends, category labels, line styles and marker shapes provide
additional distinctions. Matrix color bars show actual signed values;
probability plots stay within 0–1. Color meaning is always local to its legend.

## How to read the experiments

### 01 — Analytical parameter landscapes

Two positive-definite quadratics use distinct minima and Hessians. Explicit
gradient descent first approaches the base minimum, then starts adaptation at
that exact endpoint. Axes are two parameter coordinates and objective height;
2D contours show the same coordinates. Stars/endpoint markers identify minima
and stage boundaries. A two-dimensional quadratic is an educational abstraction,
not a projection of measured LLM weights or an empirical neural loss surface.
Practical implication: adaptation initialization and optimization objective are
separate choices.

### 02 — SFT and token loss accounting

A PyTorch `Linear(2,4)` response selector learns one demonstration with SGD,
learning rate 0.12. Every update records logits, softmax, cross-entropy and
parameter-gradient norm; bars highlight the chosen response and curves use
actual update indices. The separate $-\log(p)$ plot is analytical.

The token fixture shows true-prefix input, once-shifted next-token targets,
prompt/response/padding masks and measured cross-entropy contributions. Response
`4` and EOS contribute; prompt and padding do not. Those logits are a fixed
arithmetic example, not an autoregressive model. Teacher forcing does not test
generation. Practical implication: a chat template and loss mask must agree
before interpreting training loss.

### 03 — Symbolic instruction categories

A 7→16→2 tanh MLP first learns the common rule $x_0>0$, then either adapts only
to category 1 or jointly to four task-specific threshold rules: $x_0>0$,
$x_1>0$, $x_0+x_1>0$ and $x_0-x_1>0$. Task one-hot IDs, two Gaussian features
and an independent scalar variant ID form the inputs. Training variants are
0/0.5; evaluation variants 1/1.5 use independently generated samples.
Base/adaptation use Adam at 0.02/0.015 with fixed budgets.

Heatmaps and curves show task-specific held-out accuracy. Animation coordinates
are hidden states of the same 32 evaluation probes under one pooled, centered
PCA basis across models and updates; the retained 2D variance is displayed.
Markers denote supplied categories, not discovered semantic clusters. Task
names are numerical labels: no extraction, summarization, QA or unseen natural
instruction following is established. Practical implication: report each task
and held-out variant separately; more tasks need not improve every task.

### 04 — Reward modeling and inconsistent annotations

Response features lie in [0,1] with synthetic utility weights
$(2,0.4,1.5,1.2,-0.8)$ for factuality/conciseness/relevance/compliance/verbosity.
Labels are Bernoulli draws from their Bradley–Terry probability; extra independent
flips use rates 0/0.2/0.4. A bias-free linear reward model minimizes pairwise
logistic cross-entropy using Adam 0.04. A separate evaluation feature sample
measures expected BT cross-entropy against the known population, not noisy
training labels.

Surface axes are factuality feature, relevance feature and predicted reward;
other feature values are fixed at 0.5. Checkpoint metadata reports both probe
response scores and their implied preference probability. The sigmoid plot
maps reward difference to preference probability. No feature is measured from
real text and no annotator is human. Practical implication: reward scores
approximate the supplied preference signal rather than universal truth.

### 05 — Exact finite-policy reward and KL

Three actions use reference $(0.6,0.3,0.1)$ and supplied rewards $(0,0.6,1)$.
Exponentiated-gradient/mirror ascent computes exact expectations for
$J(p)=p^\top r-\beta D_{KL}(p\Vert q)$, with beta 0/0.2/0.5/2 and step size 0.2.
For beta > 0 the optimum is $p^*\propto q\exp(r/\beta)$; beta zero concentrates
on maximizing actions. The simplex surface uses $p_1,p_2$ and height $J$,
with $p_3=1-p_1-p_2$ and invalid cells omitted. Markers show reference/analytic
optimum and the line is the actual optimization path. This is neither PPO nor
sampled policy-gradient RLHF. Practical implication: regularization changes the
target distribution and should be measured, not inferred from the name RLHF.

### 06 — DPO and absolute response probability

NumPy gradient descent uses the tested analytical DPO derivative for one
chosen/rejected pair within three categories, starting at reference
$(0.45,0.35,0.2)$. Independent runs use beta 0.1/0.5/2, learning rate 0.2.
The surface axes are chosen logit, rejected logit and DPO loss with the third
logit fixed. Checkpoints overlay the actual path. Analytical curves show
loss, absolute margin derivative and $\sigma(\beta m)$ over adjusted margin.

$\sigma(\beta m)$ is an implied preference probability; the softmax bars are
response probabilities. The constructed valid distributions $(0.4,0.3,0.3)$
and $(0.3,0.1,0.6)$ demonstrate a larger pair ratio while both pair probabilities
fall. This counterexample is arithmetic, not a training trace. Practical
implication: pair loss alone cannot establish absolute response quality.

### 07 — Shared-assumption comparison

The reward demo here reuses Demo 05's response space, reference, rewards and
beta 0.5. DPO uses exact population BT target probabilities over all three
pairs, optimizing expected binary cross-entropy with float64 Adam 0.04.
Both methods start at the same reference; probability, expected reward, KL
and margins can be compared. Their objective values have different definitions.
Agreement with the analytic optimum relies on known rewards, complete pair
coverage and exact population assumptions. These policy optimizations do not
consume Demo 04's fitted reward model or real finite human labels. Practical
implication: theoretical connection does not imply universal practical equivalence
or a fair speed/quality ranking under different optimizers.

### 08 — Learned LoRA matrices and SVD

Rank-2 and full-rank target updates share a frozen random square base. Ranks
1/2/4/8/16 minimize squared Frobenius reconstruction error using float64 Adam
0.04, random A and zero B, with alpha/rank = 1. Synchronized heatmaps show
W0, A, B, BA, effective weights, target and residual. Row/column axes are matrix
indices, not token positions. The 3D surfaces use column, row and update entry;
color/height ranges stay fixed within each target scenario.

Learned final errors are compared with truncated SVD's optimal Frobenius error,
and spectra are calculated from the actual targets. Rank bounds do not ensure
finite-budget convergence. This is matrix reconstruction, not downstream model
quality. Practical implication: rank is a capacity constraint whose usefulness
depends on the adaptation target.

### 09 — Geometry of bounded-rank matrices

Projected gradient descent on a known 2×2 target compares rank 1 and rank 2.
The 3D surface is the fixed-$W_{10}=0.4$ slice
$W_{11}=0.4W_{01}/W_{00}$ for positive $W_{00}$; axes are $W_{00},W_{01},W_{11}$.
Trajectories are projections of all four changing entries, so they need not
lie on that fixed-coordinate slice. The rank-1 SVD solution supplies a known
constrained optimum. This projected algorithm illustrates constraints; it is
not Demo 08's factorized LoRA optimizer. Bounded-rank matrices are a nonlinear
family, not a fixed linear subspace. Practical implication: a three-coordinate
picture cannot certify the feasibility of a full high-dimensional update.

### 10 — Parameter counts and bottleneck computation

The implemented residual adapter computes $h+W_{up}\,ReLU(W_{down}h+b_{down})+b_{up}$.
The 8→2→8 fixture includes 42 trainable entries and shows its actual activation
vectors. Square-layer comparisons use $d^2$ full weights, $2rd$ LoRA entries,
and $2db+b+d$ adapter entries including biases. Full/LoRA counts exclude bias
by explicit convention; all counts scale with adapted-layer number.

Log curves compare counts and theoretical FP16 trainable checkpoints; the
surface axes are dimension, rank and $100(2r/d)$ percent. Configurations with
too many LoRA entries visibly exceed 100%. These are exact counts and assumed
storage, not measured memory, runtime or GPU demand. Practical implication:
include base copies, optimizer state and activations in a real resource estimate.

### 11 — Quantization and reconstructed-base adaptation

Original FP32 Gaussian weights are compared with FP16 round-trip values and
symmetric uniform 8/4-bit reconstruction, using one max-absolute scale and
255/15 signed levels (one bit pattern unused). Errors are measured against
the original FP32 array. Storage estimates assume packed codes plus one FP32
scale; actual simulated codes are int16 arrays, not packed memory measurements.

The mapping GIF interpolates display positions between original and discrete
values: it is explicitly an illustration, not training. Histograms show value,
count and reconstruction error. Matrix row/column heatmaps compare original,
reconstructed and adapted weights. A float64 rank-2 LoRA update trains against
a frozen reconstructed 4-bit base using Adam 0.02; it can reduce reconstruction
error. This is not NF4, double quantization, paged optimization or real QLoRA
kernels. Practical implication: weight storage estimates and total training
memory answer different questions.

### 12 — Sequential learning and retention

A 2→12→2 tanh MLP learns Task A ($x_0>0$) with Adam 0.03. It then adapts to
Task B ($x_1>0$) with Adam 0.02 using full updates, rank-1 head-only LoRA,
or full updates with a 50/50 A/B training-loss replay mixture. Both tasks have
independent held-out inputs; evaluation samples are not replayed.

Decision-boundary axes are original $x_0,x_1$; color denotes class-1 probability,
and dashed/dotted lines show the two true boundaries. Curves report each
task's accuracy throughout adaptation. There is no task ID, so conflicting
labels impose a deliberate trade-off; optimizer parameter sets also differ.
Practical implication: LoRA is not a guarantee against forgetting, and this
constructed comparison cannot rank retention methods generally.

### 13 — Reward hacking by construction

Candidates have lengths 20/60/120 and synthetic correctness 1/0.8/0.1.
Intended utility is correctness minus 0.002×length; the misspecified proxy is
length/120. Exact policy optimization compares proxy beta 0, proxy beta 0.5
and corrected-utility beta 0.5, using the common reference from Demo 05.
The 3D axes are expected length, proxy reward and intended utility; frames
show the actual recorded policy path. Curves keep all four outcomes separate,
including KL. This is a supplied misspecification, not evidence about an actual
reward model. Practical implication: limiting drift cannot repair an incorrect
objective on its own.

### 14 — Integrated post-training

The analytical study's `make_data`, `LoRALinear`, pair scoring, DPO and
evaluation helpers are reused. A separately seeded visual run uses an 8→4
linear base, rank 1 / alpha 2 adaptation, beta 0.5, with Adam 0.03 for base/SFT
and 0.01 for DPO. Split seeds are 643/644/645/646 at seed 42; sizes are
256/128/128/256 in quick mode. All three stages use 80 updates. This differs
from the older analytical study's sizes, seeds and 200/80/100 budgets.

The animation displays a fixed evaluation probe, stage-specific loss, held-out
metrics and the first 12 trainable-entry changes relative to that stage's
initial values. Those coordinates change meaning between base weights and
LoRA matrices, as the axis says. Frozen-base integrity is checked throughout
adaptation. All KL and relative-margin metrics are retrospectively computed
against the same independent final SFT reference. The loss panel shows only
the current stage's objective; losses must not be ranked across stages by
magnitude. Task retention measures only the original category
identity, not broad language capability. Practical implication: preserve the
reference and report separate task, preference and distribution metrics.

## Executed visual record — 2026-11-25, America/Sao_Paulo

Two complete seed-42 quick generations were run. The first dated UTC directory
is `outputs/visual-20261126T004701450838Z`; the second, containing presentation
refinements and explicit stage weight deltas, is
`outputs/visual-quick-20261125-02`. Both remain ignored and preserve their own
timestamps and source hashes. Neither replaces the earlier analytical JSON.

Both complete runs generated 13 GIFs, 21 PNGs and 14 interactive HTMLs,
plus the gallery, runtime, per-demo records and manifest. Numerical records
matched across all fourteen demos; the second adds stage-specific weight deltas.
The final presentation bundle, `outputs/visual-offline-20261125-03`, reuses the
second run's unchanged JSONs without new training. It improves axis wording,
displays reward/KL and preference objectives separately, and confines the pipeline
loss panel to its current stage. The final bundle is the recommended local gallery.
Selected results below retain the common seed/configuration.

| Measurement | Observed quick-run result | Interpretation candidate — author review required |
|---|---|---|
| Two analytical parameter objectives | Base loss 8.5069→0.000141; adaptation loss 1.8806→0.000127 | Descent reduces each separate quadratic from its own starting point; adaptation starts at the base endpoint |
| Single-demonstration SFT | Target probability 0.1271→0.9593; CE 2.0629→0.0415 | Likelihood fitting concentrates this toy distribution on the demonstration |
| Multitask held-out task accuracy | Base: 1.000 / 0.500 / 0.813 / 0.766; multitask: 0.813 / 0.766 / 0.906 / 0.727 | Adaptation did not improve every supplied rule |
| Reward-model expected held-out BT loss | 0.5871 / 0.5991 / 0.6642 for flip rates 0 / 0.2 / 0.4 | This draw is consistent with degraded preference prediction under extra flips |
| Exact beta-0.2 policy optimum discrepancy | Maximum probability error $1.03\times10^{-6}$ | Numerical optimization approaches the supplied analytical optimum |
| Population DPO optimum discrepancy | Final probabilities approximately 0.256966 / 0.426578 / 0.316456 | Matches the reward/KL optimum under this constructed population model |
| One-pair DPO, beta 0.5 | Adjusted margin 3.9227; chosen response probability 0.9277; implied BT probability 0.8767; loss 0.1316 | Pair-preference probability and model response probability are distinct quantities |
| Rank-2 target reconstruction | Rank-1 error 0.80899; rank-2 error 0.01984; rank-2 SVD bound approximately zero | Both rank capacity and finite optimization matter |
| Full-rank target, rank-8 reconstruction | Learned error 1.31717; SVD bound 1.26067 | Learning did not reach the best rank-8 approximation in this budget |
| Projected 2×2 update geometry | Final Frobenius error rank 1: 1.018034; rank 2: 0.00005745 | The known rank restriction excludes the unconstrained matrix optimum |
| Implemented adapter and counts | 8→2→8 adapter: 42 entries; four width-128 layers: 65,536 full / 8,192 LoRA / 8,736 adapter entries | Exact parameter counts depend on the stated architecture and bias policy, not measured runtime |
| Uniform 4-bit weight RMSE | 0.018674; reconstructed-base + LoRA 0.015023 | Adaptation reduces this reconstruction error, not a measured language error |
| Earlier Task A accuracy after B adaptation | Full 0.547; LoRA 0.352; replay 0.719, from 1.0 | LoRA did not prevent regression in this conflicting-task construction |
| Misspecified unregularized proxy | Proxy 0.350→1.000; intended utility 0.766→−0.140 | Proxy maximization can conflict with supplied utility |
| Integrated chosen-response probability | Base 0.5174; SFT 0.7736; DPO 0.9477; task accuracy 1.0 throughout | Preference shift and task retention coexist on this simple distribution |

No claim is made about general method superiority, human values, real language
quality, measured memory savings, production readiness or the full-mode seeds.

## Validation and preview handoff

The initial focused validation passed 62 tests: 30 analytical, 29 visual-core
and 3 visual-export/CLI checks, without skips or warnings. Numerical checks
cover normalization, cross-entropy, DPO derivatives (including the existing
finite-difference/autograd checks), zero LoRA initialization, rank/freezing,
merge equivalence, SVD bounds, KL, optimum agreement, quantization reconstruction
and held-out-update isolation. Rendering tests write to temporary directories.
Final validation on 2026-11-26 repeated all 62 topic tests successfully, without
skips or warnings. Repository parsing covered 545 public Python files; the
internal-link check covered 2,012 targets across 334 Markdown files. Offline
`uv pip check` found all 89 installed packages compatible, and `git diff --check`
reported no whitespace errors. This was a focused topic test run, not the full
curriculum suite or Linux CI.

The final bundle's artifact/source hashes matched its manifest. All 130 GIF
frames and 21 PNGs decoded; the gallery's 82 local targets resolved. Initial,
middle and final GIF frames plus static contact sheets were inspected. All
fourteen reused measurement records retained numerical agreement with the first
quick execution, aside from the second run's additional stage-delta field.

Chrome 158.0.8127.37, controlled through cached Playwright in an offline
1500×1100 desktop context, passed 164 assertions across all 14 interactive
pages. Checks exercised every configuration selector, final checkpoints,
playback, reset, actual 3D camera drags inside each scene's domain and nonblank
WebGL canvases. Key page screenshots were inspected. There were no JavaScript
or console errors and no external HTTP(S) requests. Twelve browser performance
warnings concerned Canvas2D readback and OpenGL `ReadPixels`; they were separate
from the warning-free numerical/export tests.

Browser rendering used software WebGL. These observations establish only the
inspected local desktop flow, not physical-GPU performance, cross-browser,
mobile, accessibility, GitHub math rendering, scientific validity or author
approval. The initial short simplex-axis label was expanded; an audit drag
that crossed two scene panels was corrected to use their actual layout domains.

Recommended candidates for later author curation:

| Candidate | Inspected size | Distinct learning question |
|---|---:|---|
| `08_lora_matrix_evolution.gif` | 624,312 bytes | Which matrices change while the base stays fixed? |
| `06_dpo_training_dynamics.gif` | 140,668 bytes | How do response probabilities, margin and pair loss evolve? |
| `07_rlhf_vs_dpo.gif` | 262,262 bytes | Which distributions/objectives agree under matched population assumptions? |
| `10_parameter_efficiency.png` | 119,163 bytes | What counts as trainable parameters versus assumed checkpoint storage? |

They remain ignored; no preview was copied, unignored or embedded.
Root README changes and final asset selection are left for author review, as
requested. No commit, push, publishing or deployment is part of this work.
