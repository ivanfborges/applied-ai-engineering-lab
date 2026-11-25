# Day 67 — interactive Transformer laboratory

The laboratory asks how the **complete encoder-decoder computation** connects
attention, features, residuals, normalization and conditional generation.
It reuses [example.py](example.py) and its synthetic data split. The separate
[analytical record](notes.md#execution-record--2026-11-25) remains unchanged;
the new paired visual experiment uses seed **67**, not analytical seed 42.

## Run locally

From the repository root, in a stable Python environment:

```powershell
python -m pip install -e ".[dev,deep-learning]"
python 05-transformers-llms/67-transformer-architecture-i/visualizations.py --list
python 05-transformers-llms/67-transformer-architecture-i/visualizations.py --all
python 05-transformers-llms/67-transformer-architecture-i/visualizations.py --quick
python 05-transformers-llms/67-transformer-architecture-i/visualizations.py --only architecture positional_encoding
python 05-transformers-llms/67-transformer-architecture-i/visualizations.py --only self_attention cross_attention
python 05-transformers-llms/67-transformer-architecture-i/visualizations.py --only representations
python 05-transformers-llms/67-transformer-architecture-i/visualizations.py --only training_experiment
python 05-transformers-llms/67-transformer-architecture-i/visualizations.py --interactive --skip-training
```

Open the printed `index.html` or an individual HTML directly from disk. Plotly
is bundled inside each interactive artifact, so no CDN, server, external API,
pretrained model or dataset download is needed. `--show` opens the local gallery
after generation; it does not start a service.

The full default generates all **16 groups / 33 artifacts**. Quick mode generates
10 core groups: architecture, positions, causality, cross-attention,
representations, teacher forcing/generation, leakage, complexity, training
progress and playground. It uses 128/32 training/held-out sequences and 80
updates per arm by default. It is a different experiment, not a faster
equal-coverage rendering of the full default. `--interactive` generates only
HTML artifacts for the selected groups (all seven interactive groups by default).

| Option | Default and meaning |
|---|---|
| `--seed` | 67; dataset/model seed; batch generator uses seed + 1 |
| `--training-steps` | 400 per arm, or 80 with `--quick`; accepted range 1–2000 |
| `--sequence-length` | 5 content tokens; bounded to 3–10 |
| `--num-heads` | 4; accepts 1, 2, 4, 8 with width 32 |
| `--fps` | 2 explanatory states/second; 1–12 |
| `--dpi` | 90; accepted range 40–200 |
| `--device` | `cpu` only; preserves the original study's CPU tensor contract |
| `--output-dir` | Topic-local `outputs/`; use another ignored directory for alternatives |
| `--skip-training` | Load a compatible cache; if missing, skip measured-training groups and label other model views untrained |

Training is invoked only when a selected group uses trained observations.
Groups 11, 12 and 15 require trained checkpoints. Cross-attention, head views,
residuals, normalization, representation evolution and playground can instead
show explicitly labeled untrained computation. `--only architecture` does not
train. Imports do not train, create files or regenerate existing assets.

## Artifact map and learning questions

The names below are filenames inside a generated run, not links to approved
public previews. All generated artifacts remain ignored pending author review.

| Group | Question and classification | Generated files |
|---|---|---|
| 01 | How do source IDs reach output probabilities? **Conceptual illustration** with configuration-derived shapes and encoder-memory K/V bridge | `01_transformer_architecture.gif` |
| 02 | How do sinusoidal features and positional similarities vary? **Mathematical visualization** and an **untrained numerical equivariance control**; distinct from learned model positions | `02_positional_encoding_heatmap.png`, `02_positional_encoding_waves.gif`, `02_positional_encoding_3d.html` |
| 03 | How do Q/K/V, scaled scores, row softmax and weighted values connect? **Untrained numerical demonstration**, with synchronized graph and heatmap | `03_encoder_self_attention.gif`, `03_encoder_attention_heatmap.png` |
| 04 | What is blocked, and why are logits different from probabilities? **Untrained numerical demonstration** holding Q/K fixed while changing mask | `04_decoder_causal_attention.gif` |
| 05 | Where do cross-attention Q/K/V originate? **Trained model observation**, or labeled untrained fallback; rectangular matrices and initialization/final comparison | `05_cross_attention.gif`, `05_cross_attention_heatmap.png` |
| 06 | What differs across heads and after concatenation? **Model observations** of individual projections, per-head weights and output projection | `06_multi_head_attention.gif`, `06_attention_heads.png`, `06_attention_heads_interactive.html` |
| 07 | How does one position expand 8→32→8 without mixing tokens? **Untrained numerical demonstration**, same FFN for all positions; fixed-input ReLU/GELU comparison | `07_feed_forward_network.gif`, `07_ffn_activations.png` |
| 08 | What does adding an update do geometrically? **Model activation observation**; 2D feature slice and a shared PCA view before normalization | `08_residual_connections.gif`, `08_residual_vectors_3d.html` |
| 09 | Which feature moments change within a token? **Model activation + mathematical visualization**, with clearly illustrative affine scale/bias; pre/post topology is **conceptual**, not a stability benchmark | `09_layer_normalization.gif`, `09_activation_distributions.png`, `09_pre_norm_vs_post_norm.png` |
| 10 | How do tokens move across actual encoder stages? **Model activation observations** in a single PCA basis; HTML includes initialized and trained checkpoints | `10_representation_evolution.gif`, `10_representation_evolution_3d.html` |
| 11 | How does the information available differ? **Trained model observation**: teacher-forced vocabulary distributions versus actual generated-prefix distributions | `11_training_vs_inference.gif` |
| 12 | Can teacher forcing hide missing causality? **Measured experimental result** plus **conceptual illustration** of the label shortcut | `12_causal_mask_experiment.png`, `12_target_leakage.gif` |
| 13 | How do information paths differ? **Conceptual illustration** of RNN hidden state, LSTM hidden/cell state and dense encoder attention; no latency benchmark | `13_transformer_vs_rnn.gif` |
| 14 | How do lengths, heads, precision and batch affect work/storage? **Theoretical complexity estimates**; no large score allocation | `14_attention_complexity.png`, `14_attention_memory_3d.html`, `14_sequence_length_scaling.gif` |
| 15 | What changes at recorded optimization checkpoints? **Measured experimental results / trained observations**, including actual failures | `15_training_progress.gif`, `15_training_metrics.png`, `15_generation_examples.gif`, `15_training_attention_evolution.html` |
| 16 | What changes when computation or visibility changes? **Precomputed model observations** with explicit counterfactual labels | `16_transformer_playground.html` |

The sentence in group 03 is only a sequence of display aliases for synthetic
IDs. Its random untrained attention does not demonstrate coreference or grammar.
The group 07 matrices form a small educational FFN, not the trained model's FFN.
Group 09's illustrative gamma/beta are not presented as learned parameters.

## Model instrumentation and geometric integrity

[transformer_lab.py](transformer_lab.py) calls the actual registered attention,
FFN and normalization modules of `Seq2SeqTransformer`; no second trainable
Transformer architecture is introduced. Explicit Q/K/V projections expose raw
dot products, scaled logits, per-head softmax, head outputs, concatenation and
the output projection. Captured computations are checked against
`nn.MultiheadAttention(..., need_weights=True, average_attn_weights=False)`
and the original model forward, including final stack normalization.

All capture occurs in evaluation mode, so dropout is disabled. The manual
composition supports the study's equal-width, bias-containing attention without
extra key/value tokens. Fully blocked rows raise an error rather than producing
an undefined softmax. `True` in this model's boolean mask means blocked.

The existing NumPy block already retains fixed parameters between calls; it
does not need refactoring. Its sinusoidal implementation and fixed-projection
attention are reused for the position/equivariance demonstration. The original
single-head educational forward and the learned multi-head model remain distinct.

The representation GIF fits one PCA across all its displayed stages. The 3D
HTML fits one PCA across **all displayed stages and training checkpoints**.
Token colors, axes and coordinate limits remain consistent inside each artifact;
PCA coordinates from different artifacts are not interchangeable. Explained
variance is reported in the interactive view. Attention and FFN branch outputs
are labeled separately from states after residual addition and normalization.

The residual 3D view fits one PCA across all shown input/update/sum triplets.
Inputs and sums use centered PCA coordinates; update vectors use the same linear
projection without subtracting the mean. This preserves projected vector
addition. The representation is still lossy and precedes LayerNorm. The residual
GIF interpolates between actual input and summed endpoint vectors on fixed axes;
intermediate fractions are geometric illustration, not extra model layers.

## Interactive controls and their meaning

- **Positions:** length/dimension controls recompute the sinusoidal formula and
  its cosine similarity in JavaScript; they do not change a trained position table.
- **Heads/training evolution:** select checkpoint, mechanism, layer, head and
  query from actual recorded tensors. Vocabulary slots and source query positions
  have different meanings; the displayed target-slot index is bounded accordingly.
- **Residuals:** choose the encoder layer; rotate and zoom a real projected
  input/update/sum triplet for token 0.
- **Representations:** choose checkpoint and encoder stage, play/pause stages,
  rotate/zoom and inspect token trajectories in a common 3D coordinate system.
- **Memory:** batch/head/precision controls recompute theoretical bytes only;
  source/target axes span 128–32,768 without constructing score tensors.
- **Playground:** length 3 or configured length, position enabled/disabled,
  causal visibility, pre/post-norm and initialized/final checkpoint select
  **real precomputed forwards**. Head, layer and query select a view. The six
  synchronized panes show encoder geometry, masked scaled logits, probabilities,
  connectivity, attention output features and vocabulary probabilities.

The playground head count is fixed by `--num-heads`; changing it requires a new
compatible model/experiment rather than relabeling the existing tensor. The
trained model uses post-norm, learned positions and the configured length.
Changing those properties without retraining is labeled **counterfactual**.
The loss of a causal mask here applies to a forward intervention, distinct from
the separately trained unmasked arm in group 12. Shorter-length cases use
prefixes from a recorded synthetic sequence and are not an evaluated new benchmark.

## Paired synthetic experiment — 2026-11-25

**Hypothesis:** missing causal masking can improve teacher-forced scores while
free-running generation remains poor because future ground-truth tokens are
unavailable. This is tested without forcing the expected result.

**Executed command:**

```powershell
.\.local\environments\n09-full-2026-11-13\Scripts\python.exe 05-transformers-llms/67-transformer-architecture-i/visualizations.py --all --training-steps 400 --seed 67
```

**Configuration:** synthetic length-5 reversal; 512 unique training sequences
and 128 disjoint held-out sequences; vocabulary 32 (BOS=0, EOS=1); width 32,
four heads, two encoder and two decoder layers, FFN 128, post-norm, ReLU,
backbone dropout 0.1; Adam at 0.001; 400 updates per arm, batch 32,
gradient clipping 1.0. Both arms start with identical state dictionaries,
data/split, batch generator and RNG seeds. The difference is decoder masking.
Snapshots at updates 0, 40, 200 and 400 are descriptive; none is selected as best.

CPU FP32, one PyTorch thread; Python 3.11.14, PyTorch 2.15.0+cpu and NumPy 2.4.6
on Windows 10 build 19045. Exact rendering/library versions are retained in each
JSON run manifest. Training losses are full-training evaluation-mode means,
not noisy dropout minibatch losses.

| Final measured quantity | Correct causal mask | No causal mask |
|---|---:|---:|
| Full-training CE | 0.053364 | 0.017075 |
| Held-out teacher-forced CE | 0.143631 | 0.031018 |
| Held-out teacher-forced token accuracy | 97.6563% | 99.8698% |
| Held-out teacher-forced exact match | 85.9375% | 99.2188% |
| Held-out autoregressive token accuracy | 95.5729% | 57.1615% |
| Held-out autoregressive exact match | 85.9375% (110/128) | 1.5625% (2/128) |

Generation starts with BOS and uses only its own predictions. The causal arm
uses causal decoder prefixes. The unmasked arm uses unrestricted attention
**within the available generated prefix**, never future labels. Both emit
exactly six tokens, including the EOS slot, without early EOS stopping. EOS
counts in all token and sequence metrics. Neither arm receives the expected
answer during generation.

**Interpretation candidate — author review required:** the unmasked arm's better
teacher-forced fit and much lower autoregressive exact match are consistent
with the future-target shortcut. The causal arm learned the bounded reversal
mapping under this budget. This is not proof that every unmasked model will
fail identically or that attention matrices causally explain individual errors.

**Limitations:** one seed, one length and architecture; no uncertainty analysis,
natural-language data, variable-length padding, EOS stopping, KV cache, latency
benchmark or pre/post-norm training comparison. The held-out split is inspected
at multiple checkpoints as a **diagnostic curve**, not claimed as an untouched
final test for subsequent tuning. Changes inspired by these curves require new
validation/final-test planning. Successful reversal does not establish semantics,
translation quality or production readiness.

The numerical invariance tests perturb future decoder inputs and require
unchanged earlier logits under causal visibility, with an unmasked sensitivity
control. The mathematical equivariance control reorders both query/key inputs
without positions and compares the correspondingly reordered outputs.

## Outputs, reuse and resource bounds

- `outputs/visualizations/<UTC-run>/`: GIFs, PNGs, seven standalone HTMLs and
  an `index.html` gallery; old runs are preserved.
- `outputs/metrics/visual_run_<UTC-run>.json`: environment, configuration,
  actual measurements, cache status, asset inventory and failures/skips.
- `outputs/checkpoints/comparison_<fingerprint>.pt`: local paired model states.
  The fingerprint includes configuration, schema, PyTorch version and numerical
  source files. Loading uses CPU and `weights_only=True`; incompatible caches
  are not silently relabeled.

Training is bounded to two small models, one CPU thread, and four checkpoints
per arm. GIF frame counts follow short sequences or explicit computational
states. Complexity plots evaluate formulas only. High DPI, longer sequences or
more steps increase costs; no hardware timing guarantee is claimed. Each HTML
bundles Plotly, so several megabytes per HTML are intentional local artifacts.
No package installation, CUDA migration or frontend framework is performed by
the generator. All Matplotlib figures are closed after export.

The first full attempt trained the paired models but failed at the architecture
diagram: a Matplotlib elbow connector used parallel angles, followed by a
secondary empty-frame writer error. The connector was corrected, an actual
export test passed, and the next full generation reused the trained cache.
The initial failed run manifest is preserved rather than rewritten as successful.

The complete successful run was `20261125T110907411819Z`: **33 assets, zero
skipped groups**. A targeted render revision, `20261125T112333850320Z`, reused
the same cache and generated nine assets for positions, multi-head projections,
residual interpolation and teacher forcing. Its three HTMLs are byte-identical
to the already browser-validated versions. The local gallery under
`outputs/visualizations/reviewed_20261125T112333Z/index.html` combines the complete
suite with those revised renders. Its manifest records the source run for every
asset; original runs and experimental records are preserved.

A separate `--quick --interactive --skip-training` smoke generated four untrained
HTMLs. It explicitly skipped the training-experiment group because no compatible
quick-mode checkpoint existed. This intentional skipped group is not a skipped
test or an omission from the full 16-group generation.

## Validation and publication boundary

Run focused checks; rendering tests write only to temporary directories:

```powershell
python scripts/validate_repo.py tests --scope 05-transformers-llms/67-transformer-architecture-i
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

[test_visual_core.py](tests/test_visual_core.py) covers attention/model agreement,
per-head probabilities, causality, rectangular cross-attention, shared PCA,
permutation equivariance, deterministic paired initialization/training, split
separation, cache behavior and theoretical storage arithmetic. Existing
[analytical tests](tests/test_transformer_architecture.py) cover target shifting,
LayerNorm behavior, finite gradients and generation consistency.
[test_visual_exports.py](tests/test_visual_exports.py) produces real GIF/HTML
exports and checks CLI selection, missing-training behavior, imports and
offline bundling. The generator decodes every GIF frame, checks durations/looping,
verifies PNGs and checks HTML runtime bundling. Those checks do not substitute
for opening controls in a browser or personal interpretation review.

The local focused suite passed **54 tests** (31 analytical, 10 visual numerical,
13 export/CLI), with no skips or warnings. Syntax parsed **481 public Python
files**; internal-link validation passed; compatibility checking found all
**81 installed packages** compatible. The initial export smoke emitted a
Matplotlib `set_bad` pending-deprecation warning; the renderer now uses
`with_extremes`, and the completed suite emitted no warnings.

All seven standalone HTMLs were opened in **Headless Chrome 158** on Windows,
offline, at 1500×1050. Checks exercised formulas, head/layer/query selectors,
rectangular attention, zero masked probabilities, checkpoint changes, projected
residual addition, PCA stages, play/pause, counterfactual computations, resets
and real pointer-driven 3D camera rotation. There were no JavaScript errors;
Plotly emitted Canvas2D readback-performance warnings. This is local desktop
evidence, not cross-browser, mobile, full accessibility or native GPU performance
validation. CLI startup initially encountered filesystem/process sandbox limits;
the successful checks kept auxiliary files local and ran each browser session
through its complete open/check/close lifecycle.

## Recommended preview set — pending author approval

Representative states of all raster exports were visually inspected; all seven
HTML views were exercised in the browser, and the PCA/playground screenshots
were also inspected visually. These six candidates answer complementary questions:

| Candidate | Bytes | Reason |
|---|---:|---|
| `01_transformer_architecture.gif` | 549,012 | Complete source-to-prediction path and the memory bridge |
| `04_decoder_causal_attention.gif` | 207,729 | Logits, visibility and zero future probabilities |
| `05_cross_attention.gif` | 168,928 | Actual rectangular source/target connectivity |
| `10_representation_evolution_3d.html` | 4,869,003 | Rotatable activation trajectories in one shared PCA basis |
| `11_training_vs_inference.gif` | 177,600 | Correct target availability, causal mask, logits and generated prefixes |
| `14_attention_complexity.png` | 129,455 | Explicit theoretical work/storage trade-offs without allocating giant tensors |

Total **6,101,727 bytes (5.82 MiB)**, within the repository's 6 MiB selected-image
review target when including the linked HTML. The five raster candidates total
about 1.18 MiB. The offline 3D HTML bundles its runtime and should be linked
rather than embedded if approved. The measured leakage chart
`12_causal_mask_experiment.png` is a useful alternative if experiment evidence
is preferred to the theoretical complexity preview.

The larger self-attention GIF and comprehensive playground HTML remain local;
the playground is about 6.59 MiB and is not proposed for versioning. No generated
output is approved, deliberately unignored or automatically published. The
topic README describes the implementation; the root EN/PT READMEs and module
index retain their pre-visual content until the requested author approval.
