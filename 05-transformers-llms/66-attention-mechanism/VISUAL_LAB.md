# Attention visual laboratory

This is a CPU-only laboratory connecting the same numerical operation across
matrix arithmetic, geometry, probability, sequence visibility and inference.
It reuses [the analytical attention reference](from_scratch.py). Synthetic
parameters are deliberately designed or randomly initialized, never trained.
**Author interpretation review required.**

## Install and navigate

Use Python 3.11+ in an isolated environment. From the repository root:

```sh
python -m pip install -e ".[dev,deep-learning]"
python 05-transformers-llms/66-attention-mechanism/visualize_attention.py --list
python 05-transformers-llms/66-attention-mechanism/visualize_attention.py --quick
python 05-transformers-llms/66-attention-mechanism/visualize_attention.py --all
```

The base dependencies already include NumPy, Matplotlib, Pillow and Plotly.
PyTorch is needed for the numerical reference tests, not for rendering. No
package installation, tokenizer resource download or network request occurs
when generating or opening the artifacts. HTML bundles the Plotly runtime and
its license; mathematical labels use plain Unicode rather than remote MathJax.

Within the topic directory, the same entry point supports:

```sh
python visualize_attention.py
python visualize_attention.py --experiment geometry
python visualize_attention.py --experiment scaling
python visualize_attention.py --experiment softmax
python visualize_attention.py --experiment masking
python visualize_attention.py --all --quick --output-dir outputs/checks
python visualize_attention.py --experiment values --open
```

Each command creates a new UTC-stamped child directory in `outputs/runs/`
(or the specified parent), preserving earlier runs. Open its `index.html`
directly from Windows Explorer. `--open` optionally asks the OS to open that
local gallery. Standalone explorers can be copied individually and opened
without the gallery, server, CDN or credentials. The raw exports remain ignored.
The generator never copies files into the public-preview directory.

## Experiments and connections

| CLI name | Question and interaction | Generated filenames |
|---|---|---|
| `projections` | Which input row and projection column create each Q/K/V cell? Four synthetic tokens, explicit partial products and dimensions | `01_qkv_projections.gif` |
| `geometry` | How do query azimuth/elevation, magnitude and key magnitude change angles, dot products and weights? Real 3D camera, synchronized table/bars, rotation/pause/reset | `02_query_key_geometry.html`, `02_query_key_alignment.gif` |
| `pipeline` | Where do transpose, dot products, scaling, normalization and value mixing enter? Active stages and first-row arithmetic | `03_attention_pipeline.gif`, `03_attention_pipeline.png` |
| `scaling` | Do Gaussian score variance and softmax sensitivity change with width? Five dimensions, fixed key count, distributions and measured derivatives | `04_attention_scaling.gif`, `04_score_distributions.png`, `04_attention_entropy.png`, `04_softmax_gradient_saturation.png` |
| `softmax` | How does one logit affect every weight? Four independent logits, temperature, presets and reset, stabilized exponentials | `05_softmax_playground.html`, `05_softmax_temperature.gif` |
| `values` | Why can output move without changing weights? Normalize nonnegative masses, inspect a tetrahedral convex hull or fix Q/K and translate V | `06_weighted_value_geometry.html`, `06_weighted_value_aggregation.gif`, `06_changing_values.gif` |
| `context` | Can the same bank input have different contextual mixtures? Switch engineered river/finance features, inspect all Q/K/V rows and weight matrices | `07_contextual_embeddings.html`, `07_contextual_attention.gif` |
| `masking` | Which keys can a query see? Full, causal and padding masks, row reveal, masked logits, weights, outputs and future-only intervention | `08_attention_masks.gif`, `08_full_vs_causal_attention.png`, `08_future_token_leakage.gif` |
| `cross` | Why can A be rectangular? Source/target inputs and projections, origin graphs, 5×5 self-attention versus 3×5 cross-attention | `09_self_vs_cross_attention.gif`, `09_cross_attention_heatmap.png` |
| `heads` | How do projection spaces produce separate outputs? Three deliberately different heads, concatenation and output projection | `10_multi_head_attention_preview.gif` |
| `complexity` | How do pair counts, one-matrix byte estimates and measured reference-call time grow? Interactive theory-only memory slider | `11_attention_complexity.png`, `11_attention_matrix_growth.gif`, `11_attention_memory.html` |
| `inference` | How are external retrieval and internal attention different, and what does caching reuse? Calculated cosine retrieval plus prefix-validated cached outputs | `12_attention_in_rag.gif`, `12_kv_cache_inference.gif` |

Blue means Q, orange K/scores, green V, purple weights/output and gray
masked/inactive states. Plots use fixed comparative axes. Full PNGs are
150 dpi; GIFs use bounded frame counts and readable pauses. Matrix animations
emphasize operations, not camera motion. The 3D camera is preserved when controls
change vector values.

The planar geometry GIF uses $d_k=2$ and two full rotations: first unit keys,
then unequal norms. Its 3D HTML uses $d_k=3$. The weighted-value GIF uses a 2D
triangle; its interactive HTML uses four values in a 3D tetrahedron. These are
different, explicitly labeled fixtures rather than interchangeable measurements.

## Mathematical contracts and simplifications

- Standard normalized attention produces a convex combination before dropout
  or later output projections. Manual nonnegative masses are normalized.
  If all four HTML masses are zero, the UI explicitly falls back to uniform
  weights. The numerical helper instead rejects a zero-total mass input.
- Boolean True permits attention. Fully masked queries raise `ValueError` in
  the shared reference. Padding masks remove keys; padded-query outputs still
  require handling in downstream loss/output logic. No mask can repair future
  information already present in permitted representations.
- Softmax uses maximum subtraction. Temperature is an optional teaching
  parameter. Derivative plots measure $\|\operatorname{diag}(p)-pp^T\|_F$
  with respect to incoming softmax logits. Differentiating scaled attention
  with respect to raw dot products also introduces $1/\sqrt{d_k}$ (and $1/T$
  when temperature is used). No full-network gradient claim is made.
  The numerical norm uses sums of diagonal/off-diagonal squares rather than
  subtracting nearly equal expanded terms near a one-hot distribution:

$$
\|J\|_F^2=\sum_i p_i^2(1-p_i)^2+\sum_{i\ne j}p_i^2p_j^2.
$$

- Gaussian scaling theory assumes independent centered unit-variance
  coordinates. A visual run uses one generator seeded at 66, 64 keys and 512
  queries per dimension (128 in quick mode), with fresh draws for widths
  4/16/64/256/1024. Entries share query/key factors; they are not independent
  experimental replicates. Variance uses `ddof=0`; entropy is mean row entropy
  in nats. There are no confidence intervals or trained vectors.
- Raw/scaled histograms use different, explicitly labeled x ranges, each fixed
  across widths. Density axes match. Theoretical variance is distinguished
  from empirical variance; both maximum probability and entropy are measured
  across multiple query rows.
- The contextual demo assigns water/finance features explicitly. All queries
  are `[1,1]`, K=2X and V=X, so every query within one context gets the same
  output. The GIF's moving circle interpolates endpoint outputs for display;
  it is not attention computed on a blended sentence or semantic learning.
- Multi-head parameters deliberately generate distinct patterns. Neither
  random initialization nor visible differences establish learned semantic
  or syntactic specialization. This is a preview of later roadmap topics.
- The RAG animation retrieves two of three synthetic document vectors by
  actual cosine. Selected vectors become token proxies for actual attention.
  No tokenizer, LLM, citation engine or generated response text exists.
- The cache animation has one frozen layer and predetermined token inputs.
  Prefill projects the first three positions; each new position appends K/V
  once and attends over the available prefix. Each cached output is checked
  against full causal-prefix recomputation. Sampling, position encodings,
  residuals and multilayer model state are omitted. Projection-row counts are
  operation counts, not measured latency savings.

## Timing and memory protocol

Full timing lengths: 128, 256, 512, 1024, 2048. Quick timing lengths: 128, 256.
Feature and value width are fixed at 32. Inputs use standard-normal float64
arrays with seed 66. Each size gets one warm-up and five timed repetitions
(three quick); the report retains every sample and its median. `perf_counter`
measures complete NumPy reference calls, including validation, allocation,
score calculation, softmax and value mixing. Rendering is outside those samples.

The entry point sets numerical thread environment defaults to one before NumPy
imports, preserving explicitly supplied user limits. `run.json` records those
settings, CPU identifier, OS, Python and package versions. These settings are not
a hard CPU/RAM cap. Tests and generation should be run serially.

Theory plots include lengths up to 16,384 without allocating their score
matrices. Byte estimates are for **one score OR weight matrix per head/example**:
$n^2\times4$ bytes for FP32 and $n^2\times2$ for FP16. Actual reference memory
includes multiple float64 intermediates. FP16 is not executed. No length above
2048 is timed. FlashAttention's reduced memory materialization/traffic does not
remove dense quadratic arithmetic; no optimized kernel is benchmarked.

## Source organization and verification

- [attention_lab.py](attention_lab.py): numerical traces, controlled fixtures,
  scaling statistics, bounded timing and incremental cache reference.
- [attention_html.py](attention_html.py): five self-contained explorers with
  actual JavaScript calculations and a numerical state for browser checks.
- [attention_figures.py](attention_figures.py): matrix/vector/scaling renderers.
- [attention_sequences.py](attention_sequences.py): sequence/mask/head and
  inference renderers.
- [visualize_attention.py](visualize_attention.py): selection, dated manifests,
  metadata and a local gallery.
- [tests/test_visual_core.py](tests/test_visual_core.py) and
  [tests/test_visual_exports.py](tests/test_visual_exports.py): invariants and
  real temporary-directory PNG/GIF/HTML exports.

```sh
python scripts/validate_repo.py tests --scope 05-transformers-llms/66-attention-mechanism
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

The topic suite includes real float64 CPU PyTorch SDPA comparisons, including
rectangular attention. Export tests inspect every GIF frame and parse HTML
script dependencies; neither proves interactive control correctness. Browser
checks compare displayed numerical states with Python references and exercise
controls and camera movement separately. GitHub rendering, other browsers,
native-GPU behavior and scientific author review remain separate validations.

## Visual execution and preview record

The AI assistant completed a quick run (`20261124T214028040479Z`) and a full
run (`20261124T214710433223Z`) on 2026-11-24. Each produced 15 GIFs, 7 PNGs,
5 standalone explorers, a gallery and `run.json`. Every GIF frame was decoded:
180 quick frames and 312 full frames. PNG decoding and embedded-script checks
passed. The earlier analytical
[2026-11-24 record](notes.md#execution-record--2026-11-24) remains a distinct
protocol with its original query counts and dimensions.

Environment: Windows 10 build 19045, Intel64 Family 6 Model 158 Stepping 9
(GenuineIntel; eight logical CPUs), Python 3.11.14, NumPy 2.4.6, Matplotlib
3.11.2, Pillow 12.3.0 and Plotly 6.9.0. All three recorded numerical thread
environment settings were one. PyTorch reference tests used 2.15.0+cpu.
These are local execution facts, not a frozen environment or CI result.

### Full-run measurements

Scaling hypothesis/configuration: independent Gaussian query/key coordinates,
512 queries and 64 keys per dimension, seed 66; dimensional scaling should
control dispersion under the stated assumptions. These are results of the
full visual protocol, not a rerun of the smaller analytical experiment:

| $d_k$ | Raw variance | Scaled variance | Raw mean entropy | Scaled mean entropy |
|---:|---:|---:|---:|---:|
| 4 | 3.666762 | 0.916691 | 2.827366 | 3.723850 |
| 16 | 15.991454 | 0.999466 | 1.387210 | 3.695677 |
| 64 | 63.528065 | 0.992626 | 0.591733 | 3.692554 |
| 256 | 254.277706 | 0.993272 | 0.287222 | 3.685462 |
| 1024 | 1031.308911 | 1.007138 | 0.133029 | 3.683146 |

At width 1024, mean row maximum probability was 0.947585 raw versus 0.106583
scaled. Raw mean softmax Jacobian norms across widths were 0.260346, 0.293229,
0.207135, 0.141448 and 0.071027: they did **not** decrease monotonically at
every width. Corresponding scaled norms were 0.178301–0.183741.

Interpretation candidate — **author review required**: in this Gaussian fixture,
scaling kept score variance near one and entropy comparatively stable while
unscaled distributions became concentrated. The derivative diagnostic captures
local softmax sensitivity, with a non-monotonic mean at smaller widths. It does
not establish gradient health or task quality in a trained network. Limitations:
one seed, shared keys across rows, finite samples and no training or independent
replication.

Complexity hypothesis/configuration: pair counts and one-matrix bytes grow
quadratically; observed reference-call time should generally increase with
sequence length. Width 32, float64, one warm-up and five timed repetitions:

| Tokens | Median complete NumPy call (ms) | One FP32 matrix (MiB) | One FP16 matrix (MiB) |
|---:|---:|---:|---:|
| 128 | 0.377700 | 0.0625 | 0.03125 |
| 256 | 2.035500 | 0.25 | 0.125 |
| 512 | 9.036900 | 1 | 0.5 |
| 1024 | 39.084500 | 4 | 2 |
| 2048 | 155.509800 | 16 | 8 |

The time column is measured; byte columns are theoretical dtype-size arithmetic.
At 16,384 tokens, theory gives 268,435,456 interactions and 1024/512 MiB for
one FP32/FP16 matrix; that length was neither allocated nor timed.

Interpretation candidate — **author review required**: time rose strongly with
length in this single local reference implementation. This is consistent with
its dense pair work, but it does not estimate universal latency, prove an
asymptotic slope or compare optimized kernels. CPU scheduling, BLAS, allocation,
checks and cache behavior are included. Individual samples and environment
remain in the ignored dated manifest; no actual peak-memory measurement exists.

The future-only control changed the last K/V while keeping earlier queries and
permitted representations fixed. Maximum earlier-output difference was
11.494365 with full attention and exactly 0 with causal attention. Cached
decoding at prefix lengths 4/5/6 differed from full causal recomputation by
at most $6.94\times10^{-17}$. External cosine retrieval selected synthetic
document IDs `[2,0]` with scores 0.987007 and 0.980581.

Interpretation candidates — **author review required**: the mask blocks the
specified forbidden interactions; the frozen-layer cache reproduces this
prefix computation while projecting only one new K/V row. The retrieved order
is a cosine calculation on three designed vectors. None establishes end-task
quality, trained-language-model behavior or measured cache speedup.

### Reviewed artifacts and browser coverage

The final local gallery is in the ignored directory
`outputs/runs/20261124T220503184125Z-review/`. It retains the full run's
numerical measurements and timing samples. Its `visual_revision` metadata
identifies the original measurement run and subsequent visual revisions:
camera preservation across rapid control updates, separated coincident vector
labels, legible time-axis ticks, unique position IDs for repeated tokens and
explicit purple attention colors. Five HTMLs and affected vector/complexity
rasters were regenerated; numerical experiments were not rerun or relabeled.
Earlier generated runs remain intact. A fresh `--all` run reproduces the
current presentation and records its own new timings.

The AI assistant inspected all seven static figures and representative
first/middle/final animation states, including the corrected vector labels.
Every final GIF frame was decoded. A final offline Chrome 158.0.8127.37 audit
passed **58 checks**, including control updates/reset, rotation/pause, actual
3D camera drag and preservation, JS/NumPy agreement, value-only intervention,
uniform fallback, contextual token-position identity, byte estimates and local
gallery links. All five explorers were also checked at a selected 430×900
viewport; desktop was 1500×1050. The final audit confirmed **SwiftShader software
WebGL**, demonstrating the browser path without native GPU rendering.

No JavaScript/console errors or external HTTP(S) requests were observed. Eight
performance warnings occurred: four Canvas2D readback advisories and four
WebGL `ReadPixels` stall advisories. No frame-rate or browser-performance
measurement is claimed. The Playwright CLI could not retain a local daemon
session; the successful checks used the already-cached Playwright library with
Chrome and process-local caches under ignored local validation material.

These checks do not establish cross-browser behavior, exhaustive accessibility,
GitHub rendering, a clean Linux install or production readiness. Interpretation
review by the author is still pending. Root EN/PT READMEs and the module index
were left at their pre-visual state, as requested.

### Six intentional public-preview candidates

All six were inspected and explicitly unignored before being linked. Copies
are byte-identical to the reviewed exports; rendering tests use temporary
directories and never overwrite them. Total: **2,431,242 bytes (2.32 MiB)**.
Only the pipeline GIF and scaling PNG are inline in the topic README:
321,055 bytes combined, below the documented inline budget.

| Preview | Bytes | Learning question | Reproduce with `--experiment` |
|---|---:|---|---|
| [Pipeline](outputs/previews/03_attention_pipeline.gif) | 184,729 | Where do transpose, scaling, normalization and value mixing occur? | `pipeline` |
| [Query–key alignment](outputs/previews/02_query_key_alignment.gif) | 1,197,490 | How do rotation and unequal norms jointly affect attention? | `geometry` |
| [Weighted values](outputs/previews/06_weighted_value_aggregation.gif) | 544,179 | How does the output move inside a convex hull? | `values` |
| [Scaling](outputs/previews/04_attention_entropy.png) | 136,326 | Which variance/entropy/concentration changes were actually measured? | `scaling` |
| [Causal visibility](outputs/previews/08_full_vs_causal_attention.png) | 186,594 | How does the same Q/K/V produce different outputs under masking? | `masking` |
| [Complexity](outputs/previews/11_attention_complexity.png) | 181,924 | How do theory, byte estimates and local measured times differ? | `complexity` |

The 1.14 MiB alignment GIF is an intentional exception to the 1 MiB review
target: 82 computed frames preserve two complete rotations, unit and unequal
key norms, and synchronized weight updates over 16.4 seconds. A single PNG
would lose the continuous alignment-to-weight relationship. Other selected
files satisfy their per-file targets; the selection is below the 6 MiB total.
No HTML, gallery, benchmark manifest or alternate animation is selected for Git.
The author should review both these candidates and the interpretation candidates
above before publication.

Final automated local checks on the reviewed source: 46 passed cases across
three isolated test files (23 analytical, 20 visual numerical and 3 export
cases), with no skips or warnings. Repository syntax parsed 472 Python files;
the internal-link checker validated 1,739 links across 294 Markdown files.
Dependency compatibility checked 81 installed packages through offline `uv pip
check` against the selected interpreter, because that environment has no `pip`
module. `git diff --check` passed; Git emitted its existing LF/CRLF conversion
advisory for the unchanged module index. Baseline hashes confirmed that the
root READMEs, module index and private study source were unchanged during this
visual phase. No commit, push, publication or deployment was performed.
