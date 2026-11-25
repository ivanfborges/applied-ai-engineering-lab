# Multi-Head Attention: interactive visual learning lab

This lab follows one forward computation from synthetic input vectors to the
model-width output, then separates geometry, routing and serving calculations.
It is intended for technical exploration rather than a trained language-model
demonstration. Every panel labels its evidence: **computed untrained tensors**,
**constructed controls**, **PCA approximations**, or **theoretical counts**.
Interpretation candidates require **author review**.

## Start with one command

From the repository root, use the existing stable Python environment with the
canonical dependencies. Installation, when needed, is through
`python -m pip install -e ".[dev,deep-learning]"`; NumPy, Matplotlib, Plotly
and Pillow already belong to the root runtime. No additional dependency,
Streamlit server, API key, model download or GPU is needed.

```powershell
python 05-transformers-llms/68-transformer-architecture-ii/visualize.py --all --show
python 05-transformers-llms/68-transformer-architecture-ii/visualize.py --quick --show
python 05-transformers-llms/68-transformer-architecture-ii/visualize.py --experiment subspaces
python 05-transformers-llms/68-transformer-architecture-ii/visualize.py --experiment scaling
python 05-transformers-llms/68-transformer-architecture-ii/visualize.py --experiment kv-cache
python 05-transformers-llms/68-transformer-architecture-ii/visualize.py --list
```

The generator prints the new `outputs/run-UTC_TIMESTAMP/index.html` path.
Open that local file to navigate the ten learning modules. Standalone Plotly
documents embed their runtime and data; they work without a server or network.
The index uses a local iframe to avoid loading every WebGL plot at once.
Each animation has a frame selector in the gallery: select a stage to pause,
open its PNG to inspect or save it, or open the complete GIF to animate.

`--quick` retains all selected questions, reducing random sample count from
4,000 to 1,000 and reducing subspace/concatenation animation stages. It does
not skip numerical checks. `--experiment NAME` can be repeated. Default seed
is **168**, distinct from the earlier analytical seed 68. `--dpi 50–160`
controls raster resolution; default is 90. `--output-dir PATH` selects the
parent for new dated run directories. Old runs and public previews are never
overwritten by the generator. `--show` opens the browser only when requested.

## Questions, controls and justified observations

| # / CLI experiment | Artifact and mathematical operation | What to change or inspect | Evidence boundary |
|---|---|---|---|
| 01 `architecture` | `01_multi_head_attention_pipeline.gif`: X→Q/K/V→heads→scores→scale→mask→softmax→AV→concat→W_O | Ten stages, actual dimensions and tensor values; parallel heads retain their colors | One random untrained causal forward, not language behavior |
| 02 `subspaces` | `02_representation_subspaces.html` and GIF: XW for Q/K/V | Head, projection, direct 3D/PCA, token pairs; rotate/zoom/hover; exact distances/angles and principal-angle overlap | Direct 3→3 maps are a geometric analogy; separate PCA views lose information and have incomparable axes |
| 03 `geometry` | `03_query_key_value_geometry.html`: dot products, cosine, softmax and weighted values | Constructed head/query, query angle/norm, key norms and scaling; synchronized values and probabilities | Norms affect dot products even when cosines match; no confidence or semantic claim |
| 04 `matrices` | `04_attention_matrix_operations.gif`: matrix products and shape changes | Inspect every PNG stage; highlighted row/column yields an actual result cell; compare packed and per-head projection | Small B=1,T=4,D=8,H=2 numerical fixture, not a performance model |
| 05 `heads` | `05_attention_heads.png` and interactive HTML | Raw/scaled/masked logits or probabilities, query and mask; individual/mean maps and final output | Word labels identify random vectors; mean maps can conceal differences but do not establish usefulness |
| 06 `graph` | `06_token_attention_graph.html` and GIF | One/all heads, one/all queries, threshold, causal/bidirectional; arrows query→key | Routing graph, not a causal explanation or verified linguistic relation |
| 07 `single-vs-multi` | `07_single_vs_multi_head.gif`: shared vs independent channel retrieval | Shared mixture sweeps a line segment; independently routed channels retrieve separate attributes | Constructed fixed-value encoding and manually specified logits; not a universal expressivity/quality ranking |
| 08 `scaling` | `08_attention_scaling.gif` and HTML | Dimension, Q/K magnitudes, scaling; paired dot-product histograms, variance, entropy and max probability | IID normal components, finite Monte Carlo sample and one six-key row; scaling does not imply uniformity |
| 09 `head-count` | `09_head_count_tradeoffs.html`: fixed-width dimensional allocation | Valid D/H, context and precision; inspect 4D² parameters and BHT² map entries | Counts are exact under standard equal-width bias-free MHA assumptions; quality is unmeasured |
| 10 `causal` | `10_causal_attention.gif`: accessibility and future-token intervention | Progressive token reveal, raw logits, mask, probabilities and measured prefix change | Reveal is an availability illustration, not trained generation; causality is tested at layer scope |
| 11 `concatenation` | `11_head_concatenation.gif`: concatenation and blockwise W_O contributions | Track head colors and selected output features; sum head contributions numerically | An average has d_head channels; concatenation has D channels. No trained semantic interpretation |
| 12 `diversity` | `12_head_diversity.html`: map/output cosine, entropy and head intervention | Select query and a head to zero; heads 0/1 share Q/K but retain different V | Output changes in an untrained model do not establish downstream importance or safe pruning |
| 13 `sharing` | `13_mha_gqa_mqa_comparison.gif` and HTML | Valid query/KV divisors; inspect connections, Q/K/V/O parameters and payload | Architectural arithmetic only; no checkpoint conversion, training or quality equivalence |
| 14 `kv-cache` | `14_kv_cache_explorer.html`: theoretical storage formula | Layers, query/KV heads, width, batch, tokens, FP32/FP16/BF16/INT8/packed INT4; curves, bars, 3D surface and capacity plane | Payload excludes weights, activations, metadata, scales, allocation and cache management; plane is hypothetical |
| 15 `complexity` | `15_attention_complexity.html`: prefill/decode MACs, map/cache storage | Width, heads and token count; linear/quadratic growth and rotatable storage surface | Arithmetic, storage and bandwidth are distinct; no latency or peak-memory measurement |
| 15b `flash` | `15_flash_attention_concept.gif`: tiled online softmax | Follow block maximum, normalizer and accumulated weighted values | Executed educational CPU identity, not a FlashAttention GPU kernel; dense pairs remain quadratic |
| 16 `validation` | `16_numerical_validation.html` | Inspect actual new parity, causality, variance and online-softmax results | Software/numerical checks do not replace author interpretation review |

The same head palette is used throughout. Token colors remain stable across
the original and projected coordinate plots. Probability maps share scale
[0,1]. Signed tensors and representations have their own signed scales and
must not be read as probabilities. Masked negative infinity is rendered as
blank in Plotly, labeled “−∞”, and never converted into a finite logit.

## Numerical contracts and scientific assumptions

[from_scratch.py](from_scratch.py) remains the attention implementation.
Its optional `return_trace=True` adds intermediate tensors to the normal
output/maps without changing the default return contract. The visual lab does
not maintain a second attention forward implementation. [visual_core.py](visual_core.py)
provides fixtures, exact geometry, count formulas and controlled interventions.
[example.py](example.py) supplies the identical-weight PyTorch reference.

The default visual fixture uses `(B,T,D)=(1,6,16)`, four heads of width four,
float64, no bias/dropout and zero training updates. The matrix-animation fixture
uses `(1,4,8)` with two width-four heads. Token text is an index label, not a
semantic embedding. Direct 3D projection examples use random 3×3 matrices;
the higher-dimensional projection examples use the actual 16×4 head blocks.

The original and each head's projected cloud receive **separate centered SVD
PCA fits**. Retained variance is shown for each fit. Rotation and translation
can differ across views, and discarded coordinates change distances. Pairwise
distance/angle readouts use all coordinates before PCA. Q-column-span overlap
uses principal angles from orthonormal bases; a low angle does not imply
functional redundancy, and arbitrary heads need not be orthogonal.

For scaling, each sample pair has independent N(0,1) components; dimensions
2–128 use nested prefixes of the same samples. Thus raw variance is expected
to be $d_h$ and scaled variance approximately one before magnitude changes.
Magnitude factors $a,b$ multiply these expectations by $a^2b^2$. Entropy is
in nats and includes exact zero probability as $0\log 0=0$.

The constructed routing case holds values fixed: the entity source has payload
`(1,0)` and the numeric source `(0,10)`, with position/context fields identifying
the sources. A shared distribution produces $(\alpha,10(1-\alpha))$; two
independent scalar value channels can approximately retrieve $(1,10)$ with
manually specified opposing logits. An arbitrary output projection or changed
encoding changes this comparison, so it cannot prove universal superiority.

For query heads $H_Q$, KV heads $H_{KV}$ and equal head width $d_h$, set
$D=H_Qd_h$. Bias-free Q/K/V/output parameters total

$$
P=2D^2+2D H_{KV}d_h.
$$

With $H_{KV}=H_Q$, this reduces to $4D^2$. Grouping requires
$H_Q\bmod H_{KV}=0$. The cache payload is

$$
C_{KV}=2BLT H_{KV}d_h s\quad\text{bytes}.
$$

The factor two stores both K and V. $B$ is batch, $L$ layers, $T$ cached tokens,
and $s$ bytes per element (including ideal $s=0.5$ for packed INT4).
FP16 and BF16 have the same payload width. Quantized calculations omit scales,
zero points, packing alignment and metadata; they are lower-bound payload
models, not allocated memory measurements.

Dense prefill arithmetic counts use four projections $4BTD^2$ MACs and
QKᵀ/AV $2BT^2D$ MACs. A single cached decoding query uses approximately
$4BD^2$ projection MACs and $2BTD$ attention MACs, with a linearly growing cache.
These formulas omit softmax and other block operations. Explicit probability
maps use $BHT^2$ elements only when materialized. Memory bandwidth is a device
property and traffic is implementation dependent; neither is measured here.

The CPU online-softmax example maintains a running maximum, normalizer and
value accumulator while visiting key blocks. It matches the dense result for
the tested head. It does not implement GPU SRAM tiling, backward recomputation
or kernel scheduling. See the [FlashAttention primary reference](references.md).

## Validation and artifacts

```powershell
python scripts/validate_repo.py tests --scope 05-transformers-llms/68-transformer-architecture-ii
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

[Numerical visual tests](tests/test_visual_core.py) cover trace identities,
fused/separate projections, PCA geometry, principal angles, paired variance,
constructed routing, equal-map/distinct-value controls, ablations, grouping,
cache precision and tiled-softmax agreement. [Export tests](tests/test_visual_exports.py)
write actual HTML/GIF/PNG files into temporary directories. The generator
decodes every raster frame, verifies GIF timing/looping and checks local HTML
references. A browser check is separate: initialization, controls, camera
interaction, console errors and external requests need actual inspection.

Each new run saves `report.json` with UTC timestamp, versions, seed streams,
configuration, hypotheses, actual results, interpretation candidates, limits,
artifact inventory and export-decoding evidence. Generation time is an observed
local duration, not an inference benchmark. These visual runs remain distinct
from the [earlier analytical record](notes.md#execution-record--2026-11-25).

Full HTMLs, GIFs, frame PNGs, reports and browser diagnostics remain ignored
under `outputs/` or local workflow records. Only a small deliberately inspected
preview selection is suitable for public documentation. Export validation
does not establish GitHub rendering, broad accessibility, cross-browser
compatibility, native-GPU performance or scientific author approval.

## Executed visual record — 2026-11-25

The AI coding assistant executed the complete collection with:

```powershell
$py = ".local/environments/n09-full-2026-11-13/Scripts/python.exe"
& $py 05-transformers-llms/68-transformer-architecture-ii/visualize.py --all
```

The final full export is retained locally in
`outputs/run-20261125T123446140910Z/`, with `report.json`, an index, the animation
gallery and standalone pages. It contains **10 GIFs, 13 HTMLs and 83 PNGs**;
82 PNGs are individual animation stages. All 106 raster/HTML artifacts passed
the generator's decoding and local-reference checks. Earlier generated runs
remain separate; presentation fixes did not replace their records. The final
record includes source SHA-256 hashes, six structured experiment records,
versions, configuration and full-precision measurements.

Configuration: seed 168; synthetic untrained `(1,6,16)` fixture with four
width-four heads; matrix example `(1,4,8)` with two heads; CPU float64; zero
bias/dropout/updates; 4,000 paired scaling samples; 90 dpi; GIF stages last
1,200 ms and loop. Versions: Python 3.11.14, NumPy 2.4.6, PyTorch 2.15.0+cpu,
Matplotlib 3.11.2, Plotly 6.9.0 and Pillow 12.3.0 on Windows x64.

| Hypothesis / control | Actual result | Interpretation candidate — author review required | Limitation |
|---|---|---|---|
| Matched-weight NumPy/PyTorch forward parity | Largest output error 4.440892e-16; largest map error 2.220446e-16 | Tested tensors agree within 1e-12 | Small CPU float64 fixture, not optimized kernels or gradients |
| Future token 4 cannot affect causal positions 0–3 | Causal prefix change 0; bidirectional change 5.098015 | Mask blocks the tested future influence | Layer-level synthetic intervention |
| Equal Q/K need not imply equal value outputs | Heads 0/1 map error 0; head-output L2 difference 3.716065 | Identical routing does not establish equivalent contributions | Untrained model; no downstream importance measurement |
| Online softmax preserves dense attention | Maximum error 2.220446e-16 over three key blocks | The tested block accumulator agrees with dense computation | CPU identity, not a GPU FlashAttention kernel |
| Scaled IID dot-product variance stays near one | Across d=2–128, scaled sample variance 0.936895–1.011173 | Observed dimension scaling is consistent with the stated IID assumptions | Finite paired Monte Carlo sample; no uncertainty interval or task-quality result |
| Independent constructed channels retrieve distinct payloads | Output approximately (0.999994, 9.999939) | Separate routing escapes the restricted shared-mixture line | Manual logits and fixed value encoding; not universal superiority |

The cache formula also returned 4, 1 and 0.125 GiB for 32/8/1 KV heads at
`B=1,L=32,T=8192,d_head=128,16 bits`. These are **analytical calculations**,
not allocated or measured device memory. No model training, linguistic
specialization, model-quality comparison or inference benchmark was executed.

## Selected public previews

Four previews were selected after raster decoding and visual inspection of
representative stages, with byte-identical copies from the final full run:

| Preview | Bytes | Learning question and curation decision |
|---|---:|---|
| [Complete forward pipeline](assets/01_multi_head_attention_pipeline.gif) | 549,887 | Ten actual tensor stages and four parallel head views |
| [Representation subspaces](assets/02_representation_subspaces.gif) | 1,989,998 | Direct 3D and high-dimensional PCA, all Q/K/V/head states |
| [Individual and average head maps](assets/05_attention_heads.png) | 145,414 | Computed causal maps with one shared probability scale and final output |
| [Concatenation and output mixing](assets/11_head_concatenation.gif) | 436,932 | Per-head numerical contributions to output features |

Total selection: **3,122,231 bytes**, below the 6 MiB topic target. The subspace
GIF exceeds the default 1 MiB per-GIF review target intentionally: 24 slow
states preserve all four Q/K/V projections in both direct and PCA modes.
A single still would omit the coordinate-space changes; reducing labels or
states would remove learning content. Only the pipeline is embedded in the
topic README; other previews are linked. Complete exports, HTML runtimes,
frame sets and run/browser records remain ignored. The generator never writes
into `assets/`; future runs cannot silently replace these selected previews.

## Browser review — 2026-11-25

The AI coding assistant used Playwright CLI 0.1.23 with Chrome 158.0.8127.37
on the final exported files. The review completed **179 scenarios** across all
eleven interactive pages, covering every dropdown option in isolation, slider
endpoints via keyboard, dependent valid KV grouping, actual 3D camera dragging
and wheel interaction, corrected 3D axis titles, local navigation, frame selection
and PNG decoding. Representative JavaScript cache/parameter/softmax calculations
matched their expected values: 1 GiB FP16/BF16 GQA payload, 0.25 GiB ideal INT4,
655,360 projection parameters for `(H_Q,H_KV,d_h)=(8,2,64)` and `[0.5,0.5]`
for two zero logits.

The final harness recorded **zero JavaScript/console errors and zero external
page requests**. Three Canvas2D readback-performance warnings occurred.
Desktop viewport was 1440×1000; the gallery and geometry flow also received a
selected 430×900 mobile check. The reported renderer was ANGLE on Intel HD
Graphics 630 via D3D11. This identifies the inspected display path, not a GPU
inference or performance benchmark; numerical reference calculations ran on CPU.

The CLI blocked `file:` navigation, so browser review used a temporary server
bound only to `127.0.0.1`; exported pages still bundle all code/data and perform
no fetches. Direct file opening was not automated. Initial sandbox/cache attempts
and an earlier favicon 404 are separate from the final successful review.
The generator now embeds an empty favicon. Local diagnostics and screenshots
remain ignored; the generator's immutable run record reports its own export
validation separately from this later browser review.

This is a local Chrome flow review, not all parameter combinations, all slider
values, comprehensive accessibility, cross-browser/GitHub rendering, model-quality
validation or author endorsement. All interpretation candidates remain pending.

## Verification snapshot — 2026-11-25

The focused topic suite passed **86 tests**: 42 analytical, 30 visual numerical
and 14 real-export/entrypoint checks, without skips or warnings. The export file
was run again after the final presentation/control updates and passed all 14.
These were isolated scoped runs, not the complete curriculum or CI.

The final repository checks parsed 490 Python files and checked 1,806 local
links across 304 documents. `node --check` accepted the offline controls;
`uv --no-cache --offline pip check --python` against the stable interpreter
reported all 81 installed packages compatible. Tracked diffs and all 15 new
public text files passed whitespace checks. Git emitted a line-ending notice
for the existing module README; no whitespace failure occurred.

Source hashes matched the final run manifest; all four preview copies matched
their source exports byte-for-byte. Git publication candidates exclude every
run output. The private raw-study hash remained unchanged. No new dependencies,
commit, push or deployment occurred; the review browser/server were closed.
