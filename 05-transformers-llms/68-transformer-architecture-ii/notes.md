# Multi-head attention: projections, routing and output mixing

## What multiple heads add

A single head supplies one distribution over keys for each query. Multiple
heads supply several distributions and separate value channels before mixing
their outputs. With fixed model width, each head is narrower; this changes
the factorization of the attention operation rather than simply replicating
a full-width head. It does not establish a universal quality ordering between
single-head and multi-head models.

Each head's projection uses all input features. Splitting projected columns
into heads is different from splitting the raw embedding into fixed feature
groups. Projection spaces may overlap and need not be orthogonal or full rank.
Heads share a training objective and interact through output mixing; separate
softmax operations do not imply statistical independence.

## Equations and dimensions

Let $X\in\mathbb{R}^{B\times T\times D}$, with $H$ heads and
$d_h=D/H$. This implementation assumes equal query/key/value head widths and
equal input/output model widths, with no bias. For head $i$:

$$
Q_i=XW_i^Q,\qquad K_i=XW_i^K,\qquad V_i=XW_i^V,
\qquad W_i^Q,W_i^K,W_i^V\in\mathbb{R}^{D\times d_h}.
$$

$$
A_i=\operatorname{softmax}_{\text{keys}}
\left(\frac{Q_iK_i^\top}{\sqrt{d_h}}+M_i\right),
\qquad Z_i=A_iV_i.
$$

Here the transpose operates on the last two axes. An additive mask is zero
for permitted entries and negative infinity for blocked entries. There must
be at least one permitted key per query per head.

$$
Y=\operatorname{Concat}(Z_1,\ldots,Z_H)W^O,
\qquad W^O\in\mathbb{R}^{D\times D}.
$$

Writing $W^O$ in row blocks $W_i^O$ also gives
$Y=\sum_i Z_iW_i^O$. This identity clarifies how output mixing combines head
contributions; standard MHA does not average head outputs. Averaging displayed
maps is a separate inspection choice that discards per-head routing detail.

| Stage | Shape in this implementation | Example with B=2, T=6, D=32, H=4 |
|---|---|---|
| Input, each packed Q/K/V projection | $(B,T,D)$ | `(2,6,32)` |
| Reshape then transpose | $(B,H,T,d_h)$ | `(2,4,6,8)` |
| Scores and probabilities | $(B,H,T,T)$ | `(2,4,6,6)` |
| Weighted values | $(B,H,T,d_h)$ | `(2,4,6,8)` |
| Transpose back then concatenate | $(B,T,D)$ | `(2,6,32)` |
| Output projection | $(B,T,D)$ | `(2,6,32)` |

Packed projection matrices concatenate per-head columns, e.g.
$W^Q=[W_1^Q\mid\cdots\mid W_H^Q]$. Thus $XW^Q$ computes all query
projections at once. A reshape alone cannot replace the transpose: tokens and
heads occupy different axes. The independent loop oracle in the tests protects
this ordering. PyTorch stores linear weights for multiplication by their
transpose; the reference copies `[Wq | Wk | Wv].T` into `in_proj_weight` and
`Wo.T` into `out_proj.weight`.

## Scaling, masks and numerical assumptions

If query/key components are independent, centered and unit variance, with
uncorrelated products across coordinates, then
$\operatorname{Var}(q\cdot k)=d_h$. Dividing by $\sqrt{d_h}$ controls this
variance under those assumptions. Trained or correlated projections need not
satisfy them. Scaling uses head width, not model width. For a fixed logit row,
increasing a positive inverse temperature concentrates its softmax; this says
nothing about whether that concentration improves a downstream task.

The forward pass subtracts the maximum permitted logit before exponentiation.
Blocked entries become exactly zero probability. Large finite logits can still
overflow during projection or dot products, before softmax; this implementation
raises an explicit error on detected overflow instead of claiming stability for
all finite inputs. Float64 is an educational comparison choice, not an inference
precision recommendation.

`blocked_mask` uses `True = blocked`, matching Day 67 and PyTorch MHA. Day 66's
NumPy visibility mask and functional PyTorch SDPA use `True = permitted`; invert
the meaning when porting. A causal mask blocks columns greater than the query
row. Key padding broadcasts across heads and queries. Combining padding and
causal masks can fully block a query, especially with left padding; this layer
rejects that case. It does not silently invent a distribution. Padding does not
zero padded-query outputs or remove them from a training loss.

`need_weights=True, average_attn_weights=False` preserves per-head maps in the
PyTorch reference. This is an inspection path; requesting maps can affect kernel
selection. When maps are unnecessary, the documented `need_weights=False`
option permits optimized execution where supported. No such speed comparison
was performed here. See the versioned [API reference](references.md).

## Parameters, arithmetic and memory

For this equal-width bias-free MHA, $P=4D^2$; adding the usual four biases gives
$4D^2+4D$. Changing $H$ at fixed $D$ preserves projection parameters, but changes
head width, routing and the number of probability entries. Unequal head sizes
or shared KV projections change these assumptions.

Dense self-attention uses $O(BTD^2+BT^2D)$ arithmetic. Explicitly retaining maps
uses $BHT^2$ elements. At fixed width the leading attention arithmetic does not
multiply by heads because each head becomes narrower, though actual runtime
still depends on layout, kernel and hardware. This NumPy implementation
materializes scores, probabilities and masks; their combined memory exceeds
one map's size.

Cross-attention instead uses target queries and separate source keys/values,
with maps $(B,H,T_q,T_k)$. It is discussed here but not implemented in this
day's self-attention layer. Encoder self-attention may be bidirectional; decoder
self-attention needs causal masking for autoregressive modeling. Position
information is a separate topic in [Day 69 of the roadmap](../../ROADMAP.md).

## MHA, GQA, MQA and KV caching

| Design | Query heads | KV heads | Sharing |
|---|---:|---:|---|
| MHA | $H$ | $H$ | Separate K/V for each query head |
| GQA | $H$ | $G$, typically dividing $H$ | K/V shared within query-head groups |
| MQA | $H$ | 1 | One K/V head shared by all query heads |

For equal key/value widths, a theoretical cache payload is

$$
C_{\mathrm{KV}}=2BLT H_{\mathrm{KV}}d_h s\quad\text{bytes},
$$

where $L$ is layer count and $s$ bytes per element. For a hypothetical
$B=1$, $L=32$, $T=8192$, $H=32$, $d_h=128$, $s=2$, the payload is 4 GiB for
MHA, 1 GiB for GQA with eight KV heads, and 0.125 GiB for MQA. These are
**analytical payload calculations**, excluding weights, temporary activations,
cache allocation/fragmentation and runtime overhead. No cache was allocated or
profiled. GQA changes projection sharing; it is not a mask toggle on an existing
MHA checkpoint. Model conversion and quality require separate validation.

FlashAttention changes dense-attention execution through tiling and online
softmax, reducing memory traffic and avoiding full map materialization. It does
not remove dense attention's quadratic pairwise arithmetic. It can coexist
with multi-head or shared-KV designs. Neither FlashAttention nor GQA/MQA is
implemented or benchmarked here; see [primary sources](references.md).

## What an attention map cannot establish

Different maps may route complementary information, but diversity alone does
not demonstrate usefulness. Identical maps can still multiply distinct value
projections and contribute differently through $W^O$. A hand-computed test
constructs exactly this case. Head ablations need downstream evaluation and
actual latency measurements to justify pruning; similar maps are insufficient.

Contextual attention can appear in embedding models, joint query-document
rerankers and generators. Joint attention in a cross-encoder reranker does not
necessarily mean an encoder-decoder cross-attention layer. Inspecting weights
does not establish answer groundedness, causal feature importance or confidence.

## Execution record — 2026-11-25

The AI coding assistant executed a new analytical run on 2026-11-25 at
11:45:15 UTC. This record does not relabel illustrative study material or any
earlier day's execution as new evidence. **Interpretation candidates below
require author review**; no personal author conclusions are claimed.

Commands from the repository root, using the existing stable interpreter:

```powershell
$py = ".local/environments/n09-full-2026-11-13/Scripts/python.exe"
& $py 05-transformers-llms/68-transformer-architecture-ii/from_scratch.py
& $py 05-transformers-llms/68-transformer-architecture-ii/example.py
& $py scripts/validate_repo.py tests --scope 05-transformers-llms/68-transformer-architecture-ii
```

Environment: Windows x64 (`Windows-10-10.0.19045-SP0`), Python 3.11.14,
NumPy 2.4.6, PyTorch 2.15.0+cpu and pytest 9.2.0. The PyTorch example sets
one PyTorch thread. Tests use the repository runner's one-thread child settings.
All attention calculations use CPU float64, zero bias, zero dropout and zero
training updates. There are no language labels or external datasets.

The default example uses seed 68 for a local NumPy generator and for each
projection initializer. The standalone NumPy smoke uses input seed 68 and
projection seed 42. The JSON record is retained locally as
`outputs/run-20261125T114515495751Z.json` (ignored, deliberately not linked as
a public asset). It contains full-precision results and provenance; the tables
below round them for readability. Future runs create separate dated records.

### 1. Numerical parity

**Hypothesis:** identical Q/K/V/output weights produce matching outputs and
per-head maps in the two implementations, including masking.

**Configuration:** $B=2$, $T=6$, $D=32$, $H=4$, $d_h=8$; independent
standard-normal synthetic vectors. One final key is padded in batch element
zero. Evaluate bidirectional and causal attention with the same input and
weights, bias/dropout disabled. Compare at `atol=rtol=1e-12`.

| Mask | Maximum output absolute error | Maximum map absolute error | Maximum row-sum error | Blocked-key weight |
|---|---:|---:|---:|---:|
| Bidirectional + padding | 6.661338e-16 | 3.330669e-16 | 2.220446e-16 | 0 |
| Causal + padding | 1.332268e-15 | 3.330669e-16 | 2.220446e-16 | 0 |

**Interpretation candidate — author review required:** these tested forward
calculations agree at float64 precision, including per-head maps and mask
composition. **Limitation:** small deterministic CPU cases do not establish
correctness for every input, backpropagation, mixed precision or kernel speed.

### 2. Causal perturbation control

**Hypothesis:** modifying positions 3–5 cannot change outputs at positions 0–2
under causal self-attention.

**Configuration:** reuse the seed-68 layer and input, without padding. Add
`3 * standard_normal` to positions 3 onward using the continuing generator
state. Keep weights fixed. Compare causal and bidirectional outputs.

**Result:** maximum causal-prefix change was 0; largest future attention weight
was 0. The bidirectional control's maximum prefix change was 4.804524.
The separate standalone smoke returned input/output shape `(2,6,32)`, maps
`(2,4,6,6)`, row-sum error 2.220446e-16 and future weight 0.

**Interpretation candidate — author review required:** the observed mask
blocks future influence within this layer. **Limitation:** this is not an
end-to-end autoregressive training or data-leakage audit.

### 3. Fixed-width head count and scaling

**Hypothesis:** at fixed standard width, more heads preserve projection
parameters and increase explicit map entries; removing head-width scaling
concentrates distributions for the same projected logits.

**Configuration:** new synthetic `(2,6,64)` input drawn from the continuing
seed-68 generator. All six head configurations receive exactly the same packed
matrices initialized with seed 68. Use bidirectional attention without padding.
Compare scaled/unscaled calculations within each head configuration; the latter
is a controlled departure from standard MHA. Entropy is averaged across all
batch/head/query rows in nats, with $0\log 0=0$.

| Heads | Head width | Projection parameters | Explicit map entries | Scaled mean entropy | Unscaled mean entropy |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 16,384 | 72 | 1.491324 | 0.250106 |
| 2 | 32 | 16,384 | 144 | 1.519664 | 0.433372 |
| 4 | 16 | 16,384 | 288 | 1.472587 | 0.595630 |
| 8 | 8 | 16,384 | 576 | 1.459280 | 0.781767 |
| 16 | 4 | 16,384 | 1,152 | 1.451632 | 1.042026 |
| 32 | 2 | 16,384 | 2,304 | 1.486444 | 1.325453 |

**Interpretation candidate — author review required:** the parameter/map
counts follow the fixed-width identities, and scaling increases entropy within
each tested configuration. **Limitation:** head count changes grouping, head
width and the scaling denominator simultaneously. Cross-row entropy differences
are not evidence of semantic diversity or task quality. No model was trained,
no seed distribution estimated and no latency or peak-memory benchmark run.

### Automated verification

The focused test file passed **42 tests**, with no skips or warnings. It covers
an independent per-head loop, hand-computed values/output mixing, multiple head
counts, framework parity, batch/head-specific masks, padding, causal invariance,
large logits, overflow errors, invalid inputs, deterministic measurements and
preservation of existing JSON records. These checks remain separate from
scientific interpretation and human approval.

## Suggested follow-ups — not executed

- Train matched single-head/multi-head models with held-out tasks, multiple
  seeds and an explicit selection policy before comparing quality.
- Extend the separate visual lab's untrained map/output-similarity and zero-head
  controls to a trained model with downstream evaluation; that trained follow-up
  remains unexecuted. Distinguish routing overlap from complete head redundancy.
- Profile inference with and without map requests on a named kernel/device,
  holding shapes and precision fixed. Report latency and peak memory separately.
- Implement cross-attention or shared KV heads only with a separate shape,
  cache and masking contract and corresponding correctness checks.
