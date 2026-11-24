# Attention: matching and information transfer

## From embeddings to Q, K and V

For input representations $X\in\mathbb{R}^{n\times d_{model}}$, a single
self-attention head forms

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

The projection parameters have shapes $d_{model}\times d_k$ for $W_Q,W_K$
and $d_{model}\times d_v$ for $W_V$. They are trained parameters; Q, K and V
are input-dependent activations. Separate maps allow matching features and
transferred features to differ. Taking Q=K in a small arithmetic fixture is
not a general property of a trained head.

In cross-attention, queries come from X while keys and values come from a
source Y, potentially with a different sequence length and source width.
Both sources must project into the same query/key width $d_k$. Values may
have a different width $d_v$.

The [Transformer paper](references.md) introduced scaled dot-product attention,
not attention as a whole; additive attention predates it. Multiple heads and
full block architecture belong to later roadmap topics.

## Scores, normalization and geometry

For query i and key j, $s_{ij}=q_i^Tk_j/\sqrt{d_k}$. Dot products combine
direction and magnitude; they are not cosine similarities unless norms are
controlled, with the scaling factor still accounted for.

Softmax normalizes independently over the permitted keys for each query:

$$
\alpha_{ij}=\frac{\exp(s_{ij}-c_i)}
{\sum_{\ell\in\mathcal{P}_i}\exp(s_{i\ell}-c_i)},
\qquad j\in\mathcal{P}_i,
\qquad c_i=\max_{\ell\in\mathcal{P}_i}s_{i\ell}.
$$

Blocked weights are zero. Subtracting a permitted row maximum preserves the
distribution and avoids exponential overflow. It cannot repair an already
overflowed dot product: the reference detects that and raises `ValueError`.
Extreme negative score differences may underflow to zero in float64.

The weighted sum is $o_i=\sum_j\alpha_{ij}v_j$. Before dropout or subsequent
projections, this lies in the convex hull of the available values. Coordinate
bounds are a useful necessary check, but do not alone prove convex-hull
membership. Changing V alone leaves A unchanged in this isolated operation;
changing K can change both A and O. The matrix A is generally asymmetric.
Even symmetric QK scores do not guarantee symmetric row-normalized weights.

The softmax Jacobian is

$$
\frac{\partial\alpha_i}{\partial s_j}
=\alpha_i(\delta_{ij}-\alpha_j).
$$

A nearly one-hot distribution can have small local derivatives. This is not
evidence that all gradients in a full network vanish.

## Why the square root?

Assume all query/key coordinates in a pair are independent, centered, with
unit variance. Each product then has variance one, giving

$$
\operatorname{Var}(q^Tk)=d_k,
\qquad \operatorname{Var}\left(q^Tk/\sqrt{d_k}\right)=1.
$$

Scaling controls score dispersion under these assumptions. Learned features
can be correlated and have different norms, so the argument is an initialization
intuition, not a universal variance guarantee. Softmax concentration also
depends on the number of keys and the actual score distribution. A separate
temperature would change that distribution, but is not needed for the standard
operation implemented here.

## Masks define visibility

An additive mask has zero at permitted positions and $-\infty$ elsewhere.
The NumPy API instead accepts a strictly boolean mask named `allowed` that
broadcasts to $(n,m)$, with `True` meaning permitted. Numeric 0/1 masks are
rejected to avoid confusing additive and visibility semantics. A key-padding
mask can have shape $(m,)$; a square causal mask permits $j\leq i$ and includes
the diagonal. Block before normalization: multiplying weights by a mask
after softmax without renormalizing changes row sums.

Every row must have at least one permitted key. This reference raises an
explicit error for an empty permitted set, rather than treating NaN or an
all-zero output as a probability distribution. This is a local API policy;
framework kernels can handle that edge differently.

PyTorch functional SDPA also uses boolean True for permitted positions; its
`MultiheadAttention.key_padding_mask` uses True for blocked padding. The
[versioned SDPA documentation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)
also requires `dropout_p=0.0` to disable dropout. Our comparison supplies an
explicit mask and float64 CPU tensors with batch/head size one.

A causal mask prevents future leakage only if earlier queries and permitted
keys/values are themselves free of future information. It cannot undo leakage
from preprocessing or upstream bidirectional layers. Padding visibility does
not automatically suppress outputs at padded query positions; callers must
handle those positions in loss/output logic.

Rectangular cross-attention is supported. Rectangular decoding with a KV
cache needs a position-aware mask: a fresh single query can attend to the
entire allowed prefix. A naive 1-by-m lower triangle would allow only the
first key. This study implements no KV cache or automatic rectangular causality.

## Order and engineering trade-offs

Without positional information or a position-dependent mask, self-attention
is permutation-equivariant: permuting input rows permutes output rows.
Positional representations and causal visibility add sequence structure;
causal masking does not replace every role of positional representations.

Dense cross-attention costs $O(nm(d_k+d_v))$ for score computation and value
aggregation, with $O(nm)$ score/weight storage in this reference. Self-attention
with comparable widths reduces to $O(n^2d)$ arithmetic. These bounds exclude
projection work, other model activations, gradients and parameters.
[FlashAttention](https://arxiv.org/abs/2205.14135) reorganizes exact attention
to reduce memory traffic and avoid storing a full attention matrix in GPU
high-bandwidth memory. Dense arithmetic remains quadratic; no kernel or speed
comparison is executed here.

In language-model inference, prefill processes prompt queries together; cached
decoding reuses previous keys/values for successive queries. KV memory grows
with context, layers, batch size, KV head count and head width. This explains
an engineering trade-off, not a latency measurement from this study.

External retrieval searches a corpus and supplies selected evidence. Attention
mixes representations of supplied context; it does not replace the retrieval
index. Task quality, latency and robustness need separate evaluation. Likewise,
weights alone do not establish an explanation: values, residual paths, heads
and subsequent layers also affect outputs. See the [interpretability reference](references.md).

## Execution record — 2026-11-24

The AI assistant ran the public `from_scratch.py` and `example.py --torch`
from the repository root on Windows, using Python 3.11.14, NumPy 2.4.6 and
PyTorch 2.15.0+cpu. CPU float64, one PyTorch thread, dropout disabled; no
training, dataset downloads or wall-clock benchmark. This is a new analytical
run; the private source contained proposed code, not earlier execution results.

### Fixed-vector controls

Configuration: $Q=K$ with rows $(1,0),(0,1),(-1,0)$;
V has rows $(2,0),(0,2),(1,1)$. The causal mask is a 3-by-3 lower triangle,
including its diagonal. The unbatched implementation uses row-wise softmax.

| Hypothesis / control | Observed result | Interpretation candidate — author review required | Limitation |
|---|---|---|---|
| The first query matches the worked weighted sum | Weights `(0.575975, 0.283995, 0.140029)`; output `(1.291980, 0.708020)` | Scores determine mixture coefficients, not a hard choice of one value | Manually chosen vectors; no learned relevance |
| Adding `(3,-2)` to every V leaves weights unchanged | Max absolute weight change `0`; first output shift `(3,-2)` | The matching and transfer roles can be separated in one operation | Q and K held fixed; not a full network intervention |
| Changing only future K/V cannot change earlier causal outputs | Last K changed to `(20,-20)` and V to `(100,-100)`; max change over the first two outputs: causal `0`, full `100.707800` | The visibility mask blocks these future interactions | Earlier Q/K/V remain fixed; says nothing about upstream leakage |
| NumPy matches PyTorch SDPA for these full and causal inputs | Max absolute errors: full `2.220e-16`, causal `1.110e-16`; tolerance `rtol=atol=1e-12` passed | The implementations agree within float64 rounding for this fixture | CPU only; no gradients, GPU, low-precision or trained-model validation |

### Scaling protocol

Hypothesis: under independent standard-normal query/key coordinates, dividing
scores by $\sqrt{d_k}$ controls their dispersion and avoids the strong entropy
decrease caused by increasing unscaled dimensions in this setup.

Configuration: `np.random.default_rng(66)`; sequential dimensions 4, 16, 64,
256; fresh Q of shape `(256,d_k)` and K of shape `(64,d_k)` at each dimension.
Each row normalizes over 64 keys. Score standard deviation uses all entries
with `ddof=0`; entropy is the mean row entropy in nats with $0\log 0=0$.

| $d_k$ | Raw score std | Scaled score std | Raw mean entropy | Scaled mean entropy |
|---:|---:|---:|---:|---:|
| 4 | 2.042787 | 1.021394 | 2.871330 | 3.707669 |
| 16 | 3.988690 | 0.997173 | 1.395581 | 3.695420 |
| 64 | 7.969005 | 0.996126 | 0.638898 | 3.692819 |
| 256 | 16.132906 | 1.008307 | 0.251409 | 3.678280 |

Interpretation candidate — **author review required**: scaled dispersion stayed
near one and scaled entropy changed little, while the unscaled distributions
became much more concentrated in this run. This supports the stated Gaussian
intuition for this configuration; it does not establish a universal entropy
trend or improvement in trained-model performance.

Limitations: one seed, 64 keys per query, fresh inputs across dimensions, finite
sample estimates and shared keys across query rows. Matrix entries are not all
independent observations. No confidence intervals, optimizer updates, real
language task, latency or memory measurements were performed. Suggestions
below remain unexecuted.

The isolated topic test file subsequently passed 23 cases (pytest 9.2.0),
including rectangular cross-attention SDPA comparisons. No skips or warnings
were reported. These assertions establish their numerical contracts, not
personal author approval of interpretations or production readiness.

## Analytical handoff suggestions and later visual work

At the analytical handoff, the following were suggestions, not executed results.
The separate [visual laboratory](VISUAL_LAB.md) now implements query rotation,
a different scaling protocol and bounded CPU timing. Their execution records
belong to that guide; they do not replace the historical analytical record
above. Multi-seed replication and training projections remain unexecuted.

- Rotate a unit query against fixed unit keys, then vary norms separately to
  distinguish angular alignment from dot-product magnitude effects.
- Repeat the scaling protocol across independent seeds and different key
  counts before making distributional claims beyond this fixture.
- Train projections on a specified task and evaluate held-out quality;
  synthetic weights alone do not demonstrate contextual semantic learning.
- Measure runtime/memory under a documented implementation and hardware
  protocol if efficiency is the question. No timing benchmark is provided here.
