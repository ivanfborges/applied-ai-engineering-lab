# Information flow, objectives and model selection

## Three choices that should remain separate

Architecture specifies stacks and connections. A mask specifies permitted
query-key pairs. An objective specifies prediction targets. Fine-tuning and
decoding add further choices; none is implied completely by the family name.

A BERT-style encoder returns one contextual vector per input token. A task
head can score a sequence or label individual tokens. Original BERT used MLM
and next-sentence prediction; NSP is not a requirement of an encoder.
For semantic search, pooling arbitrary hidden states does not establish useful
sentence geometry. Use retrieval training and held-out relevance evaluation.
See the [BERT paper](https://aclanthology.org/N19-1423/).

A GPT-style stack has causal self-attention and no mandatory encoder memory.
A complete prompt can be processed in parallel, but position $i$ still cannot
see later prompt positions. Its last prompt state can use the entire prompt.
The original [GPT paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
also illustrates why causal representations are useful for discriminative
tasks; generation and contextual representation are not exclusive capabilities.

An encoder-decoder separates input representation from target prediction.
Every target query can read the valid encoded source, while target
self-attention stays causal. This is valid when the complete source is
available before decoding; streaming tasks need additional constraints.
The [T5 paper](https://www.jmlr.org/papers/v21/20-074.html) uses a shared
text-to-text interface and span-corruption pretraining. Architecture alone
does not mandate that objective.

## Shared attention and different origins

For already projected matrices $Q\in\mathbb{R}^{m\times d_k}$,
$K\in\mathbb{R}^{n\times d_k}$, $V\in\mathbb{R}^{n\times d_v}$:

$$
A=\operatorname{softmax}_{\text{keys}}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),
\qquad O=AV\in\mathbb{R}^{m\times d_v}.
$$

$M_{ij}=0$ permits a connection; $M_{ij}=-\infty$ blocks it. Encoder
self-attention permits all valid source keys. Causal self-attention additionally
requires $j\le i$ for positions starting at the same origin. Cross-attention
uses $Q=H_dW_Q$, $K=H_eW_K$, $V=H_eW_V$, producing an $m\times n$ weight matrix.
There is no general reason to impose a triangular mask on source-target indices:
source position 4 is not a future *target* token.

`visibility` uses Boolean `True = permitted`, as does functional PyTorch SDPA.
Transformer modules use `True = blocked` for their Boolean masks, so
`causal_block_mask` explicitly inverts visibility. See the
[SDPA mask contract](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html).
All masks and Q/K/V passed to the NumPy implementation must have exact,
unbatched shapes. Cached rectangular causal attention needs position offsets;
the helper rejects that unsupported case instead of silently aligning it.

Padding masks hide key/value positions, not the outputs at padded queries.
This matters twice in an encoder-decoder: source padding must be excluded
inside the encoder and when its states become decoder memory. Hiding memory
alone cannot remove information already mixed into valid encoder states.
Losses and pooling must also exclude padded query positions as appropriate.
The educational implementation rejects fully blocked rows because no
normalized probability distribution exists over an empty support. This is its
explicit contract, not a claim that all library kernels have the same policy.

## Objectives, corruption and target alignment

For selected MLM positions $S$ and corrupted input $\widetilde{x}$:

$$
\mathcal{L}_{\mathrm{MLM}}=-\frac{1}{|S|}\sum_{i\in S}\log p_\theta(x_i\mid\widetilde{x}).
$$

The loss is over selected positions, not every visible input token. Original
BERT selected 15% of token positions and used an 80/10/10 replacement mixture
of mask/random/unchanged tokens among them. This corruption mask is distinct
from an attention mask: valid input positions remain bidirectionally visible.
This study does not implement the original corruption sampler or NSP.

For causal modeling, with the initial context/start convention explicit:

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}),\qquad
\mathcal{L}_{\mathrm{CLM}}=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

For conditional generation:

$$
p_\theta(y\mid x)=\prod_{t=1}^{m}p_\theta(y_t\mid y_{<t},x),\qquad
\mathcal{L}_{\mathrm{seq2seq}}=-\frac{1}{m}\sum_{t=1}^{m}\log p_\theta(y_t\mid y_{<t},x).
$$

These averages assume unpadded sequences; real batches need an explicit valid
target count. When conditioning a decoder-only model on a source prompt,
whether prompt tokens also contribute to the loss must be specified.

Teacher forcing makes known target prefixes available at every position in
one forward pass. For synthetic labels `[4, 7, 2]`, where `2` represents EOS,
the seq2seq inputs are `[start, 4, 7]`; position 0 predicts `4`, not `start`.
The causal diagonal is permitted because that input contains the *previous*
target. With unshifted labels, the residual path could expose the current
target even if the attention diagonal were blocked. `shift_right` requires
the start ID explicitly and neither appends EOS nor handles padded batches.
For a causal sequence `[start, a, b, eos]`, the equivalent slices are inputs
`[start, a, b]` and labels `[a, b, eos]`. Some model APIs shift internally;
verify the contract before preparing labels to avoid a double shift.

T5 span corruption replaces each selected contiguous span with a distinct
sentinel. For the code-defined text `A B C D E`, corrupting `B C` and `E`
could give source `A <s0> D <s1>` and target `<s0> B C <s1> E <s2>` (plus the
tokenizer's end marker). The target reconstructs missing spans, not every
source token. This is a schematic fixture, not a tokenizer execution.

Teacher-forced loss evaluates conditionals under ground-truth prefixes;
autoregressive inference conditions on generated prefixes. A model can fit
the former and fail in free-running use. No loss or generation is executed
in this day's random-block experiment; see [Day 67](../67-transformer-architecture-i/)
for a separate trained synthetic sequence example.

## Cost and application trade-offs

For conventional dense attention with source length $n$, target length $m$
and comparable width $d$, interaction work scales as $O(n^2d)$ for the
encoder, $O(m^2d)$ for target self-attention and $O(nmd)$ for cross-attention.
A decoder-only concatenation instead has total length $n+m$. Triangular
visibility does not imply a dense kernel performs exactly half the FLOPs.
These expressions omit projections, FFNs, vocabulary heads, batch size,
layer counts and kernel choices; they cannot establish serving speed.

A bi-encoder can precompute document vectors. A cross-encoder scores each
query-document pair jointly and therefore costs another evaluation per pair.
A generator can then condition on retrieved evidence. This decomposition
requires separate retrieval recall/ranking metrics and answer-quality checks;
architecture names cannot establish the best pipeline.

For autoregressive serving, a KV cache retains prior per-layer keys and
values. Prefill processes the prompt/source; subsequent tokens reuse cached
states. Encoder-decoder serving can also retain source memory and projected
cross-attention keys/values. Cache memory depends on layers, KV heads, head
width, sequence length, batch and dtype. Neither cache use nor latency is
measured here. The experiment runs full forwards without any cache.

Classification may favor a compact encoder or a classical baseline; variable
text output needs a generative interface. Translation is naturally expressed
as source-to-target generation, but this does not guarantee T5-style models
beat prompted decoders. Compare suitable checkpoints on task data, factuality,
output constraints and serving budgets. Conditional access does not guarantee
the model uses the source correctly or avoids hallucination. Attention weights
alone are also insufficient as causal explanations of predictions.

## Execution record — 2026-11-25

**Status:** executed by Codex; interpretation candidates require author review.
This is a new analytical run, not a rerun of another day's records.

**Hypothesis:** changing a future target cannot change earlier causal outputs;
a valid source change may reach every target; a source change hidden by both
padding masks cannot reach valid encoder outputs or target outputs.

**Configuration:** Windows, Python 3.11.14, NumPy 2.4.6, PyTorch 2.15.0+cpu;
seed 71, CPU float64, one thread, batch 1. Source shape $(1,5,16)$, decoder
input shape $(1,3,16)$. One layer per stack, four heads, FFN width 64, ReLU,
Post-LN, default layer-norm epsilon $10^{-5}$, dropout zero, evaluation and
inference modes. The encoder and causal stack have identical copied weights;
the cross-attention decoder has separate random parameters. Perturbation is
$+10$ in feature 0 of the last source or target position, in separate controls.
The padding control marks only the last source position invalid in both masks.
All vectors are independent synthetic normal draws. Zero optimizer updates.

Commands from the repository root:

```sh
python 05-transformers-llms/71-encoder-decoder-models/from_scratch.py
python 05-transformers-llms/71-encoder-decoder-models/example.py --output 05-transformers-llms/71-encoder-decoder-models/outputs/analytical_2026-11-25_seed71.json
```

The retained local JSON records UTC execution time
`2026-11-25T20:09:00.367558+00:00`. It is ignored and is not a public link.
Use another filename for a new run; keep this dated summary unchanged.

**Results:** position-wise output L2 differences, rounded to six decimals:

| Perturbation and output | Position-wise L2 difference |
|---|---|
| Valid source change → bidirectional source states | `[1.287663, 1.040008, 2.027258, 1.110425, 4.080962]` |
| Same source-vector change → causal states | `[0, 0, 0, 0, 4.080962]` |
| Valid source change → encoder-decoder targets | `[0.452726, 0.474293, 0.773795]` |
| Last target input change → encoder-decoder targets | `[0, 0, 3.723025]` |
| Masked source change → valid encoder states | `[0, 0, 0, 0]` |
| Masked source change → encoder-decoder targets | `[0, 0, 0]` |

The invariance tolerance was $10^{-12}$. Zero entries above were exactly zero
in this run. NumPy/CPU SDPA rectangular cross-attention maximum absolute
difference was $1.6653\times10^{-16}$. NumPy demo weight shapes were $5\times5$
(bidirectional and causal) and $3\times5$ (cross); largest row-sum error was
$2.2204\times10^{-16}$. Blocked probabilities were exactly zero. Seq2seq
alignment printed inputs `[0, 4, 7]` and labels `[4, 7, 2]`.

**Interpretation candidate — author review required:** these controls are
consistent with causal prefix isolation and source conditioning through
cross-attention. Bidirectional availability allowed earlier source states to
change in this initialization. Positive deltas are observations, not universal
lower bounds: permitted paths may have zero effect for other weights or inputs.

**Limitations:** one random seed and one perturbation magnitude; no pretrained
models, tokenizer, positional encodings, language-model heads, gradients,
training, free-running generation, cached inference or task-quality evaluation.
The NumPy attention core is a single-head reference rather than a complete
model; the PyTorch stacks illustrate connectivity rather than reproduce the
normalization, positions or parameterization of BERT/GPT/T5. L2 magnitudes
do not measure linguistic understanding or rank architectures. The NumPy/SDPA
comparison tests a small numerical case, not every kernel or dtype.

Local validation on the same date used the stable Python environment above:

```sh
python scripts/validate_repo.py tests --scope 05-transformers-llms/71-encoder-decoder-models
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
uv --offline --no-cache pip check --python .local/environments/n09-full-2026-11-13/Scripts/python.exe
git diff --check
```

The scoped runner passed 43 tests in one isolated file, with no skips or
warnings. Repository-wide parsing covered 511 Python files; the local-link
check covered 1,901 targets in 318 files. Dependency compatibility passed for
81 installed packages, and the whitespace check passed. These are local
checks, not the complete numerical suite, Linux CI, GitHub rendering or
author approval of the interpretation.

## Suggested follow-ups — not executed here

- Repeat perturbations across seeds, layers and positions, separating exact
  causal invariance from variation in visible-path magnitudes.
- Implement each corruption objective and task head, then train on a declared
  dataset with an untouched evaluation split and free-running checks.
- Compare adapted pretrained checkpoints on one task, recording tokenizer,
  truncation, decoding policy, parameter budgets and measured serving costs.
- Check cached decoding against full forwards with correct absolute offsets.

No conclusions from these proposed studies are claimed.
