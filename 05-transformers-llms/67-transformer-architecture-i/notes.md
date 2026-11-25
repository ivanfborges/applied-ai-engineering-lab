# Transformer Architecture I — composition and sequence contracts

## From attention to an architecture

An attention operation computes a content-dependent mixture of value vectors.
A Transformer repeatedly combines that routing with nonlinear feature
transformations. The original encoder processes the source bidirectionally;
the decoder uses a causal target prefix and the encoder's final memory.
This supports conditional sequence tasks such as translation and summarization.

| Operation | Queries | Keys and values | Connectivity |
|---|---|---|---|
| Encoder self-attention | Source states | Source states | All valid source positions |
| Decoder self-attention | Target states | Target states | Current and earlier target input positions |
| Decoder cross-attention | Decoder states after self-attention | Final encoder memory | All valid source positions |

The source is assumed fully available. Streaming sources require a different
visibility policy. Cross-attention is generally rectangular, and applying the
decoder's triangular mask to it would impose an unjustified source alignment.
Padding masks and causal masks address different restrictions.

## Equations and tensor flow

For one head, with query length $m$ and source length $n$:

$$
Q=X_qW_Q,\quad K=X_kW_K,\quad V=X_kW_V,
\qquad A=\operatorname{softmax}_{\text{keys}}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),
\qquad H=AV.
$$

$M$ is zero for allowed positions and $-\infty$ for blocked ones. Each row
must retain at least one visible key. The NumPy implementation rejects fully
blocked rows instead of normalizing an undefined distribution. Its mask API uses
boolean `True` for blocked positions, matching `nn.Transformer`; direct PyTorch
SDPA uses the inverse boolean convention (see [references](references.md)).

Multi-head attention concatenates head outputs and projects them back to
$d_{\text{model}}$. The PyTorch example uses four heads of width eight; the NumPy
block uses one head to keep composition inspectable. Head specialization is
learned, not guaranteed or manually assigned.

Ignoring dropout for the equations, the original post-norm composition is:

$$
U=\operatorname{LN}(X+\operatorname{MHA}(X,X,X)),\qquad
H_e=\operatorname{LN}(U+\operatorname{FFN}(U)).
$$

$$
A_d=\operatorname{LN}(Y+\operatorname{MaskedMHA}(Y,Y,Y)),
$$

$$
B_d=\operatorname{LN}(A_d+\operatorname{MHA}(A_d,H_e,H_e)),\qquad
D=\operatorname{LN}(B_d+\operatorname{FFN}(B_d)).
$$

The same final $H_e$ is available to each decoder layer. FFN parameters are shared
across positions, with $\operatorname{FFN}(x)=\operatorname{ReLU}(xW_1+b_1)W_2+b_2$.
Layer normalization computes mean and variance over **features of each token**,
not over sequence positions or examples. NumPy omits its learned scale and bias;
PyTorch retains them. Pre-norm instead computes $x+F(\operatorname{LN}(x))$,
altering the residual gradient path. No normalization comparison was executed.

| Default PyTorch tensor | Shape | Meaning |
|---|---|---|
| Source IDs | $(B,5)$ | Content tokens |
| Embedded source / encoder memory | $(B,5,32)$ | Model features |
| Shifted decoder IDs | $(B,6)$ | BOS followed by reversed source |
| Embedded decoder / decoder states | $(B,6,32)$ | Model features |
| Decoder visibility mask | $(6,6)$ | Block strictly upper triangle |
| Vocabulary logits | $(B,6,32)$ | Scores over token IDs |

Vocabulary size and model width happen to be equal here; their meanings differ.
`nn.Transformer` supplies neither the token lookup nor positional information nor
the vocabulary projection. The example adds all three, using
$\sqrt{d_{\text{model}}}E+P$ and a learned position table.

The NumPy block receives synthetic continuous representations and adds
sinusoidal positions:

$$
PE_{p,2i}=\sin\left(p/10000^{2i/d}\right),\qquad
PE_{p,2i+1}=\cos\left(p/10000^{2i/d}\right).
$$

Content-only unrestricted self-attention is permutation-equivariant, not
permutation-invariant. Positions enable order-sensitive representations.
Causal connectivity itself also introduces order-dependent structure; removing
positional embeddings does not remove every possible order signal in a decoder.
Neither sinusoidal values nor learned positions guarantee length extrapolation.

## Why shifting and masking are both necessary

For source `[7, 4, 9]`, the reversal training pair is:

```text
decoder input: [BOS, 9, 4, 7]
target:        [9,   4, 7, EOS]
```

The diagonal of the causal mask is allowed because the current **input** token
belongs to the already available prefix. The next target token appears at the
following decoder-input position, except at the terminal EOS slot. Without
shifting, a residual path can expose the answer at the same position even if
future attention is masked. Without masking, shifted teacher forcing exposes
future answers through self-attention.

The objective implements the conditional factorization:

$$
p_\theta(Y\mid X)=\prod_{t=1}^{m}p_\theta(y_t\mid y_{<t},X),\qquad
\mathcal{L}=-\frac{1}{Bm}\sum_{b=1}^{B}\sum_{t=1}^{m}\log p_\theta(y_{bt}\mid y_{b,<t},X_b).
$$

Teacher-forced evaluation supplies correct previous targets. Greedy evaluation
starts with BOS and supplies only previously generated predictions. Both use
the same final model in evaluation mode. Generation encodes the source once;
it recomputes decoder prefixes without a KV cache. It emits exactly six slots
for the default task and does not stop at EOS. Early EOS, BOS or missing EOS are
ordinary prediction errors against the fixed target; EOS is not stripped from
the reported metrics.

The causal tests change future decoder inputs and verify unchanged earlier
states/logits with dropout disabled. An unmasked control must respond to those
changes. Separate source perturbations verify a working source-to-decoder path.
These checks detect connectivity errors; they do not prove a learned linguistic
alignment or provide a causal explanation of trained predictions.

## Experimental protocol and assumptions

All data are synthetic integer sequences. IDs 0/1 denote BOS/EOS; IDs 2–31 are
content, including possible repeated tokens within a sequence. Unique sequence
IDs are sampled without replacement from the $30^5$ possible sequences and
decoded in base 30. A single seeded sample is divided into 512 training and
128 test sequences, guaranteeing no exact sequence overlap. Individual token
IDs and subsequences can overlap. Targets are deterministic reversals.

The default model has two encoder and two decoder layers, width 32, four heads,
FFN width 128, ReLU, post-norm and dropout 0.1 inside the backbone. It shares one
token table and one learned position table across source/target; the output head
is separate. Matrix parameters are independently Xavier initialized. There is
no additional embedding dropout, label smoothing, learning-rate schedule or
claim of reproducing the original translation recipe.

Training uses 400 Adam updates at learning rate 0.001, batch size 32 sampled with
replacement from the training set and global gradient clipping at norm 1.0.
Data/model seed is 42; mini-batch generator seed is 43. PyTorch uses CPU, one
thread and deterministic algorithms. The fixed final model is evaluated once
on the held-out set. No validation-based selection, test-based tuning or best
checkpoint selection is performed. Initial/final full-training loss is measured
in evaluation mode, so dropout does not confound that comparison.

## Execution record — 2026-11-25

New analytical execution by Codex; interpretation **author review required**.
This is distinct from the source study's unexecuted proposals.

**Hypothesis:** a small correctly masked encoder-decoder can learn some
fixed-length reversal structure under this bounded training budget, and
teacher-forced token quality may differ from free-running quality. No target
accuracy or convergence guarantee was assumed.

**Command from repository root:**

```powershell
.\.local\environments\n09-full-2026-11-13\Scripts\python.exe 05-transformers-llms/67-transformer-architecture-i/example.py --steps 400 --seed 42
```

**Configuration:** defaults documented above; Windows 10 build 19045,
Python 3.11.14, NumPy 2.4.6, PyTorch 2.15.0+cpu, CPU FP32, one PyTorch thread.
The JSON timestamp was `2026-11-25T10:33:45.874988+00:00`.

| Measured quantity | Result |
|---|---:|
| Initial full-training evaluation cross-entropy | 4.026252270 |
| Final full-training evaluation cross-entropy | 0.048757479 |
| Test teacher-forced cross-entropy | 0.108248033 |
| Test teacher-forced token accuracy (EOS included) | 98.3073% (755/768) |
| Test teacher-forced sequence exact match | 89.8438% (115/128) |
| Test autoregressive token accuracy (EOS included) | 97.1354% (746/768) |
| Test autoregressive sequence exact match | 89.8438% (115/128) |

The first test source was `[27, 16, 9, 5, 24]`; both target and generated output
were `[24, 5, 9, 16, 27, 1]`. This selected example is not representative evidence
by itself; aggregate metrics above include all 128 sequences.

**Interpretation candidate:** the loss reduction and disjoint-sequence exact
matches are consistent with learning this bounded synthetic mapping. Free-running
generation made nine more token errors than teacher forcing while exact match
was equal. A matching sequence percentage does not imply equal token behavior.
The mechanism responsible for individual errors was not investigated.

**Limitations:** one seed, one model and one length; no uncertainty interval,
multi-seed comparison, natural-language data, padding, out-of-length testing,
causal-mask training ablation or performance benchmark. The test result must not
be used to choose a new configuration and still be called an untouched test.
No translation, model ranking or production conclusion follows from these results.

**Forward-pass check:** `from_scratch.py` ran with seed 42 for parameters and
seed 7 for synthetic inputs. It returned memory `(5,8)`, decoder states `(3,8)`,
and attention shapes `(5,5)`, `(3,3)`, `(3,5)`. Maximum row-sum error was
$1.11\times10^{-16}$ and earlier-state change after a future-input perturbation
was zero. These are numerical structural observations with untrained weights.

The small public record is this table. Full JSON records remain ignored under
`outputs/`, with unique timestamps; no generated preview or checkpoint is selected.

### Local verification scope

On 2026-11-25, the topic's isolated test file passed **31 tests**, with no skips
or warnings, using pytest 9.2.0 and the environment above. The repeatability
test runs a tiny two-update configuration twice; it does not repeat the
400-update experiment. No convergence threshold is imposed by the tests.

```powershell
.\.local\environments\n09-full-2026-11-13\Scripts\python.exe 05-transformers-llms/67-transformer-architecture-i/from_scratch.py
.\.local\environments\n09-full-2026-11-13\Scripts\python.exe scripts/validate_repo.py tests --scope 05-transformers-llms/67-transformer-architecture-i
.\.local\environments\n09-full-2026-11-13\Scripts\python.exe scripts/validate_repo.py syntax
.\.local\environments\n09-full-2026-11-13\Scripts\python.exe scripts/validate_repo.py links
uv --no-cache pip check --python .local/environments/n09-full-2026-11-13/Scripts/python.exe
```

Syntax parsing and internal-link checks passed. Dependency compatibility passed
for 81 installed packages; no packages were installed or upgraded. This is local
focused validation, not the full curriculum test suite, Linux CI, GitHub math
rendering, a browser check or personal author review.

## Cost, application and common mistakes

Dense encoder self-attention has pairwise work $O(n^2d)$; decoder self-attention
has $O(m^2d)$; cross-attention has $O(mnd)$. These exclude projection work
$O((n+m)d^2)$ and FFN work $O((n+m)dd_{\text{ff}})$ per layer up to constant
factors. An explicitly materialized self-attention score tensor costs
$O(Bhn^2)$ elements. Fused kernels can avoid storing the full tensor; these
expressions are not measured allocation or latency results. Prefix recomputation
in this educational decoder adds redundant work at inference.

Encoder-only, decoder-only and encoder-decoder architectures differ in
visibility and conditional paths. Architecture does not by itself specify
pretraining objective, dataset quality or downstream performance. In a typical
decoder-only RAG prompt, retrieved passages enter causal self-attention as input
tokens; vector retrieval is not a Transformer cross-attention operation.
Attention weights are internal mixtures, not validated feature importance.

For variable-length batches, future work must address source/target padding
masks, source memory padding masks, loss exclusion and EOS-aware metrics
together. Silently adding padding to this baseline would violate its assumptions.

## Proposals at the analytical handoff

This list records the analytical handoff. The subsequent
[separate visual protocol](VISUAL_LAB.md#paired-synthetic-experiment--2026-11-25)
executes the matched masked/unmasked training comparison with seed 67 and its
own checkpoints and metrics. The seed-42 analytical record above is preserved;
the other suggested comparisons below remain unexecuted.

- Train matched masked/unmasked models and compare teacher-forced versus
  autoregressive quality. Existing unmasked tests are connectivity controls,
  not that training experiment.
- Remove learned positions with a fixed budget and independent validation
  protocol; distinguish encoder symmetry from decoder mask structure.
- Compare head counts or pre-norm/post-norm with parameter/budget differences
  recorded explicitly and multiple seeds.
- Evaluate new sequence lengths with an appropriate position strategy and
  separately defined test sets. This learned table rejects unsupported lengths.

These are suggestions, not measured findings or author conclusions.
