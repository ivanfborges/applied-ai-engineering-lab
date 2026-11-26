# Pretraining: objective, data and compute

## Likelihood is a prediction objective

For a token sequence $X=(x_1,\ldots,x_T)$, the probability chain rule gives

$$
p_\theta(X)=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

This identity assumes no token independence. A bigram model introduces a
restriction, $p(x_t\mid x_{<t})\approx p(x_t\mid x_{t-1})$, while a causal
Transformer uses a longer prefix within its context window. Here BOS supplies
the initial context, EOS is a prediction target, and BOS itself is not scored.

With logits $z$ and a one-hot target $y$, use stable log-softmax:

$$
\log p_j=z_j-m-\log\sum_k\exp(z_k-m),\qquad m=\max_k z_k.
$$

Taking a logarithm after softmax can underflow for a very unlikely target.
`from_scratch.py` computes log probabilities directly. For a mean over $K$
prediction positions,

$$
\mathcal L=-\frac1K\sum_{i=1}^K\log p_{i,y_i},\qquad
\frac{\partial\mathcal L}{\partial z_{i,j}}=\frac{p_{i,j}-y_{i,j}}K.
$$

The bigram table reuses a row for every occurrence of the same context token.
Its weight gradients must therefore **sum repeated row contributions**.
`np.add.at` implements that accumulation; indexed assignment would silently
lose repeated contributions. Tests check both autograd and finite differences.
Three short synthetic sentences supply twelve transitions; EOS never predicts
the BOS of another sentence. No generalization result is claimed for this table.

## Visibility and selected loss positions

CLM input `X[:, :-1]` predicts `X[:, 1:]`. Query position $t$ may attend to
positions $\leq t$, including itself, because its target is the *next* token.
Without the attention mask, a later input position can reveal that target.
Correct shifting and correct visibility are separate requirements.

MLM selects interior positions $M$ and predicts their original IDs from a
corrupted sequence $\widetilde X$:

$$
\mathcal L_{\mathrm{MLM}}=-\frac1{|M|}\sum_{i\in M}
\log p_\theta(x_i\mid\widetilde X).
$$

Every selected token becomes `<mask>` here. There is no 80/10/10 replacement
mixture, next-sentence objective, pretrained tokenizer or BERT reproduction.
The selected fraction is 25% of the four eligible positions by default.
Unselected targets use `ignore_index=-100`; their logits contribute zero loss
and zero direct loss gradient. They may still influence selected positions
through attention, so their hidden states and embeddings can receive gradients.

Evaluation enumerates every subset of the requested size from the four
interior positions. With one mask, this produces 48 train and 16 validation
reconstructions. Uniformly sampled training masks and exhaustive evaluation
masks address the same reconstruction objective; final train NLL is not the
last stochastic minibatch loss. All subsets have equal size, so the simple
token mean gives equal weight to each sentence/subset. Empty selections fail.

The module mask convention is Boolean `True` = **blocked**. This is the
convention of PyTorch Transformer modules, not functional SDPA. A test perturbs
a future token, checks invariance of earlier causal logits, and verifies a
nonzero change with bidirectional attention as a positive control.
Evaluation uses `eval()` and `inference_mode()`, restores the prior mode and
does not change weights or existing gradients. Dropout is zero in both phases.

## What can the synthetic corpus establish?

The construction combines ordered animals `(cat, dog, bird, rabbit)` with
foods `(fish, meat, seeds, carrots)` in
`<bos> the ANIMAL likes FOOD <eos>`. Indices $i\ne j$ enter training; $i=j$
enter validation. Vocabulary IDs come from training content plus a reserved mask.
All validation words occur in training; whole sentences do not overlap.
The sentences describe an invented grammar, not animal diets or verified facts.

Even perfect fitting cannot predict every ambiguous position deterministically.
In the empirical training distribution, CLM has four equiprobable animals after
`the` and three equiprobable foods for each animal. Three other target positions
are deterministic. Thus the analytical minimum mean training NLL is

$$
\mathcal L_{\mathrm{CLM,min}}=\frac{\log4+\log3}{5}\approx0.496981,
$$

and the corresponding expected token accuracy ceiling is
$(3+1/4+1/3)/5\approx0.716667$. This is an empirical-distribution reference,
not a claim about optimizer convergence or natural language.

For one-mask MLM, `the` and `likes` are deterministic; animal given food and
food given animal each have three alternatives in training. The analogous
minimum selected-token training NLL is $\log3/2\approx0.549306$, with expected
accuracy $(2+1/3+1/3)/4=2/3$. Changing the number of masks changes this problem.

The reserved diagonal combinations have zero empirical training probability.
A model fitting that distribution well can assign them little probability.
Higher held-out NLL is compatible with that support shift; it does not by itself
identify memorization, establish general language understanding or prove an
architectural defect. Aggregate token accuracy also includes easy fixed words.
Use position-specific metrics, larger grammars and independent test data in a
future experiment before making stronger generalization claims.

## Perplexity and data hygiene

With natural-log CLM NLL, $\mathrm{PPL}=\exp(\mathcal L)$. It measures predictive
uncertainty on a specified distribution and scoring policy. It does not measure
truth, useful reasoning, attribution, safety or instruction following.
Compare perplexities only with aligned tokenizer, evaluated text, context,
special-token treatment and loss denominator. A different token unit can change
the number without improving modeling. Byte-normalized likelihood can help
compare tokenizations when text and evaluation protocols are aligned.
Exponentiating MLM loss does not yield ordinary autoregressive perplexity.

For real corpora, record provenance and permitted use, remove unwanted sensitive
content, audit quality/language/domain mixtures, and deduplicate before deciding
evaluation boundaries. Split source documents or related groups before making
overlapping chunks. Audit exact and approximate benchmark overlaps. Packing
requires deliberate document boundaries, attention visibility and loss masks;
padding must be excluded from both attention where needed and loss. The present
fixed-length example implements none of that pipeline and demonstrates only
its explicit combination split. RAG does not remove the need to evaluate answers.

## Scale is an allocation problem

The [Kaplan study](references.md) reports empirical power-law behavior in its
experimental regime. The [Hoffmann study](references.md) studies how to allocate
a fixed training compute budget between parameters and training tokens. An
illustrative fitted loss form is

$$
L(N,D)=E+A N^{-\alpha}+B D^{-\beta},\qquad C\approx6ND.
$$

Here $N$ follows the fit's parameter-count convention, $D$ follows its data and
training-token convention, and $E,A,B,\alpha,\beta$ are fitted quantities. No
constants or exponents are fitted in this study. Constraining $D=C/(6N)$ gives
$N^*\propto C^{\beta/(\alpha+\beta)}$ and
$D^*\propto C^{\alpha/(\alpha+\beta)}$. Similar exponents suggest growing both
model size and token budget. They do not establish a universal token/parameter
ratio. A training-compute optimum also need not minimize lifetime serving cost.

The dense approximation counts roughly $2ND$ forward operations and $4ND$
backward operations. It omits explicit attention/context costs, embedding work,
optimizer operations, activation recomputation, communication, data loading and
downtime. MoE requires active/total parameter distinctions. Converting FLOPs to
time requires *measured sustained* throughput and a representative pilot,
not just accelerator peak specifications.

The script reports both total and nonembedding parameters. Its labeled `6ND`
proxy uses the latter and **processed input tokens**, including repeated
exposures. This is illustrative accounting; the approximation is especially
crude for such a tiny model. It is not a profiler result or a scaling experiment.
Loss-position count is different: MLM computes all six input representations
but scores only the selected token. Twelve thousand repeated token exposures
do not provide the diversity of twelve thousand unique corpus positions.

At scale, memory also includes activations, gradients, optimizer states and
temporary buffers. Gradient accumulation, activation checkpointing and sharding
address different parts of that budget. None is implemented or benchmarked here.

## Application decisions

Continued pretraining changes existing weights with a language-modeling
objective and can adapt domain distributions. Supervised fine-tuning teaches
desired input/output behavior; retrieval supplies external evidence at inference.
These methods can be combined, but each needs task-specific evaluation.
Frequently changing or permission-scoped facts need traceable data access;
weight updates alone cannot enforce current document permissions or citations.
Training from scratch additionally requires data, compute, recovery and evaluation
capabilities far beyond this educational CPU loop.

## Execution record — 2026-11-25

New analytical runs, separate from the unexecuted source suggestions. All
interpretations below are candidates: **author review required**.

**Hypothesis:** the miniature self-supervised models can fit training templates,
while predictions for absent animal-food combinations may remain weaker.
The NumPy hypothesis is that the explicit mean-NLL gradient fits local transitions.

**Environment/configuration:** Windows, Python 3.11.14, PyTorch 2.15.0+cpu,
NumPy 2.4.6; one CPU thread for the Transformer. Each Transformer used seed 42,
float32, 200 updates, a full batch of 12, width 32, two heads, one block, FFN 64,
learned positions, untied output projection, ReLU, post-norm, no dropout, AdamW
learning rate 0.01 and weight decay 0.01. Both used 9,581 total / 8,973
nonembedding parameters. Final-update checkpoints were evaluated; validation
was not used for early stopping or checkpoint selection. No independent test set.

| Measurement | CLM | MLM (one interior mask) |
|---|---:|---:|
| Initial train NLL | 2.740437 | 2.350907 |
| Initial validation NLL | 2.740490 | 2.356024 |
| Final train NLL | 0.498278 | 0.622623 |
| Final validation NLL | 1.573562 | 1.371625 |
| Final train token accuracy | 0.716667 | 0.666667 |
| Final validation token accuracy | 0.650000 | 0.500000 |
| Final validation autoregressive perplexity | 4.823801 | Not applicable |
| Evaluated train / validation targets | 60 / 20 | 48 / 16 |
| Processed training input tokens | 12,000 | 14,400 |
| Supervised training tokens | 12,000 | 2,400 |
| Training loop seconds, excluding initialization/evaluation | 0.798496 | 0.738203 |
| Theoretical nonembedding `6ND` FLOPs proxy | 646,056,000 | 775,267,200 |

The timing is one wall-clock observation per loop, not a throughput comparison.
Execution timestamps were `2026-11-25T22:39:27.158948+00:00` (CLM) and
`2026-11-25T22:39:40.520316+00:00` (MLM). Full reports remain ignored at
`outputs/clm-20261125-analytical-01.json` and
`outputs/mlm-20261125-analytical-01.json`; this table is the intentional public
record. Future runs must use new paths and append distinct records.

NumPy executed at `2026-11-25T22:39:43.162016+00:00`: seed 42, float64,
64 table parameters, 12 within-sentence pairs from the three strings listed in
`from_scratch.py`, 300 full-batch updates and learning rate 1.0. Training NLL
changed from **2.077227 to 0.302376**. After `learns`, probabilities for
`language` and `patterns` were each **0.483532**. This is a separate corpus,
optimizer and context restriction; its NLL does not rank it against the Transformer.

**Interpretation candidates:** CLM nearly reaches the analytical empirical
training entropy reference. Both Transformer runs show a train/validation NLL
gap on the excluded combinations; the gap alone does not isolate its cause.
MLM reaches the expected training accuracy ceiling despite residual NLL above
its optimum, illustrating that argmax accuracy does not capture probability fit.
The bigram table assigns most mass to the two observed successors of `learns`.

**Limitations:** one seed and tiny artificial grammars; incomparable CLM/MLM
losses and unequal supervision; repeated exposures, no scale sweep, no unseen
vocabulary, no real-text or downstream evaluation, no hardware profiling.
No claim about hallucinations, production readiness or empirical scaling laws.

## Suggested experiments — not executed as experiments here

Vary capacity or update budget with multiple seeds, recording unique data,
processed tokens, supervised tokens, parameters, elapsed time and held-out NLL.
Use an independent test set if selecting a configuration on validation.
Increase corpus diversity while preserving a meaningful source-level split.
Compare mask counts with exhaustive matching evaluation policies. As a separate
leakage demonstration, remove the CLM attention mask and evaluate both
teacher-forced predictions and legitimate prefix-only prediction; a lower leaky
loss would not establish usable generation. The automated invariance and
short-training tests are correctness checks, not those proposed studies.
