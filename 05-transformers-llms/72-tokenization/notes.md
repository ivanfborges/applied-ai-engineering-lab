# Tokenization: learned units and operational budgets

## The tokenizer is part of the model interface

A tokenizer comprises more than a vocabulary: normalization, pretokenization,
segmentation, token-to-ID mapping, special-token policy and decoding all matter.
IDs select rows in an embedding table and columns in output logits. Reusing a
different tokenizer with a pretrained network changes that correspondence.
Adding tokens in an open-weight workflow can require resized embeddings and
training; a hosted model's tokenizer cannot be replaced through prompt design.

Character tokenizers need character coverage and often longer sequences. Whole
word tokenizers need many entries and an unknown-word policy. Subwords reuse
fragments, but these are statistically learned pieces, not guaranteed morphemes.
All 256 base bytes cover valid UTF-8 text without an unknown ordinary-text byte;
the efficiency of representing that text still depends on the learned merges.

Normalization may intentionally lose distinctions. Unicode normalization,
lowercasing and whitespace collapsing are configuration choices. A normalized
round trip can recover normalized text without recovering the original source.
Code, identifiers, citations and OCR evidence can be damaged by aggressive cleanup.

## Three different segmentation mechanisms

| Aspect | BPE | WordPiece | Unigram |
|---|---|---|---|
| Training intuition | Add frequent adjacent pairs | Learn a useful subword vocabulary; relative-association scoring is an educational reconstruction | Estimate piece probabilities and prune a candidate vocabulary |
| Common inference | Apply learned merge priorities | Greedy longest match with continuation markers | Maximize segmentation log probability with dynamic programming |
| Ambiguity | Ranked merges resolve competing pairs | Greedy choices generally do not backtrack | Several segmentations have scores; sampling can be enabled |
| Coverage | Depends on base alphabet/bytes | Incomplete word segmentation can yield a whole-word unknown | Depends on character coverage and fallback configuration |

SentencePiece supplies training and inference infrastructure, including BPE and
Unigram. It can learn from raw sentences without an external word segmenter.
Its whitespace marker `▁` and default normalization must be understood together:
the marker alone does not guarantee original-text reconstruction. Optional byte
fallback and subword sampling are configuration features, not properties of
every SentencePiece model. See [the implementation and paper](references.md).

### BPE: frequency during training, ranks during inference

For a current symbol sequence $w$ with corpus frequency $f(w)$:

$$
C(a,b)=\sum_{w\in\mathcal D} f(w)
\sum_{i=1}^{|w|-1}\mathbf{1}[(w_i,w_{i+1})=(a,b)].
$$

Choose the pair with largest $C(a,b)$ and update the corpus. Here ties use
lexicographic ordering, making results independent of dictionary insertion
order. Count adjacent pairs including overlaps, then replace non-overlapping
occurrences from left to right: `aaaaa` merged on `(a,a)` becomes `aa, aa, a`.
Thus a counted frequency is not always the number of replacements.

Inference starts from the base symbols and repeatedly applies the best-ranked
available learned pair. It does not learn from the incoming request. If `(b,c)`
was learned before `(a,b)`, `abc` can become `a, bc` even if `ab` is a vocabulary
entry. Longest-vocabulary-match inference would instead start with `ab`.

`SimpleBPE` trains on whitespace-split words without end-of-word markers. It
allows a learned fragment at any position; changing that boundary policy would
change the tokenizer. It drops interword whitespace during training, does not
normalize, and rejects unseen code points. Its repeated scans cost roughly
$O(MN)$ for $M$ merges and initial corpus symbol count $N$, excluding Python
string copying and sorting/tie-selection details. This is an educational bound,
not a throughput measurement. Refit resets the learned alphabet and rules.

### WordPiece: avoid claiming the historical training recipe

A commonly taught score is

$$
S(a,b)=\frac{C(a,b)}{C(a)C(b)}.
$$

This rewards relative association rather than frequency alone, but is not a
verified reproduction of Google's original unpublished trainer. At inference,
the BERT implementation uses a longest-match-first vocabulary lookup with `##`
continuation pieces and a whole-word `[UNK]` on failure. Greedy search need not
find another valid segmentation after making an unsuccessful longest choice.
The original BERT code, rather than a reconstructed training equation, defines
that inference behavior; see [references](references.md).

### Unigram: probabilities over alternative paths

For a segmentation $s=(t_1,\ldots,t_m)$ of string $x$:

$$
P(s)=\prod_i p(t_i),\qquad
s^*=\arg\max_{s\in\mathcal S(x)}\sum_i\log p(t_i).
$$

The independence assumption makes dynamic programming possible. During training,
the text likelihood sums over possible segmentations rather than treating a
single greedy path as ground truth. Unigram pruning and probability estimation
differ from frequency-based BPE merges. Sampling segmentations during training
is subword regularization; deterministic best-path inference is a separate mode.
No Unigram trainer, probability estimates or sampling experiment is included here.

## Vocabulary, sequence length and memory

For vocabulary size $V$ and hidden width $d$, input embeddings contain $Vd$
weights. A bias-free output projection contributes another $Vd$ if untied.
Weight tying shares storage but does not remove vocabulary-sized logit work.
At $V=50{,}000$, $d=4{,}096$, a tied table has 204,800,000 parameters and
409,600,000 FP16 weight bytes (0.4096 decimal GB, approximately 0.3815 GiB).
This arithmetic excludes gradients, optimizer state, activations and all other
model layers. Rare pieces also need enough training exposure to learn useful
representations; a bigger vocabulary is not automatically better.

Dense full-sequence attention has $n^2$ score positions per head and core
arithmetic of order $O(n^2d_k)$ when key/value widths are comparable. Doubling
token count quadruples score positions, not necessarily measured runtime or
allocated memory. Fused kernels need not materialize the full score matrix.
Cached autoregressive decoding attends new queries to cached keys/values, with
different cost from a full prefill. KV storage grows with cached token count
for fixed layers, head layout, precision and batch size. These are scaling
arguments, not measured hardware benchmarks.

`tiktoken.n_vocab` is an ID capacity (maximum token ID plus one). It can include
unused IDs; do not equate it with the number of mergeable entries. The example
reports both mergeable entries and defined special tokens using public APIs.

## Context and monetary units

For a hypothetical decoder-only model with a shared window:

$$
T_{\mathrm{system}}+T_{\mathrm{history}}+T_{\mathrm{query}}+
T_{\mathrm{retrieval}}+T_{\mathrm{overhead}}+T_{\mathrm{reserved\ output}}\le L.
$$

Also enforce independent input and output caps. `context_budget` operates on
already-counted integers; it is not a request serializer or tokenizer. Tokenizing
components separately and adding their counts need not equal tokenizing their
concatenation: merges can cross the boundary. Count the final serialized request
with the target model's supported method, including chat formatting and tools.
The hypothetical 32,768-token example allocates 20,300 input tokens plus 4,000
reserved output tokens, leaving 8,468. A reservation is not observed output.

With invented prices $P_{\mathrm{in}}$ and $P_{\mathrm{out}}$ per million tokens:

$$
C=\frac{T_{\mathrm{in}}P_{\mathrm{in}}+T_{\mathrm{out}}P_{\mathrm{out}}}{10^6}.
$$

For 6,000 input tokens at USD 2/million and 1,000 output tokens at USD 8/million,
the calculation is USD 0.02 per request. Decimal arithmetic avoids binary floating-point
rounding for these decimal price inputs. No current provider price is asserted.
Caching, multimodal inputs, tool charges, infrastructure and provider-specific
reasoning/usage rules are excluded. Reserve budget before dispatch; reconcile
estimates with the provider's actual usage after the response.

## Reading the real-encoding probes

The eight handwritten samples exercise prose, Portuguese, terminology, Python,
JSON, an artificial OCR corruption and mixed Unicode. They are synthetic probes,
not representative language distributions. English and Portuguese differ in
length and wording, so their token counts cannot rank language equity. Unicode
character counts here are Python code points, not grapheme clusters. Word-based
fertility additionally needs a defined word segmenter, especially for scripts
without whitespace-separated words.

Compare OCR noise only to its paired clean string; the edit changes characters
and whitespace simultaneously. Increased token count on this pair cannot prove
worse OCR retrieval or answer quality. Smaller counts for an encoding do not
isolate vocabulary size as the cause: vocabularies and pretokenization differ.

The full token sequence reconstructs UTF-8 bytes. A token boundary can occur
inside an emoji, so individual byte pieces should not be independently decoded
with replacement characters. `inspect_text` rejects unpaired surrogates rather
than silently accepting a library repair. `encode_ordinary` deliberately treats
special-token-looking text literally; a chat renderer has a different contract.

## RAG and evaluation decisions

Validate ingestion chunks against the embedding tokenizer and limits. After
retrieval, validate the complete generator prompt against its own tokenizer.
Use document boundaries where possible and measure any truncation policy against
retrieval relevance, answer quality and citation preservation. Token counts alone
cannot choose a chunk size. OCR cleanup should retain source evidence and avoid
changing identifiers. Train custom tokenizer vocabularies on training data only
when evaluating generalization; test-distribution vocabulary exposure is a form
of leakage even without labels.

## Execution record — 2026-11-25

**Run:** 2026-11-25 at 21:52:06 UTC (18:52:06 America/Sao_Paulo), CPU,
Python 3.11.14 and `tiktoken` 0.14.0. The deterministic character implementation
uses no random sampling; neither does pretrained encoding. No seed is needed.
The two cached encoding table hashes match [the resource record](references.md).
The JSON-serialized, sorted synthetic probe mapping has SHA-256
`6124cc1cd121ada961a306ac831052158c908606eda6f929c1ea165be1ba5298`.

**Hypotheses:** learned pair ranks can encode novel combinations of covered
characters; two pretrained encodings can give different lengths for identical
strings; the controlled OCR edit can increase fragmentation on its paired probe.

**Configuration:** `SimpleBPE(10)`, weighted words `low:5`, `lower:2`, `newest:6`,
`widest:3`, lexical ties and no end-of-word markers. Separate fixed encodings
`cl100k_base` and `o200k_base`, ordinary-text policy, eight exact synthetic
strings in `example.py`, no normalization or tokenizer training on those probes.

**Commands executed:** `from_scratch.py`, `example.py --output` with a unique
ignored JSON destination, and the topic-scoped validation runner, using the
project's stable Python interpreter. Reproduce from the root:

```powershell
python 05-transformers-llms/72-tokenization/from_scratch.py
python 05-transformers-llms/72-tokenization/example.py --output 05-transformers-llms/72-tokenization/outputs/run-20261125-analytical.json
python scripts/validate_repo.py tests --scope 05-transformers-llms/72-tokenization
```

Use a new filename for reruns; the original JSON remains local and ignored.

**Observed results:** the first BPE merge was `(e,s)` with weighted count 9.
Ten merges grew the alphabet of 10 symbols to 20 vocabulary entries.
`lowest` became `['low', 'est']` and `newer` became `['n', 'ew', 'e', 'r']`;
all six inspected words reconstructed exactly. The 33 focused tests passed
with no skips or warnings reported, including an independent case where
rank-based merging differs from longest-vocabulary matching.

| Synthetic probe | `cl100k_base` tokens | `o200k_base` tokens |
|---|---:|---:|
| English | 6 | 6 |
| Portuguese | 14 | 10 |
| Technical | 11 | 10 |
| Python | 10 | 10 |
| JSON | 14 | 14 |
| Clean | 2 | 2 |
| OCR noise | 11 | 11 |
| Unicode | 9 | 7 |

All eight complete byte streams and decoded strings reconstructed exactly
under both encodings. `cl100k_base` reported ID capacity 100,277 with 100,256
mergeable entries and 5 defined special tokens. `o200k_base` reported ID
capacity 200,019 with 199,998 mergeable entries and 2 defined special tokens.
The gaps are why ID capacity is labeled separately from defined token counts.

**Interpretation candidates — author review required:** these probes illustrate
that count depends on the encoding, and that this artificial OCR edit consumes
9 more tokens (450% inflation relative to 2) in either encoding. They support
counting representative application text with the target interface, rather than
using a fixed characters-per-token rule. The character BPE result illustrates
composition from known symbols; it does not establish useful embeddings.

**Limitations:** one fixed configuration and eight short strings; the language
samples are not a representative parallel corpus. The two encodings differ
in more than vocabulary size. OCR edits change multiple properties at once.
No trained language model, retrieval metric, provider usage, latency, cost bill,
WordPiece training or SentencePiece execution was evaluated. The context
remainder 8,468, cost USD 0.02 and 409,600,000 weight bytes are deterministic
hypothetical calculations, not empirical systems measurements. The raw JSON,
cache and full token traces stay ignored; this table is the intentional public
textual evidence. Automated checks do not approve interpretations for the author.

## Suggested work — not executed

This list records the analytical phase's unexecuted suggestions. The later
[visual phase](VISUAL_LAB.md) implements related vocabulary, translated-probe,
OCR and chunking protocols with separately recorded configurations. They do
not replace this historical list or imply that the proposed broader evaluations
and exact settings below were run.

- Train `SimpleBPE` with 0, 2, 5, 10 and 15 merges and compare on independent
  words whose characters are covered. Separate training compression from held-out
  fragmentation and report the unsupported-character rate.
- Compare parallel translated passages with documented language/segmentation
  rules, using both bytes/token and a carefully defined fertility measure.
- Vary OCR corruption mechanisms independently across many paired documents and
  evaluate retrieval relevance alongside token counts.
- Evaluate semantic-boundary chunking with both embedding and generation
  tokenizers and a held-out question set. No RAG pipeline is executed here.

These suggestions are not results or author conclusions.
