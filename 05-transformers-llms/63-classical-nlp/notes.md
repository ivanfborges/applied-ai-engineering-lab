# Classical NLP: representation choices and information loss

## What the transformations assume

A lexical analyzer maps raw text into an ordered token sequence. If two strings
map to the same sequence, a downstream model using only that sequence cannot
recover their original distinction. Case folding can merge `US` and `us`;
deleting negation can merge opposing statements. The useful question is whether
those equivalences match the task, rather than whether text looks cleaner.

Unicode normalization addresses representation equivalence. NFC composes
canonically equivalent sequences where possible; it preserves the distinction
between accented letters. NFKC additionally folds compatibility characters,
which can be useful for some inputs but inappropriate for preserving original
notation. Case folding is a separate, potentially lossy operation: `Straße`
becomes `strasse`. Neither operation is semantic interpretation.

Keep the original document for citations, offsets, identifiers and auditing.
A normalized search representation usually cannot supply exact original offsets
without a separate mapping. Apply compatible, versioned analyzers to indexed
documents and queries; retain the same vocabulary and IDF at query time.

## Analyzer contract

The implementation uses Python's Unicode database and regex semantics:

1. Apply NFC, `casefold()`, map curly right apostrophe to ASCII apostrophe,
   collapse whitespace and apply NFC again after case folding.
2. Extract alphanumeric runs, including internal apostrophes. Underscores,
   hyphens, punctuation and emoji separate or disappear from tokens.
3. Retain single-character tokens, digits, accents and all words, including
   `not`, `without` and `never`. There is no stop-word list.
4. Extract windows per document. Punctuation does not define a sentence boundary
   here, so `approved. Never` can contribute the bigram `approved never`.

Empty strings yield no tokens. Invalid text types raise `TypeError`; malformed
orders, tokens and vocabularies fail explicitly. The dense count helper accepts
an empty corpus and returns a `(0, 0)` matrix. Documents shorter than the n-gram
order produce zero features. Scikit-learn's fitted vectorizer instead raises an
error for a completely empty vocabulary; the retrieval wrapper retains that
behavior. A query without known features returns an empty ranking, not an
arbitrary list of zero-score documents. Positive-score ties keep document order.

This contract is not a multilingual tokenizer. Some combining marks remain
separate even after NFC and are not matched by this regex; scripts requiring
language-aware word segmentation need a different analyzer. `don't` is one token,
not a grammatical analysis. `47.838/2020` becomes `47`, `838`, `2020`; use a separate
exact-identifier field if this reference must match intact. Word n-grams contain
tuples internally so boundaries are explicit; joining tokens with spaces for
library comparison is safe under this helper's no-whitespace token contract.

## Stemming and lemmatization

Stemming approximates morphological conflation through rules. A stem need not be
a dictionary word. Understemming leaves desired variants apart; overstemming
merges distinctions that matter. Derivational relationships, inflection and
synonymy are different problems: `operate`, `operator` and `operation` should not
be assumed to have interchangeable meanings merely because a rule groups them.

Lemmatization maps a form to a base form using linguistic resources or models.
POS matters: `saw` can map to `see` as a verb and `saw` as a noun. POS itself is
an inference problem when not supplied. A dictionary alone cannot resolve every
contextual ambiguity, and lemmas do not merge synonyms such as `car` and
`automobile`. English rules and lookup resources do not establish Portuguese
behavior.

The offline examples deliberately separate mechanisms:

- `suffix_stem_demo` removes at most one of `ing`, `ed`, `es`, `s`, retaining at
  least two characters. It exposes errors such as `running → runn` and
  `news → new`. It is not an implementation of Porter or Snowball.
- `lemma_lookup_demo` reads a small hand-authored English `(word, POS)` table.
  Unsupported entries raise an error; no identity fallback pretends that a
  lemma was inferred. Supplied tags and entries are teaching inputs, not measured
  morphological accuracy or automatic tagging output.

These demos explain why suffix rules and lexical knowledge differ. They should
not be used to compare real stemmer/lemmatizer quality. A real evaluation would
select a language, install a maintained analyzer, record its resource/model
version and license, and evaluate annotated examples and downstream errors.
The [NLTK references](references.md) identify real Porter and WordNet interfaces;
they are not runtime dependencies of this study.

## Counting local order

For a document with $m$ tokens and positive integer order $n$, contiguous windows
without padding produce

$$
G(m,n)=\max(0,m-n+1).
$$

This counts occurrences, including repeated windows. The observed vocabulary is
the union of distinct windows over documents. A vocabulary of $K$ tokens admits
at most $K^n$ ordered tuples of length $n$, but a finite corpus only contributes
its observed windows. Vocabulary growth is not a measured memory forecast.

For document $d$ and feature $g_j$, the count matrix is $X_{dj}=c(g_j,d)$.
`dog bites man` and `man bites dog` have identical unigram counts but distinct
bigrams. Local windows still discard absolute positions, most global order and
long-distance scope. N-gram feature extraction is also distinct from an n-gram
language model, which estimates conditional next-token probabilities and needs
boundary, smoothing and unseen-event policies. No language model is trained here.

The educational count helper allocates $O(DK)$ integers for $D$ documents and
$K$ observed features, plus window lists. It is intentionally dense and small.
The library example retains sparse document and query matrices. Only the final
length-$D$ similarity vector is dense. Large-scale retrieval needs an appropriate
index and top-k strategy; this example scores every indexed document.

## Weighting, fitting and evaluation boundaries

The library uses raw term counts, smoothed IDF and L2-normalized TF-IDF vectors.
For an index of $D$ documents with document frequency $\operatorname{df}(t)$:

$$
\operatorname{idf}(t)=\log\frac{1+D}{1+\operatorname{df}(t)}+1,
\qquad z_{dt}=c(t,d)\operatorname{idf}(t),
\qquad x_d=\frac{z_d}{\lVert z_d\rVert_2}.
$$

Zero rows remain zero. Cosine similarity is a lexical overlap score here, not a
probability of relevance. Comparing a score between different feature spaces
does not establish that one model became more confident or more accurate.

For this retrieval demonstration the six documents are the fixed index, not a
training/test split. Fitting the index is expected; adding a query to the fitted
vocabulary would change the experiment. In supervised classification, fit the
vectorizer and any corpus-dependent preprocessing only on training data within
each fold, preferably in a pipeline. The same distinction applies when tuning
an analyzer: use development judgments and reserve held-out queries.

Bigrams can add a feature for `not approved`; unigrams already retain `not` when
it is not deleted. Neither representation guarantees understanding of phrases
such as `not only approved` or long-distance negation. If preprocessing makes
opposing inputs identical, no later classifier can distinguish that pair from
those features alone.

Word n-grams offer interpretable phrases. Character n-grams can retain overlap
across spelling or OCR variations, while also creating unwanted partial matches.
Neither is universally more accurate. For hybrid retrieval, preserve exact
identifier matching, evaluate lexical and dense retrieval separately and then
their combination, and measure relevance with judgments. No dense embeddings,
BM25 or hybrid system is implemented here. Do not feed stemmed text blindly to
a pretrained model with a different native tokenizer and training distribution.

## Executed study — 2026-11-24

The AI assistant ran both scripts locally on Windows with Python 3.11.14,
NumPy 2.4.6, scikit-learn 1.9.1 and Unicode database 14.0.0. This is a new
analytical run, not a rerun of a historical experiment. There is no stochastic
sampling, model training or seed; all input text is defined in code.

From the repository root, the executed commands used the local stable interpreter:

```powershell
& .local/environments/n09-full-2026-11-13/Scripts/python.exe 05-transformers-llms/63-classical-nlp/from_scratch.py
& .local/environments/n09-full-2026-11-13/Scripts/python.exe 05-transformers-llms/63-classical-nlp/example.py
```

The library script's stdout was saved as an ignored local JSON record; no
generated asset is selected for publication. Normalization produced
`informação strasse` from decomposed/capitalized input, retaining the Portuguese
accents while folding `ß`. The morphology script printed the supplied lookup
entries and the mechanical suffix outputs, including `running → runn`,
`was → wa` and `news → new`. Those outputs describe this limited rule only.

**Experiment A — feature growth and lexical ranking**

- Hypothesis: adding observed word windows expands the feature space and changes
  overlap scores; it need not change the top-ranked document or improve relevance.
- Configuration: six synthetic documents in `example.py`, no stop-word removal,
  no stemming or lemmatization, raw TF, smoothed IDF, L2 normalization and cosine
  similarity. Fit on those documents only. Query: `environmental license not approved`.
  Compare ranges `(1, 1)`, `(1, 2)`, `(1, 3)` without feature pruning. Document IDs
  are zero-based; positive-score ties retain input order.
- Result:

| Range | Matrix shape | Nonzeros | Density | Query nonzeros | Top document ID | Top cosine |
|---|---|---:|---:|---:|---:|---:|
| 1–1 | 6 × 18 | 31 | 0.287037 | 4 | 1 | 0.885595 |
| 1–2 | 6 × 39 | 56 | 0.239316 | 6 | 1 | 0.775626 |
| 1–3 | 6 × 56 | 75 | 0.223214 | 6 | 1 | 0.641792 |

The positive-score order was `1, 0, 5, 3` in all three configurations. Documents
2 and 4 had zero overlap and were excluded from the displayed ranking.
Document 1 is `The environmental license was not approved.` Increasing the
range added features but did not change that order. Not every query window
exists in the index: `license not`, for example, is absent because the document
contains `license was not`.

- Interpretation candidate — **author review required**: local windows expose
  additional phrase features, but this example already ranks document 1 first
  with unigrams. Its decreasing cosine across feature spaces is not evidence
  of declining or improving relevance; each space has a different normalization.
- Limitation: one constructed query and six English sentences, with no relevance
  judgments, held-out queries, uncertainty estimate, retrieval-quality metric,
  timing measurement or production index. No method superiority follows.

**Experiment B — destructive negation and zero-evidence controls**

- Hypothesis: dropping `not` makes the token sequences of `The license was approved.`
  and `The license was not approved.` identical. A query consisting of one unseen
  token has no lexical evidence under the fixed index.
- Configuration: the same analyzer; remove only the literal token `not` from
  the negative sentence for the collision control. Independently query
  `unseentokenxyz` with word range `(1, 2)` on the unchanged six-document index.
- Result: tokens were distinct before removal and equal afterward. The unseen
  query had zero nonzeros and returned an empty ranking; the fitted vocabulary
  remained the corpus vocabulary.
- Interpretation candidate — **author review required**: a deterministic
  preprocessing collision makes these two statements indistinguishable to a
  downstream model using only these tokens. A zero-score ordering should not be
  presented as evidence of a match.
- Limitation: exact constructed controls, not an estimate of how often a deployed
  stop-word policy fails. No classifier, production fallback or negation parser
  was evaluated.

Automated assertions separately passed 52 focused cases with pytest 9.2.0,
including count equality with scikit-learn under the same analyzer. That parity
checks the counting mechanism; both implementations deliberately share the
tokenizer and do not independently validate its linguistic adequacy. These
checks do not approve interpretation candidates or establish relevance quality.

## Proposals from the analytical phase

These were unexecuted proposals at the time of the analytical run above. The
later [visual lab](VISUAL_LAB.md) executes separate morphology, judged-retrieval
and synthetic OCR demonstrations with its own corpora, configurations and dated
record. Those runs do not replace the original experiment or establish the
larger-corpus, real-language and production evaluations proposed here.

- Compare a maintained English stemmer and lemmatizer on annotated POS examples,
  then on development queries. Record resource versions and erroneous conflations.
- Evaluate word versus `char_wb` character n-grams under controlled OCR corruption,
  with relevance judgments defined before ranking comparisons. `char_wb` pads word
  edges and differs from raw character windows spanning spaces.
- Evaluate case/accent policies on exact identifiers, entity names and Portuguese
  minimal pairs. Retain the original fields and stratify errors by language.
- Tune feature pruning on a substantially larger corpus, recording vocabulary,
  sparse storage, quality and latency separately. The six sentences cannot
  establish large-corpus efficiency or production retrieval quality.
