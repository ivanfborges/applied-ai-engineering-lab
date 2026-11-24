# Classical NLP — interview questions

**What is the difference between normalization and tokenization?**

Normalization selects equivalences between textual forms. Tokenization selects
the units represented downstream. Order matters: canonical Unicode normalization
before regex matching prevents some equivalent spellings from segmenting
differently. A tokenizer's output need not consist of linguistic words.

**Why does lemmatization need POS, and is supplied POS enough?**

The noun `saw` and past-tense verb `saw` have different base forms. Supplied POS
removes one ambiguity but resources can still have multiple candidates or missing
entries. A lookup example with manual POS does not demonstrate an automatic
contextual lemmatizer. This study's closed fixture makes that limitation explicit.

**Would a stemmer producing dictionary words be inherently better?**

No. Stemming aims at useful conflation, not dictionary validity. Evaluate whether
it merges relevant variants and creates harmful collisions. The intentionally
bad `news → new` rule here illustrates the danger, not a benchmark against Porter.
Choose language-specific maintained tools for real use.

**Do bigrams solve bag-of-words and negation limitations?**

They preserve some adjacent order. `not approved` becomes a feature, but long
scope and composition remain unresolved. Unigrams can already retain `not`.
Deleting that word can destroy the distinction before bigrams are formed.

**Why can a smaller vocabulary be a worse representation?**

Reduction can consolidate evidence or erase signal. Merging `US` and `us`, or
legal identifiers after punctuation removal, can be harmful. Measure downstream
quality and error categories, not just vocabulary size or matrix density.

**What is leakage in a text pipeline?**

In supervised evaluation, fitting vocabulary, IDF or feature selection before
splitting lets held-out data influence training. Fit within each training fold.
Retrieval normally fits the indexed collection, but queries and their judgments
must not silently alter that representation or tune a held-out evaluation.

**What should a retrieval system do with an all-OOV query?**

Report that the representation has no matching evidence. Sorting a vector of
zeros gives a deterministic but meaningless ranking. This example returns an
empty ranking; real systems may use separately evaluated fallback retrieval.

**How do you preserve legal citations in hybrid RAG?**

Keep original text and exact identifier fields. Use separate derived lexical
representations and a dense model's native input contract. Evaluate exact
references, paraphrases, negation and OCR errors with labeled queries. A cosine
score alone is neither a relevance judgment nor evidence of reliable generation.

**A one-minute explanation**

Classical NLP chooses which information enters a statistical representation.
Tokenization defines units; normalization removes selected variation; stemming
and lemmatization consolidate forms with different assumptions; n-grams preserve
local order. Sparse features remain useful baselines for classification and
lexical retrieval. I would preserve original evidence, fit learned text statistics
at the correct boundary, test language-specific failure cases, and evaluate each
transformation against downstream judgments. Preprocessing is a task decision,
not a universal cleaning recipe.
