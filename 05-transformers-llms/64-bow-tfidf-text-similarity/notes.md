# Weighting lexical evidence

## Counts, DF and corpus state

For $N$ documents and $V$ vocabulary terms, the count matrix is
$X\in\mathbb{R}^{N\times V}$ with $X_{ij}=f(t_j,d_i)$. Binary BoW uses
$\mathbf{1}[X_{ij}>0]$. Neither unigram representation retains order: “dog bites
man” and “man bites dog” collide. Word bigrams preserve some local order at the
cost of more features; character n-grams can preserve partial spelling overlap
without learning synonym equivalence.

Document frequency is $DF(t)=\sum_{i=1}^{N}\mathbf{1}[f(t,d_i)>0]$.
An empty document contributes to $N$ even though it adds no vocabulary terms.
Repeating a term within one document does not increase its DF.

Classical IDF $\log(N/DF(t))$ is the self-information of document-level term
presence under the empirical probability $DF(t)/N$. This does not quantify
semantic importance or mutual information with a relevance label.
Our implementation uses the [scikit-learn convention](references.md):

$$
IDF(t)=1+\log\frac{N+1}{DF(t)+1}.
$$

With `smooth_idf=False`, scikit-learn uses $1+\log(N/DF(t))$; the external
$+1$ remains. A term in every document therefore has IDF one in both variants.
The implementation learns its vocabulary, so every fitted term has positive DF.
It does not expose fixed-vocabulary zero-DF behavior.

## Local frequency and normalization

Raw TF is $f(t,d)$. Sublinear TF is zero for absent terms and
$1+\log f(t,d)$ otherwise. TF-IDF before normalization is
$w_{t,d}=TF(t,d)IDF(t)$. Relative-frequency TF $f(t,d)/|d|$ is another convention;
it is not scikit-learn's default TF. If raw TF is scaled by one positive constant
per document and then L2-normalized, that scalar cancels. Sublinear TF changes
relative coordinates and does not cancel this way.

For nonzero $\mathbf{w}$, L2 normalization gives
$\hat{\mathbf{w}}=\mathbf{w}/\|\mathbf{w}\|_2$. Scaling every count by the same
positive factor leaves raw-TF cosine unchanged with fixed IDF. Repeating just
one term changes the direction. Cosine normalization does not erase every
length effect or model diminishing term returns in the same way as BM25.

$$
\cos(\mathbf{a},\mathbf{b})=
\frac{\mathbf{a}^{\mathsf T}\mathbf{b}}{\|\mathbf{a}\|_2\|\mathbf{b}\|_2}.
$$

For unit nonzero vectors, cosine equals dot product and
$\|\mathbf{a}-\mathbf{b}\|_2^2=2-2\cos(\mathbf{a},\mathbf{b})$.
Their rankings agree under those conditions. TF-IDF coordinates are nonnegative,
so cosine lies in $[0,1]$ up to rounding. General signed-vector cosine can be
negative. Zero-vector cosine is mathematically undefined; the educational
helper assigns score zero, while the retriever returns `zero_query` and no hits.
The unit-vector distance identity does not apply to zero rows.

## Fitted state and application boundaries

The analyzer lowercases and extracts Unicode word tokens of at least two
characters, matching default `TfidfVectorizer` word unigrams. It does not perform
stemming, Unicode canonical normalization or identifier-aware parsing. This is
a different explicit analyzer contract from [Day 63](../63-classical-nlp/).
`model` does not match `models`; stop words, including negation, are retained.
Keeping `not` still does not make a bag of words understand negation or entailment.

For supervised tasks, split first and fit inside training folds with a Pipeline.
For retrieval, fitting the searchable collection is intentional; incoming queries
use `transform`, and held-out relevance judgments must not guide test tuning.
Refreshing a collection changes vocabulary and IDF: version the analyzer,
vocabulary and statistics together, then rebuild document and query transforms
consistently. Appending new counts to an old index while using new IDF is invalid.

CSR stores nonzero values plus column indices and row pointers, approximately
$O(nnz+N)$ storage rather than $O(NV)$. Sparse output is not a guarantee of cheap
computation: a common query term can still match most documents. The example
computes one sparse query-to-document cosine vector and sorts its positive
entries. It scans the stored representation, with no inverted-index candidate
pruning. A full pairwise document matrix can have $O(N^2)$ nonzeros.
The from-scratch dense matrices intentionally expose the arithmetic for small
fixtures and are not replacements for library implementations.

An inverted index maps terms to posting lists, often storing frequencies and
positions. It supports candidate access; a sparse matrix is a numerical
representation. BM25 typically combines an IDF variant, bounded TF contribution
and document length relative to the corpus average. Sublinear TF-IDF still grows
logarithmically before normalization; BM25's TF factor approaches a finite
limit at fixed length. Parameter/IDF variants require explicit documentation.
The analytical example does not implement BM25. The separate
[visual laboratory](VISUAL_LAB.md) adds an educational scorer and distinct
synthetic protocols; it does not replace the analytical record below.

Lexical search can complement dense retrieval for identifiers, technical terms
and paraphrases. Raw lexical and embedding scores need not share a scale; rank
fusion is an option, not evidence of improvement. Evaluate lexical, dense and
hybrid configurations using representative relevance labels, candidate recall,
MRR/nDCG, context constraints and latency. No such evaluation occurred here.

## Execution record — 2026-11-24

New analytical run by Codex; this is separate from the private study's unexecuted
drafts and Day 63's historical analytical/visual records. No random sampling is
used. All documents are synthetic strings defined in `example.py`.

Environment: Windows, Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1,
scikit-learn 1.9.1. Reproduction from the repository root:

```powershell
python 05-transformers-llms/64-bow-tfidf-text-similarity/example.py
python 05-transformers-llms/64-bow-tfidf-text-similarity/from_scratch.py
```

Locally these commands used the explicit stable Python interpreter documented
by the repository workflow. Both scripts completed successfully. The report is
printed to stdout; no generated report or visual asset is selected for Git.

### A. Count cosine versus TF-IDF cosine

**Hypothesis:** IDF changes directional weights and may distinguish count-score
ties, while synonym-only OOV queries remain unrepresented.

**Configuration:** the six strings in `DOCUMENTS`, default lowercased word
unigrams, no stop-word removal/pruning, smoothed IDF, raw TF, L2 TF-IDF,
cosine scoring for both representations, $k=3$, positive hits only and
zero-based corpus IDs for tie breaking. Both vectorizers produced a $6\times27$
matrix with 31 nonzero entries. `model` is OOV even though `models` is indexed.

**Observed results** (scores rounded to six decimals):

| Query | Count cosine: document ID, score | TF-IDF cosine: document ID, score |
|---|---|---|
| `machine learning model` | 0: 0.632456; 2: 0.632456; 1: 0.288675 | 0: 0.548790; 2: 0.526695; 1: 0.196768 |
| `neural networks` | 1: 0.577350 | 1: 0.623073 |
| `car servicing` | 5: 0.447214 | 5: 0.447214 |
| `car repair` | `zero_query`, no hits | `zero_query`, no hits |
| empty string | `zero_query`, no hits | `zero_query`, no hits |

**Interpretation candidate — author review required:** IDF broke the count-score
tie between documents 0 and 2, although the displayed ordering stayed the same.
The partial query matched `servicing`, without evidence of synonym understanding.

**Limitation:** no relevance labels, retrieval metric, timing or real corpus.
Larger or smaller cosine scores across representations do not establish better
retrieval. The two representations have different geometry.

### B. Repetition with independent weighting controls

**Hypothesis:** sublinear TF reduces repetition's unnormalized contribution;
L2 changes magnitude separately.

**Configuration:** fit once per configuration on `machine learning model`,
`deep learning model`, `machine pipeline`. Transform variants containing
1, 3 or 20 occurrences of `machine`, followed by `learning model`. Cross raw
versus sublinear TF with `norm=None` versus `norm='l2'`. All four fits have the
same vocabulary and IDF; `machine` IDF is 1.287682.

**Observed results:**

| TF | Normalization | `machine` weights for 1 / 3 / 20 occurrences |
|---|---|---|
| raw | none | 1.287682 / 3.863046 / 25.753641 |
| sublinear | none | 1.287682 / 2.702345 / 5.145233 |
| raw | L2 | 0.577350 / 0.904534 / 0.997509 |
| sublinear | L2 | 0.577350 / 0.829279 / 0.942697 |

All six normalized rows (three per TF variant) had measured L2 norm 1.0.

**Interpretation candidate — author review required:** logarithmic TF reduced
the single repeated coordinate's dominance in this fixed-IDF fixture.
Normalization constrained magnitude without making that coordinate constant.

**Limitation:** this is a coordinate-level mechanism check, not evidence of
ranking quality, resistance to keyword stuffing or superiority over BM25.

### C. Corpus-dependent IDF

**Hypothesis:** increasing the document prevalence of `transformer` reduces IDF.

**Configuration:** initially `machine learning`, `deep learning`,
`transformer architecture`. Append four documents pairing `transformer` with
`training`, `inference`, `optimization` and `deployment`. Refit after extension.

**Observed result:** $N=3$, $DF=1$, IDF 1.693147 became $N=7$, $DF=5$,
IDF 1.287682. Both vocabulary and corpus size changed in the refit.

**Interpretation candidate — author review required:** the observed IDF change
matches the smoothed formula and illustrates that corpus statistics are fitted
state. This is not a measured retrieval-rank change.

**Limitation:** a designed seven-document extension does not characterize
operational corpus drift, refresh schedules or production quality.

### Numerical reference check

`from_scratch.py` compared vocabulary mappings, IDF, document matrices and query
transforms with scikit-learn on four synthetic documents (one empty) and three
queries (known, OOV and empty). Both raw and sublinear TF passed at absolute and
relative tolerances of $10^{-12}$. This supports implementation agreement for
the specified configurations, not retrieval effectiveness or author approval.

### Automated verification for this delivery

Using the same Python 3.11.14 environment and pytest 9.2.0:

```powershell
python scripts/validate_repo.py tests --scope 05-transformers-llms/64-bow-tfidf-text-similarity
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
uv --no-cache pip check --offline --python .local/environments/n09-full-2026-11-13/Scripts/python.exe
git diff --check
```

The focused test file passed 29 cases, without skips or warnings. Repository
syntax parsing passed for 445 Python files; internal-link checking passed for
1,625 links in 283 Markdown files. Dependency compatibility passed for 81
installed packages. `git diff --check` passed, with Git's LF-to-CRLF conversion
warning for the module index. No full curriculum test run, CI execution,
browser/GitHub math-render inspection or production verification was performed.

## Suggested work — not executed

- Compare word unigrams, word bigrams and character 3–5-grams on controlled OCR
  corruption, using held-out relevance labels rather than lexical scores as truth.
- Evaluate exact-identifier retrieval with a structured field versus default
  tokenization; test collisions caused by punctuation removal.
- Compare TF-IDF, BM25 and a dense model on a representative query set with
  frozen tuning/evaluation partitions. This would require new dependencies or
  model resources and an explicitly documented evaluation protocol.
