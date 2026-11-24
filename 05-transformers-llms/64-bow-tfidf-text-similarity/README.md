# Day 64 — Bag of Words, TF-IDF and Text Similarity

How does corpus-dependent weighting change lexical similarity, and what can a
similarity score actually establish? This study connects count vectors, document
frequency, TF-IDF and cosine geometry to a small executable lexical retriever.
These are useful foundations for sparse text classification and search baselines.

## Representation and retrieval

BoW records term counts. TF-IDF multiplies local counts by corpus-level IDF,
then optionally normalizes each row. Documents and queries must share the same
fitted vocabulary and IDF. The example compares **count cosine** with **TF-IDF
cosine**, rather than comparing an unnormalized count dot product with cosine.

The code-defined six-document collection is **synthetic**. It includes machine
learning, retrieval and automobile maintenance phrases. No dataset download,
neural model, external service or credentials are required. There are no
relevance labels: rankings demonstrate lexical mechanics, not retrieval quality.

## Files and execution

- [example.py](example.py): CSR retrieval, positive-score top-k, deterministic
  ties, empty/OOV handling, independent TF/normalization and corpus-drift controls.
- [from_scratch.py](from_scratch.py): educational dense NumPy word-unigram
  TF-IDF, fitted query transformation, normalization and cosine calculation.
- [tests/test_tfidf.py](tests/test_tfidf.py): reference agreement, fitted-state
  preservation, geometry, query policies and invalid-input checks.
- [notes.md](notes.md): formulas, assumptions, trade-offs and dated execution record.
- [interview_questions.md](interview_questions.md): questions about weighting,
  leakage, zero vectors, indexing and hybrid retrieval.
- [references.md](references.md): primary technical sources and data provenance.
- [visualizations.py](visualizations.py) and [VISUAL_LAB.md](VISUAL_LAB.md):
  separate offline visual laboratory, output catalog and execution evidence.
- [retrieval_dashboard.py](retrieval_dashboard.py): local arbitrary-query search,
  per-feature explanations, configurable sparse storage and judged evaluation.

From the repository root, using Python 3.11+ and the canonical dependencies:

```powershell
python -m pip install -e ".[dev]"
python 05-transformers-llms/64-bow-tfidf-text-similarity/example.py
python 05-transformers-llms/64-bow-tfidf-text-similarity/from_scratch.py
python scripts/validate_repo.py tests --scope 05-transformers-llms/64-bow-tfidf-text-similarity
```

`example.py` prints a JSON report with library versions, query results and
controlled measurements. It writes no files. The tiny repetition control
materializes only one feature column; the retriever keeps document and query
matrices sparse and does not build document-to-document similarities.

## Takeaways and boundaries

- DF counts documents containing a term, not total occurrences. Rare noise can
  receive high IDF; rarity does not establish semantic importance.
- Cosine is a normalized dot product. Its score is neither a probability nor a
  percentage of shared meaning. Zero vectors require an explicit policy.
- `car servicing` can match through `servicing`; `car repair` has no known
  features here. TF-IDF does not infer that `car` and `automobile` are synonyms.
- Default tokenization splits punctuation and drops single-character tokens.
  Exact identifiers may require separate structured fields or a custom analyzer.
- The corpus is fitted for retrieval. Supervised evaluations instead fit the
  vectorizer inside each training fold to protect validation/test boundaries.
- The analytical scripts remain focused on TF-IDF and cosine. The separate
  visual laboratory implements educational BM25, a postings trace, lexical RRF
  and a judged synthetic retrieval fixture. These do not establish scalable
  search, semantic retrieval, latency gains or production readiness.

Implementation and automated verification are distinct from interpretation.
The [dated analytical record](notes.md#execution-record--2026-11-24) contains
interpretation candidates: **author review required**.

## Explore the separate visual laboratory

```powershell
python 05-transformers-llms/64-bow-tfidf-text-similarity/visualizations.py --quick
python 05-transformers-llms/64-bow-tfidf-text-similarity/visualizations.py --all
python 05-transformers-llms/64-bow-tfidf-text-similarity/visualizations.py --section cosine
python -m streamlit run 05-transformers-llms/64-bow-tfidf-text-similarity/retrieval_dashboard.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

The [visual guide](VISUAL_LAB.md) catalogs 41 PNG/GIF/HTML exports across 18
sections: text-to-matrix stages, actual CSR bytes, genuine three-term vectors,
cosine controls, approximate SVD views, OCR/n-grams, score contributions,
BM25, corpus drift, rank fusion and eight judged synthetic queries. HTML pages
bundle Plotly for local offline use. Dense embedding comparison is not executed;
rank fusion uses two actual lexical methods. Generated assets and their dated
records remain ignored; five preview candidates are recommended for author
selection rather than embedded automatically.

The visual configuration and judgments differ from the original analytical
record. Its [tests](tests/test_visual_core.py),
[temporary-directory export checks](tests/test_visual_exports.py) and
[dashboard checks](tests/test_visual_app.py) provide separate implementation
evidence. Historical analytical observations remain unchanged.

[Module index](../README.md) · [Previous: Classical NLP](../63-classical-nlp/)
