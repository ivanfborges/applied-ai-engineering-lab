# Day 63 — Classical NLP

[Module index](../README.md) · [Roadmap](../../ROADMAP.md)

**Which text differences should a representation remove, and which must it preserve?**
Tokenization, normalization, stemming, lemmatization and n-grams encode that
decision before a classifier or search engine ever sees a feature vector.
This study opens Module 5 with inspectable, offline examples on synthetic text.

Tokenization chooses units; normalization makes selected forms equivalent.
Stemming merges forms through rules, while lemmatization uses linguistic
information to identify base forms. N-grams retain local sequence information.
These choices matter for ticket classification, lexical retrieval, OCR and
hybrid search. They require separate evaluation for each language and task.

## What is implemented

- NFC normalization, case folding, whitespace collapse and apostrophe unification.
- A deterministic regex analyzer retaining accents, numbers and negation.
- Contiguous word n-grams and an educational dense count matrix, with optional
  fixed vocabulary for new documents. Tests compare counts with scikit-learn.
- Sparse TF-IDF/cosine diagnostics for unigram, unigram–bigram and
  unigram–trigram features on six hand-authored synthetic administrative sentences.
- A control showing that deleting `not` makes opposing token sequences identical,
  plus explicit handling of queries with no matching features.
- A deliberately incomplete suffix stripper and a closed word/POS lemma lookup.
  These illustrate morphology and its failure modes; they are **not Porter,
  WordNet, automatic POS tagging or general English morphology implementations**.

In the analytical scripts, morphological transformations are demonstrated separately and are not applied
to the retrieval corpus. That example has no relevance labels and reports no
retrieval-quality metric or benchmark. TF-IDF is used only to expose the effect
of representation choices; Day 64 develops weighting and similarity further.

## Run from the repository root

Use Python 3.11+ and the canonical dependencies in
[pyproject.toml](../../pyproject.toml). The analytical scripts and test file below
need no NLP model or corpus download. The separate visual lab has additional
resource requirements described below.

```powershell
python -m pip install -e ".[dev]"
python 05-transformers-llms/63-classical-nlp/from_scratch.py
python 05-transformers-llms/63-classical-nlp/example.py
python scripts/validate_repo.py tests --scope 05-transformers-llms/63-classical-nlp/tests/test_classical_nlp.py
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
```

The analytical scripts print matrices, morphology demonstrations and a JSON diagnostic
report to stdout. They do not export assets. No generated preview is selected;
saved reports should stay under ignored `outputs/` or `.local/` directories.

| File | Purpose |
|---|---|
| [from_scratch.py](from_scratch.py) | Analyzer, window counts, fixed columns and explicitly limited morphology demonstrations |
| [example.py](example.py) | Sparse library example, deterministic ranking and destructive-negation control |
| [tests/test_classical_nlp.py](tests/test_classical_nlp.py) | Count parity, Unicode, invalid inputs, analyzer boundaries and query isolation |
| [notes.md](notes.md) | Theory, assumptions, execution record and proposed experiments |
| [interview_questions.md](interview_questions.md) | Questions about information loss, morphology, leakage and retrieval |
| [references.md](references.md) | Primary documentation and scope of the data |

## Separate interactive visual lab

The [visual guide](VISUAL_LAB.md) covers 13 groups of computed demonstrations:
token boundaries, Unicode, real NLTK morphology, sliding windows, vocabulary and
CSR storage, count matrices, TF-IDF surfaces, SVD geometry, cosine, statistical
language models, judged retrieval ablations, repeated OCR corruption and a
clearly illustrative RAG architecture. Its corpora and execution records are
separate from the analytical study above.

The lab uses actual Porter/Snowball/WordNet outputs, supplied POS annotations and
a locally trained character BPE. Statistical models, scores and plots are computed;
no pretrained tokenization or dense retrieval result is invented. WordNet needs
an explicit resource download. The optional dashboard provides editable inputs;
all essential demonstrations also export without Streamlit.

```powershell
python -m pip install -e ".[dev,nlp]"
python -m nltk.downloader -d .local/nltk_data wordnet
python 05-transformers-llms/63-classical-nlp/visualizations.py --all --quick
python -m streamlit run 05-transformers-llms/63-classical-nlp/app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

Start with [visualizations.py](visualizations.py), [app.py](app.py) and the shared
[numerical core](nlp_lab/core.py). Focused tests cover
[calculations](tests/test_visual_core.py), [real exports](tests/test_visual_exports.py)
and [dashboard behavior](tests/test_visual_app.py). Exports and their manifest
stay ignored under `assets/generated/`; no public visual preview is selected.
The [dated visual record](VISUAL_LAB.md#execution-record--visual-phase-2026-11-24)
keeps measured findings, illustrative architecture and unrun extensions distinct.

## Limits and takeaways

The regex splits `47.838/2020` into three tokens, discards emoji and punctuation,
and can join context across sentence boundaries inside a document. It does not
segment languages without explicit word boundaries or preserve every Unicode
combining mark. It keeps contractions such as `don't` whole rather than parsing
their grammatical structure. Original text must remain available for evidence
and identifiers. See the [analyzer contract](notes.md#analyzer-contract).

Fewer features do not establish better relevance. Bigrams expose `not approved`
but do not implement negation understanding. Query vocabulary must stay fixed;
supervised evaluation additionally requires fitting corpus statistics within
each training fold. A pretrained embedding or language model has its own
tokenizer and input contract; this analyzer is for lexical features.

Implementation, automated checks and author interpretation review are separate.
The dated [execution record](notes.md#executed-study--2026-11-24) contains bounded
observations and interpretation candidates. **Author review required.**
