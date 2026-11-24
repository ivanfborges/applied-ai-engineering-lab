# References and data provenance

Primary documentation consulted for this implementation on 2026-11-24:

- [Python 3.11: Unicode normalization](https://docs.python.org/3.11/library/unicodedata.html):
  normalization forms and the Unicode database version exposed by the runtime.
- [Python 3.11: string case folding](https://docs.python.org/3.11/library/stdtypes.html#str.casefold):
  caseless matching is a different operation from canonical normalization.
- [scikit-learn: text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):
  analyzer customization, sparse text representations and weighting conventions.
- [NLTK: Porter stemmer](https://www.nltk.org/api/nltk.stem.porter.html):
  a real rule-based stemmer with explicitly selectable algorithm variants.
- [NLTK: WordNet lemmatizer](https://www.nltk.org/api/nltk.stem.wordnet.html):
  resource-backed morphology with POS inputs and a documented fallback policy.

NLTK is further reading for the original analytical scripts; the separate
[visual lab](VISUAL_LAB.md) executes real NLTK morphology and boundary rules.
Those outputs need not match the analytical toy suffix rules. The explicit
WordNet download remains local and ignored, and follows the
[Princeton WordNet license](https://wordnet.princeton.edu/license-and-commercial-use).
No pretrained model or resource is bundled into public source or downloaded by
widgets or visualization generation.

All sentences, queries and the word/POS lemma fixture are hand-authored synthetic
teaching inputs defined in the source code. They do not represent legal records,
real user searches, an annotated evaluation collection or a linguistic benchmark.
The analytical lemma entries are supplied examples rather than measured predictions.
The visual phase instead records actual WordNet outputs and candidates. Its
synthetic relevance judgments and POS annotations are declared in source before
ranking; they are teaching labels, not author-validated real-world ground truth.
