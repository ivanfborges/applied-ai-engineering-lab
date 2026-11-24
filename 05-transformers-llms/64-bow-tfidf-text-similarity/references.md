# References and provenance

- [scikit-learn: text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
  — count vectors, sparse storage, IDF conventions and limitations of lexical features.
- [scikit-learn: TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
  — default analyzer, fitted attributes, smoothing, sublinear TF and normalization.
- [scikit-learn: cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
  — normalized dot product and sparse output contract.
- [scikit-learn: common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
  — training-only preprocessing and pipelines.
- [Manning, Raghavan and Schütze: Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
  — inverted indexes, vector-space scoring and probabilistic retrieval, including BM25.
- [Lucene BM25Similarity](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/search/similarities/BM25Similarity.html)
  — positive BM25 IDF convention and saturation/length parameters; not a claim of engine-score parity.
- [Elasticsearch: reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
  — one-based rank contributions and missing-document behavior.
- [scikit-learn: TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
  — sparse uncentered projections, fitted query transforms and explained variance.
- [SciPy: linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
  — condensed dissimilarities and average linkage; Ward requires Euclidean geometry.

The scikit-learn TF-IDF API and text-extraction guide were consulted on
2026-11-24. Execution records identify installed versions; stable documentation
can change. The added primary visual references were consulted on the same date.
The textbook remains further reading; the visual BM25 comparison uses the
explicit educational formula documented in the visual guide.

All example documents, queries and control corpora are synthetic and defined
in code. There is no external dataset, relevance annotation, neural model or
production observation. Numerical reference values come from the installed
scikit-learn implementation, not from a retrieval benchmark.
