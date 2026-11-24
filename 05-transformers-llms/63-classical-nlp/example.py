"""Offline sparse lexical diagnostics on six hand-authored synthetic documents."""

import json
import platform
import unicodedata

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from from_scratch import normalize, tokenize


DOCUMENTS = (
    "The environmental license was approved.",
    "The environmental license was not approved.",
    "The company operated without licenses.",
    "Companies operating without environmental licenses.",
    "The administrative application was rejected.",
    "The environmental inspection identified violations.",
)
QUERY = "environmental license not approved"


def lexical_diagnostics(
    documents: list[str] | tuple[str, ...], query: str, max_n: int = 1,
) -> dict:
    """Fit only the index corpus. Ties use original document order.

    This returns similarities and representation diagnostics, not relevance
    metrics. An all-out-of-vocabulary query is reported as having no matches.
    """
    if isinstance(documents, (str, bytes)) or not isinstance(documents, (list, tuple)):
        raise TypeError("documents must be a list or tuple of strings")
    if not documents or not all(isinstance(doc, str) for doc in documents):
        raise ValueError("documents must contain at least one string and no other types")
    if isinstance(max_n, bool) or not isinstance(max_n, int):
        raise TypeError("max_n must be an integer")
    if max_n < 1:
        raise ValueError("max_n must be positive")
    normalize(query)
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, token_pattern=None, lowercase=False,
        ngram_range=(1, max_n), stop_words=None,
        smooth_idf=True, norm="l2", sublinear_tf=False,
    )
    matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranking = sorted(range(len(documents)), key=lambda i: (-float(scores[i]), i))
    return {
        "ngram_range": [1, max_n],
        "shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "features": vectorizer.get_feature_names_out().tolist(),
        "query_nnz": int(query_vector.nnz),
        "no_matching_features": query_vector.nnz == 0,
        "ranking": [{"document_id": i, "score": float(scores[i])}
                    for i in ranking if scores[i] > 0],
    }


def negation_collision() -> dict:
    """Demonstrate irreversible information loss with an intentionally bad policy."""
    positive = tokenize("The license was approved.")
    negative = tokenize("The license was not approved.")
    return {
        "tokens_distinct_before": positive != negative,
        "tokens_equal_after_dropping_not": positive == [t for t in negative if t != "not"],
    }


def main() -> None:
    report = {
        "data": "six hand-authored synthetic administrative sentences; no relevance labels",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "unicode": unicodedata.unidata_version},
        "documents": list(DOCUMENTS),
        "query": QUERY,
        "normalization": {"original": "  INFORMA\u00c7A\u0303O\tStra\u00dfe  ",
                          "normalized": normalize("  INFORMA\u00c7A\u0303O\tStra\u00dfe  ")},
        "configurations": [lexical_diagnostics(DOCUMENTS, QUERY, n) for n in (1, 2, 3)],
        "negation_control": negation_collision(),
        "oov_control": lexical_diagnostics(DOCUMENTS, "unseentokenxyz", 2),
        "interpretation_status": "author review required; similarities are not relevance judgments",
    }
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
