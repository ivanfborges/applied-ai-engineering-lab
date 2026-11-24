"""Sparse lexical retrieval and deterministic synthetic weighting controls."""

import json
import platform

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from from_scratch import validate_documents


DOCUMENTS = (
    "Machine learning models improve predictions",
    "Deep learning models use neural networks",
    "Machine learning pipelines process data",
    "Vector databases support semantic search",
    "Retrieval augmented generation uses document retrieval",
    "Automobile servicing requires regular maintenance",
)
QUERIES = ("machine learning model", "neural networks", "car servicing", "car repair", "")


class LexicalRetriever:
    """Small sparse scan, with positive scores only and input-order tie breaking.

    IDs are zero-based corpus positions. This is not an inverted-index engine.
    """

    def __init__(self, documents, *, representation="tfidf"):
        self.documents = tuple(validate_documents(documents))
        if not self.documents:
            raise ValueError("documents must not be empty")
        if representation == "count":
            self.vectorizer = CountVectorizer()
        elif representation == "tfidf":
            self.vectorizer = TfidfVectorizer(norm="l2", smooth_idf=True, sublinear_tf=False)
        else:
            raise ValueError("representation must be 'count' or 'tfidf'")
        self.matrix = self.vectorizer.fit_transform(self.documents).tocsr()

    def search(self, query, *, k=3):
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be a positive integer")
        if k < 1:
            raise ValueError("k must be a positive integer")
        vector = self.vectorizer.transform([query])
        if vector.nnz == 0:
            return {"status": "zero_query", "hits": []}
        # Only positive query-document products are materialized, not an N x N matrix.
        scores = cosine_similarity(vector, self.matrix, dense_output=False).tocsr()
        candidates = [(int(i), float(s)) for i, s in zip(scores.indices, scores.data) if s > 0]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        hits = [{"document_id": i, "score": s} for i, s in candidates[:k]]
        return {"status": "ok" if hits else "no_overlap", "hits": hits}


def repetition_control():
    """Freeze vocabulary and IDF; change TF scaling and normalization independently."""
    fit_documents = ["machine learning model", "deep learning model", "machine pipeline"]
    variants = ["machine " * n + "learning model" for n in (1, 3, 20)]
    result = []
    for sublinear in (False, True):
        for norm in (None, "l2"):
            vectorizer = TfidfVectorizer(sublinear_tf=sublinear, norm=norm)
            vectorizer.fit(fit_documents)
            matrix = vectorizer.transform(variants)
            column = vectorizer.vocabulary_["machine"]
            result.append({"sublinear_tf": sublinear, "norm": norm,
                           "machine_weights": matrix[:, column].toarray().ravel().tolist(),
                           "row_norms": np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1.tolist()})
    return result


def corpus_drift_control():
    before = ["machine learning", "deep learning", "transformer architecture"]
    after = before + [f"transformer {word}" for word in
                      ("training", "inference", "optimization", "deployment")]
    result = []
    for documents in (before, after):
        vectorizer = TfidfVectorizer().fit(documents)
        result.append({"documents": len(documents),
                       "transformer_df": sum("transformer" in doc.split() for doc in documents),
                       "transformer_idf": float(vectorizer.idf_[vectorizer.vocabulary_["transformer"]])})
    return result


def run_study():
    retrieval = {}
    for representation in ("count", "tfidf"):
        retriever = LexicalRetriever(DOCUMENTS, representation=representation)
        retrieval[representation] = {
            "shape": list(retriever.matrix.shape), "nonzero": int(retriever.matrix.nnz),
            "queries": {query: retriever.search(query) for query in QUERIES},
        }
    return {
        "data": "synthetic, code-defined; no relevance labels or benchmark",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "retrieval": retrieval,
        "repetition_counts": [1, 3, 20], "repetition": repetition_control(),
        "corpus_drift": corpus_drift_control(),
        "interpretation_status": "author review required",
    }


if __name__ == "__main__":
    print(json.dumps(run_study(), indent=2, allow_nan=False))
