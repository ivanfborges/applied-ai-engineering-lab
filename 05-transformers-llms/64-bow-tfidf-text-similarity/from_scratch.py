"""Dense educational TF-IDF; deliberately limited to default word unigrams."""

from collections import Counter
import re

import numpy as np


def validate_documents(documents):
    """Materialize an iterable of strings without interpreting a string as a corpus."""
    if isinstance(documents, (str, bytes)):
        raise TypeError("documents must be an iterable of strings")
    try:
        result = list(documents)
    except TypeError as error:
        raise TypeError("documents must be an iterable of strings") from error
    if any(not isinstance(document, str) for document in result):
        raise TypeError("each document must be a string")
    return result


def tokenize(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return re.findall(r"(?u)\b\w\w+\b", text.lower())


def l2_normalize(matrix):
    """Leave zero rows at zero; mathematical cosine is undefined for these rows."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("matrix must be finite and two-dimensional")
    # Scale first so squaring finite, very large values does not overflow.
    scale = np.max(np.abs(matrix), axis=1, keepdims=True, initial=0)
    scaled = np.divide(matrix, scale, out=np.zeros_like(matrix), where=scale != 0)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    return np.divide(scaled, norms, out=np.zeros_like(matrix), where=norms != 0)


def cosine_scores(left, right):
    """Pairwise cosine with an explicit zero-vector score of zero."""
    left, right = l2_normalize(left), l2_normalize(right)
    if left.shape[1] != right.shape[1]:
        raise ValueError("matrices must have the same number of features")
    return left @ right.T


class EducationalTfidf:
    """Fit sorted vocabulary/IDF once; transform never updates corpus statistics.

    No pruning, custom analyzer, stemming, fixed vocabulary or sparse storage.
    smooth_idf=False retains sklearn's external +1, unlike log(N/DF).
    """

    def __init__(self, *, smooth_idf=True, sublinear_tf=False, norm="l2"):
        if type(smooth_idf) is not bool or type(sublinear_tf) is not bool:
            raise TypeError("smooth_idf and sublinear_tf must be booleans")
        if norm not in (None, "l2"):
            raise ValueError("norm must be None or 'l2'")
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.vocabulary_ = None
        self.idf_ = None

    def _counts(self, documents, vocabulary):
        matrix = np.zeros((len(documents), len(vocabulary)), dtype=float)
        for row, document in enumerate(documents):
            for term, count in Counter(tokenize(document)).items():
                if term in vocabulary:
                    matrix[row, vocabulary[term]] = count
        return matrix

    def fit(self, documents):
        documents = validate_documents(documents)
        terms = sorted({term for document in documents for term in tokenize(document)})
        if not terms:
            raise ValueError("cannot fit an empty vocabulary")
        vocabulary = {term: column for column, term in enumerate(terms)}
        counts = self._counts(documents, vocabulary)
        df = np.count_nonzero(counts, axis=0)
        smoothing = int(self.smooth_idf)
        idf = np.log((len(documents) + smoothing) / (df + smoothing)) + 1
        self.vocabulary_, self.idf_ = vocabulary, idf
        return self

    def transform(self, documents):
        if self.vocabulary_ is None:
            raise ValueError("fit must be called before transform")
        counts = self._counts(validate_documents(documents), self.vocabulary_)
        if self.sublinear_tf:
            nonzero = counts > 0
            counts[nonzero] = 1 + np.log(counts[nonzero])
        weighted = counts * self.idf_
        return l2_normalize(weighted) if self.norm == "l2" else weighted

    def fit_transform(self, documents):
        documents = validate_documents(documents)
        return self.fit(documents).transform(documents)


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer

    documents = ["machine learning model", "deep learning model", "machine machine pipeline", ""]
    query = ["machine pipeline", "unknown", ""]
    for sublinear in (False, True):
        model = EducationalTfidf(sublinear_tf=sublinear)
        reference = TfidfVectorizer(sublinear_tf=sublinear)
        matrix = model.fit_transform(documents)
        expected = reference.fit_transform(documents).toarray()
        assert model.vocabulary_ == reference.vocabulary_
        np.testing.assert_allclose(model.idf_, reference.idf_, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(model.transform(query), reference.transform(query).toarray(),
                                   rtol=1e-12, atol=1e-12)
        print(f"sublinear_tf={sublinear}: reference agreement passed")
    print("Synthetic fixtures only; dense educational implementation.")


if __name__ == "__main__":
    main()
