"""Numerical reference agreement, geometric identities and query policies."""

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.sparse import isspmatrix_csr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import EducationalTfidf, cosine_scores, l2_normalize, tokenize
from example import LexicalRetriever, corpus_drift_control, repetition_control


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("sublinear", [False, True])
@pytest.mark.parametrize("norm", [None, "l2"])
def test_fitted_state_and_query_parity(smooth, sublinear, norm):
    docs = ["Café café common 42", "common model", "model common common", "", "a !"]
    own = EducationalTfidf(smooth_idf=smooth, sublinear_tf=sublinear, norm=norm)
    reference = TfidfVectorizer(smooth_idf=smooth, sublinear_tf=sublinear, norm=norm)
    actual = own.fit_transform(iter(docs))
    expected = reference.fit_transform(docs).toarray()
    assert own.vocabulary_ == reference.vocabulary_
    np.testing.assert_allclose(own.idf_, reference.idf_, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    snapshot = own.idf_.copy()
    vocabulary = own.vocabulary_.copy()
    queries = ["café unknown", "model model common", "OOV", "", "a", "42"]
    np.testing.assert_allclose(own.transform(queries), reference.transform(queries).toarray(),
                               atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(own.idf_, snapshot)
    assert own.vocabulary_ == vocabulary
    assert own.transform([]).shape == (0, len(vocabulary))


def test_document_frequency_counts_documents_including_empty_rows():
    model = EducationalTfidf(norm=None).fit(["common rare rare rare", "common", ""])
    assert model.idf_[model.vocabulary_["rare"]] == pytest.approx(1 + np.log(4 / 2))
    assert model.idf_[model.vocabulary_["common"]] == pytest.approx(1 + np.log(4 / 3))
    ubiquitous = EducationalTfidf().fit(["common", "common common"])
    np.testing.assert_array_equal(ubiquitous.idf_, [1])


def test_tokenizer_boundary_matches_default_sklearn():
    text = "I DON'T naïve_42 café 1 47.383/2018 🙂"
    assert tokenize(text) == TfidfVectorizer().build_analyzer()(text)
    assert "47.383/2018" not in tokenize(text)


def test_cosine_identities_zero_policy_and_large_finite_values():
    raw = np.array([[3., 4., 0.], [6., 8., 0.], [0., 0., 7.], [0., 0., 0.]])
    unit = l2_normalize(raw)
    scores = cosine_scores(raw, raw)
    np.testing.assert_allclose(scores, cosine_similarity(raw), atol=1e-14)
    np.testing.assert_allclose(scores, scores.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(scores), [1, 1, 1, 0], atol=1e-14)
    np.testing.assert_allclose(scores, unit @ unit.T, atol=1e-14)
    distance2 = ((unit[:3, None] - unit[None, :3]) ** 2).sum(axis=2)
    np.testing.assert_allclose(distance2, 2 - 2 * scores[:3, :3], atol=1e-14)
    assert scores[0, 1] == pytest.approx(1)
    assert scores[0, 2] == 0
    np.testing.assert_allclose(l2_normalize([[1e308, 1e308]]), [[2**-.5, 2**-.5]])


@pytest.mark.parametrize("bad", [[], [""], ["a !"], ["", "!"]])
def test_empty_fit_fails(bad):
    with pytest.raises(ValueError, match="empty vocabulary"):
        EducationalTfidf().fit(bad)


@pytest.mark.parametrize("bad", [None, "one document", [12], [b"bytes"]])
def test_invalid_corpus_fails(bad):
    with pytest.raises(TypeError, match="string"):
        EducationalTfidf().fit(bad)
    with pytest.raises(TypeError, match="string"):
        LexicalRetriever(bad)


def test_invalid_state_options_and_vectors():
    with pytest.raises(ValueError, match="fit"):
        EducationalTfidf().transform(["text"])
    with pytest.raises(ValueError, match="norm"):
        EducationalTfidf(norm="l1")
    with pytest.raises(TypeError, match="booleans"):
        EducationalTfidf(sublinear_tf="false")
    for bad in ([1, 2], [[np.nan]], [[np.inf]]):
        with pytest.raises(ValueError, match="finite"):
            l2_normalize(bad)
    with pytest.raises(ValueError, match="features"):
        cosine_scores([[1, 2]], [[1]])


@pytest.mark.parametrize("representation", ["count", "tfidf"])
def test_sparse_search_reference_ties_and_zero_queries(representation):
    retriever = LexicalRetriever(["alpha beta", "alpha beta", "gamma", ""],
                                 representation=representation)
    assert isspmatrix_csr(retriever.matrix)
    result = retriever.search("alpha", k=10)
    assert [hit["document_id"] for hit in result["hits"]] == [0, 1]
    expected = cosine_similarity(retriever.vectorizer.transform(["alpha"]), retriever.matrix)[0]
    np.testing.assert_allclose([hit["score"] for hit in result["hits"]], expected[:2])
    for query in ("", "! a", "unknown"):
        assert retriever.search(query) == {"status": "zero_query", "hits": []}
    assert len(retriever.search("alpha", k=1)["hits"]) == 1
    assert retriever.search("gamma")["hits"][0]["document_id"] == 2


@pytest.mark.parametrize("bad,error", [(0, ValueError), (-1, ValueError),
                                       (True, TypeError), (1.2, TypeError)])
def test_invalid_k(bad, error):
    with pytest.raises(error, match="positive integer"):
        LexicalRetriever(["alpha"]).search("alpha", k=bad)


def test_invalid_retriever_inputs():
    with pytest.raises(ValueError, match="empty"):
        LexicalRetriever([])
    with pytest.raises(ValueError, match="representation"):
        LexicalRetriever(["alpha"], representation="dense")
    with pytest.raises(TypeError, match="string"):
        LexicalRetriever(["alpha"]).search(None)


def test_repetition_freezes_idf_and_separates_normalization():
    raw, raw_unit, log, log_unit = repetition_control()
    np.testing.assert_allclose(np.array(raw["machine_weights"]) / raw["machine_weights"][0],
                               [1, 3, 20])
    np.testing.assert_allclose(np.array(log["machine_weights"]) / log["machine_weights"][0],
                               1 + np.log([1, 3, 20]))
    for normalized in (raw_unit, log_unit):
        np.testing.assert_allclose(normalized["row_norms"], 1)
    assert log_unit["machine_weights"][-1] < raw_unit["machine_weights"][-1]


def test_drift_idf_uses_new_corpus_statistics():
    before, after = corpus_drift_control()
    assert before["transformer_idf"] == pytest.approx(1 + np.log(4 / 2))
    assert after["transformer_idf"] == pytest.approx(1 + np.log(8 / 6))
    assert after["transformer_idf"] < before["transformer_idf"]
