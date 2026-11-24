"""Independent numerical contracts for BM25, fusion, metrics and sparse fixtures."""

from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_lab import core


def test_idf_boundary_and_information_identity():
    result = core.idf_values(10, np.array([1, 10]))
    np.testing.assert_allclose(result["classical"], -np.log([.1, 1]))
    assert result["smoothed"][-1] == 1
    assert result["bm25"][-1] > 0


@pytest.mark.parametrize("n,df", [(0, 0), (10, 0), (10, 11), (10, np.nan), (-1, 1)])
def test_idf_invalid_domains(n, df):
    with pytest.raises(ValueError): core.idf_values(n, df)


def test_bm25_known_values_zero_and_saturation():
    assert core.bm25_tf(0) == 0
    assert core.bm25_tf(1, 1, 1.2, .75) == pytest.approx(1)
    assert core.bm25_tf(2, 1, 1.2, .75) == pytest.approx(4.4 / 3.2)
    curve = core.bm25_tf(np.arange(10000), 1, 1.2, .75)
    assert (np.diff(curve) > 0).all()
    assert curve[-1] < 2.2
    assert curve[-1] > 2.199
    np.testing.assert_allclose(core.bm25_tf(2, np.array([.5, 1, 2]), 1.2, 0), 4.4 / 3.2)
    assert core.bm25_tf(2, .5) > core.bm25_tf(2, 2)


@pytest.mark.parametrize("kwargs", [{"tf": -1}, {"tf": 1, "k1": 0}, {"tf": 1, "b": 2},
                                     {"tf": 1, "length_ratio": np.inf}])
def test_bm25_invalid_arguments(kwargs):
    with pytest.raises(ValueError): core.bm25_tf(**kwargs)


def test_bm25_hand_computed_scores_and_document_lengths():
    index = core.SearchIndex(["alpha alpha beta", "beta", ""])
    assert index.lengths.tolist() == [3, 1, 0]
    assert index.avgdl == pytest.approx(4 / 3)
    result = index.explain("alpha alpha", "bm25")
    expected = np.log(1 + 2.5 / 1.5) * (2 * 2.2) / (2 + 1.2 * (.25 + .75 * 3 / (4 / 3)))
    np.testing.assert_allclose(result["scores"], [expected, 0, 0])
    np.testing.assert_allclose(result["scores"], index.explain("alpha", "bm25")["scores"])
    assert result["ranking"] == [0]


@pytest.mark.parametrize("method", core.METHODS)
def test_explanations_sum_to_scores_and_zero_query_policy(method):
    index = core.SearchIndex()
    result = index.explain("document retrieval", method)
    np.testing.assert_allclose(result["contributions"].sum(axis=1), result["scores"], atol=1e-12)
    assert set(result["ranking"]).isdisjoint(result["excluded"])
    assert len(result["ranking"]) + len(result["excluded"]) == len(index.documents)
    assert index.explain("", method)["ranking"] == []
    assert index.explain("zzzzzzzz", method)["ranking"] == []
    if method != "bm25":
        q = index.vectorizers[method].transform(["document retrieval"])
        expected = cosine_similarity(q, index.matrices[method])[0]
        np.testing.assert_allclose(result["scores"], expected, atol=1e-12)


def test_rrf_one_based_ranks_missing_items_and_ties():
    order, values = core.rrf([[2, 1], [1, 3]], constant=10)
    assert order == [1, 2, 3]
    assert values[1] == pytest.approx(1 / 12 + 1 / 11)
    assert values[2] == pytest.approx(1 / 11)
    assert core.rrf([[], []]) == ([], {})
    with pytest.raises(ValueError): core.rrf([[1, 1]])
    with pytest.raises(ValueError): core.rrf([[1]], constant=0)


def test_metrics_full_relevance_denominator_missing_hits_and_graded_ndcg():
    result = core.ranking_metrics([1, 0, 2], {1: 1, 2: 2}, k=2)
    assert result["Precision@k"] == .5
    assert result["Recall@k"] == .5
    assert result["MRR"] == 1
    reference = ndcg_score([[0, 1, 3]], [[2, 3, 1]], k=2)
    assert result["nDCG@k"] == pytest.approx(reference)
    assert core.ranking_metrics([2], {2: 2}, k=3)["Precision@k"] == pytest.approx(1 / 3)
    assert core.ranking_metrics([], {2: 2}) == {"Precision@k": 0, "Recall@k": 0, "MRR": 0, "nDCG@k": 0}
    assert core.ranking_metrics([0, 1, 2], {2: 1}, k=1)["MRR"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("ranking,grades,k", [([1, 1], {1: 1}, 3), ([], {}, 3), ([], {1: 0}, 3), ([], {1: 1}, 0)])
def test_invalid_metrics(ranking, grades, k):
    with pytest.raises(ValueError): core.ranking_metrics(ranking, grades, k)


def test_storage_measurements_reconstruct_and_determinism():
    matrix, memory = core.sparse_fixture(n=10, vocabulary=50, unique=5)
    other, _ = core.sparse_fixture(n=10, vocabulary=50, unique=5)
    np.testing.assert_array_equal(matrix.toarray(), other.toarray())
    assert matrix.nnz == 50
    assert memory["dense_bytes"] == matrix.toarray().nbytes
    assert memory["csr_bytes"] == sum(a.nbytes for a in (matrix.data, matrix.indices, matrix.indptr))
    assert np.diff(matrix.indptr).tolist() == [5] * 10
    with pytest.raises(ValueError): core.sparse_fixture(unique=1001)


def test_true_three_term_vectors_and_transform_direction():
    fixture = core.three_vectors()
    assert fixture["counts"].shape == (4, 3)
    np.testing.assert_allclose(fixture["weighted"], fixture["counts"] * fixture["idf"])
    np.testing.assert_allclose(np.linalg.norm(fixture["unit"], axis=1), 1)
    np.testing.assert_allclose(cosine_similarity(fixture["unit"]), cosine_similarity(fixture["weighted"]))


def test_visual_tfidf_validation_and_drift_consistency():
    assert core.validate_numerics()["status"] == "passed"
    _, _, steps = core.drift_fixture()
    assert steps[-1]["alpha_idf"] < steps[0]["alpha_idf"]
    assert all(not entry["novel_in_frozen"] for entry in steps)
    assert len(steps[-1]["scores"]) == steps[-1]["N"]
    assert not np.allclose(steps[-1]["scores"], steps[-1]["frozen_scores"])
    assert steps[0]["ranking"][0] == 0
    assert steps[-1]["ranking"][0] == 1


def test_representation_pipeline_known_counts_df_and_l2():
    terms, matrices, idf = core.representations(["alpha alpha beta", "beta", ""])
    assert terms.tolist() == ["alpha", "beta"]
    np.testing.assert_array_equal(matrices["Counts"], [[2, 1], [0, 1], [0, 0]])
    np.testing.assert_array_equal((matrices["Counts"] > 0).sum(axis=0), [1, 2])
    np.testing.assert_allclose(idf, 1 + np.log(np.array([4 / 2, 4 / 3])))
    np.testing.assert_allclose(matrices["TF-IDF (unnormalized)"], matrices["Counts"] * idf)
    np.testing.assert_allclose(np.linalg.norm(matrices["TF-IDF (L2)"], axis=1), [1, 1, 0])


def test_evaluation_includes_failed_synonym_query_without_label_changes():
    result = core.evaluation()
    assert len(result["judgments"]) == 8
    for method in (*core.METHODS, "rrf"):
        assert len(result["per_query"][method]) == 8
        assert result["per_query"][method][5]["Recall@k"] == 0
        assert all(0 <= value <= 1 for value in result["summary"][method].values())
