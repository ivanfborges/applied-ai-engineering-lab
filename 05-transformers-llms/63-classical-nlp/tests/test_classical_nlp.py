"""Analyzer boundaries, count parity, fixed vocabularies and retrieval controls."""

from pathlib import Path
import sys
import unicodedata

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import (
    document_term_matrix, generate_ngrams, lemma_lookup_demo,
    normalize, suffix_stem_demo, tokenize,
)
from example import DOCUMENTS, QUERY, lexical_diagnostics, negation_collision


@pytest.mark.parametrize("text", ["INFORMAÇÃO", "Straße", "DON’T\tstop", "ǰ", "", " a  b \n"])
def test_normalization_is_idempotent_and_nfc(text):
    result = normalize(text)
    assert normalize(result) == result
    assert unicodedata.is_normalized("NFC", result)


def test_canonical_equivalence_accents_and_casefold():
    assert normalize("INFORMAÇÃO") == normalize("INFORMAC\u0327A\u0303O") == "informação"
    assert normalize("Straße") == "strasse"
    assert tokenize("avó avô") == ["avó", "avô"]


def test_tokenization_retains_negation_contractions_and_single_characters():
    assert tokenize("I DON’T approve; not without licença!") == [
        "i", "don't", "approve", "not", "without", "licença"]
    assert tokenize("'quoted' foo_bar 47.838/2020") == ["quoted", "foo", "bar", "47", "838", "2020"]
    assert tokenize("...🙂") == []


@pytest.mark.parametrize("bad", [None, 12, b"text", ["text"]])
def test_non_string_text_fails(bad):
    with pytest.raises(TypeError, match="string"):
        normalize(bad)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
def test_ngram_occurrence_formula_and_repetition(n):
    tokens = ["a", "a", "a"]
    grams = generate_ngrams(tokens, n)
    assert len(grams) == max(0, len(tokens) - n + 1)
    assert grams == [("a",) * n] * max(0, 4 - n)
    assert generate_ngrams([], n) == []


@pytest.mark.parametrize("n,error", [(0, ValueError), (-1, ValueError), (1.5, TypeError),
                                    (True, TypeError), ("2", TypeError)])
def test_invalid_orders_fail_even_for_empty_inputs(n, error):
    with pytest.raises(error):
        generate_ngrams([], n)
    with pytest.raises(error):
        document_term_matrix([], n)


@pytest.mark.parametrize("tokens,error", [("abc", TypeError), ([""], ValueError),
                                         ([1], ValueError), (["two words"], ValueError)])
def test_invalid_token_sequences(tokens, error):
    with pytest.raises(error):
        generate_ngrams(tokens, 1)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_counts_equal_sklearn_with_the_same_analyzer(n):
    docs = ["A a! café don't stop.", "Don't stop; not now.", "", "A café."]
    vocabulary, counts = document_term_matrix(docs, n)
    reference = CountVectorizer(tokenizer=tokenize, token_pattern=None,
                                lowercase=False, ngram_range=(n, n))
    expected = reference.fit_transform(docs)
    assert [" ".join(g) for g in vocabulary] == reference.get_feature_names_out().tolist()
    np.testing.assert_array_equal(counts, expected.toarray())


def test_known_count_matrix_and_row_order():
    vocabulary, matrix = document_term_matrix(["b a a", "b", ""])
    assert vocabulary == [("a",), ("b",)]
    np.testing.assert_array_equal(matrix, [[2, 1], [0, 1], [0, 0]])
    assert matrix.dtype == np.int64


def test_no_cross_document_grams_and_empty_shapes():
    features, matrix = document_term_matrix(["one", "two"], 2)
    assert features == [] and matrix.shape == (2, 0)
    assert document_term_matrix([])[1].shape == (0, 0)
    assert document_term_matrix(["", "!!!"])[1].shape == (2, 0)


def test_frozen_vocabulary_keeps_supplied_order_and_ignores_oov():
    vocabulary = [("b",), ("a",)]
    original = vocabulary.copy()
    features, matrix = document_term_matrix(["a unknown unknown b a"], vocabulary=vocabulary)
    assert features == original == vocabulary
    np.testing.assert_array_equal(matrix, [[1, 2]])
    assert document_term_matrix([], vocabulary=vocabulary)[1].shape == (0, 2)


@pytest.mark.parametrize("vocabulary", [[("a",), ("a",)], [("A",)], [("",)],
                                        [("two words",)], [("a", "b")], ["a"]])
def test_invalid_vocabulary_fails(vocabulary):
    with pytest.raises(ValueError):
        document_term_matrix([], vocabulary=vocabulary)


@pytest.mark.parametrize("docs", ["single document", None, iter(["a"])])
def test_invalid_corpus_container_fails(docs):
    with pytest.raises(TypeError):
        document_term_matrix(docs)


def test_morphology_demo_exposes_bad_suffix_rules_and_supplied_pos():
    assert suffix_stem_demo("studies") == "studi"
    assert suffix_stem_demo("running") == "runn"
    assert suffix_stem_demo("news") == "new"
    assert lemma_lookup_demo("running", "v") == "run"
    assert lemma_lookup_demo("saw", "v") == "see"
    assert lemma_lookup_demo("saw", "n") == "saw"
    with pytest.raises(ValueError, match="fixture"):
        lemma_lookup_demo("unknown", "n")
    with pytest.raises(ValueError, match="POS"):
        lemma_lookup_demo("saw", "verb")
    with pytest.raises(ValueError, match="one"):
        suffix_stem_demo("two words")


def test_retrieval_matches_independent_library_construction():
    result = lexical_diagnostics(DOCUMENTS, QUERY, 2)
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                lowercase=False, ngram_range=(1, 2))
    corpus = vectorizer.fit_transform(DOCUMENTS)
    query = vectorizer.transform([QUERY])
    scores = (query @ corpus.T).toarray().ravel()
    for item in result["ranking"]:
        assert item["score"] == pytest.approx(scores[item["document_id"]], abs=1e-12)
    assert result["shape"] == list(corpus.shape)
    assert result["nnz"] == corpus.nnz


def test_query_does_not_fit_vocabulary_and_zero_match_is_explicit():
    first = lexical_diagnostics(DOCUMENTS, QUERY, 2)
    oov = lexical_diagnostics(DOCUMENTS, "unseentokenxyz", 2)
    assert first["features"] == oov["features"]
    assert "unseentokenxyz" not in oov["features"]
    assert oov["no_matching_features"] and oov["query_nnz"] == 0
    assert oov["ranking"] == []
    assert lexical_diagnostics(DOCUMENTS, "", 2)["ranking"] == []


def test_ties_preserve_input_order_and_ngrams_retain_local_negation():
    result = lexical_diagnostics(["same text", "same text"], "same", 1)
    assert [item["document_id"] for item in result["ranking"]] == [0, 1]
    result = lexical_diagnostics(DOCUMENTS, QUERY, 2)
    assert "not approved" in result["features"]
    assert negation_collision() == {"tokens_distinct_before": True,
                                    "tokens_equal_after_dropping_not": True}


@pytest.mark.parametrize("documents,query,n,error", [
    ([], "a", 1, ValueError), ([""], "a", 1, ValueError),
    (["a", None], "a", 1, ValueError), ("a", "a", 1, TypeError),
    (["a"], None, 1, TypeError), (["a"], "a", 0, ValueError),
    (["a"], "a", True, TypeError),
])
def test_invalid_retrieval_inputs(documents, query, n, error):
    with pytest.raises(error):
        lexical_diagnostics(documents, query, n)
