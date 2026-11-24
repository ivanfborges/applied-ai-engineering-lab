"""Numerical and application invariants for the independently executed visual phase."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nlp_lab import core as c


def test_bpe_training_and_encoding_are_deterministic_and_lossless_for_seen_alphabet():
    a = c.EducationalBPE.train(c.SENTENCES, 40)
    b = c.EducationalBPE.train(c.SENTENCES, 40)
    assert a == b and a.merges
    for text in c.SENTENCES:
        pieces = a.encode(text)
        assert "".join(p["token"] for p in pieces) == text
        assert [p["start"] for p in pieces] == [0] + [p["end"] for p in pieces[:-1]]
        assert pieces[-1]["end"] == len(text)
        assert len(pieces) <= len(text)
    assert a.encode("🦄")[0]["token"] == "<unk>"
    assert a.encode("") == []


@pytest.mark.parametrize("strategy", c.STRATEGIES)
def test_actual_token_boundaries_and_empty_sentences(strategy):
    for sentence in c.SENTENCES:
        pieces = c.segment(sentence, strategy)
        assert pieces and all(p["start"] < p["end"] for p in pieces)
        if strategy != "byte":
            for p in pieces:
                assert p["token"] == sentence[p["start"]:p["end"]]
        else:
            recovered = bytes(int(p["token"], 16) for p in pieces).decode("utf-8")
            assert recovered == sentence
    assert c.segment("", strategy) == []


def test_linguistic_tokenizer_is_not_whitespace_and_bytes_are_not_characters():
    pieces = c.segment("I don't approve.", "linguistic")
    assert "n't" in [p["token"] for p in pieces]
    assert len(c.segment("é", "byte")) == 2
    assert len(c.segment("é", "character")) == 1
    with pytest.raises(ValueError): c.segment("text", "invented")


def test_unicode_forms_lengths_and_information_loss():
    assert len(c.normalization_step("é", "NFC")) == 1
    assert len(c.normalization_step("é", "NFD")) == 2
    assert c.codepoints("é")[0]["codepoint"] == "U+00E9"
    assert c.normalization_step("ﬁ①", "NFKC") == "fi1"
    assert c.normalization_step("US us Apple apple", "casefold") == "us us apple apple"
    assert c.normalization_step("avó avô", "accents") == "avo avo"
    assert c.normalization_step("not approved", "stopwords") == "approved"
    trace = c.normalization_trace("  Café ", ["NFD", "accents", "casefold", "whitespace"])
    assert trace[-1]["text"] == "cafe"
    with pytest.raises(ValueError): c.normalization_step("text", "unknown")


def test_real_morphology_uses_resources_and_supplied_pos():
    rows = c.morphology_rows()
    verbs = {r["word"]: r for r in rows if r["pos"] == "v"}
    assert verbs["running"]["porter"] == c.nltk_tools()[0].stem("running")
    assert verbs["was"]["lemma"] == "be"
    assert "see" in c.wordnet_candidates("saw", "v")
    assert c.wordnet_lemma("saw", "v") in c.wordnet_candidates("saw", "v")
    assert c.wordnet_lemma("saw", "n") == "saw"
    assert c.wordnet_lemma("leaves", "n") == "leaf"
    assert c.wordnet_lemma("leaves", "v") == "leave"
    pt = [r for r in rows if "portuguese" in r]
    assert all(r["portuguese"] == c.nltk_tools()[2].stem(r["word"]) for r in pt)
    with pytest.raises(ValueError): c.wordnet_lemma("saw", "bad-pos")


def test_missing_wordnet_reports_explicit_setup_and_does_not_break_counting(monkeypatch):
    def unavailable(): raise c.ResourceUnavailable("explicit setup required")
    monkeypatch.setattr(c, "wordnet_resource", unavailable)
    with pytest.raises(c.ResourceUnavailable): c.wordnet_lemma("mice", "n")
    assert len(c.morphology_rows(False)) > 0
    assert c.feature_stats(c.synthetic_corpus())[0]["features"] > 0
    assert c.retrieval_run()["results"]


@pytest.mark.parametrize("n", [1, 2, 3])
def test_vocabulary_density_memory_and_fixed_order_bound(n):
    stats, matrix, features = c.feature_stats(c.synthetic_corpus(), n)
    assert stats["nnz"] == np.count_nonzero(matrix.toarray())
    assert stats["features"] == len(features)
    assert stats["density"] + stats["sparsity"] == pytest.approx(1)
    assert stats["sparse_bytes"] == matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    assert stats["dense_bytes"] == matrix.toarray().nbytes
    base = len({t for doc in c.synthetic_corpus() for t in c.tokenize(doc)})
    fixed = {g for doc in c.synthetic_corpus() for g in c.generate_ngrams(c.tokenize(doc), n)}
    assert len(fixed) <= base ** n


def test_feature_pruning_changes_computation_and_preserves_min_df():
    corpus = c.synthetic_corpus()
    full, _, _ = c.feature_stats(corpus, 3)
    reduced, matrix, _ = c.feature_stats(corpus, 3, 4, 30)
    assert reduced["features"] <= 30 < full["features"]
    assert np.asarray((matrix > 0).sum(axis=0)).min() >= 4


@pytest.mark.parametrize("l2", [False, True])
def test_scratch_tfidf_matches_sklearn_including_zero_row(l2):
    counts = np.array([[1, 2, 0], [1, 0, 0], [1, 0, 4], [0, 0, 0]])
    df, idf, weighted = c.tfidf_from_counts(counts, l2)
    reference = TfidfTransformer(norm="l2" if l2 else None).fit(counts)
    np.testing.assert_allclose(idf, reference.idf_, rtol=1e-14)
    np.testing.assert_allclose(weighted, reference.transform(counts).toarray(), rtol=1e-14)
    np.testing.assert_array_equal(df, [3, 1, 1])
    assert np.all(weighted[-1] == 0)


@pytest.mark.parametrize("df,tf", [(0, 1), (101, 1), (1.5, 1), (1, -1), (np.nan, 1), (1, np.inf)])
def test_invalid_surface_counts_rejected(df, tf):
    with pytest.raises(ValueError): c.tfidf_weight(100, df, tf)


@pytest.mark.parametrize("a,b,expected", [([1, 0], [3, 0], 1), ([1, 0], [0, 1], 0),
                                        ([1, 0], [-1, 0], -1), ([0, 0], [1, 0], None)])
def test_cosine_geometry_and_library_convention(a, b, expected):
    assert c.cosine(np.array(a), np.array(b)) == expected
    if expected is not None:
        assert c.cosine(np.array(a), np.array(b)) == pytest.approx(cosine_similarity([a], [b])[0, 0])


def test_ranking_uses_original_space_and_zero_evidence_is_excluded():
    result = c.vector_space("word12", c.SPACE_QUERIES[0])
    assert result["ranking"] == c.rank(result["scores"])
    vectorizer = c.make_vectorizer("word12")
    matrix = vectorizer.fit_transform([text for _, text in c.TOPIC_DOCS])
    scores = cosine_similarity(vectorizer.transform([c.SPACE_QUERIES[0]]), matrix).ravel()
    np.testing.assert_allclose(result["scores"], scores)
    assert c.rank(np.array([0., 1., 1., 0.])) == [1, 2]
    assert c.vector_space("word", "zzzzunseen")["ranking"] == []


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("alpha", [0., .1, 1.])
def test_conditional_distributions_are_normalized_and_boundaries_explicit(order, alpha):
    model = c.NGramLM([c.tokenize(s) for s in c.LM_SENTENCES], order)
    assert model.BOS not in model.vocabulary and model.EOS in model.vocabulary
    for context in model.counts:
        p = model.distribution(context, alpha)
        assert p.sum() == pytest.approx(1, abs=1e-14)
        assert (p >= 0).all()
    assert model.generate(63, alpha) == model.generate(63, alpha)


def test_unsmoothed_unseen_event_zero_but_unseen_context_undefined():
    model = c.NGramLM([["a", "b"], ["a", "c"]], 2)
    assert model.distribution(("a",), 0)[model.vocabulary.index("a")] == 0
    assert model.distribution(("a",), .1)[model.vocabulary.index("a")] > 0
    with pytest.raises(ValueError, match="unseen"): model.distribution(("unseen",), 0)
    np.testing.assert_allclose(model.distribution(("unseen",), .1), np.full(4, .25))
    with pytest.raises(ValueError): model.distribution((), .1)
    with pytest.raises(ValueError): model.distribution(("a",), -.1)


@pytest.mark.parametrize("order", [0, -1, True, 1.0, 1.5, "2"])
def test_invalid_model_orders_are_rejected(order):
    with pytest.raises(ValueError): c.NGramLM([["a"]], order)


def test_no_cross_sentence_counts_and_unigram_hand_count():
    bigram = c.NGramLM([["a"], ["b"]], 2)
    assert bigram.counts[("a",)] == {bigram.EOS: 1}
    assert "b" not in bigram.counts[("a",)]
    unigram = c.NGramLM([["a"], ["b"]], 1)
    p = unigram.distribution((), 0)
    assert p[unigram.vocabulary.index(unigram.EOS)] == .5


def test_metrics_hand_cases_and_library_ndcg():
    m = c.retrieval_metrics([8, 3, 2, 1], (3, 1), 3)
    assert m["recall"] == .5 and m["rr"] == .5
    expected = (1 / np.log2(3)) / (1 + 1 / np.log2(3))
    assert m["ndcg"] == pytest.approx(expected)
    assert m["ndcg"] == pytest.approx(ndcg_score([[0, 1, 0, 1]], [[4, 3, 2, 1]], k=3))
    assert c.retrieval_metrics([], (1,), 3) == {"recall": 0, "rr": 0, "ndcg": 0}
    with pytest.raises(ValueError): c.retrieval_metrics([1, 1], (1,), 3)
    with pytest.raises(ValueError): c.retrieval_metrics([], (), 3)


def test_negation_ablation_is_real_and_changes_results():
    assert c.analyze("license approved", "drop") == c.analyze("license not approved", "drop")
    assert c.analyze("license approved", "preserve") != c.analyze("license not approved", "preserve")
    original, dropped = c.retrieval_run(), c.retrieval_run("drop")
    assert original["mean_metrics"] != dropped["mean_metrics"]
    for _, stop, morph, n in c.CONFIGS:
        result = c.retrieval_run(stop, morph, n)
        assert len(result["results"]) == len(c.CONTRAST_QUERIES)
        assert all(0 <= value <= 1 for value in result["mean_metrics"].values())


def test_corruption_is_reproducible_zero_rate_identity_and_seed_sensitive():
    text = "A fiscalização ambiental identifies environmental violations."
    assert c.corrupt_text(text, 0, 63) == text
    assert c.corrupt_text(text, .5, 63) == c.corrupt_text(text, .5, 63)
    assert c.corrupt_text(text, .5, 63) != c.corrupt_text(text, .5, 64)
    assert c.corrupt_text("", .5) == ""
    with pytest.raises(ValueError): c.corrupt_text(text, 1.1)


def test_character_and_word_representations_share_canonical_preprocessing():
    for rep in c.REPRESENTATIONS:
        v = c.make_vectorizer(rep)
        matrix = v.fit_transform(["CAFÉ licença", "unrelated text"])
        a = v.transform(["CAFÉ licença"])
        b = v.transform(["cafe\u0301 licença"])
        np.testing.assert_allclose(a.toarray(), b.toarray())


def test_ocr_trials_are_paired_and_deterministic():
    a = c.ocr_experiment(63, 2, (0, .3))
    assert a == c.ocr_experiment(63, 2, (0, .3))
    assert len(a["rows"]) == 4 * 2 * 2
    for rate in (0, .3):
        queries = [[r["query"] for r in a["examples"] if r["rate"] == rate and r["representation"] == rep] for rep in c.REPRESENTATIONS]
        assert queries.count(queries[0]) == 4
