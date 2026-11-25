"""Independent mathematical and tokenizer integration checks for the visual phase."""

import itertools
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual_core import (BPE_TEXT, UNIGRAM_PROBS, RAG_DOCUMENT, ROOT, SEED,
    bpe_states, inference_states, wordpiece_trace, association_scores,
    unigram_viterbi, exhaustive_paths, synthetic_embeddings, prepare_tokenizers,
    compare_text, vocabulary_experiment, attention_counts, corrupt,
    chunk_document, chunk_metrics, prune_context, pipeline_state)


@pytest.fixture(scope="module")
def local(tmp_path_factory):
    return prepare_tokenizers(tmp_path_factory.mktemp("models"))


def test_bpe_trace_uses_weighted_counts_and_fixed_inference_priorities():
    states = bpe_states()
    assert states[0]["weighted_tokens"] == 79
    assert states[0]["selected"] == ["e", "s"]
    assert states[0]["pairs"][0] == (["e", "s"], 9)
    assert states[-1]["weighted_tokens"] == 20
    assert states[-1]["compression"] == 79 / 20
    assert [s["vocabulary_size"] for s in states] == list(range(10, 21))
    for word in ["lowest", "newer", "slower", "newest"]:
        trace = inference_states(word)
        assert "".join(trace[-1]["pieces"]) == word
        for state in trace[:-1]:
            assert state["selected"] == state["candidates"][0]["pair"]
            assert [c["rank"] for c in state["candidates"]] == sorted(c["rank"] for c in state["candidates"])
    assert "unsupported" in inference_states("café")[0]["error"]


def test_wordpiece_greedy_continuation_whole_word_unknown_and_no_backtracking():
    vocab = {"un", "##afford", "##able", "[UNK]"}
    assert wordpiece_trace("unaffordable", vocab)["pieces"] == ["un", "##afford", "##able"]
    # a + ##bx would work, but greedy ab then missing ##x yields whole-word UNK.
    assert wordpiece_trace("abx", {"a", "ab", "##bx", "[UNK]"})["pieces"] == ["[UNK]"]
    assert wordpiece_trace("afford", vocab)["pieces"] == ["[UNK]"]


def test_relative_association_can_reverse_frequency_preference():
    scores = association_scores()
    assert scores["bpe_winner"] == 0 and scores["association_winner"] == 1
    assert scores["association"] == [100/(1000*500), 30/(40*50)]
    changed = association_scores(left=(1000, 10000))
    assert changed["association_winner"] == 0


@pytest.mark.parametrize("arguments", [
    {"joint": (0, 1)}, {"joint": (1001, 30)}, {"left": (-1, 40)},
    {"right": (math.inf, 50)}, {"left": (1,)},
])
def test_invalid_frequency_tables_are_explicit(arguments):
    with pytest.raises(ValueError):
        association_scores(**arguments)


def test_viterbi_matches_exhaustive_oracle_and_probability_product():
    for probabilities in [UNIGRAM_PROBS, {**UNIGRAM_PROBS, "un": .35, "unbeliev": .10}]:
        result = unigram_viterbi("unbelievable", probabilities)
        paths = exhaustive_paths("unbelievable", probabilities)
        assert len(paths) == 3
        best_pieces, best_probability = max(paths, key=lambda row: row[1])
        assert result["pieces"] == best_pieces
        assert result["probability"] == pytest.approx(best_probability)
        assert result["log_probability"] == pytest.approx(sum(math.log(probabilities[p]) for p in best_pieces))
    assert unigram_viterbi("unbelievable", UNIGRAM_PROBS)["pieces"] == ["unbeliev", "able"]


@pytest.mark.parametrize("text,probs", [
    ("a", {"a": .4}), ("a", {"a": 0, "b": 1}), ("a", {"a": math.nan}),
    ("a", {"": 1}), ("c", {"a": 1}), ("", {"a": 1}),
])
def test_invalid_probabilities_or_unreachable_text(text, probs):
    with pytest.raises(ValueError):
        unigram_viterbi(text, probs)


@pytest.mark.integration
def test_actual_hf_wordpiece_matches_greedy_lookup_for_single_normalized_words(local):
    tokenizer = local["WordPiece local"]
    for word in ["unbelievable", "tokenization", "lower", "intelligence", "xyz"]:
        assert wordpiece_trace(word, tokenizer.get_vocab())["pieces"] == tokenizer.encode(word).tokens


@pytest.mark.integration
def test_sentencepiece_reports_actual_whitespace_normalization_and_unknowns(local):
    for name in ["SentencePiece BPE", "SentencePiece Unigram"]:
        row = compare_text(name, "  Café   ＡＩ e\u0301.  ", local)
        assert "▁" in row["normalized"]
        assert row["decoded"] == "Café AI é."
        assert not row["exact_reconstruction"]
        assert row["tokens"] == len(row["ids"]) == len(row["pieces"])
        assert compare_text(name, "🧠", local)["unknown"] > 0
    assert local["SentencePiece BPE"].vocab_size() <= 256
    assert local["SentencePiece Unigram"].vocab_size() <= 256


@pytest.mark.integration
def test_repeated_local_training_reproduces_piece_outputs(local, tmp_path):
    repeated = prepare_tokenizers(tmp_path / "repeat")
    for name in local:
        first = compare_text(name, "Machine learning. Café", local)
        second = compare_text(name, "Machine learning. Café", repeated)
        assert first["pieces"] == second["pieces"]
        assert first["ids"] == second["ids"]


def test_embedding_lookup_is_id_stable_without_full_table_allocation():
    rows = synthetic_embeddings([17, 200, 17])
    assert rows.shape == (3, 6)
    np.testing.assert_array_equal(rows[0], rows[2])
    np.testing.assert_array_equal(rows[1], synthetic_embeddings([200])[0])


@pytest.mark.integration
def test_pipeline_uses_actual_preserving_regex_boundaries_and_real_ids():
    import tiktoken
    text = "Inteligência artificial generativa."
    state = pipeline_state(text)
    assert "".join(state["pretokenized"]) == text
    assert state["ids"] == tiktoken.get_encoding("cl100k_base").encode_ordinary(text)
    assert bytes.fromhex("".join(state["pieces"])).decode() == text
    assert np.asarray(state["selected_rows"]).shape == (len(state["ids"]), 6)


def test_train_and_held_out_coverage_are_separate_and_rejections_explicit():
    rows = vocabulary_experiment()
    train, held = [r for r in rows if r["partition"] == "train"], [r for r in rows if r["partition"] == "held_out"]
    assert all(r["word_coverage"] == 1 for r in train)
    assert all(r["rejected"] == 1 and r["word_coverage"] == .8 for r in held)
    assert train[0]["tokens"] == 79
    assert train[0]["compression_supported_characters_per_token"] == 1


@pytest.mark.parametrize("n", [1, 2, 128, 4096])
def test_attention_counts_match_small_enumeration_and_formulas(n):
    row = attention_counts(n, 64)
    assert row["dense"] == n*n and row["dense_proxy"] == n*n*64
    assert row["causal_valid"] == n*(n+1)//2
    assert row["one_cached_decode_query"] == n
    if n <= 2:
        assert row["causal_valid"] == sum(k <= q for q, k in itertools.product(range(n), repeat=2))


@pytest.mark.parametrize("mechanism", ["substitution", "insert_space", "break_word", "remove_space", "deletion"])
def test_ocr_is_reproducible_and_zero_noise_is_identity(mechanism):
    text = "Article 125 retains the legal information."
    assert corrupt(text, 0, mechanism) == text
    assert corrupt(text, .3, mechanism, SEED) == corrupt(text, .3, mechanism, SEED)
    assert corrupt(text, 1, "deletion") == ""


@pytest.mark.parametrize("encoding,strategy", list(itertools.product(["cl100k_base", "o200k_base"], ["tokens", "sentences"])))
def test_token_chunking_preserves_all_unicode_positions_and_enforces_limit(encoding, strategy):
    import tiktoken
    text = "Café 🚀 中文. " + RAG_DOCUMENT
    chunks = chunk_document(text, encoding, strategy, 24, 4)
    enc = tiktoken.get_encoding(encoding)
    assert set(i for c in chunks for i in range(c["start"], c["end"])) == set(range(len(text)))
    for c in chunks:
        assert c["text"] == text[c["start"]:c["end"]]
        assert len(enc.encode_ordinary(c["text"])) == c["tokens"] <= 24
        c["text"].encode().decode("utf-8", errors="strict")
    for before, after in zip(chunks, chunks[1:]):
        assert after["start"] > before["start"]
        assert len(enc.encode_ordinary(text[after["start"]:before["end"]])) <= 4


def test_character_chunks_expose_token_violations_and_containment_is_not_retrieval():
    chunks = chunk_document("🚀" * 20, strategy="characters", size=8, overlap=2)
    metrics = chunk_metrics(chunks, embedding_limit=8)
    assert metrics["token_limit_violations"] > 0
    assert metrics["duplicated_code_points"] > 0
    assert metrics["truncated_code_points"] == 0
    assert not metrics["retrieval_evaluated"]
    assert chunk_metrics(chunk_document(RAG_DOCUMENT, size=40, overlap=5))["evidence_contained"]


@pytest.mark.parametrize("kwargs", [{"size": 0}, {"overlap": -1}, {"size": 4, "overlap": 4}, {"strategy": "unknown"}, {"size": True}])
def test_invalid_chunk_configuration(kwargs):
    with pytest.raises(ValueError):
        chunk_document("hello", **kwargs)


def test_context_pruning_removes_lower_priorities_and_reports_irreducible_overflow():
    result = prune_context({"system": 3}, [3, 4, 5], 2, 12)
    assert result["retained"] == [3, 4] and result["removed"] == 1
    assert result["budget"]["fits"] and result["budget"]["remaining"] == 0
    impossible = prune_context({"system": 20}, [3, 4], 2, 12)
    assert impossible["retained"] == [] and not impossible["budget"]["fits"]
