"""Merge, Unicode and request-budget contracts, including real tokenizer probes."""

from decimal import Decimal
from pathlib import Path
import sys

import pytest
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import SimpleBPE, merge_pair
from example import (ENCODINGS, SAMPLES, context_budget, estimated_cost,
                     inspect_text, run_experiment, vocabulary_budget)


def test_pair_counts_are_weighted_and_ties_are_lexical():
    model = SimpleBPE(1).fit("ab ab ab cd cd")
    assert model.history == [(("a", "b"), 3)]
    assert SimpleBPE(1).fit("cd ab").history == [(("a", "b"), 1)]


def test_non_overlapping_merges_and_priority_not_longest_match():
    assert merge_pair(tuple("aaaaa"), ("a", "a")) == ("aa", "aa", "a")
    # A greedy vocabulary match chooses ab at the start; BPE first merges bc.
    model = SimpleBPE(2).fit("bc bc bc ab ab")
    assert model.history == [(("b", "c"), 3), (("a", "b"), 2)]
    assert model.encode_word("abc") == ["a", "bc"]


def test_repeated_training_refit_and_unseen_word_reconstruction():
    text = "low low lower newest widest"
    first = SimpleBPE(10).fit(text)
    second = SimpleBPE(10).fit(text)
    assert first.history == second.history
    history = first.history.copy()
    for word in ["", "lowest", "newer", "widest"]:
        assert "".join(first.encode_word(word)) == word
    assert first.history == history
    first.fit("xy xy")
    assert first.alphabet == {"x", "y"}
    assert first.history == [(("x", "y"), 2)]
    with pytest.raises(ValueError, match="unseen"):
        first.encode_word("low")


def test_zero_merges_and_early_stop():
    assert SimpleBPE(0).fit("abc").encode_word("cab") == list("cab")
    assert SimpleBPE(20).fit("a a b").history == []


@pytest.mark.parametrize("merges", [-1, True, 1.5])
def test_invalid_merge_counts(merges):
    with pytest.raises(ValueError):
        SimpleBPE(merges)


@pytest.mark.parametrize("text", ["", " \n\t", 5])
def test_empty_training_is_explicit(text):
    with pytest.raises(ValueError):
        SimpleBPE().fit(text)


def test_invalid_word_and_unfitted_model_are_explicit():
    with pytest.raises(ValueError, match="fit"):
        SimpleBPE().encode_word("a")
    model = SimpleBPE().fit("abc")
    for word in ("a b", "a\tb", "z", None):
        with pytest.raises(ValueError):
            model.encode_word(word)


def test_shared_context_and_separate_limits_at_boundary():
    assert context_budget({"prompt": 8}, 2, 10) == {
        "input_tokens": 8, "output_reserve": 2, "remaining": 0, "fits": True}
    assert not context_budget({"prompt": 9}, 2, 10)["fits"]
    assert not context_budget({"prompt": 8}, 2, 10, max_input=7)["fits"]
    assert not context_budget({"prompt": 8}, 2, 10, max_output=1)["fits"]
    assert context_budget({}, 0, 10)["remaining"] == 10


@pytest.mark.parametrize("kwargs", [
    {"components": {"prompt": -1}}, {"components": {"prompt": True}},
    {"components": {"prompt": 1.5}}, {"components": {"": 1}},
    {"components": []}, {"context_limit": 0}, {"output_reserve": -1},
    {"max_input": False}, {"max_output": -1},
])
def test_invalid_context_inputs(kwargs):
    arguments = {"components": {"prompt": 1}, "output_reserve": 1, "context_limit": 10}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        context_budget(**arguments)


def test_cost_and_weight_units_and_tying():
    assert estimated_cost(6000, 1000, 2, 8) == Decimal("0.020")
    assert estimated_cost(1, 1, 0.1, Decimal("0.2")) == Decimal("0.0000003")
    assert estimated_cost(0, 0, 0, 0) == 0
    tied = vocabulary_budget(50000, 4096)
    assert tied == {"parameters": 204800000, "fp16_weight_bytes": 409600000}
    assert vocabulary_budget(50000, 4096, tied=False)["parameters"] == 2 * tied["parameters"]


@pytest.mark.parametrize("price", [-1, float("inf"), float("nan"), True, "2"])
def test_invalid_price(price):
    with pytest.raises(ValueError):
        estimated_cost(1, 1, price, 0)


def test_invalid_token_and_vocabulary_counts():
    for count in [-1, True, 1.5]:
        with pytest.raises(ValueError):
            estimated_cost(count, 1, 2, 8)
        with pytest.raises(ValueError):
            vocabulary_budget(count, 4096)
    with pytest.raises(ValueError):
        vocabulary_budget(10, 0)
    with pytest.raises(ValueError):
        vocabulary_budget(10, 2, tied=1)


@pytest.fixture(scope="module", params=ENCODINGS)
def encoder(request):
    return tiktoken.get_encoding(request.param)


@pytest.mark.integration
def test_real_round_trip_preserves_unicode_whitespace_and_literal_special_text(encoder):
    for text in [*SAMPLES.values(), "", " \t\n", "é", "e\u0301", "🚀", "<|endoftext|>"]:
        result = inspect_text(encoder, text)
        assert result["tokens"] == len(result["ids"])
        assert b"".join(bytes.fromhex(piece) for piece in result["token_bytes_hex"]) == text.encode("utf-8")
        assert encoder.decode(result["ids"], errors="strict") == text
    special_id = encoder.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
    assert len(special_id) == 1
    assert inspect_text(encoder, "<|endoftext|>")["ids"] != special_id
    with pytest.raises(ValueError, match="UTF-8"):
        inspect_text(encoder, "\ud800")


@pytest.mark.integration
def test_token_boundary_can_split_utf8_character(encoder):
    result = inspect_text(encoder, "🚀")
    assert result["tokens"] > 1
    with pytest.raises(UnicodeDecodeError):
        bytes.fromhex(result["token_bytes_hex"][0]).decode("utf-8", errors="strict")


@pytest.mark.integration
def test_report_uses_actual_encoding_counts_and_marks_hypotheses():
    report = run_experiment()
    assert report["interpretation_status"] == "author review required"
    assert report["hypothetical_budget"]["remaining"] == 8468
    assert Decimal(report["hypothetical_cost_usd"]) == Decimal("0.02")
    for name, data in report["encodings"].items():
        assert data["id_capacity"] >= data["mergeable_tokens"] + data["special_tokens"]
        for label, text in SAMPLES.items():
            assert data["samples"][label]["ids"] == tiktoken.get_encoding(name).encode_ordinary(text)
