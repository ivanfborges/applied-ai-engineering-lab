"""Numerical contracts and generation boundaries on synthetic fixtures."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "day75_decoding", Path(__file__).resolve().parents[1] / "from_scratch.py"
)
decoder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = decoder
SPEC.loader.exec_module(decoder)
distribution = decoder.decoding_distribution


def test_temperature_identity_shift_and_hand_calculation():
    probabilities = np.array([0.6, 0.3, 0.1])
    logits = np.log(probabilities)
    np.testing.assert_allclose(distribution(logits), probabilities)
    np.testing.assert_allclose(distribution(logits + 1000), probabilities)
    np.testing.assert_allclose(distribution(logits, temperature=0.5),
                               probabilities ** 2 / np.sum(probabilities ** 2))


@pytest.mark.parametrize("temperature", [0, 1e-300, 0.1, 1, 1e300])
def test_extreme_finite_logits_stay_normalized(temperature):
    with np.errstate(all="raise"):
        result = distribution([1e308, -1e308, 0], temperature=temperature, top_p=0.9)
    assert np.isfinite(result).all()
    assert np.all(result >= 0)
    assert result.sum() == pytest.approx(1)


def test_ties_are_stable_and_greedy_limit_has_distinct_semantics():
    np.testing.assert_array_equal(distribution([2, 2, 1], temperature=0), [1, 0, 0])
    np.testing.assert_allclose(distribution([2, 2, 1], temperature=1e-10), [0.5, 0.5, 0])
    np.testing.assert_array_equal(distribution([1, 1, 1], top_k=1), [1, 0, 0])
    np.testing.assert_allclose(distribution([1, 1, 1], top_p=0.5), [0.5, 0.5, 0])


def test_top_k_ranks_logits_even_when_softmax_underflows():
    np.testing.assert_array_equal(distribution([-10000, 0, -1000], top_k=2), [0, 1, 0])


def test_top_p_includes_threshold_crossing_token_and_exact_boundary():
    np.testing.assert_allclose(distribution(np.log([0.85, 0.07, 0.04, 0.03, 0.01]), top_p=0.9),
                               [0.85 / 0.92, 0.07 / 0.92, 0, 0, 0])
    np.testing.assert_allclose(distribution([0, 0, 0, 0], top_p=0.5), [0.5, 0.5, 0, 0])
    np.testing.assert_allclose(distribution([0, 0, 0, 0], top_p=1), [0.25] * 4)
    np.testing.assert_array_equal(distribution([0, 0], top_p=1e-300), [1, 0])


def test_top_p_uses_mass_after_top_k_renormalization():
    logits = np.log([0.5, 0.3, 0.2])
    np.testing.assert_array_equal(distribution(logits, top_k=2, top_p=0.6), [1, 0, 0])
    assert np.count_nonzero(distribution(logits, top_p=0.6)) == 2


def test_top_p_candidate_count_changes_with_temperature():
    logits = np.log([0.85, 0.07, 0.04, 0.03, 0.01])
    assert np.count_nonzero(distribution(logits, temperature=0.2, top_p=0.9)) == 1
    assert np.count_nonzero(distribution(logits, temperature=2, top_p=0.9)) == 4


@pytest.mark.parametrize("logits", [[], [[1, 2]], [np.nan], [np.inf], [-np.inf]])
def test_invalid_logits(logits):
    with pytest.raises(ValueError):
        distribution(logits)


@pytest.mark.parametrize("options", [
    {"temperature": -1}, {"temperature": np.nan}, {"temperature": np.inf},
    {"temperature": True}, {"temperature": "1"}, {"top_k": 0}, {"top_k": 4},
    {"top_k": 1.5}, {"top_k": True}, {"top_p": 0}, {"top_p": 1.01},
    {"top_p": np.nan}, {"top_p": True},
])
def test_invalid_decoding_options_even_in_greedy_mode(options):
    with pytest.raises(ValueError):
        distribution([1, 2, 3], **({"temperature": 0} | options))


def test_seeded_sampling_repeats_and_excludes_filtered_tokens():
    first, second = np.random.default_rng(42), np.random.default_rng(42)
    left = [decoder.sample_token([1, 2, 0], first, top_k=2) for _ in range(100)]
    right = [decoder.sample_token([1, 2, 0], second, top_k=2) for _ in range(100)]
    assert left == right
    assert set(left) == {0, 1}
    with pytest.raises(ValueError):
        decoder.sample_token([0, 1], None)


VOCABULARY = ("Hello", " world", "!", "<EOS>")


def greeting(**options):
    return decoder.generate(decoder.greeting_model, (), VOCABULARY, eos_id=3,
                            temperature=0, **options)


@pytest.mark.parametrize("options,text,reason,ids", [
    ({}, "Hello world!", "eos", (0, 1, 2, 3)),
    ({"stop_sequences": ("!",)}, "Hello world", "stop_sequence", (0, 1, 2)),
    ({"stop_sequences": ("lo wo",)}, "Hel", "stop_sequence", (0, 1)),
    ({"max_new_tokens": 2}, "Hello world", "length", (0, 1)),
    ({"max_new_tokens": 0}, "", "length", ()),
    ({"max_new_tokens": 3, "stop_sequences": ("!",)}, "Hello world", "stop_sequence", (0, 1, 2)),
    ({"max_new_tokens": 4}, "Hello world!", "eos", (0, 1, 2, 3)),
    ({"stop_sequences": ("<EOS>",)}, "Hello world!", "eos", (0, 1, 2, 3)),
])
def test_generation_finish_reasons_and_actual_budget(options, text, reason, ids):
    result = greeting(**options)
    assert (result.text, result.finish_reason, result.sampled_ids) == (text, reason, ids)


def test_earliest_completed_stop_ignores_list_order_and_overhang():
    for stops in (("lo", "ello"), ("ello", "lo")):
        result = greeting(stop_sequences=stops)
        assert result.text == "H"
        assert result.sampled_ids == (0,)


def test_completed_stop_does_not_wait_for_unfinished_earlier_match():
    result = greeting(stop_sequences=("Hello world!", "world"))
    assert result.text == "Hello "
    assert result.sampled_ids == (0, 1)


def test_unfinished_stop_prefix_is_returned_at_budget_exhaustion():
    result = greeting(stop_sequences=("Hello world!",), max_new_tokens=2)
    assert result.text == "Hello world"
    assert result.finish_reason == "length"


def test_model_receives_prompt_and_generated_context_without_mutation():
    contexts = []

    def model(context):
        contexts.append(context)
        return [-10, 10, -10, -10] if len(context) == 1 else [-10, -10, -10, 10]

    prompt = [0]
    result = decoder.generate(model, prompt, VOCABULARY, eos_id=3, temperature=0,
                              stop_sequences=("Hello",))
    assert contexts == [(0,), (0, 1)]
    assert prompt == [0]
    assert result.text == " world"
    assert result.finish_reason == "eos"


@pytest.mark.parametrize("options", [
    {"max_new_tokens": -1}, {"max_new_tokens": True}, {"max_new_tokens": 1.5},
    {"seed": -1}, {"seed": True}, {"stop_sequences": ("",)},
    {"stop_sequences": "!"}, {"stop_sequences": (1,)},
])
def test_invalid_generation_options(options):
    with pytest.raises(ValueError):
        greeting(**options)


def test_model_shape_and_nonfinite_output_rejected():
    for logits in ([1, 2], [1, 2, 3, np.nan], [[1, 2, 3, 4]]):
        with pytest.raises(ValueError):
            decoder.generate(lambda context: logits, (), VOCABULARY, eos_id=3)


@pytest.mark.parametrize("vocabulary,prompt,eos_id", [
    ((), (), 0), (("", "EOS"), (), 1), ("abc", (), 0),
    (("a", "EOS"), (-1,), 1), (("a", "EOS"), (2,), 1),
    (("a", "EOS"), (True,), 1), (("a", "EOS"), (), 2),
])
def test_invalid_vocabulary_prompt_and_eos(vocabulary, prompt, eos_id):
    with pytest.raises(ValueError):
        decoder.generate(lambda context: [0, 1], prompt, vocabulary, eos_id=eos_id)


def test_zero_budget_makes_no_model_call_but_validates_configuration():
    def must_not_run(context):
        raise AssertionError("model called at zero budget")

    result = decoder.generate(must_not_run, (), VOCABULARY, eos_id=3, max_new_tokens=0)
    assert result.sampled_ids == ()
    with pytest.raises(ValueError):
        decoder.generate(must_not_run, (), VOCABULARY, eos_id=3,
                         max_new_tokens=0, top_p=0)
