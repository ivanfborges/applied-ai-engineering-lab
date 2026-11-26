"""Educational NumPy decoding and non-streaming fragment generation."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def decoding_distribution(
    logits: Sequence[float], *, temperature: float = 1.0,
    top_k: int | None = None, top_p: float = 1.0,
) -> np.ndarray:
    """Apply temperature, top-k, renormalization, then inclusive top-p.

    Finite float64 logits only. Ties favor the lower vocabulary index;
    temperature=0 is an explicit greedy branch, not division by zero.
    """
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("logits must be a nonempty finite vector.")
    for value, name in ((temperature, "temperature"), (top_p, "top_p")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite and real.")
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Require temperature >= 0 and 0 < top_p <= 1.")
    if top_k is not None:
        _integer(top_k, "top_k", 1)
        if top_k > scores.size:
            raise ValueError("top_k cannot exceed vocabulary size.")
    order = np.argsort(-scores, kind="stable")
    probabilities = np.zeros_like(scores)
    if temperature == 0:
        probabilities[order[0]] = 1.0
        return probabilities
    # Extreme finite gaps or tiny temperatures may become -inf; exp(-inf)=0.
    # The maximal entry always has centered score zero, so mass stays nonzero.
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp((scores - scores.max()) / temperature)
    probabilities = weights / weights.sum()
    if top_k is not None:
        probabilities[order[top_k:]] = 0.0
        probabilities /= probabilities.sum()
    if top_p < 1:
        cumulative = np.cumsum(probabilities[order])
        cumulative[-1] = 1.0
        count = int(np.searchsorted(cumulative, top_p, side="left")) + 1
        probabilities[order[count:]] = 0.0
        probabilities /= probabilities.sum()
    return probabilities


def sample_token(logits: Sequence[float], rng: np.random.Generator, **decoding) -> int:
    """Sample using an explicit caller-owned random generator."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator.")
    probabilities = decoding_distribution(logits, **decoding)
    return int(rng.choice(probabilities.size, p=probabilities))


def softmax(logits: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Stable temperature softmax; zero uses the documented greedy branch."""
    return decoding_distribution(logits, temperature=temperature)


def entropy_bits(probabilities: Sequence[float]) -> float:
    """Shannon entropy of a normalized finite categorical distribution."""
    values = np.asarray(probabilities, dtype=np.float64)
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
            or np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=1e-12)):
        raise ValueError("Expected a normalized nonnegative probability vector.")
    positive = values[values > 0]
    return float(-np.sum(positive * np.log2(positive)))


@dataclass(frozen=True)
class GenerationResult:
    text: str
    sampled_ids: tuple[int, ...]
    finish_reason: str


def generate(
    model: Callable[[tuple[int, ...]], Sequence[float]],
    prompt_ids: Sequence[int], vocabulary: Sequence[str], *, eos_id: int,
    max_new_tokens: int = 20, stop_sequences: Sequence[str] = (),
    temperature: float = 1.0, top_k: int | None = None,
    top_p: float = 1.0, seed: int = 42, rng: np.random.Generator | None = None,
) -> GenerationResult:
    """Generate concatenated synthetic fragments, not tokenizer-decoded text.

    Only newly generated text is searched. At the first step with complete
    matches, return the prefix before the earliest match. All sampled IDs count
    toward the budget, including EOS and the whole stop-triggering fragment.
    Text omits EOS and the matched stop plus any following characters.
    An explicit rng is advanced in place; otherwise seed initializes a fresh RNG.
    """
    if isinstance(vocabulary, (str, bytes)) or not vocabulary:
        raise ValueError("vocabulary must be a nonempty sequence of strings.")
    vocabulary = tuple(vocabulary)
    if any(not isinstance(piece, str) or not piece for piece in vocabulary):
        raise ValueError("vocabulary fragments must be nonempty strings.")
    _integer(eos_id, "eos_id")
    _integer(max_new_tokens, "max_new_tokens")
    _integer(seed, "seed")
    if eos_id >= len(vocabulary):
        raise ValueError("eos_id outside vocabulary.")
    context = tuple(prompt_ids)
    for token_id in context:
        _integer(token_id, "prompt token id")
        if token_id >= len(vocabulary):
            raise ValueError("prompt token id outside vocabulary.")
    if isinstance(stop_sequences, (str, bytes)):
        raise ValueError("stop_sequences must be a sequence of nonempty strings.")
    stops = tuple(stop_sequences)
    if any(not isinstance(stop, str) or not stop for stop in stops):
        raise ValueError("stop sequences must be nonempty strings.")
    decoding = dict(temperature=temperature, top_k=top_k, top_p=top_p)
    decoding_distribution(np.zeros(len(vocabulary)), **decoding)
    if rng is None:
        rng = np.random.default_rng(seed)
    elif not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a NumPy Generator or None.")
    sampled = []
    text = ""
    for _ in range(max_new_tokens):
        logits = np.asarray(model(context), dtype=np.float64)
        if logits.shape != (len(vocabulary),):
            raise ValueError("model must return one logit per vocabulary entry.")
        token_id = sample_token(logits, rng, **decoding)
        sampled.append(token_id)
        if token_id == eos_id:
            return GenerationResult(text, tuple(sampled), "eos")
        context += (token_id,)
        text += vocabulary[token_id]
        positions = [text.find(stop) for stop in stops if stop in text]
        if positions:
            return GenerationResult(text[:min(positions)], tuple(sampled), "stop_sequence")
    return GenerationResult(text, tuple(sampled), "length")


def greeting_model(context: tuple[int, ...]) -> np.ndarray:
    """Handwritten transition fixture: Hello, space-world, !, EOS."""
    next_id = 0 if not context else min(context[-1] + 1, 3)
    logits = np.full(4, -10.0)
    logits[next_id] = 10.0
    return logits


if __name__ == "__main__":
    vocabulary = ("Hello", " world", "!", "<EOS>")
    for options in ({}, {"stop_sequences": ("!",)}, {"max_new_tokens": 2}):
        print(generate(greeting_model, (), vocabulary, eos_id=3, temperature=0, **options))
