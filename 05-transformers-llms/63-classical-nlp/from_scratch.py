"""Educational tokenization, morphology fixtures and dense n-gram counts."""

from collections import Counter
from collections.abc import Sequence
import re
import unicodedata

import numpy as np


TOKEN_PATTERN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)

# Hand-authored English examples; these are not inferred POS tags or a lexicon.
LEMMA_FIXTURE = {
    ("studies", "v"): "study",
    ("companies", "n"): "company",
    ("running", "v"): "run",
    ("mice", "n"): "mouse",
    ("was", "v"): "be",
    ("better", "a"): "good",
    ("saw", "v"): "see",
    ("saw", "n"): "saw",
    ("news", "n"): "news",
}


def normalize(text: str) -> str:
    """Apply NFC, case folding, apostrophe unification and whitespace collapse."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    folded = unicodedata.normalize("NFC", text).casefold().replace("\u2019", "'")
    return unicodedata.normalize("NFC", " ".join(folded.split()))


def tokenize(text: str) -> list[str]:
    """Keep alphanumeric runs and internal apostrophes; discard other separators."""
    return TOKEN_PATTERN.findall(normalize(text))


def _sequence(values: Sequence, name: str) -> None:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence, not a single string or iterator")


def _order(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 1:
        raise ValueError("n must be positive")


def generate_ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    """Return contiguous windows without padding; keep repeated occurrences."""
    _order(n)
    _sequence(tokens, "tokens")
    if any(not isinstance(t, str) or not t or any(c.isspace() for c in t)
           for t in tokens):
        raise ValueError("tokens must be nonempty strings without whitespace")
    return [tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


def document_term_matrix(
    documents: Sequence[str],
    n: int = 1,
    vocabulary: Sequence[tuple[str, ...]] | None = None,
) -> tuple[list[tuple[str, ...]], np.ndarray]:
    """Fit sorted features or reuse supplied columns; ignore unseen n-grams.

    Empty corpora/vocabularies are valid. The dense int64 allocation is for tiny
    examples only. Supplied vocabulary tokens must already use this analyzer.
    """
    _order(n)
    _sequence(documents, "documents")
    grams = [generate_ngrams(tokenize(doc), n) for doc in documents]
    if vocabulary is None:
        features = sorted({gram for row in grams for gram in row})
    else:
        _sequence(vocabulary, "vocabulary")
        features = list(vocabulary)
        for gram in features:
            if not isinstance(gram, tuple) or len(gram) != n:
                raise ValueError("vocabulary entries must be tuples of length n")
            generate_ngrams(gram, n)
            if any(tokenize(token) != [token] for token in gram):
                raise ValueError("vocabulary tokens must already be normalized")
        if len(set(features)) != len(features):
            raise ValueError("vocabulary entries must be unique")
    columns = {gram: i for i, gram in enumerate(features)}
    matrix = np.zeros((len(documents), len(features)), dtype=np.int64)
    for row, occurrences in enumerate(grams):
        for gram, count in Counter(occurrences).items():
            if gram in columns:
                matrix[row, columns[gram]] = count
    return features, matrix


def suffix_stem_demo(word: str) -> str:
    """Strip one suffix mechanically. Deliberately NOT Porter or Snowball."""
    word = normalize(word)
    if tokenize(word) != [word]:
        raise ValueError("word must be one alphanumeric/apostrophe token")
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[:-len(suffix)]
    return word


def lemma_lookup_demo(word: str, pos: str) -> str:
    """Read a closed English teaching fixture; unsupported entries fail loudly."""
    if pos not in ("n", "v", "a"):
        raise ValueError("demo POS must be n, v or a")
    key = (normalize(word), pos)
    if key not in LEMMA_FIXTURE:
        raise ValueError("word/POS is outside the closed teaching fixture")
    return LEMMA_FIXTURE[key]


def main() -> None:
    documents = ["Machine learning models", "Machine learning learning", "Deep learning models"]
    print("Synthetic count example; dense educational matrices")
    for n in (1, 2, 3):
        vocabulary, matrix = document_term_matrix(documents, n)
        print(f"\nn={n}, features={vocabulary}\n{matrix}")
    print("\nMorphology: mechanical suffix removal versus supplied word/POS lookup")
    for word, pos in LEMMA_FIXTURE:
        print(f"{word:10} {pos}  suffix={suffix_stem_demo(word):8} "
              f"fixture lemma={lemma_lookup_demo(word, pos)}")


if __name__ == "__main__":
    main()
