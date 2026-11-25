"""Deterministic character BPE for inspecting merge learning and application."""

from collections import Counter


def merge_pair(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Replace non-overlapping occurrences, scanning from left to right."""
    result = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            result.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return tuple(result)


class SimpleBPE:
    """Whitespace-split training; single-word inference; no token IDs or fallback."""

    def __init__(self, num_merges: int = 10):
        if type(num_merges) is not int or num_merges < 0:
            raise ValueError("num_merges must be a nonnegative integer")
        self.num_merges = num_merges
        self.alphabet: set[str] = set()
        self.vocabulary: set[str] = set()
        self.merge_ranks: dict[tuple[str, str], int] = {}
        self.history: list[tuple[tuple[str, str], int]] = []

    def fit(self, text: str) -> "SimpleBPE":
        if not isinstance(text, str) or not text.split():
            raise ValueError("training text must contain at least one word")
        words = Counter(text.split())
        corpus = {tuple(word): count for word, count in words.items()}
        self.alphabet = set("".join(words))
        self.vocabulary = self.alphabet.copy()
        self.merge_ranks = {}
        self.history = []
        for rank in range(self.num_merges):
            counts = Counter()
            for symbols, frequency in corpus.items():
                for pair in zip(symbols, symbols[1:]):
                    counts[pair] += frequency
            if not counts:
                break
            pair = min(counts, key=lambda item: (-counts[item], item))
            self.merge_ranks[pair] = rank
            self.history.append((pair, counts[pair]))
            self.vocabulary.add("".join(pair))
            updated = Counter()
            for symbols, frequency in corpus.items():
                updated[merge_pair(symbols, pair)] += frequency
            corpus = updated
        return self

    def encode_word(self, word: str) -> list[str]:
        if not self.alphabet:
            raise ValueError("fit the tokenizer before encoding")
        if not isinstance(word, str) or any(char.isspace() for char in word):
            raise ValueError("word must be a string without whitespace")
        if set(word) - self.alphabet:
            raise ValueError("word contains unseen characters; no byte fallback")
        symbols = tuple(word)
        while len(symbols) > 1:
            candidates = set(zip(symbols, symbols[1:])) & self.merge_ranks.keys()
            if not candidates:
                break
            pair = min(candidates, key=self.merge_ranks.__getitem__)
            symbols = merge_pair(symbols, pair)
        return list(symbols)


def main() -> None:
    # Synthetic weighted training corpus; evaluation words do not change merges.
    text = " ".join(["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3)
    tokenizer = SimpleBPE(10).fit(text)
    print("Synthetic character BPE; lexical tie breaking; no boundary markers")
    for rank, (pair, count) in enumerate(tokenizer.history):
        print(f"{rank:02d}: {pair!r} -> {''.join(pair)!r}, weighted count={count}")
    print(f"Alphabet={len(tokenizer.alphabet)}, vocabulary={len(tokenizer.vocabulary)}")
    for word in ["low", "lower", "lowest", "newer", "newest", "widest"]:
        pieces = tokenizer.encode_word(word)
        if "".join(pieces) != word:
            raise AssertionError("word reconstruction failed")
        print(f"{word:8s} -> {pieces}")


if __name__ == "__main__":
    main()
