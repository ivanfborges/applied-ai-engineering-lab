"""Educational bag-of-words retrieval implemented from first principles.

The tiny document collection is synthetic. NumPy is used only for vector
operations; tokenization, vocabulary construction, and ranking are explicit.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np


DOCUMENTS = [
    "duplicate payments require a billing dispute",
    "password reset is available in account security",
    "api failures should include the request identifier",
]


def tokenize(text: str) -> list[str]:
    """Apply a deliberately simple lowercase alphanumeric tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocabulary(texts: list[str]) -> list[str]:
    """Return one stable, sorted dimension for every observed token."""
    return sorted({token for text in texts for token in tokenize(text)})


def vectorize(text: str, vocabulary: list[str]) -> np.ndarray:
    """Represent text using raw token counts in vocabulary order."""
    counts = Counter(tokenize(text))
    return np.array([counts[token] for token in vocabulary], dtype=float)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Measure vector direction similarity, returning zero for a zero vector."""
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def retrieve(query: str, documents: list[str]) -> tuple[str, float]:
    """Return the document with the highest bag-of-words cosine similarity."""
    if not query.strip():
        raise ValueError("The query must not be empty.")
    if not documents:
        raise ValueError("At least one document is required.")

    vocabulary = build_vocabulary(documents + [query])
    query_vector = vectorize(query, vocabulary)
    scores = [
        cosine_similarity(query_vector, vectorize(document, vocabulary))
        for document in documents
    ]
    best_index = int(np.argmax(scores))
    return documents[best_index], scores[best_index]


def main() -> None:
    query = "I have two payments for one invoice"
    document, score = retrieve(query, DOCUMENTS)
    print(f"Query: {query}")
    print(f"Document: {document}")
    print(f"Similarity: {score:.3f}")
    print("\nLimitation: lexical overlap cannot capture synonyms or semantics.")


if __name__ == "__main__":
    main()
