"""A small AI system combining classification, retrieval, and routing.

All training examples and knowledge-base documents are synthetic. The script is
educational and its outputs must not be interpreted as benchmark results.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline


TRAINING_DATA = [
    ("I was charged twice for the same purchase", "billing"),
    ("My invoice contains an incorrect amount", "billing"),
    ("I need a refund for a duplicated payment", "billing"),
    ("The subscription price changed unexpectedly", "billing"),
    ("My credit card payment failed", "billing"),
    ("Where can I download my invoice?", "billing"),
    ("The application crashes when I upload a PDF", "technical"),
    ("The API returns a 500 error", "technical"),
    ("Document processing is stuck", "technical"),
    ("The dashboard is not loading", "technical"),
    ("The integration stopped sending events", "technical"),
    ("The system is very slow today", "technical"),
    ("I cannot sign in to my account", "account"),
    ("I forgot my password", "account"),
    ("How can I change my email address?", "account"),
    ("My account has been locked", "account"),
    ("I need to enable two-factor authentication", "account"),
    ("How do I delete my account?", "account"),
]

KNOWLEDGE_BASE = [
    {
        "id": "billing-duplicate",
        "category": "billing",
        "text": (
            "If an invoice was paid twice, confirm whether both duplicate charges "
            "were settled. "
            "Pending authorizations usually disappear automatically. If both "
            "charges were settled, open a billing dispute."
        ),
    },
    {
        "id": "billing-invoice",
        "category": "billing",
        "text": (
            "Invoices can be downloaded from Settings, Billing, Invoices. "
            "Invoice corrections require a billing support request."
        ),
    },
    {
        "id": "technical-upload",
        "category": "technical",
        "text": (
            "For document upload failures, validate the file format, file size, "
            "and API response. Retry only transient 5xx errors with backoff."
        ),
    },
    {
        "id": "technical-api",
        "category": "technical",
        "text": (
            "For API 500 errors, record the request identifier, endpoint, and "
            "timestamp. Do not retry non-idempotent operations automatically."
        ),
    },
    {
        "id": "account-access",
        "category": "account",
        "text": (
            "For account access problems, use the password-reset flow. Locked "
            "accounts may require identity verification by support."
        ),
    },
    {
        "id": "account-security",
        "category": "account",
        "text": (
            "Two-factor authentication can be enabled in Settings, Security. "
            "Store recovery codes in a secure location."
        ),
    },
]


@dataclass(frozen=True)
class Decision:
    """A compact trace of one system decision."""

    query: str
    predicted_category: str
    classifier_confidence: float
    retrieved_document_id: str
    retrieval_similarity: float
    action: str
    response: str
    latency_ms: float


def build_classifier() -> Pipeline:
    """Train a lightweight classifier on the synthetic examples."""
    texts = [text for text, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]

    classifier = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    ngram_range=(1, 2), lowercase=True, stop_words="english"
                ),
            ),
            ("model", LogisticRegression(max_iter=1_000, random_state=42)),
        ]
    )
    classifier.fit(texts, labels)
    return classifier


class KnowledgeRetriever:
    """Rank synthetic knowledge-base documents with lexical similarity."""

    def __init__(self, documents: list[dict[str, str]]) -> None:
        if not documents:
            raise ValueError("At least one knowledge-base document is required.")

        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), lowercase=True, stop_words="english"
        )
        self.document_matrix = self.vectorizer.fit_transform(
            document["text"] for document in documents
        )

    def retrieve(
        self, query: str, category: str | None = None
    ) -> tuple[dict[str, str], float]:
        """Return the highest-scoring allowed document and its similarity."""
        similarities = cosine_similarity(
            self.vectorizer.transform([query]), self.document_matrix
        )[0]

        candidate_indices = [
            index
            for index, document in enumerate(self.documents)
            if category is None or document["category"] == category
        ]
        if not candidate_indices:
            raise ValueError("No knowledge-base candidates match the filter.")

        best_index = max(candidate_indices, key=lambda index: similarities[index])
        return self.documents[best_index], float(similarities[best_index])


def process_ticket(
    query: str,
    classifier: Pipeline,
    retriever: KnowledgeRetriever,
    confidence_threshold: float = 0.40,
    retrieval_threshold: float = 0.05,
) -> Decision:
    """Classify, retrieve evidence, then answer or abstain."""
    if not query.strip():
        raise ValueError("The query must not be empty.")

    started_at = time.perf_counter()
    probabilities = classifier.predict_proba([query])[0]
    classes = classifier.named_steps["model"].classes_
    best_class_index = int(np.argmax(probabilities))
    predicted_category = str(classes[best_class_index])
    confidence = float(probabilities[best_class_index])

    document, similarity = retriever.retrieve(query, category=predicted_category)

    if confidence < confidence_threshold:
        action = "human_review"
        response = "The request could not be classified with enough confidence."
    elif similarity < retrieval_threshold:
        action = "human_review"
        response = (
            "The category was identified, but no sufficiently relevant "
            "knowledge-base evidence was found."
        )
    else:
        action = "retrieval_augmented_response"
        response = (
            f"Category: {predicted_category}. Relevant guidance: {document['text']}"
        )

    return Decision(
        query=query,
        predicted_category=predicted_category,
        classifier_confidence=confidence,
        retrieved_document_id=document["id"],
        retrieval_similarity=similarity,
        action=action,
        response=response,
        latency_ms=(time.perf_counter() - started_at) * 1_000,
    )


def main() -> None:
    classifier = build_classifier()
    retriever = KnowledgeRetriever(KNOWLEDGE_BASE)
    example_queries = [
        "I paid the same invoice twice",
        "The API fails while uploading a document",
        "My account is locked and password reset failed",
        "Can you negotiate a new commercial contract for me?",
    ]

    for query in example_queries:
        decision = process_ticket(query, classifier, retriever)
        print(json.dumps(asdict(decision), indent=2))
        print("-" * 80)


if __name__ == "__main__":
    main()
