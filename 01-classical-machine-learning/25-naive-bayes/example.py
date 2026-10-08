"""A tiny, synthetic support-ticket routing example with training-only vectorization."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


TRAIN_TEXTS = [
    "invoice has duplicate payment",
    "billing charged wrong amount",
    "please refund the invoice",
    "payment amount is incorrect",
    "cannot login to account",
    "password reset link failed",
    "account access is blocked",
    "login with password fails",
]
TRAIN_LABELS = [
    "billing", "billing", "billing", "billing",
    "access", "access", "access", "access",
]
QUERIES = [
    "refund the wrong payment",
    "password reset for account",
]


def build_model():
    """Keep vocabulary fitting inside the model's training step."""
    return make_pipeline(
        CountVectorizer(lowercase=True, ngram_range=(1, 1)),
        MultinomialNB(alpha=1.0),
    )


def main():
    model = build_model().fit(TRAIN_TEXTS, TRAIN_LABELS)
    vectorizer = model.named_steps["countvectorizer"]
    classifier = model.named_steps["multinomialnb"]
    predictions = model.predict(QUERIES)

    for text, label in zip(QUERIES, predictions):
        print(f"{text!r} -> {label}")

    # A token's class contrast is more informative than its class probability alone.
    vocabulary = vectorizer.vocabulary_
    billing = np.flatnonzero(classifier.classes_ == "billing")[0]
    access = np.flatnonzero(classifier.classes_ == "access")[0]
    for token in ("invoice", "password"):
        index = vocabulary[token]
        contrast = classifier.feature_log_prob_[billing, index] - classifier.feature_log_prob_[access, index]
        print(f"{token!r} log-likelihood contrast (billing - access): {contrast:+.3f}")


if __name__ == "__main__":
    main()
