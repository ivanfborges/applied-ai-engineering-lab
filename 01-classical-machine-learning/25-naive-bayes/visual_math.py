"""Small numerical helpers for the Day 25 visual explanations."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from example import TRAIN_LABELS, TRAIN_TEXTS


def fit_text_model():
    """Fit vocabulary and model on the same code-defined synthetic training texts."""
    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 1))
    X = vectorizer.fit_transform(TRAIN_TEXTS)
    model = MultinomialNB(alpha=1.0).fit(X, TRAIN_LABELS)
    return vectorizer, model


def fit_count_model(seed=25):
    """Two interpretable count features with overlap between synthetic classes."""
    rng = np.random.default_rng(seed)
    billing = rng.poisson([3.0, 1.3], size=(28, 2))
    access = rng.poisson([1.3, 3.0], size=(28, 2))
    X = np.vstack((billing, access)).astype(int)
    y = np.array(["billing"] * len(billing) + ["access"] * len(access))
    model = MultinomialNB(alpha=1.0).fit(X, y)
    return X, y, model


def bayes_update(prior, likelihood):
    prior = np.asarray(prior, dtype=float)
    likelihood = np.asarray(likelihood, dtype=float)
    if prior.shape != (2,) or likelihood.shape != (2,):
        raise ValueError("prior and likelihood must each have two entries")
    if not np.isfinite(prior).all() or not np.isfinite(likelihood).all():
        raise ValueError("probabilities must be finite")
    if np.any(prior <= 0) or not np.isclose(prior.sum(), 1):
        raise ValueError("prior must be positive and sum to one")
    if np.any(likelihood < 0) or np.any(likelihood > 1):
        raise ValueError("likelihood entries must lie between zero and one")
    unnormalized = prior * likelihood
    if unnormalized.sum() == 0:
        raise ValueError("evidence must have positive probability")
    return unnormalized, unnormalized / unnormalized.sum()


def smoothed_probabilities(counts, alpha):
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1 or len(counts) == 0:
        raise ValueError("counts must be a nonempty vector")
    if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("counts must be finite nonnegative integers")
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
        raise ValueError("alpha must be finite and nonnegative")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be finite and nonnegative") from exc
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    denominator = counts.sum() + alpha * len(counts)
    if denominator <= 0:
        raise ValueError("at least one count or positive alpha is required")
    return (counts + alpha) / denominator


def class_indices(model):
    labels = model.classes_.tolist()
    return labels.index("billing"), labels.index("access")


def score_difference(model, X, prior_billing=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != model.feature_log_prob_.shape[1]:
        raise ValueError("X must have the fitted feature count")
    if not np.isfinite(X).all() or np.any(X < 0):
        raise ValueError("X must contain finite nonnegative values")
    billing, access = class_indices(model)
    if prior_billing is None:
        intercept = model.class_log_prior_[billing] - model.class_log_prior_[access]
    else:
        if not np.isfinite(prior_billing) or not 0 < prior_billing < 1:
            raise ValueError("prior_billing must lie strictly between zero and one")
        intercept = np.log(prior_billing) - np.log1p(-prior_billing)
    weights = model.feature_log_prob_[billing] - model.feature_log_prob_[access]
    return intercept + X @ weights


def posterior_from_difference(difference):
    difference = np.asarray(difference, dtype=float)
    if not np.isfinite(difference).all():
        raise ValueError("score differences must be finite")
    small = np.exp(-np.abs(difference))
    return np.where(difference >= 0, 1 / (1 + small), small / (1 + small))


def token_contrasts(vectorizer, model):
    billing, access = class_indices(model)
    values = model.feature_log_prob_[billing] - model.feature_log_prob_[access]
    return dict(zip(vectorizer.get_feature_names_out(), values))


def accumulated_scores(vectorizer, model, tokens):
    if isinstance(tokens, str) or not tokens or not all(isinstance(token, str) and token for token in tokens):
        raise ValueError("tokens must be a nonempty sequence of words")
    scores = []
    for count in range(len(tokens) + 1):
        text = " ".join(tokens[:count])
        X = vectorizer.transform([text])
        scores.append(model.class_log_prior_ + X @ model.feature_log_prob_.T)
    return np.vstack(scores)


def correlated_copy_posteriors(copies=4, prior=0.5, likelihood_billing=0.75, likelihood_access=0.25):
    """Compare exact one-latent-event Bayes with naive multiplication of copies."""
    if isinstance(copies, bool) or not isinstance(copies, int) or copies < 1:
        raise ValueError("copies must be a positive integer")
    if not 0 < prior < 1 or not 0 < likelihood_billing < 1 or not 0 < likelihood_access < 1:
        raise ValueError("prior and likelihoods must lie strictly between zero and one")
    exact = prior * likelihood_billing / (
        prior * likelihood_billing + (1 - prior) * likelihood_access
    )
    n = np.arange(1, copies + 1)
    log_odds = np.log(prior / (1 - prior)) + n * np.log(likelihood_billing / likelihood_access)
    return n, np.full(copies, exact), posterior_from_difference(log_odds)
