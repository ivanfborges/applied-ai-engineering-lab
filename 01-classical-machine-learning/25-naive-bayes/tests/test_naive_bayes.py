"""Checks for the educational count model and the text pipeline."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.naive_bayes import MultinomialNB


TOPIC = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOPIC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load_module("day25_scratch", "from_scratch.py")
example = load_module("day25_example", "example.py")


def test_matches_library_on_counts_and_class_order():
    X = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 3], [0, 0, 1]])
    y = np.array(["billing", "billing", "access", "access"])
    queries = np.array([[1, 0, 0], [0, 0, 2], [0, 0, 0]])
    actual = scratch.MultinomialNBFromScratch(alpha=1.0).fit(X, y)
    expected = MultinomialNB(alpha=1.0).fit(X, y)

    np.testing.assert_array_equal(actual.classes_, expected.classes_)
    np.testing.assert_allclose(actual.class_log_prior_, expected.class_log_prior_)
    np.testing.assert_allclose(actual.feature_log_prob_, expected.feature_log_prob_)
    np.testing.assert_array_equal(actual.predict(queries), expected.predict(queries))
    expected_scores = queries @ expected.feature_log_prob_.T + expected.class_log_prior_
    np.testing.assert_allclose(actual.joint_log_likelihood(queries), expected_scores)


def test_smoothing_keeps_unseen_class_token_finite():
    model = scratch.MultinomialNBFromScratch(alpha=0.5).fit(
        [[2, 0], [0, 1]], ["billing", "access"]
    )
    assert np.isfinite(model.feature_log_prob_).all()
    assert np.isfinite(model.joint_log_likelihood([[1, 1]])).all()
    # A zero-count document contributes only the prior.
    np.testing.assert_allclose(model.joint_log_likelihood([[0, 0]])[0], model.class_log_prior_)


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf"), True, "bad"])
def test_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        scratch.MultinomialNBFromScratch(alpha)


@pytest.mark.parametrize("X", [[[-1, 0]], [[0.5, 1]], [[float("nan"), 1]], [[float("inf"), 1]]])
def test_rejects_invalid_counts(X):
    with pytest.raises(ValueError, match="finite nonnegative integer"):
        scratch.MultinomialNBFromScratch().fit(X, ["a"])


def test_rejects_bad_shapes_and_unfitted_queries():
    model = scratch.MultinomialNBFromScratch()
    with pytest.raises(ValueError, match="before scoring"):
        model.predict([[1]])
    with pytest.raises(ValueError, match="at least one training row"):
        model.fit(np.empty((0, 2)), [])
    with pytest.raises(ValueError, match="one label"):
        model.fit([[1], [2]], ["a"])
    model.fit([[1], [2]], ["a", "b"])
    with pytest.raises(ValueError, match="fitted feature count"):
        model.predict([[1, 2]])


def test_pipeline_vocabulary_is_training_only():
    model = example.build_model().fit(example.TRAIN_TEXTS, example.TRAIN_LABELS)
    vectorizer = model.named_steps["countvectorizer"]
    original_vocabulary = vectorizer.vocabulary_.copy()
    model.predict(["novelword neverseen"])
    assert vectorizer.vocabulary_ == original_vocabulary
    assert vectorizer.transform(["novelword neverseen"]).nnz == 0
    np.testing.assert_array_equal(model.classes_, ["access", "billing"])
