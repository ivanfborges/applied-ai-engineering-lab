"""Hand counts, independent gradients, optimization and retrieval policies."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import (
    EmbeddingFit, analogy, cosine, glove_loss_grad, nearest, predictive_loss_grad,
    train_glove, train_predictive, training_data, unit_rows, word_vector,
)
from example import build_corpus


def finite_differences(objective, parameters):
    expected = objective(*parameters)[1]
    for parameter, gradient in zip(parameters, expected):
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + 1e-6
            plus = objective(*parameters)[0]
            parameter[index] = original - 1e-6
            minus = objective(*parameters)[0]
            parameter[index] = original
            numerical[index] = (plus - minus) / 2e-6
        np.testing.assert_allclose(gradient, numerical, atol=2e-9, rtol=2e-5)


def test_hand_counts_direction_distance_duplicates_and_sentence_boundaries():
    words, contexts, centers, pairs, counts = training_data([["a", "b", "a"], ["c", "b"]], window=2)
    assert words == ("a", "b", "c")
    np.testing.assert_array_equal(counts, [[1, 2, 0], [2, 0, 1], [0, 1, 0]])
    np.testing.assert_allclose(contexts[1], [1, 0, 0])
    assert centers[1] == 1
    assert np.sum(np.all(pairs == [1, 0], axis=1)) == 2
    assert len(pairs) == 8
    np.testing.assert_allclose(contexts.sum(axis=1), 1)
    assert np.all(training_data([["a", "b", "c"]], window=1)[4][[0, 2], [2, 0]] == 0)


@pytest.mark.parametrize("mode", ["cbow", "skipgram"])
def test_predictive_gradients_all_parameters(mode):
    words, contexts, centers, pairs, _ = training_data([["a", "a", "b"], ["b", "c"]])
    features, targets = (contexts, centers) if mode == "cbow" else (np.eye(len(words))[pairs[:, 0]], pairs[:, 1])
    rng = np.random.default_rng(4)
    parameters = [rng.normal(0, 0.3, (3, 2)), rng.normal(0, 0.3, (3, 2))]
    finite_differences(lambda v, u: predictive_loss_grad(v, u, features, targets), parameters)


def test_full_softmax_uniform_and_large_logits():
    loss, _ = predictive_loss_grad(np.ones((2, 1)), np.zeros((2, 1)), np.eye(2), np.array([0, 1]))
    assert loss == pytest.approx(np.log(2))
    loss, gradients = predictive_loss_grad(np.ones((2, 1)), np.array([[1000.], [-1000.]]), np.eye(2), np.array([1, 0]))
    assert loss == pytest.approx(1000)
    assert all(np.isfinite(g).all() for g in gradients)


def test_glove_gradients_vectors_and_biases_with_fractional_counts():
    rng = np.random.default_rng(2)
    counts = np.array([[0, 0.5, 3], [0.5, 0, 2], [3, 2, 0.]])
    parameters = [rng.normal(0, 0.3, (3, 2)), rng.normal(0, 0.3, (3, 2)), rng.normal(0, 0.2, 3), rng.normal(0, 0.2, 3)]
    finite_differences(lambda v, u, b, c: glove_loss_grad(v, u, b, c, counts, x_max=2), parameters)


def test_glove_observed_only_mean_and_capped_weights():
    counts = np.array([[0., 0.5], [4, 0]])
    zeros, bias = np.zeros((2, 1)), np.zeros(2)
    loss, _ = glove_loss_grad(zeros, zeros, bias, bias, counts, x_max=2, alpha=1)
    assert loss == pytest.approx((0.25 * np.log(0.5)**2 + np.log(4)**2) / 2)
    # Arbitrary scores at unobserved entries contribute no loss or gradient.
    counts = np.eye(2)
    loss, gradients = glove_loss_grad(np.eye(2), np.array([[0, 9.], [7, 0]]), bias, bias, counts)
    assert loss == 0
    assert all(np.all(g == 0) for g in gradients)


@pytest.mark.parametrize("method", ["cbow", "skipgram", "glove"])
def test_training_is_finite_reproducible_and_reduces_own_loss(method):
    corpus = build_corpus()
    def fit():
        return train_glove(corpus, epochs=80) if method == "glove" else train_predictive(corpus, mode=method, epochs=80)
    first, second = fit(), fit()
    assert first.vectors.shape == (len(first.words), 8)
    assert np.isfinite(first.vectors).all()
    assert len(first.losses) == 81
    assert first.losses[-1] < first.losses[0]
    np.testing.assert_array_equal(first.vectors, second.vectors)
    np.testing.assert_array_equal(first.losses, second.losses)


def test_one_update_matches_simultaneous_preupdate_gradients():
    corpus = [["a", "b", "a"]]
    words, contexts, targets, _, counts = training_data(corpus)
    rng = np.random.default_rng(65)
    v = rng.normal(0, 0.1, (len(words), 2))
    u = rng.normal(0, 0.1, v.shape)
    _, (dv, _) = predictive_loss_grad(v, u, contexts, targets)
    actual = train_predictive(corpus, dim=2, epochs=1, lr=0.2)
    np.testing.assert_allclose(actual.vectors, v - 0.2 * dv)
    zeros = np.zeros(len(words))
    _, (dv, du, _, _) = glove_loss_grad(v, u, zeros, zeros, counts)
    actual = train_glove(corpus, dim=2, epochs=1, lr=0.2)
    np.testing.assert_allclose(actual.vectors, v + u - 0.2 * (dv + du))


def test_cosine_geometry_and_scale_safety():
    assert cosine([1, 0], [0, 1]) == 0
    assert cosine([1, 2], [-1, -2]) == pytest.approx(-1)
    assert cosine([1e300, 2e300], [1e-300, 2e-300]) == pytest.approx(1)
    x = unit_rows([[1, 2], [3, -1]])
    assert cosine(x[0], x[1]) == pytest.approx(cosine(x[1], x[0]))
    assert np.sum((x[0] - x[1])**2) == pytest.approx(2 - 2 * cosine(x[0], x[1]))


def test_ranking_exclusions_ties_oov_and_constructed_analogy():
    fit = EmbeddingFit(("man", "king", "woman", "queen", "zebra"), np.array([[1., 0], [1, 1], [2, 0], [2, 1], [-1, 0]]), np.array([]))
    result = analogy(fit, "man", "king", "woman", k=10)
    assert result[0][0] == "queen"
    assert {w for w, _ in result} == {"queen", "zebra"}
    assert nearest(("z", "a"), [[1, 0], [1, 0]], [1, 0]) == [("a", 1.), ("z", 1.)]
    assert nearest(("a",), [[1, 0]], [1, 0], exclude=("a",)) == []
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        word_vector(fit, "unseen")


@pytest.mark.parametrize("corpus", [[], [[]], ["a b"], [[""]], [[3]], [[" a"]], [["a"], ["b"]], "a b"])
def test_invalid_corpus(corpus):
    with pytest.raises(ValueError):
        training_data(corpus)


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"dim": 0}, {"epochs": -1}, {"dim": True}, {"epochs": 1.5}, {"lr": np.nan}, {"lr": -1}])
@pytest.mark.parametrize("trainer", [train_predictive, train_glove])
def test_invalid_training_configuration(trainer, kwargs):
    with pytest.raises(ValueError):
        trainer([["a", "b"]], **kwargs)


@pytest.mark.parametrize("kwargs", [{"x_max": 0}, {"alpha": np.nan}, {"alpha": -1}])
def test_invalid_glove_configuration(kwargs):
    with pytest.raises(ValueError):
        train_glove([["a", "b"]], **kwargs)


@pytest.mark.parametrize("a,b", [([0, 0], [1, 2]), ([np.nan], [1]), ([np.inf], [1]), ([1, 2], [1]), ([], []), ([[1]], [[1]])])
def test_invalid_cosine(a, b):
    with pytest.raises(ValueError):
        cosine(a, b)


def test_invalid_ranking_and_mode():
    for words, vectors, query, k in [(("a",), [[1]], [1], 0), (("a",), [[1]], [1, 2], 1), (("a", "a"), [[1], [1]], [1], 1), (("a",), [[1], [1]], [1], 1)]:
        with pytest.raises(ValueError):
            nearest(words, vectors, query, k)
    with pytest.raises(ValueError):
        train_predictive([["a", "b"]], mode="sgns")
