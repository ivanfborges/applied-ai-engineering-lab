"""Numerical and artifact checks for the Day 25 visual explanations."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))

import visual_math as vm
import visualizations as viz


def test_bayes_update_normalizes_and_rejects_impossible_evidence():
    joint, posterior = vm.bayes_update([0.5, 0.5], [0.75, 0.25])
    np.testing.assert_allclose(joint, [0.375, 0.125])
    np.testing.assert_allclose(posterior, [0.75, 0.25])
    with pytest.raises(ValueError, match="positive probability"):
        vm.bayes_update([0.5, 0.5], [0, 0])


def test_smoothing_changes_zeros_and_normalizes():
    counts = [8, 5, 3, 0, 0]
    unsmoothed = vm.smoothed_probabilities(counts, 0)
    smoothed = vm.smoothed_probabilities(counts, 1)
    assert unsmoothed[3] == 0
    assert smoothed[3] > 0
    np.testing.assert_allclose(unsmoothed.sum(), 1)
    np.testing.assert_allclose(smoothed.sum(), 1)
    with pytest.raises(ValueError, match="nonnegative"):
        vm.smoothed_probabilities([-1, 1], 1)
    with pytest.raises(ValueError, match="alpha"):
        vm.smoothed_probabilities([1, 0], "bad")


def test_token_evidence_and_accumulation_match_library():
    vectorizer, model = vm.fit_text_model()
    evidence = vm.token_contrasts(vectorizer, model)
    assert evidence["invoice"] > 0
    assert evidence["password"] < 0
    tokens = ["invoice", "payment", "amount", "incorrect"]
    scores = vm.accumulated_scores(vectorizer, model, tokens)
    np.testing.assert_allclose(scores[0], model.class_log_prior_)
    X = vectorizer.transform([" ".join(tokens)])
    np.testing.assert_allclose(scores[-1], (model.class_log_prior_ + X @ model.feature_log_prob_.T)[0])
    assert model.classes_[np.argmax(scores[-1])] == model.predict(X)[0]
    assert np.isfinite(scores).all()
    with pytest.raises(ValueError, match="tokens"):
        vm.accumulated_scores(vectorizer, model, "invoice")


def test_count_score_difference_and_prior_shift():
    X, y, model = vm.fit_count_model()
    X_again, y_again, model_again = vm.fit_count_model()
    np.testing.assert_array_equal(X, X_again)
    np.testing.assert_array_equal(y, y_again)
    np.testing.assert_allclose(model.feature_log_prob_, model_again.feature_log_prob_)

    billing, access = vm.class_indices(model)
    queries = np.array([[0, 0], [2, 2], [5, 1]])
    expected = (
        model.class_log_prior_[billing] - model.class_log_prior_[access]
        + queries @ (model.feature_log_prob_[billing] - model.feature_log_prob_[access])
    )
    np.testing.assert_allclose(vm.score_difference(model, queries), expected)
    low = vm.score_difference(model, queries, prior_billing=0.1)
    high = vm.score_difference(model, queries, prior_billing=0.9)
    np.testing.assert_allclose(high - low, 2 * np.log(9))
    assert np.all(high > low)
    with pytest.raises(ValueError, match="prior_billing"):
        vm.score_difference(model, queries, prior_billing=1)


def test_correlated_copy_example_distinguishes_posteriors():
    n, exact, naive = vm.correlated_copy_posteriors()
    np.testing.assert_array_equal(n, [1, 2, 3, 4])
    np.testing.assert_allclose(exact, 0.75)
    np.testing.assert_allclose(naive[0], exact[0])
    assert np.all(np.diff(naive) > 0)
    assert naive[-1] > exact[-1]
    with pytest.raises(ValueError, match="copies"):
        vm.correlated_copy_posteriors(copies=True)


def test_representative_artifacts_are_generated(tmp_path):
    png = viz.plot_token_evidence(tmp_path)
    gif = viz.animate_laplace_smoothing(tmp_path)
    html = viz.plot_posterior_surface_3d(tmp_path)
    assert png == tmp_path / "03_token_evidence.png"
    assert gif == tmp_path / "05_laplace_smoothing.gif"
    assert html == tmp_path / "09_posterior_surface.html"
    with Image.open(png) as image:
        assert image.width > 300 and image.height > 200
    with Image.open(gif) as image:
        assert image.n_frames == 7
    assert "Plotly.newPlot" in html.read_text(encoding="utf-8")
