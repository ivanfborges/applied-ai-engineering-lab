"""Numerical mechanisms, real fit caching, recorded updates and temporary exports."""

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image
import pytest


TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))
import manifold_visual_core as core
import manifold_visualizations as scenes


def test_entropy_and_perplexity_match_a_uniform_distribution():
    p, entropy = core.entropy_probabilities(np.ones(9), beta=2)
    np.testing.assert_allclose(p, 1 / 9)
    assert np.exp(entropy) == pytest.approx(9)


@pytest.mark.parametrize("target", [5, 15, 30])
def test_binary_search_matches_entropy_and_sigma(target):
    X, _ = core.probability_sample()
    distance2 = np.sum((X[1:] - X[0]) ** 2, axis=1)
    solution, history = core.match_perplexity(distance2, target)
    raw = np.exp(-distance2 / (2 * solution["sigma"] ** 2))
    np.testing.assert_allclose(solution["probabilities"], raw / raw.sum(), rtol=1e-12, atol=1e-15)
    assert solution["perplexity"] == pytest.approx(target, rel=2e-7)
    assert 2 ** solution["entropy_bits"] == pytest.approx(solution["perplexity"])
    assert history[-1] is solution


def test_local_bandwidth_changes_but_probabilities_remain_normalized():
    X, _ = core.probability_sample()
    P, conditional, sigmas, achieved = core.perplexity_affinities(X, 15)
    np.testing.assert_allclose(conditional.sum(axis=1), 1)
    np.testing.assert_array_equal(np.diag(conditional), 0)
    np.testing.assert_allclose(P, P.T, atol=1e-15)
    assert P.sum() == pytest.approx(1)
    assert np.std(sigmas) > 0
    np.testing.assert_allclose(achieved, 15, rtol=2e-7)


def test_adaptive_affinity_gradient_and_force_sum():
    rng = np.random.default_rng(8)
    P = core.perplexity_affinities(rng.normal(size=(12, 4)), 5)[0]
    Y = rng.normal(size=(12, 2))
    _, gradient = core.loss_and_gradient(Y, P)
    attraction, repulsion = core.descent_forces(Y, P)
    np.testing.assert_allclose(attraction + repulsion, -gradient, atol=1e-15)
    numerical = np.zeros_like(Y)
    for index in np.ndindex(Y.shape):
        plus, minus = Y.copy(), Y.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (core.loss_and_gradient(plus, P)[0] - core.loss_and_gradient(minus, P)[0]) / 2e-6
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=3e-9)


def test_saved_positions_are_actual_accepted_gradient_updates():
    X = np.random.default_rng(2).normal(size=(20, 4))
    trace = core.optimization_trace(X, perplexity=5, steps=12, stride=1, learning_rate=1000)
    for i, rate in enumerate(trace["accepted_rates"]):
        Y = trace["positions"][i]
        gradient = core.loss_and_gradient(Y, trace["P"])[1]
        expected = Y - rate * gradient
        expected -= expected.mean(axis=0)
        np.testing.assert_allclose(trace["positions"][i + 1], expected, atol=1e-14)
    assert np.all(np.diff(trace["full_losses"]) <= 1e-14)
    assert trace["losses"][-1] < trace["losses"][0]
    np.testing.assert_allclose(trace["positions"].mean(axis=1), 0, atol=1e-15)
    assert np.max(trace["accepted_rates"]) <= 1000


@pytest.mark.parametrize("target", [0, 1, 71, 80, np.nan])
def test_invalid_perplexity_rejected(target):
    with pytest.raises(ValueError):
        core.match_perplexity(np.arange(71), target)


def test_unattainable_perplexity_and_oversized_dense_problem_rejected():
    with pytest.raises(ValueError, match="unattainable"):
        core.match_perplexity(np.ones(20), 5)
    with pytest.raises(ValueError, match="100"):
        core.perplexity_affinities(np.ones((101, 3)))


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"stride": 0}, {"learning_rate": 0}])
def test_invalid_descent_controls(kwargs):
    with pytest.raises(ValueError):
        core.optimization_trace(np.ones((20, 3)), **kwargs)


@pytest.mark.parametrize("k", [5, 15])
def test_umap_fuzzy_union_matches_the_actual_fitted_graph(k):
    stages = core.umap_graph_stages(k, epochs=30)
    directed = stages["directed"]
    expected = directed + directed.T - directed * directed.T
    np.testing.assert_allclose(stages["graph"], expected, atol=2e-6)
    np.testing.assert_array_equal(stages["indices"][:, 0], np.arange(75))
    np.testing.assert_array_equal(np.diag(stages["graph"]), 0)
    assert stages["graph"].min() >= 0
    assert stages["graph"].max() <= 1 + 1e-7
    assert stages["Y"].shape == (75, 2)
    assert np.isfinite(stages["Y"]).all()


def test_cache_uses_input_parameters_and_retains_original_timing(tmp_path):
    X = np.random.default_rng(8).normal(size=(80, 5))
    Y, first = core.fit_embedding(X, "PCA", tmp_path)
    repeated, cached = core.fit_embedding(X, "PCA", tmp_path)
    np.testing.assert_array_equal(Y, repeated)
    assert not first["cache_hit"] and cached["cache_hit"]
    assert first["fit_seconds"] == cached["fit_seconds"]
    core.fit_embedding(X, "PCA", tmp_path, n_components=3)
    core.fit_embedding(X + 1, "PCA", tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 3


@pytest.mark.parametrize("method", ["PCA", "t-SNE", "UMAP"])
def test_real_three_dimensional_fits(method, tmp_path):
    X, labels, _ = core.digits_sample(110)
    Y, info = core.fit_embedding(X, method, tmp_path, n_components=3, iterations=350, epochs=30)
    assert Y.shape == (110, 3)
    assert np.isfinite(Y).all()
    assert np.linalg.matrix_rank(Y - Y.mean(axis=0)) == 3
    assert info["parameters"]["n_components"] == 3
    assert info["fit_seconds"] > 0
    scores = core.evaluate(X, Y)
    assert all(0 <= score <= 1 for score in scores.values())


def test_metrics_and_selected_neighbor_categories():
    X = np.random.default_rng(4).normal(size=(80, 2))
    scores = core.evaluate(X, X)
    assert all(score == pytest.approx(1) for score in scores.values())
    A = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])
    B = np.array([[2, 3], [2, 3], [1, 3], [1, 2]])
    colors, recall = scenes.neighbor_categories(A, B, 0, 4)
    np.testing.assert_array_equal(colors, [4, 1, 3, 2])
    assert recall == 0.5


def test_group_geometry_is_translation_and_rotation_invariant():
    X, labels = core.geometry_sample()
    rotated = X @ np.array([[0, -1], [1, 0]]) + 100
    a, b = core.group_geometry(X, labels), core.group_geometry(rotated, labels)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-12)
    # Library embeddings are float32; public run records must still be JSON-safe.
    assert json.loads(json.dumps(core.group_geometry(X.astype(np.float32), labels)))


def test_temporary_gif_png_and_offline_html_exports(tmp_path):
    scenes.prepare(tmp_path)
    kernels = scenes.kernels(tmp_path, {})
    result = scenes.optimization(tmp_path, {"toy_steps": 20})
    with Image.open(tmp_path / result["files"][0]) as gif:
        assert gif.n_frames >= 3
        gif.seek(gif.n_frames - 1)
        gif.load()
    for filename in (kernels["files"][0], result["files"][1]):
        with Image.open(tmp_path / filename) as png:
            assert png.width >= 700 and png.height >= 350
            png.load()
    fig = scenes.go.Figure(scenes.go.Scatter(x=[0, 1], y=[1, 0]))
    scenes.plot_page(tmp_path, "test.html", "Test", "Concept", "p sums to 1", fig,
                     "seed=42", "Inspect", "Limitation", "Takeaway")
    text = (tmp_path / "interactive/test.html").read_text(encoding="utf-8")
    assert 'src="../plotly.min.js"' in text
    assert 'src="https://' not in text
    assert "Limitation" in text and "Takeaway" in text
    scenes.write_index(tmp_path, {"optimization": result, "kernels": kernels})
    assert (tmp_path / "index.html").is_file()


def test_cli_exposes_the_documented_groups(capsys):
    spec = importlib.util.spec_from_file_location("day40_visual_cli", TOPIC / "visualize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--list"]) == 0
    output = capsys.readouterr().out
    assert all(flag in output for flag in ("--quick", "--interactive", "--animations", "--experiments", "--all"))
