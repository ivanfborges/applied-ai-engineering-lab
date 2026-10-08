"""Numerical checks and lightweight rendering checks for the ExtraTrees visual lab."""

import importlib.util
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

TOPIC = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("day28_visuals", TOPIC / "visualize_extratrees.py")
visuals = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(visuals)


def test_gini_and_gain_by_hand():
    assert visuals.gini_impurity([0, 0, 0, 1]) == 0.375
    assert visuals.split_gain([0, 1, 2, 3], [0, 0, 1, 1], 1.5) == 0.5
    assert visuals.split_gain([0, 1, 2, 3], [0, 1, 1, 0], 1.5) == 0
    assert np.isnan(visuals.split_gain([0, 1], [0, 1], 2))
    assert np.isnan(visuals.split_gain([0, 1], [0, 1], -1))


@pytest.mark.parametrize("labels", [[], [0, 2], [0, np.nan], [[0], [1]]])
def test_invalid_labels(labels):
    with pytest.raises(ValueError):
        visuals.gini_impurity(labels)


def test_invalid_split_data():
    with pytest.raises(ValueError):
        visuals.split_gain([0], [0, 1], 0.5)
    with pytest.raises(ValueError):
        visuals.split_gain([0, np.inf], [0, 1], 0.5)


def test_variance_identities():
    np.testing.assert_allclose(visuals.ensemble_variance([1, 10, 100], 0), [1, 0.1, 0.01])
    np.testing.assert_allclose(visuals.ensemble_variance([1, 10, 100], 1), 1)
    assert visuals.ensemble_variance(100, 0.5) == 0.505
    assert visuals.ensemble_variance(10, 0.5, sigma_squared=2) == 1.1
    M, rho = np.meshgrid([1, 10, 100], [0, 0.5, 1])
    result = visuals.ensemble_variance(M, rho)
    assert result.shape == (3, 3)
    assert np.all(result >= rho)
    assert np.all(np.diff(result, axis=1) <= 0)


@pytest.mark.parametrize("M,rho,sigma", [
    (0, 0.5, 1), (1.5, 0.5, 1), (2, -0.1, 1),
    (2, 1.1, 1), (2, np.nan, 1), (2, 0.5, -1),
])
def test_invalid_variance_inputs(M, rho, sigma):
    with pytest.raises(ValueError):
        visuals.ensemble_variance(M, rho, sigma)


def test_diversity_matches_pairwise_definitions():
    predictions = np.array([[0, 0, 1, 1], [0, 1, 1, 0], [1, 0, 1, 0]], dtype=float)
    result = visuals.tree_diversity(predictions)
    expected = np.mean([np.mean(a != b) for a, b in combinations(predictions, 2)])
    assert result["mean_disagreement"] == pytest.approx(expected)
    np.testing.assert_allclose(result["correlation_matrix"], np.corrcoef(predictions), atol=1e-15)
    np.testing.assert_allclose(result["disagreement_matrix"].diagonal(), 0)
    assert result["defined_correlation_pairs"] == 3
    complemented = visuals.tree_diversity(1 - predictions)
    np.testing.assert_allclose(result["disagreement_matrix"], complemented["disagreement_matrix"])


def test_constant_predictions_keep_correlations_undefined():
    result = visuals.tree_diversity([[0, 0, 0], [1, 1, 1]])
    assert result["mean_disagreement"] == 1
    assert result["mean_correlation"] is None
    assert result["defined_correlation_pairs"] == 0
    assert np.isnan(result["correlation_matrix"]).all()
    mixed = visuals.tree_diversity([[0, 0, 0], [0, 1, 0], [1, 0, 1]])
    assert mixed["defined_correlation_pairs"] == 1
    assert mixed["mean_correlation"] == pytest.approx(-1)


@pytest.mark.parametrize("predictions", [
    [[0, 1]], [[0], [1]], [[0, 2], [0, 1]], [[0, np.nan], [0, 1]],
])
def test_invalid_diversity(predictions):
    with pytest.raises(ValueError):
        visuals.tree_diversity(predictions)


def test_nested_averaging():
    probabilities = np.array([[0.1, 0.9], [0.5, 0.3], [0.9, 0.3]])
    result = visuals.cumulative_probabilities(probabilities, [1, 2, 3])
    np.testing.assert_allclose(result, [[0.1, 0.9], [0.3, 0.6], [0.5, 0.5]])


@pytest.mark.parametrize("counts", [[0], [4], [1.5], []])
def test_invalid_prefix_counts(counts):
    with pytest.raises(ValueError):
        visuals.cumulative_probabilities([[0.1], [0.5], [0.9]], counts)


def test_data_and_model_controls():
    first, second = visuals.make_datasets(), visuals.make_datasets()
    for name in first:
        for a, b in zip(first[name], second[name]):
            np.testing.assert_array_equal(a, b)
    assert first["moons"][0].shape == (280, 2)
    assert first["moons"][1].shape == (120, 2)
    assert first["tabular"][0].shape == (700, 10)
    for family in ["RF", "ET"]:
        model = visuals.make_model(family, n_estimators=5, max_features=2)
        assert model.n_estimators == 5
        assert model.max_features == 2
        assert model.bootstrap is False
        assert model.estimator.splitter == ("best" if family == "RF" else "random")


def test_static_render_and_closed_figures(tmp_path):
    path, records = visuals.plot_split_search(visuals.make_datasets()["split"], tmp_path, dpi=80)
    assert len(records) == 2
    with Image.open(tmp_path / path) as image:
        assert image.width > 500 and image.height > 300
        assert image.format == "PNG"
    assert not plt.get_fignums()


def test_cli_help():
    process = subprocess.run([sys.executable, str(TOPIC / "visualize_extratrees.py"), "--help"],
                             capture_output=True, text=True, check=True)
    assert "--output-dir" in process.stdout
    assert "--skip-animations" in process.stdout
