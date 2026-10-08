"""Numerical and CLI smoke checks for the Random Forest visual lab."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "visualizations.py"
SPEC = importlib.util.spec_from_file_location("day27_visualizations", MODULE_PATH)
visual = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(visual)


def test_bootstrap_membership_matches_draws():
    membership, samples = visual.bootstrap_membership(12, 5, seed=4)
    assert membership.shape == (12, 5)
    for tree, indices in enumerate(samples):
        assert len(indices) == 12
        np.testing.assert_array_equal(np.flatnonzero(membership[:, tree]), np.unique(indices))
    with pytest.raises(ValueError):
        visual.bootstrap_membership(1, 2)


def test_unique_fraction_simulation_is_repeatable_and_near_expectation():
    sizes, means, bounds = visual.simulate_bootstrap_unique_fraction(
        sizes=(20, 100), repeats=600, seed=8
    )
    _, again, _ = visual.simulate_bootstrap_unique_fraction(sizes=(20, 100), repeats=600, seed=8)
    np.testing.assert_array_equal(means, again)
    expected = 1 - (1 - 1 / sizes) ** sizes
    np.testing.assert_allclose(means, expected, atol=0.02)
    assert np.all(bounds[:, 0] <= means)
    assert np.all(means <= bounds[:, 1])
    with pytest.raises(ValueError):
        visual.simulate_bootstrap_unique_fraction(repeats=0)


def test_variance_curve_and_pairwise_agreement():
    counts = np.array([1, 2, 10])
    np.testing.assert_allclose(visual.variance_curve(counts, 0), 1 / counts)
    np.testing.assert_array_equal(visual.variance_curve(counts, 1), np.ones(3))
    assert visual.pairwise_agreement(np.array([[0, 1], [0, 1], [1, 1]])) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        visual.variance_curve([0], 0.5)
    with pytest.raises(ValueError):
        visual.variance_curve([1], 1.2)
    with pytest.raises(ValueError):
        visual.pairwise_agreement(np.array([[0, 1]]))


def test_one_view_cli_writes_only_requested_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(visual, "OUTPUT", tmp_path)
    paths = visual.main(["--only", "07_correlation_variance"])
    assert [path.name for path in paths] == ["07_correlation_variance.png"]
    assert paths[0].is_file()
    assert paths[0].stat().st_size > 1_000
    assert len(visual.ALL) == 16
