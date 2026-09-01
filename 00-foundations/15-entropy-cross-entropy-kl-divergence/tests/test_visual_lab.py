"""Numerical and smoke tests for the Day 15 visual generator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


TOPIC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIR))

from from_scratch import entropy  # noqa: E402
from visual_lab import (  # noqa: E402
    ASSET_GENERATORS,
    binary_entropy,
    continuous_kl,
    distribution_path,
    expected_asset_names,
    mixture_density,
    normal_density,
    plot_self_information,
    selected_generators,
    simplex_grid,
    stable_softmax,
)


def test_binary_entropy_endpoints_and_maximum() -> None:
    values = binary_entropy(np.array([0.0, 0.5, 1.0]))
    assert values[[0, 2]] == pytest.approx(np.zeros(2))
    assert values[1] == pytest.approx(np.log(2.0))


def test_temperature_flattens_fixed_logits_and_increases_entropy() -> None:
    logits = np.array([3.0, 1.0, 0.0])
    cold = stable_softmax(logits, temperature=0.2)
    hot = stable_softmax(logits, temperature=5.0)
    assert cold.sum() == pytest.approx(1.0)
    assert hot.sum() == pytest.approx(1.0)
    assert cold[0] > hot[0]
    assert entropy(cold) < entropy(hot)


def test_distribution_path_preserves_endpoints_and_normalization() -> None:
    start = np.array([1.0, 0.0, 0.0])
    end = np.full(3, 1.0 / 3.0)
    path = distribution_path(start, end, steps=9)
    assert path[0] == pytest.approx(start)
    assert path[-1] == pytest.approx(end)
    assert path.sum(axis=1) == pytest.approx(np.ones(9))
    assert np.all(path >= 0.0)


def test_simplex_grid_is_strictly_inside_probability_simplex() -> None:
    points = simplex_grid(resolution=12)
    assert points.shape[1] == 3
    assert np.all(points > 0.0)
    assert points.sum(axis=1) == pytest.approx(np.ones(len(points)))


def test_continuous_kl_examples_are_finite_and_nonnegative() -> None:
    x = np.linspace(-6.0, 6.0, 2400)
    target = mixture_density(x)
    approximation = normal_density(x, mean=0.0, std=2.1)
    value = continuous_kl(x, target, approximation)
    assert np.isfinite(value)
    assert value >= 0.0


def test_asset_manifest_and_cli_groups_are_complete() -> None:
    expected = expected_asset_names()
    assert len(expected) == 18
    assert len(set(expected)) == 18
    assert len(ASSET_GENERATORS) == 18
    assert len(selected_generators("entropy")) == 5
    assert len(selected_generators("kl")) == 5
    assert len(selected_generators("softmax")) == 4
    assert len(selected_generators("llm")) == 4
    assert len(selected_generators("all")) == 18


def test_static_plot_smoke_writes_a_decodable_png(tmp_path: Path) -> None:
    output = plot_self_information(tmp_path)
    assert output == tmp_path / "self_information.png"
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.inf, np.nan])
def test_stable_softmax_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError):
        stable_softmax(np.array([1.0, 0.0]), temperature=temperature)
