"""Tests for the Day 15 educational numerical utilities."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


TOPIC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC_DIR))

from from_scratch import (  # noqa: E402
    categorical_nll_from_logits,
    cross_entropy,
    entropy,
    kl_divergence,
    log_softmax,
    softmax,
    validate_distribution,
)


def test_entropy_extremes_use_natural_logarithms() -> None:
    assert entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert entropy(np.full(3, 1.0 / 3.0)) == pytest.approx(math.log(3.0))


def test_cross_entropy_decomposes_into_entropy_and_kl() -> None:
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.6, 0.3, 0.1])
    assert cross_entropy(p, q) == pytest.approx(entropy(p) + kl_divergence(p, q))
    assert kl_divergence(p, q) >= 0.0
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-15)


def test_kl_is_asymmetric_for_distinct_distributions() -> None:
    p = np.array([0.8, 0.1, 0.1])
    q = np.array([0.4, 0.3, 0.3])
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p))


def test_support_mismatch_has_infinite_cost_without_clipping() -> None:
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([1.0, 0.0, 0.0])
    assert math.isinf(cross_entropy(p, q))
    assert math.isinf(kl_divergence(p, q))


@pytest.mark.parametrize(
    "distribution",
    [
        np.array([]),
        np.array([[0.5, 0.5]]),
        np.array([0.6, 0.6]),
        np.array([1.1, -0.1]),
        np.array([np.nan, 1.0]),
    ],
)
def test_distribution_validation_rejects_invalid_inputs(distribution: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_distribution(distribution)


def test_log_softmax_is_stable_and_shift_invariant() -> None:
    logits = np.array([[1_000.0, 999.0, 998.0], [-1_000.0, -999.0, -998.0]])
    probabilities = softmax(logits)
    assert np.all(np.isfinite(log_softmax(logits)))
    assert probabilities.sum(axis=-1) == pytest.approx(np.ones(2))
    assert probabilities == pytest.approx(softmax(logits + 50_000.0))


def test_categorical_nll_matches_true_class_probability() -> None:
    logits = np.array([[2.0, 1.0, 0.1]])
    target = np.array([0])
    expected = -math.log(softmax(logits)[0, 0])
    assert categorical_nll_from_logits(logits, target) == pytest.approx(expected)


def test_masked_token_mean_excludes_padding() -> None:
    logits = np.array(
        [
            [[2.0, 0.0], [0.0, 2.0], [100.0, -100.0]],
            [[1.0, 0.0], [0.0, 1.0], [-100.0, 100.0]],
        ]
    )
    targets = np.array([[0, 1, 1], [0, 1, 0]])
    mask = np.array([[True, True, False], [True, True, False]])
    per_position = categorical_nll_from_logits(
        logits, targets, mask=mask, reduction="none"
    )
    mean_loss = categorical_nll_from_logits(
        logits, targets, mask=mask, reduction="mean"
    )
    assert np.all(per_position[~mask] == 0.0)
    assert mean_loss == pytest.approx(per_position[mask].mean())


@pytest.mark.parametrize(
    ("targets", "mask", "error"),
    [
        (np.array([0.0]), None, TypeError),
        (np.array([3]), None, ValueError),
        (np.array([[0]]), None, ValueError),
        (np.array([0]), np.array([1]), TypeError),
        (np.array([0]), np.array([False]), ValueError),
    ],
)
def test_categorical_nll_rejects_invalid_targets_or_masks(
    targets: np.ndarray, mask: np.ndarray | None, error: type[Exception]
) -> None:
    logits = np.array([[2.0, 1.0, 0.0]])
    with pytest.raises(error):
        categorical_nll_from_logits(logits, targets, mask=mask)
