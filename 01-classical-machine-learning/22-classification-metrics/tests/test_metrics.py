"""Check binary metric semantics and library parity."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


spec = importlib.util.spec_from_file_location(
    "day22_metrics", Path(__file__).resolve().parents[1] / "from_scratch.py"
)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


def test_counts_and_metric_denominators():
    actual = [1, 1, 1, 0, 0, 0]
    predicted = [1, 0, 1, 1, 0, 0]
    result = core.classification_metrics(actual, predicted)
    assert [result[key] for key in ("tn", "fp", "fn", "tp")] == [2, 1, 1, 2]
    assert sum(result[key] for key in ("tn", "fp", "fn", "tp")) == len(actual)
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "actual,predicted",
    [
        ([0, 0, 1, 1], [0, 0, 0, 0]),
        ([0, 0, 0], [0, 0, 0]),
        ([1, 1, 1], [1, 1, 1]),
        ([0, 1, 0, 1], [1, 1, 1, 1]),
    ],
)
def test_sklearn_parity_including_undefined_rates(actual, predicted):
    result = core.classification_metrics(actual, predicted)
    expected = confusion_matrix(actual, predicted, labels=[0, 1])
    np.testing.assert_array_equal(
        expected, [[result["tn"], result["fp"]], [result["fn"], result["tp"]]]
    )
    assert result["accuracy"] == pytest.approx(accuracy_score(actual, predicted))
    assert result["precision"] == pytest.approx(
        precision_score(actual, predicted, zero_division=0)
    )
    assert result["recall"] == pytest.approx(
        recall_score(actual, predicted, zero_division=0)
    )
    assert result["f1"] == pytest.approx(f1_score(actual, predicted, zero_division=0))


@pytest.mark.parametrize(
    "actual,predicted",
    [
        ([], []),
        ([0], []),
        ([[0, 1]], [[0, 1]]),
        ([0, 2], [0, 1]),
        ([0, 1], [0, np.nan]),
    ],
)
def test_invalid_inputs_fail_explicitly(actual, predicted):
    with pytest.raises(ValueError):
        core.classification_metrics(actual, predicted)
