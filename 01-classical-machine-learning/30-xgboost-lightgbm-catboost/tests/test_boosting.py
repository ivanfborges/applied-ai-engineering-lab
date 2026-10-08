"""Check the numerical objective, split constraints, and validation boundaries."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import fit_newton_stump, leaf_weight, logistic_derivatives, sigmoid, split_gain
from example import lightgbm_frames, make_data, split_indices


def test_logistic_derivatives_against_finite_differences():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    z = np.array([-2.0, -0.4, 0.8, 2.1])
    step = 1e-4
    loss = lambda scores: np.logaddexp(0.0, scores) - y * scores
    g, h = logistic_derivatives(y, z)
    np.testing.assert_allclose(g, (loss(z + step) - loss(z - step)) / (2 * step), atol=1e-8)
    np.testing.assert_allclose(h, (loss(z + step) - 2 * loss(z) + loss(z - step)) / step**2, atol=1e-7)


def test_extreme_logits_are_finite_without_overflow():
    with np.errstate(over="raise", invalid="raise"):
        p = sigmoid([-1000.0, 0.0, 1000.0])
        g, h = logistic_derivatives([1, 0, 0], [-1000.0, 0.0, 1000.0])
    np.testing.assert_allclose(p, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(g, [-1.0, 0.5, 1.0])
    np.testing.assert_allclose(h, [0.0, 0.25, 0.0])
    assert logistic_derivatives([1], [40])[1][0] > 0


def test_leaf_weight_minimizes_quadratic_objective():
    gradient, hessian, penalty = -2.7, 1.3, 2.0
    weight = leaf_weight(gradient, hessian, penalty)
    objective = lambda w: gradient * w + 0.5 * (hessian + penalty) * w**2
    assert gradient + (hessian + penalty) * weight == pytest.approx(0.0)
    assert objective(weight) < objective(weight - 0.1)
    assert objective(weight) < objective(weight + 0.1)
    assert abs(leaf_weight(gradient, hessian, 10)) < abs(weight)


def test_gain_matches_direct_regularized_objective_difference():
    gl, hl, gr, hr, penalty, gamma = 2.0, 1.0, -1.0, 0.5, 1.0, 0.2
    objective = lambda g, h, w: g * w + 0.5 * (h + penalty) * w**2
    parent = objective(gl + gr, hl + hr, leaf_weight(gl + gr, hl + hr, penalty)) + gamma
    children = (objective(gl, hl, leaf_weight(gl, hl, penalty))
                + objective(gr, hr, leaf_weight(gr, hr, penalty)) + 2 * gamma)
    assert split_gain(gl, hl, gr, hr, penalty, gamma) == pytest.approx(parent - children)


def test_stump_selects_best_partition_against_brute_force_oracle():
    x = np.array([3.0, 0.0, 1.0, 1.0, 2.0, 4.0])
    y = np.array([1, 0, 0, 1, 1, 1])
    z = np.array([-0.2, 0.0, 0.4, -0.3, 0.7, -0.1])
    g, h = logistic_derivatives(y, z)
    scores = []
    for threshold in np.unique(x)[:-1]:
        mask = x <= threshold
        def minimum_objective(part):
            G, H = g[part].sum(), h[part].sum()
            w = -G / (H + 1.0)
            return G * w + 0.5 * (H + 1.0) * w**2
        whole = np.ones(len(x), dtype=bool)
        gain = minimum_objective(whole) - minimum_objective(mask) - minimum_objective(~mask)
        scores.append((gain, threshold))
    expected_gain, expected_threshold = max(scores, key=lambda pair: pair[0])
    stump = fit_newton_stump(x, y, z)
    assert stump.threshold == expected_threshold
    assert stump.gain == pytest.approx(expected_gain)
    permutation = np.array([4, 2, 0, 5, 3, 1])
    assert fit_newton_stump(x[permutation], y[permutation], z[permutation]) == stump


@pytest.mark.parametrize("kwargs", [{"gamma": 100.0}, {"min_child_weight": 2.0}])
def test_constraints_prevent_split_but_retain_parent_update(kwargs):
    x, y = [0, 1, 2, 3], [0, 0, 1, 1]
    stump = fit_newton_stump(x, y, np.zeros(4), **kwargs)
    assert stump.threshold is None
    assert stump.gain == 0.0
    np.testing.assert_array_equal(stump.predict(x), np.zeros(4))


def test_constant_feature_returns_regularized_parent_leaf():
    stump = fit_newton_stump([2, 2, 2], [1, 1, 1], [0, 0, 0])
    assert stump.threshold is None
    np.testing.assert_allclose(stump.predict([1, 2, 3]), np.full(3, 1.5 / 1.75))


@pytest.mark.parametrize("x,y,z,kwargs", [
    ([], [], [], {}), ([np.nan], [0], [0], {}), ([0, 1], [0], [0], {}),
    ([0], [2], [0], {}), ([0], [0], [np.inf], {}),
    ([0], [0], [0], {"gamma": -1}), ([0], [0], [0], {"reg_lambda": -1}),
    ([0], [0], [0], {"min_child_weight": -1}),
    ([0], [0], [1000], {"reg_lambda": 0}),
])
def test_invalid_stump_inputs_raise_explicit_errors(x, y, z, kwargs):
    with pytest.raises(ValueError):
        fit_newton_stump(x, y, z, **kwargs)


def test_split_indices_are_disjoint_complete_and_reproducible():
    _, y, _ = make_data("numeric")
    partitions = split_indices(y)
    all_rows = np.concatenate(list(partitions.values()))
    np.testing.assert_array_equal(np.sort(all_rows), np.arange(len(y)))
    assert len(np.unique(all_rows)) == len(y)
    assert {name: len(rows) for name, rows in partitions.items()} == {
        "train": 1440, "validation": 480, "test": 480}
    for name, rows in partitions.items():
        np.testing.assert_array_equal(rows, split_indices(y)[name])
        assert set(y[rows]) == {0, 1}


def test_category_vocabulary_comes_only_from_training_and_inputs_are_unchanged():
    frames = {"train": pd.DataFrame({"merchant": ["b", "a"]}),
              "validation": pd.DataFrame({"merchant": ["a", "unseen"]}),
              "test": pd.DataFrame({"merchant": ["b", "another"]})}
    result = lightgbm_frames(frames, ["merchant"])
    assert list(result["train"]["merchant"].cat.categories) == ["a", "b"]
    assert result["validation"]["merchant"].cat.codes.tolist() == [0, -1]
    assert result["test"]["merchant"].cat.codes.tolist() == [1, -1]
    assert frames["validation"]["merchant"].tolist() == ["a", "unseen"]


def test_one_tree_agrees_with_xgboost_when_objectives_and_base_margin_match():
    xgb = pytest.importorskip("xgboost", reason="requires optional boosting extra")
    x = np.arange(8.0)
    y = np.array([0] * 4 + [1] * 4)
    stump = fit_newton_stump(x, y, np.zeros(8), reg_lambda=1.0)
    model = xgb.XGBClassifier(n_estimators=1, max_depth=1, learning_rate=1.0,
        objective="binary:logistic", tree_method="exact", base_score=0.5,
        min_child_weight=0.0, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
        subsample=1.0, colsample_bytree=1.0, n_jobs=1, random_state=42)
    model.fit(x.reshape(-1, 1), y)
    np.testing.assert_allclose(model.predict(x.reshape(-1, 1), output_margin=True),
                               stump.predict(x), atol=1e-6)
