"""Check actual training states, information boundaries and visual artifact integrity."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.decomposition import PCA

VISUALS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISUALS))
import visual_core as core

spec = importlib.util.spec_from_file_location("day43_visual_renderer", VISUALS / "visualizations.py")
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)


@pytest.fixture(scope="module")
def lab():
    return core.build_lab(factors=3, epochs=2, frames=8)


def test_instrumented_training_matches_the_existing_study_optimizer():
    matrix = np.array([[1, 0, 1, 0], [0, 1, 0, 0]])
    trace = core.trace_bpr(matrix, factors=3, epochs=3, seed=43, frames=8)
    p, q = core.fit_bpr(matrix, factors=3, epochs=3, seed=43)
    np.testing.assert_allclose(trace["p"], p, atol=1e-14)
    np.testing.assert_allclose(trace["q"], q, atol=1e-14)
    assert trace["history"][0]["epoch"] == 0
    assert trace["history"][-1]["epoch"] == 3
    for snapshot in trace["steps"]:
        u, i, j = snapshot["user"], snapshot["positive"], snapshot["unobserved"]
        assert matrix[u, i] == 1 and matrix[u, j] == 0
        old = snapshot["before"]
        g = 1 / (1 + np.exp(old[0] @ (old[1] - old[2])))
        expected = np.array([old[0] + .05 * (g * (old[1] - old[2]) - .01 * old[0]),
                             old[1] + .05 * (g * old[0] - .01 * old[1]),
                             old[2] + .05 * (-g * old[0] - .01 * old[2])])
        np.testing.assert_allclose(snapshot["after"], expected)
        after = snapshot["after"]
        np.testing.assert_allclose(snapshot["after_scores"], [after[0] @ after[1], after[0] @ after[2]])


def test_epoch_losses_are_computed_from_the_recorded_factors(lab):
    for history, state in zip(lab.trace["history"], lab.trace["states"]):
        scores = state["p"] @ state["q"].T
        losses = []
        for u, row in enumerate(lab.matrix):
            positive = np.flatnonzero(row == 1)
            unobserved = np.flatnonzero(row == 0)
            losses.extend(np.logaddexp(0, -(scores[u, positive, None] - scores[u, unobserved][None, :])).ravel())
        assert history["pair_loss"] == pytest.approx(np.mean(losses))


def test_shared_projection_fits_one_basis_to_both_populations():
    rng = np.random.default_rng(43)
    p, q = rng.normal(size=(8, 6)), rng.normal(size=(12, 6))
    pp, qq, meta = core.shared_projection(p, q)
    expected = PCA(3, svd_solver="full").fit_transform(np.vstack([p, q]))
    np.testing.assert_allclose(np.vstack([pp, qq]), expected, atol=1e-12)
    assert 0 < meta["variance_retained"] <= 1


@pytest.mark.parametrize("dimensions", [2, 3])
def test_low_dimensional_projection_preserves_dot_products(dimensions):
    rng = np.random.default_rng(43)
    p, q = rng.normal(size=(4, dimensions)), rng.normal(size=(5, dimensions))
    pp, qq, _ = core.shared_projection(p, q)
    np.testing.assert_allclose(pp @ qq.T, p @ q.T)


def test_changing_only_test_targets_cannot_change_fitted_scores(lab, monkeypatch):
    original = core.synthetic_history
    def different_targets(seed):
        sequences, features = original(seed)
        for u, row in enumerate(sequences):
            allowed = np.setdiff1d(np.arange(len(features)), row)
            sequences[u, 15] = allowed[0]
        return sequences, features
    monkeypatch.setattr(core, "synthetic_history", different_targets)
    other = core.build_lab(factors=3, epochs=2, frames=8)
    assert not np.array_equal(lab.targets, other.targets)
    np.testing.assert_array_equal(lab.matrix, other.matrix)
    for name in lab.scores:
        np.testing.assert_array_equal(lab.scores[name], other.scores[name])
    np.testing.assert_array_equal(lab.hybrid(), other.hybrid())


def test_svd_reconstruction_and_content_profiles_have_no_test_pairs(lab):
    assert (lab.matrix[np.arange(80), lab.targets] == 0).all()
    np.testing.assert_allclose(lab.scores["SVD"], lab.svd_p @ lab.svd_q.T)
    np.testing.assert_allclose(lab.scores["BPR"], lab.trace["p"] @ lab.trace["q"].T)
    np.testing.assert_allclose(lab.scores["Content"], core.content_scores(lab.matrix, lab.features))


@pytest.mark.parametrize("k", [1, 5, 10, 20])
def test_retrieval_ceiling_seen_filtering_and_computed_metrics(lab, k):
    rows = core.candidate_analysis(lab, k=k)
    assert np.all(np.diff([r["candidate_recall"] for r in rows]) >= 0)
    for row in rows:
        assert row["metrics"]["ndcg"] <= row["metrics"]["recall"] <= row["candidate_recall"]
        assert row["mean_candidates"] <= 2 * row["budget"]
        for u, (pool, ranked) in enumerate(zip(row["pools"], row["rankings"])):
            assert not lab.seen[u, pool].any()
            assert not lab.seen[u, ranked].any()
            assert set(ranked) <= set(pool)
        expected = core.single_positive_metrics(row["rankings"], lab.targets, k)
        assert expected == row["metrics"]


def test_hybrid_endpoints_and_input_errors(lab):
    np.testing.assert_array_equal(lab.hybrid(0), lab.content_z)
    np.testing.assert_array_equal(lab.hybrid(1), lab.cf_z)
    with pytest.raises(ValueError):
        lab.hybrid(1.1)
    with pytest.raises(ValueError):
        core.build_lab(factors=1)


def test_objective_losses_have_the_expected_arithmetic_and_shift_properties():
    scores, labels = np.array([.3, 1.2, .8, -.5, .1]), np.array([1, 0, 1, 0, 0])
    result = core.objective_losses(scores, labels)
    shifted = core.objective_losses(scores + 4, labels)
    assert result["pointwise"] != pytest.approx(shifted["pointwise"])
    assert result["pairwise"] == pytest.approx(shifted["pairwise"])
    assert result["listwise"] == pytest.approx(shifted["listwise"])
    expected = np.mean([np.log1p(np.exp(-(scores[i] - scores[j])))
                        for i in [0, 2] for j in [1, 3, 4]])
    assert result["pairwise"] == pytest.approx(expected)


def test_relevant_item_at_rank_two_has_larger_ndcg_than_rank_eight():
    late, early = np.zeros(10), np.zeros(10)
    late[7] = early[1] = 1
    old, new = core.ranking_contributions(late), core.ranking_contributions(early)
    assert old["ndcg"][-1] == pytest.approx(1 / np.log2(9))
    assert new["ndcg"][-1] == pytest.approx(1 / np.log2(3))
    assert new["ndcg"][-1] > old["ndcg"][-1]


def test_cold_start_has_zero_history_and_real_content_candidate_source(lab):
    rows = core.cold_states(lab, steps=(0, 3, 15))
    np.testing.assert_array_equal(rows[0]["cf"], 0)
    np.testing.assert_array_equal(rows[0]["content"], 0)
    for row in rows:
        assert row["cf"][-1] == 0
        assert not row["seen"][row["top"]].any()
        assert set(row["content_candidates"]) <= set(row["pool"])
        assert row["new_item_in_pool"] == (row["new_item"] in row["pool"])
        assert np.isfinite(row["hybrid"]).all()


def test_exposure_clicks_and_hidden_preferences_are_separate_and_deterministic():
    states = core.exposure_simulation(rounds=4)
    repeated = core.exposure_simulation(rounds=4)
    for index, (state, other) in enumerate(zip(states, repeated)):
        np.testing.assert_array_equal(state["exposure"], other["exposure"])
        np.testing.assert_array_equal(state["true_ctr"], states[0]["true_ctr"])
        assert (state["clicks"] <= state["exposure"]).all()
        if index:
            old = states[index - 1]
            assert (state["exposure"] - old["exposure"]).sum() == 240
            assert (state["clicks"] - old["clicks"] >= 0).all()
            assert (state["clicks"] - old["clicks"] <= state["exposure"] - old["exposure"]).all()


def test_rendering_smoke_uses_temporary_files_and_gif_changes(lab, tmp_path):
    output = render.prepare_output(tmp_path)
    paths = render.metrics_scene(lab, output, 8)
    paths += render.objectives_scene(lab, output, 8)
    paths += render.dot_product_scene(lab, output, 8)
    for path in paths:
        result = render.inspect_artifact(path)
        assert result["bytes"] > 1000
        if path.suffix == ".gif":
            assert result["frames"] == 7 and result["frame_changes"]
        if path.suffix == ".html":
            document = path.read_text(encoding="utf-8")
            assert '<script src="https://' not in document
            assert 'cdn.plot.ly' not in document[:1000]
            assert 'window.labReady' in document
