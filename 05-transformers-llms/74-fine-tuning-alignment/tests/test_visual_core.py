"""Independent numerical identities and held-out-data isolation for the visual lab."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visual_core as core


@pytest.fixture(scope="module")
def reports():
    config = core.LabConfig(42, True)
    return {name: core.compute_demo(name, config) for name in core.DEMOS}


def test_all_demo_names_have_real_experiments_and_metadata(reports):
    assert set(reports) == set(core.DEMOS) == set(core.FUNCTIONS)
    for name, report in reports.items():
        assert report["demo"] == name and report["synthetic"]
        assert report["interpretation_status"] == "author review required"
        assert report["limitation"]


@pytest.mark.parametrize("logits", [[1, 2, 3], [-1000, 0, 1000], [0, 0, 0]])
def test_softmax_normalizes_without_overflow(logits):
    p = core.softmax(logits)
    np.testing.assert_allclose(p.sum(), 1)
    assert np.isfinite(p).all() and (p >= 0).all()


@pytest.mark.parametrize("p,q", [([.4,.6],[.5,.5]), ([0,1],[.2,.8]), ([.1,.2,.7],[.1,.2,.7])])
def test_kl_matches_definition_and_is_nonnegative(p, q):
    p, q = np.array(p), np.array(q)
    active = p > 0
    expected = (p[active] * np.log(p[active] / q[active])).sum()
    assert core.kl_divergence(p, q) == pytest.approx(expected)
    assert core.kl_divergence(p, q) >= -1e-12


@pytest.mark.parametrize("p,q", [([.1,.2],[.5,.5]), ([.4,.6],[0,1]), ([-.1,1.1],[.5,.5])])
def test_kl_rejects_invalid_probability_vectors(p, q):
    with pytest.raises(ValueError):
        core.kl_divergence(p, q)


def test_landscape_adaptation_start_equals_pretraining_end(reports):
    rows = reports["landscape"]["records"]
    pre = [r for r in rows if r["stage"] == "pretraining"]
    fine = [r for r in rows if r["stage"] == "fine-tuning"]
    np.testing.assert_array_equal(pre[-1]["position"], fine[0]["position"])
    assert pre[-1]["loss"] < pre[0]["loss"]
    assert fine[-1]["loss"] < fine[0]["loss"]


def test_sft_recorded_ce_equals_negative_target_log_probability(reports):
    report = reports["sft"]
    for row in report["records"]:
        np.testing.assert_allclose(core.softmax(row["logits"]), row["probabilities"], atol=1e-7)
        assert row["loss"] == pytest.approx(-np.log(row["probabilities"][report["target"]]), abs=1e-6)
        assert np.isfinite(row["gradient"])
    token = report["token"]
    assert token["selected_sum"] == pytest.approx(np.sum(token["cross_entropy"][token["mask"]]))


def test_positive_beta_policy_matches_analytic_optimum_and_objective_improves(reports):
    for beta, report in reports["rlhf"]["runs"].items():
        objective = np.array([v["objective"] for v in report["records"]])
        assert (np.diff(objective) >= -1e-12).all()
        if float(beta) > 0:
            assert report["max_abs_optimum_error"] < .005
        for row in report["records"]:
            assert row["kl"] >= -1e-12
            assert row["objective"] == pytest.approx(row["reward"] - float(beta)*row["kl"])


def test_population_dpo_matches_shared_reward_optimum(reports):
    report = reports["comparison"]
    np.testing.assert_allclose(report["dpo"][-1]["probabilities"], report["rlhf"]["optimum"], atol=.005)
    np.testing.assert_allclose(report["dpo"][0]["probabilities"], report["reference"], atol=1e-15)


def test_dpo_neutral_loss_and_bt_probability_are_distinct_from_response_probability(reports):
    report = reports["dpo"]
    for beta, rows in report["runs"].items():
        assert rows[0]["loss"] == pytest.approx(np.log(2))
        assert rows[0]["bt_preference_probability"] == pytest.approx(.5)
        assert rows[-1]["margin"] > rows[0]["margin"]
        assert rows[-1]["loss"] < rows[0]["loss"]
        for row in rows:
            assert np.sum(row["probabilities"]) == pytest.approx(1)
            assert row["bt_preference_probability"] == pytest.approx(1 / (1 + np.exp(-float(beta) * row["margin"])))
    ref, policy = report["counterexample"].values()
    assert policy[0] < ref[0] and policy[1] < ref[1]
    assert policy[0] / policy[1] > ref[0] / ref[1]


def test_reward_model_scores_and_bt_probability_match_feature_calculations(reports):
    report = reports["reward"]
    probe = np.array(report["probe_features"])
    for records in report["noise"].values():
        row = records[-1]
        np.testing.assert_allclose(probe @ row["weights"], row["probe_scores"])
        scores = row["probe_scores"]
        assert row["probe_preference_probability"] == pytest.approx(1 / (1+np.exp(-(scores[0]-scores[1]))))
        assert np.isfinite(row["expected_test_loss"])


def test_matrix_lora_zero_initialization_rank_freezing_and_svd_bound(reports):
    for target in reports["lora"]["targets"].values():
        for rank, run in target["ranks"].items():
            first, last = run["records"][0], run["records"][-1]
            assert np.count_nonzero(first["update"]) == 0
            assert np.linalg.norm(first["A"]) > 0
            np.testing.assert_allclose(np.array(last["B"]) @ np.array(last["A"]), last["update"])
            assert run["numerical_rank"] <= int(rank)
            assert run["frozen_unchanged"]
            assert last["error"] >= run["svd_error"] - 1e-8


def test_geometry_converges_to_rank_constrained_svd_solution(reports):
    report = reports["geometry"]
    np.testing.assert_allclose(report["runs"]["1"][-1]["matrix"], report["optimal_rank1"], atol=1e-4)
    assert report["runs"]["1"][-1]["error"] > report["runs"]["2"][-1]["error"]


def test_adapter_count_includes_both_biases_and_residual_matches():
    with core.isolated_cpu(42):
        adapter = core.BottleneckAdapter(8, 2)
        x = torch.randn(3, 8)
        assert sum(p.numel() for p in adapter.parameters()) == core.parameter_counts(8,1,2)["adapter"]
        torch.testing.assert_close(adapter(x), x + adapter.up(torch.relu(adapter.down(x))))
    assert core.parameter_counts(8,1,2,4)["adapter"] == 4 * 42


@pytest.mark.parametrize("bits", [4, 8])
def test_quantization_error_reconstruction_scale_and_storage(bits):
    original = np.linspace(-.7,.8,33).reshape(3,11)
    report = core.quantize_uniform(original,bits)
    np.testing.assert_array_equal(report["reconstructed"], report["codes"] * report["scale"])
    assert report["error"] == pytest.approx(np.sqrt(np.mean((original-report["reconstructed"])**2)))
    assert abs(original-report["reconstructed"]).max() <= report["scale"] / 2 + 1e-12
    assert report["theoretical_bytes"] == (original.size*bits+7)//8+4
    assert np.max(abs(report["codes"])) <= 2**(bits-1)-1


def test_quantized_base_stays_frozen_and_error_is_measured(reports):
    report = reports["qlora"]
    assert report["base_unchanged"]
    final = np.array(report["records"][-1]["effective"])
    expected = np.sqrt(np.mean((np.array(report["original"])-final)**2))
    assert report["records"][-1]["error"] == pytest.approx(expected)


def test_hacking_proxy_and_intended_utility_are_separate(reports):
    report = reports["hacking"]
    for run in report["runs"].values():
        for row in run["records"]:
            assert row["proxy"] == pytest.approx(row["probabilities"] @ report["proxy_reward"])
            assert row["intended"] == pytest.approx(row["probabilities"] @ report["intended_utility"])
    unregularized = report["runs"]["proxy beta=0"]["records"]
    assert unregularized[-1]["proxy"] > unregularized[0]["proxy"]
    assert unregularized[-1]["intended"] < unregularized[0]["intended"]


def test_pipeline_reference_and_frozen_weights_and_neutral_dpo_start(reports):
    report = reports["pipeline"]
    assert report["final_reference_unchanged"]
    for row in report["records"]:
        assert row["base_frozen_unchanged"]
        assert row["metrics"]["kl_policy_to_sft_reference"] >= -2e-6
    initial_dpo = next(row for row in report["records"] if row["stage"] == "dpo")
    assert initial_dpo["loss"] == pytest.approx(np.log(2))
    assert initial_dpo["metrics"]["kl_policy_to_sft_reference"] == 0


def test_evaluation_data_cannot_change_pipeline_gradient_updates(monkeypatch):
    config = core.LabConfig(74, True)
    first = core.compute_demo("pipeline", config)
    original = core.make_data
    def perturbed(n, seed, concise_probability=.5):
        result = original(n,seed,concise_probability)
        if seed == config.seed+604:
            return (result[0]*-3, result[1], result[2], result[3])
        return result
    monkeypatch.setattr(core,"make_data",perturbed)
    second = core.compute_demo("pipeline", config)
    for a,b in zip(first["records"],second["records"]):
        np.testing.assert_array_equal(a["weights"],b["weights"])
    assert first["records"][-1]["metrics"] != second["records"][-1]["metrics"]


def test_numerical_runs_restore_global_rng_and_thread_count():
    rng = torch.random.get_rng_state().clone(); threads = torch.get_num_threads()
    config = core.LabConfig(4,True)
    first = core.serializable(core.compute_demo("sft",config))
    second = core.serializable(core.compute_demo("sft",config))
    assert first == second
    assert torch.equal(rng,torch.random.get_rng_state())
    assert threads == torch.get_num_threads()
    assert replace(config,quick=False).dimension > config.dimension
    assert replace(config,quick=False).repetitions > config.repetitions


@pytest.mark.parametrize("kwargs", [{"seed":-1},{"seed":True},{"quick":1}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        core.LabConfig(**kwargs)
