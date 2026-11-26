"""Scientific contracts for the independently measured visual training phase."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example import clm_batch, token_loss
from visual_core import (ExposedTransformer, LabConfig, Scaling, cached_experiments, compute_budget,
                         corpus_statistics, cosine_matrix, cross_entropy_logits,
                         evaluate_lab, make_lab_corpus, optimal_allocation, pooled_pca,
                         prefix_evaluation, random_mlm_batch, scaling_loss, softmax,
                         train_visual_run)


def tiny_config(**kwargs):
    return LabConfig(steps=4, width=8, heads=2, layers=1, ffn=16, batch_size=8, snapshots=3, **kwargs)


def test_corpus_split_precedes_batching_and_reserved_subject_verb_contexts_are_absent():
    corpus = make_lab_corpus()
    vocabulary = corpus["vocabulary"]
    assert len(corpus["validation"]) == 64
    assert len(corpus["datasets"]["more_unique"]) == 192
    val = {tuple(row) for row in corpus["validation"]}
    for sequences in corpus["datasets"].values():
        assert not val & {tuple(row) for row in sequences}
    train = corpus["datasets"]["diverse"]
    decode = lambda row: [vocabulary[i] for i in row]
    assert any(decode(row)[:4] == ["<bos>", "the", "cat", "eats"] for row in train)
    assert not any(decode(row)[:4] == ["<bos>", "the", "cat", "likes"] for row in train)
    assert set(np.array(corpus["validation"]).ravel()) <= set(np.array(train).ravel())
    assert corpus_statistics(corpus["datasets"]["duplicated"], vocabulary)["unique_examples"] == 24
    assert corpus_statistics(corpus["datasets"]["single_grammar"], vocabulary)["grammars"] == 1
    assert make_lab_corpus() == corpus


@pytest.mark.parametrize("probability", [.001, .25, 1])
def test_mlm_mask_probability_excludes_special_tokens_and_always_supervises(probability):
    corpus = make_lab_corpus()
    original = torch.tensor(corpus["datasets"]["diverse"][:8])
    mask_id = corpus["vocabulary"].index("<mask>")
    inputs, targets = random_mlm_batch(original, mask_id, probability, torch.Generator().manual_seed(5))
    selected = targets != -100
    assert selected.any(1).all() and not selected[:, [0, -1]].any()
    assert (inputs[selected] == mask_id).all()
    torch.testing.assert_close(targets[selected], original[selected])
    torch.testing.assert_close(inputs[~selected], original[~selected])
    logits = torch.zeros(8, 6, len(corpus["vocabulary"]), requires_grad=True)
    token_loss(logits, targets).backward()
    assert (logits.grad[~selected] == 0).all()
    if probability == 1:
        assert selected.sum() == 32


def test_actual_per_head_weights_and_causal_prefix_invariance():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        model = ExposedTransformer(12, tiny_config()).eval()
        tokens = torch.tensor([[0, 1, 2, 3, 4]])
        changed = tokens.clone()
        changed[:, -1] = 5
        with torch.inference_mode():
            before, weights, hidden = model(tokens, True, True)
            after, _, _ = model(changed, True)
            short, _, _ = model(tokens[:, :3], True)
            torch.testing.assert_close(before[:, :-1], after[:, :-1], atol=1e-6, rtol=0)
            torch.testing.assert_close(before[:, :3], short, atol=1e-6, rtol=0)
            assert weights[0].shape == (1, 2, 5, 5) and hidden.shape == (1, 5, 8)
            assert (weights[0].triu(1) == 0).all()
            torch.testing.assert_close(weights[0].sum(-1), torch.ones(1, 2, 5))
            unrestricted, _, _ = model(tokens, False)
            perturbed, _, _ = model(changed, False)
            assert torch.linalg.vector_norm(unrestricted[:, :-1] - perturbed[:, :-1]) > 1e-5


def test_prefix_only_evaluation_matches_causal_teacher_forcing_and_preserves_mode():
    corpus = make_lab_corpus()
    sequences = torch.tensor(corpus["validation"][:4])
    model = ExposedTransformer(len(corpus["vocabulary"]), tiny_config()).train()
    expected = evaluate_lab(model, clm_batch(sequences), True)
    actual = prefix_evaluation(model, sequences, True)
    assert model.training
    assert expected["nll"] == pytest.approx(actual["nll"], abs=1e-6)
    assert expected["accuracy"] == actual["accuracy"]
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("objective", ["clm", "mlm"])
def test_actual_training_records_are_finite_reproducible_and_budgeted(objective):
    config, corpus = tiny_config(), make_lab_corpus()
    report, _ = train_visual_run(config, corpus, objective=objective, causal=objective == "clm")
    rerun, _ = train_visual_run(config, corpus, objective=objective, causal=objective == "clm")
    assert report["processed_training_tokens"] == config.steps * config.batch_size * (5 if objective == "clm" else 6)
    assert report["parameters"] > report["nonembedding_parameters"] > 0
    assert all(np.isfinite(row["gradient_norm"]) and np.isfinite(row["batch_nll"]) for row in report["trace"])
    assert report["supervised_training_tokens"] > 0
    for a, b in zip(report["snapshots"], rerun["snapshots"]):
        assert a["train"] == b["train"] and a["validation"] == b["validation"]
        np.testing.assert_allclose(a["embedding"], b["embedding"], atol=1e-7, rtol=0)
        np.testing.assert_allclose(np.sum(a["probabilities"], axis=-1), 1, atol=1e-6)
    # No assertion about monotone loss or a predetermined validation ranking.
    assert (report["prefix_only_validation"] is not None) == (objective == "clm")


def test_softmax_numerical_stability_and_explicit_logit_gradients():
    loss, gradient = cross_entropy_logits([1000, -1000, 0], 1)
    assert loss == 2000 and np.isfinite(gradient).all()
    logits = np.array([-.8, .4, 0])
    loss, gradient = cross_entropy_logits(logits, 0)
    for i in range(3):
        plus, minus = logits.copy(), logits.copy()
        plus[i] += 1e-6
        minus[i] -= 1e-6
        estimate = (cross_entropy_logits(plus, 0)[0] - cross_entropy_logits(minus, 0)[0]) / 2e-6
        assert gradient[i] == pytest.approx(estimate, abs=1e-9)
    assert softmax(logits).sum() == pytest.approx(1)


def test_pooled_pca_uses_a_shared_coordinate_system_and_cosines_are_original_dimensional():
    matrices = np.random.default_rng(73).normal(size=(3, 8, 6))
    matrices[2, 0] = matrices[0, 0]
    coords, variance = pooled_pca(matrices)
    assert coords.shape == (3, 8, 3) and variance.shape == (3,)
    np.testing.assert_allclose(coords[2, 0], coords[0, 0], atol=1e-12)
    assert 0 < variance.sum() <= 1
    cosine = cosine_matrix(matrices[0])
    np.testing.assert_allclose(np.diag(cosine), 1)
    np.testing.assert_allclose(cosine, cosine.T)


@pytest.mark.parametrize("budget", [1e19, 1e22, 1e25])
def test_compute_optimum_is_derived_and_matches_a_dense_numeric_grid(budget):
    scaling = Scaling()
    optimum = optimal_allocation(budget, scaling)
    assert 6 * optimum["parameters"] * optimum["tokens"] == pytest.approx(budget)
    grid = optimum["parameters"] * np.logspace(-2, 2, 4001)
    losses = scaling_loss(grid, budget / (6 * grid), scaling)
    assert optimum["loss"] == pytest.approx(losses.min(), rel=1e-12)
    n, d = optimum["parameters"] / 1e9, optimum["tokens"] / 1e9
    assert scaling.alpha * scaling.A / n ** scaling.alpha == pytest.approx(scaling.beta * scaling.B / d ** scaling.beta)


def test_compute_units_cost_and_parallelism_assumptions():
    a = compute_budget(7e9, 140e9, 8, 300, .4, 2)
    b = compute_budget(7e9, 140e9, 16, 300, .4, 2)
    assert a["training_flops"] == 5.88e21
    assert a["sustained_flops_per_second"] == 8 * 300e12 * .4
    assert a["tokens_per_parameter"] == 20
    assert a["seconds"] == 2 * b["seconds"]
    assert a["accelerator_hours"] == b["accelerator_hours"]
    assert a["accelerator_cost"] == 2 * a["accelerator_hours"]


def test_saved_experiments_reuse_exact_records_and_reject_configuration_drift(tmp_path):
    config = tiny_config()
    first = cached_experiments(config, tmp_path, ["clm"])
    path = tmp_path / "clm.json"
    original = path.read_bytes()
    second = cached_experiments(config, tmp_path, ["clm"])
    assert first == second and path.read_bytes() == original
    with pytest.raises(ValueError, match="mismatch"):
        cached_experiments(replace(config, seed=74), tmp_path, ["clm"])
    assert path.read_bytes() == original


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"width": 7}, {"layers": True}, {"batch_size": -1},
    {"masking_probability": 0}, {"masking_probability": 1.1}, {"learning_rate": float("nan")},
    {"seed": -1}, {"seed": True}, {"snapshots": 1}, {"device": "remote"}])
def test_invalid_config_errors_are_explicit(kwargs):
    with pytest.raises(ValueError):
        LabConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"parameters": 0}, {"tokens": -1}, {"accelerators": 1.5},
    {"utilization": 1.1}, {"utilization": 0}, {"hourly_cost": -1}, {"peak_tflops": float("inf")}])
def test_invalid_compute_inputs_are_rejected(kwargs):
    values = dict(parameters=1e9, tokens=1e10, accelerators=8, peak_tflops=300, utilization=.4, hourly_cost=2)
    values.update(kwargs)
    with pytest.raises(ValueError):
        compute_budget(**values)
