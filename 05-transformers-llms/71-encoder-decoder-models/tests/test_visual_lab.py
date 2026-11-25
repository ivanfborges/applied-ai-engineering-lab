"""Numerical controls plus real temporary raster/HTML export checks."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from from_scratch import attention, visibility
from visual_experiments import (
    SOURCE, TARGET, model_experiment, attention_experiment, shared_pca,
    objective_experiment, corrupt_spans, generation_experiment, softmax,
    cache_experiment, cache_bytes, connection_counts, rag_experiment,
    validate_experiments,
)
from visual_html import attention_page, geometry_payload
import visual_figures as figures
from visualizations import generate, inspect_artifact


@pytest.fixture(scope="module")
def model():
    return model_experiment(generation_steps=3)


@pytest.fixture(scope="module")
def attn(model):
    return attention_experiment(model)


def test_real_layer_traces_causality_and_visible_positive_controls(model, attn):
    report = validate_experiments(model, attn, cache_experiment())
    assert all(report["checks"].values())
    assert all(error < 1e-12 for error in model["trace_errors"].values())
    assert np.max(model["deltas"]["decoder"][:-1]) <= 1e-12
    assert np.max(model["target_deltas"][:-1]) <= 1e-12
    assert np.min(model["deltas"]["encoder"][:-1]) > 1e-6
    assert np.min(model["deltas"]["cross"]) > 1e-6
    assert model["baseline"]["cross"].shape == (len(TARGET), 16)


def test_attention_uses_shared_projections_and_rectangular_source_memory(attn):
    for key in ("q", "k", "v", "logits"):
        np.testing.assert_array_equal(attn["encoder"][key], attn["decoder"][key])
    assert attn["cross"]["weights"].shape == (len(TARGET), len(SOURCE))
    for item in attn.values():
        expected, weights = attention(item["q"], item["k"], item["v"], item["allowed"])
        np.testing.assert_array_equal(item["output"], expected)
        np.testing.assert_array_equal(item["weights"], weights)
        assert np.all(item["weights"][~item["allowed"]] == 0)
        np.testing.assert_allclose(item["weights"].sum(axis=1), 1, atol=1e-12)


def test_shared_pca_is_one_orthonormal_transform_and_displacements_contract(model):
    arrays = [model["embeddings"]["encoder"], model["baseline"]["encoder"], model["perturbed"]["encoder"]]
    pca = shared_pca(arrays)
    np.testing.assert_allclose(pca["basis"].T @ pca["basis"], np.eye(3), atol=1e-12)
    for raw, xyz in zip(arrays, pca["coordinates"]):
        np.testing.assert_allclose(xyz, (raw-pca["mean"]) @ pca["basis"])
    projected = np.linalg.norm(pca["coordinates"][2]-pca["coordinates"][1], axis=1)
    full = np.linalg.norm(arrays[2]-arrays[1], axis=1)
    assert np.all(projected <= full + 1e-12)
    payload = geometry_payload(model)
    assert payload["basis"].shape == (16, 3)
    np.testing.assert_array_equal(payload["families"]["decoder"]["full_l2"], model["deltas"]["decoder"])


def test_objectives_align_loss_masks_shifted_labels_and_span_targets():
    data = objective_experiment()
    mlm = data["mlm"]
    selected_losses = [-np.log(mlm["probabilities"][i, data["vocab"].index(mlm["labels"][i])]) for i in mlm["selected"]]
    assert mlm["mean_selected_loss"] == pytest.approx(np.mean(selected_losses))
    assert mlm["inputs"][5] == mlm["labels"][5]  # Selected unchanged token is still scored.
    assert data["causal"]["inputs"][1:] == data["causal"]["labels"][:-1]
    assert data["span"]["inputs"][0] == "<start>"
    assert data["span"]["inputs"][1:] == data["span_target"][:-1]
    assert data["span_target"] == ["<s0>", "bank", "approved", "<s1>", "loan.", "<s2>", "<eos>"]
    for i, j in data["leak_pairs"]:
        assert data["causal"]["inputs"][j] == data["causal"]["labels"][i]
        assert not visibility(len(data["causal"]["inputs"]), len(data["causal"]["inputs"]), causal=True)[i, j]


@pytest.mark.parametrize("spans", [[(-1, 2)], [(2, 2)], [(0, 9)], [(0, 2), (1, 3)], [(0, 1), (1, 2)], []])
def test_invalid_spans_fail_explicitly(spans):
    with pytest.raises(ValueError):
        corrupt_spans(SOURCE, spans)


@pytest.mark.parametrize("prompt,total", [(1, 1), (1, 8), (3, 10), (5, 12)])
def test_absolute_position_cache_matches_prefix_recomputation(prompt, total):
    cache = cache_experiment(prompt=prompt, total=total)
    np.testing.assert_allclose(cache["cached_outputs"], cache["uncached_outputs"], rtol=0, atol=1e-12)
    assert cache["cached_projection_vectors"][-1] == 2*total
    assert cache["uncached_projection_vectors"][-1] == 2*sum(range(prompt, total+1))
    assert cache["cache_bytes"][-1] == 2*total*8*8


def test_cache_memory_and_complexity_formulas():
    assert cache_bytes(1024, 12, 4, 64, 2) == 2*1024*12*4*64*2
    assert cache_bytes(1024, 12, 4, 64, 2, batch=2) == 2*cache_bytes(1024, 12, 4, 64, 2)
    counts = connection_counts(6, 7)
    assert counts == {"encoder": 36, "decoder": 91, "cross": 106}
    assert connection_counts(np.array([1, 2]), 0)["cross"].tolist() == [1, 4]
    with pytest.raises(ValueError):
        cache_bytes(0, 12, 4, 64, 2)
    with pytest.raises(ValueError):
        connection_counts(-1, 3)


def test_temperature_sampling_is_reproducible_and_greedy_argmax_invariant():
    data = generation_experiment()
    greedy = [run for run in data["runs"] if run["mode"] == "greedy"]
    assert all([step["chosen"] for step in run["steps"]] == [step["chosen"] for step in greedy[0]["steps"]] for run in greedy)
    for run in data["runs"]:
        for step in run["steps"]:
            np.testing.assert_allclose(step["probabilities"], softmax(step["logits"], run["temperature"]))
    assert [s["chosen"] for s in data["runs"][-1]["steps"]] == [s["chosen"] for s in generation_experiment()["runs"][-1]["steps"]]
    assert softmax([0, 1], 0.25).max() > softmax([0, 1], 2).max()


@pytest.mark.parametrize("temperature", [0, -1, np.nan, np.inf])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        softmax([0, 1], temperature)


def test_rag_vectors_and_pair_scores_match_declared_mechanism():
    data = rag_experiment()
    np.testing.assert_allclose(np.linalg.norm(data["document_vectors"], axis=1), 1)
    np.testing.assert_allclose(data["cosine"], data["document_vectors"] @ data["query_vector"])
    np.testing.assert_allclose(data["joint_scores"], 0.4*data["cosine"] + 0.4*data["pair_features"]["bank_and_loan"] + 0.2*data["pair_features"]["approval_term"])
    assert set(data["reranked"]) == set(data["retrieved"])
    assert list(data["reranked"]) == [0, 3, 2, 1]


def test_html_payload_and_heatmap_are_the_computed_values(tmp_path, attn):
    page = tmp_path / "attention.html"
    attention_page(page, attn)
    text = page.read_text(encoding="utf-8")
    payload = json.loads(text.split('<script id="lab-data" type="application/json">', 1)[1].split('</script>', 1)[0])
    np.testing.assert_array_equal(payload["decoder"]["weights"], attn["decoder"]["weights"])
    assert inspect_artifact(page)["embedded_plotly"]
    fig = figures.plt.figure()
    try:
        figures.attention_panels(fig, attn, "probabilities")
        for ax, family in zip(fig.axes[:3], ("encoder", "decoder", "cross")):
            image = ax.images[0].get_array()
            np.testing.assert_allclose(image.data, attn[family]["weights"])
            np.testing.assert_array_equal(np.ma.getmaskarray(image), ~attn[family]["allowed"])
    finally:
        figures.plt.close(fig)


@pytest.mark.rendering
def test_real_gif_frames_are_decodable_and_change(tmp_path, attn):
    output = tmp_path / "masking.gif"
    figures.save_gif(output, lambda fig, i: figures.masking_frame(fig, attn, i), 4, quick=True)
    info = inspect_artifact(output)
    assert info["frames"] == 4
    with Image.open(output) as image:
        first = np.asarray(image.convert("RGB"))
        image.seek(1)
        assert np.any(first != np.asarray(image.convert("RGB")))


@pytest.mark.rendering
def test_no_gif_cli_path_produces_real_static_and_offline_interactive_exports(tmp_path):
    directory = tmp_path / "run"
    report = generate(directory, section="complexity", quick=True, no_gif=True)
    assert not report["failures"]
    assert {entry["name"] for entry in report["artifacts"]} == {"attention_complexity.png", "attention_complexity_log_scale.png", "attention_complexity_3d.html"}
    assert (directory/"index.html").is_file()
    assert (directory/"numerical_data.json").is_file()
    with pytest.raises(FileExistsError):
        generate(directory, section="complexity", quick=True)
