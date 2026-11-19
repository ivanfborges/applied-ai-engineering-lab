"""Numerical replay invariants and small real-export tests."""

import copy
import json
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visual_core as core
import visual_figures as visuals
from visualize_training_loop import create_run_directory, generate


@pytest.fixture(scope="module")
def small_lab():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    result = core.run_lab(core.LabConfig(samples=80, epochs=3, frames=4, grid_size=12, batch_size=17))
    yield result
    torch.set_num_threads(previous)


def test_partitions_scaling_orders_and_unequal_tail(small_lab):
    data, run = small_lab["data"], small_lab["mlp"]
    ids = data["ids"]
    np.testing.assert_array_equal(np.sort(np.concatenate(list(ids.values()))), np.arange(80))
    np.testing.assert_allclose(data["mean"], data["raw"][ids["train"]].mean(0))
    np.testing.assert_allclose(data["scale"], data["raw"][ids["train"]].std(0))
    assert run["total_steps"] == 3 * 3
    for order in run["orders"]:
        np.testing.assert_array_equal(np.sort(order), np.sort(ids["train"]))
    assert len(run["events"][-1]["ids"]) == 14
    np.testing.assert_array_equal(run["orders"], small_lab["linear"]["orders"])


def test_event_forward_chain_rule_and_parameter_gradients_replay(small_lab):
    for event in small_lab["mlp"]["events"]:
        model = core.make_model("mlp", 0)
        core.restore(model, event["state_before"])
        x, y = torch.from_numpy(event["x"]), torch.from_numpy(event["y"])
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        np.testing.assert_allclose(logits.detach().numpy(), event["logits_before"], atol=1e-7)
        assert loss.item() == pytest.approx(event["batch_loss_before"])
        for name, parameter in model.named_parameters():
            np.testing.assert_allclose(parameter.grad.numpy(), event["gradient"][name], atol=1e-7)
        sample = event["sample"]
        np.testing.assert_allclose(sample["hidden"], np.maximum(sample["z1"], 0))
        assert sample["dlogit"] == pytest.approx((sample["probability"] - sample["y"]) / len(event["ids"]), abs=1e-7)
        np.testing.assert_allclose(sample["dhidden"], event["state_before"]["output.weight"][0] * sample["dlogit"], atol=1e-7)
        np.testing.assert_allclose(sample["dz1"], sample["dhidden"] * (sample["z1"] > 0), atol=1e-7)


def test_event_after_update_metrics_grid_and_ids_are_synchronized(small_lab):
    data = small_lab["data"]
    changed = False
    for event in small_lab["mlp"]["events"]:
        assert set(event["ids"]).issubset(set(data["ids"]["train"]))
        np.testing.assert_array_equal(event["x"], data["x"][event["ids"]])
        model = core.make_model("mlp", 0)
        core.restore(model, event["state_after"])
        for name, delta in event["delta"].items():
            np.testing.assert_array_equal(delta, event["state_after"][name] - event["state_before"][name])
            changed |= bool(np.any(delta != 0))
        with torch.inference_mode():
            after = model(torch.from_numpy(event["x"]))
            expected = torch.nn.functional.binary_cross_entropy_with_logits(after, torch.from_numpy(event["y"]))
        np.testing.assert_allclose(after.numpy(), event["logits_after"], atol=1e-7)
        assert expected.item() == pytest.approx(event["batch_loss_after"])
        np.testing.assert_allclose(core.predict_grid(model, data), event["grid_probability"], atol=1e-7)
        for partition in ("train", "validation"):
            ids = data["ids"][partition]
            assert core.measure(model, data["x"][ids], data["y"][ids]) == event[partition]
    assert changed


def test_online_loss_uses_actual_batch_sizes(small_lab):
    data = small_lab["data"]
    config = core.LabConfig(samples=80, epochs=2, frames=6, grid_size=12, batch_size=17)
    run = core.record_training(data, config)
    for epoch in (1, 2):
        rows = [e for e in run["events"] if e["epoch"] == epoch]
        assert len(rows) == 3
        expected = sum(e["batch_loss_before"] * len(e["ids"]) for e in rows) / 48
        assert run["history"][epoch]["online_loss"] == pytest.approx(expected)


def test_validation_changes_never_change_training_parameters(small_lab):
    data = copy.deepcopy(small_lab["data"])
    config = core.LabConfig(samples=80, epochs=3, frames=4, grid_size=12, batch_size=17)
    data["x"][data["ids"]["validation"]] += 20
    data["y"][data["ids"]["test"]] = 1 - data["y"][data["ids"]["test"]]
    repeat = core.record_training(data, config)
    for actual, expected in zip(repeat["checkpoints"], small_lab["mlp"]["checkpoints"]):
        for name in actual["state"]:
            np.testing.assert_array_equal(actual["state"][name], expected["state"][name])


def test_selected_checkpoint_reproduces_metrics_and_test(small_lab):
    selected = small_lab[small_lab["selected"]]
    best = selected["best_epoch"]
    assert best == min(range(1, 4), key=lambda i: selected["history"][i]["validation_loss"])
    model = core.make_model(selected["kind"], 0)
    core.restore(model, selected["checkpoints"][best]["state"])
    before = core.snapshot(model)
    data = small_lab["data"]
    ids = data["ids"]["test"]
    assert core.measure(model, data["x"][ids], data["y"][ids]) == small_lab["selected_test"]
    for key, value in model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), before[key])


def test_logistic_surface_path_and_actual_sgd_updates_share_objective(small_lab):
    record = small_lab["logistic"]
    w, b = np.meshgrid(record["weights"], record["biases"])
    np.testing.assert_allclose(record["surface"], core.logistic_objective(record["x"], record["y"], np.stack([w, b], -1)))
    for run in record["runs"].values():
        np.testing.assert_array_equal(run["path"][0], [-1.5, 1.])
        logits = torch.tensor(run["path"][:, 0, None] * record["x"] + run["path"][:, 1, None])
        target = torch.tensor(np.broadcast_to(record["y"], logits.shape).copy())
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(1)
        np.testing.assert_allclose(expected.numpy(), run["loss"], atol=1e-14)
        assert run["uphill_steps"] == int((np.diff(run["loss"]) > 0).sum())
        assert np.isfinite(run["gradient"]).all()
        if run["optimizer"] == "SGD":
            np.testing.assert_allclose(np.diff(run["path"], axis=0), -run["learning_rate"] * run["gradient"], atol=1e-14)
    assert record["runs"][record["primary"]]["uphill_steps"] > 0


def test_dropout_and_autograd_switches_are_independent(small_lab):
    probe = small_lab["dropout"]
    for name, run in probe.items():
        assert all(flag == (not run["inference"]) for flag in run["requires_grad"])
        assert np.ptp(run["probability"]) > 0 if run["training"] else np.ptp(run["probability"]) == 0
    np.testing.assert_array_equal(probe["train + grad"]["probability"], probe["train + inference"]["probability"])
    np.testing.assert_array_equal(probe["eval + grad"]["probability"], probe["eval + inference"]["probability"])


class ExternalResources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("src", "href") and value and value.startswith(("https://", "http://", "//")):
                self.external.append((tag, name, value))


@pytest.mark.rendering
def test_real_gif_png_and_html_exports(small_lab, tmp_path):
    visuals.theme()
    visuals.dataset_visual(small_lab, tmp_path, 4)
    visuals.forward_visual(small_lab, tmp_path)
    visuals.backward_visual(small_lab, tmp_path)
    visuals.boundary_visual(small_lab, tmp_path, 4)
    visuals.mode_visual(small_lab, tmp_path)
    visuals.landscape_html(small_lab, tmp_path, 4)
    visuals.optimizer_html(small_lab, tmp_path)
    visuals.dashboard_html(small_lab, tmp_path)
    for path in tmp_path.glob("*.gif"):
        with Image.open(path) as image:
            assert image.n_frames >= 4
            assert image.width >= 800
            assert image.info["duration"] >= 400
            for frame in range(image.n_frames):
                image.seek(frame)
                image.convert("RGB").load()
    for path in tmp_path.glob("*.png"):
        with Image.open(path) as image:
            image.verify()
    for path in tmp_path.glob("*.html"):
        document = path.read_text(encoding="utf-8")
        parser = ExternalResources(); parser.feed(document)
        assert parser.external == []
        assert "Plotly.newPlot" in document
        assert path.stat().st_size < 12 * 1024**2


@pytest.mark.rendering
def test_quick_generation_records_manifest_and_fresh_directory(tmp_path):
    torch.set_num_threads(1)
    config = core.LabConfig(samples=80, epochs=2, frames=4, grid_size=12)
    destination = generate(config, tmp_path, mode="quick")
    manifest = json.loads((destination / "manifest.json").read_text())
    records = json.loads((destination / "numerical_records.json").read_text())
    assert len(list(destination.glob("*.png"))) == 4
    assert len(list(destination.glob("*.gif"))) == 1
    assert records["mlp"]["events"][-1]["step"] == records["mlp"]["total_steps"]
    assert manifest["configuration"]["samples"] == 80
    assert create_run_directory(tmp_path) != destination


@pytest.mark.parametrize("kwargs", [{"frames": 200}, {"samples": 10}, {"epochs": 0}, {"grid_size": 500},
                                    {"learning_rate": float("nan")}, {"batch_size": True}])
def test_invalid_or_excessive_config_rejected(kwargs):
    with pytest.raises(ValueError):
        core.LabConfig(**kwargs).validate()


def test_cli_invalid_frames_fails_before_writing(tmp_path):
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "visualize_training_loop.py"),
               "--frames", "1000", "--output-dir", str(tmp_path / "absent")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert not (tmp_path / "absent").exists()
