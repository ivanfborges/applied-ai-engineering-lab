"""Actual exports in temporary directories; no public preview writes."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual_core import Config
from visualizations import _Resources, generate, validate_exports
from visualizations import TOPICS
from visual_html import EXPLANATIONS, PAGES


def test_every_cli_group_and_html_mode_has_an_explanation():
    assert set(TOPICS) <= set(EXPLANATIONS)
    assert {mode for pages in PAGES.values() for _, mode in pages} <= set(EXPLANATIONS)


@pytest.mark.rendering
def test_real_ffn_and_complete_block_animations(tmp_path):
    folder, report = generate(tmp_path, ["ffn", "transformer-block"], Config(quick=True))
    assert report["interpretation_status"] == "author review required"
    for filename in ["ffn_information_flow.gif", "transformer_block_forward_pass.gif", "transformer_block_backward_pass.gif"]:
        with Image.open(folder / filename) as picture:
            assert picture.n_frames >= 8
            assert picture.width >= 800
            assert picture.info["loop"] == 0
    assert len(validate_exports(folder)) == 5
    other, _ = generate(tmp_path, ["ffn"], Config(quick=True), formats=("html",))
    assert other != folder


@pytest.mark.rendering
def test_standalone_html_and_csv_export(tmp_path):
    folder, report = generate(tmp_path, ["geometry", "normalization", "gradients"], Config(depth=1, d_model=8, quick=True), formats=("html",))
    assert len(report["artifacts"]) == 5
    page = (folder / "layer_normalization_geometry.html").read_text(encoding="utf-8")
    assert 'id="lab-data"' in page
    resources = _Resources()
    resources.feed(page)
    assert not any(link.startswith(("https:", "http:", "//")) for link in resources.links)
    assert "window.labMath" in page
    assert (folder / "dense_gradient_measurements.csv").read_text().startswith("experiment,seed,depth")
    record = json.loads((folder / "run.json").read_text())
    assert record["configuration"]["quick"]
    assert record["experiments"]["dense"]["rows"] > 0


def test_cli_validation_and_invalid_dimension(tmp_path):
    script = Path(__file__).resolve().parents[1] / "visualizations.py"
    command = [sys.executable, str(script), "--validate-only"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["ffn_other_tokens_change"] == 0
    result = subprocess.run([sys.executable, str(script), "--d-model", "7", "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0 and "multiple of 4" in result.stderr
    assert not list(tmp_path.iterdir())
