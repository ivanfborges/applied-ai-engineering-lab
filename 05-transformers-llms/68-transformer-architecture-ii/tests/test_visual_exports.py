"""Real offline exports in temporary directories; no public-preview regeneration."""

import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visualize import GROUPS, generate, validate_exports
from visual_html import PAGES, write_page


@pytest.mark.rendering
@pytest.mark.parametrize("page", list(PAGES))
def test_each_html_bundles_runtime_and_data_without_external_dependencies(tmp_path, page):
    path = write_page(page, tmp_path, samples=1000)
    exports = validate_exports(tmp_path)
    assert exports[0]["bytes"] > 1_000_000
    source = path.read_text(encoding="utf-8")
    assert "const DATA=" in source and "window.labReady" in source
    assert "author review required" in source.lower()


@pytest.mark.rendering
def test_real_pipeline_matrices_gif_frames_and_static_heatmaps(tmp_path):
    folder = generate(tmp_path, ["architecture", "matrices", "heads", "geometry"], quick=True, dpi=50)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert report["review_status"] == "author review required"
    for filename, minimum in (("01_multi_head_attention_pipeline.gif", 10), ("04_attention_matrix_operations.gif", 11)):
        with Image.open(folder / filename) as image:
            assert image.n_frames >= minimum and image.info["loop"] == 0
            image.seek(image.n_frames - 1); image.load()
    with Image.open(folder / "05_attention_heads.png") as image:
        assert image.width >= 600 and image.height >= 350
    assert len(list((folder / "frames").rglob("*.png"))) == 21
    assert "Numerical" not in (folder / "index.html").read_text(encoding="utf-8")
    assert report["results"]["causal_prefix_max_abs_change"] < 1e-12


def test_new_run_directory_preserves_previous_record_and_individual_selection(tmp_path):
    first = generate(tmp_path, ["head-count", "sharing"], quick=True)
    original = (first / "report.json").read_bytes()
    second = generate(tmp_path, ["validation"], quick=True)
    assert first != second and (first / "report.json").read_bytes() == original
    assert set(json.loads((second / "report.json").read_text())["artifacts"]) == {"validation"}
    assert len(GROUPS) == 17


def test_invalid_selection_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError):
        generate(tmp_path, ["unknown"])
    assert not list(tmp_path.iterdir())
