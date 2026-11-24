"""Real temporary-directory exports; never regenerate curated previews in tests."""

from html.parser import HTMLParser
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_figures import pipeline
from attention_html import context_page, geometry_page, memory_page, softmax_page, values_page
from visualize_attention import EXPERIMENTS, main


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.external = []
    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.external.extend(value for name, value in attrs if name == "src")


def test_htmls_bundle_runtime_and_controls(tmp_path):
    for name, generate in (("geometry", geometry_page), ("softmax", softmax_page), ("values", values_page), ("context", context_page), ("memory", memory_page)):
        path = tmp_path/f"{name}.html"
        generate(path)
        source = path.read_text(encoding="utf-8")
        parser = Scripts()
        parser.feed(source)
        assert parser.external == []
        assert "plotly.js" in source.lower()
        assert 'id="reset"' in source and "window.labReady" in source and "window.labState" in source


def verify_gif(path, minimum_frames):
    with Image.open(path) as image:
        assert image.format == "GIF" and image.n_frames >= minimum_frames
        for frame in range(image.n_frames):
            image.seek(frame)
            image.load()
            assert image.info["duration"] > 0
        assert image.info.get("loop") == 0


@pytest.mark.rendering
def test_actual_pipeline_gif_and_high_resolution_png(tmp_path):
    pipeline(tmp_path/"pipeline.gif", tmp_path/"pipeline.png")
    verify_gif(tmp_path/"pipeline.gif", 9)
    with Image.open(tmp_path/"pipeline.png") as image:
        assert image.format == "PNG" and image.width >= 1900
        image.verify()


@pytest.mark.rendering
def test_cli_scoped_generation_and_dated_manifest(tmp_path, capsys):
    assert main(["--list"]) is None
    assert len(EXPERIMENTS) == 12
    assert "geometry" in capsys.readouterr().out
    folder = main(["--experiment", "complexity", "--quick", "--output-dir", str(tmp_path)])
    report = json.loads((folder/"run.json").read_text())
    assert report["experiments"] == ["complexity"] and report["quick"]
    assert report["seed"] == 66
    assert len(report["assets"]) == 3
    assert (folder/"index.html").is_file()
    assert [row["length"] for row in report["timings"]] == [128, 256]
    assert all(row["median_seconds"] > 0 for row in report["timings"])
    verify_gif(folder/"11_attention_matrix_growth.gif", 8)
    assert not (folder/"03_attention_pipeline.gif").exists()
