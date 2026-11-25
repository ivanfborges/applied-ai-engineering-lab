"""Real minimal exports use temporary output paths, preserving author previews."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visualize import generate, inspect_artifact


@pytest.mark.rendering
def test_bpe_generator_writes_real_distinct_frames_and_measured_history(tmp_path):
    manifest = generate(tmp_path / "run", section="bpe", quick=True)
    assert len(manifest["artifacts"]) == 3
    info = inspect_artifact(tmp_path / "run/02_bpe_training.gif")
    assert info["frames"] > 2 and info["width"] >= 1000
    data = json.loads((tmp_path / "run/experiments.json").read_text())
    assert data["bpe"][0]["weighted_tokens"] == 79
    assert (tmp_path / "run/index.html").is_file()
    with pytest.raises(FileExistsError):
        generate(tmp_path / "run", section="bpe", quick=True)


@pytest.mark.rendering
def test_unigram_export_has_embedded_plotly_data_and_live_controls(tmp_path):
    manifest = generate(tmp_path / "run", section="unigram", quick=True)
    source = (tmp_path / "run/05_unigram_interactive.html").read_text(encoding="utf-8")
    assert 'id="lab-data"' in source
    assert "p-" in source and "Plotly.react" in source
    assert '<script src="http' not in source
    assert manifest["interpretation_status"] == "author review required"
