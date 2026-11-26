"""Temporary exports and CLI checks, without overwriting historical artifacts."""

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))
from visual_render import Gallery


@pytest.mark.rendering
def test_conditioning_exports_readable_gif_and_self_contained_html(tmp_path):
    gallery = Gallery(tmp_path, seed=42, quick=True, duration=2, fps=4, width=800, height=600,
                      dpi=100, input_price=.5, output_price=1.5)
    gallery.conditioning()
    gallery.dashboard()
    with Image.open(tmp_path / "animations" / "01_prompt_conditioning.gif") as gif:
        assert gif.n_frames >= 6 and gif.size == (800, 600)
        for frame in range(gif.n_frames):
            gif.seek(frame)
            gif.convert("RGB").load()
    page = (tmp_path / "prompt_engineering_dashboard.html").read_text(encoding="utf-8")
    assert "cdn.plot.ly" not in page.split("</head>")[1]
    assert '<script src="http' not in page
    assert "Misleading few-shot" in page
    assert "SYNTHETIC ILLUSTRATION" in page
    assert "data:image/gif;base64," in page


@pytest.mark.integration
def test_default_is_simulation_and_cli_rejects_bad_render_settings(tmp_path):
    result = subprocess.run([sys.executable, str(TOPIC / "visualize.py"), "--duration", "nan", "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "invalid seed/render limits" in result.stderr
    assert not list(tmp_path.iterdir())
    result = subprocess.run([sys.executable, str(TOPIC / "visualize.py"), "--help"], capture_output=True, text=True, check=True)
    assert "offline simulation by default" in result.stdout
    assert "--max-requests" in result.stdout and "--results-file" in result.stdout


def test_replay_invalid_file_does_not_substitute_simulation(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"evidence_type": "made_up"}), encoding="utf-8")
    result = subprocess.run([sys.executable, str(TOPIC / "visualize.py"), "--results-file", str(source), "--output-dir", str(tmp_path / "runs")], capture_output=True, text=True)
    assert result.returncode == 2
    assert "No simulation substituted" in result.stderr
    assert not list((tmp_path / "runs").rglob("*.gif"))
