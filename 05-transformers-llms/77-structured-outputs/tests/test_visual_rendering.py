"""Real temporary exports; public previews and historical runs are untouched."""

from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visual_core
import visual_html
import visual_render


@pytest.mark.rendering
@pytest.mark.parametrize("demo", ["decoding", "masking", "grammar", "validation", "strictness", "retries", "surface", "correlation", "guardrails"])
def test_real_static_exports(tmp_path, demo):
    lab = visual_core.compute_lab(samples=100)
    paths = visual_render.render_static(demo, tmp_path, lab, preview=True)
    assert paths
    for name in paths:
        with Image.open(tmp_path / name) as image:
            image.load()
            assert image.width >= 1000 and image.height >= 500
            if name.endswith('.gif'):
                assert image.n_frames >= 4 and image.info['loop'] == 0


@pytest.mark.parametrize("demo", list(visual_html.FILES))
def test_standalone_html_is_self_contained(tmp_path, demo):
    files = visual_html.render_html(demo, tmp_path, visual_core.compute_lab(samples=100))
    for name in files:
        data = (tmp_path / name).read_text(encoding='utf-8')
        assert 'plotly.js' in data and 'const LAB=' in data
        assert '<script src=' not in data
        assert 'Author interpretation review required' in data


def test_cli_list_and_invalid_inputs_from_another_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'visualizations.py'
    run = subprocess.run([sys.executable, str(script), '--list'], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert all(name + ':' in run.stdout for name in visual_core.DEMOS)
    bad = subprocess.run([sys.executable, str(script), '--samples', '0'], cwd=tmp_path, capture_output=True, text=True)
    assert bad.returncode != 0 and 'samples' in bad.stderr


def test_cli_preserves_existing_output_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'visualizations.py'
    marker = tmp_path / 'keep.txt'
    marker.write_text('original', encoding='utf-8')
    run = subprocess.run([sys.executable, str(script), '--demo', 'space', '--output-dir', str(tmp_path)], capture_output=True, text=True)
    assert run.returncode != 0 and marker.read_text(encoding='utf-8') == 'original'
