"""Small real exports remain in pytest temporary directories, never public previews."""

from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual_core import LabConfig, make_lab_corpus, train_visual_run
from visual_html import compute_html
from visual_render import render_embeddings, render_probabilities, write_gallery


@pytest.mark.rendering
def test_measured_probability_and_shared_pca_exports_are_real_and_local(tmp_path):
    config = LabConfig(steps=2, width=8, heads=2, layers=1, ffn=16, batch_size=4, snapshots=3)
    report, _ = train_visual_run(config, make_lab_corpus())
    render_probabilities(tmp_path, report)
    render_embeddings(tmp_path, report)
    for name in ("token_probability_evolution.gif", "embedding_evolution_2d.gif"):
        with Image.open(tmp_path / name) as image:
            assert image.n_frames == 3 and image.width >= 900
            image.seek(image.n_frames - 1)
            image.load()
    for name in ("prediction_entropy_evolution.png", "embedding_cosine_similarity.png"):
        with Image.open(tmp_path / name) as image:
            assert image.width >= 1500
    page = (tmp_path / "embedding_evolution_3d.html").read_text(encoding="utf-8")
    assert 'src="plotly.min.js"' in page and 'src="https://' not in page
    assert "explained_variance" in page and (tmp_path / "plotly.min.js").stat().st_size > 100000
    assets = write_gallery(tmp_path)
    assert len(assets) == 6
    assert (tmp_path / "index.html").exists()


@pytest.mark.rendering
def test_compute_simulator_exposes_all_assumptions_without_external_scripts(tmp_path):
    compute_html(tmp_path / "training_compute_simulator.html")
    page = (tmp_path / "training_compute_simulator.html").read_text(encoding="utf-8")
    for field in ("parameters", "tokens", "accelerators", "peak", "utilization", "cost"):
        assert f'id="{field}"' in page
    assert "analytical_estimate:true" in page and 'src="https://' not in page
