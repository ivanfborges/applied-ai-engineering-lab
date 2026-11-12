"""Small exports use temporary files and never overwrite public previews."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_visuals import generate
from visual_charts import surface_figure


def test_offline_exports_and_animated_content(tmp_path):
    record = generate(tmp_path, frames=3, repetitions=1000)
    assert record["config"]["n_per_arm"] > 14000
    for name in ("sampling_distributions.gif", "power_growth.gif", "effective_sample_size.gif"):
        with Image.open(tmp_path / name) as gif:
            assert gif.n_frames == 3
            first = gif.convert("RGB").copy()
            gif.seek(2)
            assert ImageChops.difference(first, gif.convert("RGB")).getbbox()
    with Image.open(tmp_path / "effect_size_sample_size.png") as png:
        assert png.width >= 1600
        assert png.height >= 900
    html = (tmp_path / "power_surface_3d.html").read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "<script src=" not in html
    records = list(tmp_path.glob("visual_run_*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["config"]["seed"] == 46


def test_surface_has_valid_probabilities():
    fig = surface_figure(0.99, 0.05, 0.8)
    values = np.asarray(fig.data[0].z)
    assert values.shape == (45, 55)
    assert np.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all()
    assert np.asarray(fig.data[0].y).max() / 100 + 0.99 < 1
