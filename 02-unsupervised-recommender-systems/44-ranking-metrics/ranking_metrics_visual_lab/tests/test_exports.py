"""Export smoke checks render only into pytest's temporary directory."""

from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_gifs
import metrics as lab


@pytest.mark.parametrize("kind", export_gifs.KINDS)
def test_animation_states_remain_valid_and_deterministic(kind):
    first = export_gifs.states(kind)
    assert first == export_gifs.states(kind)
    for state in first:
        assert len(set(state["order"])) == len(state["order"])
        scores = lab.evaluate_query(state["order"], state["labels"], state["k"])
        assert all(0 <= value <= 1+1e-12 for value in scores.values())


def test_temporary_gif_decodes_and_closes_figure(tmp_path):
    path = export_gifs.render_animation("ap", tmp_path / "ap.gif", fps=2, dpi=50)
    with Image.open(path) as gif:
        assert gif.n_frames >= 6
        assert gif.size == (480, 290)
        for index in range(gif.n_frames):
            gif.seek(index)
            gif.load()
            assert gif.info["duration"] > 0
    assert not export_gifs.plt.get_fignums()
