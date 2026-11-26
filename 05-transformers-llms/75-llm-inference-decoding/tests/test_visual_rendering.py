"""Real exports go to temporary directories and never overwrite previews."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visual_core import LabConfig, PRESETS, compute_group, landscape
from visual_render import render_group, stepped_landscape


@pytest.mark.rendering
@pytest.mark.parametrize("group", ["temperature", "stopping", "interactions"])
def test_real_visual_exports(tmp_path, group):
    report = compute_group(group, LabConfig(quick=True))
    if group == "temperature":
        report = report[:3]
    names = render_group(group, tmp_path, report)
    assert names
    for name in names:
        path = tmp_path / name
        assert path.stat().st_size > 1000
        if path.suffix in (".png", ".gif"):
            with Image.open(path) as image:
                assert image.width >= 800
                if path.suffix == ".gif":
                    assert image.n_frames >= 2
        else:
            text = path.read_text(encoding="utf-8")
            assert "Plotly.newPlot" in text
            assert '<script src="https://' not in text


def test_surface_has_independent_flat_tiles_and_actual_hover_values():
    grid = landscape(PRESETS["Moderate"], [0.1, 1, 3], [0.05, 0.9, 1], 3)
    figure = stepped_landscape([grid])
    trace = figure.data[0]
    assert len(trace.x) == 9 * 4
    assert len(trace.i) == 9 * 2
    for tile in range(9):
        height = np.asarray(trace.z)[4 * tile:4 * tile + 4]
        assert np.ptp(height) == 0
        row, column = divmod(tile, 3)
        assert height[0] == grid["entropy"][row, column]
        assert trace.customdata[4 * tile][4] == grid["candidates"][row, column]
