"""Small real exports in temporary directories, without touching dated outputs."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text_lab import figures, interactive
from visualizations import SECTIONS, main, validate_asset


@pytest.mark.rendering
def test_real_png_gif_and_offline_interactive_exports(tmp_path):
    targets = ["20_similarity_metrics_comparison.png", "19_l2_normalization.gif", "16_cosine_similarity_2d.html"]
    export = figures.Exporter(tmp_path, quick=True, targets=targets)
    figures.normalization(export)
    interactive.generate(export, "cosine")
    records = [validate_asset(export.path(name)) for name in targets]
    assert len(export.paths) == 3
    assert records[1]["frames"] >= 9
    assert records[1]["duration_ms"] > 5000
    source = export.path(targets[2]).read_text(encoding="utf-8")
    assert "window.labState" in source
    assert '<script src=' not in source
    assert 'id="angle"' in source


def test_cli_list_and_asset_registry(capsys):
    assert main(["--list"]) == 0
    assert "cosine" in capsys.readouterr().out
    filenames = [name for _, names in SECTIONS.values() for name in names]
    assert len(filenames) == len(set(filenames)) == 41


@pytest.mark.rendering
def test_cli_individual_asset_and_historical_directory_guard(tmp_path):
    assert main(["--asset", "30_term_score_contributions.png", "--output-dir", str(tmp_path)]) == 0
    assert len(list(tmp_path.rglob("*.png"))) == 1
    assert len(list((tmp_path / "results").glob("experiments-*.json"))) == 1
    with pytest.raises(ValueError, match="historical"):
        main(["--section", "tfidf", "--output-dir", str(tmp_path)])
