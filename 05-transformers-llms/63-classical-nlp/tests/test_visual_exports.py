"""Real bounded exports into temporary directories; no tracked preview overwrite."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nlp_lab import figures
from visualizations import OUTPUTS, main, validate_asset


@pytest.mark.rendering
def test_real_window_animation_and_standalone_html(tmp_path):
    result = figures.ngrams(tmp_path, True, 63)
    assert result["counts"] == {"1": 6, "2": 5, "3": 4, "4": 3}
    for name in OUTPUTS["ngrams"][1]:
        info = validate_asset(tmp_path / name)
        assert info["bytes"] > 1000
        if name.endswith(".gif"): assert info["frames"] > 10


@pytest.mark.rendering
def test_real_tfidf_pngs_and_interactive_surface(tmp_path):
    result = figures.tfidf(tmp_path, True, 63)
    assert result["df"][result["features"].index("document")] == 4
    for name in OUTPUTS["tfidf"][1]: assert validate_asset(tmp_path / name)["bytes"] > 1000


def test_cli_list_does_not_generate_artifacts(capsys, tmp_path):
    assert main(["--list", "--output-dir", str(tmp_path)]) == 0
    assert "language_model" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_external_html_dependencies_are_rejected(tmp_path):
    file = tmp_path / "bad.html"
    file.write_text('<html><script src="https://example.com/library.js"></script>' + ' ' * 100 + '</html>')
    with pytest.raises(ValueError, match="External"): validate_asset(file)
