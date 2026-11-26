"""Real small exports under tmp_path; no training or preview replacement at import."""

import importlib
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example import Config
from transformer_figures import architecture, causal
from transformer_html import attention_page, memory_page, playground_page, positions_page, representations_page, residual_page
from transformer_lab import default_trace, playground_cases
from visualizations import GROUPS, ScriptLinks, main, validate_assets


def small_config():
    return Config(seed=67, sequence_length=3, d_model=8, heads=2, layers=1,
                  d_ff=16, train_size=12, test_size=4, steps=2)


def test_all_seven_html_pages_are_self_contained_and_expose_controls(tmp_path):
    config = small_config()
    _, trace = default_trace(config)
    paths = [tmp_path / f"view{i}.html" for i in range(7)]
    positions_page(paths[0])
    attention_page(paths[1], [(0, trace)])
    residual_page(paths[2], trace)
    representations_page(paths[3], [(0, trace)])
    memory_page(paths[4])
    attention_page(paths[5], [(0, trace)], "Training snapshots")
    playground_page(paths[6], playground_cases(config))
    for path in paths:
        source = path.read_text(encoding="utf-8")
        parser = ScriptLinks()
        parser.feed(source)
        assert parser.external == []
        assert "window.labReady" in source and "window.labState" in source
        assert 'id="reset"' in source and "plotly.js" in source.lower()
        assert "author interpretation review required" in source
        assert path.stat().st_size < 10 * 1024 ** 2


@pytest.mark.rendering
def test_actual_architecture_and_causal_gifs_decode_every_frame(tmp_path):
    config = small_config()
    _, trace = default_trace(config)
    architecture(tmp_path, config, fps=2, dpi=40)
    attention = next(row for row in trace["attention"] if row["mechanism"] == "decoder")
    causal(tmp_path, attention, trace["decoder_input"], fps=2, dpi=40)
    assets = validate_assets(tmp_path)
    assert len(assets) == 2
    assert assets[0]["frames"] == 14
    with Image.open(tmp_path / "04_decoder_causal_attention.gif") as image:
        assert image.n_frames == 4


def test_cli_lists_sixteen_questions_without_writes(tmp_path, capsys):
    assert main(["--list", "--output-dir", str(tmp_path)]) is None
    assert len(GROUPS) == 16
    assert "target_leakage" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_cli_skips_missing_training_explicitly(tmp_path):
    folder = main(["--only", "training_inference", "target_leakage", "--skip-training", "--training-steps", "1", "--output-dir", str(tmp_path)])
    report = json.loads(next((tmp_path / "metrics").glob("*.json")).read_text())
    assert len(report["skipped"]) == 2
    assert report["training_available"] is False
    assert report["assets"] == []
    assert folder.joinpath("index.html").is_file()
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("reverse_html_order", [False, True])
def test_cli_interactive_only_no_unrelated_training_or_raster(tmp_path, monkeypatch, reverse_html_order):
    folder = main(["--only", "positional_encoding", "--interactive", "--output-dir", str(tmp_path)])
    original_glob = Path.glob

    def ordered_glob(directory, pattern):
        paths = list(original_glob(directory, pattern))
        if directory == folder and pattern == "*.html":
            paths.sort(key=lambda path: path.name, reverse=reverse_html_order)
        return iter(paths)

    # Exercise both filesystem enumeration orders after generating real files.
    monkeypatch.setattr(Path, "glob", ordered_glob)
    assert sorted(path.name for path in folder.glob("*.html")) == ["02_positional_encoding_3d.html", "index.html"]
    assert not list(folder.glob("*.gif"))
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("args", [["--training-steps", "0"], ["--sequence-length", "2"],
    ["--dpi", "10"], ["--fps", "0"], ["--device", "cuda"], ["--only", "architecture", "--interactive"]])
def test_cli_rejects_invalid_or_unsupported_options(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_validation_rejects_external_runtime(tmp_path):
    (tmp_path / "bad.html").write_text('<script src="https://example.org/runtime.js"></script>', encoding="utf-8")
    with pytest.raises(ValueError, match="bundle"):
        validate_assets(tmp_path)


def test_import_does_not_train_or_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import transformer_lab
    import visualizations
    with monkeypatch.context() as patch:
        patch.setattr(transformer_lab, "train_comparison", lambda *args, **kwargs: pytest.fail("import attempted training"))
        importlib.reload(visualizations)
    importlib.reload(visualizations)
    assert list(tmp_path.iterdir()) == []
