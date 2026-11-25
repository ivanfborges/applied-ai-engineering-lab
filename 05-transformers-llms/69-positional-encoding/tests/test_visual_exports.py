"""Real bounded exports in temporary directories; no public preview overwrites."""

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visual_core import Config
from visualizations import GROUPS, generate, validate_exports


@pytest.mark.rendering
def test_all_html_and_static_exports_decode_and_resolve_offline(tmp_path):
    folder=generate(tmp_path,GROUPS,Config(seq_len=12,dim=8,frames=4,dpi=60),("html","png"))
    record=json.loads((folder/"report.json").read_text(encoding="utf-8"))
    assert len(record["selected"])==13
    assert record["training_updates"]==0
    assert record["review_status"]=="author review required"
    assert len(list(folder.glob("*.html")))==17
    assert len(list(folder.glob("*.png")))==7
    assert (folder/"plotly.min.js").stat().st_size>1_000_000
    assert len(validate_exports(folder))==24
    assert all(item["bytes"]>0 for item in record["validated_exports"])


@pytest.mark.rendering
def test_real_rope_animations_have_multiple_readable_frames(tmp_path):
    config=Config(seq_len=8,dim=4,frames=4,fps=4,dpi=60)
    folder=generate(tmp_path,["rope"],config,("gif",))
    for name in ("rope_rotation.gif","rope_relative_shift_invariance.gif"):
        with Image.open(folder/name) as image:
            assert image.n_frames>=4
            assert image.info["loop"]==0
            first=image.convert("RGB").tobytes()
            image.seek(2)
            assert first!=image.convert("RGB").tobytes()
    assert len(list((folder/"frames").glob("*.png")))==6


def test_cli_list_and_invalid_dimensions_do_not_generate_outputs(tmp_path):
    script=Path(__file__).resolve().parents[1]/"visualizations.py"
    listed=subprocess.run([sys.executable,str(script),"--list"],cwd=tmp_path,capture_output=True,text=True,check=True)
    assert "rope" in listed.stdout and "extrapolation" in listed.stdout
    invalid=subprocess.run([sys.executable,str(script),"--demo","rope","--dim","3","--output-dir",str(tmp_path/"invalid")],capture_output=True,text=True)
    assert invalid.returncode==2
    assert not (tmp_path/"invalid").exists()
