"""Real temporary exports and command/provenance contracts; no tracked assets touched."""

import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visual_core import LabConfig,compute_demo
from visual_render import render_demo
from visual_html import write_gallery
from visualize_fine_tuning import main


class SmokeConfig(LabConfig):
    @property
    def frames(self):
        return 3


@pytest.mark.rendering
def test_dpo_real_gif_png_and_offline_html_exports(tmp_path):
    config=SmokeConfig(42,True)
    report=compute_demo("dpo",config)
    render_demo("dpo",tmp_path,report,config)
    with Image.open(tmp_path/"06_dpo_training_dynamics.gif") as image:
        assert image.n_frames == 3 and image.info["loop"] == 0
        frames=[]
        for index in range(image.n_frames):
            image.seek(index); frames.append(np.array(image.convert("RGB")))
        assert not np.array_equal(frames[0],frames[-1])
    with Image.open(tmp_path/"06_dpo_loss_and_gradients.png") as image:
        assert image.width > 700 and np.asarray(image).std() > 10
    page=(tmp_path/"06_dpo_loss_surface_3d.html").read_text(encoding="utf-8")
    assert re.findall(r'<script src="([^"]+)"',page) == ["plotly.min.js"]
    assert (tmp_path/"plotly.min.js").is_file()
    assert 'id="checkpoint"' in page and 'id="reset"' in page
    payload=json.loads(re.search(r'type="application/json">(.*?)</script>',page,re.S).group(1))
    assert len(payload)==3
    assert all(len(frames)==3 for frames in payload.values())
    assets=write_gallery(tmp_path,{"dpo":report})
    gallery=(tmp_path/"index.html").read_text(encoding="utf-8")
    assert all((tmp_path/name).is_file() and name in gallery for name in assets)


@pytest.mark.rendering
def test_main_reuse_preserves_original_measurements_and_refuses_overwrite(tmp_path):
    original=tmp_path/"first"; reused=tmp_path/"reuse"
    main(["--quick","--demo","efficiency","--output-dir",str(original)])
    content=(original/"experiments/efficiency.json").read_bytes()
    main(["--demo","efficiency","--reuse-run",str(original),"--output-dir",str(reused)])
    assert (original/"experiments/efficiency.json").read_bytes()==content
    assert (reused/"experiments/efficiency.json").read_bytes()==content
    manifest=json.loads((reused/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["public_previews"]==[]
    assert manifest["reused_measurements"]["efficiency"]["sha256"]
    with pytest.raises(SystemExit) as exc:
        main(["--quick","--demo","efficiency","--output-dir",str(original)])
    assert exc.value.code==2


def test_list_and_invalid_demo_do_not_create_outputs(capsys,tmp_path):
    main(["--list"])
    assert "pipeline" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--demo","missing","--output-dir",str(tmp_path/"not-created")])
    assert exc.value.code==2
    assert not (tmp_path/"not-created").exists()
