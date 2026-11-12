"""Small rendering check in a temporary directory, never tracked previews."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from animations import export_gifs


def test_two_gifs_have_distinct_frames_and_configuration_manifest(tmp_path):
    files=export_gifs(tmp_path,n=500,seed=47,frames=6,width=600,height=360)
    assert len(files)==2
    for file in files:
        with Image.open(file) as im:
            assert im.size==(600,360)
            assert im.n_frames==6
            hashes=[]
            for i in range(im.n_frames):
                im.seek(i)
                hashes.append(hash(np.asarray(im.convert("RGB")).tobytes()))
            assert len(set(hashes))==6
    manifest=json.loads((files[0].parent/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["configuration"]["seed"]==47
    assert manifest["interpretation_status"]=="author review required"
