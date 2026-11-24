"""Real temporary-directory raster/GIF/HTML exports and CLI selection."""

from html.parser import HTMLParser
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from embedding_html import cosine_explorer, embedding_space
from embedding_lab import make_corpus, train_model
from visualize_embeddings import architecture_animation, decorate, evolution, main, window_animation


def test_animation_frame_caption_replaces_previous_caption():
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    try:
        decorate(fig,"Repeated title","first frame caption")
        decorate(fig,"Repeated title","second frame caption")
        assert [text.get_text() for text in fig.texts]==["Repeated title","second frame caption"]
    finally:
        plt.close(fig)


class ExternalScripts(HTMLParser):
    def __init__(self):
        super().__init__(); self.sources=[]
    def handle_starttag(self,tag,attrs):
        if tag=="script":
            self.sources.extend(v for k,v in attrs if k=="src")


@pytest.mark.rendering
def test_actual_rasters_gifs_and_html_in_temporary_directory(tmp_path):
    corpus=make_corpus(size=2)
    model=train_model(corpus,epochs=3)
    evolution(tmp_path/"evolution.gif",model)
    window_animation(tmp_path/"window.gif",2)
    architecture_animation(tmp_path/"architecture.gif","cbow",65)
    for name in ("evolution.gif","window.gif","architecture.gif"):
        with Image.open(tmp_path/name) as image:
            assert image.format=="GIF"
            assert image.n_frames>=4
            for frame in range(image.n_frames):
                image.seek(frame); image.load()
                assert image.info["duration"]>0
    cosine_explorer(tmp_path/"cosine.html")
    embedding_space(tmp_path/"space.html",model,{w:1 for w in model.fit.words})
    for name in ("cosine.html","space.html"):
        source=(tmp_path/name).read_text(encoding="utf-8")
        parser=ExternalScripts();parser.feed(source)
        assert parser.sources==[]
        assert "window.labReady" in source


@pytest.mark.rendering
def test_independent_cli_exports_only_requested_demo_and_dated_manifest(tmp_path):
    folder=main(["--demo","cosine","--quick","--epochs","2","--corpus-size","2","--output-dir",str(tmp_path),"--no-show"])
    import json
    report=json.loads((folder/"results"/"run.json").read_text())
    assert len(report["assets"])==5
    assert (folder/"html"/"22_cosine_similarity_interactive.html").is_file()
    assert not list((folder/"gifs").iterdir())
    for path in (folder/"figures").glob("*.png"):
        with Image.open(path) as image:
            assert image.format=="PNG"
            image.verify()
