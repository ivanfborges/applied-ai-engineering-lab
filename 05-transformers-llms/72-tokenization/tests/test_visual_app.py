"""Headless interactive checks across the ten live dashboard sections."""

from pathlib import Path
import sys

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import SECTIONS

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.integration
def test_dashboard_sections_and_material_controls():
    app = AppTest.from_file(str(APP), default_timeout=40).run()
    assert not app.exception
    for name in SECTIONS:
        app.selectbox(key="section").select(name).run()
        assert not app.exception, (name, app.exception)
    app.number_input(key="cost_input").set_value(12000).run()
    assert not app.exception
    assert app.metric[0].value == "0.03200"
    app.slider(key="cost_reduction").set_value(50).run()
    assert app.metric[0].value == "0.02000"
    app.selectbox(key="section").select("WordPiece and Unigram").run()
    app.segmented_control(key="algorithm").set_value("Unigram").run()
    app.slider(key="ug_unbeliev").set_value(1).run()
    assert not app.exception
    app.selectbox(key="section").select("RAG and context").run()
    app.segmented_control(key="rag_view").set_value("Context window").run()
    app.selectbox(key="context_limit").select(8192).run()
    app.toggle(key="prune").set_value(True).run()
    assert not app.exception
    app.selectbox(key="section").select("Tokenizer comparison").run()
    app.multiselect(key="tokenizers").set_value(["SentencePiece BPE", "SentencePiece Unigram", "WordPiece local", "Character BPE"]).run()
    assert not app.exception
