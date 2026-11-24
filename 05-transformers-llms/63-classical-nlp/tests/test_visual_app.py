"""Optional dashboard navigation and controls change the computed output."""

from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import SECTIONS


def test_all_sections_run_without_app_exceptions():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    for section in SECTIONS:
        app.selectbox(key="section").select(section).run()
        assert not app.exception, (section, app.exception)
        assert not app.warning or section in ("Normalization", "Document vector space")


def test_tokenization_ngram_and_lm_controls_change_computation():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    app.text_area(key="token_text").set_value("é").run()
    app.selectbox(key="strategy").select("byte").run()
    assert app.metric[0].value == "2"
    app.selectbox(key="section").select("N-grams").run()
    app.text_input(key="ngram_text").set_value("a a a").run()
    app.slider(key="ngram_n").set_value(4).run()
    assert '"occurrences": 0' in app.json[0].value
    app.selectbox(key="section").select("Statistical language models").run()
    app.selectbox(key="lm_order").select(3).run()
    app.slider(key="alpha").set_value(.5).run()
    assert not app.exception
    assert app.metric[0].value == "1.0000000000"
