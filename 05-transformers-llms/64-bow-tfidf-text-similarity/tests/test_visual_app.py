"""Dashboard query policies, score explanations and view switching."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest


def test_dashboard_methods_empty_query_and_views():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "retrieval_dashboard.py"), default_timeout=45).run()
    assert not app.exception
    assert app.metric[0].value != "0"
    app.segmented_control(key="method").set_value("bm25").run()
    assert not app.exception
    assert app.slider(key="k1").value == 1.2
    app.slider(key="b").set_value(0.).run()
    assert not app.exception
    app.text_input(key="query").set_value("").run()
    assert not app.exception
    assert app.metric[0].value == "0"
    assert "No lexical evidence" in app.info[0].value
    app.text_input(key="query").set_value("document retrieval").run()
    app.segmented_control(key="method").set_value("char").run()
    assert not app.exception
    assert len(app.dataframe[0].value) > 0
    app.segmented_control(key="view").set_value("Sparse storage").run()
    assert not app.exception
    assert int(app.metric[0].value) > int(app.metric[1].value)
    app.segmented_control(key="view").set_value("Evaluation").run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 6
