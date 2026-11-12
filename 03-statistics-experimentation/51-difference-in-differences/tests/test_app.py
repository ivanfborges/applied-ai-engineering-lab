"""Headless widget behavior; browser/WebGL rendering is checked separately."""

from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))


def open_tab(app, label):
    app.session_state["chapter"] = label
    app.run()
    assert not app.exception


def test_all_seven_tabs_and_failure_controls():
    app = AppTest.from_file(str(TOPIC / "app.py"), default_timeout=45).run()
    assert not app.exception
    assert len(app.tabs) == 7
    app.slider(key="effect").set_value(6.0).run()
    assert "+6.000" in app.metric[0].value
    for label in ["DiD Estimator", "Parallel Trends", "Regression & Uncertainty", "Event Study", "When DiD Fails", "3D Exploration"]:
        open_tab(app, label)
    open_tab(app, "Parallel Trends")
    app.slider(key="slope").set_value(1.0).run()
    assert not app.exception
    assert app.warning
    open_tab(app, "When DiD Fails")
    for scenario in ["Spillover effects", "Anticipation effects", "Differential untreated trends"]:
        app.selectbox(key="failure").select(scenario).run()
        assert not app.exception
        assert app.warning


def test_sampling_run_and_stale_result_guard():
    app = AppTest.from_file(str(TOPIC / "app.py"), default_timeout=60).run()
    open_tab(app, "Regression & Uncertainty")
    app.select_slider(key="draws").set_value(100).run()
    app.button(key="run_sampling").click().run()
    assert not app.exception
    assert "sampling_run" in app.session_state
    assert any("Mean estimate" in element.value for element in app.markdown)
    app.slider(key="noise").set_value(3.0).run()
    assert not app.exception
    assert any("Parameters changed" in element.value for element in app.info)
    app.slider(key="reg_slope").set_value(2.0).run()
    app.button(key="run_sampling").click().run()
    assert not app.exception
    assert any("boundary estimate" in element.value for element in app.caption)
