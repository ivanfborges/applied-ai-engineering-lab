"""Application behavior stays independent of browser or external services."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "interactive_demo.py"


def test_controls_greedy_presets_and_reset():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    app.slider(key="temperature").set_value(0.0).run()
    assert not app.exception
    assert app.metric[0].value == "-0.0000" or app.metric[0].value == "0.0000"
    assert app.metric[3].value == "1"
    app.segmented_control(key="preset").set_value("Diffuse").run()
    app.slider(key="temperature").set_value(3.0).run()
    assert int(app.metric[3].value) == 10
    app.slider(key="top_k").set_value(3).run()
    app.slider(key="top_p").set_value(0.5).run()
    assert int(app.metric[3].value) <= 3
    app.button(key="reset").click().run()
    assert app.slider(key="temperature").value == 1.0
    assert app.slider(key="top_k").value == 10
    assert app.slider(key="top_p").value == 1.0
    assert app.segmented_control(key="preset").value == "Moderate"
    assert not app.exception


def test_every_section_and_zero_budget():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    for section in app.selectbox(key="section").options:
        app.selectbox(key="section").select(section).run()
        assert not app.exception, section
        if section == "Stopping conditions":
            app.slider(key="budget").set_value(0).run()
            assert app.metric[0].value == "length"
            assert app.metric[1].value == "False"
        if section == "Sequence search":
            app.slider(key="beam_width").set_value(1).run()
            assert not app.exception
