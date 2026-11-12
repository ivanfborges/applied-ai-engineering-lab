"""Headless Streamlit checks; separate from real-browser evidence."""
from pathlib import Path
import sys
from streamlit.testing.v1 import AppTest

TOPIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPIC))


def test_all_eight_modules_and_reset():
    app = AppTest.from_file(str(TOPIC / "app.py"), default_timeout=60).run()
    assert not app.exception
    options = list(app.selectbox(key="module").options)
    assert len(options) == 8
    for module in options:
        app.selectbox(key="module").set_value(module).run()
        assert not app.exception, module
    app.number_input(key="baseline").set_value(0.2).run()
    next(button for button in app.button if button.label == "Reset to defaults").click().run()
    assert app.number_input(key="baseline").value == 0.1
    assert app.selectbox(key="module").value == options[0]
    assert not app.exception


def test_simulation_is_explicit_and_retains_run_configuration():
    app = AppTest.from_file(str(TOPIC / "app.py"), default_timeout=60).run()
    app.selectbox(key="module").set_value("05 · Monte Carlo simulation").run()
    assert "simulation_result" not in app.session_state
    next(button for button in app.button if button.label == "Run simulation").click().run()
    assert not app.exception
    initial = app.session_state["simulation_result"]["rejection_rate"]
    app.number_input(key="alpha").set_value(0.06).run()
    assert app.session_state["simulation_result"]["rejection_rate"] == initial
    assert any("Inputs changed" in warning.value for warning in app.warning)
    app.selectbox(key="simulation_mode").set_value("H0 · no population effect")
    next(button for button in app.button if button.label == "Run simulation").click().run()
    assert app.session_state["simulation_result"]["config"]["mode"] == "null"
    assert not app.exception


def test_invalid_population_and_zero_effect_do_not_crash():
    app = AppTest.from_file(str(TOPIC / "app.py"), default_timeout=60).run()
    app.number_input(key="baseline").set_value(0.999).run()
    assert app.error
    assert not app.exception
    app.number_input(key="effect_pp").set_value(0.0).run()
    assert not app.exception
