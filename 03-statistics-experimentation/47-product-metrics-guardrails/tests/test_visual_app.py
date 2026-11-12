"""Interactive application checks in Streamlit; browser rendering is separate."""
from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

TOPIC=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TOPIC))


def test_all_nine_sections_and_reset():
    app=AppTest.from_file(str(TOPIC/"app.py"),default_timeout=60).run()
    assert not app.exception
    sections=list(app.selectbox(key="module").options)
    assert len(sections)==9
    for section in sections:
        app.selectbox(key="module").set_value(section).run()
        assert not app.exception,section
    assert "export_paths" not in app.session_state
    app.number_input(key="treatment_conversion").set_value(.3).run()
    app.button(key="reset").click().run()
    assert app.number_input(key="treatment_conversion").value==.15
    assert app.selectbox(key="module").value==sections[0]
    assert not app.exception


def test_controls_change_metrics_and_zero_denominator_is_safe():
    app=AppTest.from_file(str(TOPIC/"app.py"),default_timeout=60).run()
    initial=app.session_state["observed_lift"]
    app.number_input(key="treatment_conversion").set_value(.8).run()
    assert app.session_state["observed_lift"]>initial
    app.slider(key="n").set_value(500).run()
    assert app.session_state["sample_counts"]=={"control":500,"treatment":500}
    app.number_input(key="control_conversion").set_value(0)
    app.number_input(key="treatment_conversion").set_value(0).run()
    app.selectbox(key="module").set_value("07 · Simulated rollout decision").run()
    assert not app.exception
    assert app.session_state["decision_status"]!="Candidate meets the simulated rollout criteria"


def test_hard_stop_and_imperfect_detection_are_visible():
    app=AppTest.from_file(str(TOPIC/"app.py"),default_timeout=60).run()
    app.selectbox(key="module").set_value("07 · Simulated rollout decision").run()
    app.checkbox(key="inject_critical").set_value(True).run()
    assert "violates" in app.session_state["decision_status"]
    app.slider(key="detection").set_value(.5).run()
    assert not app.exception
    table=app.dataframe[0].value
    assert table.loc[table.check=="Safety incidence degradation","status"].iloc[0]=="INCONCLUSIVE"


def test_export_button_uses_current_configuration_only_on_request(monkeypatch,tmp_path):
    import animations
    calls=[]
    def export(*args):
        calls.append(args)
        return [tmp_path/"latency.gif",tmp_path/"uncertainty.gif"]
    monkeypatch.setattr(animations,"export_gifs",export)
    app=AppTest.from_file(str(TOPIC/"app.py"),default_timeout=60).run()
    app.number_input(key="treatment_median").set_value(3.).run()
    app.selectbox(key="module").set_value("09 · Export GIFs").run()
    assert not calls
    app.button(key="export_gifs").click().run()
    assert len(calls)==1
    assert calls[0][2].median==3.
    assert "export_paths" in app.session_state
    app.run()
    assert len(calls)==1
    assert not app.exception
