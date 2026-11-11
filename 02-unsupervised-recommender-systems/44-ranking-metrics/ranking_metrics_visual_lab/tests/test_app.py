"""Headless interaction coverage; browser/Plotly animation checks are separate."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
SECTIONS = ["Ranking playground", "Precision vs recall", "Average precision", "Reciprocal rank",
            "DCG and NDCG", "3D ranking landscape", "Metric comparison", "RAG retrieval simulation"]


def start(section="Ranking playground"):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    if section != SECTIONS[0]:
        app.radio(key="section").set_value(section).run()
    return app


@pytest.mark.parametrize("section", SECTIONS)
def test_all_sections_render_without_exceptions(section):
    app = start(section)
    assert not app.exception
    assert not app.error
    assert app.get("plotly_chart")


def test_swap_grade_shuffle_reset_and_shared_cutoff():
    app = start()
    initial = app.session_state["play_order"].copy()
    app.number_input(key="play_to").set_value(4).run()
    app.button(key="play_swap").click().run()
    assert app.session_state["play_order"][0] == initial[3]
    app.selectbox(key="play_document").set_value("D01").run()
    app.selectbox(key="play_grade").set_value(0).run()
    app.button(key="play_set_grade").click().run()
    assert app.session_state["play_labels"]["D01"] == 0
    app.button(key="play_random").click().run()
    shuffled = app.session_state["play_order"].copy()
    app.button(key="play_random").click().run()
    assert app.session_state["play_order"] == shuffled
    app.slider(key="cutoff").set_value(3).run()
    assert app.metric[0].label == "Precision@3"
    app.button(key="reset_dataset").click().run()
    assert app.session_state["play_order"] == initial
    assert app.session_state["play_labels"]["D01"] == 3
    assert not app.exception


def test_no_positive_edits_show_undefined_values_without_crashing():
    app = start()
    app.session_state["play_labels"] = {item: 0 for item in app.session_state["play_order"]}
    app.run()
    assert not app.exception
    assert "undefined" in [metric.value for metric in app.metric]
    assert app.warning


def test_ap_steps_cutoff_changes_and_ndcg_gain_toggle():
    app = start("Average precision")
    assert app.metric[2].value == "0.6042"
    app.slider(key="ap_step").set_value(5).run()
    app.slider(key="cutoff").set_value(1).run()
    assert not app.exception and not app.error
    assert app.slider(key="ap_step").value <= 1
    app.slider(key="cutoff").set_value(5).run()
    app.radio(key="section").set_value("DCG and NDCG").run()
    assert app.metric[4].value == "0.8251"
    app.selectbox(key="gain").set_value("linear").run()
    assert app.metric[4].value == "0.7783"
    app.button(key="ndcg_swap").click().run()
    assert not app.exception and not app.error


def test_landscape_excludes_collisions_and_rag_small_pool():
    app = start("3D ranking landscape")
    app.slider(key="landscape_high").set_value(2).run()
    assert "2" not in app.selectbox(key="landscape_medium").options
    app.selectbox(key="landscape_medium").set_value(8).run()
    app.radio(key="section").set_value("RAG retrieval simulation").run()
    app.slider(key="rag_n").set_value(1).run()
    app.slider(key="rag_budget").set_value(180).run()
    assert not app.exception and not app.error
    assert "0.1111" == app.metric[0].value
