"""Local, synthetic visual laboratory for Day 44 ranking metrics."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as lab


st.set_page_config(page_title="Ranking metrics | Day 44", layout="wide")

SECTIONS = ["Ranking playground", "Precision vs recall", "Average precision",
            "Reciprocal rank", "DCG and NDCG", "3D ranking landscape",
            "Metric comparison", "RAG retrieval simulation"]
LABELS = {"precision": "Precision", "recall": "Recall", "ap": "AP", "rr": "RR", "ndcg": "NDCG"}
BLUE, GREEN, INK = "#2463EB", "#23866F", "#152238"


def polish(fig, title, height=380):
    fig.update_layout(template="plotly_white", title=dict(text=title, font_size=17),
                      height=height, margin=dict(l=25, r=25, t=65, b=45),
                      font=dict(family="Arial, sans-serif", color=INK, size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.2),
                      hoverlabel=dict(bgcolor="white", font_color=INK))
    return fig


def show(fig, key):
    st.plotly_chart(fig, key=key, width="stretch", theme=None,
                    config={"displaylogo": False, "scrollZoom": False}, alt=fig.layout.title.text)


def ranking_chart(order, labels, k, title="Current order"):
    ranks = list(range(1, len(order) + 1))
    grades = [int(labels[item]) for item in order]
    fig = go.Figure(go.Bar(
        x=[1] * len(order), y=ranks, orientation="h", width=0.77,
        text=[f"{item}  ·  grade {grade}  ·  {lab.GRADE_NAMES[grade]}" for item, grade in zip(order, grades)],
        textposition="inside", insidetextanchor="start",
        textfont=dict(color=["white" if grade >= 2 else INK for grade in grades], size=13),
        marker=dict(color=[lab.COLORS[grade] for grade in grades],
                    opacity=[1 if rank <= k else 0.3 for rank in ranks]),
        customdata=[[item, grade, "inside K" if rank <= k else "outside K"]
                    for rank, item, grade in zip(ranks, order, grades)],
        hovertemplate="Rank %{y} · %{customdata[0]}<br>Grade %{customdata[1]} · %{customdata[2]}<extra></extra>"))
    fig.update_xaxes(visible=False, range=[0, 1.02])
    fig.update_yaxes(autorange="reversed", tickmode="array", tickvals=ranks,
                     ticktext=[f"{rank:02d}" for rank in ranks], title="Rank", showgrid=False)
    if k < len(order):
        fig.add_hline(y=k + 0.5, line_dash="dot", line_color=BLUE,
                      annotation_text=f"K = {k}", annotation_position="top right")
    return polish(fig, title, max(260, 85 + 34 * len(order)))


def score_panel(order, labels, k, gain, previous=None):
    try:
        scores = lab.evaluate_query(order, labels, k, gain=gain)
    except ValueError as error:
        if "No relevant items" not in str(error):
            raise
        st.warning("No relevant items in the judged universe. Recall, AP and NDCG are undefined. Set at least one grade above zero.")
        scores = {"precision": 0.0, "recall": None, "ap": None, "rr": 0.0, "ndcg": None}
    with st.container(horizontal=True):
        for metric, name in LABELS.items():
            value = scores[metric]
            delta = None
            if previous and value is not None and previous.get(metric) is not None:
                delta = round(value - previous[metric], 4)
            st.metric(f"{name}@{k}", "undefined" if value is None else f"{value:.4f}",
                      delta=delta, delta_color="off", border=True)
    return scores


def load_scenario():
    order, labels = lab.fixture(lab.SCENARIOS[st.session_state.scenario])
    st.session_state.play_order, st.session_state.play_labels = order, labels
    st.session_state.pop("play_previous", None)


def reset_ndcg():
    st.session_state.ndcg_order, st.session_state.ndcg_labels = lab.fixture()


def choose_order(prefix, mode):
    st.session_state[f"{prefix}_order"] = lab.reorder(
        st.session_state[f"{prefix}_order"], st.session_state[f"{prefix}_labels"], mode,
        int(st.session_state.seed))


def swap(prefix):
    order = st.session_state[f"{prefix}_order"].copy()
    a, b = st.session_state[f"{prefix}_from"] - 1, st.session_state[f"{prefix}_to"] - 1
    order[a], order[b] = order[b], order[a]
    st.session_state[f"{prefix}_order"] = order


def set_grade(prefix):
    labels = st.session_state[f"{prefix}_labels"].copy()
    labels[st.session_state[f"{prefix}_document"]] = int(st.session_state[f"{prefix}_grade"])
    st.session_state[f"{prefix}_labels"] = labels


def edit_controls(prefix):
    n = len(st.session_state[f"{prefix}_order"])
    with st.container(horizontal=True):
        st.number_input("Position A", 1, n, 1, key=f"{prefix}_from", width=140)
        st.number_input("Position B", 1, n, n, key=f"{prefix}_to", width=140)
        st.button("Swap positions", key=f"{prefix}_swap", on_click=swap, args=(prefix,))
    with st.container(horizontal=True):
        st.selectbox("Document to edit", sorted(st.session_state[f"{prefix}_labels"]),
                     key=f"{prefix}_document", width=190)
        st.selectbox("New relevance grade", [0, 1, 2, 3], index=3, key=f"{prefix}_grade", width=190)
        st.button("Set grade", key=f"{prefix}_set_grade", on_click=set_grade, args=(prefix,))


def animate_controls(fig):
    fig.update_layout(updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.22,
        buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 1000, "redraw": True},
                     "transition": {"duration": 250}, "fromcurrent": True, "mode": "immediate"}]),
                 dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0},
                     "transition": {"duration": 0}, "mode": "immediate"}])])],
        margin_b=95)
    return fig


def playground(k, gain):
    st.write("Move the same judged documents, then inspect which summaries notice the change. Faded cards stay in the judged universe.")
    for col, formula in zip(st.columns(3), [r"P@K=\frac{\mathrm{hits}@K}{K}",
            r"R@K=\frac{\mathrm{hits}@K}{R_q}", r"AP@K=\frac{\sum_{i=1}^{K}P@i\,r_i}{R_q}"]):
        with col:
            st.latex(formula)
    with st.container(horizontal=True):
        for mode, title in (("ideal", "Ideal order"), ("poor", "Poor order"), ("random", "Seeded shuffle")):
            st.button(title, key=f"play_{mode}", on_click=choose_order, args=("play", mode))
    edit_controls("play")
    order, labels = st.session_state.play_order, st.session_state.play_labels
    previous = st.session_state.get("play_previous")
    scores = score_panel(order, labels, k, gain, previous)
    if all(value is not None for value in scores.values()):
        st.info(lab.explain_change(previous, scores) if previous and all(value is not None for value in previous.values())
                else "Initial state. Swap two cards, change a grade or move K to compare the next state.")
    st.session_state.play_previous = scores
    show(ranking_chart(order, labels, k), "play_ranking")
    st.caption("Gray = 0 · Yellow = 1 · Blue = 2 · Green = 3. Binary metrics treat every grade above zero as relevant.")
    st.markdown("**Try it:** swap two relevant cards inside K. Binary metrics can stay fixed while NDCG changes. Changing grades also changes the full relevant count and ideal gain.")
    st.caption("Production takeaway: define the relevance rubric and evaluation universe before comparing search or recommendation rankings.")


def precision_recall(k, gain):
    order, labels = st.session_state.play_order, st.session_state.play_labels
    st.write("Increase K to spend more result slots. Recall keeps previously found positives; precision depends on the next document.")
    st.latex(r"P@K=\frac{H_K}{K}\qquad R@K=\frac{H_K}{R_q}")
    scores = score_panel(order, labels, k, gain)
    if scores["recall"] is None:
        return
    rows = lab.cutoff_curve(order, labels, gain)
    fig = go.Figure()
    for metric, color in (("precision", BLUE), ("recall", GREEN)):
        fig.add_scatter(x=[row["k"] for row in rows], y=[row[metric] for row in rows],
                        mode="lines+markers", name=LABELS[metric], line_color=color)
        fig.add_scatter(x=[k], y=[scores[metric]], mode="markers", showlegend=False,
                        marker=dict(size=15, color=color, line=dict(width=2, color="white")))
    fig.add_vline(x=k, line_dash="dot", line_color=INK)
    fig.update_xaxes(title="K · available result slots", dtick=1)
    fig.update_yaxes(title="Metric", range=[0, 1.05])
    left, right = st.columns([1.3, 1])
    with left:
        show(polish(fig, "One numerator, two denominators", 430), "pr_curve")
    with right:
        show(ranking_chart(order, labels, k, "Inside and outside K"), "pr_ranking")
    hits = sum(labels[item] > 0 for item in order[:k])
    total = sum(grade > 0 for grade in labels.values())
    st.info(f"{hits} hits: Precision = {hits}/{k}; Recall = {hits}/{total}. R_q includes relevant documents below K. Use the sidebar's four scenarios to compare early, scattered, late and sparse relevance.")
    st.subheader("Same coverage, different internal order")
    a, b, qrels, _ = lab.comparison("Same P/R, different AP")
    comparison_panels(a, b, qrels, 5, gain, "pr_equal")
    st.caption("Production takeaway: a retrieval budget can improve evidence coverage without putting useful evidence early enough for the final display.")


def ap_animation(k, step):
    order, labels = lab.fixture([1, 0, 1, 1, 0, 1])
    rows = lab.trace(order, labels, k)[:k]
    def state(current):
        values = [row["ap_contribution"] if row["rank"] <= current else 0 for row in rows]
        numerator = sum(values)
        text = ("Before scanning" if not current else
                f"Rank {current}: hit={rows[current-1]['relevant']} · cumulative hits={rows[current-1]['hits_so_far']} · P@{current}={rows[current-1]['precision_at_rank']:.3f}")
        return go.Bar(x=list(range(1, k + 1)), y=values,
                      marker_color=[lab.COLORS[int(labels[item])] for item in order[:k]],
                      text=[f"{value:.3f}" if value else "" for value in values], textposition="outside"), dict(
            annotations=[dict(text=text + f"<br>AP numerator = {numerator:.4f}; running AP = {numerator:.4f}/4 = {numerator/4:.4f}",
                              x=0, y=1.18, xref="paper", yref="paper", showarrow=False, xanchor="left")],
            shapes=[] if not current else [dict(type="line", x0=current, x1=current, y0=0, y1=1.12,
                                                line=dict(color=INK, width=2, dash="dot"))])
    bar, layout = state(step)
    fig = go.Figure(data=[bar], frames=[go.Frame(name=str(i), data=[state(i)[0]], layout=state(i)[1]) for i in range(k+1)])
    polish(fig, "Only relevant ranks add precision to the numerator", 410)
    fig.update_layout(**layout, margin_t=110)
    fig.update_xaxes(title="Scanned rank i", dtick=1)
    fig.update_yaxes(title="Contribution P@i × relevance_i", range=[0, 1.18])
    return animate_controls(fig)


def average_precision(k, gain):
    k = min(k, 6)
    st.write("Scan [1, 0, 1, 1, 0, 1]. The sixth item remains relevant even when K=5. Its missing contribution is still penalized by the denominator of four.")
    st.latex(r"AP@K=\frac{\sum_{i=1}^{K}P@i\,r_i}{R_q},\quad R_q=4")
    step = st.slider("AP scan step", 0, k, min(st.session_state.get("ap_step", 0), k), key="ap_step")
    order, labels = lab.fixture([1, 0, 1, 1, 0, 1])
    rows = lab.trace(order, labels, k)
    score_panel(order, labels, k, gain)
    show(ap_animation(k, step), "ap_animation")
    st.caption("Play animates the chart's annotations. The step slider controls the table and values below; cards above always show the final metric at K.")
    if step:
        row = rows[step - 1]
        st.info(f"Rank {step}: {'relevant' if row['relevant'] else 'irrelevant'}. {row['hits_so_far']} hits / {step} positions = {row['precision_at_rank']:.4f}. Add {row['ap_contribution']:.4f}; numerator {row['ap_numerator']:.4f}; running AP {row['running_ap']:.4f}.")
    st.dataframe(pd.DataFrame(rows[:step], columns=["rank", "relevant", "hits_so_far", "precision_at_rank", "ap_contribution", "ap_numerator", "running_ap"]), hide_index=True)
    earlier = order.copy()
    earlier[:k] = sorted(earlier[:k], key=lambda item: -labels[item])
    initial, improved = lab.evaluate_query(order, labels, k), lab.evaluate_query(earlier, labels, k)
    st.markdown(f"**Same selected documents, earlier hits:** AP@{k} changes from **{initial['ap']:.4f}** to **{improved['ap']:.4f}** when the top-K hits are grouped first. Precision and recall stay fixed. At K=5, the original AP is 0.6042.")
    comparison_panels(order, earlier, labels, k, gain, "ap_compare")
    st.caption("Production takeaway: AP is useful when multiple relevant sources matter. Here its denominator is R_q, never min(R_q, K).")


def movement_figure(kind, initial_position=5, k=5, gain="exponential", selected_position=5):
    if kind == "rr":
        base, labels = lab.first_hit_order(5)
        selected_id = "D05"
    else:
        base, labels = lab.fixture()
        selected_id = "D01"
    rest = [item for item in base if item != selected_id]
    def state(position):
        order = rest.copy()
        order.insert(position - 1, selected_id)
        values = lab.evaluate_query(order, labels, k, gain=gain)
        rows = lab.trace(order, labels, k, gain)
        height = [float(labels[item] > 0) for item in order] if kind == "rr" else [row["dcg_contribution"] for row in rows]
        text = (f"First relevant rank = {position} · RR@{k} = 1/{position} = {values['rr']:.4f}" if kind == "rr" else
                f"Grade 3 at rank {position} · {gain} gain · NDCG@{k} = {values['ndcg']:.4f}")
        bar = go.Bar(x=list(range(1, len(order)+1)), y=height,
                     marker_color=[lab.COLORS[int(labels[item])] for item in order],
                     text=order, textposition="outside", hovertemplate="Rank %{x} · %{text}<br>Contribution %{y:.4f}<extra></extra>")
        return bar, dict(annotations=[dict(text=text, x=0, y=1.12, xref="paper", yref="paper", showarrow=False, xanchor="left")])
    bar, layout = state(selected_position)
    fig = go.Figure(data=[bar], frames=[go.Frame(name=str(pos), data=[state(pos)[0]], layout=state(pos)[1]) for pos in range(initial_position, 0, -1)])
    polish(fig, "Move one relevant document toward rank one", 390)
    fig.update_layout(**layout, margin_t=100)
    fig.update_xaxes(title="Rank", dtick=1)
    fig.update_yaxes(title="Binary relevance" if kind == "rr" else "Discounted gain contribution", range=[0, 1.3 if kind == "rr" else 8.5])
    return animate_controls(fig)


def reciprocal_rank(k, gain):
    st.write("Only the first acceptable result matters to RR. Move the one relevant document from rank five to rank one.")
    st.latex(r"RR@K=\begin{cases}1/h_q & h_q\leq K\\0 & \text{otherwise}\end{cases}")
    st.latex(r"MRR@K=\frac{1}{Q}\sum_q RR@K(q)")
    position = st.slider("First relevant position", 1, 5, 5, key="rr_position")
    order, labels = lab.first_hit_order(position)
    st.metric("RR@5 for this query", f"{lab.evaluate_query(order, labels, 5)['rr']:.4f}", border=True)
    show(movement_figure("rr", selected_position=position), "rr_animation")
    curve = go.Figure(go.Scatter(x=list(range(1, 6)), y=[1/i for i in range(1, 6)], mode="lines+markers", line_color=BLUE))
    curve.add_scatter(x=[position], y=[1/position], mode="markers", marker=dict(size=16, color=GREEN), showlegend=False)
    curve.update_xaxes(title="First relevant rank", dtick=1)
    curve.update_yaxes(title="Reciprocal rank", range=[0, 1.1])
    show(polish(curve, "The first hit has a reciprocal reward", 280), "rr_curve")
    st.subheader("Three queries → one MRR")
    rankings, qrels = {}, {}
    for col, query, default in zip(st.columns(3), ["Q1", "Q2", "Q3"], [1, 3, 5]):
        with col:
            hit = st.number_input(f"{query} first hit (0 = absent)", 0, 5, default, key=f"rr_{query}")
            rankings[query], qrels[query] = lab.first_hit_order(hit)
    result = lab.evaluate_queries(rankings, qrels, 5)
    st.dataframe(pd.DataFrame([{"query": query, "RR@5": row["rr"]} for query, row in result["per_query"].items()]), hide_index=True)
    st.metric("MRR@5 across three queries", f"{result['macro']['mrr']:.4f}", border=True)
    st.subheader("One hit versus five hits: RR stays one")
    ids, same_universe = lab.fixture([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    comparison_panels([ids[0], *ids[5:9]], ids[:5], same_universe, 5, gain, "rr_blind")
    st.info("The displayed relevance patterns are [1,0,0,0,0] and [1,1,1,1,1]. Both use the same ten-item universe with five positives. RR=1 for both, but recall and AP reveal the additional recovered evidence.")
    st.caption("Production takeaway: report RR for individual queries and MRR for a collection; add coverage metrics when several pieces of evidence are needed.")


def ndcg(k, gain):
    k = min(k, 6)
    st.write("Grades produce gain; rank produces a discount. Multiply them, sum the selected contributions, then normalize against an ideal ordering of all judged IDs.")
    for col, formula in zip(st.columns(2), [r"DCG@K=\sum_{i=1}^{K}\frac{G(g_i)}{\log_2(i+1)}",
                                           r"NDCG@K=\frac{DCG@K}{IDCG@K}"]):
        with col:
            st.latex(formula)
    st.caption(f"Active gain: {gain} — " + ("G(g)=2^g−1" if gain == "exponential" else "G(g)=g") + f". Dedicated six-item example, K={k}.")
    st.button("Reset NDCG example", key="ndcg_reset", on_click=reset_ndcg)
    edit_controls("ndcg")
    order, labels = st.session_state.ndcg_order, st.session_state.ndcg_labels
    scores = score_panel(order, labels, k, gain)
    if scores["ndcg"] is None:
        return
    details = lab.dcg_details(order, labels, k, gain)
    with st.container(horizontal=True):
        st.metric(f"DCG@{k}", f"{details['dcg']:.4f}", border=True)
        st.metric(f"IDCG@{k}", f"{details['idcg']:.4f}", border=True)
        st.metric("Achieved / ideal gain", f"{details['dcg']:.4f} / {details['idcg']:.4f}", border=True)
    for col, ranking, title in zip(st.columns(2), [order, details["ideal_order"]], ["Predicted ranking", "Ideal over all six judged IDs"]):
        with col:
            show(ranking_chart(ranking, labels, k, title), f"ndcg_{title}")
    rows = details["rows"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = [row["rank"] for row in rows]
    fig.add_trace(go.Bar(x=x, y=[row["gain"] for row in rows], name="Undiscounted gain", marker_color="#B4C4D9"), secondary_y=False)
    fig.add_trace(go.Bar(x=x, y=[row["dcg_contribution"] for row in rows], name="Selected discounted gain",
                         marker_color=[lab.COLORS[int(row["grade"])] for row in rows]), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=[row["discount"] for row in rows], name="Discount (right axis)", mode="lines+markers", line_color=INK), secondary_y=True)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Rank", dtick=1)
    fig.update_yaxes(title="Gain", secondary_y=False)
    fig.update_yaxes(title="Discount", range=[0, 1.1], secondary_y=True)
    show(polish(fig, "Gain × discount = contribution; only ranks ≤ K count", 400), "ndcg_breakdown")
    st.dataframe(pd.DataFrame(rows)[["rank", "document", "grade", "inside_k", "gain", "discount", "dcg_contribution"]], hide_index=True)
    st.info(f"Current DCG / IDCG = {scores['ndcg']:.4f}. The ideal denominator includes every judged item, even below K. Under exponential gain, grades 1, 2 and 3 contribute 1, 3 and 7 before discounting.")
    st.subheader("Watch a grade-three document move forward")
    show(movement_figure("ndcg", k=k, gain=gain), "ndcg_animation")
    st.caption("This animation uses the original fixed six-item fixture, independently of the edits above. Play changes the chart's NDCG annotation.")
    st.markdown("**Why grade matters:** for the same rank-discount increase Δ, a grade-three document gains 7Δ under exponential gain, versus 1Δ for grade one. A swap's total change also includes the displaced document.")
    st.caption("Production takeaway: gain and discount encode utility assumptions. A high NDCG alone does not prove evidence completeness or answer correctness.")


@st.cache_data(max_entries=12, show_spinner=False)
def cached_landscape(k, gain):
    return lab.landscape(k, gain)


def ranking_landscape(k, gain):
    k = min(k, 8)
    st.write("Place grade-three D01 and grade-two D02 at distinct ranks. The other six documents keep their relative order. Every cell is an evaluated ranking, not an interpolated state.")
    st.latex(r"z(p_3,p_2)=NDCG@K(\pi(p_3,p_2)),\qquad p_3\ne p_2")
    rows, labels = cached_landscape(k, gain)
    best, worst = max(rows, key=lambda row: row["ndcg"]), min(rows, key=lambda row: row["ndcg"])
    matrix = np.full((8, 8), np.nan)
    for row in rows:
        matrix[row["medium_position"] - 1, row["high_position"] - 1] = row["ndcg"]
    heat = go.Figure(go.Heatmap(z=matrix, x=list(range(1, 9)), y=list(range(1, 9)),
                               zmin=0, zmax=1, colorscale="Blues", colorbar_title="NDCG",
                               hoverongaps=False, hovertemplate="Grade 3 rank %{x}<br>Grade 2 rank %{y}<br>NDCG %{z:.4f}<extra></extra>"))
    heat.update_xaxes(title="Grade 3 position", dtick=1)
    heat.update_yaxes(title="Grade 2 position", dtick=1, autorange="reversed")
    for row, color, name in ((best, GREEN, "Best (one if tied)"), (worst, "#BB5260", "Worst (one if tied)")):
        heat.add_scatter(x=[row["high_position"]], y=[row["medium_position"]], mode="markers", name=name,
                         marker=dict(size=13, color=color, symbol="diamond", line=dict(color="white", width=2)))
    show(polish(heat, f"56 valid rankings · diagonal is impossible · K={k}", 450), "landscape_heatmap")
    selected_x = st.slider("Grade 3 position", 1, 8, 1, key="landscape_high")
    valid_y = [rank for rank in range(1, 9) if rank != selected_x]
    selected_y = st.selectbox("Grade 2 position (distinct)", valid_y, key="landscape_medium")
    selected = next(row for row in rows if row["high_position"] == selected_x and row["medium_position"] == selected_y)
    score_panel(selected["ranking"], labels, k, gain)
    show(ranking_chart(selected["ranking"], labels, k, "Selected valid configuration"), "landscape_selected")
    if st.toggle("Explore the same values in 3D", value=True, key="show_3d"):
        fig = go.Figure(go.Scatter3d(x=[row["high_position"] for row in rows],
            y=[row["medium_position"] for row in rows], z=[row["ndcg"] for row in rows],
            mode="markers", marker=dict(size=6, color=[row["ndcg"] for row in rows], colorscale="Blues", cmin=0, cmax=1),
            text=[" → ".join(row["ranking"]) for row in rows],
            hovertemplate="Grade 3 rank %{x}<br>Grade 2 rank %{y}<br>NDCG %{z:.4f}<br>%{text}<extra></extra>", name="Valid configurations"))
        for row, color, name in ((best, GREEN, "Best (one if tied)"), (worst, "#BB5260", "Worst (one if tied)"), (selected, "#E5AE38", "Selected")):
            fig.add_scatter3d(x=[row["high_position"]], y=[row["medium_position"]], z=[row["ndcg"]],
                             mode="markers", name=name, marker=dict(size=10, color=color, symbol="diamond"))
        fig.update_layout(scene=dict(xaxis_title="Grade 3 position", yaxis_title="Grade 2 position", zaxis_title="NDCG@K", zaxis_range=[0, 1.05]), uirevision=f"landscape-{k}-{gain}")
        show(polish(fig, "Rotate to inspect the discrete utility landscape", 500), "landscape_3d")
    for col, row, title in zip(st.columns(2), [best, worst], ["Best valid configuration", "Worst valid configuration"]):
        with col:
            st.markdown(f"**{title}** · NDCG={row['ndcg']:.4f} (one if tied)")
            show(ranking_chart(row["ranking"], labels, k, title), f"landscape_{title}")
    st.caption("Production takeaway: early placement has greater discounted value, especially for large gains. The heatmap often communicates this more clearly than 3D; both show exactly the same 56 states.")


def comparison_panels(a, b, labels, k, gain, prefix):
    rows = []
    for col, order, system in zip(st.columns(2), [a, b], ["System A", "System B"]):
        with col:
            show(ranking_chart(order, labels, k, system), f"{prefix}_{system}")
        rows.append({"System": system, **lab.evaluate_query(order, labels, k, gain=gain)})
    st.dataframe(pd.DataFrame(rows).rename(columns=LABELS), hide_index=True,
                 column_config={name: st.column_config.NumberColumn(format="%.4f") for name in LABELS.values()})
    return rows


def metric_comparison(k, gain):
    name = st.selectbox("Comparison scenario", list(lab.COMPARISONS), key="comparison_scenario")
    a, b, labels, reference = lab.comparison(name)
    st.write("Both systems rank the same judged documents. The objective determines which differences matter; there is no universal winner.")
    st.latex(r"\pi_A\ne\pi_B,\quad K_A=K_B,\quad \mathcal{D}_A=\mathcal{D}_B")
    use_reference = st.toggle("Use the scenario's illustrative cutoff", value=True, key="comparison_reference")
    chosen_k = reference if use_reference else min(k, 6)
    rows = comparison_panels(a, b, labels, chosen_k, gain, "comparison")
    preferences = []
    for metric, label in LABELS.items():
        first, second = rows[0][metric], rows[1][metric]
        preference = "tie" if np.isclose(first, second) else "A" if first > second else "B"
        preferences.append(f"{label}: {preference}")
    st.info(f"At K={chosen_k}, {gain} gain — " + " · ".join(preferences))
    st.write("Precision and recall count selected positives; AP also uses their ranks; RR stops at the first hit; NDCG distinguishes grades and discounts. Changing K can remove or reverse a displayed disagreement.")
    st.caption("Production takeaway: choose an acceptable-answer, multi-evidence or graded-utility objective before selecting a metric. Inspect the per-query trade-offs rather than declaring a universally superior system.")


def rag(k, gain):
    st.write("A larger candidate pool creates opportunities; a reranker only reorders the available IDs. Scores here are generated from grades plus seeded noise, not learned from data.")
    st.latex(r"\begin{aligned}Recall@N&=\frac{|C_N\cap Rel|}{R_q}\\ Recall@K&\leq Recall@N\quad(K\leq N)\end{aligned}")
    n = st.slider("Candidate pool N", 1, 20, 8, key="rag_n")
    budget = st.slider("Hypothetical context budget (tokens)", 180, 3600, 900, step=180, key="rag_budget")
    result = lab.rag_simulation(int(st.session_state.seed), n, k, budget, gain)
    effective = result["effective_k"]
    st.markdown(f"**Query → retrieve {n} candidates → reorder the same {n} IDs → top {effective} context documents → hypothetical LLM context**")
    st.caption(f"Requested K={k}; effective K=min(K, N, budget/180)={effective}. Equal synthetic size: 180 tokens per document; {result['context_tokens']} of {budget} tokens used. No real tokenizer or LLM is involved.")
    st.metric(f"Stage A · corpus Recall@{n}", f"{result['candidate_recall']:.4f}", border=True)
    st.subheader("Stage B · before and after reranking")
    rows = comparison_panels(result["candidates"], result["reranked"], result["judgments"], effective, gain, "rag")
    st.caption("System A = retriever order; System B = reranker order. AP and NDCG retain the complete 20-document judged universe; neither normalizes only over candidates.")
    table = []
    for item in result["candidates"]:
        table.append({"document": item, "grade": result["judgments"][item],
                      "retriever rank": result["candidates"].index(item)+1,
                      "reranker rank": result["reranked"].index(item)+1,
                      "retriever score": result["retriever_scores"][item],
                      "reranker score": result["reranker_scores"][item],
                      "in final context": item in result["reranked"][:effective]})
    movement = go.Figure()
    for row in table:
        movement.add_scatter(x=["Retrieved", "Reranked"], y=[row["retriever rank"], row["reranker rank"]],
                             text=[row["document"]]*2, mode="lines+markers+text", textposition="middle right",
                             line=dict(color=lab.COLORS[row["grade"]]), showlegend=False,
                             hovertemplate="%{text}<br>Rank %{y}<extra></extra>")
    movement.add_hline(y=effective+0.5, line_dash="dot", annotation_text="Context cutoff")
    movement.update_yaxes(autorange="reversed", title="Candidate position", dtick=1)
    show(polish(movement, "The same documents move; excluded IDs cannot enter", 430), "rag_movement")
    st.dataframe(pd.DataFrame(table), hide_index=True)
    missed = [item for item in result["excluded"] if result["judgments"][item] > 0]
    st.info(f"Relevant documents outside the pool: {', '.join(missed) or 'none'}. Candidate recall stays {result['candidate_recall']:.4f} after reranking. End-to-end recall at K={effective} is {rows[1]['recall']:.4f}.")
    with st.expander("Inspect every excluded document"):
        st.dataframe(pd.DataFrame([{"document": item, "grade": result["judgments"][item], "retriever score": result["retriever_scores"][item]}
                                   for item in result["excluded"]], columns=["document", "grade", "retriever score"]), hide_index=True)
    curves = lab.cutoff_curve(result["reranked"], result["judgments"], gain)[:n]
    fig = go.Figure()
    for metric, color in (("recall", GREEN), ("ndcg", BLUE)):
        fig.add_scatter(x=[row["k"] for row in curves], y=[row[metric] for row in curves], name=LABELS[metric], mode="lines+markers", line_color=color)
    fig.add_vline(x=effective, line_dash="dot", annotation_text="Effective K")
    fig.update_xaxes(title="Context documents (180 hypothetical tokens each)", dtick=1)
    fig.update_yaxes(title="Metric", range=[0, 1.05])
    show(polish(fig, "More context changes coverage and discounted utility", 330), "rag_context_curve")
    st.warning("Retrieval quality is not generated-answer quality. These metrics do not check factual grounding, citations, contradictions, evidence sufficiency or whether an LLM uses the context correctly.")
    st.caption("Production takeaway: evaluate candidate recovery, final ordering, answer grounding, latency and cost separately. A reranker can improve order and still leave crucial evidence unavailable.")


st.session_state.setdefault("scenario", "Scattered evidence")
st.session_state.setdefault("seed", 44)
if "play_order" not in st.session_state:
    load_scenario()
if "ndcg_order" not in st.session_state:
    reset_ndcg()

with st.sidebar:
    st.caption("APPLIED AI ENGINEERING LAB · DAY 44")
    st.title("Ranking lab")
    section = st.radio("Explore a concept", SECTIONS, key="section")
    st.selectbox("Query scenario", list(lab.SCENARIOS), key="scenario", on_change=load_scenario)
    k = st.slider("Shared cutoff K", 1, 10, 5, key="cutoff")
    gain = st.selectbox("NDCG gain convention", ["exponential", "linear"], key="gain")
    st.number_input("Random seed", 0, 2**32-1, key="seed", step=1)
    st.button("Reset dataset", on_click=load_scenario, key="reset_dataset")
    st.caption("Scenario and reset control the playground and precision/recall views. Dedicated six-item examples cap K at six; the landscape caps it at eight. RAG uses this seed and K with its own pool and budget.")
    st.caption("Synthetic, fully judged data · no network or model API · interpretations require author review.")

st.caption("VISUAL LEARNING / INFORMATION RETRIEVAL / RECOMMENDATION / RAG")
st.title(section)
st.caption(f"Day 44 · {gain} gain for NDCG · AP denominator = all relevant judged documents")
handlers = dict(zip(SECTIONS, [playground, precision_recall, average_precision, reciprocal_rank,
                              ndcg, ranking_landscape, metric_comparison, rag]))
try:
    handlers[section](k, gain)
except ValueError as error:
    st.error(f"This configuration cannot be evaluated: {error}")
st.caption("Learning lab, not a production evaluator. All comparisons are constructed demonstrations; author review required.")
