"""Local Python-backed decoding playground; no network or model downloads."""

import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

TOPIC = Path(__file__).resolve().parent
if str(TOPIC) not in sys.path:
    sys.path.insert(0, str(TOPIC))

from visual_core import (LabConfig, PRESETS, TOKENS, STRUCTURED_VOCABULARY, compute_group,
                         landscape, stages, sample_path, structured_logits, trace_generation)
from visual_render import BLUE, GREEN, GRAY, ORANGE, stepped_landscape


DEFAULTS = dict(preset="Moderate", temperature=1.0, top_k=10, top_p=1.0, seed=275, trials=10_000)


def reset():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


@st.cache_data(max_entries=12)
def cached_grid(preset, top_k):
    return landscape(PRESETS[preset], np.linspace(0.1, 3, 25), np.linspace(0.05, 1, 25), top_k)


def playground(logits, options, seed, trials):
    state = stages(logits, **options)
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(TOKENS), trials, p=state["final"])
    empirical = np.bincount(draws, minlength=len(TOKENS)) / trials
    selected = int(draws[0])
    with st.container(horizontal=True):
        st.metric("Entropy (bits)", f"{state['entropy']:.4f}")
        st.metric("Effective vocabulary", f"{state['effective']:.3f}")
        st.metric("Maximum probability", f"{state['maximum']:.4f}")
        st.metric("Candidates", str(state["candidates"]))
        st.metric("Original mass retained", f"{state['retained_original_mass']:.4f}")
    st.info(f"First seeded draw: {TOKENS[selected]} (token {selected}). Transformations are deterministic; token selection uses PCG64.")
    headings = ("1. Original logits", "2. Scaled logits z/T", "3. Temperature softmax",
                "4. Top-k retained mass (raw)", "5. Top-p retained mass (after k normalization)", "6. Final normalized policy")
    figure = make_subplots(rows=2, cols=3, subplot_titles=headings)
    values = [state["logits"], state["scaled_logits"], state["tempered"], state["raw_k"], state["raw_p"], state["final"]]
    for index, data in enumerate(values):
        row, column = index // 3 + 1, index % 3 + 1
        if data is None:
            figure.add_annotation(text="T=0: greedy, no division", x=0.5, y=0.5, row=row, col=column, showarrow=False)
        else:
            colors = BLUE if index < 3 else [ORANGE if index == 5 and token == selected else GREEN if value > 0 else GRAY
                                            for token, value in enumerate(data)]
            figure.add_trace(go.Bar(x=list(TOKENS), y=data, marker_color=colors), row=row, col=column)
        figure.update_yaxes(title_text="Logit" if index < 2 else "Probability / mass", row=row, col=column)
    figure.update_layout(height=650, showlegend=False, template="plotly_white", margin=dict(t=60, b=40))
    st.plotly_chart(figure, key="pipeline_chart", width="stretch", alt="Six stages from original logits to the filtered policy; orange marks the sampled token")
    frequencies = go.Figure()
    frequencies.add_bar(name="Theoretical", x=TOKENS, y=state["final"], marker_color=BLUE)
    frequencies.add_bar(name="Empirical", x=TOKENS, y=empirical, marker_color=ORANGE)
    frequencies.update_layout(barmode="group", yaxis_title="Probability / empirical frequency", template="plotly_white")
    st.plotly_chart(frequencies, key="frequencies_chart", width="stretch", alt="Theoretical probabilities compared with actual categorical sampling frequencies")
    tv = float(0.5 * np.abs(empirical - state["final"]).sum())
    st.metric("Total variation distance", f"{tv:.5f}")
    st.caption("The RNG resets to the chosen seed on each UI rerun. Empirical frequencies are finite-sample observations, not exact probabilities.")
    st.dataframe(dict(token=TOKENS, original=state["original"], softmax=state["tempered"],
                      top_k_raw=state["raw_k"], top_k_normalized=state["after_k"], top_p_raw=state["raw_p"],
                      final=state["final"], empirical=empirical), alt="Exact numerical values for every decoding stage")


def main():
    st.set_page_config(page_title="LLM decoding laboratory", layout="wide")
    st.title("LLM inference and decoding laboratory")
    st.caption("Synthetic logits and inspectable toy transitions. No pretrained model, API call or production benchmark.")
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
    with st.sidebar:
        st.header("Decoding controls")
        st.button("Reset controls", key="reset", on_click=reset)
        st.segmented_control("Distribution", list(PRESETS), key="preset", required=True)
        st.slider("Temperature", 0.0, 3.0, step=0.05, key="temperature")
        st.slider("Top-k", 1, 10, key="top_k", help="10 equals the whole vocabulary; no candidate removed by top-k.")
        st.slider("Top-p", 0.05, 1.0, step=0.05, key="top_p")
        st.number_input("Random seed", min_value=0, max_value=1_000_000, step=1, key="seed")
        st.number_input("Sampling trials", min_value=100, max_value=50_000, step=100, key="trials")
    section = st.selectbox("Explore", ("Probability playground", "Parameter landscapes", "Sequence search", "Stopping conditions",
                                       "Reproducibility", "Sampling convergence", "Prefill and KV cache"), key="section")
    options = {key: st.session_state[key] for key in ("temperature", "top_k", "top_p")}
    logits = PRESETS[st.session_state.preset]
    seed, trials = int(st.session_state.seed), int(st.session_state.trials)
    st.caption("Order: Temperature → Softmax → Top-k → renormalize → Top-p → Sampling. Lower token ID wins ties. Engine conventions may differ.")
    if section == "Probability playground":
        playground(logits, options, seed, trials)
    elif section == "Parameter landscapes":
        metric = st.segmented_control("Surface height", ("entropy", "candidates"), default="entropy", key="height", required=True)
        grid = cached_grid(st.session_state.preset, options["top_k"])
        st.plotly_chart(stepped_landscape([grid], metric), width="stretch", key="surface_chart",
                        alt="Flat tiles show actual computed parameter-grid values, without smoothing candidate transitions")
        st.caption("Each tile extends one computed grid sample to its neighboring midpoints; the true threshold boundary is only resolved at grid precision. This is parameter space, not physical geometry.")
    elif section == "Sequence search":
        width = st.slider("Beam width", 1, 8, 2, key="beam_width")
        report = compute_group("generation", LabConfig(seed=seed, beam_width=width))
        rows = [dict(strategy=name, sequence="".join("AB"[i] for i in path["ids"]),
                     model_log_probability=path["log_probability"], policy_log_probability=path["policy_log_probability"])
                for name, path in report["paths"].items()]
        for name, path in (("Beam", report["beam"]["best"]), ("Exact enumeration", report["exact"][0])):
            rows.append(dict(strategy=name, sequence="".join("AB"[i] for i in path["ids"]),
                             model_log_probability=path["log_probability"], policy_log_probability=None))
        st.dataframe(rows, alt="Sequence and model joint log probability for each decoding or search strategy")
        path = sample_path(seed, temperature=options["temperature"], top_k=min(2, options["top_k"]), top_p=options["top_p"])
        st.write("Current controls on the two-token tree (top-k capped at two):", "".join("AB"[i] for i in path["ids"]))
        st.dataframe([dict(step=i + 1, context="".join("AB"[v] for v in row["context"]) or "<BOS>",
                           probability_A=row["state"]["original"][0], policy_A=row["state"]["final"][0],
                           selected="AB"[row["selected"]], model_log_probability=row["model_log_probability"])
                      for i, row in enumerate(path["steps"])], alt="Conditional model probability, transformed policy and chosen token at each step")
        st.caption("Fixed depth 3; no EOS, no length normalization. Exact enumeration covers all eight paths. Model likelihood and transformed-policy likelihood are distinct; neither measures usefulness.")
    elif section == "Stopping conditions":
        budget = st.slider("Maximum new tokens", 0, 8, 8, key="budget")
        delimiter = st.checkbox("Generate a split <END> delimiter before EOS", value=True, key="delimiter")
        stop = st.text_input("Stop sequence (empty disables it)", value="<END>", key="stop")
        result = trace_generation(lambda context: structured_logits(context, delimiter), STRUCTURED_VOCABULARY, 5,
                                  temperature=0, max_new_tokens=budget, stop_sequences=(stop,) if stop else ())
        st.code(result["result"]["text"], language="json")
        st.metric("Finish reason", result["result"]["finish_reason"])
        try:
            json.loads(result["result"]["text"])
            valid = True
        except json.JSONDecodeError:
            valid = False
        st.metric("Syntactically valid JSON", str(valid))
        st.dataframe([dict(step=i + 1, selected=STRUCTURED_VOCABULARY[row["selected"]], remaining=row["remaining"],
                           text=row["text"], reason=row["reason"]) for i, row in enumerate(result["trace"])],
                     alt="Fragment selection, output text, remaining budget and termination reason")
        st.caption("Matched stops and EOS are excluded from text but their sampled IDs consume budget. Cross-token matching searches generated text only. Output caps, context-window capacity and semantic completeness are separate constraints.")
    elif section == "Reproducibility":
        report = compute_group("reproducibility", LabConfig(seed=seed))
        for name, row in report.items():
            st.subheader(name)
            st.write(f"Unique: {row['unique']}; pairwise exact: {row['exact_agreement']:.3f}; mean pairwise token: {row['token_agreement']:.3f}")
            st.dataframe(np.where(row["matrix"] == -1, "padding", np.where(row["matrix"] == 0, "A", np.where(row["matrix"] == 1, "B", "EOS"))),
                         alt=f"Sixteen generated sequences for {name}")
        st.caption("Agreement includes EOS; unequal lengths count missing tokens as mismatches and double padding is excluded. Shared RNG is advanced across runs; reset mode initializes a new RNG each run. No hosted-inference guarantee.")
    elif section == "Sampling convergence":
        report = compute_group("sampling", LabConfig(seed=seed, samples=trials))
        figure = go.Figure()
        for row in report:
            figure.add_scatter(x=[s["samples"] for s in row["snapshots"]], y=[s["total_variation"] for s in row["snapshots"]], name=row["name"], mode="lines+markers")
        figure.update_layout(xaxis=dict(title="Nested sampling trials", type="log"), yaxis_title="Total variation distance", template="plotly_white")
        st.plotly_chart(figure, key="convergence_chart", width="stretch", alt="Actual finite-sample total variation errors for six decoding configurations")
        st.caption("Nested prefixes of one sampled stream per configuration; shared seed initialization across configurations. Error is not required to decrease at every point.")
    else:
        report = compute_group("cache", LabConfig(seed=seed))
        theory = report["theory"]
        st.dataframe({key: theory[key] for key in ("lengths", "new_positions", "reused_positions", "uncached_projection_positions", "cached_kv_bytes")},
                     alt="Theoretical context lengths, projected and reused K/V positions, and retained bytes")
        st.metric("Single-layer cache equivalence maximum absolute error", f"{report['numerical_check']['max_absolute_error']:.3e}")
        st.caption("Prefill predicts output 1. Feed the previous generated token to predict the next. Cache K/V, not past queries. Counts assume batch 1, 2 layers, 2 KV heads, head dimension 8, float32. These are theoretical counts plus a fixed-input one-layer calculation, not measured latency.")
    st.caption("Author interpretation review required. Lower sampling entropy does not guarantee correct, grounded or schema-valid output.")


if __name__ == "__main__":
    main()
