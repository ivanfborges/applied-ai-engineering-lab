"""Local Streamlit learning lab. Training requires an explicit form submission."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from experiments import train_visuals
from visual_math import batchnorm_trace, dropout_moments, stopping_replay
from visual_render import (COLORS, LABELS, bn_running_plot, boundary_plot, decay_contour,
                           decay_surface, dropout_network_plot, learning_plot, run_by_name,
                           state_at, stopping_plot, style)


st.set_page_config(page_title="Regularization · Day 57", layout="wide")
st.title("Regularization, made observable")
st.caption("Day 57 · Local CPU learning lab · Synthetic data · Interpretation review required")

PAGES = ["Generalization and overfitting", "Dropout", "Weight decay and optimization",
         "Batch normalization", "Early stopping", "Integrated comparison"]
page = st.sidebar.selectbox("Explore", PAGES, key="page")
st.sidebar.caption("Mathematical simulations are immediate. Real training runs only when requested.")


@st.cache_data(max_entries=2, show_spinner=False)
def completed_training(epochs, n, seed, patience, min_delta):
    indicator = st.progress(0., text="Preparing synthetic train/validation/test partitions…")
    def update(fraction, name, epoch):
        if epoch % 10 == 0 or fraction == 1:
            indicator.progress(fraction, text=f"Training {LABELS[name]} · epoch {epoch}")
    result = train_visuals(epochs, n, seed, patience=patience, min_delta=min_delta, progress=update)
    indicator.empty()
    return result


with st.sidebar.form("training"):
    st.subheader("Real training budget")
    epochs = st.slider("Maximum epochs", 40, 200, 120, 20, key="train_epochs")
    n = st.slider("Synthetic observations", 100, 1000, 400, 100, key="train_n")
    seed = st.number_input("Training seed", 0, 100_000, 157, 1, key="train_seed")
    patience = st.slider("Training patience", 1, 40, 15, key="train_patience")
    min_delta = st.number_input("Training min_delta", 0., .02, .0001, .0001, format="%.4f", key="train_delta")
    submitted = st.form_submit_button("Run six training configurations", key="train_button", width="stretch")
if submitted:
    st.session_state["study"] = completed_training(epochs, n, seed, patience, min_delta)

study = st.session_state.get("study")
if study is not None:
    config = study["configuration"]
    st.sidebar.success(f"Completed: seed {config['seed']} · n={config['n']} · max epochs {config['epochs_max']}")
    st.sidebar.caption("Changing form controls does not replace this result until Run is pressed. Test metrics remain hidden.")


def chart(figure, key, alt):
    st.plotly_chart(figure, key=key, width="stretch", alt=alt)


def need_training():
    if study is None:
        st.info("Run the six training configurations in the sidebar to inspect real checkpoints. The other simulation pages work immediately.")
        st.stop()


if page == PAGES[0]:
    st.header("When fitting becomes overfitting")
    st.markdown("Follow actual two-dimensional decision boundaries alongside evaluation-mode losses. A flexible boundary can fit training noise; complexity alone does not prove overfitting.")
    need_training()
    candidate = st.selectbox("Compare baseline with", list(LABELS)[1:], format_func=LABELS.get, index=4, key="boundary_candidate")
    epoch = st.slider("Training timeline", 0, config["epochs_max"], 0, key="boundary_epoch")
    retained = st.toggle("Show retained checkpoints", key="retained")
    columns = st.columns(2)
    for col, name in zip(columns, ("baseline", candidate)):
        run = run_by_name(study, name)
        state = run["selected"] if retained else state_at(run, epoch)
        with col:
            chart(boundary_plot(study, run, state), f"boundary_{name}", "Actual synthetic decision boundary with training and validation samples")
            st.metric("Validation − training CE", f"{state['validation']['loss']-state['train']['loss']:+.4f}")
    chart(learning_plot({**study, "runs": [run_by_name(study, name) for name in ("baseline", candidate)]}), "paired_curves", "Observed training and validation loss, without extrapolation")
    st.caption("Circles: training; crosses: validation. No test points shown. Stopped candidates freeze at their restored best checkpoint after termination. Boundaries are genuine 2D model outputs, not projections.")
    st.info("Compare the gap with held-out performance. A smaller training loss is insufficient, and regularization can underfit.")

elif page == PAGES[1]:
    st.header("Dropout changes a forward pass")
    st.latex(r"\tilde h=\frac{mh}{1-p},\quad m\sim\mathrm{Bernoulli}(1-p),\quad \mathrm{Var}(\tilde h\mid h)=\frac{p}{1-p}h^2")
    p = st.slider("Drop probability p", 0., .8, .3, .05, key="drop_p")
    pass_index = st.slider("Forward pass / mask", 0, 30, 0, key="drop_pass")
    chart(dropout_network_plot(p, pass_index), "drop_network", "Two hidden layers with actual masks, activation magnitudes and output probabilities")
    moments = dropout_moments(p)
    first, second, third = st.columns(3)
    first.metric("Retained draws (2,000 × 3)", f"{moments['retained']:.3f}")
    second.metric("Mean for h = 1", f"{moments['mean'][1]:.4f}")
    third.metric("Variance for h = 1", f"{moments['variance'][1]:.4f}")
    fig = go.Figure(go.Histogram(x=moments["samples"][:, 1], name="Training: masked/scaled", opacity=.75, xbins=dict(size=.1)))
    fig.add_vline(x=1, line_color=COLORS[1], annotation_text="Inference = original h")
    fig.update_xaxes(title="Activation after dropout (h=1)")
    fig.update_yaxes(title="Count across 2,000 passes")
    chart(style(fig, "Training distribution vs ordinary inference identity", 380), "drop_hist", "Dropout adds zero mass and rescales retained activations")
    ps = np.linspace(0, .8, 33)
    variance = go.Figure(go.Scatter(x=ps, y=ps/(1-ps), name="Theory, h=1", line=dict(color=COLORS[0])))
    variance.add_trace(go.Scatter(x=ps, y=[dropout_moments(float(v))["variance"][1] for v in ps],
                                 mode="markers", name="Empirical, h=1", marker=dict(color=COLORS[1])))
    variance.update_xaxes(title="Drop probability")
    variance.update_yaxes(title="Conditional variance")
    chart(style(variance, "Activation variance increases with p", 380), "drop_variance", "Dropout activation variance compared with its conditional formula")
    st.dataframe(pd.DataFrame({"Original h / inference": moments["activation"], "Training mean": moments["mean"],
                               "Training variance": moments["variance"], "Theory variance": moments["target_variance"]}),
                 hide_index=True, alt="Empirical dropout moments compared with conditional expectations")
    st.caption("Fixed, untrained network; changing masks does not update weights. A naturally zero ReLU is different from a dropped unit. Expectation preservation applies to an activation, not an entire nonlinear prediction. Monte Carlo error is expected.")

elif page == PAGES[2]:
    st.header("Decay changes the update, not just the surface")
    st.latex(r"L(w)=\tfrac12(w-c)^TA(w-c),\quad J(w)=L(w)+\tfrac\lambda2\|w\|^2")
    st.caption("A=[[3,1],[1,1]], c=[1.6,−1]. L2 minimum solves (A+λI)w=Ac. Fixed start [−2,2].")
    decay = st.slider("Decay / L2 coefficient λ", 0., 1., .3, .05, key="decay")
    steps = st.slider("Optimization updates", 10, 120, 80, 10, key="decay_steps")
    chart(decay_surface(decay, steps), "decay_3d", "Original and regularized surfaces with four optimizer trajectories and analytic minima")
    chart(decay_contour(decay, steps), "decay_2d", "Parameter paths, norms and reference objective values")
    st.latex(r"w^+_{SGD}=(1-\eta\lambda)w-\eta g;\quad w^+_{AdamW}=(1-\eta\lambda)w-\eta\hat m/(\sqrt{\hat v}+\epsilon)")
    st.info("SGD/L2 equals multiplicative decay here. Adam/L2 includes λw in its moments; AdamW excludes it. AdamW paths are lifted onto J only as a reference: this is not gradient descent on J.")
    st.caption("Mathematical simulation; no neural-network loss is being approximated. SGD lr=.12; both adaptive methods lr=.08, betas=(.9,.999), eps=1e-8, no momentum for SGD. Different optimizer families are not rate-tuned or benchmarked.")

elif page == PAGES[3]:
    st.header("BatchNorm transforms a distribution")
    st.latex(r"y=\gamma\frac{x-\mu_B}{\sqrt{v_B+\epsilon}}+\beta")
    controls = st.columns(4)
    gamma = controls[0].slider("Gamma γ", -3., 3., 1., .1, key="bn_gamma")
    beta = controls[1].slider("Beta β", -5., 5., 0., .1, key="bn_beta")
    exponent = controls[2].slider("log10 epsilon", -8, 0, -5, key="bn_eps")
    batch_size = controls[3].select_slider("Batch size", [2, 4, 8, 16, 32, 64, 128], value=16, key="bn_size")
    batch = st.slider("Observed mini-batch", 1, 40, 40, key="bn_batch") - 1
    feature = st.selectbox("Synthetic feature", [0, 1, 2], index=2, format_func=lambda j: f"Feature {j+1}", key="bn_feature")
    trace = batchnorm_trace(batch_size, gamma, beta, 10.**exponent)
    fig = go.Figure()
    for key, name, color in (("raw", "Raw", COLORS[0]), ("normalized", "Before affine", COLORS[1]), ("affine", "After gamma/beta", COLORS[2])):
        fig.add_trace(go.Histogram(x=trace[key][batch, :, feature], name=name, opacity=.55, marker_color=color))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Activation")
    fig.update_yaxes(title="Count")
    chart(style(fig, "Same finite batch: raw → normalized → scale and shift", 400), "bn_hist", "BatchNorm normalization followed by gamma scaling and beta shifting")
    st.dataframe(pd.DataFrame([{"Stage": key, "Mean": trace[key][batch, :, feature].mean(),
                                "Biased variance": trace[key][batch, :, feature].var()} for key in ("raw", "normalized", "affine", "inference")]),
                 hide_index=True, alt="Numeric activation means and variances at each normalization stage")
    chart(bn_running_plot(trace, feature), "bn_running", "Current mini-batch statistics versus exponential running estimates and population truth")
    modes = go.Figure()
    for key, color in (("affine", COLORS[0]), ("inference", COLORS[1])):
        modes.add_trace(go.Scatter(y=trace[key][batch, :, feature], mode="lines+markers", name="Training" if key == "affine" else "Inference", line=dict(color=color)))
    modes.update_xaxes(title="Sample within same batch")
    modes.update_yaxes(title="Activation after affine")
    chart(style(modes, "Same input, training statistics vs running statistics", 350), "bn_modes", "BatchNorm train and inference outputs can differ on the same observations")
    st.caption("Stationary Gaussian feature simulation, not model training. Running mean starts at 0, variance at 1, momentum .1. Training uses biased variance; running variance uses m/(m−1) correction. Inference shown after this batch's update. Smaller batches are noisier; this does not establish a generalization benefit.")

elif page == PAGES[4]:
    st.header("Validation chooses the checkpoint")
    st.markdown("Replay a stopping rule on real recorded baseline training. Changing the rule selects an observed prefix; it does not generate a new training trajectory or query the test set.")
    need_training()
    baseline = run_by_name(study, "baseline")
    maximum = st.slider("Replay maximum epochs", 1, len(baseline["history"]), len(baseline["history"]), key="es_max")
    replay_patience = st.slider("Replay patience", 1, 40, 15, key="es_patience")
    delta = st.number_input("Replay min_delta", 0., .1, .0001, .0001, format="%.4f", key="es_delta")
    replay = stopping_replay(baseline["history"], replay_patience, delta, maximum)
    current = st.slider("Observed replay epoch", 1, replay["stop_epoch"], replay["stop_epoch"], key="es_current") if replay["stop_epoch"] > 1 else 1
    chart(stopping_plot(baseline, replay, current), "es_curves", "Validation loss selects the best observed checkpoint while patience counts significant progress")
    best, last = (baseline["snapshots"][replay[key]] for key in ("best_epoch", "stop_epoch"))
    left, middle, right = st.columns(3)
    left.metric("Retained checkpoint", str(best["epoch"]))
    middle.metric("Last observed checkpoint", str(last["epoch"]))
    right.metric("Linear parameter distance", f"{np.linalg.norm(best['weights']-last['weights']):.4f}")
    st.dataframe(pd.DataFrame([{"Checkpoint": name, "Epoch": state["epoch"], "Validation CE": state["validation"]["loss"],
                                "Training CE": state["train"]["loss"]} for name, state in (("Best", best), ("Last at stop", last))]),
                 hide_index=True, alt="Best versus last-at-stop checkpoint validation losses")
    for column, name, state in zip(st.columns(2), ("best", "last"), (best, last)):
        with column:
            chart(boundary_plot(study, baseline, state), f"es_{name}", "Recorded checkpoint boundary for comparison")
    st.caption("Checkpoint panels compare the complete replay rule, independently of the timeline slider above. Strict minima update the copied best state; min_delta resets patience separately. Real early-stopping candidates in the integrated run actually terminate and restore all parameters and BatchNorm buffers. No deployment is implied.")

else:
    st.header("Six mechanisms, one evaluation protocol")
    need_training()
    chart(learning_plot(study), "all_losses", "All executed training and validation loss histories")
    chart(learning_plot(study, "accuracy"), "all_accuracy", "All executed training and validation accuracy histories")
    gap = go.Figure()
    rows = []
    for color, run in zip(COLORS, study["runs"]):
        history, selected = run["history"], run["selected"]
        gap.add_trace(go.Scatter(x=[r["epoch"] for r in history],
                                y=[r["validation"]["loss"]-r["train"]["loss"] for r in history],
                                name=LABELS[run["name"]], line=dict(color=color)))
        rows.append({"Configuration": LABELS[run["name"]], "Train CE": selected["train"]["loss"],
                     "Validation CE": selected["validation"]["loss"], "Train accuracy": selected["train"]["accuracy"],
                     "Validation accuracy": selected["validation"]["accuracy"], "Selected epoch": selected["epoch"],
                     "Epochs run": run["epochs_run"], "Updates": run["steps"], "Linear norm (incl. biases)": selected["weight_norm"],
                     "Seconds (incl. snapshots)": run["duration_seconds"], "Checkpoint policy": run["checkpoint_policy"]})
    gap.update_xaxes(title="Executed epoch")
    gap.update_yaxes(title="Validation CE − train CE")
    chart(style(gap, "Generalization gaps", 400), "all_gaps", "Validation minus training loss across the six configurations")
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, alt="Measured configuration metrics, selected checkpoints, effort and parameter norms")
    bars = go.Figure(go.Bar(x=table["Configuration"], y=table["Validation CE"], marker_color=COLORS))
    bars.update_yaxes(title="Retained validation CE", rangemode="tozero")
    chart(style(bars, "Retained validation checkpoints", 380), "all_retained", "Validation loss at the retained checkpoint, lower is better")
    st.metric("Selected using validation only", LABELS[study["selected_configuration"]])
    selected_name = st.selectbox("Inspect retained boundary", list(LABELS), format_func=LABELS.get, key="integrated_boundary")
    run = run_by_name(study, selected_name)
    chart(boundary_plot(study, run, run["selected"]), "integrated_map", "Retained model boundary with train and validation observations")
    st.caption("One seed per dashboard run. Bias/BN parameters are excluded from decay; reported linear norm includes biases. ES candidates retain their best observed validation state; others retain the final epoch. Durations include snapshots and cannot benchmark methods. Test set remains unevaluated in the dashboard. CLI exports support repeated seeds.")
