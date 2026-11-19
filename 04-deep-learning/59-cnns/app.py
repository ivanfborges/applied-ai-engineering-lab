"""CNN Visual Learning Lab — Day 59. Local, synthetic, CPU-only interactions."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments import load_model, run_comparison, selected_example, translation_predictions
from from_scratch import correlate2d, output_size, pool2d
from visual_core import (FILTERS, architecture_table, convolution_steps, edge_image,
                         receptive_mask, rf_table, shift_image)
from visualizations import (architecture_figure, channels_figure, comparison_figure,
                            curves_figure, failure_figure, feature_figure, filters_plotly,
                            learning_figure, pooling_figure, receptive_figure,
                            sliding_figure, stride_figure, translation_figure)


ROOT = Path(__file__).resolve().parent
SECTIONS = (
    "01 · Sliding convolution", "02 · Filters and feature maps", "03 · RGB convolution in 3D",
    "04 · Stride and padding", "05 · Max versus average pooling", "06 · Receptive fields",
    "07 · Translation and predictions", "08 · CNN architecture in 3D",
    "09 · Training and learned filters", "10 · CNN versus MLP", "11 · Failure analysis",
)
EXPLANATIONS = {
    "01": ("A kernel window, its nine products, their sum and the growing output.",
           "Y[i,j] = sum X[i+u,j+v] W[u,v], with no kernel reversal.",
           "Move the window to another output position.", "Positive/negative contributions can reinforce or cancel; zero weights contribute nothing.",
           "Every learned spatial kernel reuses this local operation.", "Handcrafted single-channel filter; no learned bias or activation."),
    "02": ("One input, six handcrafted local operators and signed responses.", "A padded cross-correlation, followed by no nonlinearity.",
           "Choose an operator in the chart's filter menu.", "Edge responses have signs; blur smooths and identity preserves interior values.",
           "Connect kernel coefficients to feature responses before inspecting learned weights.", "All outputs use the same signed scale; handcrafted filters are not learned semantics."),
    "03": ("RGB input planes, one filter's three channel slices and two output maps.",
           "Sum spatial products per input channel, sum channels, then add one output-channel bias.",
           "Choose the output filter and the output pixel; rotate/zoom the scene.",
           "The selected 3x3x3 filter spans R, G and B; it produces one map, not three.",
           "Understand Conv2d weight layout and why output channels are separate learned detectors.",
           "Planes use layout offsets and symbolic channel spacing. Values are colors, not surface heights; weights are illustrative."),
    "04": ("Real input boundary, padded image, all sampled centers and the output map.",
           "O = floor((N + 2P - D(K-1) - 1) / S + 1); D = 1 here.",
           "Change N, K, S, P and the highlighted window.", "Stride changes the sampling grid; padding adds zeros and changes border averages.",
           "Track resolution, boundary effects and lost detail before designing a network.", "Only symmetric zero padding, square normalized box kernels and floor mode; invalid sizes are rejected."),
    "05": ("One 8x8 map compressed into two 4x4 summaries.", "Max or arithmetic mean over 2x2 windows, stride 2.",
           "Move the highlighted pooling window.", "A sparse high response can dominate max pooling while average pooling dilutes it.",
           "Downsampling can save later computation and discard small structures.", "No learned parameters, padding or ceil mode. Neither summary guarantees translation invariance."),
    "06": ("A selected neuron and its exact structural support on the original input.",
           "r[l] = r[l-1] + (K-1) j[l-1]; j[l] = j[l-1] S.",
           "Choose a layer and neuron coordinates, including a border neuron.",
           "Nominal span grows 1,3,4,8,12; real-image support is clipped by padding.",
           "Check whether a local feature has enough input context for a task.",
           "Theoretical support is potential influence. Effective gradient sensitivity is sample/weight dependent and is not measured in this panel."),
    "07": ("Zero-filled shifts, a stride-1 detector, aligned differences and actual classifier probabilities.",
           "Compare C(Tx) with T(Cx) on overlapping interior windows; compare predictions separately.",
           "Shift the input left or right; inspect strided and pooled summaries.",
           "Aligned interior responses agree while classifier confidence may change.",
           "Sampling grids, borders and trained heads affect robustness to object position.",
           "Gray pixels are excluded. Outer shifts may crop the bar. Lines connect measured even displacements; current probabilities are computed separately."),
    "08": ("Symbolic boxes sized by spatial width/height and number of channels.",
           "Two Conv-ReLU-Pool blocks, global spatial mean, then a linear classifier.",
           "Rotate, zoom and hover over layers.", "32→16→8 spatial resolution; 1→8→16 channels; 1,282 trainable parameters.",
           "Separate channel count from image resolution and inspect where context grows.",
           "Boxes encode shapes, not measured activations. x includes layer-layout offsets; global aggregation covers the input union."),
    "09": ("Actual saved kernel weights, training/validation curves and before/after feature maps.",
           "Adam updates from cross-entropy gradients; each displayed checkpoint is an executed epoch.",
           "Select a saved epoch, first-layer channel and test example.",
           "Use the fixed weight scale and shared activation scale rather than independently normalizing frames.",
           "Relate optimization to changes in representations and predictions.",
           "One channel/input does not establish interpretability. Validation curves are descriptive; final epoch is fixed in advance."),
    "10": ("Paired CNN/MLP runs: parameters, learning curves, clean tests and stress sweeps.",
           "Both models optimize mean cross-entropy with Adam 0.01 on identical batches and splits.",
           "Inspect seed-specific results and transformation curves.",
           "Clean accuracy can hide sensitivity to translation or noise.",
           "Compare model behavior using actual measurements under explicit constraints.",
           "Equal epochs/rate are not equal capacity or compute; neither model is tuned. Synthetic results do not rank model families."),
    "11": ("Real mistakes from transformed test images, selected for inspection.",
           "Run final fixed-epoch models on dx=+8 and additional noise std=1, then rank confident errors.",
           "Inspect which model, true label, predicted label and condition produced each case.",
           "A confident softmax output can still be wrong; if no errors exist, uncertain successes are labeled honestly.",
           "Make aggregate scores actionable by examining failure conditions.",
           "This is a deliberately selected diagnostic gallery, not another unbiased evaluation sample or a calibration study."),
}


@st.cache_data(max_entries=3, show_spinner=False)
def read_report(path, modified_ns):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(max_entries=3, show_spinner=False)
def cached_comparison(seed, epochs):
    torch.set_num_threads(1)
    # Keep the progress element inside the cached function so Streamlit can
    # replay it in another session without referencing an external layout.
    progress_bar = st.progress(0., text="Preparing paired runs")
    measured = run_comparison(seeds=(seed, seed + 1), epochs=epochs,
                              progress=lambda value, text: progress_bar.progress(value, text=text))
    progress_bar.progress(1., text="Measurements ready; identical configurations may come from cache")
    return measured


def show(fig, description):
    st.pyplot(fig, width="stretch", alt=description)
    plt.close(fig)


kernel_icon = np.array([[[37, 99, 166], [240, 240, 240], [194, 91, 37]]] * 3, dtype=np.uint8)
st.set_page_config(page_title="CNN Visual Learning Lab — Day 59", page_icon=Image.fromarray(kernel_icon), layout="wide")
torch.set_num_threads(1)
st.title("CNN Visual Learning Lab — Day 59")
st.caption("Pixels → shared filters → feature maps → spatial summaries → predictions. Synthetic data; local CPU execution.")
section = st.sidebar.selectbox("Explore a mechanism", SECTIONS, key="section")
with st.sidebar.expander("Run the matched learning experiment"):
    seed = st.number_input("Base seed", min_value=0, max_value=10000, value=59, step=1, key="training_seed")
    epochs = st.selectbox("Epoch budget", [5, 10, 20], index=2, key="training_epochs")
    st.caption("Two seeds, CNN and MLP; 256/128/128 train/validation/test images. Identical requests reuse cached computation.")
    train_clicked = st.button("Run CPU experiments", key="run_training", type="primary", width="stretch")
if train_clicked:
    measured = cached_comparison(int(seed), epochs)
    st.session_state["measured_report"] = measured

existing = sorted((ROOT / "outputs" / "visual-lab").glob("run-*/experiment.json"))
latest_path = existing[-1] if existing else None
report = st.session_state.get("measured_report")
if report is None and latest_path:
    report = read_report(str(latest_path), latest_path.stat().st_mtime_ns)
if report:
    st.sidebar.caption(f"Displayed run: seeds {report['configuration']['seeds']}; {report['configuration']['epochs']} epochs. Author review required.")
    st.sidebar.download_button("Download measured record", json.dumps(report, indent=2), "cnn_visual_experiment.json", "application/json", key="download_report")
else:
    st.sidebar.info("No training runs automatically. Use the button above or generate the visual collection.")

code = section[:2]
st.header(section)
labels = ("What you see", "Operation", "Change", "Observe", "Practical use", "Limit")
with st.expander("How to read this visualization", expanded=True):
    for label, explanation in zip(labels, EXPLANATIONS[code]):
        st.markdown(f"**{label}:** {explanation}")

if code == "01":
    step = st.slider("Output position (row-major order)", 0, 35, 0, key="convolution_step")
    show(sliding_figure(step), "Sliding kernel, selected patch, products, weighted sum and progressively built output")
elif code == "02":
    st.plotly_chart(filters_plotly(), width="stretch", key="filter_chart", alt="Filter selector compares handcrafted kernels and signed responses on a common scale")
elif code == "03":
    controls = st.columns(3)
    selected_filter = controls[0].selectbox("Output channel", [0, 1], key="rgb_channel")
    row = controls[1].slider("Output row", 0, 5, 2, key="rgb_row")
    col = controls[2].slider("Output column", 0, 5, 2, key="rgb_col")
    st.markdown("Input **3×8×8** · weights **2×3×3×3** · each filter **3×3×3** · output **2×6×6**")
    st.plotly_chart(channels_figure(row, col, selected_filter, show_menu=False), width="stretch", key="rgb_chart",
                    config={"scrollZoom": True, "displaylogo": False}, alt="RGB planes, per-input-channel kernels, contributions and two output feature maps")
elif code == "04":
    controls = st.columns(4)
    size = controls[0].slider("Input dimension N", 4, 24, 8, key="input_size")
    kernel = controls[1].selectbox("Kernel K", [3, 5, 7], key="kernel_size")
    stride = controls[2].selectbox("Stride S", [1, 2, 3], index=1, key="stride")
    padding = controls[3].selectbox("Padding P", [0, 1, 2, 3], index=1, key="padding")
    st.latex(r"O=\left\lfloor\frac{N+2P-D(K-1)-1}{S}+1\right\rfloor,\quad D=1")
    try:
        out = output_size(size, kernel, stride, padding)
    except ValueError as exc:
        st.error(f"Invalid geometry: {exc}. Increase N or padding, or reduce K.")
    else:
        step = st.slider("Highlighted output position", 0, out**2 - 1, 0, key=f"stride_step_{size}_{kernel}_{stride}_{padding}") if out > 1 else 0
        st.metric("Output spatial dimensions", f"{out} × {out}")
        show(stride_figure(size, kernel, stride, padding, step), "Padded input boundary, kernel sampling positions and resulting output dimensions")
elif code == "05":
    step = st.slider("Pooling window (row-major order)", 0, 15, 0, key="pool_step")
    show(pooling_figure(step), "Synchronized max and average pooling over the same feature-map window")
elif code == "06":
    table = rf_table()
    layer = st.selectbox("Layer", list(range(5)), index=4, format_func=lambda index: table[index]["layer"], key="rf_layer")
    side = table[layer]["size"]
    controls = st.columns(2)
    row = controls[0].slider("Neuron row", 0, side - 1, side // 2, key=f"rf_row_{layer}")
    col = controls[1].slider("Neuron column", 0, side - 1, side // 2, key=f"rf_col_{layer}")
    mask = receptive_mask(layer, row, col)
    st.caption(f"Nominal span: {table[layer]['span']}×{table[layer]['span']}; jump: {table[layer]['jump']}; real input pixels: {mask.sum()}.")
    show(receptive_figure(layer, row, col), "Selected neuron and exact potential input support, clipped at every padded layer")
    st.dataframe(table, hide_index=True, alt="Spatial size, theoretical receptive-field span and effective stride for each layer")
elif code == "08":
    st.plotly_chart(architecture_figure(), width="stretch", key="architecture_chart",
                    config={"scrollZoom": True, "displaylogo": False}, alt="CNN tensor boxes with spatial dimensions, channel counts and parameter counts")
    st.dataframe([dict(layer=name, shape=str(shape), parameters=params, receptive_field=span)
                  for name, shape, params, span in architecture_table()], hide_index=True,
                 alt="CNN layer shapes, parameter counts and local receptive fields")
else:
    if report is None:
        st.info("Run CPU experiments in the sidebar, or generate the collection first. These panels use actual measurements.")
    elif code == "07":
        default_sample = translation_predictions(report)["sample"]
        sample = st.slider("Translation test example", 0, report["configuration"]["test_size"] - 1, default_sample, key="translation_sample")
        dx = st.slider("Horizontal displacement (pixels)", -8, 8, 0, key="translation_dx")
        show(translation_figure(report, dx, sample=sample), "Shifted image, aligned stride-one convolution responses and measured classifier probability sweep")
        image, label = selected_example(report, sample=sample)
        moved = torch.from_numpy(shift_image(image.numpy(), dx))
        cols = st.columns(2)
        for kind, col in zip(("cnn", "mlp"), cols):
            model = load_model(kind, report["runs"][0]["models"][kind]["checkpoints"][-1]["state"])
            with torch.no_grad():
                before = torch.softmax(model(image), 1)[0, label].item()
                after = torch.softmax(model(moved), 1)[0, label].item()
            col.metric(f"{kind.upper()} P(true class) at dx={dx}", f"{after:.4f}", delta=f"{after - before:+.4f}")
        original = image[0, 0].numpy()
        changed = moved[0, 0].numpy()
        kernel = FILTERS["Vertical edge"]
        summaries = []
        for name, operation in (("Stride-1 feature mean", lambda x: correlate2d(x, kernel, padding=1).mean()),
                                ("Stride-2 feature mean", lambda x: correlate2d(x, kernel, stride=2, padding=1).mean()),
                                ("Max-pool feature mean", lambda x: pool2d(correlate2d(x, kernel, padding=1)).mean()),
                                ("Average-pool feature mean", lambda x: pool2d(correlate2d(x, kernel, padding=1), mode="average").mean())):
            summaries.append({"aggregation": name, "original": float(operation(original)), "shifted": float(operation(changed))})
        st.dataframe(summaries, hide_index=True, alt="Measured changes in strided and pooled global feature summaries after translation")
        st.caption("These handcrafted signed-feature summaries are a separate mechanism check, not internal classifier logits. Stride 2 has no integer output alignment for odd input shifts.")
    elif code == "09":
        checkpoints = report["runs"][0]["models"]["cnn"]["checkpoints"]
        controls = st.columns(3)
        checkpoint = controls[0].selectbox("Actual checkpoint", list(range(len(checkpoints))), index=len(checkpoints) - 1,
                                          format_func=lambda index: f"epoch {checkpoints[index]['epoch']}", key="checkpoint")
        channel = controls[1].slider("First-layer channel", 0, 7, 0, key="feature_channel")
        sample = controls[2].slider("Test example", 0, report["configuration"]["test_size"] - 1, 0, key="feature_sample")
        show(learning_figure(report, checkpoint), "Eight actual first-layer kernels at a measured checkpoint with one fixed weight scale")
        show(feature_figure(report, channel, sample, checkpoint), "Input, initial and selected-checkpoint feature maps on a shared scale and actual predictions")
        show(curves_figure(report), "Measured full-training loss and validation accuracy over epochs for both models and seeds")
    elif code == "10":
        show(comparison_figure(report), "Paired CNN and MLP architectures, parameter counts, learning curves and held-out stress measurements")
    elif code == "11":
        show(failure_figure(report), "Real transformed-input classification errors, with labels and confidence values")
        st.caption("Class 0: vertical; class 1: horizontal. The synthetic center support and transform protocol are recorded in the downloadable experiment JSON.")

if latest_path and code in ("01", "04", "05", "06", "07", "09"):
    gifs = sorted(latest_path.parent.glob(f"{code}_*.gif"))
    same_run = report is not None and report["created_utc"] == read_report(str(latest_path), latest_path.stat().st_mtime_ns)["created_utc"]
    if gifs and (code in ("01", "04", "05", "06") or same_run):
        if st.checkbox("Play the exported animation", key=f"play_{code}"):
            st.image(gifs[0].read_bytes(), width="stretch", alt=f"Animated demonstration for {section}")

st.caption("Educational implementations and synthetic measurements. Interpretation candidates require author review; software checks do not establish real-image performance.")
