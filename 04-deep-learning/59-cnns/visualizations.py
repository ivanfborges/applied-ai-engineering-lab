"""Scientific figures and bounded animations; shared with the interactive lab."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from experiments import bars, load_model, run_comparison, selected_example, translation_predictions
from from_scratch import correlate2d, output_size, pool2d
from visual_core import (FILTERS, RF_LAYERS, architecture_table, convolution_steps,
                         edge_image, equivariance, receptive_mask, rf_table, rgb_demo,
                         scene, shift_image)


BLUE, ORANGE, INK = "#2563a6", "#c25b25", "#172b46"
EXPECTED = (
    "01_convolution_sliding.gif", "02_filters_comparison.png", "02_filters_interactive.html",
    "03_rgb_convolution_3d.html", "04_stride_padding.gif", "05_pooling_comparison.gif",
    "06_receptive_field_growth.gif", "07_translation_equivariance.gif",
    "08_cnn_architecture_3d.html", "09_filter_learning.gif", "09_training_curves.png",
    "10_cnn_vs_mlp.png", "11_feature_maps_and_failures.png",
)


def figure(rows, cols, title, *, size=None):
    fig, axes = plt.subplots(rows, cols, figsize=size or (4.1 * cols, 3.5 * rows), squeeze=False)
    fig.set_facecolor("#f8fafc")
    fig.suptitle(title, fontsize=16, fontweight="bold", color=INK, y=0.98)
    fig.subplots_adjust(top=0.86, bottom=0.13, left=0.07, right=0.94, wspace=0.42, hspace=0.6)
    return fig, axes


def footer(fig, text):
    fig.text(0.035, 0.025, text, fontsize=9, color=INK, va="bottom")


def heat(ax, values, title, *, limits=None, signed=False, numbers=False, colorbar=False):
    values = np.asarray(values)
    if limits is None:
        maximum = max(float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1., 1e-8)
        limits = (-maximum, maximum) if signed else (0., maximum)
    cmap = plt.get_cmap("RdBu_r" if signed else "viridis").with_extremes(bad="#e2e8f0")
    artist = ax.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="nearest")
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    ax.set_xlabel("column j", fontsize=8)
    ax.set_ylabel("row i", fontsize=8)
    for dimension, setter in ((values.shape[1], ax.set_xticks), (values.shape[0], ax.set_yticks)):
        setter(range(dimension) if dimension <= 8 else [0, dimension // 2, dimension - 1])
    ax.tick_params(labelsize=8)
    if numbers:
        for (row, col), value in np.ndenumerate(values):
            if np.isfinite(value):
                color = "white" if abs(value - sum(limits) / 2) > (limits[1] - limits[0]) * .23 else INK
                rounded = 0. if abs(value) < .005 else value
                label = f"{rounded:.2f}".rstrip("0").rstrip(".")
                ax.text(col, row, label, ha="center", va="center", fontsize=11, color=color)
    if colorbar:
        fig = ax.figure
        fig.colorbar(artist, ax=ax, fraction=.045, pad=.04).ax.tick_params(labelsize=8)
    return artist


def box(ax, row, col, height, width, color="#e8ae26"):
    ax.add_patch(Rectangle((col - .5, row - .5), width, height, fill=False, edgecolor=color, lw=2.6))


def sliding_figure(index=0):
    image, kernel = edge_image(), FILTERS["Vertical edge"]
    _, output, steps = convolution_steps(image, kernel)
    index = int(np.clip(index, 0, len(steps) - 1))
    current = steps[index]
    progressive = np.full_like(output, np.nan)
    progressive.flat[:index + 1] = output.flat[:index + 1]
    fig, axes = figure(2, 3, f"01 · A shared 3x3 kernel, one weighted sum at a time   [{index + 1}/{len(steps)}]", size=(12, 7.8))
    heat(axes[0, 0], image, "Input image · synthetic vertical edge", limits=(0, 1), numbers=True)
    box(axes[0, 0], current["top"], current["left"], 3, 3)
    heat(axes[0, 1], kernel, "Kernel W · negative / zero / positive", limits=(-1, 1), signed=True, numbers=True)
    heat(axes[0, 2], current["patch"], "Selected 3x3 patch", limits=(0, 1), numbers=True)
    heat(axes[1, 0], current["products"], "Element-wise product: patch x W", limits=(-1, 1), signed=True, numbers=True)
    axes[1, 1].axis("off")
    rows = current["products"].sum(1)
    axes[1, 1].text(.5, .8, "Sum the nine products", ha="center", fontsize=13, color=INK)
    axes[1, 1].text(.5, .54, " + ".join(f"({value:.0f})" for value in rows), ha="center", fontsize=19)
    axes[1, 1].text(.5, .2, f"Y[{current['row']},{current['col']}] = {current['sum']:.0f}", ha="center", fontsize=23, color=ORANGE)
    heat(axes[1, 2], progressive, "Output feature map · gray = not computed", limits=(-3, 3), signed=True, numbers=True)
    box(axes[1, 2], current["row"], current["col"], 1, 1)
    footer(fig, "Y[i,j] = sum Input[i+u,j+v] x W[u,v]  |  Cross-correlation: no kernel flip. Stride 1, no padding, no bias.\nHandcrafted edge detector; CNN coefficients are learned. Fixed color scales across frames.")
    return fig


def filters_figure(selected=None):
    choices = list(FILTERS) if selected is None else [selected]
    image = scene()
    maximum = max(np.abs(correlate2d(image, kernel, padding=1)).max() for kernel in FILTERS.values())
    fig, axes = figure(len(choices), 3, "02 · Same image, different local questions", size=(10.8, 2.7 * len(choices) + 1.3))
    for row, name in enumerate(choices):
        kernel = FILTERS[name]
        heat(axes[row, 0], image, "Synthetic lines, corner and disk", limits=(0, 1))
        heat(axes[row, 1], kernel, name, limits=(-5, 5), signed=True, numbers=True)
        heat(axes[row, 2], correlate2d(image, kernel, padding=1), "Response (signed; no ReLU)",
             limits=(-maximum, maximum), signed=True, colorbar=True)
    footer(fig, "Handcrafted filters for intuition. Zero padding 1; stride 1. All responses share one signed scale.\nA large signed response indicates contrast, not a probability or an object label.")
    return fig


def filters_plotly():
    image = scene()
    maximum = max(np.abs(correlate2d(image, k, padding=1)).max() for k in FILTERS.values())
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Input", "3x3 kernel", "Signed feature map"))
    for index, (name, kernel) in enumerate(FILTERS.items()):
        for col, values, scale, limits in (
            (1, image, "Viridis", (0, 1)), (2, kernel, "RdBu_r", (-5, 5)),
            (3, correlate2d(image, kernel, padding=1), "RdBu_r", (-maximum, maximum))):
            fig.add_trace(go.Heatmap(z=values, colorscale=scale, zmin=limits[0], zmax=limits[1],
                                    showscale=col == 3, visible=index == 0,
                                    texttemplate="%{z:.2g}" if col == 2 else None,
                                    hovertemplate="row=%{y}, col=%{x}<br>value=%{z:.4f}<extra></extra>"), row=1, col=col)
    buttons = [dict(label=name, method="update", args=[{"visible": [i // 3 == index for i in range(18)]}])
               for index, name in enumerate(FILTERS)]
    fig.update_layout(title="Handcrafted filters · same input and fixed response scale", height=490,
                      margin=dict(t=125, b=65), updatemenus=[dict(buttons=buttons, x=0, y=1.3, xanchor="left")])
    fig.update_yaxes(autorange="reversed", scaleanchor=None)
    for col in range(1, 4):
        fig.update_xaxes(title_text="column", row=1, col=col)
        fig.update_yaxes(title_text="row", row=1, col=col)
    return fig


def stride_figure(size=8, kernel=3, stride=2, padding=1, step=0):
    image = edge_image(size)
    weights = np.ones((kernel, kernel)) / kernel**2
    padded, output, steps = convolution_steps(image, weights, stride=stride, padding=padding)
    current = steps[int(np.clip(step, 0, len(steps) - 1))]
    fig, axes = figure(1, 3, f"04 · N={size}, K={kernel}, S={stride}, P={padding}, D=1  ->  O={output.shape[0]}", size=(12, 4.8))
    heat(axes[0, 0], image, f"Input {size} x {size}", limits=(0, 1))
    heat(axes[0, 1], padded, f"Zero-padded input {padded.shape}", limits=(0, 1))
    box(axes[0, 1], padding, padding, size, size, BLUE)
    for item in steps:
        axes[0, 1].plot(item["left"] + (kernel - 1) / 2, item["top"] + (kernel - 1) / 2,
                        ".", color="#e8ae26", markersize=3)
    box(axes[0, 1], current["top"], current["left"], kernel, kernel)
    heat(axes[0, 2], output, f"Output {output.shape} · normalized box filter", limits=(0, 1), colorbar=True)
    box(axes[0, 2], current["row"], current["col"], 1, 1)
    footer(fig, f"O = floor((N + 2P - D(K - 1) - 1) / S + 1) = {output.shape[0]}  |  Yellow dots = sampled window centers.\nBlue rectangle = real input boundary. Fixed normalized kernel; boundary averages include padded zeros.")
    return fig


def pooling_input():
    image = np.arange(64, dtype=float).reshape(8, 8) / 8 - 2
    image[3, 4] = 9
    return image


def pooling_figure(step=0):
    image = pooling_input()
    maximum, average = pool2d(image), pool2d(image, mode="average")
    step = int(np.clip(step, 0, 15))
    row, col = divmod(step, 4)
    patch = image[2 * row:2 * row + 2, 2 * col:2 * col + 2]
    fig, axes = figure(1, 3, f"05 · Two summaries of the same 2x2 window   [{step + 1}/16]", size=(12, 4.8))
    heat(axes[0, 0], image, "Input feature map · 8 x 8", limits=(-2, 9), signed=True, numbers=True)
    box(axes[0, 0], row * 2, col * 2, 2, 2)
    for ax, result, title in ((axes[0, 1], maximum, "Max pooling · 4 x 4"),
                              (axes[0, 2], average, "Average pooling · 4 x 4")):
        progressive = np.full_like(result, np.nan)
        progressive.flat[:step + 1] = result.flat[:step + 1]
        heat(ax, progressive, title, limits=(-2, 9), signed=True, numbers=True)
        box(ax, row, col, 1, 1)
    footer(fig, f"Window = {np.array2string(patch.flatten(), precision=2)}   max = {patch.max():.3f}   mean = {patch.mean():.3f}\nStride 2, no padding. Same color scale. A sparse peak survives max pooling but is diluted by the average; spatial detail is lost.")
    return fig


def receptive_figure(layer=4, row=None, col=None):
    table = rf_table()
    size = table[layer]["size"]
    row, col = size // 2 if row is None else row, size // 2 if col is None else col
    mask = receptive_mask(layer, row, col)
    fig, axes = figure(1, 3, "06 · Which original pixels can influence this neuron?", size=(12, 5.4))
    image = scene(20)
    heat(axes[0, 0], image, "Input · yellow = structural support", limits=(0, 1))
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = [1., .7, .05, .55]
    axes[0, 0].imshow(overlay)
    layer_map = np.zeros((size, size))
    layer_map[row, col] = 1
    heat(axes[0, 1], layer_map, f"{table[layer]['layer']} · neuron ({row},{col})", limits=(0, 1))
    box(axes[0, 1], row, col, 1, 1)
    axes[0, 2].axis("off")
    lines = [f"{'layer':18}  r    j    grid"]
    for index, item in enumerate(table):
        prefix = ">" if index == layer else " "
        lines.append(f"{prefix}{item['layer']:17} {item['span']:2}   {item['jump']:2}   {item['size']:2}x{item['size']}")
    axes[0, 2].text(0, .95, "\n\n".join(lines), family="monospace", fontsize=9, va="top", color=INK)
    footer(fig, f"r[l] = r[l-1] + (K-1) j[l-1];  j[l] = j[l-1] S   |   Nominal span {table[layer]['span']}x{table[layer]['span']}; {mask.sum()} real pixels.\nPadding is clipped at every layer. Potential influence is not equal influence: effective sensitivity depends on data, weights and nonlinearities.")
    return fig


def translation_figure(report, dx=0, *, sample=None, predictions=None):
    predictions = predictions or translation_predictions(report, sample)
    images, label = selected_example(report, sample=predictions["sample"])
    values = equivariance(images[0, 0].numpy(), FILTERS["Vertical edge"], dx)
    fig, axes = figure(2, 3, f"07 · Shift the input; align the feature maps   dx={dx:+d}", size=(12, 8))
    limits = (-1, 2)
    heat(axes[0, 0], values["input"], "Original test bar (with synthetic noise)", limits=limits, signed=True)
    heat(axes[0, 1], values["shifted"], "Shifted input · zero fill, no wrap", limits=limits, signed=True)
    feature_limit = max(np.abs(values["original_map"]).max(), np.abs(values["shifted_map"]).max())
    heat(axes[0, 2], values["original_map"], "C(x) · handcrafted vertical detector", limits=(-feature_limit, feature_limit), signed=True)
    heat(axes[1, 0], values["shifted_map"], "C(Tx) · compare to shifted C(x)", limits=(-feature_limit, feature_limit), signed=True)
    heat(axes[1, 1], values["difference"], "C(Tx) - T(Cx) · valid overlap only", limits=(-.01, .01), signed=True, colorbar=True)
    for kind, color in (("cnn", BLUE), ("mlp", ORANGE)):
        sweep = predictions["results"][kind]
        axes[1, 2].plot([x["dx"] for x in sweep], [x["true_class_probability"] for x in sweep],
                        "o-", color=color, label=kind.upper())
    axes[1, 2].axvline(dx, color=INK, linestyle="--", alpha=.6)
    axes[1, 2].set(xlabel="horizontal displacement (pixels)", ylabel=f"P(true class {label})",
                   title=f"Actual classifiers · first seed, sample {predictions['sample']}", ylim=(-.02, 1.02))
    axes[1, 2].legend(fontsize=9)
    footer(fig, f"Stride-1 operator alignment error on valid interior: {values['max_error']:.2e}. Gray = excluded borders / missing overlap.\nClassifier curves include learned layers, pooling and global/dense heads. Separate models and operation: equivariance is not classifier invariance.")
    return fig


def pixel_plane(values, x_offset=0, y_offset=0):
    """Duplicate cell edges so Surface colors remain discrete, not interpolated."""
    height, width = values.shape
    rows = np.repeat(np.arange(height), 2)
    cols = np.repeat(np.arange(width), 2)
    x_edges = cols + np.tile([-.5, .5], width) + x_offset
    y_edges = rows + np.tile([-.5, .5], height) + y_offset
    x, y = np.meshgrid(x_edges, y_edges)
    logical_col, logical_row = np.meshgrid(cols, rows)
    return x, y, np.repeat(np.repeat(values, 2, axis=0), 2, axis=1), np.stack([logical_row, logical_col], -1)


def channels_figure(row=2, col=2, output_channel=0, *, show_menu=True):
    fig = go.Figure()
    annotations = []
    for selected in range(2):
        data = rgb_demo(row, col, selected)
        first_trace = len(fig.data)
        for channel, name in enumerate(("R", "G", "B")):
            x, y, colors, logical = pixel_plane(data["input"][channel])
            fig.add_trace(go.Surface(x=x, y=y, z=np.full(x.shape, channel * 2), surfacecolor=colors, customdata=logical,
                                    colorscale={"R": "Reds", "G": "Greens", "B": "Blues"}[name], cmin=0, cmax=1,
                                    showscale=False, name=f"input c={channel} ({name})",
                                    hovertemplate=f"input c={channel} ({name})<br>row=%{{customdata[0]}}, col=%{{customdata[1]}}<br>value=%{{surfacecolor:.3f}}<extra></extra>"))
            fig.add_trace(go.Scatter3d(x=[col - .5, col + 2.5, col + 2.5, col - .5, col - .5],
                                      y=[row - .5, row - .5, row + 2.5, row + 2.5, row - .5], z=[channel * 2 + .05] * 5,
                                      mode="lines", line=dict(color="#ebac26", width=6), showlegend=False, hoverinfo="skip"))
            kx, ky, colors, logical = pixel_plane(data["weights"][selected, channel], 13, row)
            fig.add_trace(go.Surface(x=kx, y=ky, z=np.full(kx.shape, channel * 2),
                                    surfacecolor=colors, colorscale="RdBu_r", cmin=-1, cmax=1,
                                    showscale=False, name=f"W[{selected},{channel},u,v]",
                                    customdata=logical,
                                    hovertemplate=f"W[out={selected}, in={channel}]<br>u=%{{customdata[0]}}, v=%{{customdata[1]}}<br>weight=%{{surfacecolor:.3f}}<extra></extra>"))
        for out in range(2):
            ox, oy, colors, logical = pixel_plane(data["output"][out], 22)
            fig.add_trace(go.Surface(x=ox, y=oy, z=np.full(ox.shape, out * 2), surfacecolor=colors,
                                    colorscale="RdBu_r", cmin=-2, cmax=2, showscale=out == selected,
                                    opacity=1 if out == selected else .35, colorbar=dict(title="response", len=.55),
                                    customdata=logical, name=f"Y[out={out}]",
                                    hovertemplate=f"Y[out={out}]<br>i=%{{customdata[0]}}, j=%{{customdata[1]}}<br>value=%{{surfacecolor:.3f}}<extra></extra>"))
        for trace in fig.data[first_trace:]:
            trace.visible = selected == output_channel
        contributions = data["contributions"]
        total = contributions.sum() + data["bias"][selected]
        annotations.append(dict(x=.5, y=-.06, xref="paper", yref="paper", showarrow=False,
                                text=f"Y[{selected},{row},{col}] = R {contributions[0]:+.3f} + G {contributions[1]:+.3f} + B {contributions[2]:+.3f} + bias {data['bias'][selected]:+.3f} = {total:.3f}"))
    count = len(fig.data) // 2
    buttons = [dict(label=f"Output filter {out}", method="update",
                    args=[{"visible": [i // count == out for i in range(len(fig.data))]}, {"annotations": [annotations[out]]}]) for out in range(2)]
    fig.update_layout(title="03 · RGB planes → W[out,in,u,v] → two output maps", height=630,
                      margin=dict(t=110, b=85, l=0, r=30), annotations=[annotations[output_channel]],
                      updatemenus=[dict(buttons=buttons, active=output_channel, x=0, y=1.12)] if show_menu else [],
                      scene=dict(xaxis_title="input → kernel → output (layout offsets)", yaxis_title="row coordinate",
                                 zaxis_title="channel plane (not response height)", yaxis=dict(autorange="reversed"),
                                 camera=dict(eye=dict(x=1.6, y=-2.2, z=1.6)),
                                 aspectmode="manual", aspectratio=dict(x=2.5, y=1, z=.8)))
    return fig


def architecture_figure():
    fig = go.Figure()
    offset = 0
    palette = ["#8798aa", BLUE, BLUE, "#218b83", BLUE, BLUE, "#218b83", ORANGE, "#805bb6"]
    for (name, (channels, height, width), parameters, span), color in zip(architecture_table(), palette):
        # Boxes use actual W,H,C dimensions; x has an added layer-layout offset.
        x = np.array([0, width, width, 0, 0, width, width, 0]) + offset
        y = [0, 0, height, height, 0, 0, height, height]
        z = [0, 0, 0, 0, channels, channels, channels, channels]
        text = f"{name}<br>C x H x W = {channels} x {height} x {width}<br>parameters = {parameters}<br>RF span = {span}"
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z,
                               i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
                               j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
                               k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
                               color=color, opacity=.7, name=name, hovertext=text, hoverinfo="text", showlegend=True))
        for channel in sorted(set([0, channels // 2, channels])):
            fig.add_trace(go.Scatter3d(x=[offset, offset + width, offset + width, offset, offset],
                                      y=[0, 0, height, height, 0], z=[channel] * 5,
                                      mode="lines", line=dict(color=color, width=2), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=[offset + width / 2], y=[-8], z=[0], mode="text",
                                  text=[f"{name}<br>{channels}×{height}×{width}<br>{parameters} params"],
                                  textfont=dict(size=10), showlegend=False, hoverinfo="skip"))
        offset += width + 12
    fig.update_layout(title="08 · Spatial resolution shrinks; channels increase · 1,282 parameters", height=700,
                      margin=dict(t=65, b=40, l=0, r=0), legend=dict(orientation="h", y=-.05),
                      scene=dict(xaxis_title="layer offset + spatial width W", yaxis_title="spatial height H",
                                 zaxis_title="channels C", aspectmode="manual", aspectratio=dict(x=3.5, y=1, z=.8),
                                 camera=dict(eye=dict(x=1.2, y=-2.3, z=1.5))))
    return fig


def learning_figure(report, checkpoint=0):
    model_data = report["runs"][0]["models"]["cnn"]
    saved = model_data["checkpoints"]
    current = saved[checkpoint]
    maximum = max(np.abs(np.asarray(item["state"]["conv1.weight"])).max() for item in saved)
    fig = plt.figure(figsize=(11, 7.7), facecolor="#f8fafc")
    grid = fig.add_gridspec(3, 4, top=.87, bottom=.15, left=.06, right=.95, hspace=.65)
    fig.suptitle(f"09 · Actual learned kernels · epoch {current['epoch']} · seed {report['runs'][0]['seed']}",
                 fontsize=16, fontweight="bold", color=INK)
    weights = np.asarray(current["state"]["conv1.weight"])[:, 0]
    for channel in range(8):
        ax = fig.add_subplot(grid[channel // 4, channel % 4])
        heat(ax, weights[channel], f"W[out={channel}, in=0]", limits=(-maximum, maximum), signed=True, numbers=True)
    ax = fig.add_subplot(grid[2, :])
    history = model_data["history"]
    ax.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], color=BLUE, label="train CE")
    ax.plot([row["epoch"] for row in history], [row["val_loss"] for row in history], color=ORANGE, label="validation CE")
    ax.axvline(current["epoch"], linestyle="--", color=INK)
    ax.set(xlabel="actual completed epoch", ylabel="cross-entropy")
    ax.legend(fontsize=9)
    footer(fig, "One fixed signed weight scale across all channels and saved epochs. No interpolated weights or smoothed observations.\nFilters optimize the task loss; appearances do not establish their semantic meaning. Biases are learned but not drawn as kernels.")
    return fig


def curves_figure(report):
    fig, axes = figure(1, 2, "09 · Training changes measured loss and validation predictions", size=(11, 4.8))
    for index, run in enumerate(report["runs"]):
        for kind, color in (("cnn", BLUE), ("mlp", ORANGE)):
            history = run["models"][kind]["history"]
            style = "-" if index == 0 else "--"
            label = f"{kind.upper()} seed {run['seed']}"
            axes[0, 0].plot([x["epoch"] for x in history], [x["train_loss"] for x in history], style, color=color, label=label)
            axes[0, 1].plot([x["epoch"] for x in history], [x["val_accuracy"] for x in history], style, color=color, label=label)
    axes[0, 0].set(title="Full-training loss after updates", xlabel="epoch", ylabel="cross-entropy")
    axes[0, 1].set(title="Validation accuracy (not test)", xlabel="epoch", ylabel="accuracy", ylim=(-.02, 1.02))
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    footer(fig, "Paired synthetic splits and minibatch order; Adam 0.01; fixed final epoch, no best-seed selection.\nEach curve is an executed model. Different parameter counts and untuned hyperparameters prevent a universal ranking.")
    return fig


def comparison_figure(report):
    fig, axes = figure(3, 2, "10 · Spatial tensor CNN versus flattened-image MLP · measured comparison", size=(12, 11))
    axes[0, 0].axis("off")
    axes[0, 0].text(0, .92, "CNN: 1×32×32 → Conv8 → ReLU → Pool2\n"
                          "          → Conv16 → ReLU → Pool2 → mean → logits\n\n"
                          "MLP: 1×32×32 → flatten 1024\n"
                          "          → Linear32 → ReLU → logits\n\n"
                          "Same images, labels, batches, rate and epochs.\n"
                          "Parameters / arithmetic are not matched.", va="top", fontsize=11, color=INK)
    axes[0, 1].axis("off")
    rows = []
    for run in report["runs"]:
        for kind in ("cnn", "mlp"):
            data = run["models"][kind]
            rows.append([kind.upper(), run["seed"], f"{data['parameters']:,}", f"{data['test']['accuracy']:.3f}", f"{data['test']['loss']:.4f}"])
    table = axes[0, 1].table(cellText=rows, colLabels=["Model", "Seed", "Params", "Test acc", "Test CE"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.08, 1.8)
    axes[0, 1].set_title("Final fixed-epoch clean holdout", fontsize=12)
    for run_index, run in enumerate(report["runs"]):
        for kind, color in (("cnn", BLUE), ("mlp", ORANGE)):
            data = run["models"][kind]
            style = "-" if run_index == 0 else "--"
            label = f"{kind.upper()} seed {run['seed']}"
            axes[1, 0].plot([r["epoch"] for r in data["history"]], [r["train_loss"] for r in data["history"]], style, color=color, label=label)
            axes[1, 1].plot([r["epoch"] for r in data["history"]], [r["val_accuracy"] for r in data["history"]], style, color=color)
            axes[2, 0].plot([r["dx"] for r in data["shift_sweep"]], [r["accuracy"] for r in data["shift_sweep"]], style + "o", color=color)
            axes[2, 1].plot([r["additional_noise_std"] for r in data["noise_sweep"]], [r["accuracy"] for r in data["noise_sweep"]], style + "o", color=color)
    axes[1, 0].set(title="Training loss", xlabel="epoch", ylabel="cross-entropy")
    axes[1, 1].set(title="Validation accuracy", xlabel="epoch", ylabel="accuracy", ylim=(-.02, 1.02))
    axes[2, 0].set(title="Horizontal translation · test set", xlabel="zero-filled displacement (pixels)", ylabel="accuracy", ylim=(-.02, 1.02))
    axes[2, 1].set(title="Extra Gaussian noise · test set", xlabel="additional noise standard deviation", ylabel="accuracy", ylim=(-.02, 1.02))
    axes[1, 0].legend(fontsize=8)
    for ax in axes[1:].flat:
        ax.grid(alpha=.2)
    footer(fig, "Training centers 14..17; shifts are a new condition and outer shifts may crop signal. Noise perturbations are paired between models.\nSynthetic data only. Seed ranges are not confidence intervals. These observations require author review.")
    return fig


def feature_figure(report, channel=0, sample=0, checkpoint=-1):
    images, label = selected_example(report, sample=sample)
    checkpoints = report["runs"][0]["models"]["cnn"]["checkpoints"]
    initial = load_model("cnn", checkpoints[0]["state"])
    selected = load_model("cnn", checkpoints[checkpoint]["state"])
    with torch.no_grad():
        before = initial.features(images)[0][0, channel].numpy()
        after = selected.features(images)[0][0, channel].numpy()
        probabilities = torch.softmax(selected(images), 1)[0].numpy()
    maximum = max(before.max(), after.max(), 1e-8)
    fig, axes = figure(1, 4, f"Learned feature inspection · sample {sample}, channel {channel} · true class {label}", size=(13, 4.5))
    heat(axes[0, 0], images[0, 0].numpy(), "Input bar + noise", limits=(-.5, 1.5), signed=True)
    heat(axes[0, 1], before, "First-layer ReLU · epoch 0", limits=(0, maximum))
    heat(axes[0, 2], after, f"First-layer ReLU · epoch {checkpoints[checkpoint]['epoch']}", limits=(0, maximum), colorbar=True)
    axes[0, 3].bar(["vertical", "horizontal"], probabilities, color=[BLUE, ORANGE])
    axes[0, 3].set(ylim=(0, 1), ylabel="softmax probability", title="Actual CNN prediction")
    footer(fig, "Before/after activations share a scale. This is one input/channel, not a semantic attribution or a calibrated confidence guarantee.")
    return fig


def failure_cases(report):
    """Find real errors under the measured shifts/noise; fall back to uncertainty."""
    run = report["runs"][0]
    images, labels = bars(report["configuration"]["test_size"], run["dataset_seeds"][2])
    noise = torch.from_numpy(np.random.default_rng(run["dataset_seeds"][0] + 3).normal(size=images.shape).astype(np.float32))
    cases = []
    for kind in ("cnn", "mlp"):
        model = load_model(kind, run["models"][kind]["checkpoints"][-1]["state"])
        for condition, altered in (("dx=+8 (zero fill)", torch.from_numpy(shift_image(images.numpy(), 8))),
                                   ("extra noise std=1", images + noise)):
            with torch.no_grad():
                probabilities = torch.softmax(model(altered), 1).numpy()
            prediction = probabilities.argmax(1)
            for sample in range(len(images)):
                cases.append({"kind": kind, "condition": condition, "sample": sample,
                              "image": altered[sample, 0].numpy(), "label": int(labels[sample]),
                              "predicted": int(prediction[sample]), "confidence": float(probabilities[sample].max()),
                              "error": bool(prediction[sample] != int(labels[sample]))})
    # Prefer high-confidence real errors. If none exist, label uncertain successes honestly.
    cases.sort(key=lambda item: (bool(item["error"]), float(item["confidence"]) if item["error"] else -float(item["confidence"])), reverse=True)
    return cases


def failure_figure(report):
    cases = failure_cases(report)
    fig, axes = figure(1, 4, "11 · Inspect measured failures, not an invented success story", size=(13, 4.8))
    selected = cases[:4]
    minimum = min(item["image"].min() for item in selected)
    maximum = max(item["image"].max() for item in selected)
    for ax, item in zip(axes.flat, selected):
        title = f"{item['kind'].upper()} · {'ERROR' if item['error'] else 'correct / uncertain'}\n{item['condition']} · sample {item['sample']}"
        heat(ax, item["image"], title, limits=(minimum, maximum), signed=True)
        ax.set_xlabel(f"true={item['label']}, predicted={item['predicted']}\nP(predicted)={item['confidence']:.3f}", fontsize=9)
    errors = sum(item["error"] for item in cases)
    footer(fig, f"Class 0 = vertical, class 1 = horizontal. {errors} error cases across both models and these two transformed sets (first seed).\nSelection is diagnostic and deliberately biased toward confident mistakes. It is not an additional unbiased test estimate.")
    return fig


def gif(path, builders, *, duration=250):
    frames = []
    try:
        for build in builders:
            fig = build()
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).convert("P", palette=Image.Palette.ADAPTIVE))
            plt.close(fig)
        if not frames:
            raise ValueError("an animation needs at least one frame")
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=False,
                       disposal=2)
    finally:
        for frame in frames:
            frame.close()


def png(path, fig):
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_html(path, fig):
    fig.write_html(path, include_plotlyjs=True, full_html=True,
                   config={"displaylogo": False, "scrollZoom": True, "toImageButtonOptions": {"format": "png"}})


def generate_collection(output_dir=None, *, report=None, seeds=(59, 60), epochs=20,
                        frame_limit=None, progress=None):
    """Create a fresh collection. frame_limit is for bounded export tests only."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "outputs" / "visual-lab" / f"run-{stamp}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new or empty output directory; previous runs are preserved")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if report is None:
        report = run_comparison(seeds=seeds, epochs=epochs, progress=progress)
    (output / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    predictions = translation_predictions(report)
    image, _ = selected_example(report, sample=predictions["sample"])
    diagnostics = {"hypothesis": "Interior stride-1 equivariance does not imply unchanged finite-classifier predictions.",
                   "configuration": {"source_experiment_utc": report["created_utc"], "kernel": "handcrafted vertical 3x3", "stride": 1, "padding": 1,
                                     "boundary": "overlap of complete interior windows only", "displacements": list(range(-8, 9, 2))},
                   "results": {"per_image_predictions": predictions,
                               "interior_max_errors": [equivariance(image[0, 0].numpy(), FILTERS["Vertical edge"], dx)["max_error"] for dx in range(-8, 9, 2)]},
                   "interpretation_candidate": "Compare operator alignment separately from actual class confidence, without claiming invariance outside these cases.",
                   "interpretation_status": "author review required",
                   "limitation": "One example selected by label; outer shifts can crop signal; no causal or general robustness inference."}
    (output / "mechanism_checks.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    def pick(values):
        values = list(values)
        if frame_limit is None or len(values) <= frame_limit:
            return values
        return [values[index] for index in np.linspace(0, len(values) - 1, frame_limit, dtype=int)]
    tasks = [
        (EXPECTED[0], lambda path: gif(path, [lambda i=i: sliding_figure(i) for i in pick(range(36))], duration=200)),
        (EXPECTED[1], lambda path: png(path, filters_figure())),
        (EXPECTED[2], lambda path: write_html(path, filters_plotly())),
        (EXPECTED[3], lambda path: write_html(path, channels_figure())),
        (EXPECTED[4], lambda path: gif(path, [lambda s=s, p=p, i=i: stride_figure(stride=s, padding=p, step=i)
                                            for s, p, i in pick([(s, p, i) for s, p in [(1, 0), (2, 0), (3, 0), (2, 1)]
                                                                for i in (0, 2, 4)])], duration=450)),
        (EXPECTED[5], lambda path: gif(path, [lambda i=i: pooling_figure(i) for i in pick(range(16))], duration=350)),
        (EXPECTED[6], lambda path: gif(path, [lambda i=i: receptive_figure(i) for i in pick(range(5))], duration=950)),
        (EXPECTED[7], lambda path: gif(path, [lambda dx=dx: translation_figure(report, dx, predictions=predictions) for dx in pick(range(-8, 9, 2))], duration=500)),
        (EXPECTED[8], lambda path: write_html(path, architecture_figure())),
        (EXPECTED[9], lambda path: gif(path, [lambda i=i: learning_figure(report, i)
                                            for i in pick(range(len(report["runs"][0]["models"]["cnn"]["checkpoints"])))], duration=750)),
        (EXPECTED[10], lambda path: png(path, curves_figure(report))),
        (EXPECTED[11], lambda path: png(path, comparison_figure(report))),
        (EXPECTED[12], lambda path: png(path, failure_figure(report))),
        ("09_features_before_after.png", lambda path: png(path, feature_figure(report))),
    ]
    for index, (name, render) in enumerate(tasks):
        if progress:
            progress(index / len(tasks), f"Rendering {name}")
        render(output / name)
    artifacts = []
    for name in EXPECTED + ("09_features_before_after.png",):
        path = output / name
        item = {"name": name, "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix in (".png", ".gif"):
            with Image.open(path) as image:
                item.update(width=image.width, height=image.height, frames=getattr(image, "n_frames", 1))
        artifacts.append(item)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "artifacts": artifacts,
                "frame_limit": frame_limit, "interpretation_status": "author review required"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if progress:
        progress(1., f"Saved {len(artifacts)} artifacts to {output}")
    return output, report
