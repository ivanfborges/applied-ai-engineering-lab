"""Matplotlib PNG/GIF exports of computed Day 70 tensors and diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation, patches, pyplot as plt
import numpy as np

from visual_core import (activation, activation_derivative, ffn_fixture, geometry_fixture,
                         norm_steps, parameter_counts, residual_fixture, swiglu_fixture)

COLORS = {"attention": "#D55E00", "ffn": "#0072B2", "norm": "#7B4AA1",
          "residual": "#008767", "gradient": "#B83E50", "plain": "#667788"}
SERIES = ["#0072B2", "#D55E00", "#008767", "#7B4AA1"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .16, "figure.facecolor": "#F8FAFC"})


def _finish(fig, title, caption):
    fig.suptitle(title, fontsize=15, fontweight="bold", y=.98)
    fig.text(.04, .025, caption, fontsize=9, color="#334155", va="bottom")
    fig.subplots_adjust(top=.86, bottom=.16, left=.08, right=.96, wspace=.3, hspace=.4)


def _png(fig, folder, filename):
    path = Path(folder) / filename
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=.2)
    plt.close(fig)
    return path


def _gif(fig, update, frames, folder, filename, fps=3):
    movie = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    path = Path(folder) / filename
    movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=85)
    plt.close(fig)
    return path


def _heatmap(ax, value, title, bound=None, annotate=False):
    value = np.atleast_2d(value)
    bound = max(float(np.max(np.abs(value))), 1e-9) if bound is None else bound
    image = ax.imshow(value, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.grid(False)
    ax.set_title(f"{title}  {tuple(value.shape)}", fontsize=11)
    ax.set_xlabel("Feature coordinate")
    ax.set_ylabel("Token / row")
    if annotate:
        for i, j in np.ndindex(value.shape):
            ax.text(j, i, f"{value[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(value[i, j]) > .55 * bound else "#142132")
    return image


def visualize_ffn_information_flow(folder, config, data):
    f = data["ffn"]
    stages = [("Input", f["x"]), ("Linear expansion", f["expanded"]),
              ("GELU (tanh)", f["hidden"]), ("Linear projection / output", f["output"])]
    bound = max(np.max(np.abs(value)) for _, value in stages)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 4]})

    def draw(frame):
        token, stage = divmod(frame, 4)
        for ax in axes:
            ax.clear()
        axes[0].set_xlim(0, 10)
        axes[0].set_ylim(-1, 1)
        axes[0].axis("off")
        for j, (label, value) in enumerate(stages):
            x = j * 2.55
            axes[0].add_patch(patches.FancyBboxPatch((x, -.5), 2.2, 1, boxstyle="round,pad=.1",
                             facecolor=COLORS["ffn"] if stage == j else "#DCE8F3", alpha=.85))
            axes[0].text(x + 1.1, 0, f"{label}\n{value.shape}", ha="center", va="center", fontsize=9,
                         color="white" if j == stage else "#142132")
            if j < 3:
                axes[0].annotate("", (x + 2.5, 0), (x + 2.2, 0), arrowprops={"arrowstyle": "->"})
        title, value = stages[stage]
        _heatmap(axes[1], value, title, bound, annotate=True)
        axes[1].add_patch(patches.Rectangle((-.5, token - .5), value.shape[1], 1,
                                           fill=False, edgecolor="#E69F00", linewidth=3))
        _finish(fig, f"Shared 8 → 32 → 8 FFN · token {token} · stage {stage + 1}/4",
                "Actual synthetic values, rounded to 2 decimals. One color scale across all stages.\n"
                "The same W1/W2 act on each row; no token mixing. Random weights; no semantic improvement claim.")
    return [_gif(fig, draw, 16, folder, "ffn_information_flow.gif", fps=2)]


def visualize_ffn_geometry(folder, config, data):
    f = data["geometry"]
    arrays = [a for variant in f["variants"].values() for a in variant]
    bound = max(2.5, max(np.abs(a).max() for a in arrays)) * 1.08
    fig = plt.figure(figsize=(12, 5.8))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    names = list(f["variants"])
    frames_per_stage = 4 if config.quick else 8
    def draw(frame):
        stage = min(frame // frames_per_stage, 3)
        fraction = (frame % frames_per_stage) / max(1, frames_per_stage - 1)
        for ax, name in zip(axes, names):
            ax.clear()
            start = f["variants"][name][stage]
            end = f["variants"][name][min(stage + 1, 3)]
            xyz = (1 - fraction) * start + fraction * end
            ax.scatter(*xyz.T, c=np.array(SERIES)[f["labels"]], s=12, alpha=.8)
            for point in (0, 1, 2):
                ax.text(*xyz[point], str(point), fontsize=9)
            ax.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-bound, bound),
                   xlabel="Coordinate 0", ylabel="Coordinate 1", zlabel="Coordinate 2", title=name)
            ax.view_init(22, -55)
        _finish(fig, f"3 → 12 → 3 · matched weights · stage {stage + 1}",
                "Original 3D input/output; hidden coordinates 0–2 of 12 (not a complete hidden representation).\n"
                "Interpolation is a display transition between feature spaces. IDs/colors persist; random weights do not imply better separation.")
    return [_gif(fig, draw, frames_per_stage * 4, folder, "linear_vs_nonlinear_ffn.gif", fps=4)]


def visualize_activation_functions(folder, config, data):
    x = np.linspace(-5, 5, 501)
    paths = []
    for derivative, filename in [(False, "activation_functions.png"), (True, "activation_derivatives.png")]:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for name, color in zip(("identity", "relu", "gelu", "silu"), SERIES):
            ax.plot(x, activation_derivative(x, name) if derivative else activation(x, name), label=name, color=color, lw=2)
        ax.set(xlabel="Input x", ylabel="Derivative" if derivative else "f(x)")
        ax.legend(ncol=4)
        _finish(fig, "Local gradient" if derivative else "Activation functions · identical scalar input",
                "ReLU=max(0,x); GELU uses the tanh approximation; SiLU=x·sigmoid(x); identity=x.\n"
                "Derivatives from float64 PyTorch autograd; ReLU derivative at zero is 0. No trained-model quality inference.")
        paths.append(_png(fig, folder, filename))
    return paths


def visualize_residual_connections(folder, config, data):
    f = data["residual"]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection="3d")
    bound = max(np.max(np.abs(f["residual"])), np.max(np.abs(f["replacement"]))) * 1.15
    frames = 16 if config.quick else 32
    def draw(frame):
        ax.clear()
        count = min(1 + round(frame * (len(f["residual"]) - 1) / (frames - 1)), len(f["residual"]))
        for row, (key, color) in enumerate([("residual", COLORS["residual"]), ("replacement", COLORS["plain"])]):
            xyz = f[key][:count]
            ax.plot(*xyz.T, "o-", color=color, label=key)
            ax.text2D(.02, .84 - row * .07, f"{key}: {np.round(xyz[-1], 2)}",
                      transform=ax.transAxes, color=color, fontsize=9)
        ax.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-bound, bound),
               xlabel="Feature 0", ylabel="Feature 1", zlabel="Feature 2")
        ax.view_init(24, -55)
        ax.legend(loc="upper left")
        _finish(fig, f"Residual stream versus replacement · {count - 1} layers",
                "Actual 3D states: x_next=x+tanh(xW+b) versus x_next=tanh(xW+b); same parameters.\n"
                "Residual updates can change direction and magnitude; addition does not guarantee information preservation.")
    return [_gif(fig, draw, frames, folder, "residual_information_flow.gif", fps=3)]


def _profile_plot(ax, rows, metric, title, seed=None):
    rows = rows if seed is None else [r for r in rows if r["seed"] == seed]
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    for condition, color in zip(conditions, SERIES):
        color = {"plain": COLORS["plain"], "residual": COLORS["residual"],
                 "pre": COLORS["ffn"], "post": COLORS["attention"], "pre+final-LN": COLORS["norm"]}.get(condition, color)
        selected = [r for r in rows if r["condition"] == condition]
        indices = sorted(set(r["layer"] for r in selected))
        values = [[r[metric] for r in selected if r["layer"] == i and r.get(metric) is not None] for i in indices]
        means = np.array([np.mean(v) if v else np.nan for v in values])
        minimum = np.array([min(v) if v else np.nan for v in values])
        maximum = np.array([max(v) if v else np.nan for v in values])
        zeros = means == 0
        ax.plot(indices, np.where(zeros, np.nan, means) if "gradient" in metric else means,
                color=color, label=condition, lw=1.8)
        ax.fill_between(indices, np.where(minimum > 0, minimum, np.nan) if "gradient" in metric else minimum,
                        maximum, color=color, alpha=.12)
        if zeros.any():
            ax.text(.01, .02, f"{condition}: {zeros.sum()} exact zeros omitted from log axis", transform=ax.transAxes, fontsize=7)
    if "gradient" in metric:
        ax.set_yscale("log")
    ax.set(xlabel="Layer boundary (0=input)", ylabel=metric.replace("_", " "), title=title)
    ax.legend(fontsize=8)


def visualize_gradient_propagation(folder, config, data):
    rows = data["dense"]
    depth = max(r["depth"] for r in rows)
    chosen = [r for r in rows if r["depth"] == depth]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric in zip(axes.flat, ["gradient_l2", "gradient_rms", "relative_gradient", "activation_rms"]):
        _profile_plot(ax, chosen, metric, f"Depth={depth}")
    for condition, color in (("plain", COLORS["plain"]), ("residual", COLORS["residual"])):
        samples = np.concatenate([r["gradient_samples"] for r in chosen if r["condition"] == condition and r["layer"] == 0])
        axes[1, 1].hist(samples, bins=24, histtype="step", lw=2, color=color, label=condition)
    axes[1, 1].set(title="Input gradient distribution", xlabel="Signed gradient", ylabel="Count")
    axes[1, 1].legend()
    axes[1, 2].axis("off")
    axes[1, 2].text(0, .9, "Actual PyTorch tanh networks\nMatched weights, inputs and MSE target\nNo norm or optimizer steps\n\nLine: seed mean\nBand: seed min/max\nNot a confidence interval", va="top")
    _finish(fig, "Measured dense-network propagation at initialization",
            "Each boundary retains its actual autograd gradient. A larger norm is not a better learning signal.\n"
            "Gradient distributions sample at most 64 coordinates per seed; inputs/targets are synthetic.")
    paths = [_png(fig, folder, "gradient_propagation_experiment.png")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    depths = sorted(set(r["depth"] for r in rows))
    def draw(i):
        for ax in axes:
            ax.clear()
        chosen = [r for r in rows if r["depth"] == depths[i]]
        _profile_plot(axes[0], chosen, "relative_gradient", "Relative to final-state gradient")
        _profile_plot(axes[1], chosen, "activation_rms", "Activation scale")
        axes[0].set_ylim(1e-18, 1e3)
        _finish(fig, f"Actual autograd · network depth {depths[i]}",
                "Paired independently initialized layers; deeper networks extend the same prefix within a seed.\n"
                "Bands are seed ranges. Any values outside the common log axis remain present in CSV.")
    paths.append(_gif(fig, draw, len(depths), folder, "gradient_propagation_depth.gif", fps=1))
    return paths


def visualize_layer_normalization(folder, config, data):
    x = np.array([2., 4., 6., 8., 1., -2., 3., 5.])
    gamma = np.linspace(.6, 1.4, len(x))
    beta = np.linspace(-.4, .4, len(x))
    s = norm_steps(x, gamma, beta)
    values = [x, np.full_like(x, s["mean"].item()), s["centered"], s["centered"]**2,
              s["normalized"], s["scaled"], s["output"], s["output"]]
    names = ["Input", "Mean across hidden features", "Subtract mean", "Squared centered values → population variance",
             "Divide by sqrt(variance + epsilon)", "Multiply feature-wise gamma", "Add feature-wise beta", "Final representation"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 2]})
    def draw(i):
        for ax in axes:
            ax.clear()
        value = values[i]
        axes[0].bar(np.arange(len(x)), value, color=COLORS["norm"])
        axes[0].set_ylim(min(-5, value.min() - 1), max(10, value.max() + 2))
        for j, v in enumerate(value):
            axes[0].text(j, v + .15, f"{v:.2f}", ha="center", fontsize=9)
        axes[0].set(xlabel="Hidden feature", ylabel="Actual stage value", title=names[i])
        axes[1].hist(value, bins=8, color=COLORS["norm"], alpha=.7)
        axes[1].set(xlabel="Stage value", ylabel="Feature count", title="Eight features; no Gaussian assumption")
        text = (f"Original mean={s['mean'].item():.4f}  variance={s['variance'].item():.4f}\n"
                f"sqrt(v+eps)={s['std'].item():.4f}  eps={s['eps']:g}\n"
                f"stage mean={value.mean():.4f}  variance={value.var():.4f}  RMS={np.sqrt(np.mean(value**2)):.4f}\n"
                f"gamma={np.round(gamma, 2)}\nbeta={np.round(beta, 2)}")
        _finish(fig, f"LayerNorm · step {i + 1}/8", text)
        fig.subplots_adjust(bottom=.28)
    paths = [_gif(fig, draw, len(values), folder, "layer_normalization_step_by_step.gif", fps=1)]
    rng = np.random.default_rng(config.seed)
    tensors = [rng.normal(3, 2, size=(96, 8)), np.ones((96, 8)) * 3,
               3 + rng.normal(scale=1e-7, size=(96, 8))]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.5))
    for ax, tensor, name in zip(axes, tensors, ("Nonconstant", "Constant", "Near-constant")):
        normalized = norm_steps(tensor)["normalized"]
        ax.hist(tensor.flatten(), bins=30, alpha=.5, label="input", density=True)
        ax.hist(normalized.flatten(), bins=30, alpha=.5, label="normalized", density=True)
        ax.set(title=f"{name}\noutput variance={normalized.var():.3g}", xlabel="Feature value", ylabel="Density")
        ax.legend(fontsize=8)
    _finish(fig, "LayerNorm distributions and epsilon sensitivity",
            "Synthetic feature ensembles; each row is normalized separately. Constant input maps to zero before beta.\n"
            "Near-zero variance is dominated by epsilon; output does not have to have unit variance or a Gaussian shape.")
    paths.append(_png(fig, folder, "layer_normalization_distributions.png"))
    return paths


def visualize_normalization_comparison(folder, config, data):
    base = data["normalization"]["variants"]["base"]["input"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    def draw(frame):
        for ax in axes:
            ax.clear()
        mode, token = divmod(frame, 4)
        _heatmap(axes[0], base.reshape(8, 3), "B=2, T=4, D=3; rows=(batch,token)", annotate=True)
        if mode < 2:
            axes[0].add_patch(patches.Rectangle((-.5, token - .5), 3, 1, fill=False, linewidth=3, edgecolor=COLORS["norm"]))
        else:
            axes[0].add_patch(patches.Rectangle((token % 3 - .5, -.5), 1, 8, fill=False, linewidth=3, edgecolor=COLORS["attention"]))
        names = ["LayerNorm: feature mean + variance", "RMSNorm: feature mean square", "BatchNorm train: B,T per channel"]
        axes[1].axis("off")
        axes[1].text(0, .95, names[mode] + "\n\nLN/RMS reduce D for each (B,T).\nBN1d layout B,D,T reduces B,T.\n\nHighlight = statistics group,\nnot an attention relationship.\n\nBN eval uses fixed running moments;\nother current samples cannot affect it.", va="top", fontsize=12)
        _finish(fig, "Which elements share normalization statistics?",
                "Actual synthetic tensor. LN/RMS do not couple tokens; training BatchNorm couples B,T for each feature.\n"
                "Running variance uses PyTorch's unbiased update; training normalization uses population variance.")
    return [_gif(fig, draw, 12, folder, "normalization_axes.gif", fps=2)]


def visualize_pre_post_norm(folder, config, data):
    rows = data["transformer"]
    depth = config.depth
    selected = [r for r in rows if r["depth"] == depth]
    paths = []
    specs = [("gradient_l2", "pre_norm_vs_post_norm_gradients.png", "Absolute state-gradient L2"),
             ("activation_rms", "pre_norm_vs_post_norm_activations.png", "Raw residual stream RMS"),
             ("ffn_update_ratio", "residual_update_ratio.png", "FFN branch RMS / pre-addition stream RMS")]
    for metric, filename, title in specs:
        fig, ax = plt.subplots(figsize=(10, 5.8))
        _profile_plot(ax, selected, metric, title)
        _finish(fig, f"Real Transformer blocks · depth {depth} · {config.d_model} features",
                "Shared initial parameters; real four-head bidirectional attention; synthetic random-target mean MSE; zero updates.\n"
                "Lines=seed means, bands=min/max. Pre+final-LN changes readout, not internal activations. Not a quality ranking.")
        paths.append(_png(fig, folder, filename))
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.5))
    depths = sorted(set(r["depth"] for r in rows))
    for ax, condition in zip(axes, ("pre", "post", "pre+final-LN")):
        matrix = np.full((len(depths), max(depths) + 1), np.nan)
        for i, d in enumerate(depths):
            for layer in range(d + 1):
                values = [r["gradient_l2"] for r in rows if r["depth"] == d and r["layer"] == layer and r["condition"] == condition]
                mean = np.mean(values)
                matrix[i, layer] = np.log10(mean) if mean > 0 else np.nan
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=-4, vmax=1)
        ax.grid(False)
        ax.set(title=condition, xlabel="Layer boundary", yticks=np.arange(len(depths)), yticklabels=depths, ylabel="Network depth")
    fig.colorbar(im, ax=list(axes), label="log10(mean gradient L2)", fraction=.025, pad=.02)
    _finish(fig, "Measured Transformer gradient heatmap",
            "Common color range; missing layers are blank, not zero. Every measured value remains in CSV.\n"
            "One loss at initialization; attention is unmasked. No convergence or generalization claim.")
    fig.subplots_adjust(right=.88)
    paths.append(_png(fig, folder, "transformer_gradient_heatmap.png"))
    return paths


def visualize_swiglu(folder, config, data):
    f = data["swiglu"]
    names = [("Input", "x"), ("Independent gate projection", "gate_projection"),
             ("Independent feature projection", "features"), ("SiLU gate", "activated_gate"),
             ("Element-wise product", "gated_features"), ("Down projection", "output")]
    bound = max(np.max(np.abs(f[key])) for _, key in names)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    def draw(i):
        ax.clear()
        title, key = names[i]
        _heatmap(ax, f[key], title, bound, annotate=True)
        _finish(fig, f"SwiGLU · stage {i + 1}/6",
                f"Actual tensors; dense width={f['dense_width']} ({f['dense_count']} params), gated width={f['gated_width']} ({f['gated_count']} params), biases enabled.\n"
                "SiLU gates are signed and unbounded above; their product may suppress, amplify or flip feature signs.")
    paths = [_gif(fig, draw, len(names), folder, "swiglu_information_flow.gif", fps=1)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    widths = np.arange(4, 65, 4)
    for gated, label, color in [(False, "GELU FFN", COLORS["ffn"]), (True, "SwiGLU FFN", COLORS["norm"])]:
        axes[0].plot(widths, [parameter_counts(8, int(h), gated=gated)["ffn"] for h in widths], label=label, color=color)
    axes[0].set(xlabel="Intermediate width", ylabel="Parameters including biases")
    axes[0].legend()
    axes[1].bar(["GELU", "SwiGLU"], [f["dense_count"], f["gated_count"]], color=[COLORS["ffn"], COLORS["norm"]])
    axes[1].set(title=f"Nearest budget: h={f['dense_width']} versus {f['gated_width']}", ylabel="Parameters")
    _finish(fig, "Two projections versus three projections",
            "Biases included: dense=2dh+h+d; gated=3dh+2h+d. Norms and attention excluded here.\n"
            "Approximate parameter matching does not match latency, activation statistics or model quality.")
    paths.append(_png(fig, folder, "ffn_parameter_comparison.png"))
    return paths


def visualize_parameter_complexity(folder, config, data):
    counts = parameter_counts(config.d_model, config.d_ff)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    names = ["Attention", "FFN", "Two LayerNorms"]
    values = [counts[k] for k in ("attention", "ffn", "normalization")]
    axes[0].bar(names, values, color=[COLORS["attention"], COLORS["ffn"], COLORS["norm"]])
    for i, v in enumerate(values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom")
    axes[0].set(ylabel="Parameters per block", title="Conventional MHA, biases and affine norms")
    axes[1].bar(["FP32", "FP16", "BF16", "INT8 ideal"], [counts["total"] * config.depth * size / 2**20 for size in (4, 2, 2, 1)], color=SERIES)
    axes[1].set(ylabel="Weight storage estimate (MiB)", title=f"{config.depth} blocks; weights only")
    _finish(fig, "Transformer parameter and idealized storage budget",
            f"d={config.d_model}, h={config.d_ff}; no embeddings, readout, optimizer states, activations or KV cache.\n"
            "INT8 excludes scales/metadata. Storage and arithmetic estimates are not peak memory or measured latency.")
    return [_png(fig, folder, "transformer_parameter_distribution.png")]


def visualize_transformer_block(folder, config, data):
    f = data["block"]
    names = ["input", "norm1", "attention", "residual1", "norm2", "expansion", "gelu", "projection", "residual2", "output"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1.4, 3]})
    def diagram(ax, active):
        ax.axis("off")
        ax.set_xlim(-.5, 10)
        ax.set_ylim(-.7, 1.5)
        for i, name in enumerate(names):
            color = COLORS["norm"] if name.startswith("norm") else COLORS["attention"] if name == "attention" else COLORS["residual"] if "residual" in name else COLORS["ffn"]
            ax.add_patch(patches.FancyBboxPatch((i, 0), .75, .7, boxstyle="round,pad=.06", facecolor=color, alpha=1 if i == active else .35))
            ax.text(i + .375, .35, name, rotation=55, ha="center", va="center", fontsize=8)
            if i < 9:
                ax.annotate("", (i + 1, .35), (i + .78, .35), arrowprops={"arrowstyle": "->", "lw": 1})
        for start, stop, y in [(0, 3, 1.05), (3, 8, 1.35)]:
            ax.plot([start + .35, start + .35, stop + .35, stop + .35], [.7, y, y, .7], color=COLORS["residual"], lw=2)
            ax.text((start + stop) / 2, y + .02, "identity residual", ha="center", fontsize=8, color=COLORS["residual"])
    def forward(i):
        for ax in axes:
            ax.clear()
        diagram(axes[0], i)
        value = f["trace"][names[i]][0]
        _heatmap(axes[1], value, names[i], annotate=True)
        _finish(fig, f"Complete Pre-LN block · forward {i + 1}/10",
                "Actual PyTorch B=1,T=4,d=8,h=32,two-head activations. Colors: orange attention, blue FFN, purple norm, green residual.\n"
                "Residual sums have two input paths. No position encoding, masks, dropout or training; color scale is local to each stage.")
    paths = [_gif(fig, forward, len(names), folder, "transformer_block_forward_pass.gif", fps=1)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    contributions = list(f["contributions"].items())
    backward_names = list(reversed(names))
    def backward(i):
        for ax in axes:
            ax.clear()
        if i < len(backward_names):
            key = backward_names[i]
            value = f["gradients"][key][0]
            _heatmap(axes[0], value, f"dLoss/d({key})", annotate=True)
        else:
            label, value = contributions[i - len(backward_names)]
            _heatmap(axes[0], value[0], label, annotate=True)
        labels = ["FFN identity", "FFN branch", "Middle total", "Attention branch", "Input total"]
        magnitudes = [np.linalg.norm(value) for _, value in contributions]
        axes[1].barh(labels, magnitudes, color=[COLORS["residual"], COLORS["gradient"], COLORS["residual"], COLORS["gradient"], COLORS["gradient"]])
        axes[1].set(xlabel="Actual contribution L2 norm", title="Vector sums, not sums of norms")
        _finish(fig, "Complete block backward · both learned and identity paths",
                f"Middle = upstream + FFN VJP; input = middle + attention VJP. Reconstruction error={f['decomposition_max_abs_error']:.2e}.\n"
                "Autograd follows the real graph; branch vectors may cancel. Reverse stage order is pedagogical, not an execution-time trace.")
    paths.append(_gif(fig, backward, len(backward_names) + len(contributions), folder, "transformer_block_backward_pass.gif", fps=2))
    fig, ax = plt.subplots(figsize=(13, 4.8))
    diagram(ax, 9)
    for i, name in enumerate(names):
        ax.text(i + .3, -.3, str(f["trace"][name].shape), rotation=35, fontsize=8, ha="center")
    _finish(fig, "Pre-Norm Transformer block · actual operation graph",
            "Arrows are schematic dependencies; tensor shapes and the two bypasses correspond to the computed block.\n"
            "Attention mixes token positions; the shared FFN acts independently at each position.")
    paths.append(_png(fig, folder, "transformer_block_architecture.png"))
    return paths


RENDERERS = {
    "ffn": visualize_ffn_information_flow, "geometry": visualize_ffn_geometry,
    "activations": visualize_activation_functions, "residual": visualize_residual_connections,
    "gradients": visualize_gradient_propagation, "normalization": visualize_layer_normalization,
    "normalization-comparison": visualize_normalization_comparison,
    "pre-post-norm": visualize_pre_post_norm, "swiglu": visualize_swiglu,
    "complexity": visualize_parameter_complexity, "transformer-block": visualize_transformer_block,
}
